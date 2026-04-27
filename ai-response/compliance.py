from __future__ import annotations

import re
from typing import Any, Iterable, Sequence


HIGH_RISK_AUTHORITIES = {"prokuratura", "markaziy bank"}
MEDIUM_RISK_AUTHORITIES = {"soliq"}
REQUIRED_HEADINGS = ("KIRISH", "ASOS", "HUQUQIY TAHLIL", "XULOSA")
CITATION_PATTERN = re.compile(r"\[[^\]]+\]")
FORMALITY_MARKERS = (
    "mazkur",
    "huquqiy",
    "qonuniy",
    "vakolat",
    "mutanosiblik",
    "minimal oshkor etish",
    "qisman",
    "aniqlashtirish",
    "asoslantirilgan",
    "belgilangan tartibda",
)
UNSAFE_DISCLOSURE_MARKERS = (
    "to‘liq taqdim etiladi",
    "barcha ma’lumot",
    "hammasi beriladi",
    "to‘liq tarix",
    "barchasi oshkor etiladi",
)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _get_value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _unique_preserve_order(values: Sequence[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _as_bool(value: Any) -> bool:
    return bool(value)


def _authority_floor(authority: str) -> tuple[str, int]:
    normalized = authority.strip().lower()
    if normalized in HIGH_RISK_AUTHORITIES:
        return "HIGH", 75
    if normalized in MEDIUM_RISK_AUTHORITIES:
        return "MEDIUM", 50
    return "LOW", 25


def _level_from_score(score: int) -> str:
    if score >= 70:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    return "LOW"


def _max_level(left: str, right: str) -> str:
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    return left if order[left] >= order[right] else right


def _extract_citations(text: str) -> list[str]:
    return [match.group(0) for match in CITATION_PATTERN.finditer(text)]


def _contains_any(text: str, phrases: Sequence[str]) -> bool:
    normalized = _normalize_text(text)
    return any(phrase.lower() in normalized for phrase in phrases)


def _word_count(text: str) -> int:
    return len([token for token in re.split(r"\s+", text.strip()) if token])


def response_needs_refinement(response_text: str) -> bool:
    """
    Lightweight quality gate used by the agent to decide whether a second pass is needed.
    """
    normalized = _normalize_text(response_text)

    if _word_count(response_text) < 90:
        return True

    if not all(heading in response_text for heading in REQUIRED_HEADINGS):
        return True

    if not CITATION_PATTERN.search(response_text):
        return True

    if not _contains_any(normalized, FORMALITY_MARKERS):
        return True

    if _contains_any(normalized, UNSAFE_DISCLOSURE_MARKERS):
        return True

    return False


def evaluate_risk(classification: Any) -> tuple[str, int, list[str]]:
    """
    Returns:
        risk_level: LOW | MEDIUM | HIGH
        risk_score: 0..100
        warnings: list[str]
    """
    authority = str(_get_value(classification, "authority", "other") or "other").strip().lower()
    bank_secrecy = _as_bool(_get_value(classification, "bank_secrecy", False))
    personal_data = _as_bool(_get_value(classification, "personal_data", False))
    financial_info = _as_bool(_get_value(classification, "financial_info", False))
    issues = _get_value(classification, "issues", []) or []

    base_level, base_score = _authority_floor(authority)
    score = base_score
    warnings: list[str] = []

    if authority in HIGH_RISK_AUTHORITIES:
        warnings.append("Yuqori riskli vakolatli organ so‘rovi aniqlandi.")
    elif authority in MEDIUM_RISK_AUTHORITIES:
        warnings.append("O‘rta riskli soliq so‘rovi aniqlandi.")
    else:
        warnings.append("Umumiy toifadagi so‘rov aniqlandi.")

    if bank_secrecy:
        score += 20
        warnings.append("Bank siri bilan bog‘liq axborot mavjud.")
    if personal_data:
        score += 18
        warnings.append("Shaxsga doir ma’lumotlar mavjud.")
    if financial_info:
        score += 12
        warnings.append("Moliyaviy axborot mavjud.")

    if bank_secrecy and personal_data:
        score += 8
        warnings.append("Bank siri va shaxsiy ma’lumotlar birgalikda himoyani kuchaytiradi.")

    if isinstance(issues, (list, tuple)) and issues:
        issue_set = {str(item).strip().lower() for item in issues}
        if "bank_secrecy" in issue_set:
            score += 4
        if "personal_data" in issue_set:
            score += 4
        if "financial_info" in issue_set:
            score += 3

    score = max(0, min(100, score))
    risk_level = _max_level(base_level, _level_from_score(score))
    return risk_level, score, _unique_preserve_order(warnings)


def _check_citation_alignment(response_text: str, retrieved_chunks: Sequence[dict[str, Any]]) -> bool:
    citations_in_response = _extract_citations(response_text)
    if not retrieved_chunks:
        return bool(citations_in_response)

    retrieved_citations = []
    for chunk in retrieved_chunks:
        citation = str(chunk.get("citation", "")).strip()
        if citation:
            retrieved_citations.append(citation)

    if not retrieved_citations:
        return bool(citations_in_response)

    normalized_response = _normalize_text(response_text)
    for citation in retrieved_citations:
        if _normalize_text(citation) in normalized_response:
            return True
    return bool(citations_in_response)


def evaluate_compliance(
    *,
    response_text: str,
    classification: Any,
    retrieved_chunks: Sequence[dict[str, Any]],
    risk_level: str,
    risk_score: int,
) -> tuple[int, list[str]]:
    """
    Returns:
        compliance_score: 0..100
        warnings: list[str]
    """
    score = 100
    warnings: list[str] = []

    normalized_response = _normalize_text(response_text)
    words = _word_count(response_text)

    for heading in REQUIRED_HEADINGS:
        if heading not in response_text:
            score -= 15
            warnings.append(f"Yetishmayotgan bo‘lim: {heading}")

    if not CITATION_PATTERN.search(response_text):
        score -= 20
        warnings.append("Huquqiy sitatalar topilmadi.")

    if not _check_citation_alignment(response_text, retrieved_chunks):
        score -= 10
        warnings.append("RAG manbalari javobda yetarli aks etmagan.")

    if words < 90:
        score -= 15
        warnings.append("Javob matni yetarli darajada batafsil emas.")

    if not _contains_any(normalized_response, FORMALITY_MARKERS):
        score -= 10
        warnings.append("Rasmiy-huquqiy uslub kuchsiz.")

    authority = str(_get_value(classification, "authority", "other") or "other").strip().lower()
    bank_secrecy = _as_bool(_get_value(classification, "bank_secrecy", False))
    personal_data = _as_bool(_get_value(classification, "personal_data", False))

    if risk_level == "HIGH" and not _contains_any(normalized_response, ("minimal oshkor etish", "qisman", "aniqlashtirish", "rad etiladi")):
        score -= 12
        warnings.append("Yuqori risk uchun minimal oshkor etish yoki rad etish pozitsiyasi aniq emas.")

    if bank_secrecy and "bank siri" not in normalized_response:
        score -= 8
        warnings.append("Bank siri bo‘yicha alohida tahlil yetishmaydi.")

    if personal_data and "shaxs" not in normalized_response and "personal" not in normalized_response:
        score -= 8
        warnings.append("Shaxsga doir ma’lumotlar bo‘yicha tahlil yetishmaydi.")

    if _contains_any(normalized_response, UNSAFE_DISCLOSURE_MARKERS):
        score -= 20
        warnings.append("Noqonuniy yoki haddan tashqari oshkor etish ifodalari topildi.")

    if authority in HIGH_RISK_AUTHORITIES and risk_score < 60:
        score -= 6
        warnings.append("Risk balli yuqori vakolatga nisbatan past ko‘rinadi.")

    if "xulosa" not in response_text.lower():
        score -= 8
        warnings.append("Xulosa qismi aniq ajratilmagan.")

    score = max(0, min(100, score))
    return score, _unique_preserve_order(warnings)


__all__ = [
    "evaluate_compliance",
    "evaluate_risk",
    "response_needs_refinement",
]
