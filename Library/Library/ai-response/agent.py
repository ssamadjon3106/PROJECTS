from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from compliance import evaluate_compliance, evaluate_risk, response_needs_refinement
from knowledge_base import LEGAL_DOCS_DIR
from rag import DEFAULT_TOP_K, LegalRAGIndex, build_legal_context_text, load_default_rag_index


load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip() or None
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_TOP_K_RETRIEVAL = int(os.getenv("LEGAL_RAG_TOP_K", str(DEFAULT_TOP_K)))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1200"))

AUTHORITY_KEYWORDS = {
    "prokuratura": ("prokuratura", "prokuror", "tergov", "jinoyat", "nazorat"),
    "soliq": ("soliq", "deklaratsiya", "hisobot", "tekshiruv", "soliq organi"),
    "markaziy bank": ("markaziy bank", "mb", "regulyator", "nazorat", "monetary"),
}

ENTITY_KEYWORDS = (
    "bank",
    "bank siri",
    "hisobvaraq",
    "hisobvaraqlar",
    "tranzaksiya",
    "tranzaksiyalar",
    "karta",
    "kredit",
    "depozit",
    "mijoz",
    "pasport",
    "telefon",
    "soliq",
    "deklaratsiya",
    "tekshiruv",
    "prokuratura",
    "markaziy bank",
    "sud",
    "kontragent",
    "tushum",
    "chiqim",
    "qoldiq",
    "omonat",
)

SENSITIVE_KEYWORDS = {
    "bank_secrecy": ("bank siri", "hisobvaraq", "tranzaksiya", "qoldiq", "depozit", "kredit", "karta"),
    "personal_data": ("pasport", "telefon", "manzil", "jshshir", "jshsh", "ism", "familiya", "tug'ilgan"),
    "financial_info": ("hisobvaraq", "tranzaksiya", "tushum", "chiqim", "qoldiq", "bank", "depozit", "kredit"),
}

HIGH_CONFIDENCE_THRESHOLD = 80
MEDIUM_CONFIDENCE_THRESHOLD = 55

ROUTING_AUTO_REPLY = "AUTO_REPLY"
ROUTING_BANK_REVIEW = "BANK_REVIEW"
ROUTING_HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"


@dataclass(frozen=True)
class LegalClassification:
    authority: str
    intent: str
    entities: list[str]
    issues: list[str]
    bank_secrecy: bool
    personal_data: bool
    financial_info: bool


@dataclass(frozen=True)
class LegalAgentResult:
    user_input: str
    classification: dict[str, Any]
    risk_level: str
    risk_score: int
    compliance_score: int
    confidence_score: int
    confidence_level: str
    routing_mode: str
    review_required: bool
    review_reason: str
    warnings: list[str]
    retrieved_chunks: list[dict[str, Any]]
    retrieved_context: str
    draft_response: str
    final_response: str
    used_llm: bool
    llm_model: str | None
    refinement_passes: int


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    normalized = _clean_text(text)
    return any(keyword in normalized for keyword in keywords)


def _unique_preserve_order(values: Sequence[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _clamp_score(score: int) -> int:
    return max(0, min(100, score))


def detect_authority(user_input: str) -> str:
    normalized = _clean_text(user_input)
    for authority, keywords in AUTHORITY_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return authority
    return "other"


def detect_intent(user_input: str) -> str:
    normalized = _clean_text(user_input)

    intent_rules = (
        ("ma'lumot taqdim etish", ("ma'lumot", "axborot", "taqdim et", "bering", "so'radi", "talab qildi")),
        ("qisman javob", ("qisman", "faqat", "zarur qism", "minimal", "cheklangan")),
        ("rad etish", ("rad et", "bermaslik", "inkor", "noma'lum", "mavhum")),
        ("aniqlashtirish", ("aniqlashtir", "qo'shimcha", "rekvizit", "to'ldir", "aniq")),
        ("tekshiruv", ("tekshiruv", "audit", "nazorat", "sorov", "so'rov", "monitoring")),
    )

    for intent, keywords in intent_rules:
        if any(keyword in normalized for keyword in keywords):
            return intent
    return "huquqiy baholash"


def detect_entities(user_input: str) -> list[str]:
    normalized = _clean_text(user_input)
    entities = [keyword for keyword in ENTITY_KEYWORDS if keyword in normalized]
    return _unique_preserve_order(entities)


def detect_issues(user_input: str) -> dict[str, bool]:
    normalized = _clean_text(user_input)
    return {
        "bank_secrecy": any(keyword in normalized for keyword in SENSITIVE_KEYWORDS["bank_secrecy"]),
        "personal_data": any(keyword in normalized for keyword in SENSITIVE_KEYWORDS["personal_data"]),
        "financial_info": any(keyword in normalized for keyword in SENSITIVE_KEYWORDS["financial_info"]),
    }


def classify_request(user_input: str) -> LegalClassification:
    authority = detect_authority(user_input)
    intent = detect_intent(user_input)
    entities = detect_entities(user_input)
    issues = detect_issues(user_input)

    issue_list = [name for name, enabled in issues.items() if enabled]
    return LegalClassification(
        authority=authority,
        intent=intent,
        entities=entities,
        issues=issue_list,
        bank_secrecy=issues["bank_secrecy"],
        personal_data=issues["personal_data"],
        financial_info=issues["financial_info"],
    )


def build_rag_query(user_input: str, classification: LegalClassification) -> str:
    parts = [user_input]

    if classification.authority != "other":
        parts.append(classification.authority)

    if classification.bank_secrecy:
        parts.extend(
            [
                "bank siri ma'lumot berish tartibi",
                "lex.uz bank siri to'g'risida qonun",
                "530-II bank siri",
                "prokuratura sud davlat ijrosi soliq organlari",
            ]
        )

    if classification.personal_data:
        parts.append("shaxsga doir ma'lumotlar himoyasi")

    if classification.financial_info:
        parts.append("moliyaviy axborot bank ko'chirmasi")

    if classification.intent:
        parts.append(classification.intent)

    if classification.authority == "prokuratura":
        parts.append("prokurorning sanksiyasi bank siri")
    elif classification.authority == "soliq":
        parts.append("davlat soliq xizmati organlari bank siri")
    elif classification.authority == "markaziy bank":
        parts.append("markaziy bank nazorat bank siri")
    elif classification.intent in {"qisman javob", "aniqlashtirish", "rad etish"}:
        parts.append("minimal oshkor etish rad qilish aniqlashtirish")

    return " | ".join(parts)


@lru_cache(maxsize=1)
def get_rag_index(base_dir: str = str(LEGAL_DOCS_DIR)) -> LegalRAGIndex:
    return load_default_rag_index(base_dir)


def retrieve_legal_context(
    user_input: str,
    classification: LegalClassification,
    *,
    top_k: int = DEFAULT_TOP_K_RETRIEVAL,
    base_dir: str | os.PathLike[str] = LEGAL_DOCS_DIR,
) -> tuple[list[dict[str, Any]], str]:
    rag_index = get_rag_index(str(base_dir))
    query = build_rag_query(user_input, classification)
    retrieved_chunks = rag_index.retrieve(query, top_k=top_k)
    retrieved_context = rag_index.build_context(query, top_k=top_k)
    return retrieved_chunks, retrieved_context


def get_openai_client() -> Optional[OpenAI]:
    if not OPENAI_API_KEY:
        return None

    client_kwargs: dict[str, Any] = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        client_kwargs["base_url"] = OPENAI_BASE_URL
    elif LLM_PROVIDER == "openrouter":
        client_kwargs["base_url"] = "https://openrouter.ai/api/v1"

    return OpenAI(**client_kwargs)


def format_retrieved_sources(retrieved_chunks: Sequence[dict[str, Any]]) -> str:
    if not retrieved_chunks:
        return "[Konstitutsiya, 5-modda; Bank qonuni, 12-modda]"

    citations: list[str] = []
    for chunk in retrieved_chunks:
        citation = str(chunk.get("citation", "")).strip()
        if citation:
            citations.append(citation)
    citations = _unique_preserve_order(citations)
    return "; ".join(citations) if citations else "[Konstitutsiya, 5-modda; Bank qonuni, 12-modda]"


def build_system_prompt() -> str:
    return (
        "Siz O‘zbekistondagi davlat organlari so‘rovlari uchun ishlaydigan huquqiy AI agentisiz. "
        "Faqat berilgan huquqiy kontekstga tayangan holda javob yozing. "
        "Javob mutlaqo o‘zbek tilida, rasmiy-huquqiy uslubda bo‘lsin. "
        "Hech qachon umumiy gapirmang, har bir xulosa uchun aniq huquqiy asos keltiring. "
        "Natija quyidagi 4 bo‘limdan iborat bo‘lsin va sarlavhalar aynan shunday yozilsin:\n"
        "KIRISH\nASOS\nHUQUQIY TAHLIL\nXULOSA\n"
        "Har bir bo‘limda tegishli qonunlarga havola bering va kvadrat qavs ichida sitata yozing, masalan: "
        "[Bank qonuni, 12-modda]."
    )


def build_user_prompt(
    user_input: str,
    classification: LegalClassification,
    retrieved_context: str,
    *,
    revision_hint: str | None = None,
) -> str:
    revision_section = f"\nQayta ishlash talabi: {revision_hint}\n" if revision_hint else ""
    return (
        f"Foydalanuvchi so‘rovi:\n{user_input}\n\n"
        f"Classification:\n"
        f"- authority: {classification.authority}\n"
        f"- intent: {classification.intent}\n"
        f"- entities: {', '.join(classification.entities) if classification.entities else 'yo‘q'}\n"
        f"- issues: {', '.join(classification.issues) if classification.issues else 'yo‘q'}\n"
        f"{revision_section}\n"
        "RAG orqali olingan huquqiy kontekst:\n"
        f"{retrieved_context if retrieved_context else 'Kontekst topilmadi, lekin mavjud qonun normalariga asoslanib ehtiyotkor javob yozing.'}\n\n"
        "Talablar:\n"
        "1) Rasmiy hukm emas, huquqiy tahlil yozing.\n"
        "2) Bank siri, personal ma'lumot va soliq siri bo‘lsa, minimal oshkor etish tamoyiliga amal qiling.\n"
        "3) Noaniq bo‘lsa, aniqlashtirish yoki qisman rad etish yo‘lini tushuntiring.\n"
        "4) Xulosa aniq bo‘lsin: taqdim etish / qisman taqdim etish / rad etish / aniqlashtirish.\n"
        "5) Jumlalar formal, qisqa va aniq bo‘lsin."
    )


def _build_llm_messages(
    user_input: str,
    classification: LegalClassification,
    retrieved_context: str,
    revision_hint: str | None = None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt()},
        {
            "role": "user",
            "content": build_user_prompt(
                user_input=user_input,
                classification=classification,
                retrieved_context=retrieved_context,
                revision_hint=revision_hint,
            ),
        },
    ]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.7, min=0.5, max=4.0),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _call_openai_chat(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> str:
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    return content.strip()


def generate_llm_response(
    user_input: str,
    classification: LegalClassification,
    retrieved_context: str,
    *,
    revision_hint: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> tuple[str, bool]:
    client = get_openai_client()
    if client is None:
        return "", False

    messages = _build_llm_messages(
        user_input=user_input,
        classification=classification,
        retrieved_context=retrieved_context,
        revision_hint=revision_hint,
    )
    try:
        return _call_openai_chat(messages, model=model, max_tokens=max_tokens), True
    except Exception:
        return "", False


def build_fallback_response(
    user_input: str,
    classification: LegalClassification,
    retrieved_chunks: Sequence[dict[str, Any]],
    risk_level: str,
    warnings: Sequence[str],
    *,
    revision_hint: str | None = None,
) -> str:
    citations = format_retrieved_sources(retrieved_chunks)

    if classification.authority == "prokuratura":
        conclusion = (
            "So‘rov vakolatli organ tomonidan yuborilgan bo‘lsa ham, bank siri va personal ma’lumotlar "
            "faqat aniq ko‘rsatilgan hajmda taqdim etilishi lozim. Umumiy yoki mavhum talab bo‘lsa, "
            "so‘rov aniqlashtiriladi."
        )
    elif classification.authority == "markaziy bank":
        conclusion = (
            "Markaziy bank nazorat vakolati doirasida zarur hajmdagi axborot berilishi mumkin, "
            "ammo mijozga oid ortiqcha ma’lumotlar oshkor etilmaydi."
        )
    elif classification.authority == "soliq":
        conclusion = (
            "Soliq organi uchun faqat tekshiruv va hisobot maqsadiga zarur bo‘lgan qism taqdim etiladi; "
            "bank siri va shaxsiy ma’lumotlar bo‘yicha minimal oshkor etish qo‘llaniladi."
        )
    else:
        conclusion = (
            "Vakolat, huquqiy asos yoki maqsad yetarli aniqlanmagan bo‘lsa, so‘rov aniqlashtiriladi "
            "yoki rad etiladi."
        )

    issues_text = []
    if classification.bank_secrecy:
        issues_text.append("bank siri")
    if classification.personal_data:
        issues_text.append("shaxsga doir ma’lumotlar")
    if classification.financial_info:
        issues_text.append("moliyaviy axborot")

    issues_sentence = ", ".join(issues_text) if issues_text else "aniq maxfiylik riski aniqlanmadi"
    warning_sentence = "; ".join(warnings) if warnings else "Muhim huquqiy ogohlantirishlar aniqlanmadi."

    revision_note = (
        f"\nQayta ishlash uchun ichki ko‘rsatma: {revision_hint}"
        if revision_hint
        else ""
    )

    return (
        "KIRISH\n"
        f"Foydalanuvchi so‘rovi davlat organi talabiga taalluqli bo‘lib, aniqlangan risklar: {issues_sentence}. "
        f"So‘rov {classification.authority} toifasiga mansub deb baholandi.{revision_note}\n\n"
        "ASOS\n"
        f"RAG orqali topilgan manbalar: {citations}. "
        "Konstitutsiyada shaxsiy hayot, maxfiylik va vakolat doirasi cheklanganligi belgilangan. "
        "Bank qonunida bank siri faqat qonunda ko‘rsatilgan asoslarda oshkor etilishi mumkin. "
        "Soliq bo‘yicha so‘rovlar ham faqat vakolat va zarur hajm bilan cheklanadi. "
        f"[Konstitutsiya, 3-modda; 5-modda; 8-modda] {citations}\n\n"
        "HUQUQIY TAHLIL\n"
        f"So‘rov matnida {classification.intent} elementi mavjud. "
        f"Bank siri va moliyaviy axborot mavjudligi sababli minimal oshkor etish tamoyili majburiy qo‘llanadi. "
        f"{conclusion} [Bank qonuni, 9-modda; 11-modda; 12-modda]\n\n"
        "XULOSA\n"
        f"{conclusion} {warning_sentence} [Bank qonuni, 12-modda; {citations}]"
    ).strip()


def self_check_response(
    response_text: str,
    classification: LegalClassification,
    retrieved_chunks: Sequence[dict[str, Any]],
    risk_level: str,
    risk_score: int,
) -> tuple[bool, int, list[str]]:
    warnings: list[str] = []
    score = 100

    required_headings = ("KIRISH", "ASOS", "HUQUQIY TAHLIL", "XULOSA")
    for heading in required_headings:
        if heading not in response_text:
            warnings.append(f"Yetishmayotgan bo‘lim: {heading}")
            score -= 20

    if "[" not in response_text or "]" not in response_text:
        warnings.append("Huquqiy sitatalar topilmadi")
        score -= 20

    if classification.bank_secrecy and "bank siri" not in _clean_text(response_text):
        warnings.append("Bank siri bo‘yicha aniq tahlil yetishmaydi")
        score -= 10

    if classification.personal_data and "shaxsiy" not in _clean_text(response_text):
        warnings.append("Shaxsiy ma’lumotlar bo‘yicha tahlil yetishmaydi")
        score -= 10

    if risk_level == "HIGH" and risk_score < 70:
        warnings.append("Risk darajasi yuqori bo‘lishi kerak, lekin ball past")
        score -= 10

    if len(response_text.split()) < 120:
        warnings.append("Javob juda qisqa")
        score -= 10

    if response_needs_refinement(response_text):
        warnings.append("Matn uslubiy jihatdan kuchaytirilishi kerak")
        score -= 10

    citations = format_retrieved_sources(retrieved_chunks)
    if citations and citations not in response_text:
        warnings.append("RAG sitatalari javobda yetarli aks etmagan")
        score -= 5

    score = _clamp_score(score)
    return score >= 80, score, warnings


def calculate_confidence_score(
    *,
    classification: LegalClassification,
    risk_level: str,
    compliance_score: int,
    retrieved_chunks: Sequence[dict[str, Any]],
    response_text: str,
    used_llm: bool,
) -> int:
    score = 50
    score += int(min(compliance_score, 100) * 0.35)

    if risk_level == "LOW":
        score += 12
    elif risk_level == "MEDIUM":
        score -= 5
    else:
        score -= 22

    if classification.authority == "other":
        score += 6
    elif classification.authority in {"prokuratura", "markaziy bank"}:
        score -= 12

    if classification.bank_secrecy:
        score -= 6
    if classification.personal_data:
        score -= 5

    chunk_count = len(list(retrieved_chunks))
    if chunk_count >= 3:
        score += 8
    elif chunk_count == 0:
        score -= 12

    if not used_llm:
        score -= 3

    if response_needs_refinement(response_text):
        score -= 15

    if len(response_text.split()) < 120:
        score -= 6

    return _clamp_score(score)


def determine_routing_mode(
    *,
    confidence_score: int,
    confidence_level: str,
    risk_level: str,
) -> tuple[str, bool, str]:
    if risk_level == "HIGH":
        return (
            ROUTING_HUMAN_REVIEW_REQUIRED,
            True,
            "Risk darajasi yuqori bo‘lgani uchun majburiy inson ko‘rigi talab etiladi.",
        )

    if confidence_level == "HIGH" and confidence_score >= HIGH_CONFIDENCE_THRESHOLD:
        return (
            ROUTING_AUTO_REPLY,
            False,
            "Javob yetarli darajada ishonchli; avtomatik qaytarish mumkin.",
        )

    if confidence_level == "MEDIUM":
        return (
            ROUTING_BANK_REVIEW,
            True,
            "Confidence o‘rtacha; bank xodimi tomonidan review talab qilinadi.",
        )

    return (
        ROUTING_HUMAN_REVIEW_REQUIRED,
        True,
        "Confidence past; qo‘lda ko‘rish va qayta ishlash talab etiladi.",
    )


def confidence_level_from_score(score: int) -> str:
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return "HIGH"
    if score >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def generate_final_response(
    user_input: str,
    *,
    top_k: int = DEFAULT_TOP_K_RETRIEVAL,
    base_dir: str | os.PathLike[str] = LEGAL_DOCS_DIR,
    model: str = DEFAULT_MODEL,
    revision_hint: str | None = None,
) -> LegalAgentResult:
    classification = classify_request(user_input)
    risk_level, risk_score, risk_warnings = evaluate_risk(classification)
    effective_top_k = max(top_k, 6) if classification.bank_secrecy or classification.authority in {"prokuratura", "soliq", "markaziy bank"} else top_k
    retrieved_chunks, retrieved_context = retrieve_legal_context(
        user_input,
        classification,
        top_k=effective_top_k,
        base_dir=base_dir,
    )

    llm_response, used_llm = generate_llm_response(
        user_input=user_input,
        classification=classification,
        retrieved_context=retrieved_context,
        revision_hint=revision_hint,
        model=model,
        max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    )

    if not llm_response:
        llm_response = build_fallback_response(
            user_input=user_input,
            classification=classification,
            retrieved_chunks=retrieved_chunks,
            risk_level=risk_level,
            warnings=risk_warnings,
            revision_hint=revision_hint,
        )

    is_good, compliance_preview_score, self_check_warnings = self_check_response(
        llm_response,
        classification,
        retrieved_chunks,
        risk_level,
        risk_score,
    )

    final_response = llm_response
    refinement_passes = 0

    if not is_good:
        refinement_passes += 1
        revision_hint_first_pass = (
            revision_hint
            or "Javobni yanada rasmiylashtir, bank siri va shaxsiy ma'lumotlar bo'yicha "
            "aniq minimal oshkor etish qoidasini yoz, sitatalarni kuchaytir."
        )
        refined_response, refined_used_llm = generate_llm_response(
            user_input=user_input,
            classification=classification,
            retrieved_context=retrieved_context,
            revision_hint=revision_hint_first_pass,
            model=model,
        )
        if refined_response:
            final_response = refined_response
            used_llm = used_llm or refined_used_llm
        else:
            final_response = build_fallback_response(
                user_input=user_input,
                classification=classification,
                retrieved_chunks=retrieved_chunks,
                risk_level=risk_level,
                warnings=risk_warnings + self_check_warnings,
                revision_hint=revision_hint_first_pass,
            )

    compliance_score, compliance_warnings = evaluate_compliance(
        response_text=final_response,
        classification=classification,
        retrieved_chunks=retrieved_chunks,
        risk_level=risk_level,
        risk_score=risk_score,
    )

    if compliance_score < 80:
        refinement_passes += 1
        second_hint = (
            revision_hint
            or "Ikkinchi qayta ishlash: sarlavhalarni saqla, sitatalarni har bo‘limga joylashtir, "
            "xulosa qismini aniqroq va huquqiyroq qil."
        )
        second_response, second_used_llm = generate_llm_response(
            user_input=user_input,
            classification=classification,
            retrieved_context=retrieved_context,
            revision_hint=second_hint,
            model=model,
        )
        if second_response:
            final_response = second_response
            used_llm = used_llm or second_used_llm
        else:
            final_response = build_fallback_response(
                user_input=user_input,
                classification=classification,
                retrieved_chunks=retrieved_chunks,
                risk_level=risk_level,
                warnings=risk_warnings + compliance_warnings + self_check_warnings,
                revision_hint=second_hint,
            )
        compliance_score, compliance_warnings = evaluate_compliance(
            response_text=final_response,
            classification=classification,
            retrieved_chunks=retrieved_chunks,
            risk_level=risk_level,
            risk_score=risk_score,
        )

    confidence_score = calculate_confidence_score(
        classification=classification,
        risk_level=risk_level,
        compliance_score=compliance_score,
        retrieved_chunks=retrieved_chunks,
        response_text=final_response,
        used_llm=used_llm,
    )
    confidence_level = confidence_level_from_score(confidence_score)
    routing_mode, review_required, review_reason = determine_routing_mode(
        confidence_score=confidence_score,
        confidence_level=confidence_level,
        risk_level=risk_level,
    )

    combined_warnings = _unique_preserve_order(
        list(risk_warnings) + list(self_check_warnings) + list(compliance_warnings)
    )

    if review_required:
        combined_warnings = _unique_preserve_order(
            combined_warnings + [review_reason]
        )

    return LegalAgentResult(
        user_input=user_input,
        classification=asdict(classification),
        risk_level=risk_level,
        risk_score=risk_score,
        compliance_score=compliance_score,
        confidence_score=confidence_score,
        confidence_level=confidence_level,
        routing_mode=routing_mode,
        review_required=review_required,
        review_reason=review_reason,
        warnings=combined_warnings,
        retrieved_chunks=list(retrieved_chunks),
        retrieved_context=retrieved_context,
        draft_response=llm_response,
        final_response=final_response,
        used_llm=used_llm,
        llm_model=model if used_llm else None,
        refinement_passes=refinement_passes,
    )


def process_legal_request(
    user_input: str,
    *,
    top_k: int = DEFAULT_TOP_K_RETRIEVAL,
    base_dir: str | os.PathLike[str] = LEGAL_DOCS_DIR,
    model: str = DEFAULT_MODEL,
    revision_hint: str | None = None,
) -> dict[str, Any]:
    result = generate_final_response(
        user_input=user_input,
        top_k=top_k,
        base_dir=base_dir,
        model=model,
        revision_hint=revision_hint,
    )
    return {
        "user_input": result.user_input,
        "classification": result.classification,
        "risk_level": result.risk_level,
        "risk_score": result.risk_score,
        "compliance_score": result.compliance_score,
        "confidence_score": result.confidence_score,
        "confidence_level": result.confidence_level,
        "routing_mode": result.routing_mode,
        "review_required": result.review_required,
        "review_reason": result.review_reason,
        "warnings": result.warnings,
        "retrieved_chunks": result.retrieved_chunks,
        "retrieved_context": result.retrieved_context,
        "draft_response": result.draft_response,
        "final_response": result.final_response,
        "used_llm": result.used_llm,
        "llm_model": result.llm_model,
        "refinement_passes": result.refinement_passes,
    }


__all__ = [
    "LegalAgentResult",
    "LegalClassification",
    "build_fallback_response",
    "classify_request",
    "confidence_level_from_score",
    "determine_routing_mode",
    "generate_final_response",
    "process_legal_request",
    "retrieve_legal_context",
    "self_check_response",
]
