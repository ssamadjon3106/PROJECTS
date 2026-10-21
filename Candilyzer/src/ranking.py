import pandas as pd

CONFIDENCE_MULTIPLIER = {
    "High": 1.0,
    "Medium": 0.9,
    "Low": 0.8,
}

MATCH_SCORE = {"Met": 1.0, "Partial": 0.5, "Not Met": 0.0}
IMPORTANCE_WEIGHT = {"Must Have": 2.0, "Nice to Have": 1.0}


def score_evaluation(evaluation: dict) -> dict:
    requirements = evaluation.get("requirements_analysis") or []
    weighted_points = 0.0
    total_weight = 0.0
    must_have_total = 0
    must_have_met = 0

    for requirement in requirements:
        importance = requirement.get("importance", "Nice to Have")
        match = requirement.get("match", "Not Met")
        weight = IMPORTANCE_WEIGHT.get(importance, 1.0)
        weighted_points += MATCH_SCORE.get(match, 0.0) * weight
        total_weight += weight
        if importance == "Must Have":
            must_have_total += 1
            must_have_met += int(match == "Met")

    coverage = 100.0 * weighted_points / total_weight if total_weight else 0.0
    confidence = evaluation.get("confidence", "Low")
    score = max(
        0.0,
        min(100.0, coverage * CONFIDENCE_MULTIPLIER.get(confidence, 0.8)),
    )

    return {
        "score": round(score, 1),
        "coverage": round(coverage, 1),
        "must_have_met": must_have_met,
        "must_have_total": must_have_total,
    }


def evidence_level(score: float) -> str:
    """Describe submitted evidence strength without making an employment decision."""
    if score >= 80:
        return "Strong evidence"
    if score >= 60:
        return "Moderate evidence"
    if score >= 40:
        return "Limited evidence"
    return "Insufficient evidence"


def build_df(results: list) -> pd.DataFrame:
    rows = []

    for r in results:
        p = r.get("parsed")
        if not p:
            continue

        breakdown = score_evaluation(p)

        rows.append({
            "Candidate": r["username"],
            "Score": breakdown["score"],
            "Evidence coverage": breakdown["coverage"],
            "Must-have fit": f"{breakdown['must_have_met']}/{breakdown['must_have_total']}",
            "Evidence level": evidence_level(breakdown["score"]),
            "Confidence": p.get("confidence", "Low"),
            "Verification points": len(p.get("verification_points") or []),
        })

    return pd.DataFrame(rows)


def rank(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(
        by=["Score", "Evidence coverage", "Candidate"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def top_k(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    return df.head(k)
