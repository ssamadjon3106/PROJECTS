import pandas as pd

DECISION_SCORE = {
    "Strong Hire": 100,
    "Hire": 85,
    "Interview Recommended": 70,
    "Reject": 40
}

CONFIDENCE_MULTIPLIER = {
    "High": 1.0,
    "Medium": 0.9,
    "Low": 0.75
}


def build_df(results: list) -> pd.DataFrame:
    rows = []
    skipped = []

    for r in results:
        p = r.get("parsed")
        if not p:
            skipped.append(r.get("username", "Unknown"))
            continue

        decision = p.get("decision", "Reject")
        confidence = p.get("confidence", "Low")

        base_score = DECISION_SCORE.get(decision, 0)
        multiplier = CONFIDENCE_MULTIPLIER.get(confidence, 0.75)
        final_score = round(base_score * multiplier, 1)

        rows.append({
            "Candidate": r["username"],
            "Decision": decision,
            "Confidence": confidence,
            "Score": final_score
        })

    if skipped:
        print(f"[ranking] Skipped candidates (no parsed result): {skipped}")

    return pd.DataFrame(rows)


def rank(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(by="Score", ascending=False).reset_index(drop=True)


def top_k(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    return df.head(k)