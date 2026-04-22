import json
import re


def evaluate_one(candidate: dict, agent, job: str) -> dict:
    username = candidate.get("username", "Unknown")

    if not candidate.get("summary"):
        return {
            "username": username,
            "parsed": None,
            "raw": f"No summary available: {candidate.get('error', 'unknown reason')}"
        }

    try:
        result = agent.run(f"""
Job Description:
{job}

Candidate Profile:
{candidate['summary']}
""")
        response = getattr(result, "content", str(result))

        clean = re.sub(r"```json|```", "", response).strip()

        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            parsed = None
            print(f"[evaluator] JSON parse failed for {username}. Raw: {clean[:200]}")

        return {"username": username, "parsed": parsed, "raw": response}

    except Exception as e:
        return {"username": username, "parsed": None, "raw": str(e)}