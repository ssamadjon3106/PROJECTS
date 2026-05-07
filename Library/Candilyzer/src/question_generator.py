from agno.agent import Agent
from agno.models.nebius import Nebius


def build_question_generator(api_key: str) -> Agent:
    return Agent(
        model=Nebius(
            id="nvidia/nemotron-3-super-120b-a12b",
            api_key=api_key
        ),
        instructions=(
            "Generate 5–7 targeted technical interview questions based on the job, "
            "candidate profile, and evaluation result. "
            "Focus on gaps and areas that need validation."
        )
    )


def generate_questions(agent, job: str, summary: str, evaluation: dict) -> str:
    result = agent.run(f"""
Job Description:
{job}

Candidate Summary:
{summary}

Evaluation Result:
Decision: {evaluation.get('decision')}
Confidence: {evaluation.get('confidence')}
Strengths: {evaluation.get('strengths')}
Gaps: {evaluation.get('gaps')}
Red Flags: {evaluation.get('red_flags', [])}

Generate 5-7 targeted interview questions. Focus especially on gaps and red flags.
""")
    content = getattr(result, "content", None)
    if isinstance(content, list):
        return " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
    return str(content) if content else "Could not generate questions."