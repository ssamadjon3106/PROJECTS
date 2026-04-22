from concurrent.futures import ThreadPoolExecutor
from agno.agent import Agent
from agno.models.nebius import Nebius
from src.config import COMPRESS_MODEL, MAX_WORKERS 
import streamlit as st


def build_compressor(api_key: str) -> Agent:
    return Agent(
        model=Nebius(id=COMPRESS_MODEL, api_key=api_key),
        instructions=(
            "Summarize the candidate profile into: "
            "key technical skills, years of experience, notable projects, and tech stack. "
            "Be concise and factual."
        )
    )

@st.cache_data
def compress_all(data_list: list, agent: Agent) -> list:

    def _compress(d: dict) -> dict:
        username = d.get("username", "Unknown")

        if d.get("error"):
            return {"username": username, "summary": None, "error": d["error"]}

        github = d.get("github") or "N/A"
        linkedin = d.get("linkedin") or "N/A"
        resume = d.get("resume") or "N/A"

        if not any([d.get("github"), d.get("linkedin"), d.get("resume")]):
            return {"username": username, "summary": None, "error": "No data sources available"}
        try:
            combined = f"GitHub:\n{github}\n\nLinkedIn:\n{linkedin}\n\nResume:\n{resume}"
            result = agent.run(combined)
            summary = getattr(result, "content", str(result))  # fixed: extract string from RunResponse
            return {"username": username, "summary": summary, "error": None}

        except Exception as e:
            return {"username": username, "summary": None, "error": str(e)}

    with ThreadPoolExecutor(MAX_WORKERS) as executor:
        futures = [executor.submit(_compress, d) for d in data_list]
        return [f.result() for f in futures]
