import time
from typing import Any

from src.config import LLM_ATTEMPTS, LLM_BASE_DELAY_SECONDS


class LLMServiceError(RuntimeError):
    """Raised when an LLM call remains unavailable after bounded retries."""


def extract_content(result: Any) -> str:
    content = getattr(result, "content", result)
    if isinstance(content, list):
        parts = []
        for item in content:
            parts.append(item.get("text", "") if isinstance(item, dict) else str(item))
        content = " ".join(part for part in parts if part)
    text = str(content).strip() if content is not None else ""
    if not text:
        raise LLMServiceError("The model returned an empty response.")
    return text


def run_with_retry(agent: Any, prompt: str) -> str:
    last_error: Exception | None = None
    for attempt in range(LLM_ATTEMPTS):
        try:
            return extract_content(agent.run(prompt))
        except Exception as exc:  # Provider SDKs expose several transient error types.
            last_error = exc
            if attempt + 1 < LLM_ATTEMPTS:
                time.sleep(LLM_BASE_DELAY_SECONDS * (2**attempt))

    message = str(last_error).strip() if last_error else "Unknown provider error"
    raise LLMServiceError(
        f"AI service unavailable after {LLM_ATTEMPTS} attempts: {message[:240]}"
    ) from last_error
