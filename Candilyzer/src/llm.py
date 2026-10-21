import time
from typing import Any

from src.config import LLM_ATTEMPTS, LLM_BASE_DELAY_SECONDS


class LLMServiceError(RuntimeError):
    """Raised when an LLM call remains unavailable after bounded retries."""


RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


def _status_code(exc: Exception) -> int | None:
    for attribute in ("status_code", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _is_retryable(exc: Exception) -> bool:
    status_code = _status_code(exc)
    if status_code is not None:
        return status_code in RETRYABLE_STATUS_CODES
    return isinstance(exc, (ConnectionError, TimeoutError)) or status_code is None


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
            if not _is_retryable(exc):
                break
            if attempt + 1 < LLM_ATTEMPTS:
                time.sleep(LLM_BASE_DELAY_SECONDS * (2**attempt))

    raise LLMServiceError("The AI service is temporarily unavailable.") from last_error
