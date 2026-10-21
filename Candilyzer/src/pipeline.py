from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import MAX_WORKERS
from src.evaluator import evaluate_one
from src.evaluator_factory import create_evaluator
from src.schemas import EvaluationResult


def evaluate_all(candidates: list[dict], api_key: str, job: str) -> list[EvaluationResult]:
    """Evaluate candidates concurrently while preserving their input order."""

    def _evaluate(candidate: dict) -> EvaluationResult:
        # One agent per candidate prevents conversation state from crossing candidates.
        return evaluate_one(candidate, create_evaluator(api_key), job)

    ordered: list[EvaluationResult | None] = [None] * len(candidates)
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(candidates) or 1)) as executor:
        futures = {
            executor.submit(_evaluate, candidate): index
            for index, candidate in enumerate(candidates)
        }
        for future in as_completed(futures):
            ordered[futures[future]] = future.result()

    return [result for result in ordered if result is not None]
