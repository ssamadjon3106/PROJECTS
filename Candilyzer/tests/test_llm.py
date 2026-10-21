import unittest
from unittest.mock import patch

from src.llm import LLMServiceError, run_with_retry


class FakeResult:
    content = "ok"


class FlakyAgent:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def run(self, _prompt: str) -> FakeResult:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("temporary provider failure")
        return FakeResult()


class LLMRetryTests(unittest.TestCase):
    @patch("src.llm.time.sleep")
    def test_retries_transient_failures(self, _sleep) -> None:
        agent = FlakyAgent(failures=2)
        self.assertEqual(run_with_retry(agent, "prompt"), "ok")
        self.assertEqual(agent.calls, 3)

    @patch("src.llm.time.sleep")
    def test_raises_after_bounded_attempts(self, _sleep) -> None:
        with self.assertRaises(LLMServiceError):
            run_with_retry(FlakyAgent(failures=3), "prompt")


if __name__ == "__main__":
    unittest.main()
