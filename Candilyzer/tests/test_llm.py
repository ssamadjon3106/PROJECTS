import unittest
from unittest.mock import Mock, patch

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


class ProviderError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__("provider details must not reach the UI")
        self.status_code = status_code


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

    @patch("src.llm.time.sleep")
    def test_does_not_retry_authentication_failures(self, sleep) -> None:
        agent = FlakyAgent(failures=0)
        agent.run = Mock(side_effect=ProviderError(401))

        with self.assertRaisesRegex(LLMServiceError, "temporarily unavailable"):
            run_with_retry(agent, "prompt")

        agent.run.assert_called_once()
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
