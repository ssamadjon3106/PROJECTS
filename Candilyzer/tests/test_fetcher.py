import unittest
from unittest.mock import Mock, patch

from src.fetcher import GITHUB_HEADERS, _github_get, fetch_candidate


class GitHubRequestTests(unittest.TestCase):
    @patch("src.fetcher.requests.get")
    def test_invalid_token_retries_public_endpoint(self, get: Mock) -> None:
        get.side_effect = [Mock(status_code=401), Mock(status_code=200)]

        response = _github_get("https://api.github.com/users/octocat", "invalid")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(get.call_count, 2)
        self.assertEqual(get.call_args_list[1].kwargs["headers"], GITHUB_HEADERS)

    @patch("src.fetcher.requests.get")
    def test_forbidden_token_retries_public_endpoint(self, get: Mock) -> None:
        get.side_effect = [Mock(status_code=403), Mock(status_code=200)]

        response = _github_get("https://api.github.com/users/octocat", "limited")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(get.call_count, 2)

    @patch("src.fetcher.requests.get")
    def test_non_auth_failure_is_not_retried(self, get: Mock) -> None:
        get.return_value = Mock(status_code=404)

        response = _github_get("https://api.github.com/users/missing", "token")

        self.assertEqual(response.status_code, 404)
        get.assert_called_once()

    @patch("src.fetcher.cached_github", side_effect=RuntimeError("secret provider detail"))
    def test_source_failures_are_sanitized(self, _github: Mock) -> None:
        with self.assertLogs("src.fetcher", level="ERROR"):
            result = fetch_candidate(
                {"username": "octocat", "github_username": "octocat"},
                "token",
                "",
            )

        self.assertFalse(result["source_status"]["github"])
        self.assertNotIn("secret provider detail", result["error"])
        self.assertEqual(
            result["source_errors"]["github"],
            "GitHub evidence is temporarily unavailable.",
        )


if __name__ == "__main__":
    unittest.main()
