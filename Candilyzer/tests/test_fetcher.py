import unittest
from unittest.mock import Mock, patch

from src.fetcher import _github_get


class GitHubRequestTests(unittest.TestCase):
    @patch("src.fetcher.requests.get")
    def test_invalid_token_retries_public_endpoint(self, get: Mock) -> None:
        get.side_effect = [Mock(status_code=401), Mock(status_code=200)]

        response = _github_get("https://api.github.com/users/octocat", "invalid")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(get.call_count, 2)
        self.assertNotIn("headers", get.call_args_list[1].kwargs)

    @patch("src.fetcher.requests.get")
    def test_non_auth_failure_is_not_retried(self, get: Mock) -> None:
        get.return_value = Mock(status_code=404)

        response = _github_get("https://api.github.com/users/missing", "token")

        self.assertEqual(response.status_code, 404)
        get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
