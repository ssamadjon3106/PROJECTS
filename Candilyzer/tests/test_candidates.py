import unittest

from src.candidates import CandidateInputError, build_candidates, split_values


class CandidateInputTests(unittest.TestCase):
    def test_split_values_accepts_lines_and_commas(self) -> None:
        self.assertEqual(split_values("alpha\nbeta, gamma"), ["alpha", "beta", "gamma"])

    def test_sources_are_paired_by_position(self) -> None:
        candidates = build_candidates(
            "octocat\nsecond-user",
            "https://www.linkedin.com/in/octocat",
            [("resume.pdf", "Resume evidence")],
        )

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["username"], "octocat")
        self.assertEqual(candidates[0]["resume"], "Resume evidence")
        self.assertEqual(candidates[1]["github_username"], "second-user")
        self.assertIsNone(candidates[1]["linkedin"])

    def test_invalid_github_username_is_rejected(self) -> None:
        with self.assertRaises(CandidateInputError):
            build_candidates("bad/user", "", [])

    def test_non_linkedin_url_is_rejected(self) -> None:
        with self.assertRaises(CandidateInputError):
            build_candidates("", "https://example.com/in/person", [])


if __name__ == "__main__":
    unittest.main()
