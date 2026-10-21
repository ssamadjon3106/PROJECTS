import unittest

from src.ranking import build_df, rank, score_evaluation


class RankingTests(unittest.TestCase):
    def test_score_is_derived_from_requirement_evidence(self) -> None:
        evaluation = {
            "requirements_analysis": [
                {"importance": "Must Have", "match": "Met"},
                {"importance": "Must Have", "match": "Partial"},
                {"importance": "Nice to Have", "match": "Not Met"},
            ],
            "confidence": "High",
            "red_flags": [],
        }
        breakdown = score_evaluation(evaluation)
        self.assertEqual(breakdown["coverage"], 60.0)
        self.assertEqual(breakdown["score"], 60.0)
        self.assertEqual(breakdown["must_have_met"], 1)

    def test_red_flags_apply_a_bounded_penalty(self) -> None:
        evaluation = {
            "requirements_analysis": [
                {"importance": "Must Have", "match": "Met"},
            ],
            "confidence": "High",
            "red_flags": ["one", "two", "three", "four"],
        }
        self.assertEqual(score_evaluation(evaluation)["score"], 85.0)

    def test_rank_orders_by_score(self) -> None:
        results = [
            {
                "username": "Lower",
                "parsed": {
                    "requirements_analysis": [
                        {"importance": "Must Have", "match": "Partial"}
                    ],
                    "confidence": "High",
                    "red_flags": [],
                    "decision": "Interview Recommended",
                },
            },
            {
                "username": "Higher",
                "parsed": {
                    "requirements_analysis": [
                        {"importance": "Must Have", "match": "Met"}
                    ],
                    "confidence": "High",
                    "red_flags": [],
                    "decision": "Hire",
                },
            },
        ]
        self.assertEqual(rank(build_df(results)).iloc[0]["Candidate"], "Higher")


if __name__ == "__main__":
    unittest.main()
