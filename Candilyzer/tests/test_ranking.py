import unittest

from src.ranking import build_df, evidence_level, rank, score_evaluation


class RankingTests(unittest.TestCase):
    def test_score_is_derived_from_requirement_evidence(self) -> None:
        evaluation = {
            "requirements_analysis": [
                {"importance": "Must Have", "match": "Met"},
                {"importance": "Must Have", "match": "Partial"},
                {"importance": "Nice to Have", "match": "Not Met"},
            ],
            "confidence": "High",
            "verification_points": [],
        }
        breakdown = score_evaluation(evaluation)
        self.assertEqual(breakdown["coverage"], 60.0)
        self.assertEqual(breakdown["score"], 60.0)
        self.assertEqual(breakdown["must_have_met"], 1)

    def test_evidence_level_is_not_a_hiring_decision(self) -> None:
        self.assertEqual(evidence_level(80.0), "Strong evidence")
        self.assertEqual(evidence_level(59.9), "Limited evidence")

    def test_rank_orders_by_score(self) -> None:
        results = [
            {
                "username": "Lower",
                "parsed": {
                    "requirements_analysis": [
                        {"importance": "Must Have", "match": "Partial"}
                    ],
                    "confidence": "High",
                    "verification_points": [],
                },
            },
            {
                "username": "Higher",
                "parsed": {
                    "requirements_analysis": [
                        {"importance": "Must Have", "match": "Met"}
                    ],
                    "confidence": "High",
                    "verification_points": [],
                },
            },
        ]
        self.assertEqual(rank(build_df(results)).iloc[0]["Candidate"], "Higher")


if __name__ == "__main__":
    unittest.main()
