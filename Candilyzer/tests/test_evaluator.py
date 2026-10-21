import json
import unittest

from src.evaluator import EvaluationValidationError, parse_evaluation


def valid_payload() -> dict:
    return {
        "requirements_analysis": [
            {
                "requirement": "Python",
                "importance": "Must Have",
                "match": "Met",
                "evidence": "Resume: shipped a Python API.",
            }
        ],
        "strengths": ["Production API experience"],
        "gaps": [],
        "verification_points": [],
        "overall_summary": "The candidate meets the demonstrated requirement.",
        "confidence": "High",
        "confidence_reason": "Two sources agree.",
    }


class EvaluationParserTests(unittest.TestCase):
    def test_parses_fenced_json(self) -> None:
        parsed = parse_evaluation(f"```json\n{json.dumps(valid_payload())}\n```")
        self.assertEqual(parsed["confidence"], "High")
        self.assertEqual(parsed["requirements_analysis"][0]["match"], "Met")

    def test_ignores_text_after_json_object(self) -> None:
        parsed = parse_evaluation(f"{json.dumps(valid_payload())}\nThis should be ignored")
        self.assertEqual(parsed["confidence"], "High")

    def test_rejects_unknown_confidence_value(self) -> None:
        payload = valid_payload()
        payload["confidence"] = "Certain"
        with self.assertRaises(EvaluationValidationError):
            parse_evaluation(json.dumps(payload))

    def test_rejects_non_string_verification_points(self) -> None:
        payload = valid_payload()
        payload["verification_points"] = [{"claim": "Python"}]
        with self.assertRaises(EvaluationValidationError):
            parse_evaluation(json.dumps(payload))

    def test_requires_requirement_evidence(self) -> None:
        payload = valid_payload()
        payload["requirements_analysis"][0]["evidence"] = ""
        with self.assertRaises(EvaluationValidationError):
            parse_evaluation(json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
