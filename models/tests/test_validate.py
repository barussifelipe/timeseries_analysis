import unittest

from judge.validate import validate


class ReviewTest(unittest.TestCase):
    def test_scoring_rules(self):
        review = {
            "task": "Example task",
            "sources": ["task request", "implementation diff"],
            "scores": {"requirements": 30, "correctness": 25, "verification": 20, "grounding": 15, "maintainability": 10},
            "findings": [],
            "total": 100,
            "pass": True,
        }
        validate(review)
        review["scores"]["correctness"] = 24
        review["total"] = 99
        review["findings"] = [{"category": "correctness", "severity": "critical", "evidence": "Broken output", "fix": "Correct the output"}]
        review["pass"] = False
        validate(review)
        review["pass"] = True
        with self.assertRaises(ValueError):
            validate(review)
        review["pass"] = False
        review["findings"] = []
        with self.assertRaises(ValueError):
            validate(review)
        review["scores"]["correctness"] = 20
        review["total"] = 95
        review["pass"] = True
        review["findings"] = [{"category": "correctness", "severity": "minor", "evidence": "Edge case", "fix": "Handle it"}]
        validate(review)
        review["scores"]["correctness"] = 19
        review["total"] = 94
        with self.assertRaises(ValueError):
            validate(review)


if __name__ == "__main__":
    unittest.main()
