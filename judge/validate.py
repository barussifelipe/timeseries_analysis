"""Validate a cold review against its schema and scoring rules."""

import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError


SCHEMA = json.loads(Path(__file__).with_name("score.schema.json").read_text(encoding="utf-8"))
WEIGHTS = {key: spec["maximum"] for key, spec in SCHEMA["properties"]["scores"]["properties"].items()}
Draft202012Validator.check_schema(SCHEMA)


def validate(review):
    Draft202012Validator(SCHEMA).validate(review)
    scores = review["scores"]
    findings = review["findings"]
    if review["total"] != sum(scores.values()):
        raise ValueError("total must equal the sum of scores")
    for category, maximum in WEIGHTS.items():
        if scores[category] < maximum and not any(f["category"] == category for f in findings):
            raise ValueError(f"deduction in {category} needs a finding")
    for finding in findings:
        if scores[finding["category"]] == WEIGHTS[finding["category"]]:
            raise ValueError(f"finding in {finding['category']} needs a deduction")
    expected_pass = review["total"] >= 95 and not any(f["severity"] == "critical" for f in findings)
    if review["pass"] != expected_pass:
        raise ValueError("pass must match the 95-point threshold and critical-finding rule")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python judge/validate.py REVIEW.json")
    try:
        review = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        validate(review)
    except (OSError, ValueError, TypeError, KeyError, ValidationError) as error:
        raise SystemExit(f"invalid review: {error}") from error
    print(f"valid review: {review['total']}/100, {'PASS' if review['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
