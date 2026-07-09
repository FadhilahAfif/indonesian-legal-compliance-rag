"""Validate the versioned legal evaluation set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REQUIRED_FIELDS = {
    "id",
    "category",
    "question",
    "expected_answer",
    "regulation",
    "page",
    "article",
    "relevant_chunk",
    "answerable",
    "review_status",
}
ALLOWED_CATEGORIES = {
    "single_article",
    "user_case",
    "multi_context",
    "ambiguous",
    "out_of_scope",
}


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"line {line_number}: invalid JSON: {error}") from error
    return cases


def validate_cases(cases: list[dict], require_reviewed: bool = False) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()

    for line_number, case in enumerate(cases, 1):
        prefix = f"line {line_number}"
        missing = REQUIRED_FIELDS - case.keys()
        if missing:
            errors.append(f"{prefix}: missing {', '.join(sorted(missing))}")
            continue
        if case["id"] in seen_ids:
            errors.append(f"{prefix}: duplicate id {case['id']}")
        seen_ids.add(case["id"])
        if case["category"] not in ALLOWED_CATEGORIES:
            errors.append(f"{prefix}: invalid category {case['category']}")
        if not isinstance(case["answerable"], bool):
            errors.append(f"{prefix}: answerable must be boolean")
        for field in ("regulation", "page", "article", "relevant_chunk"):
            if not isinstance(case[field], list):
                errors.append(f"{prefix}: {field} must be a list")
        if case["answerable"] and any(
            not case[field] for field in ("expected_answer", "regulation", "page", "article", "relevant_chunk")
        ):
            errors.append(f"{prefix}: answerable case has incomplete ground truth")
        if not case["answerable"] and any(
            case[field] for field in ("regulation", "page", "article", "relevant_chunk")
        ):
            errors.append(f"{prefix}: unanswerable case must not contain citations")
        if case["review_status"] not in {"draft", "reviewed"}:
            errors.append(f"{prefix}: review_status must be draft or reviewed")

    counts = Counter(case.get("category") for case in cases)
    if len(cases) < 60:
        errors.append(f"need at least 60 cases; found {len(cases)}")
    if sum(not case.get("answerable", True) for case in cases) < 15:
        errors.append("need at least 15 unanswerable cases")
    missing_categories = ALLOWED_CATEGORIES - counts.keys()
    if missing_categories:
        errors.append(f"missing categories: {', '.join(sorted(missing_categories))}")
    unreviewed = sum(case.get("review_status") != "reviewed" for case in cases)
    if require_reviewed and unreviewed:
        errors.append(f"{unreviewed} cases have not been manually reviewed")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("data/eval_cases.jsonl"),
    )
    parser.add_argument("--require-reviewed", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.path)
    errors = validate_cases(cases, args.require_reviewed)
    if errors:
        raise SystemExit("Evaluation set validation failed:\n- " + "\n- ".join(errors))

    counts = Counter(case["category"] for case in cases)
    answerable = sum(case["answerable"] for case in cases)
    reviewed = sum(case["review_status"] == "reviewed" for case in cases)
    print(
        f"OK {len(cases)} cases; {answerable} answerable; "
        f"{len(cases) - answerable} unanswerable; {reviewed} reviewed"
    )
    print("Categories:", ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))


if __name__ == "__main__":
    main()
