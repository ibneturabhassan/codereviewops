"""Deterministic finding matching and evaluation."""

from __future__ import annotations

from collections.abc import Sequence

from codereviewops.models import (
    MAX_FINDINGS_PER_TASK,
    EvaluationResult,
    ExpectedFinding,
    Finding,
    FindingMatch,
    ProhibitedHit,
    ReviewReport,
)


def _overlaps(expected: ExpectedFinding, actual: Finding) -> bool:
    return (
        expected.category == actual.category
        and expected.file == actual.file
        and expected.line_start <= actual.line_end
        and actual.line_start <= expected.line_end
    )


def _maximum_matching(
    expected: Sequence[ExpectedFinding], actual: Sequence[Finding]
) -> list[FindingMatch]:
    candidates = [
        [actual_index for actual_index, finding in enumerate(actual) if _overlaps(label, finding)]
        for label in expected
    ]
    actual_to_expected: dict[int, int] = {}

    def augment(expected_index: int, seen_actual: set[int]) -> bool:
        for actual_index in candidates[expected_index]:
            if actual_index in seen_actual:
                continue
            seen_actual.add(actual_index)
            previous = actual_to_expected.get(actual_index)
            if previous is None or augment(previous, seen_actual):
                actual_to_expected[actual_index] = expected_index
                return True
        return False

    for expected_index in range(len(expected)):
        augment(expected_index, set())

    return [
        FindingMatch(expected_index=expected_index, actual_index=actual_index)
        for actual_index, expected_index in sorted(
            actual_to_expected.items(), key=lambda pair: (pair[1], pair[0])
        )
    ]


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _prohibited_hits(phrases: Sequence[str], findings: Sequence[Finding]) -> list[ProhibitedHit]:
    hits: list[ProhibitedHit] = []
    for actual_index, finding in enumerate(findings):
        searchable = _normalized_text(
            " ".join((finding.title, finding.evidence, finding.reasoning))
        )
        for phrase in phrases:
            normalized_phrase = _normalized_text(phrase)
            if normalized_phrase and normalized_phrase in searchable:
                hits.append(ProhibitedHit(phrase=phrase, actual_index=actual_index))
    return hits


def evaluate_review(
    expected: Sequence[ExpectedFinding], must_not_find: Sequence[str], review: ReviewReport
) -> EvaluationResult:
    """Compare a review to golden labels using one-to-one overlap matching."""

    if len(expected) > MAX_FINDINGS_PER_TASK:
        raise ValueError(f"expected findings exceed the limit of {MAX_FINDINGS_PER_TASK}")
    if len(review.findings) > MAX_FINDINGS_PER_TASK:
        raise ValueError(f"review findings exceed the limit of {MAX_FINDINGS_PER_TASK}")

    matched = _maximum_matching(expected, review.findings)
    matched_expected = {match.expected_index for match in matched}
    matched_actual = {match.actual_index for match in matched}
    missed = [index for index in range(len(expected)) if index not in matched_expected]
    hallucinated = [index for index in range(len(review.findings)) if index not in matched_actual]
    prohibited = _prohibited_hits(must_not_find, review.findings)
    true_positive = len(matched)
    actual_count = len(review.findings)
    expected_count = len(expected)
    precision = (
        true_positive / actual_count if actual_count else (1.0 if expected_count == 0 else 0.0)
    )
    recall = true_positive / expected_count if expected_count else 1.0
    hallucination_rate = len(hallucinated) / actual_count if actual_count else 0.0
    return EvaluationResult(
        schema_version="1.0",
        matched=matched,
        missed_expected_indices=missed,
        hallucinated_actual_indices=hallucinated,
        prohibited_hits=prohibited,
        precision=precision,
        recall=recall,
        hallucination_rate=hallucination_rate,
        task_success=not missed and not hallucinated and not prohibited,
    )
