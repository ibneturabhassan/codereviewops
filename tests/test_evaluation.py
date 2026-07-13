from __future__ import annotations

import pytest

from codereviewops.evaluation import evaluate_review
from codereviewops.models import MAX_FINDINGS_PER_TASK


def test_matching_requires_category_file_and_inclusive_overlap(
    expected_factory, finding_factory, report_factory
) -> None:
    expected = [expected_factory(line_start=10, line_end=20)]
    touching = finding_factory(line_start=20, line_end=25)
    result = evaluate_review(expected, [], report_factory([touching]))
    assert [(match.expected_index, match.actual_index) for match in result.matched] == [(0, 0)]
    assert result.task_success


def test_matching_is_maximum_cardinality_one_to_one_and_deterministic(
    expected_factory, finding_factory, report_factory
) -> None:
    expected = [
        expected_factory(line_start=1, line_end=10, description="broad"),
        expected_factory(line_start=2, line_end=10, description="narrow"),
    ]
    actual = [
        finding_factory(line_start=2, line_end=10, title="shared"),
        finding_factory(line_start=1, line_end=1, title="broad only"),
    ]
    first = evaluate_review(expected, [], report_factory(actual))
    second = evaluate_review(expected, [], report_factory(actual))
    pairs = [(match.expected_index, match.actual_index) for match in first.matched]
    assert pairs == [(0, 1), (1, 0)]
    assert first == second
    assert first.task_success


def test_duplicate_actual_finding_is_hallucinated(
    expected_factory, finding_factory, report_factory
) -> None:
    result = evaluate_review(
        [expected_factory()],
        [],
        report_factory([finding_factory(), finding_factory(title="duplicate")]),
    )
    assert len(result.matched) == 1
    assert result.hallucinated_actual_indices == [1]
    assert result.precision == 0.5
    assert result.recall == 1.0
    assert result.hallucination_rate == 0.5
    assert not result.task_success


def test_category_or_file_mismatch_is_a_miss_and_hallucination(
    expected_factory, finding_factory, report_factory
) -> None:
    result = evaluate_review(
        [expected_factory()],
        [],
        report_factory([finding_factory(category="security", file="src/other.py")]),
    )
    assert result.missed_expected_indices == [0]
    assert result.hallucinated_actual_indices == [0]
    assert not result.task_success


@pytest.mark.parametrize(
    ("expected_count", "actual_count", "precision", "recall", "hallucination_rate"),
    [
        (0, 0, 1.0, 1.0, 0.0),
        (1, 0, 0.0, 0.0, 0.0),
        (0, 1, 0.0, 1.0, 1.0),
    ],
)
def test_zero_denominators(
    expected_factory,
    finding_factory,
    report_factory,
    expected_count: int,
    actual_count: int,
    precision: float,
    recall: float,
    hallucination_rate: float,
) -> None:
    expected = [expected_factory()] if expected_count else []
    actual = [finding_factory()] if actual_count else []
    result = evaluate_review(expected, [], report_factory(actual))
    assert result.precision == precision
    assert result.recall == recall
    assert result.hallucination_rate == hallucination_rate


def test_prohibited_phrase_uses_casefold_collapsed_whitespace_and_selected_fields(
    expected_factory, finding_factory, report_factory
) -> None:
    finding = finding_factory(
        title="Possible sql",
        evidence="SQL\n   INJECTION occurs here",
        reasoning="Unsupported claim",
    )
    result = evaluate_review([expected_factory()], ["sql injection"], report_factory([finding]))
    assert [(hit.phrase, hit.actual_index) for hit in result.prohibited_hits] == [
        ("sql injection", 0)
    ]
    assert not result.hallucinated_actual_indices
    assert not result.task_success


def test_hallucinated_finding_can_also_be_prohibited(finding_factory, report_factory) -> None:
    finding = finding_factory(title="SQL injection")
    result = evaluate_review([], ["sql injection"], report_factory([finding]))
    assert result.hallucinated_actual_indices == [0]
    assert len(result.prohibited_hits) == 1


def test_direct_evaluator_rejects_too_many_expected_findings(
    expected_factory,
    report_factory,
) -> None:
    expected = [expected_factory()] * (MAX_FINDINGS_PER_TASK + 1)
    with pytest.raises(ValueError, match="expected findings exceed"):
        evaluate_review(expected, [], report_factory([]))


def test_direct_evaluator_rejects_too_many_review_findings(
    finding_factory,
    report_factory,
) -> None:
    oversized_review = report_factory([]).model_copy(
        update={"findings": [finding_factory()] * (MAX_FINDINGS_PER_TASK + 1)}
    )
    with pytest.raises(ValueError, match="review findings exceed"):
        evaluate_review([], [], oversized_review)
