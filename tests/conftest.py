from __future__ import annotations

from typing import Any

import pytest

from codereviewops.models import ExpectedFinding, Finding, ReviewReport


@pytest.fixture
def expected_factory():
    def build(**overrides: Any) -> ExpectedFinding:
        values: dict[str, Any] = {
            "category": "bug",
            "file": "src/service.py",
            "line_start": 10,
            "line_end": 20,
            "description": "A bug",
        }
        values.update(overrides)
        return ExpectedFinding.model_validate(values)

    return build


@pytest.fixture
def finding_factory():
    def build(**overrides: Any) -> Finding:
        values: dict[str, Any] = {
            "title": "A finding",
            "severity": "high",
            "category": "bug",
            "file": "src/service.py",
            "line_start": 10,
            "line_end": 20,
            "evidence": "Observed behavior",
            "reasoning": "It matters",
            "recommendation": "Fix it",
            "confidence": 0.9,
        }
        values.update(overrides)
        return Finding.model_validate(values)

    return build


@pytest.fixture
def report_factory(finding_factory):
    def build(findings: list[Finding] | None = None, **overrides: Any) -> ReviewReport:
        values: dict[str, Any] = {
            "schema_version": "1.0",
            "summary": "Review summary",
            "overall_assessment": "needs_changes",
            "findings": [finding_factory()] if findings is None else findings,
            "tests_run": [],
            "limitations": [],
        }
        values.update(overrides)
        return ReviewReport.model_validate(values)

    return build
