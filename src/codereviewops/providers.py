"""Review providers for Milestone 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from codereviewops.models import ReviewContext, ReviewReport


class ProviderError(ValueError):
    """Raised when a provider cannot produce a valid review."""


class ReviewProvider(Protocol):
    """Provider boundary that receives review context without golden labels."""

    def review(self, context: ReviewContext) -> ReviewReport:
        """Return a structured review for the supplied context."""
        ...


class ReplayProvider:
    """Deterministic provider backed by a validated local response file."""

    def __init__(self, replay_path: Path) -> None:
        try:
            raw = json.loads(replay_path.read_text(encoding="utf-8"))
            self._report = ReviewReport.model_validate(raw)
        except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
            raise ProviderError(f"invalid replay response {replay_path}: {exc}") from exc

    def review(self, context: ReviewContext) -> ReviewReport:
        del context
        return self._report.model_copy(deep=True)
