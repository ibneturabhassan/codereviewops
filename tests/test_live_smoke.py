from __future__ import annotations

import os

import pytest

from codereviewops.models import ReviewContext, ReviewReport
from codereviewops.providers import GroqProvider, MistralProvider

pytestmark = pytest.mark.live


def test_opt_in_live_provider_returns_valid_review() -> None:
    if os.environ.get("CODEREVIEWOPS_RUN_LIVE") != "1":
        pytest.skip("set CODEREVIEWOPS_RUN_LIVE=1 to enable one live request")
    provider_name = os.environ.get("CODEREVIEWOPS_LIVE_PROVIDER")
    model = os.environ.get("CODEREVIEWOPS_LIVE_MODEL")
    if provider_name not in {"groq", "mistral"} or not model:
        pytest.skip("live provider and model are not configured")
    key_name = "GROQ_API_KEY" if provider_name == "groq" else "MISTRAL_API_KEY"
    api_key = os.environ.get(key_name)
    if not api_key:
        pytest.skip(f"{key_name} is not configured")
    provider = (
        GroqProvider(model=model, api_key=api_key)
        if provider_name == "groq"
        else MistralProvider(model=model, api_key=api_key)
    )
    result = provider.review(
        ReviewContext(
            schema_version="1.0",
            task_id="live_smoke",
            title="Review one synthetic changed line",
            issue_description="The value must remain true.",
            diff_text=(
                "diff --git a/example.py b/example.py\n@@ -1 +1 @@\n-value = True\n+value = True"
            ),
        )
    )
    assert isinstance(result.report, ReviewReport)
    assert result.report.tests_run == []
