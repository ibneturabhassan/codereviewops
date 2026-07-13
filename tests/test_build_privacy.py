from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]
REQUIRED_PRIVATE_BUILD_EXCLUSIONS = {
    "/.codex",
    "/AGENTS.md",
    "/01_CodeReviewOps_SPEC.md",
    "/.uv-cache",
    "/.venv",
    "/artifacts",
    "/dist",
}


def test_hatch_build_globally_excludes_private_and_generated_paths() -> None:
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    exclusions = set(configuration["tool"]["hatch"]["build"]["exclude"])
    assert exclusions >= REQUIRED_PRIVATE_BUILD_EXCLUSIONS
