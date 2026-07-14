from __future__ import annotations

import re
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


def test_docker_context_is_deny_by_default() -> None:
    entries = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert entries[0] == "**"
    assert "!runner/" in entries
    assert "!runner/Dockerfile" in entries
    assert not any(entry.startswith("!src") or entry.startswith("!tests") for entry in entries)


def test_ci_docker_build_has_explicit_context() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    match = re.search(
        r"- name: Build pinned tool runner\s+run: >-\s+"
        r"(?P<command>(?:docker build\s+|--[^\n]+\s+|\.\s*)+)",
        workflow,
    )
    assert match is not None
    assert match.group("command").split()[-1] == "."
