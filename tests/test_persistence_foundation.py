from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from alembic import command
from pydantic import SecretStr
from sqlalchemy import CheckConstraint, ForeignKeyConstraint
from typer.testing import CliRunner

from codereviewops.cli import app
from codereviewops.config import AppSettings, ConfigurationError
from codereviewops.database import alembic_config, create_database_engine
from codereviewops.db_models import Base

ROOT = Path(__file__).parents[1]
URL = "postgresql+psycopg://reviewer:top-secret@localhost/codereviewops"


def test_settings_are_environment_only_typed_and_secret(tmp_path: Path) -> None:
    settings = AppSettings.from_environment(
        {
            "CODEREVIEWOPS_DATABASE_URL": URL,
            "CODEREVIEWOPS_BENCHMARK_ROOT": str(tmp_path),
            "UNRELATED": "ignored",
        }
    )
    assert settings.benchmark_root == tmp_path.resolve()
    assert isinstance(settings.database_url, SecretStr)
    assert "top-secret" not in repr(settings)
    assert "top-secret" not in str(settings)
    assert settings.database_url_value() == URL


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {"CODEREVIEWOPS_DATABASE_URL": URL},
        {"CODEREVIEWOPS_BENCHMARK_ROOT": "."},
        {
            "CODEREVIEWOPS_DATABASE_URL": "sqlite:///local.db",
            "CODEREVIEWOPS_BENCHMARK_ROOT": ".",
        },
    ],
)
def test_settings_fail_safely_without_disclosing_values(environment: dict[str, str]) -> None:
    with pytest.raises(ConfigurationError) as captured:
        AppSettings.from_environment(environment)
    assert "top-secret" not in str(captured.value)
    assert "sqlite" not in str(captured.value)


def test_metadata_contains_exact_foundation_tables_and_restrictive_foreign_keys() -> None:
    assert set(Base.metadata.tables) == {
        "benchmark_definitions",
        "review_runs",
        "review_results",
        "findings",
        "tool_traces",
        "human_feedback",
        "idempotency_records",
    }
    for table in Base.metadata.tables.values():
        assert {"id", "created_at"} <= set(table.columns.keys())
        for constraint in table.constraints:
            if isinstance(constraint, ForeignKeyConstraint):
                assert constraint.ondelete == "RESTRICT"


def test_metadata_has_named_checks_uniques_and_required_indexes() -> None:
    checks = {
        constraint.name
        for table in Base.metadata.tables.values()
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert {
        "ck_review_runs_status_payload_consistency",
        "ck_review_runs_provider_model_consistency",
        "ck_findings_matched_expected_consistency",
        "ck_human_feedback_kind_finding_consistency",
        "ck_idempotency_records_exactly_one_resource",
    } <= checks
    indexes = {index.name for table in Base.metadata.tables.values() for index in table.indexes}
    assert {
        "ix_benchmark_definitions_catalog_filters",
        "ix_review_runs_status_created_id",
        "ix_findings_run_category_severity",
        "ix_human_feedback_run_created_id",
        "ix_idempotency_records_created_at",
    } <= indexes


def test_initial_migration_renders_postgresql_offline_without_url_disclosure(
    tmp_path: Path,
) -> None:
    settings = AppSettings(
        database_url=SecretStr(URL),
        benchmark_root=tmp_path,
    )
    configuration = alembic_config(ROOT / "alembic.ini")
    configuration.attributes["settings"] = settings
    output = io.StringIO()
    with redirect_stdout(output):
        command.upgrade(configuration, "head", sql=True)
    sql = output.getvalue()
    for table in Base.metadata.tables:
        assert f"CREATE TABLE {table}" in sql
    assert "DROP TABLE" not in sql
    assert "top-secret" not in sql
    assert "20260717_0001" in sql
    assert "uq_findings_run_id_id" in sql
    assert "fk_human_feedback_run_id_finding_id_findings" in sql
    assert "position" in sql.lower()

    metadata_checks = {
        constraint.name
        for table in Base.metadata.tables.values()
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert set(re.findall(r"CONSTRAINT (ck_[a-z0-9_]+) CHECK", sql)) == metadata_checks
    down = io.StringIO()
    with redirect_stdout(down):
        command.downgrade(configuration, "head:base", sql=True)
    assert all(f"DROP TABLE {table}" in down.getvalue() for table in Base.metadata.tables)


def test_db_upgrade_cli_missing_configuration_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODEREVIEWOPS_DATABASE_URL", raising=False)
    monkeypatch.delenv("CODEREVIEWOPS_BENCHMARK_ROOT", raising=False)
    result = CliRunner().invoke(app, ["db", "upgrade"])
    assert result.exit_code == 2
    assert "database upgrade failed" in result.output
    assert "DATABASE_URL" not in result.output


def test_database_driver_normalization_and_rejection(tmp_path: Path) -> None:
    bare = AppSettings(
        database_url=SecretStr("postgresql://reviewer:top-secret@localhost/db"),
        benchmark_root=tmp_path,
    )
    explicit = AppSettings(database_url=SecretStr(URL), benchmark_root=tmp_path)
    assert bare.database_url_value().startswith("postgresql+psycopg://")
    assert explicit.database_url_value() == URL
    assert "top-secret" not in repr(bare)
    engine = create_database_engine(bare)
    try:
        assert engine.url.drivername == "postgresql+psycopg"
    finally:
        engine.dispose()
    for url in (
        "postgresql+psycopg2://localhost/db",
        "postgresql+asyncpg://localhost/db",
        "sqlite:///local.db",
    ):
        with pytest.raises(ValueError):
            AppSettings(database_url=SecretStr(url), benchmark_root=tmp_path)


def test_feedback_ownership_and_posix_path_constraints_match_contract() -> None:
    findings = Base.metadata.tables["findings"]
    assert any(c.name == "uq_findings_run_id_id" for c in findings.constraints)
    feedback = Base.metadata.tables["human_feedback"]
    ownership = next(
        c
        for c in feedback.foreign_key_constraints
        if c.name == "fk_human_feedback_run_id_finding_id_findings"
    )
    assert [item.parent.name for item in ownership.elements] == ["run_id", "finding_id"]
    assert [item.target_fullname for item in ownership.elements] == [
        "findings.run_id",
        "findings.id",
    ]
    expression = str(
        next(c for c in findings.constraints if c.name == "ck_findings_file_relative").sqltext
    )
    assert "position(chr(92) in file) = 0" in expression
    assert "file !~ '^/'" in expression
    assert "file !~ '^[A-Za-z]:'" in expression
    assert "file !~ '(^|/)\\.\\.(/|$)'" in expression
    for malicious in (
        "..\\secret.py",
        "dir\\..\\secret.py",
        "\\absolute.py",
        "dir/../secret.py",
    ):
        assert "\\" in malicious or "/../" in malicious
