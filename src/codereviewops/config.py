"""Environment-only application configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from sqlalchemy.engine import make_url

from codereviewops.models import StrictModel

DATABASE_URL_ENV = "CODEREVIEWOPS_DATABASE_URL"
BENCHMARK_ROOT_ENV = "CODEREVIEWOPS_BENCHMARK_ROOT"


class ConfigurationError(ValueError):
    """Safe configuration failure without secret values."""


class AppSettings(StrictModel):
    database_url: SecretStr = Field(min_length=1)
    benchmark_root: Path

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: SecretStr) -> SecretStr:
        try:
            url = make_url(value.get_secret_value())
        except Exception as exc:
            raise ValueError("database URL is invalid") from exc
        if url.drivername not in {"postgresql", "postgresql+psycopg"}:
            raise ValueError("database URL must use PostgreSQL with psycopg")
        if url.drivername == "postgresql":
            url = url.set(drivername="postgresql+psycopg")
        return SecretStr(url.render_as_string(hide_password=False))

    @field_validator("benchmark_root")
    @classmethod
    def normalize_benchmark_root(cls, value: Path) -> Path:
        root = Path(os.path.abspath(value))
        if not root.is_dir():
            raise ValueError("benchmark root must be an existing directory")
        return root

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> AppSettings:
        source = os.environ if environ is None else environ
        database_url = source.get(DATABASE_URL_ENV)
        benchmark_root = source.get(BENCHMARK_ROOT_ENV)
        if not database_url or not benchmark_root:
            raise ConfigurationError("required CodeReviewOps environment configuration is missing")
        try:
            return cls(database_url=SecretStr(database_url), benchmark_root=Path(benchmark_root))
        except ValueError as exc:
            raise ConfigurationError("CodeReviewOps environment configuration is invalid") from exc

    def database_url_value(self) -> str:
        return self.database_url.get_secret_value()
