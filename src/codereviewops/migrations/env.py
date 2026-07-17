"""Credential-safe Alembic environment."""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from codereviewops.config import AppSettings
from codereviewops.db_models import Base

config = context.config
target_metadata = Base.metadata


def _settings() -> AppSettings:
    supplied = config.attributes.get("settings")
    return supplied if isinstance(supplied, AppSettings) else AppSettings.from_environment()


def run_migrations_offline() -> None:
    context.configure(
        url=_settings().database_url_value(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _settings().database_url_value()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
