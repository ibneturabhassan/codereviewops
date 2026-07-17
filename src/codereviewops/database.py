"""Synchronous PostgreSQL engine, session, and migration lifecycle."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from codereviewops.config import AppSettings

SessionFactory = sessionmaker[Session]


def create_database_engine(settings: AppSettings) -> Engine:
    return create_engine(
        settings.database_url_value(),
        pool_pre_ping=True,
    )


def create_session_factory(engine: Engine) -> SessionFactory:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope(factory: SessionFactory) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def alembic_config(path: Path | None = None) -> Config:
    configuration = Config(str(path or Path("alembic.ini")))
    configuration.set_main_option(
        "script_location",
        str(Path(__file__).with_name("migrations")),
    )
    return configuration


def upgrade_database(settings: AppSettings, revision: str = "head") -> None:
    configuration = alembic_config()
    configuration.attributes["settings"] = settings
    command.upgrade(configuration, revision)
