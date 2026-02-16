"""Database engine and session management."""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.db.models import Base

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Return the SQLAlchemy engine (singleton)."""
    global _engine
    if _engine is None:
        settings = get_settings()
        connect_args: dict = {}
        if settings.database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _engine = create_engine(
            settings.database_url,
            connect_args=connect_args,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return a session factory bound to the engine."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory


def get_session() -> Session:
    """Create and return a new session."""
    return get_session_factory()()


def _find_alembic_ini() -> str | None:
    """Locate alembic.ini by walking up from this file's directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        candidate = os.path.join(here, "alembic.ini")
        if os.path.isfile(candidate):
            return candidate
        here = os.path.dirname(here)
    return None


def run_migrations() -> None:
    """Run Alembic 'upgrade head' programmatically.

    Raises FileNotFoundError if alembic.ini cannot be located.
    """
    from alembic import command
    from alembic.config import Config

    ini_path = _find_alembic_ini()
    if ini_path is None:
        raise FileNotFoundError("alembic.ini not found")

    cfg = Config(ini_path)
    cfg.set_main_option("sqlalchemy.url", get_settings().database_url)
    command.upgrade(cfg, "head")


def init_db() -> None:
    """Initialise the database.

    Attempts to run Alembic migrations first (production path).
    Falls back to ``Base.metadata.create_all()`` when Alembic is
    unavailable (e.g. in-memory test databases).
    """
    try:
        run_migrations()
    except Exception:
        Base.metadata.create_all(bind=get_engine())


def reset_engine() -> None:
    """Reset the engine singleton (useful for testing)."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
