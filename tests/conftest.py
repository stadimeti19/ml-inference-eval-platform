"""Shared test fixtures."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base
from app.inference.model import MNISTClassifier


@pytest.fixture()
def db_session() -> Session:
    """In-memory SQLite session with tables created."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture()
def tmp_artifacts(tmp_path):
    """Temporary artifacts directory."""
    d = tmp_path / "artifacts"
    d.mkdir()
    return str(d)


@pytest.fixture()
def sample_model(tmp_path) -> str:
    """Save a randomly-initialised MNISTClassifier and return its path."""
    model = MNISTClassifier()
    path = str(tmp_path / "model.pt")
    torch.save(model.state_dict(), path)
    return path


@pytest.fixture(autouse=True)
def _set_env_for_tests(tmp_path, monkeypatch):
    """Point DATABASE_URL to a temporary SQLite and artifacts to tmp."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("MODEL_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    (tmp_path / "artifacts").mkdir(exist_ok=True)

    # Reset the singleton engine so each test gets a fresh one
    from app.db import session as sess_mod
    sess_mod.reset_engine()
    yield
    sess_mod.reset_engine()
