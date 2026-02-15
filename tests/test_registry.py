"""Tests for the model registry."""

from __future__ import annotations

from app.db import repositories as repo


def test_register_model(db_session):
    mv = repo.register_model(
        db_session,
        model_name="test_model",
        model_version="v1.0.0",
        artifact_path="/tmp/model.pt",
        git_sha="abc123",
        tags={"env": "test"},
    )
    assert mv.model_name == "test_model"
    assert mv.model_version == "v1.0.0"
    assert mv.status == "staging"


def test_promote_model(db_session):
    repo.register_model(
        db_session,
        model_name="m",
        model_version="v1",
        artifact_path="/tmp/m.pt",
    )
    promoted = repo.promote_model(db_session, model_name="m", model_version="v1")
    assert promoted is not None
    assert promoted.status == "prod"


def test_rollback_model(db_session):
    repo.register_model(
        db_session,
        model_name="m",
        model_version="v1",
        artifact_path="/tmp/m1.pt",
    )
    repo.register_model(
        db_session,
        model_name="m",
        model_version="v2",
        artifact_path="/tmp/m2.pt",
    )
    repo.promote_model(db_session, model_name="m", model_version="v2")

    rolled = repo.rollback_model(db_session, model_name="m")
    assert rolled is not None
    # After rollback the most recent staging version should become prod
    assert rolled.status == "prod"


def test_list_models(db_session):
    repo.register_model(
        db_session, model_name="a", model_version="v1", artifact_path="/tmp/a.pt"
    )
    repo.register_model(
        db_session, model_name="a", model_version="v2", artifact_path="/tmp/a2.pt"
    )
    repo.register_model(
        db_session, model_name="b", model_version="v1", artifact_path="/tmp/b.pt"
    )

    all_models = repo.list_models(db_session)
    assert len(all_models) == 3

    a_models = repo.list_models(db_session, model_name="a")
    assert len(a_models) == 2


def test_get_prod_model(db_session):
    repo.register_model(
        db_session, model_name="m", model_version="v1", artifact_path="/tmp/m.pt"
    )
    assert repo.get_prod_model(db_session, model_name="m") is None

    repo.promote_model(db_session, model_name="m", model_version="v1")
    prod = repo.get_prod_model(db_session, model_name="m")
    assert prod is not None
    assert prod.model_version == "v1"
