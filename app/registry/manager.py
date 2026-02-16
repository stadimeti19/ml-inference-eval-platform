"""High-level model registry operations."""

from __future__ import annotations

import os
import shutil
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db import repositories as repo
from app.db.models import ModelVersion
from app.db.session import get_session, init_db

logger = get_logger(__name__)


def _artifact_dir(model_name: str, model_version: str) -> str:
    settings = get_settings()
    return os.path.join(settings.model_artifacts_dir, model_name, model_version)


def register(
    model_name: str,
    model_version: str,
    artifact_path: str,
    git_sha: str | None = None,
    tags: dict[str, Any] | None = None,
    architecture: str = "default",
) -> ModelVersion:
    """Copy a model artifact into the registry and record it in the DB."""
    init_db()

    dest_dir = _artifact_dir(model_name, model_version)
    os.makedirs(dest_dir, exist_ok=True)

    dest_file = os.path.join(dest_dir, "model.pt")
    if os.path.abspath(artifact_path) != os.path.abspath(dest_file):
        shutil.copy2(artifact_path, dest_file)

    session = get_session()
    try:
        existing = repo.get_model(
            session, model_name=model_name, model_version=model_version
        )
        if existing:
            logger.warning(
                "model_already_registered",
                model_name=model_name,
                model_version=model_version,
            )
            return existing

        mv = repo.register_model(
            session,
            model_name=model_name,
            model_version=model_version,
            artifact_path=dest_file,
            git_sha=git_sha,
            tags=tags,
            architecture=architecture,
        )
        logger.info(
            "model_registered",
            model_name=model_name,
            model_version=model_version,
            artifact_path=dest_file,
            architecture=architecture,
        )
        return mv
    finally:
        session.close()


def promote(model_name: str, model_version: str) -> ModelVersion | None:
    """Set *model_version* to production status."""
    init_db()
    session = get_session()
    try:
        mv = repo.promote_model(
            session, model_name=model_name, model_version=model_version
        )
        if mv:
            logger.info(
                "model_promoted",
                model_name=model_name,
                model_version=model_version,
            )
        return mv
    finally:
        session.close()


def rollback(model_name: str) -> ModelVersion | None:
    """Revert to the previous production version."""
    init_db()
    session = get_session()
    try:
        mv = repo.rollback_model(session, model_name=model_name)
        if mv:
            logger.info(
                "model_rolled_back",
                model_name=model_name,
                new_prod_version=mv.model_version,
            )
        return mv
    finally:
        session.close()


def list_models(model_name: str | None = None) -> list[ModelVersion]:
    """List registered model versions."""
    init_db()
    session = get_session()
    try:
        return repo.list_models(session, model_name=model_name)
    finally:
        session.close()
