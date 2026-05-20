"""High-level model registry operations."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db import repositories as repo
from app.db.models import ModelVersion
from app.db.session import get_session, init_db

logger = get_logger(__name__)


@dataclass(frozen=True)
class RollbackSummary:
    model_name: str
    previous_prod_version: str | None
    new_prod_version: str | None
    rolled_back: bool
    reason: str


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
        repo.create_deployment_event(
            session,
            model_name=model_name,
            version=model_version,
            previous_status=None,
            new_status=mv.status,
            event_type="register",
            reason=f"Registered artifact {dest_file}",
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
        before = repo.get_model(
            session, model_name=model_name, model_version=model_version
        )
        previous_status = before.status if before else None
        mv = repo.promote_model(
            session, model_name=model_name, model_version=model_version
        )
        if mv:
            repo.create_deployment_event(
                session,
                model_name=model_name,
                version=model_version,
                previous_status=previous_status,
                new_status="prod",
                event_type="promote",
                reason=f"Promoted {model_name}@{model_version} to production",
            )
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
    summary = rollback_with_summary(model_name)
    if not summary.rolled_back or summary.new_prod_version is None:
        return None
    session = get_session()
    try:
        return repo.get_model(
            session, model_name=model_name, model_version=summary.new_prod_version
        )
    finally:
        session.close()


def rollback_with_summary(model_name: str) -> RollbackSummary:
    """Rollback production to the previous prod version when available.

    The preferred rollback target comes from deployment audit history. If
    there is no previous prod event, the newest staging model is used as a
    conservative local-dev fallback.
    """
    init_db()
    session = get_session()
    try:
        current = repo.get_prod_model(session, model_name=model_name)
        if current is None:
            return RollbackSummary(
                model_name=model_name,
                previous_prod_version=None,
                new_prod_version=None,
                rolled_back=False,
                reason="No production model is currently set",
            )

        target_version = repo.get_previous_prod_version_from_events(
            session, model_name=model_name, current_version=current.model_version
        )
        if target_version is None:
            candidates = [
                m
                for m in repo.list_models(session, model_name=model_name)
                if m.status == "staging" and m.model_version != current.model_version
            ]
            if candidates:
                target_version = candidates[0].model_version

        if target_version is None:
            reason = "No previous production or staging rollback target is available"
            repo.create_deployment_event(
                session,
                model_name=model_name,
                version=current.model_version,
                previous_status="prod",
                new_status="prod",
                event_type="rollback",
                reason=reason,
            )
            return RollbackSummary(
                model_name=model_name,
                previous_prod_version=current.model_version,
                new_prod_version=current.model_version,
                rolled_back=False,
                reason=reason,
            )

        target = repo.get_model(
            session, model_name=model_name, model_version=target_version
        )
        if target is None:
            return RollbackSummary(
                model_name=model_name,
                previous_prod_version=current.model_version,
                new_prod_version=None,
                rolled_back=False,
                reason=f"Rollback target {target_version} not found",
            )

        previous_status = target.status
        mv = repo.promote_model(
            session, model_name=model_name, model_version=target_version
        )
        if mv:
            reason = (
                f"Rolled back {model_name} from {current.model_version} "
                f"to {target_version}"
            )
            repo.create_deployment_event(
                session,
                model_name=model_name,
                version=target_version,
                previous_status=previous_status,
                new_status="prod",
                event_type="rollback",
                reason=reason,
            )
            logger.info(
                "model_rolled_back",
                model_name=model_name,
                new_prod_version=mv.model_version,
            )
            return RollbackSummary(
                model_name=model_name,
                previous_prod_version=current.model_version,
                new_prod_version=target_version,
                rolled_back=True,
                reason=reason,
            )
        return RollbackSummary(
            model_name=model_name,
            previous_prod_version=current.model_version,
            new_prod_version=None,
            rolled_back=False,
            reason=f"Could not promote rollback target {target_version}",
        )
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
