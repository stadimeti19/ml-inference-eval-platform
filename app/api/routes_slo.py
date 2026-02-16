"""SLO policy management and evaluation endpoints."""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel

from app.core.logging import get_logger
from app.db import repositories as repo
from app.db.session import get_session
from app.eval.slo import run_slo_gate

logger = get_logger(__name__)
router = APIRouter(prefix="/slo", tags=["slo"])


# -------------------------------------------------------------------
# Request / Response schemas
# -------------------------------------------------------------------

class CreatePolicyRequest(BaseModel):
    name: str
    model_name: str
    constraints: dict[str, float]


class PolicyResponse(BaseModel):
    id: str
    name: str
    model_name: str
    constraints: dict[str, float]
    created_at: str


class SloCheckRequest(BaseModel):
    model_name: str
    model_version: str
    policy_name: str


class SloCheckResponse(BaseModel):
    passed: bool
    policy_name: str
    model_name: str
    model_version: str
    checks: list[dict[str, Any]]


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------

@router.post("/policies", response_model=PolicyResponse, status_code=201)
def create_policy(req: CreatePolicyRequest) -> PolicyResponse:
    """Create a new SLO policy."""
    session = get_session()
    try:
        existing = repo.get_slo_policy(session, name=req.name)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"SLO policy '{req.name}' already exists",
            )
        policy = repo.create_slo_policy(
            session,
            name=req.name,
            model_name=req.model_name,
            constraints=req.constraints,
        )
        return PolicyResponse(
            id=policy.id,
            name=policy.name,
            model_name=policy.model_name,
            constraints=json.loads(policy.constraints),
            created_at=str(policy.created_at),
        )
    finally:
        session.close()


@router.get("/policies", response_model=list[PolicyResponse])
def list_policies(
    model_name: Optional[str] = Query(None, description="Filter by model name"),
) -> list[PolicyResponse]:
    """List SLO policies, optionally filtered by model name."""
    session = get_session()
    try:
        if model_name:
            policies = repo.get_slo_policies_for_model(
                session, model_name=model_name
            )
        else:
            from app.db.models import SloPolicy
            policies = list(session.query(SloPolicy).all())

        return [
            PolicyResponse(
                id=p.id,
                name=p.name,
                model_name=p.model_name,
                constraints=json.loads(p.constraints),
                created_at=str(p.created_at),
            )
            for p in policies
        ]
    finally:
        session.close()


@router.get("/policies/{name}", response_model=PolicyResponse)
def get_policy(name: str) -> PolicyResponse:
    """Get a specific SLO policy by name."""
    session = get_session()
    try:
        policy = repo.get_slo_policy(session, name=name)
        if policy is None:
            raise HTTPException(
                status_code=404, detail=f"SLO policy '{name}' not found"
            )
        return PolicyResponse(
            id=policy.id,
            name=policy.name,
            model_name=policy.model_name,
            constraints=json.loads(policy.constraints),
            created_at=str(policy.created_at),
        )
    finally:
        session.close()


@router.delete("/policies/{name}")
def delete_policy(name: str) -> Response:
    """Delete an SLO policy by name."""
    session = get_session()
    try:
        deleted = repo.delete_slo_policy(session, name=name)
        if not deleted:
            raise HTTPException(
                status_code=404, detail=f"SLO policy '{name}' not found"
            )
        return Response(status_code=204)
    finally:
        session.close()


@router.post("/check", response_model=SloCheckResponse)
def check_slo(req: SloCheckRequest) -> SloCheckResponse:
    """Evaluate a model version against a named SLO policy.

    Returns pass/fail with per-constraint details.  The result is also
    persisted as a ``GateResult`` for audit.
    """
    result = run_slo_gate(
        model_name=req.model_name,
        model_version=req.model_version,
        policy_name=req.policy_name,
    )
    details = json.loads(result.details) if result.details else {}
    return SloCheckResponse(
        passed=result.passed,
        policy_name=req.policy_name,
        model_name=req.model_name,
        model_version=req.model_version,
        checks=details.get("checks", []),
    )
