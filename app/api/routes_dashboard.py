"""Server-rendered dashboard for the ML Inference Platform."""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import func

from app.db import repositories as repo
from app.db.models import BatchJob, GateResult, ShadowResult, SloPolicy
from app.db.session import get_session
from app.inference.cache import get_cache_stats
from app.llm.providers import get_llm_models, provider_status

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "templates")
templates = Jinja2Templates(directory=_TEMPLATE_DIR)


# -------------------------------------------------------------------
# Dashboard HTML
# -------------------------------------------------------------------


@router.get("", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    """Render the main dashboard page."""
    session = get_session()
    try:
        # Models
        models = repo.list_models(session)
        for m in models:
            m.metrics_parsed = json.loads(m.metrics) if m.metrics else {}  # type: ignore[attr-defined]

        # Prod model metrics
        prod_models = [m for m in models if m.status == "prod"]
        staging_models = [m for m in models if m.status == "staging"]
        prod_model = prod_models[0] if prod_models else None
        staging_model = staging_models[0] if staging_models else None
        prod_metrics: dict[str, Any] = {}
        if prod_model and prod_model.metrics:
            prod_metrics = json.loads(prod_model.metrics)
        staging_metrics: dict[str, Any] = {}
        if staging_model and staging_model.metrics:
            staging_metrics = json.loads(staging_model.metrics)

        # Gate results
        gate_results = list(
            session.query(GateResult)
            .order_by(GateResult.created_at.desc())
            .limit(50)
            .all()
        )
        for g in gate_results:
            g.details_parsed = json.loads(g.details) if g.details else {}  # type: ignore[attr-defined]

        # SLO policies
        slo_policies = list(session.query(SloPolicy).all())

        # Deployment events and batch status
        deployment_events = repo.list_deployment_events(session, limit=25)
        batch_rows = (
            session.query(BatchJob.status, func.count(BatchJob.id))
            .group_by(BatchJob.status)
            .all()
        )
        batch_status = {status: int(count) for status, count in batch_rows}
        batch_total = sum(batch_status.values())
        cache_stats = get_cache_stats()
        llm_models = get_llm_models()
        llm_provider_status = provider_status()

        # Shadow data
        shadow_pairs = _get_shadow_pairs(session)
        shadow_total = sum(s["total"] for s in shadow_pairs)
        shadow_agg_rate = 0.0
        if shadow_total > 0:
            total_agreed = sum(s["agreed"] for s in shadow_pairs)
            shadow_agg_rate = total_agreed / shadow_total

        # Chart data for shadow
        shadow_chart_agree = {
            "agreed": sum(s["agreed"] for s in shadow_pairs),
            "disagreed": shadow_total - sum(s["agreed"] for s in shadow_pairs),
        }
        shadow_chart_latency = {
            "labels": [
                f"{s['prod_version']} vs {s['shadow_version']}" for s in shadow_pairs
            ],
            "prod": [round(s["avg_prod_ms"], 2) for s in shadow_pairs],
            "shadow": [round(s["avg_shadow_ms"], 2) for s in shadow_pairs],
        }

        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "models": models,
                "prod_model": prod_model,
                "staging_model": staging_model,
                "prod_metrics": prod_metrics,
                "staging_metrics": staging_metrics,
                "gate_results": gate_results,
                "deployment_events": deployment_events,
                "slo_policies": slo_policies,
                "batch_status": batch_status,
                "batch_total": batch_total,
                "cache_stats": cache_stats,
                "llm_models": llm_models,
                "llm_provider_status": llm_provider_status,
                "shadow_pairs": shadow_pairs,
                "shadow_total": shadow_total,
                "shadow_agg_rate": shadow_agg_rate,
                "shadow_chart_agree": shadow_chart_agree,
                "shadow_chart_latency": shadow_chart_latency,
            },
        )
    finally:
        session.close()


def _get_shadow_pairs(session: Any) -> list[dict[str, Any]]:
    """Build per-(prod, shadow) pair summaries."""
    from sqlalchemy import case
    from sqlalchemy import func as sa_func

    rows = (
        session.query(
            ShadowResult.model_name,
            ShadowResult.prod_version,
            ShadowResult.shadow_version,
            sa_func.count().label("total"),
            sa_func.sum(
                case((ShadowResult.agreed == True, 1), else_=0)  # noqa: E712
            ).label("agreed_raw"),
            sa_func.avg(ShadowResult.prod_latency_ms).label("avg_prod_ms"),
            sa_func.avg(ShadowResult.shadow_latency_ms).label("avg_shadow_ms"),
        )
        .group_by(
            ShadowResult.model_name,
            ShadowResult.prod_version,
            ShadowResult.shadow_version,
        )
        .all()
    )
    pairs = []
    for r in rows:
        total = int(r.total)
        agreed = int(r.agreed_raw or 0)
        pairs.append(
            {
                "model_name": r.model_name,
                "prod_version": r.prod_version,
                "shadow_version": r.shadow_version,
                "total": total,
                "agreed": agreed,
                "rate": agreed / total if total > 0 else 0.0,
                "avg_prod_ms": float(r.avg_prod_ms or 0),
                "avg_shadow_ms": float(r.avg_shadow_ms or 0),
            }
        )
    return pairs


# -------------------------------------------------------------------
# Dashboard API actions (called via JS fetch)
# -------------------------------------------------------------------


class PromoteRequest(BaseModel):
    model_name: str
    model_version: str


class RollbackRequest(BaseModel):
    model_name: str


@router.post("/api/promote")
def api_promote(req: PromoteRequest) -> dict:
    from app.registry.manager import promote

    mv = promote(model_name=req.model_name, model_version=req.model_version)
    if mv is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Model not found")
    return {
        "model_name": mv.model_name,
        "model_version": mv.model_version,
        "status": mv.status,
    }


@router.post("/api/rollback")
def api_rollback(req: RollbackRequest) -> dict:
    from app.registry.manager import rollback_with_summary

    summary = rollback_with_summary(model_name=req.model_name)
    return {
        "model_name": summary.model_name,
        "model_version": summary.new_prod_version,
        "previous_prod_version": summary.previous_prod_version,
        "rolled_back": summary.rolled_back,
        "reason": summary.reason,
        "status": "prod" if summary.rolled_back else "unchanged",
    }
