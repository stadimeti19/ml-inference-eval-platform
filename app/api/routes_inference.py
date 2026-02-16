"""Online inference endpoints with shadow/canary deployment support."""

from __future__ import annotations

import time
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.core.logging import get_logger
from app.core.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    SHADOW_DISAGREEMENT,
    SHADOW_LATENCY,
    SHADOW_REQUEST_COUNT,
)
from app.db import repositories as repo
from app.db.session import get_session
from app.inference.cache import get_model_cached
from app.inference.predict import predict_single

logger = get_logger(__name__)
router = APIRouter()


# -------------------------------------------------------------------
# Request / Response schemas
# -------------------------------------------------------------------

class PredictRequest(BaseModel):
    model_name: str
    model_version: Optional[str] = None
    shadow_version: Optional[str] = None
    image_b64: str


class ShadowDetail(BaseModel):
    shadow_version: str
    shadow_prediction: int
    shadow_latency_ms: float
    agreed: bool


class PredictResponse(BaseModel):
    prediction: int
    latency_ms: float
    model_version: str
    shadow: Optional[ShadowDetail] = None


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------

@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Run single-image inference, optionally with shadow model comparison.

    When ``shadow_version`` is provided, the candidate model runs in
    parallel (sequentially in this implementation) against the same
    input.  The **production** prediction is always returned to the
    caller; shadow results are logged for offline analysis.
    """
    start = time.perf_counter()
    session = get_session()
    try:
        # --- Resolve prod model ---
        if req.model_version:
            mv = repo.get_model(
                session, model_name=req.model_name, model_version=req.model_version
            )
        else:
            mv = repo.get_prod_model(session, model_name=req.model_name)

        if mv is None:
            REQUEST_COUNT.labels(
                endpoint="/predict", model_name=req.model_name, status="404"
            ).inc()
            raise HTTPException(
                status_code=404,
                detail=f"No model found for {req.model_name}"
                + (f"@{req.model_version}" if req.model_version else " (prod)"),
            )

        # --- Run prod inference ---
        model = get_model_cached(
            mv.model_name, mv.model_version, mv.artifact_path,
            architecture=mv.architecture,
        )
        prediction, latency_ms = predict_single(model, req.image_b64)

        total_ms = (time.perf_counter() - start) * 1000.0
        REQUEST_LATENCY.labels(endpoint="/predict", model_name=req.model_name).observe(
            total_ms / 1000.0
        )
        REQUEST_COUNT.labels(
            endpoint="/predict", model_name=req.model_name, status="200"
        ).inc()

        # --- Shadow inference (fire-and-forget, never affects prod response) ---
        shadow_detail: ShadowDetail | None = None
        if req.shadow_version:
            shadow_detail = _run_shadow(
                session=session,
                model_name=req.model_name,
                prod_version=mv.model_version,
                shadow_version=req.shadow_version,
                image_b64=req.image_b64,
                prod_prediction=prediction,
                prod_latency_ms=latency_ms,
            )

        return PredictResponse(
            prediction=prediction,
            latency_ms=round(latency_ms, 3),
            model_version=mv.model_version,
            shadow=shadow_detail,
        )
    finally:
        session.close()


def _run_shadow(
    *,
    session: Any,
    model_name: str,
    prod_version: str,
    shadow_version: str,
    image_b64: str,
    prod_prediction: int,
    prod_latency_ms: float,
) -> ShadowDetail | None:
    """Run the shadow model and record comparison.  Never raises."""
    try:
        shadow_mv = repo.get_model(
            session, model_name=model_name, model_version=shadow_version
        )
        if shadow_mv is None:
            logger.warning(
                "shadow_model_not_found",
                model_name=model_name,
                shadow_version=shadow_version,
            )
            return None

        shadow_model = get_model_cached(
            shadow_mv.model_name, shadow_mv.model_version,
            shadow_mv.artifact_path, architecture=shadow_mv.architecture,
        )
        shadow_pred, shadow_latency = predict_single(shadow_model, image_b64)
        agreed = prod_prediction == shadow_pred

        # Record metrics
        SHADOW_REQUEST_COUNT.labels(
            model_name=model_name, shadow_version=shadow_version
        ).inc()
        SHADOW_LATENCY.labels(
            model_name=model_name, shadow_version=shadow_version
        ).observe(shadow_latency / 1000.0)
        if not agreed:
            SHADOW_DISAGREEMENT.labels(
                model_name=model_name, shadow_version=shadow_version
            ).inc()

        # Persist to DB
        repo.save_shadow_result(
            session,
            model_name=model_name,
            prod_version=prod_version,
            shadow_version=shadow_version,
            prod_prediction=prod_prediction,
            shadow_prediction=shadow_pred,
            prod_latency_ms=prod_latency_ms,
            shadow_latency_ms=shadow_latency,
        )

        logger.info(
            "shadow_inference",
            model_name=model_name,
            prod_version=prod_version,
            shadow_version=shadow_version,
            prod_pred=prod_prediction,
            shadow_pred=shadow_pred,
            agreed=agreed,
        )

        return ShadowDetail(
            shadow_version=shadow_version,
            shadow_prediction=shadow_pred,
            shadow_latency_ms=round(shadow_latency, 3),
            agreed=agreed,
        )
    except Exception:
        logger.exception("shadow_inference_failed", shadow_version=shadow_version)
        return None


# -------------------------------------------------------------------
# Shadow results API
# -------------------------------------------------------------------

@router.get("/shadow/results")
def shadow_results(
    model_name: str = Query(..., description="Model name"),
    shadow_version: str = Query(..., description="Shadow model version"),
    prod_version: Optional[str] = Query(None, description="Filter by prod version"),
) -> dict:
    """Return aggregated shadow comparison metrics."""
    session = get_session()
    try:
        return repo.get_shadow_summary(
            session,
            model_name=model_name,
            shadow_version=shadow_version,
            prod_version=prod_version,
        )
    finally:
        session.close()


@router.get("/models")
def list_models() -> list[dict]:
    """List all registered model versions."""
    session = get_session()
    try:
        models = repo.list_models(session)
        return [
            {
                "model_name": m.model_name,
                "model_version": m.model_version,
                "status": m.status,
                "architecture": m.architecture,
                "created_at": str(m.created_at),
            }
            for m in models
        ]
    finally:
        session.close()
