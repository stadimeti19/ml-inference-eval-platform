"""Online inference endpoints."""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.logging import get_logger
from app.core.metrics import REQUEST_COUNT, REQUEST_LATENCY
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
    image_b64: str


class PredictResponse(BaseModel):
    prediction: int
    latency_ms: float
    model_version: str


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------

@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Run single-image inference."""
    start = time.perf_counter()
    session = get_session()
    try:
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

        model = get_model_cached(mv.model_name, mv.model_version, mv.artifact_path)
        prediction, latency_ms = predict_single(model, req.image_b64)

        total_ms = (time.perf_counter() - start) * 1000.0
        REQUEST_LATENCY.labels(endpoint="/predict", model_name=req.model_name).observe(
            total_ms / 1000.0
        )
        REQUEST_COUNT.labels(
            endpoint="/predict", model_name=req.model_name, status="200"
        ).inc()

        return PredictResponse(
            prediction=prediction,
            latency_ms=round(latency_ms, 3),
            model_version=mv.model_version,
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
                "created_at": str(m.created_at),
            }
            for m in models
        ]
    finally:
        session.close()
