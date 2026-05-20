"""Multi-provider LLM inference and comparison endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.llm.eval import compare_llm_models
from app.llm.providers import (
    get_llm_model,
    get_llm_models,
    get_provider,
    provider_status,
)

router = APIRouter(prefix="/llm", tags=["llm"])


class LLMGenerateRequest(BaseModel):
    model: str = Field(..., description="Provider-qualified model id")
    prompt: str
    temperature: float = 0.2
    max_tokens: int = 512
    json_mode: bool = False


class LLMCompareRequest(BaseModel):
    models: list[str] = Field(..., min_length=1)
    prompt: str
    task: str = "general"
    temperature: float = 0.2
    max_tokens: int = 512
    json_mode: bool = False


@router.get("/models")
def list_llm_models() -> list[dict[str, Any]]:
    """List configured hosted/local LLM model candidates."""
    return [
        {
            "id": m.id,
            "provider": m.provider,
            "model": m.model,
            "display_name": m.display_name,
            "family": m.family,
            "status": m.status,
            "context_window": m.context_window,
            "input_cost_per_1k": m.input_cost_per_1k,
            "output_cost_per_1k": m.output_cost_per_1k,
            "supports_json": m.supports_json,
            "supports_tools": m.supports_tools,
            "supports_vision": m.supports_vision,
            "notes": m.notes,
        }
        for m in get_llm_models()
    ]


@router.get("/provider-status")
def llm_provider_status() -> list[dict[str, Any]]:
    """Return whether provider API keys are configured and live mode is enabled."""
    return provider_status()


@router.post("/generate")
def generate(req: LLMGenerateRequest) -> dict[str, Any]:
    """Generate from one provider-backed LLM.

    Without provider keys, this returns deterministic mock output. To enable
    live calls later, set `LLM_ENABLE_LIVE_PROVIDERS=true` and the relevant
    provider API key env var.
    """
    model = get_llm_model(req.model)
    if model is None:
        raise HTTPException(status_code=404, detail=f"LLM model '{req.model}' not found")
    provider = get_provider(model)
    result = provider.generate(
        model=model,
        prompt=req.prompt,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        json_mode=req.json_mode,
    )
    return {
        "model_id": result.model_id,
        "provider": result.provider,
        "output": result.output,
        "latency_ms": result.latency_ms,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "estimated_cost_usd": result.estimated_cost_usd,
        "live": result.live,
        "error": result.error,
    }


@router.post("/compare")
def compare(req: LLMCompareRequest) -> dict[str, Any]:
    """Compare the same prompt across provider models."""
    return compare_llm_models(
        model_ids=req.models,
        prompt=req.prompt,
        task=req.task,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        json_mode=req.json_mode,
    )
