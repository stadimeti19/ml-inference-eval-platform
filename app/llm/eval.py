"""Lightweight LLM comparison and placeholder scoring utilities."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from app.llm.providers import (
    LLMGenerationResult,
    get_llm_model,
    get_provider,
)


def compare_llm_models(
    *,
    model_ids: list[str],
    prompt: str,
    task: str = "general",
    temperature: float = 0.2,
    max_tokens: int = 512,
    json_mode: bool = False,
) -> dict[str, Any]:
    """Run the same prompt across providers and rank candidate models.

    The quality and bias scores are placeholders. They give the platform a
    stable output shape now, and can later be replaced by deterministic graders,
    human labels, or LLM-as-judge evals.
    """
    results: list[dict[str, Any]] = []
    for model_id in model_ids:
        model = get_llm_model(model_id)
        if model is None:
            results.append(
                {
                    "model_id": model_id,
                    "error": "model not registered",
                    "quality_score": 0.0,
                    "bias_risk_score": 1.0,
                }
            )
            continue
        provider = get_provider(model)
        generation = provider.generate(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        results.append(_score_result(generation, task=task, json_mode=json_mode))

    ranked = sorted(
        results,
        key=lambda r: (
            float(r.get("quality_score", 0.0)),
            -float(r.get("bias_risk_score", 1.0)),
            -float(r.get("estimated_cost_usd", 999.0)),
            -float(r.get("latency_ms", 999999.0)),
        ),
        reverse=True,
    )
    best = ranked[0] if ranked else None
    return {
        "task": task,
        "prompt_chars": len(prompt),
        "models_compared": len(model_ids),
        "recommended_model": best.get("model_id") if best else None,
        "recommendation_reason": _recommendation_reason(best),
        "results": ranked,
    }


def _score_result(
    generation: LLMGenerationResult,
    *,
    task: str,
    json_mode: bool,
) -> dict[str, Any]:
    data = asdict(generation)
    data["quality_score"] = _placeholder_quality_score(
        output=generation.output,
        error=generation.error,
        json_mode=json_mode,
    )
    data["bias_risk_score"] = _placeholder_bias_risk_score(generation.output)
    data["task"] = task
    return data


def _placeholder_quality_score(
    *,
    output: str,
    error: str | None,
    json_mode: bool,
) -> float:
    if error:
        return 0.0
    if not output.strip():
        return 0.0
    score = 0.70
    if len(output.split()) >= 12:
        score += 0.10
    if json_mode:
        import json

        try:
            json.loads(output)
            score += 0.15
        except Exception:
            score -= 0.20
    if "mock:" in output:
        score -= 0.05
    return round(max(0.0, min(score, 1.0)), 4)


def _placeholder_bias_risk_score(output: str) -> float:
    text = output.lower()
    risky_terms = [
        "always",
        "never",
        "obviously",
        "everyone",
        "nobody",
        "inferior",
    ]
    hits = sum(1 for term in risky_terms if term in text)
    return round(min(1.0, hits * 0.12), 4)


def _recommendation_reason(best: dict[str, Any] | None) -> str:
    if not best:
        return "No models were compared."
    if best.get("error"):
        return "No healthy model response was available."
    return (
        f"Selected {best['model_id']} based on placeholder quality, "
        "latency, cost, and bias-risk scoring."
    )
