"""Provider adapters for hosted and local LLMs.

Live API calls are disabled by default. Without API keys, providers return
deterministic mock responses so the platform, dashboard, and tests work offline.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from app.core.config import Settings, get_settings


@dataclass(frozen=True)
class LLMModelConfig:
    id: str
    provider: str
    model: str
    display_name: str
    family: str
    status: str
    context_window: int
    input_cost_per_1k: float
    output_cost_per_1k: float
    supports_json: bool = True
    supports_tools: bool = False
    supports_vision: bool = False
    notes: str = ""


@dataclass(frozen=True)
class LLMGenerationResult:
    model_id: str
    provider: str
    output: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    live: bool
    error: str | None = None


DEFAULT_LLM_MODELS: tuple[LLMModelConfig, ...] = (
    LLMModelConfig(
        id="openai:gpt-4o-mini",
        provider="openai",
        model="gpt-4o-mini",
        display_name="GPT-4o mini",
        family="OpenAI GPT",
        status="candidate",
        context_window=128000,
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.00060,
        supports_tools=True,
        supports_vision=True,
        notes="Fast, cost-efficient OpenAI baseline.",
    ),
    LLMModelConfig(
        id="openai:gpt-4.1-mini",
        provider="openai",
        model="gpt-4.1-mini",
        display_name="GPT-4.1 mini",
        family="OpenAI GPT",
        status="candidate",
        context_window=1000000,
        input_cost_per_1k=0.00040,
        output_cost_per_1k=0.00160,
        supports_tools=True,
        supports_vision=True,
        notes="Higher-capability OpenAI comparison model.",
    ),
    LLMModelConfig(
        id="gemini:gemini-2.0-flash",
        provider="gemini",
        model="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        family="Google Gemini",
        status="candidate",
        context_window=1000000,
        input_cost_per_1k=0.00010,
        output_cost_per_1k=0.00040,
        supports_vision=True,
        notes="Low-latency Gemini comparison model.",
    ),
    LLMModelConfig(
        id="gemini:gemini-1.5-pro",
        provider="gemini",
        model="gemini-1.5-pro",
        display_name="Gemini 1.5 Pro",
        family="Google Gemini",
        status="candidate",
        context_window=1000000,
        input_cost_per_1k=0.00125,
        output_cost_per_1k=0.00500,
        supports_vision=True,
        notes="Larger Gemini model for quality-focused evals.",
    ),
    LLMModelConfig(
        id="anthropic:claude-3-5-haiku-latest",
        provider="anthropic",
        model="claude-3-5-haiku-latest",
        display_name="Claude 3.5 Haiku",
        family="Anthropic Claude",
        status="candidate",
        context_window=200000,
        input_cost_per_1k=0.00080,
        output_cost_per_1k=0.00400,
        supports_tools=True,
        supports_vision=True,
        notes="Fast Claude model for low-latency tasks.",
    ),
    LLMModelConfig(
        id="anthropic:claude-3-5-sonnet-latest",
        provider="anthropic",
        model="claude-3-5-sonnet-latest",
        display_name="Claude 3.5 Sonnet",
        family="Anthropic Claude",
        status="candidate",
        context_window=200000,
        input_cost_per_1k=0.00300,
        output_cost_per_1k=0.01500,
        supports_tools=True,
        supports_vision=True,
        notes="Quality-focused Claude comparison model.",
    ),
    LLMModelConfig(
        id="local:mock-llm",
        provider="local",
        model="mock-llm",
        display_name="Local Mock LLM",
        family="Local placeholder",
        status="dev",
        context_window=8192,
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        notes="Offline placeholder for local/self-hosted models.",
    ),
)


class BaseLLMProvider:
    provider_name = "base"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def generate(
        self,
        *,
        model: LLMModelConfig,
        prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMGenerationResult:
        return self._mock_generate(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )

    def _mock_generate(
        self,
        *,
        model: LLMModelConfig,
        prompt: str,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMGenerationResult:
        start = time.perf_counter()
        input_tokens = estimate_tokens(prompt)
        digest = hashlib.sha256(f"{model.id}:{prompt}".encode()).hexdigest()[:8]
        if json_mode:
            output = json.dumps({
                "summary": f"Mock response from {model.display_name}",
                "decision": "candidate",
                "trace_id": digest,
            })
        else:
            clipped = " ".join(prompt.split()[:32])
            output = (
                f"[mock:{model.id}] {clipped}"
                if clipped
                else f"[mock:{model.id}] Ready for provider integration."
            )
        output_tokens = min(max_tokens, estimate_tokens(output))
        latency_ms = (time.perf_counter() - start) * 1000.0
        return LLMGenerationResult(
            model_id=model.id,
            provider=model.provider,
            output=output,
            latency_ms=round(latency_ms, 3),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimate_cost(model, input_tokens, output_tokens),
            live=False,
        )


class OpenAIProvider(BaseLLMProvider):
    provider_name = "openai"

    def generate(self, **kwargs: Any) -> LLMGenerationResult:
        model = kwargs["model"]
        if not self.settings.llm_enable_live_providers or not self.settings.openai_api_key:
            return super().generate(**kwargs)
        start = time.perf_counter()
        prompt = kwargs["prompt"]
        max_tokens = kwargs["max_tokens"]
        try:
            response = httpx.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                json={
                    "model": model.model,
                    "input": prompt,
                    "temperature": kwargs["temperature"],
                    "max_output_tokens": max_tokens,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            body = response.json()
            output = body.get("output_text") or _extract_openai_text(body)
            return _result_from_output(model, prompt, output, start, live=True)
        except Exception as exc:
            fallback = super().generate(**kwargs)
            return LLMGenerationResult(**{**asdict(fallback), "error": str(exc)})


class GeminiProvider(BaseLLMProvider):
    provider_name = "gemini"

    def generate(self, **kwargs: Any) -> LLMGenerationResult:
        model = kwargs["model"]
        if not self.settings.llm_enable_live_providers or not self.settings.gemini_api_key:
            return super().generate(**kwargs)
        start = time.perf_counter()
        prompt = kwargs["prompt"]
        try:
            response = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model.model}:generateContent",
                params={"key": self.settings.gemini_api_key},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=60.0,
            )
            response.raise_for_status()
            body = response.json()
            output = _extract_gemini_text(body)
            return _result_from_output(model, prompt, output, start, live=True)
        except Exception as exc:
            fallback = super().generate(**kwargs)
            return LLMGenerationResult(**{**asdict(fallback), "error": str(exc)})


class AnthropicProvider(BaseLLMProvider):
    provider_name = "anthropic"

    def generate(self, **kwargs: Any) -> LLMGenerationResult:
        model = kwargs["model"]
        if not self.settings.llm_enable_live_providers or not self.settings.anthropic_api_key:
            return super().generate(**kwargs)
        start = time.perf_counter()
        prompt = kwargs["prompt"]
        max_tokens = kwargs["max_tokens"]
        try:
            response = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.settings.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": model.model,
                    "max_tokens": max_tokens,
                    "temperature": kwargs["temperature"],
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60.0,
            )
            response.raise_for_status()
            body = response.json()
            output = _extract_anthropic_text(body)
            return _result_from_output(model, prompt, output, start, live=True)
        except Exception as exc:
            fallback = super().generate(**kwargs)
            return LLMGenerationResult(**{**asdict(fallback), "error": str(exc)})


class LocalProvider(BaseLLMProvider):
    provider_name = "local"


def get_llm_models() -> list[LLMModelConfig]:
    return list(DEFAULT_LLM_MODELS)


def get_llm_model(model_id: str) -> LLMModelConfig | None:
    return next((m for m in DEFAULT_LLM_MODELS if m.id == model_id), None)


def get_provider(model: LLMModelConfig, settings: Settings | None = None) -> BaseLLMProvider:
    providers: dict[str, type[BaseLLMProvider]] = {
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        "anthropic": AnthropicProvider,
        "local": LocalProvider,
    }
    return providers.get(model.provider, BaseLLMProvider)(settings=settings)


def provider_status(settings: Settings | None = None) -> list[dict[str, Any]]:
    settings = settings or get_settings()
    key_map = {
        "openai": bool(settings.openai_api_key),
        "gemini": bool(settings.gemini_api_key),
        "anthropic": bool(settings.anthropic_api_key),
        "local": True,
    }
    return [
        {
            "provider": provider,
            "configured": configured,
            "live_enabled": settings.llm_enable_live_providers and configured,
            "mode": "live"
            if settings.llm_enable_live_providers and configured
            else "mock",
        }
        for provider, configured in key_map.items()
    ]


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def estimate_cost(model: LLMModelConfig, input_tokens: int, output_tokens: int) -> float:
    return round(
        (input_tokens / 1000.0 * model.input_cost_per_1k)
        + (output_tokens / 1000.0 * model.output_cost_per_1k),
        8,
    )


def _result_from_output(
    model: LLMModelConfig,
    prompt: str,
    output: str,
    start: float,
    *,
    live: bool,
) -> LLMGenerationResult:
    input_tokens = estimate_tokens(prompt)
    output_tokens = estimate_tokens(output)
    return LLMGenerationResult(
        model_id=model.id,
        provider=model.provider,
        output=output,
        latency_ms=round((time.perf_counter() - start) * 1000.0, 3),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimate_cost(model, input_tokens, output_tokens),
        live=live,
    )


def _extract_openai_text(body: dict[str, Any]) -> str:
    texts = []
    for item in body.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                texts.append(content.get("text", ""))
    return "\n".join(t for t in texts if t) or json.dumps(body)[:1000]


def _extract_gemini_text(body: dict[str, Any]) -> str:
    parts = body.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "\n".join(p.get("text", "") for p in parts if p.get("text")) or json.dumps(body)[:1000]


def _extract_anthropic_text(body: dict[str, Any]) -> str:
    parts = body.get("content", [])
    return "\n".join(p.get("text", "") for p in parts if p.get("type") == "text") or json.dumps(body)[:1000]
