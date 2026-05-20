# ML Inference & Evaluation Platform

This platform is a production-style MLOps system for serving, evaluating, and safely rolling out machine learning models. It provides online inference, batch evaluation, model registry management, regression gates, SLO checks, shadow testing, benchmarking, observability, deployment audit trails, and a placeholder multi-provider LLM comparison layer.

The goal is to demonstrate how real ML teams prevent unsafe model releases. Instead of promoting a model based only on accuracy, the platform compares accuracy, latency, throughput, cache behavior, error rate, and shadow disagreement before allowing a candidate model into production. This mirrors production ML infrastructure where model quality, reliability, cost, and performance all matter.

## LLM Provider Comparison

The dashboard also includes an LLM Providers tab for comparing hosted and local language models across providers such as OpenAI, Gemini, Anthropic, and a local mock model. It runs in offline mock mode by default, so the feature is safe to demo without API keys. You can enable live provider calls later by setting the following variables:

```bash
LLM_ENABLE_LIVE_PROVIDERS=true
OPENAI_API_KEY=...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
```
