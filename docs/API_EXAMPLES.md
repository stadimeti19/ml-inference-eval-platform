# API examples

The interactive OpenAPI explorer at <http://localhost:8000/docs> is the complete
and authoritative route reference. These examples show the main workflows.

Set the base URL once:

```bash
export PLATFORM_URL=http://localhost:8000
```

## Inspect service and models

```bash
curl -s "$PLATFORM_URL/health" | python -m json.tool
curl -s "$PLATFORM_URL/models" | python -m json.tool
curl -s "$PLATFORM_URL/metrics/inference" | python -m json.tool
```

The Prometheus exposition endpoint is separate from the JSON observability
routes:

```bash
curl -s "$PLATFORM_URL/metrics"
```

## Online inference

`image_b64` is a base64-encoded grayscale image. Omitting `model_version` routes
the request to the current production version. Add `shadow_version` to evaluate
a candidate while preserving the production response.

```bash
curl -s -X POST "$PLATFORM_URL/predict" \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: demo-prediction-001' \
  -d '{
    "model_name": "mnist_cnn",
    "image_b64": "<base64-image>",
    "shadow_version": "v2.0.0"
  }' | python -m json.tool
```

Inspect accumulated shadow comparisons:

```bash
curl -s "$PLATFORM_URL/shadow/summary" | python -m json.tool
curl -s "$PLATFORM_URL/shadow/results?limit=20" | python -m json.tool
```

## Submit a batch evaluation

Start the RQ worker before submitting when running without Docker:

```bash
python -m app.jobs.worker
```

Submit and then poll the returned job ID:

```bash
curl -s -X POST "$PLATFORM_URL/batch/submit" \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"mnist_cnn","dataset_id":"mnist_10000"}' \
  | python -m json.tool

curl -s "$PLATFORM_URL/batch/status/<job-id>" | python -m json.tool
```

## Define and enforce an SLO

```bash
curl -s -X POST "$PLATFORM_URL/slo/policies" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "interactive-mnist",
    "model_name": "mnist_cnn",
    "constraints": {"accuracy_min": 0.97, "p95_ms_max": 15.0}
  }' | python -m json.tool

curl -s -X POST "$PLATFORM_URL/slo/check" \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "mnist_cnn",
    "model_version": "v2.0.0",
    "policy_name": "interactive-mnist"
  }' | python -m json.tool
```

## Promote and roll back

The dashboard actions expose the same registry operations used by the CLI:

```bash
curl -s -X POST "$PLATFORM_URL/dashboard/api/promote" \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"mnist_cnn","model_version":"v2.0.0"}' \
  | python -m json.tool

curl -s -X POST "$PLATFORM_URL/models/mnist_cnn/rollback" \
  | python -m json.tool

curl -s "$PLATFORM_URL/deployments/events?model_name=mnist_cnn" \
  | python -m json.tool
```

## Compare LLM providers

This runs offline mock adapters unless live provider mode and credentials are
enabled.

```bash
curl -s -X POST "$PLATFORM_URL/llm/compare" \
  -H 'Content-Type: application/json' \
  -d '{
    "models": [
      "openai:gpt-4o-mini",
      "gemini:gemini-2.0-flash",
      "anthropic:claude-3-5-haiku-latest"
    ],
    "prompt": "Classify this support request and recommend the next action.",
    "task": "support_triage",
    "json_mode": true
  }' | python -m json.tool
```

Provider readiness and mock/live mode can be checked independently:

```bash
curl -s "$PLATFORM_URL/llm/provider-status" | python -m json.tool
```

## Progressive release control

Register the exact evaluation dataset, then save append-only reports for the
baseline and candidate. Repeating an identical report request returns the same
content-addressed report.

```bash
curl -s -X POST "$PLATFORM_URL/release/datasets" \
  -H 'Content-Type: application/json' \
  -d '{"name":"mnist_test","version":"v1","uri":"torchvision://MNIST/test","checksum":"builtin:mnist-test-v1"}'

curl -s -X POST "$PLATFORM_URL/release/evaluations" \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"mnist_cnn","model_version":"v2.0.0","dataset_version_id":"<dataset-id>","metrics":{"accuracy":0.98,"p95_ms":12.0}}'

curl -s "$PLATFORM_URL/release/evaluation-runs?model_name=mnist_cnn" \
  | python -m json.tool
```

Create a reusable policy and a deployment. The policy supports `min`, `max`,
`max_drop`, and `max_increase_pct` rules.

```bash
curl -s -X POST "$PLATFORM_URL/release/gate-policies" \
  -H 'Content-Type: application/json' \
  -d '{"name":"safe-release","model_name":"mnist_cnn","constraints":{"accuracy":{"min":0.97,"max_drop":0.01},"p95_ms":{"max_increase_pct":10}}}'

curl -s -X POST "$PLATFORM_URL/release/deployments" \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"mnist_cnn","candidate_version":"v2.0.0","min_requests":20,"max_error_rate":0.02,"max_avg_latency_ms":20}'

curl -s -X POST "$PLATFORM_URL/release/deployments/mnist_cnn/evaluate" \
  -H 'Content-Type: application/json' \
  -d '{"policy_name":"safe-release","candidate_report_id":"<candidate-report-id>","baseline_report_id":"<baseline-report-id>"}'
```

Advance an approved deployment through shadow and canary. Invalid state jumps
are rejected. Canary assignment uses `routing_key`, or the request ID when it is
omitted, so the same caller is routed consistently.

```bash
curl -s -X POST "$PLATFORM_URL/release/deployments/mnist_cnn/transition" \
  -H 'Content-Type: application/json' \
  -d '{"target_state":"shadow","reason":"offline shadow checks passed"}'

curl -s -X POST "$PLATFORM_URL/release/deployments/mnist_cnn/transition" \
  -H 'Content-Type: application/json' \
  -d '{"target_state":"canary","reason":"begin 10% rollout","traffic_percentage":10}'
```

Canary requests update persisted error and latency health. Once `min_requests`
is reached, a breached threshold sets the deployment to `rolled_back`, stops
candidate traffic, records an audit event, and continues serving production.
