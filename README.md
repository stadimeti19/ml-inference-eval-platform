# ml-inference-eval-platform

A production-grade **Foundation Model Inference & Evaluation Platform** built with FastAPI, PyTorch, PostgreSQL, Redis, and RQ. Designed for teams that need versioned model management, real-time and batch inference, automated regression gates, and observability — without the bloat.

---

## Architecture

```
                          ┌──────────────┐
                          │ platform_cli │
                          │  (Click CLI) │
                          └──────┬───────┘
                   register │ promote │ rollback │ gate
                             │
              ┌──────────────▼──────────────┐
              │        Model Registry       │
              │  artifacts/{name}/{version}  │
              └──────────────┬──────────────┘
                             │ metadata
         ┌───────────────────▼───────────────────┐
         │             PostgreSQL                 │
         │  model_versions │ batch_jobs │ gates   │
         └───────────────────┬───────────────────-┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                     │
   ┌────▼─────┐     ┌───────▼────────┐    ┌──────▼───────┐
   │ FastAPI   │     │  RQ Worker     │    │  Eval /      │
   │ /predict  │     │  batch jobs    │    │  Regression  │
   │ /batch    │     │                │    │  Gates       │
   │ /health   │     └───────┬────────┘    └──────────────┘
   │ /metrics  │             │
   └────┬──────┘       ┌─────▼─────┐
        │              │   Redis   │
        │              │   Queue   │
   ┌────▼──────┐       └───────────┘
   │Prometheus │
   │  metrics  │
   └───────────┘
```

## Features

| Feature | Description |
|---|---|
| **Model Registry** | Versioned model storage with promote/rollback lifecycle |
| **Online Inference** | Real-time `/predict` API with model caching |
| **Batch Inference** | Async jobs via Redis Queue with full metric computation |
| **Regression Gates** | Automated accuracy & latency checks before promotion |
| **Load Testing** | Concurrent request harness with QPS & percentile reporting |
| **Observability** | Prometheus metrics, structured JSON logs, Grafana dashboard |
| **Docker** | Full docker-compose stack (API + Worker + Postgres + Redis) |

---

## Project Structure

```
app/
    api/              FastAPI routers (inference, batch, health)
    core/             Config, structured logging, Prometheus metrics
    db/               SQLAlchemy models, session management, repositories
    registry/         Model versioning logic
    inference/        PyTorch model definition, prediction, caching
    jobs/             RQ queue, worker, batch task
    eval/             Evaluation metrics, regression gates
    datasets/         MNIST download & loading

platform_cli/         CLI entrypoint (python -m platform_cli)
scripts/              Training script, load test harness
ops/                  Dockerfile, docker-compose, Grafana dashboard
tests/                Unit tests
artifacts/            Model artifacts (gitignored)
reports/              Load test reports (gitignored)
```

---

## Quickstart (Docker)

```bash
# 1. Start the full stack
make up

# 2. Train and register the demo model inside the container
make docker-train

# 3. Test the API
curl http://localhost:8000/health
curl http://localhost:8000/models

# 4. Stop everything
make down
```

## Quickstart (No Docker)

Requires Python 3.10+. Uses SQLite (no Postgres/Redis needed for basic workflow).

```bash
# 1. Install dependencies
make setup

# 2. Train and register the MNIST model
make train

# 3. Start the API server
make serve

# In another terminal:
# 4. Run inference
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name": "mnist_cnn", "image_b64": "<base64-encoded-png>"}'

# 5. Run tests
make test
```

---

## API Examples

### Health Check

```bash
curl http://localhost:8000/health
# {"status":"ok","version":"0.1.0"}
```

### List Models

```bash
curl http://localhost:8000/models
# [{"model_name":"mnist_cnn","model_version":"v1.0.0","status":"prod","created_at":"..."}]
```

### Single Prediction

```bash
# Generate a test image as base64 (Python one-liner):
IMAGE_B64=$(python -c "
import base64, io, numpy as np
from PIL import Image
img = Image.fromarray(np.random.randint(0,255,(28,28),dtype=np.uint8))
buf = io.BytesIO(); img.save(buf, format='PNG')
print(base64.b64encode(buf.getvalue()).decode())
")

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"model_name\": \"mnist_cnn\", \"image_b64\": \"$IMAGE_B64\"}"
# {"prediction":3,"latency_ms":1.234,"model_version":"v1.0.0"}
```

### Submit Batch Job

```bash
curl -X POST http://localhost:8000/batch/submit \
  -H "Content-Type: application/json" \
  -d '{"model_name":"mnist_cnn","dataset_id":"mnist_1000","config":{"batch_size":64}}'
# {"job_id":"<uuid>","status":"queued"}
```

### Check Batch Status

```bash
curl http://localhost:8000/batch/status/<job_id>
# {"job_id":"...","status":"succeeded","result_metrics":{"accuracy":0.98,...}}
```

### Prometheus Metrics

```bash
curl http://localhost:8000/metrics
# HELP request_latency_seconds Latency of HTTP requests in seconds
# TYPE request_latency_seconds histogram
# ...
```

---

## Model Registry Workflow

The registry manages model versions with a staging/prod lifecycle:

```bash
# Register a new model version
python -m platform_cli register \
  --model_name mnist_cnn \
  --model_version v1.0.0 \
  --artifact_path ./artifacts/mnist_cnn/v1.0.0/model.pt

# Promote to production
python -m platform_cli promote \
  --model_name mnist_cnn \
  --model_version v1.0.0

# List all versions
python -m platform_cli list --model_name mnist_cnn

# Rollback to previous prod version
python -m platform_cli rollback --model_name mnist_cnn
```

**Lifecycle:** When a model is registered, it starts in `staging`. Promoting sets it to `prod` and demotes any existing prod version. Rollback reverts to the most recently created staging version.

**Storage:** Artifacts are stored at `./artifacts/{model_name}/{model_version}/model.pt`. Metadata (version, git SHA, tags, status, metrics) is stored in the database.

---

## Batch Inference Workflow

1. Submit a job via `POST /batch/submit` specifying model and dataset.
2. The job is enqueued in Redis Queue (or executed synchronously if Redis is unavailable).
3. The RQ worker loads the model, runs inference over the dataset, and computes:
   - **Accuracy**
   - **Latency percentiles** (p50, p95, p99)
   - **Throughput** (queries per second)
4. Results are stored in the database and retrievable via `GET /batch/status/{job_id}`.

The `dataset_id` field follows the pattern `mnist_<N>` where `<N>` is the number of test samples (e.g., `mnist_1000`).

---

## Regression Gates

Gates compare a candidate model version against a baseline to prevent regressions:

```bash
python -m platform_cli gate \
  --model_name mnist_cnn \
  --candidate_version v2.0.0 \
  --baseline_version v1.0.0
```

**Rules:**
- Accuracy must not drop more than **1%**
- p95 latency must not worsen more than **10%**

**Prerequisites:** Both versions must have evaluation metrics stored (run batch inference first).

The gate outputs a PASS/FAIL summary with per-check details and stores the result in the database.

---

## Load Testing

The load test harness sends concurrent requests to the `/predict` endpoint:

```bash
python scripts/loadtest.py \
  --url http://localhost:8000 \
  --concurrency 10 \
  --total 200 \
  --model_name mnist_cnn
```

**Output:**
- QPS (queries per second)
- p50, p95, p99 latency
- Error count
- JSON report saved to `./reports/loadtest_<timestamp>.json`

Or use the Makefile shortcut:

```bash
make run-loadtest
```

---

## Observability

### Structured Logging

All log output is JSON-formatted via `structlog`:

```json
{"event": "model_registered", "model_name": "mnist_cnn", "model_version": "v1.0.0", "timestamp": "2025-01-15T10:30:00Z", "level": "info"}
```

### Prometheus Metrics

Exposed at `GET /metrics`:

| Metric | Type | Labels |
|---|---|---|
| `request_latency_seconds` | Histogram | endpoint, model_name |
| `request_count` | Counter | endpoint, model_name, status |
| `batch_job_duration_seconds` | Histogram | model_name |
| `queue_depth` | Gauge | — |

### Grafana Dashboard

Import `ops/grafana-dashboard.json` into Grafana. Panels include:
- Request rate over time
- Latency percentiles (p50/p95/p99)
- Batch job duration
- Queue depth

---

## Adding a New Model Type

1. **Define the model** in `app/inference/model.py` (or a new file):
   ```python
   class MyModel(nn.Module):
       def __init__(self):
           super().__init__()
           # ...
       def forward(self, x):
           # ...
   ```

2. **Create a training script** in `scripts/` that trains and saves the model.

3. **Register it**:
   ```bash
   python -m platform_cli register \
     --model_name my_model \
     --model_version v1.0.0 \
     --artifact_path /path/to/model.pt
   ```

4. **Update `load_model()`** in `app/inference/model.py` to dispatch based on model name if you have multiple architectures.

5. **Update prediction logic** in `app/inference/predict.py` if the input format differs from MNIST.

---

## Makefile Reference

| Command | Description |
|---|---|
| `make setup` | Install Python dependencies |
| `make train` | Train MNIST model and register as v1.0.0 |
| `make serve` | Start API server (local, no Docker) |
| `make test` | Run unit tests |
| `make register-demo` | Train + register demo model |
| `make run-batch` | Submit a batch job via the API |
| `make run-loadtest` | Run load test against local API |
| `make gate-demo` | Run regression gate (v1.0.0 vs itself) |
| `make up` | Start Docker Compose stack |
| `make down` | Stop and remove Docker Compose stack |
| `make clean` | Remove artifacts, reports, caches |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'app'`

Make sure you're running commands from the repository root. The `pyproject.toml` sets `pythonpath = ["."]` for pytest.

### Redis connection refused (local dev)

Batch jobs fall back to synchronous execution when Redis is unavailable. For async processing, start Redis:

```bash
# macOS
brew install redis && redis-server

# Or use Docker
docker run -d -p 6379:6379 redis:7-alpine
```

### `torch.load` warnings

If you see `FutureWarning` about `weights_only`, the codebase already passes `weights_only=True` to `torch.load`. Ensure you're on PyTorch >= 2.1.

### Database issues

For local dev, the platform uses SQLite by default (`./platform.db`). To reset:

```bash
rm platform.db
```

For Docker, the Postgres data persists in a Docker volume. To reset:

```bash
make down  # uses -v to remove volumes
make up
```

### Tests fail with segfault (exit code 139)

This can happen in restricted sandbox environments. Run tests directly:

```bash
python -m pytest tests/ -v
```

---

## Tech Stack

- **Python 3.11+** / FastAPI / Uvicorn
- **PyTorch** / TorchVision
- **SQLAlchemy** (Postgres + SQLite)
- **Redis** / RQ (async job queue)
- **Prometheus** client + Grafana
- **Docker** + Docker Compose
- **structlog** (JSON logging)
- **Click** (CLI)
- **httpx** (load testing)
