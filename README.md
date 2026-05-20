# ML Inference & Evaluation Platform

Production-style MLOps platform for model serving, evaluation, rollout safety, and observability. The demo model is an MNIST PyTorch classifier, while the platform pieces mirror common ML platform engineering workflows: registry, online inference, batch evaluation, gates, shadow traffic, benchmarking, metrics, and deployment audit history.

## What It Includes

- FastAPI online inference API
- PyTorch model training and loading
- Versioned model registry with `staging` and `prod` lifecycle
- Batch evaluation through Redis Queue, with local synchronous fallback
- Regression gates and SLO gates
- Shadow/canary comparison APIs
- Reproducible inference benchmarking
- Model cache instrumentation
- Deployment event audit trail
- JSON and Prometheus observability endpoints
- Server-rendered dashboard
- Docker Compose stack for API, worker, Postgres, and Redis
- CLI for registry, promotion, rollback, and gates

## Project Layout

```text
app/
  api/          FastAPI routes
  core/         config, logging, metrics
  db/           SQLAlchemy models and repositories
  datasets/     MNIST loading utilities
  eval/         metrics, regression gates, SLO gates
  inference/    model definitions, prediction, cache
  jobs/         Redis Queue worker and batch tasks
  registry/     model lifecycle operations
  templates/    dashboard HTML

platform_cli/   CLI entrypoint
scripts/        training, benchmarking, demos
ops/            Dockerfile, compose stack, Grafana dashboard
tests/          unit and API tests
migrations/     Alembic migrations
```
