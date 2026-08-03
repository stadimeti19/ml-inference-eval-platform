# ML Inference & Evaluation Platform

A production-style MLOps reference project for serving, evaluating, and safely
releasing machine-learning models. It demonstrates the control plane around a
model: version registration, offline evaluation, regression gates, shadow
traffic, SLO enforcement, promotion, rollback, and operational visibility.

The included MNIST classifiers keep the machine-learning workload small enough
to run locally. The platform workflow—not MNIST—is the focus.

## What it demonstrates

- Online inference with model-version selection and an in-process model cache
- Asynchronous batch evaluation with Redis and RQ workers
- A model registry backed by SQLAlchemy and versioned Alembic migrations
- Candidate-versus-baseline gates for accuracy and p95 latency regressions
- Shadow inference that records agreement and latency without changing responses
- Configurable SLO policies, deployment events, promotion, and rollback
- Prometheus metrics and an included Grafana dashboard definition
- Offline-by-default comparisons across OpenAI, Gemini, Anthropic, and local LLMs
- A server-rendered operational dashboard and interactive OpenAPI documentation

## Release workflow

```text
train -> register -> evaluate -> gate -> promote
                       |          |
                       |          +-> reject if quality or latency regresses
                       +-> shadow candidate against production traffic
```

The demo gate allows at most a 1% accuracy drop and a 10% p95-latency increase.
Every gate and deployment decision is recorded for later inspection.

## Architecture

```text
                         +-------------------+
 clients / dashboard -->| FastAPI API       |--> model cache --> PyTorch model
                         | control + serving |
                         +---------+---------+
                                   |
                    +--------------+--------------+
                    |                             |
             PostgreSQL / SQLite             Redis queue
          registry, gates, audit                 |
                                           RQ worker
                                                |
                                         batch evaluation

 Prometheus scrapes /metrics; Grafana visualizes the exported metrics.
```

## Quick start

Requirements: Python 3.11+ and, for background jobs, Redis.

```bash
python -m venv .venv
source .venv/bin/activate
make setup-dev
make migrate
make train
make serve
```

Then open:

- Dashboard: <http://localhost:8000/dashboard>
- OpenAPI explorer: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/health>
- Prometheus metrics: <http://localhost:8000/metrics>

Run the complete train/evaluate/gate/promote demonstration with:

```bash
make pipeline
```

The default local database is `platform.db`. Model files are placed in
`artifacts/`; both are ignored by Git.

## Docker environment

The Docker environment starts PostgreSQL, Redis, the API, and an RQ worker:

```bash
make up
make docker-train
```

Stop it with `make down`. This command also removes the Compose volumes and their
database, dataset, and model-artifact data.

## Common workflows

```bash
make pipeline       # train two versions, evaluate, gate, and promote the winner
make benchmark      # load and latency benchmark against the running API
make shadow         # send traffic to prod while evaluating a shadow candidate
make llm-compare    # compare provider adapters (mock mode by default)
make rollback-demo  # restore the previous eligible model version
make bad-model-demo # show a candidate being rejected
```

See [docs/API_EXAMPLES.md](docs/API_EXAMPLES.md) for end-to-end API requests and
response shapes.

## LLM provider mode

LLM calls are deterministic mocks by default, making tests and demos safe and
reproducible without credentials. Live calls are opt-in:

```bash
export LLM_ENABLE_LIVE_PROVIDERS=true
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

If a live provider call fails, the adapter returns its mock result together with
the provider error. Model identifiers and published prices in this demo are
configuration examples and should be reviewed before real cost decisions.

## Development quality checks

```bash
make lint       # Ruff rules
make typecheck  # mypy across application and scripts
make test-cov   # test suite with branch coverage threshold
make quality    # all three checks
```

Pull requests and pushes to `main` run the same checks in GitHub Actions. The CI
workflow uploads `coverage.xml` for inspection.

## Repository map

```text
app/api/          FastAPI routes and application setup
app/eval/         metrics, regression gates, and SLO evaluation
app/inference/    PyTorch loading, caching, and prediction
app/registry/     model registration, promotion, and rollback
app/jobs/         Redis/RQ background processing
app/llm/          provider adapters and comparison scoring
app/templates/    dashboard HTML
app/static/       dashboard CSS and JavaScript
migrations/       database schema history
ops/              Docker and Grafana configuration
scripts/          training, pipeline, benchmark, and failure demos
tests/            API, registry, inference, gate, SLO, and dashboard tests
```

## Scope and limitations

This is a portfolio/reference implementation, not a hosted production service.
It currently has no authentication, authorization, rate limiting, distributed
model cache, cloud artifact store, or automated public deployment. Shadow work
runs within the request path, and the demo's fixed gate thresholds are not a
replacement for domain-specific release policy. Those boundaries are deliberate
and documented so the project does not overstate its production readiness.

## License

No license has been selected yet. Treat the repository as all rights reserved
until a license file is added.
