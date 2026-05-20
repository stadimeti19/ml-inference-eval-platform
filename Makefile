.PHONY: setup up down test register-demo run-batch run-loadtest gate-demo train train-v2 pipeline migrate benchmark shadow rollback-demo bad-model-demo clean

export PYTHONPATH := $(shell pwd)

# ---------------------------------------------------------------------------
# Local development (no Docker)
# ---------------------------------------------------------------------------

setup:
	pip install -r requirements.txt

migrate:
	alembic upgrade head

train:
	python scripts/train_mnist.py --model_version v1.0.0 --architecture default --promote

train-v2:
	python scripts/train_mnist.py --model_version v2.0.0 --architecture large

register-demo: train
	@echo "Model trained and registered as mnist_cnn@v1.0.0 (prod)"

pipeline:
	@echo "=== Running full CI/CD pipeline: train v1 + v2, eval, gate, promote ==="
	python scripts/run_pipeline.py

run-batch:
	@echo "Submitting batch job via API..."
	curl -s -X POST http://localhost:8000/batch/submit \
		-H "Content-Type: application/json" \
		-d '{"model_name":"mnist_cnn","dataset_id":"mnist_1000"}' | python -m json.tool

run-loadtest:
	python scripts/loadtest.py --url http://localhost:8000 --concurrency 10 --total 100

benchmark:
	python scripts/benchmark_inference.py --url http://localhost:8000 --requests 1000 --concurrency 50 --model_version v1.0.0 --candidate_version v2.0.0

shadow:
	python scripts/benchmark_inference.py --url http://localhost:8000 --requests 100 --concurrency 10 --shadow_version v2.0.0 --payload_source mnist_test --payload_mode random_per_request --output shadow_benchmark_results.json

gate-demo:
	python -m platform_cli gate \
		--model_name mnist_cnn \
		--candidate_version v1.0.0 \
		--baseline_version v1.0.0

rollback-demo:
	python -m platform_cli rollback --model_name mnist_cnn

bad-model-demo:
	python scripts/simulate_bad_model.py --model_name mnist_cnn

test:
	python -m pytest tests/ -v

serve:
	PYTHONPATH=$(shell pwd) uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

up:
	docker compose -f ops/docker-compose.yml up --build -d

down:
	docker compose -f ops/docker-compose.yml down -v

docker-train:
	docker compose -f ops/docker-compose.yml exec api python scripts/train_mnist.py

docker-batch:
	docker compose -f ops/docker-compose.yml exec api \
		curl -s -X POST http://localhost:8000/batch/submit \
			-H "Content-Type: application/json" \
			-d '{"model_name":"mnist_cnn","dataset_id":"mnist_1000"}' | python -m json.tool

clean:
	rm -rf artifacts/ reports/ data/ __pycache__ .pytest_cache *.db platform.db
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
