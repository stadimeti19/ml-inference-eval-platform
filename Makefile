.PHONY: setup up down test register-demo run-batch run-loadtest gate-demo train clean

export PYTHONPATH := $(shell pwd)

# ---------------------------------------------------------------------------
# Local development (no Docker)
# ---------------------------------------------------------------------------

setup:
	pip install -r requirements.txt

train:
	python scripts/train_mnist.py

register-demo: train
	@echo "Model trained and registered as mnist_cnn@v1.0.0 (prod)"

run-batch:
	@echo "Submitting batch job via API..."
	curl -s -X POST http://localhost:8000/batch/submit \
		-H "Content-Type: application/json" \
		-d '{"model_name":"mnist_cnn","dataset_id":"mnist_1000"}' | python -m json.tool

run-loadtest:
	python scripts/loadtest.py --url http://localhost:8000 --concurrency 10 --total 100

gate-demo:
	python -m platform_cli gate \
		--model_name mnist_cnn \
		--candidate_version v1.0.0 \
		--baseline_version v1.0.0

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
	rm -rf artifacts/ reports/ data/ __pycache__ .pytest_cache *.db
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
