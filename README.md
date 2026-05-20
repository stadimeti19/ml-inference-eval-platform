# ML Inference & Evaluation Platform

This platform is a production-style MLOps system for serving, evaluating, and safely rolling out machine learning models. It provides online inference, batch evaluation, model registry management, regression gates, SLO checks, shadow testing, benchmarking, observability, and deployment audit trails.

The goal is to demonstrate how real ML teams prevent unsafe model releases. Instead of promoting a model based only on accuracy, the platform compares accuracy, latency, throughput, cache behavior, error rate, and shadow disagreement before allowing a candidate model into production. This mirrors production ML infrastructure where model quality, reliability, cost, and performance all matter.