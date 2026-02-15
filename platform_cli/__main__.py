"""CLI entrypoint: python -m platform_cli <command>."""

from __future__ import annotations

import json
import sys

import click

from app.db.session import init_db


@click.group()
def cli() -> None:
    """ML Inference & Evaluation Platform CLI."""
    init_db()


# -------------------------------------------------------------------
# Model Registry
# -------------------------------------------------------------------

@cli.command()
@click.option("--model_name", required=True, help="Name of the model")
@click.option("--model_version", required=True, help="Semantic version string")
@click.option("--artifact_path", required=True, help="Path to the .pt file")
@click.option("--git_sha", default=None, help="Git commit SHA")
@click.option("--tags", default=None, help="JSON string of tags")
def register(
    model_name: str,
    model_version: str,
    artifact_path: str,
    git_sha: str | None,
    tags: str | None,
) -> None:
    """Register a model version in the registry."""
    from app.registry.manager import register as reg

    parsed_tags = json.loads(tags) if tags else None
    mv = reg(
        model_name=model_name,
        model_version=model_version,
        artifact_path=artifact_path,
        git_sha=git_sha,
        tags=parsed_tags,
    )
    click.echo(f"Registered {mv.model_name}@{mv.model_version} -> {mv.artifact_path}")


@cli.command()
@click.option("--model_name", required=True)
@click.option("--model_version", required=True)
def promote(model_name: str, model_version: str) -> None:
    """Promote a model version to production."""
    from app.registry.manager import promote as prom

    mv = prom(model_name=model_name, model_version=model_version)
    if mv is None:
        click.echo("ERROR: model version not found", err=True)
        sys.exit(1)
    click.echo(f"Promoted {mv.model_name}@{mv.model_version} to prod")


@cli.command()
@click.option("--model_name", required=True)
def rollback(model_name: str) -> None:
    """Rollback to the previous production version."""
    from app.registry.manager import rollback as rb

    mv = rb(model_name=model_name)
    if mv is None:
        click.echo("ERROR: no version to rollback to", err=True)
        sys.exit(1)
    click.echo(f"Rolled back {mv.model_name} -> now prod: {mv.model_version}")


@cli.command("list")
@click.option("--model_name", default=None, help="Filter by model name")
def list_cmd(model_name: str | None) -> None:
    """List registered models."""
    from app.registry.manager import list_models

    models = list_models(model_name=model_name)
    if not models:
        click.echo("No models registered.")
        return
    for mv in models:
        tags_str = mv.tags or "{}"
        click.echo(
            f"  {mv.model_name}@{mv.model_version}  "
            f"status={mv.status}  created={mv.created_at}  tags={tags_str}"
        )


# -------------------------------------------------------------------
# Regression Gate
# -------------------------------------------------------------------

@cli.command()
@click.option("--model_name", required=True)
@click.option("--candidate_version", required=True)
@click.option("--baseline_version", required=True)
def gate(model_name: str, candidate_version: str, baseline_version: str) -> None:
    """Run regression gate between candidate and baseline."""
    from app.eval.gates import run_regression_gate

    result = run_regression_gate(
        model_name=model_name,
        candidate_version=candidate_version,
        baseline_version=baseline_version,
    )
    status = "PASS" if result.passed else "FAIL"
    details = json.loads(result.details) if result.details else {}
    click.echo(f"Gate result: {status}")
    click.echo(json.dumps(details, indent=2))
    if not result.passed:
        sys.exit(1)


if __name__ == "__main__":
    cli()
