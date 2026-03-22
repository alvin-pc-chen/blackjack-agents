"""CLI entry point for running blackjack experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from .experiment import ExperimentRunner, load_config


@click.group()
def main() -> None:
    """Blackjack experiment runner."""
    pass


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--output-dir", default=None, help="Override output directory")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def run(config_path: str, output_dir: str | None, verbose: bool) -> None:
    """Run an experiment from a YAML config file."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config = load_config(config_path)
    if output_dir:
        config.output_dir = output_dir

    runner = ExperimentRunner(config)
    record = runner.run()

    click.echo(f"\nExperiment: {config.experiment_name}")
    click.echo(f"Rounds played: {record.summary.total_rounds}")
    click.echo(f"ID: {record.experiment_id}\n")

    for ps in record.summary.player_summaries:
        click.echo(
            f"  {ps.player_name:15s} ({ps.agent_type:20s}): "
            f"W={ps.wins:3d} L={ps.losses:3d} P={ps.pushes:3d} "
            f"BJ={ps.blackjacks:2d} "
            f"win%={ps.win_rate:.1%}  net={ps.net_units:+.1f}u"
        )


@main.command()
@click.argument("results_path", type=click.Path(exists=True))
def summarize(results_path: str) -> None:
    """Print summary statistics from a results JSON file."""
    data = json.loads(Path(results_path).read_text())
    summary = data.get("summary", {})

    click.echo(f"Experiment: {data.get('config', {}).get('experiment_name', '?')}")
    click.echo(f"Rounds: {summary.get('total_rounds', '?')}")
    click.echo(f"Timestamp: {data.get('timestamp', '?')}\n")

    for ps in summary.get("player_summaries", []):
        click.echo(
            f"  {ps['player_name']:15s} ({ps['agent_type']:20s}): "
            f"W={ps['wins']:3d} L={ps['losses']:3d} P={ps['pushes']:3d} "
            f"BJ={ps['blackjacks']:2d} "
            f"win%={ps['win_rate']:.1%}  net={ps['net_units']:+.1f}u"
        )
