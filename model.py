"""Entry point for baseline modeling experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modeling import BaselineExperiment, ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline modeling experiments")
    parser.add_argument("config", type=Path, help="Path to experiment configuration YAML")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional override for the tracking output directory",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = ExperimentConfig.from_file(args.config)
    if args.output is not None:
        config.tracking.output_directory = args.output
    experiment = BaselineExperiment(config)
    result = experiment.run()
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
