from __future__ import annotations

import argparse
import json
from pathlib import Path

from viasvr.evaluation import evaluate_transcript


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser.

    `argparse` is Python's standard-library tool for command-line interfaces.
    It parses flags such as `--reference-text` into Python values.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate Vietnamese AVSR predictions with WER/CER."
    )
    parser.add_argument("--reference-text", type=str)
    parser.add_argument("--prediction-text", type=str)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.reference_text is None or args.prediction_text is None:
        raise SystemExit(
            "Provide both --reference-text and --prediction-text."
        )

    result = evaluate_transcript(
        reference=args.reference_text,
        prediction=args.prediction_text,
    )

    payload = result.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
