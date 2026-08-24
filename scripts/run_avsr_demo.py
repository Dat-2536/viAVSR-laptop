#!/usr/bin/env python3
"""Run raw webcam media through the complete Vietnamese AVSR pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from viavsr.demo import run_end_to_end_demo
from viavsr.inference import DEFAULT_BEAM_SIZE, DEFAULT_CTC_WEIGHT
from viavsr.preprocessing.face_tracking import DEFAULT_DETECTION_MAX_SIZE
from viavsr.preprocessing.media import TARGET_FRAME_RATE

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "demo"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run face tracking, mouth-ROI extraction, Vietnamese AVSR inference, "
            "and optional WER/CER evaluation with one command."
        )
    )
    parser.add_argument(
        "--media", required=True, type=Path, help="Raw video with audio."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Runtime configuration (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Generated artifact root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--tracking-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for face detection and landmark tracking.",
    )
    parser.add_argument(
        "--decoder",
        choices=("ctc_greedy", "joint_beam_search"),
        default="joint_beam_search",
        help="Transcript decoder (default: joint_beam_search).",
    )
    parser.add_argument("--beam-size", type=int, default=DEFAULT_BEAM_SIZE)
    parser.add_argument("--ctc-weight", type=float, default=DEFAULT_CTC_WEIGHT)
    parser.add_argument(
        "--reference-text",
        help="Optional ground truth; enables WER/CER in the final report.",
    )
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--frame-rate", type=int, default=TARGET_FRAME_RATE)
    parser.add_argument(
        "--max-detection-size",
        type=int,
        default=DEFAULT_DETECTION_MAX_SIZE,
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help="Override the configured face-detection confidence threshold.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_end_to_end_demo(
        config_path=args.config,
        media_path=args.media,
        output_root=args.output_root,
        tracking_device=args.tracking_device,
        decoder=args.decoder,
        beam_size=args.beam_size,
        ctc_weight=args.ctc_weight,
        reference_text=args.reference_text,
        max_duration_seconds=args.max_duration,
        frame_rate=args.frame_rate,
        max_detection_size=args.max_detection_size,
        confidence_threshold=args.confidence_threshold,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
