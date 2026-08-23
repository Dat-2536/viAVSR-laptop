from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from viavsr.evaluation import evaluate_transcript
from viavsr.inference import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_CTC_WEIGHT,
    load_model_assets_config,
    load_vietnamese_avsr_assets,
    recognize_prepared_av,
)
from viavsr.inference.errors import ModelAssetsError
from viavsr.inference.reporting import redact_secrets, write_json_report
from viavsr.preprocessing import MediaInputError, prepare_mouth_roi_media

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs/inference/media_inference.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Vietnamese AVSR inference on a prepared 96x96 mouth-ROI video "
            "with embedded audio."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--media", required=True, type=Path)
    parser.add_argument("--max-duration", default=15.0, type=float)
    parser.add_argument(
        "--decoder",
        choices=("ctc_greedy", "joint_beam_search"),
        default="ctc_greedy",
    )
    parser.add_argument("--beam-size", default=DEFAULT_BEAM_SIZE, type=int)
    parser.add_argument("--ctc-weight", default=DEFAULT_CTC_WEIGHT, type=float)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    stage = "preprocessing"
    try:
        started = time.perf_counter()
        prepared = prepare_mouth_roi_media(
            args.media, max_duration_seconds=args.max_duration
        )
        preprocessing_seconds = time.perf_counter() - started

        stage = "asset_loading"
        config = load_model_assets_config(args.config)
        assets = load_vietnamese_avsr_assets(config)

        stage = "inference"
        result = recognize_prepared_av(
            assets,
            prepared,
            decoder=args.decoder,
            beam_size=args.beam_size,
            ctc_weight=args.ctc_weight,
        )
        payload = {
            "status": "passed",
            "media": prepared.metadata.to_dict(),
            "input_shapes": prepared.shape_report(),
            "preprocessing_seconds": preprocessing_seconds,
            "result": result.to_dict(),
        }
        if args.reference_text is not None:
            payload["evaluation"] = evaluate_transcript(
                reference=args.reference_text,
                prediction=result.transcript,
            ).to_dict()
        write_json_report(args.output.expanduser().resolve(), payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except (MediaInputError, ModelAssetsError) as exc:
        if isinstance(exc, ModelAssetsError):
            stage = exc.stage
        payload = {
            "status": "failed",
            "stage": stage,
            "error": {
                "type": type(exc).__name__,
                "message": redact_secrets(str(exc)),
            },
        }
        write_json_report(args.output.expanduser().resolve(), payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
