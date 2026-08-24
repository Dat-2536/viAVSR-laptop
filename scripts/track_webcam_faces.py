from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from viavsr.preprocessing import (
    FANFaceLandmarker,
    MediaInputError,
    load_face_tracking_quality_policy,
    save_face_tracking_artifacts,
    track_face_landmarks,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs/preprocessing/face_tracks"
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/config.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect and track one face as 68-point FAN landmarks across one or "
            "more webcam recordings."
        )
    )
    parser.add_argument("--media", required=True, nargs="+", type=Path)
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--frame-rate", default=25, type=int)
    parser.add_argument("--max-detection-size", default=640, type=int)
    parser.add_argument(
        "--confidence-threshold",
        default=None,
        type=float,
        help="Override face_tracking.min_detection_confidence from --config.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        policy = load_face_tracking_quality_policy(args.config)
        if args.confidence_threshold is not None:
            policy = replace(
                policy,
                min_detection_confidence=args.confidence_threshold,
            )
    except (MediaInputError, ValueError) as exc:
        payload = {
            "status": "failed",
            "stage": "configuration",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    try:
        landmarker = FANFaceLandmarker(
            device=args.device,
            confidence_threshold=policy.min_detection_confidence,
        )
    except MediaInputError as exc:
        payload = {
            "status": "failed",
            "stage": "backend_initialization",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    results: list[dict[str, object]] = []
    failed = False
    for media_path in args.media:
        print(f"Tracking face landmarks: {media_path}", file=sys.stderr)
        started = time.perf_counter()
        stem = media_path.stem
        artifact_path = output_dir / f"{stem}_face_track.npz"
        report_path = output_dir / f"{stem}_face_track.json"
        try:
            artifact_path.unlink(missing_ok=True)
            sequence = track_face_landmarks(
                media_path,
                landmarker=landmarker,
                frame_rate=args.frame_rate,
                max_detection_size=args.max_detection_size,
                policy=policy,
            )
            result = save_face_tracking_artifacts(
                sequence,
                artifact_path=artifact_path,
                report_path=report_path,
            )
            result["tracking_seconds"] = time.perf_counter() - started
            report_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            results.append(result)
            if not sequence.quality_passed:
                failed = True
        except (MediaInputError, OSError) as exc:
            failed = True
            result = {
                "status": "failed",
                "quality_status": "failed",
                "quality_issues": ["tracking_failed"],
                "quality_thresholds": asdict(policy),
                "media_path": str(media_path.expanduser().resolve()),
                "artifact_path": str(artifact_path),
                "report_path": str(report_path),
                "tracking_seconds": time.perf_counter() - started,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
            report_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            results.append(result)

    payload = {
        "status": "failed" if failed else "passed",
        "processed": len(results),
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
