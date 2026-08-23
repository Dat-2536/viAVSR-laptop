from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from viavsr.preprocessing import (
    FANFaceLandmarker,
    MediaInputError,
    save_face_tracking_artifacts,
    track_face_landmarks,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs/preprocessing/face_tracks"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect and track one face as 68-point FAN landmarks across one or "
            "more webcam recordings."
        )
    )
    parser.add_argument("--media", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--frame-rate", default=25, type=int)
    parser.add_argument("--max-detection-size", default=640, type=int)
    parser.add_argument("--confidence-threshold", default=0.8, type=float)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        landmarker = FANFaceLandmarker(
            device=args.device,
            confidence_threshold=args.confidence_threshold,
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
        try:
            sequence = track_face_landmarks(
                media_path,
                landmarker=landmarker,
                frame_rate=args.frame_rate,
                max_detection_size=args.max_detection_size,
            )
            stem = media_path.stem
            artifact_path = output_dir / f"{stem}_face_track.npz"
            report_path = output_dir / f"{stem}_face_track.json"
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
        except MediaInputError as exc:
            failed = True
            results.append(
                {
                    "status": "failed",
                    "media_path": str(media_path.expanduser().resolve()),
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )

    payload = {
        "status": "failed" if failed else "passed",
        "processed": len(results),
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
