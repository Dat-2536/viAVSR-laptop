from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from viavsr.preprocessing import MediaInputError, export_aligned_mouth_roi_video

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKS_DIR = REPOSITORY_ROOT / "outputs/preprocessing/face_tracks"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs/preprocessing/mouth_roi"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Affine-align tracked webcam faces, crop 96x96 grayscale mouth ROIs, "
            "and preserve synchronized source audio."
        )
    )
    parser.add_argument("--media", required=True, nargs="+", type=Path)
    parser.add_argument("--tracks-dir", default=DEFAULT_TRACKS_DIR, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tracks_dir = args.tracks_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    failed = False
    for media_path in args.media:
        stem = media_path.stem
        track_path = tracks_dir / f"{stem}_face_track.npz"
        output_path = output_dir / f"{stem}_mouth96.mp4"
        report_path = output_dir / f"{stem}_mouth96.json"
        print(f"Preparing aligned mouth ROI: {media_path}", file=sys.stderr)
        started = time.perf_counter()
        try:
            export = export_aligned_mouth_roi_video(
                media_path,
                track_path,
                output_path,
            )
            result = {
                "status": "passed",
                **export.to_dict(),
                "preprocessing_seconds": time.perf_counter() - started,
                "report_path": str(report_path),
            }
            report_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            results.append(result)
        except MediaInputError as exc:
            failed = True
            result = {
                "status": "failed",
                "media_path": str(media_path.expanduser().resolve()),
                "track_path": str(track_path),
                "output_path": str(output_path),
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
