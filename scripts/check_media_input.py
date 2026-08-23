from __future__ import annotations

import argparse
import json
from pathlib import Path

from viavsr.preprocessing import MediaInputError, probe_av_media


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a webcam recording before Vietnamese AVSR preprocessing."
    )
    parser.add_argument("--media", required=True, type=Path)
    parser.add_argument("--max-duration", default=15.0, type=float)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _write_optional_report(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = build_parser().parse_args()
    try:
        metadata = probe_av_media(args.media)
        duration_compatible = metadata.duration_seconds <= args.max_duration
        if metadata.has_mouth_roi_resolution and duration_compatible:
            next_stage = "ready_for_inference"
        elif not duration_compatible:
            next_stage = "split_or_record_a_shorter_clip"
        else:
            next_stage = "face_alignment_and_mouth_roi_extraction_required"
        payload = {
            "status": "passed",
            "contains_audio_and_video": True,
            "duration_compatible": duration_compatible,
            "metadata": metadata.to_dict(),
            "next_stage": next_stage,
        }
        exit_code = 0 if duration_compatible else 1
    except MediaInputError as exc:
        payload = {
            "status": "failed",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        exit_code = 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    _write_optional_report(args.output, payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
