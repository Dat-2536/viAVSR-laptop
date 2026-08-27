"""One-command orchestration for the raw-media AVSR demo."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from viavsr.evaluation import evaluate_transcript
from viavsr.inference import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_CTC_WEIGHT,
    load_model_assets_config,
    load_vietnamese_avsr_assets,
    recognize_prepared_av,
)
from viavsr.inference.errors import ModelAssetsError
from viavsr.inference.recognition import DecoderName
from viavsr.inference.reporting import redact_secrets, write_json_report
from viavsr.preprocessing import (
    FANFaceLandmarker,
    MediaInputError,
    export_aligned_mouth_roi_video,
    export_mouth_roi_display_video,
    load_face_tracking_quality_policy,
    prepare_audio_only_media,
    prepare_mouth_roi_media,
    probe_av_media,
    save_face_tracking_artifacts,
    track_face_landmarks,
)
from viavsr.preprocessing.face_tracking import DEFAULT_DETECTION_MAX_SIZE, DeviceRequest
from viavsr.preprocessing.media import TARGET_FRAME_RATE

DEMO_REPORT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class DemoArtifactPaths:
    """Deterministic artifact paths for one raw-media demo run."""

    run_directory: Path
    face_track: Path
    face_tracking_report: Path
    mouth_roi: Path
    mouth_roi_report: Path
    mouth_roi_display: Path
    mouth_roi_display_report: Path
    report: Path

    @classmethod
    def for_media(cls, media_path: Path, output_root: Path) -> DemoArtifactPaths:
        run_directory = output_root.expanduser().resolve() / media_path.stem
        return cls(
            run_directory=run_directory,
            face_track=run_directory / "face_track.npz",
            face_tracking_report=run_directory / "face_tracking.json",
            mouth_roi=run_directory / "mouth96.mp4",
            mouth_roi_report=run_directory / "mouth_roi.json",
            mouth_roi_display=run_directory / "mouth96_display.mp4",
            mouth_roi_display_report=run_directory / "mouth_roi_display.json",
            report=run_directory / "report.json",
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "run_directory": str(self.run_directory),
            "face_track": str(self.face_track),
            "face_tracking_report": str(self.face_tracking_report),
            "mouth_roi": str(self.mouth_roi),
            "mouth_roi_report": str(self.mouth_roi_report),
            "mouth_roi_display": str(self.mouth_roi_display),
            "mouth_roi_display_report": str(self.mouth_roi_display_report),
            "report": str(self.report),
        }

    def invalidate(self) -> None:
        """Remove only stale generated files for this exact media stem."""

        self.run_directory.mkdir(parents=True, exist_ok=True)
        for path in (
            self.face_track,
            self.face_tracking_report,
            self.mouth_roi,
            self.mouth_roi_report,
            self.mouth_roi_display,
            self.mouth_roi_display_report,
            self.report,
        ):
            path.unlink(missing_ok=True)


def _validate_run_options(
    *,
    decoder: DecoderName,
    beam_size: int,
    ctc_weight: float,
    max_duration_seconds: float,
    frame_rate: int,
    max_detection_size: int,
) -> None:
    if decoder not in {"ctc_greedy", "joint_beam_search"}:
        raise ValueError(f"Unsupported decoder: {decoder}")
    if beam_size <= 0:
        raise ValueError("beam_size must be greater than zero.")
    if not 0.0 <= ctc_weight <= 1.0:
        raise ValueError("ctc_weight must be between 0 and 1.")
    if max_duration_seconds <= 0:
        raise ValueError("max_duration_seconds must be greater than zero.")
    if frame_rate <= 0:
        raise ValueError("frame_rate must be greater than zero.")
    if max_detection_size <= 0:
        raise ValueError("max_detection_size must be greater than zero.")


def _finish_report(
    payload: dict[str, Any],
    paths: DemoArtifactPaths,
    started_at: float,
) -> dict[str, Any]:
    payload["timings_seconds"]["total"] = time.perf_counter() - started_at
    write_json_report(paths.report, payload)
    return payload


def _failure(
    *,
    payload: dict[str, Any],
    paths: DemoArtifactPaths,
    started_at: float,
    stage: str,
    error_type: str,
    message: str,
) -> dict[str, Any]:
    payload.update(
        {
            "status": "failed",
            "stage": stage,
            "error": {"type": error_type, "message": redact_secrets(message)},
        }
    )
    if stage in {"face_tracking_backend", "face_tracking"}:
        write_json_report(paths.face_tracking_report, payload["error"])
    elif stage == "mouth_roi":
        write_json_report(paths.mouth_roi_report, payload["error"])
    elif stage == "mouth_roi_display":
        write_json_report(paths.mouth_roi_display_report, payload["error"])
    return _finish_report(payload, paths, started_at)


def run_end_to_end_demo(
    *,
    config_path: Path,
    media_path: Path,
    output_root: Path,
    tracking_device: DeviceRequest = "auto",
    decoder: DecoderName = "joint_beam_search",
    beam_size: int = DEFAULT_BEAM_SIZE,
    ctc_weight: float = DEFAULT_CTC_WEIGHT,
    reference_text: str | None = None,
    max_duration_seconds: float = 15.0,
    frame_rate: int = TARGET_FRAME_RATE,
    max_detection_size: int = DEFAULT_DETECTION_MAX_SIZE,
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    """Run raw media through best-effort visual processing and AVSR inference.

    All generated files are written below output_root/media-stem. Failed face
    tracking falls back to audio-only inference when the media audio is usable.
    """

    started_at = time.perf_counter()
    resolved_media = media_path.expanduser().resolve()
    resolved_config = config_path.expanduser().resolve()
    paths = DemoArtifactPaths.for_media(resolved_media, output_root)
    payload: dict[str, Any] = {
        "schema_version": DEMO_REPORT_SCHEMA_VERSION,
        "status": "running",
        "stage": "initialization",
        "request": {
            "config_path": str(resolved_config),
            "media_path": str(resolved_media),
            "tracking_device": tracking_device,
            "decoder": decoder,
            "beam_size": beam_size,
            "ctc_weight": ctc_weight,
            "reference_text": reference_text,
            "max_duration_seconds": max_duration_seconds,
            "frame_rate": frame_rate,
            "max_detection_size": max_detection_size,
            "confidence_threshold": confidence_threshold,
        },
        "artifacts": paths.to_dict(),
        "timings_seconds": {},
    }
    stage = "output_initialization"

    try:
        paths.invalidate()

        stage = "configuration"
        _validate_run_options(
            decoder=decoder,
            beam_size=beam_size,
            ctc_weight=ctc_weight,
            max_duration_seconds=max_duration_seconds,
            frame_rate=frame_rate,
            max_detection_size=max_detection_size,
        )
        quality_policy = load_face_tracking_quality_policy(resolved_config)
        if confidence_threshold is not None:
            quality_policy = replace(
                quality_policy,
                min_detection_confidence=confidence_threshold,
            )

        stage = "media_preflight"
        stage_started = time.perf_counter()
        raw_metadata = probe_av_media(resolved_media)
        payload["raw_media"] = raw_metadata.to_dict()
        payload["timings_seconds"]["media_preflight"] = (
            time.perf_counter() - stage_started
        )
        if raw_metadata.duration_seconds > max_duration_seconds:
            raise MediaInputError(
                "Media duration "
                f"{raw_metadata.duration_seconds:.3f}s exceeds the configured "
                f"maximum of {max_duration_seconds:.3f}s."
            )

        stage = "face_tracking_backend"
        stage_started = time.perf_counter()
        landmarker = FANFaceLandmarker(
            device=tracking_device,
            confidence_threshold=quality_policy.min_detection_confidence,
        )
        payload["timings_seconds"]["face_tracking_backend"] = (
            time.perf_counter() - stage_started
        )

        stage = "face_tracking"
        stage_started = time.perf_counter()
        visual_fallback: dict[str, str] | None = None
        try:
            sequence = track_face_landmarks(
                resolved_media,
                landmarker=landmarker,
                frame_rate=frame_rate,
                policy=quality_policy,
                max_detection_size=max_detection_size,
            )
        except MediaInputError as exc:
            tracking_seconds = time.perf_counter() - stage_started
            visual_fallback = {
                "stage": stage,
                "type": type(exc).__name__,
                "message": redact_secrets(str(exc)),
            }
            face_report = {
                "status": "unavailable",
                "quality_status": "unavailable",
                "stage": stage,
                "error": {
                    "type": visual_fallback["type"],
                    "message": visual_fallback["message"],
                },
                "tracking_seconds": tracking_seconds,
            }
            write_json_report(paths.face_tracking_report, face_report)
            payload["face_tracking"] = face_report
            payload["timings_seconds"]["face_tracking"] = tracking_seconds
        else:
            face_report = save_face_tracking_artifacts(
                sequence,
                artifact_path=paths.face_track,
                report_path=paths.face_tracking_report,
            )
            face_report["tracking_seconds"] = time.perf_counter() - stage_started
            write_json_report(paths.face_tracking_report, face_report)
            payload["face_tracking"] = face_report
            payload["timings_seconds"]["face_tracking"] = face_report[
                "tracking_seconds"
            ]

            if not sequence.quality_passed:
                visual_fallback = {
                    "stage": "face_tracking_quality",
                    "type": "FaceTrackingQualityError",
                    "message": redact_secrets("; ".join(sequence.quality_issues)),
                }

        stage = "mouth_roi_display"
        stage_started = time.perf_counter()
        display_track_path = paths.face_track if paths.face_track.is_file() else None
        try:
            display_result = export_mouth_roi_display_video(
                resolved_media,
                paths.mouth_roi_display,
                track_path=display_track_path,
            )
        except MediaInputError as exc:
            display_report = {
                "status": "unavailable",
                "artifact_role": "ui_visualization_only",
                "used_for_inference": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": redact_secrets(str(exc)),
                },
                "processing_seconds": time.perf_counter() - stage_started,
            }
            payload.setdefault("warnings", []).append(
                "mouth_roi_display_unavailable"
            )
        else:
            display_report = display_result.to_dict()
            display_report["status"] = "passed"
            display_report["artifact_role"] = "ui_visualization_only"
            display_report["used_for_inference"] = False
            display_report["processing_seconds"] = (
                time.perf_counter() - stage_started
            )
            payload["face_tracking"]["visual_availability"] = (
                display_result.visual_availability
            )
            write_json_report(
                paths.face_tracking_report,
                payload["face_tracking"],
            )
        write_json_report(paths.mouth_roi_display_report, display_report)
        payload["mouth_roi_display"] = display_report
        payload["timings_seconds"]["mouth_roi_display"] = display_report[
            "processing_seconds"
        ]

        inference_mode = "audio_visual"
        if visual_fallback is not None:
            inference_mode = "audio_only_fallback"
            payload.setdefault("warnings", []).append(
                "visual_preprocessing_unavailable_audio_only_fallback"
            )
            payload["modality_decision"] = {
                "policy": "automatic",
                "selected_mode": inference_mode,
                "visual_input_used": False,
                "visual_gap_policy": "visual_input_not_used",
                "display_gap_policy": "no_visual_signal_placeholder",
                "fallback_reason": visual_fallback,
            }
            stage = "audio_preprocessing"
            stage_started = time.perf_counter()
            prepared = prepare_audio_only_media(
                resolved_media,
                max_duration_seconds=max_duration_seconds,
            )
            payload["timings_seconds"]["audio_preprocessing"] = (
                time.perf_counter() - stage_started
            )
        else:
            payload["modality_decision"] = {
                "policy": "automatic",
                "selected_mode": inference_mode,
                "visual_input_used": True,
                "visual_gap_policy": "interpolated_landmarks",
                "display_gap_policy": "no_visual_signal_placeholder",
            }
            stage = "mouth_roi"
            stage_started = time.perf_counter()
            mouth_result = export_aligned_mouth_roi_video(
                resolved_media,
                paths.face_track,
                paths.mouth_roi,
            )
            mouth_report = mouth_result.to_dict()
            mouth_report["processing_seconds"] = time.perf_counter() - stage_started
            write_json_report(paths.mouth_roi_report, mouth_report)
            payload["mouth_roi"] = mouth_report
            payload["timings_seconds"]["mouth_roi"] = mouth_report["processing_seconds"]

            stage = "av_preprocessing"
            stage_started = time.perf_counter()
            prepared = prepare_mouth_roi_media(
                paths.mouth_roi,
                max_duration_seconds=max_duration_seconds,
            )
            payload["timings_seconds"]["av_preprocessing"] = (
                time.perf_counter() - stage_started
            )

        payload["prepared_input"] = {
            "media": prepared.metadata.to_dict(),
            "input_shapes": prepared.shape_report(),
        }

        stage = "model_loading"
        stage_started = time.perf_counter()
        model_config = load_model_assets_config(resolved_config)
        assets = load_vietnamese_avsr_assets(model_config)
        payload["model"] = assets.report.to_dict()
        payload["timings_seconds"]["model_loading"] = (
            time.perf_counter() - stage_started
        )

        stage = "inference"
        result = recognize_prepared_av(
            assets,
            prepared,
            decoder=decoder,
            beam_size=beam_size,
            ctc_weight=ctc_weight,
            inference_mode=inference_mode,
        )
        payload["result"] = result.to_dict()
        payload["timings_seconds"]["inference"] = result.inference_seconds

        if reference_text is not None:
            stage = "evaluation"
            payload["evaluation"] = evaluate_transcript(
                reference_text,
                result.transcript,
            ).to_dict()

        payload.update({"status": "passed", "stage": "complete"})
        return _finish_report(payload, paths, started_at)
    except ModelAssetsError as exc:
        return _failure(
            payload=payload,
            paths=paths,
            started_at=started_at,
            stage=exc.stage,
            error_type=type(exc).__name__,
            message=str(exc),
        )
    except (MediaInputError, OSError, TypeError, ValueError) as exc:
        return _failure(
            payload=payload,
            paths=paths,
            started_at=started_at,
            stage=stage,
            error_type=type(exc).__name__,
            message=str(exc),
        )
