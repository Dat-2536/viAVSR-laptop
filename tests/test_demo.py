from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from viavsr import demo
from viavsr.preprocessing import MediaInputError
from viavsr.preprocessing.face_tracking import FaceTrackingQualityPolicy
from viavsr.preprocessing.media import MediaMetadata


def _metadata(path: Path, *, mouth_roi: bool = False) -> MediaMetadata:
    size = 96 if mouth_roi else 1920
    height = 96 if mouth_roi else 1080
    return MediaMetadata(
        path=str(path),
        duration_seconds=4.0,
        video_width=size,
        video_height=height,
        frame_rate=25.0,
        audio_sample_rate=16_000,
        audio_channels=1,
    )


def _install_successful_stages(
    monkeypatch: pytest.MonkeyPatch,
    *,
    raw_media: Path,
    calls: list[str],
) -> None:
    monkeypatch.setattr(
        demo,
        "load_face_tracking_quality_policy",
        lambda _path: FaceTrackingQualityPolicy(),
    )
    monkeypatch.setattr(demo, "probe_av_media", lambda _path: _metadata(raw_media))
    monkeypatch.setattr(
        demo,
        "FANFaceLandmarker",
        lambda **_kwargs: SimpleNamespace(name="fake", device="cpu"),
    )

    def track(
        _media: Path,
        *,
        landmarker: object,
        frame_rate: int,
        max_detection_size: int,
        policy: FaceTrackingQualityPolicy,
    ):
        assert landmarker is not None
        assert frame_rate == 25
        assert max_detection_size == 640
        assert isinstance(policy, FaceTrackingQualityPolicy)
        calls.append("track")
        return SimpleNamespace(quality_passed=True, quality_issues=[])

    monkeypatch.setattr(demo, "track_face_landmarks", track)

    def save_track(_sequence, *, artifact_path: Path, report_path: Path):
        calls.append("save_track")
        artifact_path.write_bytes(b"track")
        return {
            "status": "passed",
            "quality_status": "passed",
            "artifact_path": str(artifact_path),
            "report_path": str(report_path),
        }

    monkeypatch.setattr(demo, "save_face_tracking_artifacts", save_track)

    def export_display(_media, output: Path, *, track_path: Path | None):
        calls.append("display")
        assert track_path is not None
        output.write_bytes(b"display")
        availability = {
            "frame_rate": 25,
            "frame_count": 100,
            "valid_frames": 100,
            "missing_frames": 0,
            "coverage": 1.0,
            "missing_intervals": [],
        }
        return SimpleNamespace(
            visual_availability=availability,
            to_dict=lambda: {
                "output_path": str(output),
                "visual_availability": availability,
            },
        )

    monkeypatch.setattr(demo, "export_mouth_roi_display_video", export_display)

    def export_mouth(_media, _track, output: Path):
        calls.append("mouth")
        output.write_bytes(b"mouth")
        return SimpleNamespace(
            to_dict=lambda: {"status": "passed", "output_path": str(output)}
        )

    monkeypatch.setattr(demo, "export_aligned_mouth_roi_video", export_mouth)
    prepared = SimpleNamespace(
        metadata=_metadata(raw_media.with_name("mouth96.mp4"), mouth_roi=True),
        shape_report=lambda: {"videos": [1, 1, 100, 88, 88]},
    )
    monkeypatch.setattr(
        demo,
        "prepare_mouth_roi_media",
        lambda *_args, **_kwargs: calls.append("prepare") or prepared,
    )
    monkeypatch.setattr(
        demo,
        "load_model_assets_config",
        lambda _path: calls.append("load_config") or object(),
    )
    assets = SimpleNamespace(
        report=SimpleNamespace(
            to_dict=lambda: {
                "repository_id": "example/vi-model",
                "revision": "main",
                "tokenizer_vocabulary_size": 2048,
            }
        )
    )
    monkeypatch.setattr(
        demo,
        "load_vietnamese_avsr_assets",
        lambda _config: calls.append("load_model") or assets,
    )
    result = SimpleNamespace(
        transcript="xin chao",
        inference_seconds=0.25,
        to_dict=lambda: {
            "transcript": "xin chao",
            "decoder": "joint_beam_search",
            "inference_seconds": 0.25,
        },
    )

    def recognize(actual_assets, actual_prepared, **_kwargs):
        assert actual_assets is assets
        assert actual_prepared is prepared
        assert _kwargs["inference_mode"] == "audio_visual"
        calls.append("infer")
        return result

    monkeypatch.setattr(demo, "recognize_prepared_av", recognize)
    monkeypatch.setattr(
        demo,
        "evaluate_transcript",
        lambda _reference, _prediction: SimpleNamespace(
            to_dict=lambda: {"wer": 0.0, "cer": 0.0}
        ),
    )


def _install_audio_fallback_downstream(
    monkeypatch: pytest.MonkeyPatch,
    *,
    raw_media: Path,
    calls: list[str],
) -> None:
    prepared = SimpleNamespace(
        metadata=_metadata(raw_media),
        shape_report=lambda: {
            "videos": [1, 1, 100, 88, 88],
            "audios": [1, 104, 100],
        },
    )
    monkeypatch.setattr(
        demo,
        "prepare_audio_only_media",
        lambda *_args, **_kwargs: calls.append("prepare_audio") or prepared,
    )

    def export_display(_media, output: Path, *, track_path: Path | None):
        calls.append("display")
        output.write_bytes(b"display")
        valid_frames = 50 if track_path is not None else 0
        availability = {
            "frame_rate": 25,
            "frame_count": 100,
            "valid_frames": valid_frames,
            "missing_frames": 100 - valid_frames,
            "coverage": valid_frames / 100,
            "missing_intervals": [
                {
                    "start_frame": valid_frames,
                    "end_frame_exclusive": 100,
                    "frame_count": 100 - valid_frames,
                    "start_seconds": valid_frames / 25,
                    "end_seconds": 4.0,
                    "duration_seconds": (100 - valid_frames) / 25,
                }
            ],
        }
        return SimpleNamespace(
            visual_availability=availability,
            to_dict=lambda: {"visual_availability": availability},
        )

    monkeypatch.setattr(demo, "export_mouth_roi_display_video", export_display)
    monkeypatch.setattr(
        demo,
        "export_aligned_mouth_roi_video",
        lambda *_args, **_kwargs: pytest.fail("mouth export must not run"),
    )
    monkeypatch.setattr(
        demo,
        "prepare_mouth_roi_media",
        lambda *_args, **_kwargs: pytest.fail("AV preparation must not run"),
    )
    monkeypatch.setattr(
        demo,
        "load_model_assets_config",
        lambda _path: calls.append("load_config") or object(),
    )
    assets = SimpleNamespace(
        report=SimpleNamespace(to_dict=lambda: {"repository_id": "example/vi-model"})
    )
    monkeypatch.setattr(
        demo,
        "load_vietnamese_avsr_assets",
        lambda _config: calls.append("load_model") or assets,
    )
    result = SimpleNamespace(
        transcript="audio transcript",
        inference_seconds=0.2,
        to_dict=lambda: {
            "transcript": "audio transcript",
            "inference_mode": "audio_only_fallback",
            "visual_input_used": False,
        },
    )

    def recognize(actual_assets, actual_prepared, **kwargs):
        assert actual_assets is assets
        assert actual_prepared is prepared
        assert kwargs["inference_mode"] == "audio_only_fallback"
        calls.append("infer_audio")
        return result

    monkeypatch.setattr(demo, "recognize_prepared_av", recognize)


def test_end_to_end_demo_writes_consolidated_success_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_media = tmp_path / "webcam_001.mp4"
    raw_media.write_bytes(b"raw")
    calls: list[str] = []
    _install_successful_stages(monkeypatch, raw_media=raw_media, calls=calls)

    payload = demo.run_end_to_end_demo(
        config_path=tmp_path / "config.yaml",
        media_path=raw_media,
        output_root=tmp_path / "outputs",
        reference_text="xin chao",
        beam_size=3,
        ctc_weight=0.1,
    )

    paths = demo.DemoArtifactPaths.for_media(raw_media, tmp_path / "outputs")
    assert payload["status"] == "passed"
    assert payload["schema_version"] == 2
    assert payload["stage"] == "complete"
    assert payload["evaluation"] == {"wer": 0.0, "cer": 0.0}
    assert payload["request"]["decoder"] == "joint_beam_search"
    assert payload["mouth_roi_display"]["visual_availability"]["coverage"] == 1.0
    assert payload["mouth_roi_display"]["used_for_inference"] is False
    assert payload["modality_decision"]["visual_gap_policy"] == (
        "interpolated_landmarks"
    )
    assert json.loads(paths.report.read_text(encoding="utf-8")) == payload
    assert paths.face_track.is_file()
    assert paths.mouth_roi.is_file()
    assert paths.mouth_roi_display.is_file()
    assert calls == [
        "track",
        "save_track",
        "display",
        "mouth",
        "prepare",
        "load_config",
        "load_model",
        "infer",
    ]


def test_display_export_failure_does_not_suppress_transcription(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_media = tmp_path / "webcam_001.mp4"
    raw_media.write_bytes(b"raw")
    calls: list[str] = []
    _install_successful_stages(monkeypatch, raw_media=raw_media, calls=calls)
    monkeypatch.setattr(
        demo,
        "export_mouth_roi_display_video",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MediaInputError("display encoder unavailable")
        ),
    )

    payload = demo.run_end_to_end_demo(
        config_path=tmp_path / "config.yaml",
        media_path=raw_media,
        output_root=tmp_path / "outputs",
    )

    paths = demo.DemoArtifactPaths.for_media(raw_media, tmp_path / "outputs")
    assert payload["status"] == "passed"
    assert payload["result"]["transcript"] == "xin chao"
    assert payload["mouth_roi_display"]["status"] == "unavailable"
    assert payload["mouth_roi_display"]["used_for_inference"] is False
    assert "mouth_roi_display_unavailable" in payload["warnings"]
    assert not paths.mouth_roi_display.exists()
    assert paths.mouth_roi_display_report.is_file()
    assert paths.mouth_roi.is_file()


def test_quality_failure_falls_back_to_audio_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_media = tmp_path / "turning_away.mp4"
    raw_media.write_bytes(b"raw")
    calls: list[str] = []
    monkeypatch.setattr(
        demo,
        "load_face_tracking_quality_policy",
        lambda _path: FaceTrackingQualityPolicy(),
    )
    monkeypatch.setattr(demo, "probe_av_media", lambda _path: _metadata(raw_media))
    monkeypatch.setattr(
        demo,
        "FANFaceLandmarker",
        lambda **_kwargs: SimpleNamespace(name="fake", device="cpu"),
    )
    sequence = SimpleNamespace(
        quality_passed=False,
        quality_issues=["trailing_missing_run_above_maximum:8>0"],
    )
    monkeypatch.setattr(
        demo,
        "track_face_landmarks",
        lambda *_args, **_kwargs: calls.append("track") or sequence,
    )

    def save_failed(_sequence, *, artifact_path: Path, report_path: Path):
        calls.append("save_track")
        artifact_path.write_bytes(b"failed-track")
        return {
            "status": "failed",
            "quality_status": "failed",
            "quality_issues": sequence.quality_issues,
            "artifact_path": str(artifact_path),
            "report_path": str(report_path),
        }

    monkeypatch.setattr(demo, "save_face_tracking_artifacts", save_failed)
    _install_audio_fallback_downstream(
        monkeypatch,
        raw_media=raw_media,
        calls=calls,
    )

    payload = demo.run_end_to_end_demo(
        config_path=tmp_path / "config.yaml",
        media_path=raw_media,
        output_root=tmp_path / "outputs",
    )

    paths = demo.DemoArtifactPaths.for_media(raw_media, tmp_path / "outputs")
    assert payload["status"] == "passed"
    assert payload["stage"] == "complete"
    assert payload["face_tracking"]["quality_status"] == "failed"
    assert payload["modality_decision"]["selected_mode"] == "audio_only_fallback"
    assert payload["modality_decision"]["fallback_reason"]["stage"] == (
        "face_tracking_quality"
    )
    assert payload["modality_decision"]["visual_gap_policy"] == (
        "visual_input_not_used"
    )
    assert payload["result"]["transcript"] == "audio transcript"
    assert paths.face_track.is_file()
    assert not paths.mouth_roi.exists()
    assert paths.mouth_roi_display.is_file()
    assert payload["mouth_roi_display"]["visual_availability"]["coverage"] == 0.5
    assert calls == [
        "track",
        "save_track",
        "display",
        "prepare_audio",
        "load_config",
        "load_model",
        "infer_audio",
    ]


def test_no_face_error_invalidates_stale_visuals_and_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_media = tmp_path / "webcam.mp4"
    raw_media.write_bytes(b"raw")
    paths = demo.DemoArtifactPaths.for_media(raw_media, tmp_path / "outputs")
    paths.run_directory.mkdir(parents=True)
    for stale_path in (
        paths.face_track,
        paths.face_tracking_report,
        paths.mouth_roi,
        paths.mouth_roi_report,
        paths.mouth_roi_display,
        paths.mouth_roi_display_report,
        paths.report,
    ):
        stale_path.write_bytes(b"stale")

    monkeypatch.setattr(
        demo,
        "load_face_tracking_quality_policy",
        lambda _path: FaceTrackingQualityPolicy(),
    )
    monkeypatch.setattr(demo, "probe_av_media", lambda _path: _metadata(raw_media))
    monkeypatch.setattr(
        demo,
        "FANFaceLandmarker",
        lambda **_kwargs: SimpleNamespace(name="fake", device="cpu"),
    )
    monkeypatch.setattr(
        demo,
        "track_face_landmarks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MediaInputError("No face was detected in any video frame.")
        ),
    )
    calls: list[str] = []
    _install_audio_fallback_downstream(
        monkeypatch,
        raw_media=raw_media,
        calls=calls,
    )

    payload = demo.run_end_to_end_demo(
        config_path=tmp_path / "config.yaml",
        media_path=raw_media,
        output_root=tmp_path / "outputs",
    )

    assert payload["status"] == "passed"
    assert payload["stage"] == "complete"
    assert payload["modality_decision"]["selected_mode"] == "audio_only_fallback"
    assert payload["result"]["visual_input_used"] is False
    assert not paths.face_track.exists()
    assert not paths.mouth_roi.exists()
    assert not paths.mouth_roi_report.exists()
    assert paths.mouth_roi_display.is_file()
    assert payload["mouth_roi_display"]["visual_availability"]["coverage"] == 0.0
    assert payload["face_tracking"]["visual_availability"]["missing_frames"] == 100
    face_report = json.loads(paths.face_tracking_report.read_text(encoding="utf-8"))
    assert face_report["status"] == "unavailable"
    assert face_report["error"]["message"] == (
        "No face was detected in any video frame."
    )
    assert json.loads(paths.report.read_text(encoding="utf-8"))["status"] == "passed"
    assert calls == [
        "display",
        "prepare_audio",
        "load_config",
        "load_model",
        "infer_audio",
    ]


def test_media_preflight_failure_remains_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_media = tmp_path / "silent.mp4"
    raw_media.write_bytes(b"raw")
    monkeypatch.setattr(
        demo,
        "load_face_tracking_quality_policy",
        lambda _path: FaceTrackingQualityPolicy(),
    )
    monkeypatch.setattr(
        demo,
        "probe_av_media",
        lambda _path: (_ for _ in ()).throw(
            MediaInputError("Media file has no audio stream")
        ),
    )
    monkeypatch.setattr(
        demo,
        "FANFaceLandmarker",
        lambda **_kwargs: pytest.fail("tracking backend must not load"),
    )

    payload = demo.run_end_to_end_demo(
        config_path=tmp_path / "config.yaml",
        media_path=raw_media,
        output_root=tmp_path / "outputs",
    )

    assert payload["status"] == "failed"
    assert payload["stage"] == "media_preflight"
    assert payload["error"]["message"] == "Media file has no audio stream"


@pytest.mark.parametrize(
    ("status", "exit_code"),
    [("passed", 0), ("failed", 1)],
)
def test_cli_exit_code_reflects_pipeline_status(
    status: str,
    exit_code: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import run_avsr_demo

    monkeypatch.setattr(
        run_avsr_demo,
        "run_end_to_end_demo",
        lambda **_kwargs: {"status": status},
    )

    actual = run_avsr_demo.main(["--media", "sample.mp4"])

    assert actual == exit_code
    assert json.loads(capsys.readouterr().out) == {"status": status}
