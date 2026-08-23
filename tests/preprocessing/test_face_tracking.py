import json

import numpy as np
import pytest

from viavsr.preprocessing.errors import MediaInputError
from viavsr.preprocessing.face_tracking import (
    FaceCandidate,
    bounding_box_iou,
    build_tracked_sequence,
    interpolate_missing_rows,
    maximum_false_run,
    save_face_tracking_artifacts,
    scaled_detection_size,
    select_tracked_face,
    track_face_landmarks,
)
from viavsr.preprocessing.media import MediaMetadata


def _candidate(
    box: tuple[float, float, float, float],
    *,
    confidence: float = 0.99,
    landmark_value: float = 1.0,
) -> FaceCandidate:
    return FaceCandidate(
        bounding_box=np.asarray(box, dtype=np.float32),
        detection_confidence=confidence,
        landmarks=np.full((68, 2), landmark_value, dtype=np.float32),
        landmark_scores=np.full(68, 0.9, dtype=np.float32),
    )


def _metadata() -> MediaMetadata:
    return MediaMetadata(
        path="/tmp/webcam.mp4",
        duration_seconds=4.0,
        video_width=1920,
        video_height=1080,
        frame_rate=15.0,
        audio_sample_rate=48_000,
        audio_channels=2,
    )


def test_scaled_detection_size_preserves_aspect_ratio_without_upscaling() -> None:
    assert scaled_detection_size(1920, 1080, 640) == (640, 360)
    assert scaled_detection_size(320, 240, 640) == (320, 240)


def test_track_face_landmarks_rejects_invalid_processing_parameters() -> None:
    with pytest.raises(MediaInputError, match="frame rate must be positive"):
        track_face_landmarks("missing.mp4", frame_rate=0)

    with pytest.raises(
        MediaInputError,
        match="maximum detection size must be positive",
    ):
        track_face_landmarks("missing.mp4", max_detection_size=0)


def test_bounding_box_iou() -> None:
    left = np.asarray([0, 0, 10, 10], dtype=np.float32)
    right = np.asarray([5, 5, 15, 15], dtype=np.float32)

    assert bounding_box_iou(left, right) == pytest.approx(25 / 175)


def test_tracker_selects_largest_initial_face_then_preserves_identity() -> None:
    target = _candidate((10, 10, 50, 50))
    other = _candidate((100, 100, 130, 130))
    initial = select_tracked_face([other, target], previous_box=None)
    assert initial is target

    moved_target = _candidate((12, 11, 52, 51))
    larger_other = _candidate((80, 80, 160, 160))
    selected = select_tracked_face(
        [larger_other, moved_target],
        previous_box=target.bounding_box,
    )

    assert selected is moved_target


def test_interpolate_missing_rows_fills_internal_and_edge_gaps() -> None:
    values = np.asarray(
        [
            [np.nan, np.nan],
            [2.0, 4.0],
            [np.nan, np.nan],
            [6.0, 8.0],
            [np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    detected = np.asarray([False, True, False, True, False])

    result = interpolate_missing_rows(values, detected)

    np.testing.assert_allclose(
        result,
        np.asarray(
            [
                [2.0, 4.0],
                [2.0, 4.0],
                [4.0, 6.0],
                [6.0, 8.0],
                [6.0, 8.0],
            ],
            dtype=np.float32,
        ),
    )


def test_interpolate_missing_rows_rejects_sequence_without_a_face() -> None:
    values = np.full((3, 2), np.nan, dtype=np.float32)

    with pytest.raises(MediaInputError, match="No face was detected"):
        interpolate_missing_rows(values, np.zeros(3, dtype=np.bool_))


def test_maximum_false_run() -> None:
    mask = np.asarray([True, False, False, True, False])
    assert maximum_false_run(mask) == 2


def test_build_tracked_sequence_interpolates_missing_frame() -> None:
    frame_zero = _candidate((10, 10, 50, 50), landmark_value=0.0)
    frame_one = _candidate((12, 12, 52, 52), landmark_value=2.0)
    frame_three = _candidate((16, 16, 56, 56), landmark_value=6.0)

    result = build_tracked_sequence(
        [[frame_zero], [frame_one], [], [frame_three]],
        media=_metadata(),
        processing_width=640,
        processing_height=360,
        frame_rate=25,
        backend="fake",
        device="cpu",
    )

    assert result.frame_count == 4
    assert result.detected_frames == 3
    assert result.interpolated_frames == 1
    assert result.detection_rate == pytest.approx(0.75)
    np.testing.assert_allclose(result.landmarks[2], 4.0)
    np.testing.assert_allclose(result.bounding_boxes[2], [14, 14, 54, 54])
    assert np.isnan(result.landmark_scores[2]).all()


def test_save_face_tracking_artifacts_writes_numeric_npz_and_json(tmp_path) -> None:
    sequence = build_tracked_sequence(
        [[_candidate((10, 10, 50, 50))], [_candidate((11, 10, 51, 50))]],
        media=_metadata(),
        processing_width=640,
        processing_height=360,
        frame_rate=25,
        backend="fake",
        device="cpu",
    )
    artifact_path = tmp_path / "track.npz"
    report_path = tmp_path / "track.json"

    payload = save_face_tracking_artifacts(
        sequence,
        artifact_path=artifact_path,
        report_path=report_path,
    )

    with np.load(artifact_path, allow_pickle=False) as artifact:
        assert artifact["landmarks"].shape == (2, 68, 2)
        assert artifact["bounding_boxes"].shape == (2, 4)
        assert artifact["detected"].tolist() == [True, True]
        assert artifact["original_resolution"].tolist() == [1920, 1080]
        assert artifact["processing_resolution"].tolist() == [640, 360]
        assert artifact["frame_rate"].tolist() == [25]

    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["landmark_topology"] == "ibug_68"
    assert saved_report["detection_rate"] == 1.0
    assert saved_report["artifact_path"] == str(artifact_path.resolve())
