from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch

from .errors import MediaInputError
from .media import TARGET_FRAME_RATE, MediaMetadata, probe_av_media

DeviceRequest = Literal["auto", "cpu", "cuda"]
LANDMARK_COUNT = 68
DEFAULT_DETECTION_MAX_SIZE = 640


@dataclass(frozen=True)
class FaceCandidate:
    """One detected face and its FAN 68-point landmark prediction."""

    bounding_box: np.ndarray
    detection_confidence: float
    landmarks: np.ndarray
    landmark_scores: np.ndarray

    def __post_init__(self) -> None:
        if self.bounding_box.shape != (4,):
            raise ValueError("bounding_box must have shape [4].")
        if self.landmarks.shape != (LANDMARK_COUNT, 2):
            raise ValueError("landmarks must have shape [68, 2].")
        if self.landmark_scores.shape != (LANDMARK_COUNT,):
            raise ValueError("landmark_scores must have shape [68].")


class FaceLandmarker(Protocol):
    """Backend interface used by the deterministic temporal tracker."""

    name: str
    device: str

    def detect(self, frame_rgb: np.ndarray) -> list[FaceCandidate]: ...


@dataclass(frozen=True)
class TrackedFaceSequence:
    """Frame-aligned face boxes and 68-point landmarks for one video."""

    media: MediaMetadata
    processing_width: int
    processing_height: int
    frame_rate: int
    backend: str
    device: str
    landmarks: np.ndarray
    bounding_boxes: np.ndarray
    landmark_scores: np.ndarray
    detection_confidences: np.ndarray
    detected: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.landmarks.shape[0])

    @property
    def detected_frames(self) -> int:
        return int(self.detected.sum())

    @property
    def interpolated_frames(self) -> int:
        return self.frame_count - self.detected_frames

    @property
    def detection_rate(self) -> float:
        return self.detected_frames / self.frame_count

    def report(self) -> dict[str, Any]:
        valid_confidences = self.detection_confidences[self.detected]
        valid_landmark_scores = self.landmark_scores[self.detected]
        widths = self.bounding_boxes[:, 2] - self.bounding_boxes[:, 0]
        heights = self.bounding_boxes[:, 3] - self.bounding_boxes[:, 1]
        frame_area = self.media.video_width * self.media.video_height
        face_area_ratios = widths * heights / frame_area
        return {
            "status": "passed",
            "media": self.media.to_dict(),
            "backend": self.backend,
            "device": self.device,
            "frame_rate": self.frame_rate,
            "frame_count": self.frame_count,
            "processing_resolution": [
                self.processing_width,
                self.processing_height,
            ],
            "landmark_topology": "ibug_68",
            "detected_frames": self.detected_frames,
            "interpolated_frames": self.interpolated_frames,
            "detection_rate": self.detection_rate,
            "maximum_missing_run": maximum_false_run(self.detected),
            "mean_detection_confidence": float(valid_confidences.mean()),
            "mean_landmark_confidence": float(valid_landmark_scores.mean()),
            "median_face_area_ratio": float(np.median(face_area_ratios)),
        }


def resolve_tracking_device(requested: DeviceRequest) -> str:
    """Resolve an explicit or automatic tracking device."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise MediaInputError(
            "CUDA was requested for face tracking, but torch.cuda.is_available() "
            "is false. Use --device cpu or --device auto."
        )
    return requested


class FANFaceLandmarker:
    """RetinaFace detection plus compatible 68-point FAN landmarks."""

    name = "face_alignment_1.5.0_retinaface_fan4"

    def __init__(
        self,
        *,
        device: DeviceRequest = "auto",
        confidence_threshold: float = 0.8,
    ) -> None:
        self.device = resolve_tracking_device(device)
        try:
            import face_alignment
        except ImportError as exc:
            raise MediaInputError(
                "Face tracking requires face-alignment==1.5.0. "
                "Install the project dependencies with pip install -e '.[dev]'."
            ) from exc

        try:
            self._predictor = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                device=self.device,
                flip_input=False,
                face_detector="retinaface",
                face_detector_kwargs={
                    "confidence_threshold": confidence_threshold,
                },
                compile=False,
            )
        except Exception as exc:
            raise MediaInputError(
                f"Could not initialize RetinaFace/FAN on {self.device}: {exc}"
            ) from exc

    def detect(self, frame_rgb: np.ndarray) -> list[FaceCandidate]:
        """Detect every face and return its box, confidence, and landmarks."""
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise MediaInputError(
                f"Expected an RGB frame [H, W, 3], got {list(frame_rgb.shape)}."
            )
        try:
            landmarks, scores, boxes = self._predictor.get_landmarks_from_image(
                frame_rgb,
                return_bboxes=True,
                return_landmark_score=True,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise MediaInputError(f"RetinaFace/FAN prediction failed: {exc}") from exc
        if landmarks is None or scores is None or boxes is None:
            return []

        candidates: list[FaceCandidate] = []
        for points, point_scores, raw_box in zip(landmarks, scores, boxes, strict=True):
            box = np.asarray(raw_box, dtype=np.float32)
            candidates.append(
                FaceCandidate(
                    bounding_box=box[:4].copy(),
                    detection_confidence=float(box[4]),
                    landmarks=np.asarray(points, dtype=np.float32),
                    landmark_scores=np.asarray(point_scores, dtype=np.float32),
                )
            )
        return candidates


def scaled_detection_size(
    width: int, height: int, max_dimension: int
) -> tuple[int, int]:
    """Return an even, aspect-preserving size without upscaling."""
    if width <= 0 or height <= 0 or max_dimension <= 0:
        raise ValueError("Dimensions must be positive.")
    scale = min(1.0, max_dimension / max(width, height))
    scaled_width = max(2, round(width * scale / 2) * 2)
    scaled_height = max(2, round(height * scale / 2) * 2)
    return scaled_width, scaled_height


def _require_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise MediaInputError(
            "Required executable 'ffmpeg' was not found in the active environment."
        )
    return executable


def _read_exact(stream: Any, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def iter_resampled_rgb_frames(
    path: Path,
    *,
    width: int,
    height: int,
    frame_rate: int = TARGET_FRAME_RATE,
) -> Iterator[np.ndarray]:
    """Stream CFR RGB frames from FFmpeg without loading a video into memory."""
    ffmpeg = _require_ffmpeg()
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vf",
        f"fps={frame_rate},scale={width}:{height}:flags=bicubic,format=rgb24",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise MediaInputError(f"Could not start FFmpeg: {exc}") from exc
    assert process.stdout is not None
    assert process.stderr is not None
    frame_bytes = width * height * 3
    completed = False
    try:
        while True:
            payload = _read_exact(process.stdout, frame_bytes)
            if not payload:
                break
            if len(payload) != frame_bytes:
                raise MediaInputError(
                    "FFmpeg returned an incomplete RGB frame while tracking faces."
                )
            yield (
                np.frombuffer(payload, dtype=np.uint8).copy().reshape(height, width, 3)
            )
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        completed = True
        if return_code != 0:
            raise MediaInputError(f"FFmpeg video decoding failed: {stderr}")
    finally:
        process.stdout.close()
        process.stderr.close()
        if not completed and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def bounding_box_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Calculate intersection over union for two xyxy boxes."""
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(
        0.0, float(right[3] - right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _candidate_area(candidate: FaceCandidate) -> float:
    box = candidate.bounding_box
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def select_tracked_face(
    candidates: Sequence[FaceCandidate], previous_box: np.ndarray | None
) -> FaceCandidate | None:
    """Select the largest initial face, then preserve identity temporally."""
    if not candidates:
        return None
    if previous_box is None:
        return max(
            candidates,
            key=lambda item: (_candidate_area(item), item.detection_confidence),
        )

    previous_center = (previous_box[:2] + previous_box[2:]) / 2.0
    previous_diagonal = math.hypot(
        float(previous_box[2] - previous_box[0]),
        float(previous_box[3] - previous_box[1]),
    )
    previous_diagonal = max(previous_diagonal, 1.0)

    def tracking_score(candidate: FaceCandidate) -> float:
        box = candidate.bounding_box
        center = (box[:2] + box[2:]) / 2.0
        center_distance = float(np.linalg.norm(center - previous_center))
        normalized_distance = center_distance / previous_diagonal
        return (
            3.0 * bounding_box_iou(previous_box, box)
            - normalized_distance
            + 0.25 * candidate.detection_confidence
        )

    return max(candidates, key=tracking_score)


def _scale_candidate(
    candidate: FaceCandidate, *, scale_x: float, scale_y: float
) -> FaceCandidate:
    box = candidate.bounding_box.copy()
    box[[0, 2]] *= scale_x
    box[[1, 3]] *= scale_y
    landmarks = candidate.landmarks.copy()
    landmarks[:, 0] *= scale_x
    landmarks[:, 1] *= scale_y
    return FaceCandidate(
        bounding_box=box,
        detection_confidence=candidate.detection_confidence,
        landmarks=landmarks,
        landmark_scores=candidate.landmark_scores.copy(),
    )


def maximum_false_run(mask: np.ndarray) -> int:
    """Return the longest consecutive run of false values."""
    longest = current = 0
    for value in mask:
        if bool(value):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def interpolate_missing_rows(values: np.ndarray, detected: np.ndarray) -> np.ndarray:
    """Linearly fill internal gaps and extend nearest values at sequence ends."""
    if values.shape[0] != detected.shape[0]:
        raise ValueError("values and detected must have the same frame count.")
    valid_indices = np.flatnonzero(detected)
    if len(valid_indices) == 0:
        raise MediaInputError("No face was detected in any video frame.")
    all_indices = np.arange(len(values))
    flattened = values.reshape(len(values), -1)
    result = np.empty_like(flattened, dtype=np.float32)
    for column in range(flattened.shape[1]):
        result[:, column] = np.interp(
            all_indices,
            valid_indices,
            flattened[valid_indices, column],
        )
    return result.reshape(values.shape)


def build_tracked_sequence(
    candidates_by_frame: Sequence[Sequence[FaceCandidate]],
    *,
    media: MediaMetadata,
    processing_width: int,
    processing_height: int,
    frame_rate: int,
    backend: str,
    device: str,
) -> TrackedFaceSequence:
    """Select a temporal identity and interpolate missed detections."""
    frame_count = len(candidates_by_frame)
    if frame_count == 0:
        raise MediaInputError("Video contains no decodable frames.")
    landmarks = np.full((frame_count, LANDMARK_COUNT, 2), np.nan, dtype=np.float32)
    boxes = np.full((frame_count, 4), np.nan, dtype=np.float32)
    scores = np.full((frame_count, LANDMARK_COUNT), np.nan, dtype=np.float32)
    confidences = np.full(frame_count, np.nan, dtype=np.float32)
    detected = np.zeros(frame_count, dtype=np.bool_)
    previous_box: np.ndarray | None = None

    for frame_index, candidates in enumerate(candidates_by_frame):
        selected = select_tracked_face(candidates, previous_box)
        if selected is None:
            continue
        landmarks[frame_index] = selected.landmarks
        boxes[frame_index] = selected.bounding_box
        scores[frame_index] = selected.landmark_scores
        confidences[frame_index] = selected.detection_confidence
        detected[frame_index] = True
        previous_box = selected.bounding_box

    interpolated_landmarks = interpolate_missing_rows(landmarks, detected)
    interpolated_boxes = interpolate_missing_rows(boxes, detected)
    return TrackedFaceSequence(
        media=media,
        processing_width=processing_width,
        processing_height=processing_height,
        frame_rate=frame_rate,
        backend=backend,
        device=device,
        landmarks=interpolated_landmarks,
        bounding_boxes=interpolated_boxes,
        landmark_scores=scores,
        detection_confidences=confidences,
        detected=detected,
    )


def track_face_landmarks(
    path: Path | str,
    *,
    landmarker: FaceLandmarker | None = None,
    device: DeviceRequest = "auto",
    frame_rate: int = TARGET_FRAME_RATE,
    max_detection_size: int = DEFAULT_DETECTION_MAX_SIZE,
) -> TrackedFaceSequence:
    """Detect and temporally track one 68-point face across a media file."""
    if frame_rate <= 0:
        raise MediaInputError("Face-tracking frame rate must be positive.")
    if max_detection_size <= 0:
        raise MediaInputError("Face-tracking maximum detection size must be positive.")
    media_path = Path(path).expanduser().resolve()
    metadata = probe_av_media(media_path)
    processing_width, processing_height = scaled_detection_size(
        metadata.video_width,
        metadata.video_height,
        max_detection_size,
    )
    active_landmarker = landmarker or FANFaceLandmarker(device=device)
    scale_x = metadata.video_width / processing_width
    scale_y = metadata.video_height / processing_height
    candidates_by_frame: list[list[FaceCandidate]] = []
    for frame in iter_resampled_rgb_frames(
        media_path,
        width=processing_width,
        height=processing_height,
        frame_rate=frame_rate,
    ):
        candidates_by_frame.append(
            [
                _scale_candidate(item, scale_x=scale_x, scale_y=scale_y)
                for item in active_landmarker.detect(frame)
            ]
        )
    return build_tracked_sequence(
        candidates_by_frame,
        media=metadata,
        processing_width=processing_width,
        processing_height=processing_height,
        frame_rate=frame_rate,
        backend=active_landmarker.name,
        device=active_landmarker.device,
    )


def save_face_tracking_artifacts(
    sequence: TrackedFaceSequence,
    *,
    artifact_path: Path | str,
    report_path: Path | str,
) -> dict[str, Any]:
    """Write a safe numeric NPZ handoff and a human-readable JSON report."""
    artifact = Path(artifact_path).expanduser().resolve()
    report = Path(report_path).expanduser().resolve()
    artifact.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact,
        landmarks=sequence.landmarks,
        bounding_boxes=sequence.bounding_boxes,
        landmark_scores=sequence.landmark_scores,
        detection_confidences=sequence.detection_confidences,
        detected=sequence.detected,
        original_resolution=np.asarray(
            [sequence.media.video_width, sequence.media.video_height],
            dtype=np.int32,
        ),
        processing_resolution=np.asarray(
            [sequence.processing_width, sequence.processing_height],
            dtype=np.int32,
        ),
        frame_rate=np.asarray([sequence.frame_rate], dtype=np.int32),
    )
    payload = sequence.report()
    payload["artifact_path"] = str(artifact)
    payload["report_path"] = str(report)
    report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload
