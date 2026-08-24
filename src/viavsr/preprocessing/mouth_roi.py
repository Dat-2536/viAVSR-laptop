from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .errors import MediaInputError
from .face_tracking import (
    FACE_TRACK_ARTIFACT_VERSION,
    LANDMARK_COUNT,
    iter_resampled_rgb_frames,
)
from .media import (
    MOUTH_ROI_SIZE,
    MediaMetadata,
    probe_av_media,
    validate_demo_media,
)

MEAN_FACE_PATH = Path(__file__).with_name("assets") / "20words_mean_face.json"
ALIGNED_FACE_SIZE = 256
MOUTH_START_INDEX = 48
MOUTH_STOP_INDEX = 68
SMOOTHING_WINDOW = 12
STABLE_LANDMARK_INDICES = (28, 33, 36, 39, 42, 45, 48, 54)


@dataclass(frozen=True)
class FaceTrackArtifact:
    """Validated numeric face-track arrays consumed by mouth alignment."""

    landmarks: np.ndarray
    detected: np.ndarray
    original_width: int
    original_height: int
    frame_rate: int
    artifact_version: int
    quality_passed: bool

    @property
    def frame_count(self) -> int:
        return int(self.landmarks.shape[0])


@dataclass(frozen=True)
class MouthROIExportResult:
    """Metadata for one synchronized aligned-mouth video export."""

    source_path: str
    track_path: str
    output_path: str
    frame_count: int
    frame_rate: int
    mouth_roi_size: int
    detected_frames: int
    interpolated_frames: int
    output_media: MediaMetadata

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_media"] = self.output_media.to_dict()
        return payload


def load_mean_face(path: Path | str = MEAN_FACE_PATH) -> np.ndarray:
    """Load the attributed AVSRCocktail 68-point mean-face reference."""
    asset_path = Path(path).expanduser().resolve()
    if not asset_path.is_file():
        raise MediaInputError(f"Mean-face asset does not exist: {asset_path}")
    try:
        payload = json.loads(asset_path.read_text(encoding="utf-8"))
        points = np.asarray(payload["points"], dtype=np.float32)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise MediaInputError(f"Could not load mean-face asset: {exc}") from exc
    if points.shape != (LANDMARK_COUNT, 2) or not np.isfinite(points).all():
        raise MediaInputError(
            f"Mean-face asset must contain finite [68, 2] points, got {points.shape}."
        )
    return points


def load_face_track_artifact(path: Path | str) -> FaceTrackArtifact:
    """Load and validate a safe numeric face-track NPZ artifact."""
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.is_file():
        raise MediaInputError(f"Face-track artifact does not exist: {artifact_path}")
    try:
        with np.load(artifact_path, allow_pickle=False) as payload:
            landmarks = np.asarray(payload["landmarks"], dtype=np.float32).copy()
            detected = np.asarray(payload["detected"], dtype=np.bool_).copy()
            original_resolution = np.asarray(
                payload["original_resolution"], dtype=np.int32
            )
            frame_rate_value = np.asarray(payload["frame_rate"], dtype=np.int32)
            artifact_version_value = np.asarray(
                payload["artifact_version"],
                dtype=np.int32,
            )
            quality_passed_value = np.asarray(payload["quality_passed"], dtype=np.bool_)
    except (OSError, ValueError, KeyError) as exc:
        raise MediaInputError(
            "Could not load VIAVSR-7 face-track artifact with quality metadata: "
            f"{exc}. Rerun scripts/track_webcam_faces.py."
        ) from exc

    if landmarks.ndim != 3 or landmarks.shape[1:] != (LANDMARK_COUNT, 2):
        raise MediaInputError(
            f"Tracked landmarks must have shape [T, 68, 2], got {landmarks.shape}."
        )
    if detected.shape != (len(landmarks),):
        raise MediaInputError("Face-track detection mask has the wrong shape.")
    if (
        original_resolution.shape != (2,)
        or frame_rate_value.shape != (1,)
        or artifact_version_value.shape != (1,)
        or quality_passed_value.shape != (1,)
    ):
        raise MediaInputError("Face-track metadata arrays have invalid shapes.")
    if not np.isfinite(landmarks).all():
        raise MediaInputError("Face-track landmarks contain non-finite values.")
    width, height = (int(value) for value in original_resolution)
    frame_rate = int(frame_rate_value[0])
    if width <= 0 or height <= 0 or frame_rate <= 0:
        raise MediaInputError("Face-track dimensions and frame rate must be positive.")
    artifact_version = int(artifact_version_value[0])
    if artifact_version != FACE_TRACK_ARTIFACT_VERSION:
        raise MediaInputError(
            "Unsupported face-track artifact version "
            f"{artifact_version}; expected {FACE_TRACK_ARTIFACT_VERSION}. "
            "Rerun scripts/track_webcam_faces.py."
        )
    quality_passed = bool(quality_passed_value[0])
    if not quality_passed:
        raise MediaInputError(
            "Face-track quality gates failed. Inspect the paired tracking JSON "
            "report and record the sample again before mouth-ROI extraction."
        )
    return FaceTrackArtifact(
        landmarks=landmarks,
        detected=detected,
        original_width=width,
        original_height=height,
        frame_rate=frame_rate,
        artifact_version=artifact_version,
        quality_passed=quality_passed,
    )


def smooth_landmarks(
    landmarks: np.ndarray, *, window_size: int = SMOOTHING_WINDOW
) -> np.ndarray:
    """Apply the upstream centered smoothing while preserving frame translation."""
    if landmarks.ndim != 3 or landmarks.shape[1:] != (LANDMARK_COUNT, 2):
        raise MediaInputError(f"Expected landmarks [T, 68, 2], got {landmarks.shape}.")
    if len(landmarks) == 0 or window_size <= 0:
        raise MediaInputError("Landmarks and smoothing window must be non-empty.")
    half_window = window_size // 2
    smoothed = np.empty_like(landmarks, dtype=np.float32)
    for frame_index in range(len(landmarks)):
        margin = min(
            half_window,
            frame_index,
            len(landmarks) - 1 - frame_index,
        )
        local = landmarks[frame_index - margin : frame_index + margin + 1].mean(axis=0)
        local += landmarks[frame_index].mean(axis=0) - local.mean(axis=0)
        smoothed[frame_index] = local
    return smoothed


def estimate_alignment_transform(
    landmarks: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Estimate the official partial-affine transform from stable landmarks."""
    stable = np.asarray(STABLE_LANDMARK_INDICES)
    transform, _ = cv2.estimateAffinePartial2D(
        landmarks[stable],
        reference[stable],
        method=cv2.LMEDS,
    )
    if transform is None or transform.shape != (2, 3):
        raise MediaInputError("Could not estimate a stable face alignment transform.")
    return np.asarray(transform, dtype=np.float32)


def align_and_crop_mouth_frame(
    frame_rgb: np.ndarray,
    landmarks: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Align one face to 256x256 and cut its 96x96 grayscale mouth ROI."""
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise MediaInputError(
            f"Expected an RGB frame [H, W, 3], got {frame_rgb.shape}."
        )
    transform = estimate_alignment_transform(landmarks, reference)
    grayscale = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    aligned = cv2.warpAffine(
        grayscale,
        transform,
        dsize=(ALIGNED_FACE_SIZE, ALIGNED_FACE_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    transformed_landmarks = landmarks @ transform[:, :2].transpose() + transform[:, 2]
    mouth_center = transformed_landmarks[MOUTH_START_INDEX:MOUTH_STOP_INDEX].mean(
        axis=0
    )
    half_size = MOUTH_ROI_SIZE // 2
    center_x, center_y = (round(value) for value in mouth_center)
    x_min = center_x - half_size
    y_min = center_y - half_size
    x_max = x_min + MOUTH_ROI_SIZE
    y_max = y_min + MOUTH_ROI_SIZE
    if x_min < 0 or y_min < 0 or x_max > ALIGNED_FACE_SIZE or y_max > ALIGNED_FACE_SIZE:
        raise MediaInputError(
            "Aligned mouth crop falls outside the 256x256 reference frame."
        )
    patch = aligned[y_min:y_max, x_min:x_max]
    if patch.shape != (MOUTH_ROI_SIZE, MOUTH_ROI_SIZE):
        raise MediaInputError(f"Unexpected mouth patch shape: {patch.shape}.")
    return np.ascontiguousarray(patch, dtype=np.uint8)


def _require_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise MediaInputError("Required executable 'ffmpeg' was not found.")
    return executable


def _encode_mouth_video(
    patches: Iterable[np.ndarray],
    *,
    source_path: Path,
    output_path: Path,
    frame_rate: int,
) -> int:
    ffmpeg = _require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path == output_path:
        raise MediaInputError("Mouth-ROI output must differ from the source video.")
    command = [
        ffmpeg,
        "-y",
        "-nostdin",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s:v",
        f"{MOUTH_ROI_SIZE}x{MOUTH_ROI_SIZE}",
        "-r",
        str(frame_rate),
        "-i",
        "pipe:0",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-shortest",
        str(output_path),
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise MediaInputError(f"Could not start FFmpeg mouth encoder: {exc}") from exc
    assert process.stdin is not None
    assert process.stderr is not None
    frame_count = 0
    succeeded = False
    try:
        for patch in patches:
            if patch.shape != (MOUTH_ROI_SIZE, MOUTH_ROI_SIZE):
                raise MediaInputError(f"Cannot encode mouth patch {patch.shape}.")
            process.stdin.write(np.ascontiguousarray(patch).tobytes())
            frame_count += 1
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        if return_code != 0:
            raise MediaInputError(f"FFmpeg mouth encoding failed: {stderr}")
        succeeded = True
    except (BrokenPipeError, OSError) as exc:
        raise MediaInputError(f"Could not write mouth frames to FFmpeg: {exc}") from exc
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        process.stderr.close()
        if not process.stdin.closed:
            process.stdin.close()
        if not succeeded:
            output_path.unlink(missing_ok=True)
    return frame_count


def export_aligned_mouth_roi_video(
    source_path: Path | str,
    track_path: Path | str,
    output_path: Path | str,
    *,
    mean_face_path: Path | str = MEAN_FACE_PATH,
) -> MouthROIExportResult:
    """Export an aligned 96x96 mouth video with synchronized source audio."""
    source = Path(source_path).expanduser().resolve()
    track_file = Path(track_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    source_metadata = probe_av_media(source)
    track = load_face_track_artifact(track_file)
    if (track.original_width, track.original_height) != (
        source_metadata.video_width,
        source_metadata.video_height,
    ):
        raise MediaInputError(
            "Face-track original resolution does not match the source video."
        )
    reference = load_mean_face(mean_face_path)
    smoothed = smooth_landmarks(track.landmarks)
    decoded_frames = iter_resampled_rgb_frames(
        source,
        width=track.original_width,
        height=track.original_height,
        frame_rate=track.frame_rate,
    )

    consumed_frames = 0

    def mouth_patches() -> Iterable[np.ndarray]:
        nonlocal consumed_frames
        for frame_index, frame in enumerate(decoded_frames):
            if frame_index >= track.frame_count:
                raise MediaInputError(
                    "Decoded video has more frames than the face-track artifact."
                )
            consumed_frames += 1
            yield align_and_crop_mouth_frame(
                frame,
                smoothed[frame_index],
                reference,
            )

    encoded_frames = _encode_mouth_video(
        mouth_patches(),
        source_path=source,
        output_path=output,
        frame_rate=track.frame_rate,
    )
    if consumed_frames != track.frame_count or encoded_frames != track.frame_count:
        output.unlink(missing_ok=True)
        raise MediaInputError(
            "Face-track/video frame mismatch: "
            f"track={track.frame_count}, decoded={consumed_frames}, "
            f"encoded={encoded_frames}."
        )
    output_metadata = probe_av_media(output)
    validate_demo_media(output_metadata)
    return MouthROIExportResult(
        source_path=str(source),
        track_path=str(track_file),
        output_path=str(output),
        frame_count=encoded_frames,
        frame_rate=track.frame_rate,
        mouth_roi_size=MOUTH_ROI_SIZE,
        detected_frames=int(track.detected.sum()),
        interpolated_frames=track.frame_count - int(track.detected.sum()),
        output_media=output_metadata,
    )
