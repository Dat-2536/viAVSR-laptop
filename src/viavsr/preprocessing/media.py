from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from python_speech_features import logfbank

from .errors import MediaInputError

TARGET_SAMPLE_RATE = 16_000
TARGET_FRAME_RATE = 25
MOUTH_ROI_SIZE = 96
MODEL_VIDEO_SIZE = 88
AUDIO_FILTERBANK_BINS = 26
AUDIO_STACK_ORDER = 4
AUDIO_FEATURE_DIM = AUDIO_FILTERBANK_BINS * AUDIO_STACK_ORDER
AUDIO_SAMPLES_PER_VIDEO_FRAME = TARGET_SAMPLE_RATE // TARGET_FRAME_RATE
VIDEO_MEAN = 0.421
VIDEO_STD = 0.165


@dataclass(frozen=True)
class MediaMetadata:
    """Audio/video stream metadata used by the webcam preflight."""

    path: str
    duration_seconds: float
    video_width: int
    video_height: int
    frame_rate: float
    audio_sample_rate: int
    audio_channels: int

    @property
    def has_mouth_roi_resolution(self) -> bool:
        return (
            self.video_width == MOUTH_ROI_SIZE and self.video_height == MOUTH_ROI_SIZE
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["has_mouth_roi_resolution"] = self.has_mouth_roi_resolution
        return payload


@dataclass(frozen=True)
class PreparedAVInput:
    """Batch-one tensors accepted by the released AV-HuBERT encoder."""

    videos: torch.Tensor
    audios: torch.Tensor
    video_lengths: torch.Tensor
    audio_lengths: torch.Tensor
    metadata: MediaMetadata

    def shape_report(self) -> dict[str, list[int]]:
        return {
            "videos": list(self.videos.shape),
            "audios": list(self.audios.shape),
            "video_lengths": list(self.video_lengths.shape),
            "audio_lengths": list(self.audio_lengths.shape),
        }


def _require_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise MediaInputError(
            f"Required executable '{name}' was not found. Install FFmpeg in the "
            "viavsr environment."
        )
    return executable


def _run_command(command: list[str]) -> bytes:
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except OSError as exc:
        raise MediaInputError(f"Could not run {command[0]}: {exc}") from exc
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise MediaInputError(f"{command[0]} failed: {message}")
    return result.stdout


def _parse_frame_rate(value: str) -> float:
    try:
        rate = float(Fraction(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise MediaInputError(
            f"Invalid video frame rate reported by ffprobe: {value!r}"
        ) from exc
    if rate <= 0:
        raise MediaInputError(f"Video frame rate must be positive, got {rate}.")
    return rate


def _parse_probe_payload(path: Path, payload: dict[str, Any]) -> MediaMetadata:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise MediaInputError("ffprobe did not return a stream list.")
    video = next(
        (stream for stream in streams if stream.get("codec_type") == "video"), None
    )
    audio = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"), None
    )
    if video is None:
        raise MediaInputError(f"Media file has no video stream: {path}")
    if audio is None:
        raise MediaInputError(f"Media file has no audio stream: {path}")

    try:
        duration = float(payload.get("format", {}).get("duration", 0.0))
        width = int(video["width"])
        height = int(video["height"])
        sample_rate = int(audio["sample_rate"])
        channels = int(audio["channels"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaInputError(f"Incomplete stream metadata for {path}: {exc}") from exc
    if duration <= 0:
        raise MediaInputError(f"Media duration must be positive, got {duration}.")

    return MediaMetadata(
        path=str(path),
        duration_seconds=duration,
        video_width=width,
        video_height=height,
        frame_rate=_parse_frame_rate(str(video.get("avg_frame_rate", "0/1"))),
        audio_sample_rate=sample_rate,
        audio_channels=channels,
    )


def probe_av_media(path: Path | str) -> MediaMetadata:
    """Inspect a media file and require both video and audio streams."""
    media_path = Path(path).expanduser().resolve()
    if not media_path.is_file():
        raise MediaInputError(f"Media file does not exist: {media_path}")
    ffprobe = _require_executable("ffprobe")
    output = _run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height,avg_frame_rate,sample_rate,channels:format=duration",
            "-of",
            "json",
            str(media_path),
        ]
    )
    try:
        payload = json.loads(output)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MediaInputError(
            f"Could not parse ffprobe output for {media_path}."
        ) from exc
    if not isinstance(payload, dict):
        raise MediaInputError("ffprobe output must be a JSON object.")
    return _parse_probe_payload(media_path, payload)


def validate_demo_media(
    metadata: MediaMetadata, *, max_duration_seconds: float = 15.0
) -> None:
    """Validate duration and the prepared mouth-ROI inference boundary."""
    if metadata.duration_seconds > max_duration_seconds:
        raise MediaInputError(
            f"Media is {metadata.duration_seconds:.2f}s; inference accepts at most "
            f"{max_duration_seconds:.2f}s per segment."
        )
    if not metadata.has_mouth_roi_resolution:
        raise MediaInputError(
            "Inference requires an affine-aligned 96x96 mouth-ROI video "
            "with embedded audio. This file does not have the required prepared "
            f"resolution ({metadata.video_width}x{metadata.video_height}); face "
            "alignment and mouth tracking belong to the raw-webcam preprocessing stage."
        )


def _decode_video_frames(path: Path) -> np.ndarray:
    ffmpeg = _require_executable("ffmpeg")
    output = _run_command(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-vf",
            f"fps={TARGET_FRAME_RATE},format=gray",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ]
    )
    frame_size = MOUTH_ROI_SIZE * MOUTH_ROI_SIZE
    if not output or len(output) % frame_size != 0:
        raise MediaInputError(
            "FFmpeg returned incomplete 96x96 grayscale video frames."
        )
    return (
        np.frombuffer(output, dtype=np.uint8)
        .copy()
        .reshape(-1, MOUTH_ROI_SIZE, MOUTH_ROI_SIZE)
    )


def _decode_audio_waveform(path: Path) -> np.ndarray:
    ffmpeg = _require_executable("ffmpeg")
    output = _run_command(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    if not output or len(output) % np.dtype(np.float32).itemsize != 0:
        raise MediaInputError("FFmpeg returned an empty or incomplete audio waveform.")
    return np.frombuffer(output, dtype=np.float32).copy()


def preprocess_video_frames(frames: np.ndarray) -> torch.Tensor:
    """Center-crop and normalize grayscale 96x96 mouth frames."""
    if frames.ndim != 3 or frames.shape[1:] != (MOUTH_ROI_SIZE, MOUTH_ROI_SIZE):
        raise MediaInputError(
            "Expected grayscale mouth frames with shape [T, 96, 96], got "
            f"{list(frames.shape)}."
        )
    if len(frames) == 0:
        raise MediaInputError("Video contains no decodable frames.")
    crop_start = (MOUTH_ROI_SIZE - MODEL_VIDEO_SIZE) // 2
    crop_end = crop_start + MODEL_VIDEO_SIZE
    cropped = frames[:, crop_start:crop_end, crop_start:crop_end]
    tensor = torch.from_numpy(np.ascontiguousarray(cropped)).float().div_(255.0)
    tensor = tensor.sub_(VIDEO_MEAN).div_(VIDEO_STD)
    return tensor.unsqueeze(0).unsqueeze(0)


def match_audio_to_video(waveform: np.ndarray, video_frames: int) -> np.ndarray:
    """Pad or trim audio to the upstream 640 samples-per-video-frame ratio."""
    if waveform.ndim != 1:
        raise MediaInputError(
            f"Expected mono waveform [T], got {list(waveform.shape)}."
        )
    target_samples = video_frames * AUDIO_SAMPLES_PER_VIDEO_FRAME
    if len(waveform) < target_samples:
        waveform = np.pad(waveform, (0, target_samples - len(waveform)))
    return np.ascontiguousarray(waveform[:target_samples], dtype=np.float32)


def preprocess_audio_waveform(waveform: np.ndarray, video_frames: int) -> torch.Tensor:
    """Create normalized 104-bin stacked log-filterbank features."""
    matched = match_audio_to_video(waveform, video_frames)
    features = logfbank(
        matched,
        samplerate=TARGET_SAMPLE_RATE,
        nfilt=AUDIO_FILTERBANK_BINS,
    ).astype(np.float32)
    remainder = len(features) % AUDIO_STACK_ORDER
    if remainder:
        features = np.pad(features, ((0, AUDIO_STACK_ORDER - remainder), (0, 0)))
    stacked = features.reshape(-1, AUDIO_FEATURE_DIM)
    if len(stacked) != video_frames:
        raise MediaInputError(
            "Audio/video feature alignment failed: "
            f"{len(stacked)} audio frames for {video_frames} video frames."
        )
    tensor = torch.from_numpy(stacked)
    tensor = F.layer_norm(tensor, tensor.shape[1:])
    return tensor.transpose(0, 1).unsqueeze(0)


def prepare_mouth_roi_media(
    path: Path | str, *, max_duration_seconds: float = 15.0
) -> PreparedAVInput:
    """Convert one prepared mouth-ROI media file into batch-one model tensors."""
    media_path = Path(path).expanduser().resolve()
    metadata = probe_av_media(media_path)
    validate_demo_media(metadata, max_duration_seconds=max_duration_seconds)
    frames = _decode_video_frames(media_path)
    waveform = _decode_audio_waveform(media_path)
    videos = preprocess_video_frames(frames)
    audios = preprocess_audio_waveform(waveform, len(frames))
    video_lengths = torch.tensor([len(frames)], dtype=torch.long)
    audio_lengths = torch.tensor([audios.shape[-1]], dtype=torch.long)
    return PreparedAVInput(
        videos=videos,
        audios=audios,
        video_lengths=video_lengths,
        audio_lengths=audio_lengths,
        metadata=metadata,
    )
