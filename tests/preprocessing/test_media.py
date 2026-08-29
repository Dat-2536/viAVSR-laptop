from pathlib import Path

import numpy as np
import pytest
import torch

from viavsr.preprocessing import media
from viavsr.preprocessing.errors import MediaInputError
from viavsr.preprocessing.media import (
    AUDIO_FEATURE_DIM,
    AUDIO_SAMPLES_PER_VIDEO_FRAME,
    MODEL_VIDEO_SIZE,
    VIDEO_MEAN,
    VIDEO_STD,
    MediaMetadata,
    _parse_probe_payload,
    match_audio_to_video,
    prepare_audio_only_media,
    preprocess_audio_waveform,
    preprocess_video_frames,
    validate_demo_media,
)


def _metadata(**overrides: object) -> MediaMetadata:
    values: dict[str, object] = {
        "path": "/tmp/sample.mp4",
        "duration_seconds": 4.0,
        "video_width": 96,
        "video_height": 96,
        "frame_rate": 25.0,
        "audio_sample_rate": 16_000,
        "audio_channels": 1,
    }
    values.update(overrides)
    return MediaMetadata(**values)  # type: ignore[arg-type]


def test_parse_probe_payload_requires_audio_and_video() -> None:
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "width": 96,
                "height": 96,
                "avg_frame_rate": "25/1",
            },
            {
                "codec_type": "audio",
                "sample_rate": "48000",
                "channels": 2,
            },
        ],
        "format": {"duration": "3.5"},
    }

    metadata = _parse_probe_payload(Path("sample.webm"), payload)

    assert metadata.duration_seconds == 3.5
    assert metadata.frame_rate == 25.0
    assert metadata.audio_sample_rate == 48_000
    assert metadata.audio_channels == 2


def test_parse_probe_payload_rejects_missing_audio() -> None:
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "width": 96,
                "height": 96,
                "avg_frame_rate": "25/1",
            }
        ],
        "format": {"duration": "3.5"},
    }

    with pytest.raises(MediaInputError, match="no audio stream"):
        _parse_probe_payload(Path("silent.mp4"), payload)


def test_validate_demo_media_rejects_raw_webcam_resolution() -> None:
    metadata = _metadata(video_width=1280, video_height=720)

    with pytest.raises(MediaInputError, match="face alignment and mouth tracking"):
        validate_demo_media(metadata)


def test_validate_demo_media_rejects_long_segment() -> None:
    metadata = _metadata(duration_seconds=16.0)

    with pytest.raises(MediaInputError, match="accepts at most 15.00s"):
        validate_demo_media(metadata)


def test_preprocess_video_frames_matches_upstream_shape_and_normalization() -> None:
    frames = np.stack(
        [
            np.zeros((96, 96), dtype=np.uint8),
            np.full((96, 96), 255, dtype=np.uint8),
        ]
    )

    result = preprocess_video_frames(frames)

    assert result.shape == (1, 1, 2, MODEL_VIDEO_SIZE, MODEL_VIDEO_SIZE)
    assert result[0, 0, 0, 0, 0].item() == pytest.approx(-VIDEO_MEAN / VIDEO_STD)
    assert result[0, 0, 1, 0, 0].item() == pytest.approx((1.0 - VIDEO_MEAN) / VIDEO_STD)


def test_match_audio_to_video_pads_and_trims() -> None:
    short = np.ones(100, dtype=np.float32)
    long = np.ones(AUDIO_SAMPLES_PER_VIDEO_FRAME * 3, dtype=np.float32)

    padded = match_audio_to_video(short, video_frames=2)
    trimmed = match_audio_to_video(long, video_frames=2)

    assert padded.shape == (AUDIO_SAMPLES_PER_VIDEO_FRAME * 2,)
    assert np.all(padded[:100] == 1.0)
    assert np.all(padded[100:] == 0.0)
    assert trimmed.shape == (AUDIO_SAMPLES_PER_VIDEO_FRAME * 2,)


def test_preprocess_audio_waveform_matches_verified_feature_contract() -> None:
    video_frames = 20
    sample_count = video_frames * AUDIO_SAMPLES_PER_VIDEO_FRAME
    time = np.arange(sample_count, dtype=np.float32) / 16_000
    waveform = np.sin(2 * np.pi * 440 * time).astype(np.float32)

    result = preprocess_audio_waveform(waveform, video_frames)

    assert result.shape == (1, AUDIO_FEATURE_DIM, video_frames)
    assert torch.isfinite(result).all()
    assert result[0, :, 0].mean().item() == pytest.approx(0.0, abs=1e-5)


def test_preprocess_video_frames_rejects_unprepared_full_frames() -> None:
    frames = np.zeros((2, 720, 1280), dtype=np.uint8)

    with pytest.raises(MediaInputError, match=r"\[T, 96, 96\]"):
        preprocess_video_frames(frames)


def test_prepare_audio_only_media_accepts_raw_webcam_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _metadata(
        duration_seconds=2.0,
        video_width=1280,
        video_height=720,
    )
    waveform = np.zeros(2 * 16_000, dtype=np.float32)
    monkeypatch.setattr(media, "probe_av_media", lambda path: metadata)
    monkeypatch.setattr(media, "_decode_audio_waveform", lambda path: waveform)

    prepared = prepare_audio_only_media("raw-webcam.mp4")

    assert prepared.audios.shape == (1, AUDIO_FEATURE_DIM, 50)
    assert prepared.videos.shape == (1, 1, 50, MODEL_VIDEO_SIZE, MODEL_VIDEO_SIZE)
    assert prepared.video_lengths.tolist() == [50]
    assert prepared.audio_lengths.tolist() == [50]
    assert torch.count_nonzero(prepared.videos).item() == 0
