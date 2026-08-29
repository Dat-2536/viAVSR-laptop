from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

PROGRESS_SCHEMA_VERSION = 1
AutomaticInferenceMode = Literal[
    "audio_visual_interval_gated",
    "audio_only_experimental",
]


@dataclass(frozen=True)
class DropoutInterval:
    """One half-open missing-visual interval."""

    start_frame: int
    end_frame_exclusive: int

    @property
    def frame_count(self) -> int:
        return self.end_frame_exclusive - self.start_frame

    def to_dict(self, *, frame_rate: int) -> dict[str, int | float]:
        return {
            "start_frame": self.start_frame,
            "end_frame_exclusive": self.end_frame_exclusive,
            "frame_count": self.frame_count,
            "start_seconds": self.start_frame / frame_rate,
            "end_seconds": self.end_frame_exclusive / frame_rate,
            "duration_seconds": self.frame_count / frame_rate,
        }


@dataclass(frozen=True)
class VisualDropoutMask:
    """Deterministic frame mask and metadata for one synthetic corruption."""

    availability: np.ndarray
    target_dropout_fraction: float
    actual_dropout_fraction: float
    base_seed: int
    effective_seed: int
    intervals: tuple[DropoutInterval, ...]

    def to_dict(self, *, frame_rate: int) -> dict[str, Any]:
        return {
            "target_dropout_fraction": self.target_dropout_fraction,
            "actual_dropout_fraction": self.actual_dropout_fraction,
            "base_seed": self.base_seed,
            "effective_seed": self.effective_seed,
            "frame_count": int(self.availability.size),
            "masked_frames": int((~self.availability).sum()),
            "intervals": [
                interval.to_dict(frame_rate=frame_rate) for interval in self.intervals
            ],
        }


def _stable_seed(*, base_seed: int, sample_index: int, dropout_fraction: float) -> int:
    payload = f"{base_seed}:{sample_index}:{dropout_fraction:.8f}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _allocate(total: int, parts: int, rng: np.random.Generator) -> np.ndarray:
    if total < 0 or parts <= 0:
        raise ValueError("Allocation requires a non-negative total and positive parts.")
    if total == 0:
        return np.zeros(parts, dtype=np.int64)
    probabilities = np.full(parts, 1.0 / parts)
    return rng.multinomial(total, probabilities).astype(np.int64)


def generate_contiguous_visual_dropout(
    *,
    frame_count: int,
    dropout_fraction: float,
    base_seed: int,
    sample_index: int,
    interval_count: int = 2,
) -> VisualDropoutMask:
    """Generate exact-size, separated visual-dropout intervals.

    For ordinary utterances the intervals are kept away from the first and last
    frame. Tiny inputs that cannot satisfy this layout fall back to one interval.
    """

    if frame_count <= 1:
        raise ValueError("frame_count must be greater than one.")
    if not 0.0 < dropout_fraction < 1.0:
        raise ValueError("dropout_fraction must be strictly between zero and one.")
    if interval_count <= 0:
        raise ValueError("interval_count must be greater than zero.")

    masked_frames = min(
        frame_count - 1,
        max(1, round(frame_count * dropout_fraction)),
    )
    unmasked_frames = frame_count - masked_frames
    effective_seed = _stable_seed(
        base_seed=base_seed,
        sample_index=sample_index,
        dropout_fraction=dropout_fraction,
    )
    rng = np.random.default_rng(effective_seed)

    if unmasked_frames >= 2:
        actual_interval_count = min(
            interval_count,
            masked_frames,
            max(1, unmasked_frames - 1),
        )
        interval_lengths = np.ones(actual_interval_count, dtype=np.int64)
        interval_lengths += _allocate(
            masked_frames - actual_interval_count,
            actual_interval_count,
            rng,
        )
        gaps = np.ones(actual_interval_count + 1, dtype=np.int64)
        gaps += _allocate(
            unmasked_frames - (actual_interval_count + 1),
            actual_interval_count + 1,
            rng,
        )
    else:
        actual_interval_count = 1
        interval_lengths = np.asarray([masked_frames], dtype=np.int64)
        gaps = np.asarray([unmasked_frames, 0], dtype=np.int64)
        if bool(rng.integers(0, 2)):
            gaps = gaps[::-1]

    availability = np.ones(frame_count, dtype=np.bool_)
    intervals: list[DropoutInterval] = []
    cursor = int(gaps[0])
    for index, length in enumerate(interval_lengths):
        end = cursor + int(length)
        availability[cursor:end] = False
        intervals.append(
            DropoutInterval(
                start_frame=cursor,
                end_frame_exclusive=end,
            )
        )
        cursor = end + int(gaps[index + 1])

    if cursor != frame_count or int((~availability).sum()) != masked_frames:
        raise RuntimeError("Synthetic visual-dropout construction was inconsistent.")

    return VisualDropoutMask(
        availability=availability,
        target_dropout_fraction=dropout_fraction,
        actual_dropout_fraction=masked_frames / frame_count,
        base_seed=base_seed,
        effective_seed=effective_seed,
        intervals=tuple(intervals),
    )


def select_automatic_inference_mode(
    *,
    visual_coverage: float,
    audio_only_coverage_threshold: float,
) -> AutomaticInferenceMode:
    """Select a mode using input quality only, never transcript accuracy."""

    if not 0.0 <= visual_coverage <= 1.0:
        raise ValueError("visual_coverage must be between zero and one.")
    if not 0.0 <= audio_only_coverage_threshold <= 1.0:
        raise ValueError("audio_only_coverage_threshold must be between zero and one.")
    if visual_coverage <= audio_only_coverage_threshold:
        return "audio_only_experimental"
    return "audio_visual_interval_gated"


def condition_key(
    *,
    sample_index: int,
    condition: str,
    dropout_fraction: float | None = None,
    seed: int | None = None,
) -> str:
    suffix = ""
    if dropout_fraction is not None:
        suffix += f":dropout={dropout_fraction:.6f}"
    if seed is not None:
        suffix += f":seed={seed}"
    return f"sample={sample_index}:condition={condition}{suffix}"


def expected_condition_keys(
    *,
    sample_index: int,
    dropout_fractions: tuple[float, ...],
    seeds: tuple[int, ...],
) -> set[str]:
    keys = {
        condition_key(sample_index=sample_index, condition="clean_av"),
        condition_key(sample_index=sample_index, condition="audio_only"),
    }
    for fraction in dropout_fractions:
        for seed in seeds:
            for condition in (
                "corrupted_av",
                "interval_gated_av",
                "automatic",
            ):
                keys.add(
                    condition_key(
                        sample_index=sample_index,
                        condition=condition,
                        dropout_fraction=fraction,
                        seed=seed,
                    )
                )
    return keys


def protocol_id(protocol: dict[str, Any]) -> str:
    canonical = json.dumps(
        protocol,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class BenchmarkProgressStore:
    """Append-only, single-writer progress store for resumable evaluation."""

    def __init__(self, path: Path | str, *, protocol: dict[str, Any]) -> None:
        self.path = Path(path).expanduser().resolve()
        self.protocol = protocol
        self.protocol_id = protocol_id(protocol)
        self.records: dict[str, dict[str, Any]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._load()
        else:
            self._append_line(
                {
                    "record_type": "protocol",
                    "schema_version": PROGRESS_SCHEMA_VERSION,
                    "protocol_id": self.protocol_id,
                    "protocol": protocol,
                }
            )

    def _load(self) -> None:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError(f"Progress file is empty: {self.path}")
        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Progress header is invalid JSON: {self.path}") from exc
        if (
            header.get("record_type") != "protocol"
            or header.get("schema_version") != PROGRESS_SCHEMA_VERSION
        ):
            raise ValueError("Progress file has an unsupported header.")
        if header.get("protocol_id") != self.protocol_id:
            raise ValueError(
                "Existing progress belongs to a different benchmark protocol."
            )

        for line_number, line in enumerate(lines[1:], start=2):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Progress record on line {line_number} is invalid JSON."
                ) from exc
            if record.get("record_type") != "result":
                raise ValueError(f"Progress line {line_number} is not a result record.")
            if record.get("protocol_id") != self.protocol_id:
                raise ValueError(
                    f"Progress line {line_number} has a mismatched protocol."
                )
            key = record.get("key")
            if not isinstance(key, str) or not key:
                raise ValueError(f"Progress line {line_number} has no result key.")
            if key in self.records:
                raise ValueError(f"Duplicate progress result key: {key}")
            self.records[key] = record

    def _append_line(self, value: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        key = record.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError("Progress result requires a non-empty key.")
        if key in self.records:
            raise ValueError(f"Progress result already exists: {key}")
        stored = {
            **record,
            "record_type": "result",
            "protocol_id": self.protocol_id,
        }
        self._append_line(stored)
        self.records[key] = stored
        return stored
