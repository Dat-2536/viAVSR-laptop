from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class RoundTripCase:
    input_text: str
    normalized_text: str
    token_ids: list[int]
    decoded_text: str
    status: Literal["passed", "failed", "unsupported"]
    contains_unknown: bool


@dataclass(frozen=True)
class VocabularyDimensions:
    sentencepiece_pieces: int
    units_entries: int
    asr_tokenizer: int
    config_odim: int
    model_odim: int
    ctc_output: int
    decoder_embedding: int
    decoder_output: int


@dataclass(frozen=True)
class ModelAssetsReport:
    status: Literal["passed", "failed"]
    repository_id: str
    model_revision: str
    model_implementation_revision: str
    tokenizer_repository: str
    tokenizer_revision: str
    tokenizer_model_sha256: str
    tokenizer_units_sha256: str
    model_class: str
    device: str
    dtype: str
    eval_mode: bool
    parameter_count: int
    vocabulary: VocabularyDimensions
    vocabulary_compatible: bool
    round_trip_cases: list[RoundTripCase] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoadedAVSRAssets:
    model: Any
    tokenizer: Any
    report: ModelAssetsReport


@dataclass(frozen=True)
class InferenceResult:
    transcript: str
    token_ids: list[int]
    decoder: Literal["ctc_greedy", "joint_beam_search"]
    input_video_frames: int
    encoder_frames: int
    inference_seconds: float
    device: str
    dtype: str
    beam_size: int | None = None
    ctc_weight: float | None = None
    hypothesis_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}
