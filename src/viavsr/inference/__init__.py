"""Model loading and validation for Vietnamese AVSR assets."""

from .config import ModelAssetsConfig, load_model_assets_config
from .decoding import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_CTC_WEIGHT,
    JointBeamSearchHypothesis,
    decode_joint_ctc_attention,
)
from .model_assets import load_vietnamese_avsr_assets
from .recognition import recognize_prepared_av
from .schemas import InferenceResult, LoadedAVSRAssets, ModelAssetsReport
from .tokenizer import VietnameseSentencePieceTokenizer

__all__ = [
    "DEFAULT_BEAM_SIZE",
    "DEFAULT_CTC_WEIGHT",
    "InferenceResult",
    "JointBeamSearchHypothesis",
    "LoadedAVSRAssets",
    "ModelAssetsConfig",
    "ModelAssetsReport",
    "VietnameseSentencePieceTokenizer",
    "decode_joint_ctc_attention",
    "load_model_assets_config",
    "load_vietnamese_avsr_assets",
    "recognize_prepared_av",
]
