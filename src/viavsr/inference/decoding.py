from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .errors import InferenceError
from .vendor.avsrcocktail.nets.batch_beam_search import BatchBeamSearch
from .vendor.avsrcocktail.nets.scorers.ctc import CTCPrefixScorer
from .vendor.avsrcocktail.nets.scorers.length_bonus import LengthBonus

DEFAULT_BEAM_SIZE = 3
DEFAULT_CTC_WEIGHT = 0.1


@dataclass(frozen=True)
class JointBeamSearchHypothesis:
    """Best token sequence and score returned by joint decoding."""

    token_ids: list[int]
    score: float


def validate_joint_beam_search_parameters(*, beam_size: int, ctc_weight: float) -> None:
    """Validate the two configurable upstream beam-search parameters."""
    if beam_size <= 0:
        raise InferenceError(
            f"beam_size must be positive, got {beam_size}.",
            stage="decoding",
        )
    if not 0.0 <= ctc_weight <= 1.0:
        raise InferenceError(
            f"ctc_weight must be between 0 and 1, got {ctc_weight}.",
            stage="decoding",
        )


def build_joint_ctc_attention_beam_search(
    model: Any,
    token_list: list[str],
    *,
    beam_size: int = DEFAULT_BEAM_SIZE,
    ctc_weight: float = DEFAULT_CTC_WEIGHT,
) -> BatchBeamSearch:
    """Build the pinned AVSRCocktail joint decoder with released defaults."""
    validate_joint_beam_search_parameters(
        beam_size=beam_size,
        ctc_weight=ctc_weight,
    )
    try:
        vocabulary_size = len(token_list)
        scorers = {
            "decoder": model.decoder,
            "ctc": CTCPrefixScorer(model.ctc, model.eos),
            "length_bonus": LengthBonus(vocabulary_size),
            "lm": None,
        }
        weights = {
            "decoder": 1.0 - ctc_weight,
            "ctc": ctc_weight,
            "length_bonus": 0.0,
            "lm": 0.0,
        }
        search = BatchBeamSearch(
            beam_size=beam_size,
            vocab_size=vocabulary_size,
            weights=weights,
            scorers=scorers,
            sos=model.sos,
            eos=model.eos,
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
        )
    except (AssertionError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise InferenceError(
            f"Could not construct joint CTC/attention beam search: {exc}",
            stage="decoding",
        ) from exc
    search.eval()
    return search


def decode_joint_ctc_attention(
    model: Any,
    encoder_features: torch.Tensor,
    token_list: list[str],
    *,
    beam_size: int = DEFAULT_BEAM_SIZE,
    ctc_weight: float = DEFAULT_CTC_WEIGHT,
) -> JointBeamSearchHypothesis:
    """Decode one encoder sequence using upstream joint CTC/attention search."""
    if encoder_features.ndim != 2:
        raise InferenceError(
            "Joint beam search expects encoder features shaped [T, D], "
            f"got {list(encoder_features.shape)}.",
            stage="decoding",
        )
    search = build_joint_ctc_attention_beam_search(
        model,
        token_list,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
    )
    try:
        hypotheses = search(encoder_features)
    except (IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise InferenceError(
            f"Joint CTC/attention beam search failed: {exc}",
            stage="decoding",
        ) from exc
    if not hypotheses:
        raise InferenceError(
            "Joint CTC/attention beam search returned no completed hypothesis.",
            stage="decoding",
        )

    best = hypotheses[0]
    sequence = [int(token_id) for token_id in best.yseq.tolist()]
    if len(sequence) < 2 or sequence[0] != int(model.sos):
        raise InferenceError(
            "Joint beam-search hypothesis is missing its start token.",
            stage="decoding",
        )
    if sequence[-1] != int(model.eos):
        raise InferenceError(
            "Joint beam-search hypothesis is missing its end token.",
            stage="decoding",
        )
    return JointBeamSearchHypothesis(
        token_ids=sequence[1:-1],
        score=float(best.score),
    )
