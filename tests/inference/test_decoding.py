from types import SimpleNamespace

import pytest
import torch
from torch import nn

from viavsr.inference import decoding
from viavsr.inference.decoding import (
    build_joint_ctc_attention_beam_search,
    decode_joint_ctc_attention,
    validate_joint_beam_search_parameters,
)
from viavsr.inference.errors import InferenceError
from viavsr.inference.vendor.avsrcocktail.nets.scorer_interface import (
    BatchScorerInterface,
)


class _FakeDecoder(BatchScorerInterface, nn.Module):
    def __init__(self) -> None:
        nn.Module.__init__(self)

    def batch_score(self, ys, states, xs):
        del states, xs
        return torch.zeros((len(ys), 5)), [None] * len(ys)


class _FakeCTC(nn.Module):
    pass


def _fake_model() -> SimpleNamespace:
    return SimpleNamespace(
        decoder=_FakeDecoder(),
        ctc=_FakeCTC(),
        sos=4,
        eos=4,
    )


def test_build_joint_decoder_matches_released_default_weights() -> None:
    search = build_joint_ctc_attention_beam_search(
        _fake_model(),
        ["<blank>", "a", "b", "c", "<eos>"],
    )

    assert search.beam_size == 3
    assert search.weights["decoder"] == pytest.approx(0.9)
    assert search.weights["ctc"] == pytest.approx(0.1)
    assert search.weights["length_bonus"] == 0.0
    assert search.weights["lm"] == 0.0
    assert search.pre_beam_score_key == "decoder"


@pytest.mark.parametrize(
    ("beam_size", "ctc_weight", "message"),
    [
        (0, 0.1, "beam_size must be positive"),
        (3, -0.1, "ctc_weight must be between"),
        (3, 1.1, "ctc_weight must be between"),
    ],
)
def test_validate_joint_decoder_parameters(
    beam_size: int,
    ctc_weight: float,
    message: str,
) -> None:
    with pytest.raises(InferenceError, match=message) as caught:
        validate_joint_beam_search_parameters(
            beam_size=beam_size,
            ctc_weight=ctc_weight,
        )

    assert caught.value.stage == "decoding"


def test_decode_joint_decoder_strips_sos_and_eos(monkeypatch) -> None:
    class FakeSearch:
        def __call__(self, features):
            assert features.shape == (6, 4)
            return [
                SimpleNamespace(
                    yseq=torch.tensor([4, 2, 3, 4]),
                    score=torch.tensor(-2.5),
                )
            ]

    monkeypatch.setattr(
        decoding,
        "build_joint_ctc_attention_beam_search",
        lambda *args, **kwargs: FakeSearch(),
    )

    result = decode_joint_ctc_attention(
        _fake_model(),
        torch.zeros((6, 4)),
        ["<blank>", "a", "b", "c", "<eos>"],
    )

    assert result.token_ids == [2, 3]
    assert result.score == pytest.approx(-2.5)


def test_decode_joint_decoder_rejects_invalid_feature_shape() -> None:
    with pytest.raises(InferenceError, match=r"\[T, D\]"):
        decode_joint_ctc_attention(
            _fake_model(),
            torch.zeros((1, 6, 4)),
            ["<blank>", "a", "b", "c", "<eos>"],
        )
