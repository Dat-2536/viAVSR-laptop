import hashlib
from pathlib import Path

import pytest

from viavsr.inference.errors import TokenizerAssetError
from viavsr.inference.tokenizer import VietnameseSentencePieceTokenizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPOSITORY_ROOT / "assets/tokenizers/vi/unigram2048.model"
UNITS_PATH = REPOSITORY_ROOT / "assets/tokenizers/vi/unigram2048_units.txt"


@pytest.fixture(scope="module")
def released_tokenizer():
    if not MODEL_PATH.is_file() or not UNITS_PATH.is_file():
        pytest.skip("Run scripts/fetch_tokenizer_assets.py to install tokenizer assets.")
    return VietnameseSentencePieceTokenizer(MODEL_PATH, UNITS_PATH)


def test_released_vocabulary_dimensions(released_tokenizer):
    assert released_tokenizer.sentencepiece_vocabulary_size == 2048
    assert released_tokenizer.units_vocabulary_size == 2055
    assert released_tokenizer.asr_vocabulary_size == 2057


@pytest.mark.parametrize(
    ("text", "decoded"),
    [
        ("hôm nay thời tiết rất đẹp", "hôm nay thời tiết rất đẹp"),
        ("tôi đang học máy học", "tôi đang học máy học"),
        ("xin chào các bạn", "xin chào các bạn"),
        ("xin xin chào", "xin xin chào"),
        ("TÔI Đang Học", "tôi đang học"),
    ],
)
def test_supported_vietnamese_round_trip(released_tokenizer, text: str, decoded: str):
    token_ids = released_tokenizer.encode(text)
    assert token_ids
    assert released_tokenizer.unknown_token_id not in token_ids
    assert released_tokenizer.decode(token_ids) == decoded


def test_numbers_are_reported_through_unknown_token(released_tokenizer):
    token_ids = released_tokenizer.encode("năm 2026")
    assert released_tokenizer.unknown_token_id in token_ids
    assert "<unk>" in released_tokenizer.decode(token_ids)


def test_missing_file_has_clear_error(tmp_path: Path):
    with pytest.raises(TokenizerAssetError, match="Missing SentencePiece model"):
        VietnameseSentencePieceTokenizer(
            tmp_path / "unigram2048.model", tmp_path / "unigram2048_units.txt"
        )


def test_english_tokenizer_name_is_rejected(tmp_path: Path):
    with pytest.raises(TokenizerAssetError, match="unigram5000"):
        VietnameseSentencePieceTokenizer(
            tmp_path / "unigram5000.model", tmp_path / "unigram5000_units.txt"
        )


def test_wrong_checksum_is_rejected(tmp_path: Path):
    model = tmp_path / "unigram2048.model"
    units = tmp_path / "unigram2048_units.txt"
    model.write_bytes(b"not the released model")
    units.write_text("<unk> 1\n", encoding="utf-8")
    with pytest.raises(TokenizerAssetError, match="SHA-256 mismatch"):
        VietnameseSentencePieceTokenizer(model, units)


@pytest.mark.parametrize(
    "contents,match",
    [
        ("<unk> 1\nbroken\n", "Malformed"),
        ("<unk> 1\n<unk> 2\n", "Duplicate"),
        ("<unk> 1\n▁ 3\n", "contiguous"),
    ],
)
def test_malformed_units_are_rejected(
    tmp_path: Path, monkeypatch, contents: str, match: str
):
    model = tmp_path / "unigram2048.model"
    units = tmp_path / "unigram2048_units.txt"
    model.write_bytes(b"placeholder")
    units.write_text(contents, encoding="utf-8")
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    units_hash = hashlib.sha256(units.read_bytes()).hexdigest()

    with pytest.raises(TokenizerAssetError, match=match):
        VietnameseSentencePieceTokenizer(
            model,
            units,
            expected_model_sha256=model_hash,
            expected_units_sha256=units_hash,
        )


def test_mismatched_model_and_units_are_rejected(tmp_path: Path, monkeypatch):
    model = tmp_path / "unigram2048.model"
    units = tmp_path / "unigram2048_units.txt"
    model.write_bytes(b"placeholder")
    units.write_text("<unk> 1\n▁ 2\n", encoding="utf-8")

    class FakeSentencePiece:
        def __init__(self, **kwargs):
            pass

        def get_piece_size(self):
            return 3

        def id_to_piece(self, index):
            return ("<unk>", "▁", "missing-piece")[index]

    monkeypatch.setattr(
        "viavsr.inference.tokenizer.spm.SentencePieceProcessor", FakeSentencePiece
    )
    with pytest.raises(TokenizerAssetError, match="missing pieces"):
        VietnameseSentencePieceTokenizer(
            model,
            units,
            expected_model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            expected_units_sha256=hashlib.sha256(units.read_bytes()).hexdigest(),
        )
