import unicodedata
import pytest

from viasvr.evaluation import normalize_vietnamese_text


def test_normalization_lowercases_and_collapses_whitespace():
    assert (
        normalize_vietnamese_text("  Hôm Nay,   Trời RẤT đẹp! ")
        == "hôm nay trời rất đẹp"
    )


def test_normalization_preserves_diacritics():
    assert normalize_vietnamese_text("Tôi đang học") == "tôi đang học"


def test_normalization_normalizes_unicode_to_nfc():
    decomposed = unicodedata.normalize("NFD", "tôi")
    normalized = normalize_vietnamese_text(decomposed)

    assert normalized == "tôi"
    assert unicodedata.is_normalized("NFC", normalized)


def test_normalization_rejects_non_string():
    with pytest.raises(TypeError):
        normalize_vietnamese_text(123)  # type: ignore[arg-type]
