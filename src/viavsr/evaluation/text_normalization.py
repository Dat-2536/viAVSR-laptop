import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_TO_SPACE_RE = re.compile(r"""[.,!?;:"'()\[\]{}…“”‘’\-–—/\\]+""")


def normalize_vietnamese_text(text: str) -> str:
    """Normalize Vietnamese text before WER/CER evaluation.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.lower()
    normalized = _PUNCTUATION_TO_SPACE_RE.sub(" ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized