from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path

import sentencepiece as spm

from .errors import TokenizerAssetError

TOKENIZER_MODEL_SHA256 = "21ca39e799b64044d75edccd9016fac0315e64f89bdd43fbd3089607dceb9d64"
TOKENIZER_UNITS_SHA256 = "ea7b25e67a302305ffdb59909419c08822b3607a6b03871adef2bcb9f6ebec25"
TOKENIZER_REVISION = "ad644a77e8e3177aa7422510302c11de5282fa26"
_WHITESPACE_RE = re.compile(r"\s+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_tokenizer_text(text: str) -> str:
    """Normalize tokenizer input while preserving Vietnamese diacritics."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    return _WHITESPACE_RE.sub(
        " ", unicodedata.normalize("NFC", text).lower()
    ).strip()


class VietnameseSentencePieceTokenizer:
    """Vietnamese SentencePiece-to-ASR-ID mapping released with ViCocktail."""

    def __init__(
        self,
        model_path: Path,
        units_path: Path,
        *,
        expected_model_sha256: str = TOKENIZER_MODEL_SHA256,
        expected_units_sha256: str = TOKENIZER_UNITS_SHA256,
    ) -> None:
        self.model_path = Path(model_path)
        self.units_path = Path(units_path)
        self._reject_english_asset_names()
        self._require_file(self.model_path, "SentencePiece model")
        self._require_file(self.units_path, "token units")
        self.model_sha256 = self._verify_hash(
            self.model_path, expected_model_sha256
        )
        self.units_sha256 = self._verify_hash(
            self.units_path, expected_units_sha256
        )
        self.units = self._load_units()
        self.piece_to_id = {piece: token_id for piece, token_id in self.units}
        self.token_list = ["<blank>"] + [piece for piece, _ in self.units] + ["<eos>"]

        try:
            self.sentencepiece = spm.SentencePieceProcessor(
                model_file=str(self.model_path)
            )
        except Exception as exc:
            raise TokenizerAssetError(
                f"Could not load SentencePiece model {self.model_path}: {exc}",
                stage="tokenizer",
            ) from exc
        self._validate_model_units_pair()

    @property
    def sentencepiece_vocabulary_size(self) -> int:
        return int(self.sentencepiece.get_piece_size())

    @property
    def units_vocabulary_size(self) -> int:
        return len(self.units)

    @property
    def asr_vocabulary_size(self) -> int:
        return len(self.token_list)

    @property
    def unknown_token_id(self) -> int:
        return self.piece_to_id["<unk>"]

    def encode(self, text: str) -> list[int]:
        pieces = self.sentencepiece.encode(
            normalize_tokenizer_text(text), out_type=str
        )
        unknown = self.unknown_token_id
        return [self.piece_to_id.get(piece, unknown) for piece in pieces]

    def decode(self, token_ids: Sequence[int]) -> str:
        pieces: list[str] = []
        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError("token IDs must be integers")
            if token_id == -1:
                continue
            if token_id < 0 or token_id >= len(self.token_list):
                raise ValueError(f"token ID out of range: {token_id}")
            piece = self.token_list[token_id]
            if piece in {"<blank>", "<eos>"}:
                continue
            pieces.append(piece)
        return "".join(pieces).replace("▁", " ").strip()

    def _reject_english_asset_names(self) -> None:
        for path in (self.model_path, self.units_path):
            if "unigram5000" in path.name.lower():
                raise TokenizerAssetError(
                    f"English unigram5000 tokenizer is not allowed: {path}",
                    stage="tokenizer",
                )

    @staticmethod
    def _require_file(path: Path, label: str) -> None:
        if not path.is_file():
            raise TokenizerAssetError(
                f"Missing {label} file: {path}", stage="tokenizer"
            )

    @staticmethod
    def _verify_hash(path: Path, expected: str) -> str:
        actual = sha256_file(path)
        if actual != expected:
            raise TokenizerAssetError(
                f"SHA-256 mismatch for {path}: expected {expected}, got {actual}",
                stage="tokenizer",
            )
        return actual

    def _load_units(self) -> list[tuple[str, int]]:
        units: list[tuple[str, int]] = []
        seen_pieces: set[str] = set()
        seen_ids: set[int] = set()
        try:
            lines = self.units_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise TokenizerAssetError(
                f"Could not read tokenizer units {self.units_path}: {exc}",
                stage="tokenizer",
            ) from exc
        for line_number, line in enumerate(lines, start=1):
            try:
                piece, raw_id = line.rsplit(maxsplit=1)
                token_id = int(raw_id)
            except (ValueError, TypeError) as exc:
                raise TokenizerAssetError(
                    f"Malformed tokenizer unit at line {line_number}: {line!r}",
                    stage="tokenizer",
                ) from exc
            if piece in seen_pieces or token_id in seen_ids:
                raise TokenizerAssetError(
                    f"Duplicate tokenizer piece or ID at line {line_number}: {line!r}",
                    stage="tokenizer",
                )
            seen_pieces.add(piece)
            seen_ids.add(token_id)
            units.append((piece, token_id))
        expected_ids = list(range(1, len(units) + 1))
        if [token_id for _, token_id in units] != expected_ids:
            raise TokenizerAssetError(
                "Tokenizer unit IDs must be ordered and contiguous from 1.",
                stage="tokenizer",
            )
        if not units or units[0] != ("<unk>", 1):
            raise TokenizerAssetError(
                "Tokenizer units must map <unk> to ASR token ID 1.",
                stage="tokenizer",
            )
        return units

    def _validate_model_units_pair(self) -> None:
        sentencepiece_tokens = {
            self.sentencepiece.id_to_piece(index)
            for index in range(self.sentencepiece_vocabulary_size)
        }
        required = sentencepiece_tokens - {"<s>", "</s>"}
        missing = sorted(required - set(self.piece_to_id))
        if missing:
            preview = ", ".join(repr(piece) for piece in missing[:5])
            raise TokenizerAssetError(
                f"Tokenizer model and units are incompatible; missing pieces: {preview}",
                stage="tokenizer",
            )
