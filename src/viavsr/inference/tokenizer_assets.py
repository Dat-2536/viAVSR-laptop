from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .errors import TokenizerAssetError
from .tokenizer import (
    TOKENIZER_MODEL_SHA256,
    TOKENIZER_REVISION,
    TOKENIZER_UNITS_SHA256,
    sha256_file,
)

_RAW_ROOT = (
    "https://raw.githubusercontent.com/nguyenvulebinh/viCocktail/"
    f"{TOKENIZER_REVISION}"
)


@dataclass(frozen=True)
class TokenizerDownload:
    path: Path
    sha256: str
    downloaded: bool


def _download_verified(url: str, destination: Path, expected_hash: str) -> TokenizerDownload:
    if destination.is_file() and sha256_file(destination) == expected_hash:
        return TokenizerDownload(destination, expected_hash, False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != expected_hash:
            raise TokenizerAssetError(
                f"Downloaded tokenizer hash mismatch for {destination.name}: "
                f"expected {expected_hash}, got {actual_hash}",
                stage="tokenizer_download",
            )
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as stream:
            stream.write(payload)
            temporary_path = Path(stream.name)
        os.replace(temporary_path, destination)
        temporary_path = None
        return TokenizerDownload(destination, actual_hash, True)
    except TokenizerAssetError:
        raise
    except Exception as exc:
        raise TokenizerAssetError(
            f"Could not download tokenizer asset {url}: {exc}",
            stage="tokenizer_download",
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def fetch_tokenizer_assets(
    model_path: Path, units_path: Path
) -> tuple[TokenizerDownload, TokenizerDownload]:
    """Download the pinned Vietnamese tokenizer files and verify their hashes."""
    model = _download_verified(
        f"{_RAW_ROOT}/unigram2048.model", model_path, TOKENIZER_MODEL_SHA256
    )
    units = _download_verified(
        f"{_RAW_ROOT}/unigram2048_units.txt", units_path, TOKENIZER_UNITS_SHA256
    )
    return model, units
