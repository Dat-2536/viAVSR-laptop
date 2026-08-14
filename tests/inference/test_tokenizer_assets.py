import hashlib
from pathlib import Path

import pytest

from viavsr.inference.errors import TokenizerAssetError
from viavsr.inference.tokenizer_assets import _download_verified


class FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self.payload


def test_download_is_verified_atomic_and_idempotent(tmp_path: Path, monkeypatch):
    payload = b"verified tokenizer bytes"
    expected = hashlib.sha256(payload).hexdigest()
    calls = 0

    def urlopen(url, timeout):
        nonlocal calls
        calls += 1
        return FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    destination = tmp_path / "assets/unigram2048.model"

    first = _download_verified("https://example.test/model", destination, expected)
    second = _download_verified("https://example.test/model", destination, expected)

    assert first.downloaded is True
    assert second.downloaded is False
    assert destination.read_bytes() == payload
    assert calls == 1
    assert not list(destination.parent.glob(".unigram2048.model.*"))


def test_bad_download_does_not_replace_existing_file(tmp_path: Path, monkeypatch):
    destination = tmp_path / "unigram2048.model"
    destination.write_bytes(b"old corrupt file")
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: FakeResponse(b"bad download")
    )

    with pytest.raises(TokenizerAssetError, match="hash mismatch"):
        _download_verified("https://example.test/model", destination, "0" * 64)

    assert destination.read_bytes() == b"old corrupt file"
