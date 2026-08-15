from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

_BEARER_RE = re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]+")


def redact_secrets(message: str) -> str:
    """Remove known Hugging Face credentials from error text."""
    redacted = message
    for variable in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        secret = os.environ.get(variable)
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return _BEARER_RE.sub("Bearer [REDACTED]", redacted)


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    """Write a UTF-8 JSON report atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            temporary_path = Path(stream.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
