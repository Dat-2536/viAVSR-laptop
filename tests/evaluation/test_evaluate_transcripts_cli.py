import json
import sys
from pathlib import Path

import pytest

from scripts.evaluate_transcripts import main


def test_cli_prints_and_writes_evaluation_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    output_path = tmp_path / "metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_transcripts.py",
            "--reference-text",
            "Tôi đang học.",
            "--prediction-text",
            "tôi đang đọc",
            "--output",
            str(output_path),
        ],
    )

    main()

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert printed == written
    assert written["wer"] == pytest.approx(1 / 3)
    assert written["word_substitutions"] == 1


def test_cli_requires_both_transcripts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_transcripts.py", "--reference-text", "xin chào"],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
