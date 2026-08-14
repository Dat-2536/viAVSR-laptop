import json
from pathlib import Path

from viavsr.inference.reporting import redact_secrets, write_json_report
from viavsr.inference.schemas import ModelAssetsReport, VocabularyDimensions


def test_report_preserves_vietnamese_and_is_atomic(tmp_path: Path):
    path = tmp_path / "nested/report.json"
    write_json_report(path, {"status": "passed", "text": "xin chào"})
    assert json.loads(path.read_text(encoding="utf-8"))["text"] == "xin chào"
    assert not list(path.parent.glob(".report.json.*"))


def test_hugging_face_token_is_redacted(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    message = redact_secrets("request failed with hf_secret_value and Bearer abc.def-1")
    assert "hf_secret_value" not in message
    assert "abc.def-1" not in message
    assert message.count("[REDACTED]") == 2


def test_success_report_has_stable_nested_vocabulary_schema():
    report = ModelAssetsReport(
        status="passed",
        repository_id="owner/model",
        model_revision="model-revision",
        model_implementation_revision="code-revision",
        tokenizer_repository="owner/tokenizer",
        tokenizer_revision="tokenizer-revision",
        tokenizer_model_sha256="model-hash",
        tokenizer_units_sha256="units-hash",
        model_class="package.Model",
        device="cpu",
        dtype="float32",
        eval_mode=True,
        parameter_count=123,
        vocabulary=VocabularyDimensions(
            sentencepiece_pieces=2048,
            units_entries=2055,
            asr_tokenizer=2057,
            config_odim=2057,
            model_odim=2057,
            ctc_output=2057,
            decoder_embedding=2057,
            decoder_output=2057,
        ),
        vocabulary_compatible=True,
    )

    payload = report.to_dict()
    assert payload["status"] == "passed"
    assert payload["vocabulary"]["sentencepiece_pieces"] == 2048
    assert payload["vocabulary"]["asr_tokenizer"] == 2057
