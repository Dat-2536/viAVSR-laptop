from __future__ import annotations

import argparse
import logging
from pathlib import Path

from viavsr.inference import load_model_assets_config, load_vietnamese_avsr_assets
from viavsr.inference.errors import ModelAssetsError
from viavsr.inference.reporting import redact_secrets, write_json_report


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPOSITORY_ROOT / "outputs/model_assets/model_assets_report.json"
DEFAULT_LOG = REPOSITORY_ROOT / "outputs/model_assets/check_model_assets.log"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load and validate Vietnamese AV-HuBERT model assets."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    return parser


def _logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("viavsr.model_assets")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _passed_lines(report) -> list[str]:
    return [
        "Model loaded: PASSED",
        "Tokenizer loaded: PASSED",
        f"SentencePiece vocabulary: {report.vocabulary.sentencepiece_pieces}",
        f"Tokenizer units: {report.vocabulary.units_entries}",
        f"ASR tokenizer vocabulary: {report.vocabulary.asr_tokenizer}",
        f"Model output vocabulary: {report.vocabulary.model_odim}",
        "Vocabulary compatibility: PASSED",
        "Vietnamese round-trip: PASSED",
        f"Device: {report.device}",
    ]


def main() -> int:
    args = build_parser().parse_args()
    logger = _logger(args.log.resolve())
    stage = "config"
    try:
        config = load_model_assets_config(args.config)
        stage = "asset_loading"
        assets = load_vietnamese_avsr_assets(config)
        write_json_report(args.output.resolve(), assets.report.to_dict())
        for line in _passed_lines(assets.report):
            print(line)
            logger.info(line)
        return 0
    except Exception as exc:
        if isinstance(exc, ModelAssetsError):
            stage = exc.stage
        message = redact_secrets(str(exc))
        payload = {
            "status": "failed",
            "stage": stage,
            "error": {
                "type": type(exc).__name__,
                "message": message,
            },
        }
        write_json_report(args.output.resolve(), payload)
        line = f"Asset validation: FAILED ({stage}): {message}"
        print(line)
        logger.error(line)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
