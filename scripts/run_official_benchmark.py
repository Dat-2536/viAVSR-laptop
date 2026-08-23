from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from viavsr.evaluation import evaluate_transcript
from viavsr.inference import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_CTC_WEIGHT,
    load_model_assets_config,
    load_vietnamese_avsr_assets,
    recognize_prepared_av,
)
from viavsr.inference.errors import InferenceError, ModelAssetsError
from viavsr.inference.reporting import redact_secrets, write_json_report
from viavsr.preprocessing import MediaInputError, prepare_mouth_roi_media

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs/official_benchmark"
DEFAULT_DATASET_REPOSITORY = "nguyenvulebinh/ViCocktail"
DEFAULT_DATASET_REVISION = "6a42aa56b095281ee2511a7094897c0739919bdc"
DATASETS_SERVER = "https://datasets-server.huggingface.co"
HUGGING_FACE_API = "https://huggingface.co/api/datasets"
DATA_REVISION_RE = re.compile(r"@([0-9a-f]{40})/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run joint CTC/attention decoding on an official ViCocktail "
            "clean-test smoke subset."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--dataset-repository", default=DEFAULT_DATASET_REPOSITORY)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--split", default="test")
    parser.add_argument("--offset", default=0, type=int)
    parser.add_argument("--count", default=10, type=int)
    parser.add_argument("--max-duration", default=30.0, type=float)
    parser.add_argument("--beam-size", default=DEFAULT_BEAM_SIZE, type=int)
    parser.add_argument("--ctc-weight", default=DEFAULT_CTC_WEIGHT, type=float)
    return parser


def _logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("viavsr.official_benchmark")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.FileHandler(path, encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _fetch_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "viavsr-official-benchmark/1.0"})
    try:
        with urlopen(request, timeout=180) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not fetch official dataset data: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError("Official dataset API returned a non-object response.")
    return payload


def _decode_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Official dataset field {field!r} is not a string.")
    try:
        return base64.b64decode(value, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not decode official dataset field {field!r}.") from exc


def _download_samples(
    *,
    repository: str,
    config: str,
    split: str,
    offset: int,
    count: int,
    media_dir: Path,
) -> list[dict[str, Any]]:
    query = urlencode(
        {
            "dataset": repository,
            "config": config,
            "split": split,
            "offset": offset,
            "length": count,
        }
    )
    rows = _fetch_json(f"{DATASETS_SERVER}/rows?{query}").get("rows")
    if not isinstance(rows, list) or len(rows) != count:
        raise RuntimeError(
            f"Expected {count} official rows, received {len(rows or [])}."
        )
    media_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    for wrapper in rows:
        if not isinstance(wrapper, dict) or not isinstance(wrapper.get("row"), dict):
            raise TypeError("Official dataset row is malformed.")
        row_index = wrapper.get("row_idx")
        row = wrapper["row"]
        if not isinstance(row_index, int):
            raise TypeError("Official dataset row index is malformed.")
        video_value = row.get("video")
        source_url = row.get("__url__")
        if not isinstance(video_value, str) or not isinstance(source_url, str):
            raise TypeError("Official dataset video provenance is malformed.")
        try:
            video = base64.b64decode(video_value, validate=True)
            expected_frames = int(_decode_text(row.get("length"), "length"))
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Could not decode official audiovisual payload.") from exc
        media_path = media_dir / f"{row_index:010d}.mp4"
        media_path.write_bytes(video)
        revision_match = DATA_REVISION_RE.search(source_url)
        samples.append(
            {
                "row_index": row_index,
                "key": row.get("__key__"),
                "sample_id": _decode_text(row.get("sample_id"), "sample_id"),
                "reference": _decode_text(row.get("label"), "label"),
                "expected_frames": expected_frames,
                "source_url": source_url,
                "data_revision": (
                    revision_match.group(1) if revision_match is not None else None
                ),
                "media_path": media_path,
                "media_sha256": hashlib.sha256(video).hexdigest(),
            }
        )
    return samples


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, int | float]:
    evaluations = [sample["evaluation"] for sample in samples]
    reference_words = sum(item["reference_words"] for item in evaluations)
    reference_characters = sum(item["reference_characters"] for item in evaluations)
    word_errors = sum(
        item["word_substitutions"] + item["word_deletions"] + item["word_insertions"]
        for item in evaluations
    )
    character_errors = sum(
        item["char_substitutions"] + item["char_deletions"] + item["char_insertions"]
        for item in evaluations
    )
    duration = sum(sample["media"]["duration_seconds"] for sample in samples)
    inference_seconds = sum(sample["result"]["inference_seconds"] for sample in samples)
    count = len(samples)
    return {
        "successful_samples": count,
        "total_duration_seconds": duration,
        "reference_words": reference_words,
        "word_errors": word_errors,
        "corpus_wer": word_errors / reference_words if reference_words else 0.0,
        "macro_wer": sum(item["wer"] for item in evaluations) / count,
        "reference_characters": reference_characters,
        "character_errors": character_errors,
        "corpus_cer": (
            character_errors / reference_characters if reference_characters else 0.0
        ),
        "macro_cer": sum(item["cer"] for item in evaluations) / count,
        "total_preprocessing_seconds": sum(
            sample["preprocessing_seconds"] for sample in samples
        ),
        "total_inference_seconds": inference_seconds,
        "inference_real_time_factor": inference_seconds / duration if duration else 0.0,
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.count <= 0 or args.offset < 0:
        raise SystemExit("--count must be positive and --offset must be non-negative.")

    output_dir = args.output_dir.expanduser().resolve()
    media_dir = output_dir / "media"
    predictions_dir = output_dir / "predictions"
    report_path = output_dir / "benchmark_report.json"
    log_path = output_dir / "execution.log"
    logger = _logger(log_path)
    started_at = datetime.now(UTC)
    wall_started = time.perf_counter()

    try:
        metadata = _fetch_json(
            f"{HUGGING_FACE_API}/{quote(args.dataset_repository, safe='/')}"
        )
        actual_revision = metadata.get("sha")
        if actual_revision != args.dataset_revision:
            raise RuntimeError(
                "Official dataset revision mismatch: expected "
                f"{args.dataset_revision}, got {actual_revision}."
            )
        logger.info(
            "Dataset verified: %s@%s split=%s offset=%d count=%d",
            args.dataset_repository,
            actual_revision,
            args.split,
            args.offset,
            args.count,
        )
        samples = _download_samples(
            repository=args.dataset_repository,
            config=args.dataset_config,
            split=args.split,
            offset=args.offset,
            count=args.count,
            media_dir=media_dir,
        )
        for sample in samples:
            logger.info(
                "Downloaded row=%d sample_id=%s frames=%d sha256=%s",
                sample["row_index"],
                sample["sample_id"],
                sample["expected_frames"],
                sample["media_sha256"],
            )

        asset_started = time.perf_counter()
        assets = load_vietnamese_avsr_assets(load_model_assets_config(args.config))
        asset_loading_seconds = time.perf_counter() - asset_started
        logger.info(
            "Assets loaded: model=%s@%s tokenizer=%s@%s device=%s",
            assets.report.repository_id,
            assets.report.model_revision,
            assets.report.tokenizer_repository,
            assets.report.tokenizer_revision,
            assets.report.device,
        )

        passed: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for sample in samples:
            row_index = sample["row_index"]
            logger.info("Running joint decoder for row=%d", row_index)
            try:
                preprocessing_started = time.perf_counter()
                prepared = prepare_mouth_roi_media(
                    sample["media_path"],
                    max_duration_seconds=args.max_duration,
                )
                preprocessing_seconds = time.perf_counter() - preprocessing_started
                inference = recognize_prepared_av(
                    assets,
                    prepared,
                    decoder="joint_beam_search",
                    beam_size=args.beam_size,
                    ctc_weight=args.ctc_weight,
                )
                evaluation = evaluate_transcript(
                    reference=sample["reference"],
                    prediction=inference.transcript,
                )
                dataset_sample = {
                    key: value for key, value in sample.items() if key != "media_path"
                }
                payload = {
                    "status": "passed",
                    "dataset_sample": dataset_sample,
                    "media": prepared.metadata.to_dict(),
                    "input_shapes": prepared.shape_report(),
                    "preprocessing_seconds": preprocessing_seconds,
                    "result": inference.to_dict(),
                    "evaluation": evaluation.to_dict(),
                }
                passed.append(payload)
                write_json_report(predictions_dir / f"{row_index:010d}.json", payload)
                logger.info(
                    "PASSED row=%d WER=%.6f CER=%.6f transcript=%s",
                    row_index,
                    evaluation.wer,
                    evaluation.cer,
                    inference.transcript,
                )
            except (
                MediaInputError,
                InferenceError,
                OSError,
                RuntimeError,
                ValueError,
            ) as exc:
                failure = {
                    "status": "failed",
                    "row_index": row_index,
                    "sample_id": sample["sample_id"],
                    "error": {
                        "type": type(exc).__name__,
                        "message": redact_secrets(str(exc)),
                    },
                }
                failures.append(failure)
                write_json_report(predictions_dir / f"{row_index:010d}.json", failure)
                logger.error(
                    "FAILED row=%d error=%s",
                    row_index,
                    failure["error"]["message"],
                )

        aggregate = _aggregate(passed) if passed else {}
        status = "passed" if not failures else "partial_failure"
        report = {
            "status": status,
            "scope": (
                "Deterministic official clean-test prefix smoke subset; not a "
                "full-dataset or statistically representative benchmark."
            ),
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "wall_seconds": time.perf_counter() - wall_started,
            "dataset": {
                "repository_id": args.dataset_repository,
                "repository_revision": actual_revision,
                "data_revisions": sorted(
                    {
                        sample["data_revision"]
                        for sample in samples
                        if sample["data_revision"] is not None
                    }
                ),
                "config": args.dataset_config,
                "split": args.split,
                "offset": args.offset,
                "requested_samples": args.count,
            },
            "decoder": {
                "name": "joint_beam_search",
                "beam_size": args.beam_size,
                "ctc_weight": args.ctc_weight,
                "attention_weight": 1.0 - args.ctc_weight,
                "language_model_weight": 0.0,
                "length_bonus_weight": 0.0,
            },
            "asset_loading_seconds": asset_loading_seconds,
            "model_and_tokenizer": assets.report.to_dict(),
            "aggregate": aggregate,
            "samples": passed,
            "failures": failures,
            "artifacts": {
                "report": str(report_path),
                "execution_log": str(log_path),
                "predictions_directory": str(predictions_dir),
                "media_directory": str(media_dir),
            },
        }
        write_json_report(report_path, report)
        logger.info(
            "Benchmark complete: status=%s passed=%d failed=%d corpus_WER=%s corpus_CER=%s",
            status,
            len(passed),
            len(failures),
            aggregate.get("corpus_wer"),
            aggregate.get("corpus_cer"),
        )
        print(
            json.dumps(
                {
                    "status": status,
                    "report": str(report_path),
                    "execution_log": str(log_path),
                    "aggregate": aggregate,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if not failures else 1
    except (ModelAssetsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        message = redact_secrets(str(exc))
        logger.error("Benchmark aborted: %s", message)
        write_json_report(
            report_path,
            {
                "status": "failed",
                "error": {"type": type(exc).__name__, "message": message},
                "artifacts": {"execution_log": str(log_path)},
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
