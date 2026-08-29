from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import torch
from run_official_benchmark import (
    DATASETS_SERVER,
    DEFAULT_DATASET_REPOSITORY,
    DEFAULT_DATASET_REVISION,
    HUGGING_FACE_API,
    _download_samples,
    _fetch_json,
)

from viavsr.evaluation import evaluate_transcript
from viavsr.evaluation.robustness import (
    BenchmarkProgressStore,
    condition_key,
    expected_condition_keys,
    generate_contiguous_visual_dropout,
    select_automatic_inference_mode,
)
from viavsr.inference import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_CTC_WEIGHT,
    load_model_assets_config,
    load_vietnamese_avsr_assets,
    recognize_prepared_av,
)
from viavsr.inference.errors import InferenceError, ModelAssetsError
from viavsr.inference.reporting import redact_secrets, write_json_report
from viavsr.preprocessing import (
    MediaInputError,
    PreparedAVInput,
    prepare_mouth_roi_media,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs/visual_dropout_benchmark"
DEFAULT_DROPOUT_LEVELS = (0.1, 0.3, 0.5)
DEFAULT_SEEDS = (17, 29, 43)


@dataclass(frozen=True)
class ArtifactPaths:
    output: Path
    work: Path
    media: Path
    report: Path
    log: Path
    results: Path

    @classmethod
    def build(cls, output: Path) -> ArtifactPaths:
        output = output.expanduser().resolve()
        work = output / ".work"
        return cls(
            output=output,
            work=work,
            media=work / "media",
            report=output / "benchmark_report.json",
            log=output / "execution.log",
            results=output / "results.jsonl",
        )

    def clean_work(self) -> None:
        work = self.work.resolve()
        if work.parent != self.output.resolve():
            raise ValueError("Benchmark work directory escaped its output directory.")
        shutil.rmtree(work, ignore_errors=True)

    def prepare(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        self.clean_work()
        self.media.mkdir(parents=True)

    def artifacts(self) -> dict[str, str]:
        return {
            "report": str(self.report),
            "execution_log": str(self.log),
            "results": str(self.results),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a resumable paired ViCocktail visual-dropout benchmark."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--dataset-repository", default=DEFAULT_DATASET_REPOSITORY)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--split", default="test")
    parser.add_argument("--offset", default=0, type=int)
    parser.add_argument(
        "--count", default=0, type=int, help="0 means all remaining rows."
    )
    parser.add_argument("--page-size", default=10, type=int)
    parser.add_argument("--max-duration", default=30.0, type=float)
    parser.add_argument("--beam-size", default=DEFAULT_BEAM_SIZE, type=int)
    parser.add_argument("--ctc-weight", default=DEFAULT_CTC_WEIGHT, type=float)
    parser.add_argument(
        "--dropout-levels", nargs="+", type=float, default=list(DEFAULT_DROPOUT_LEVELS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--dropout-intervals", default=2, type=int)
    parser.add_argument("--audio-only-coverage-threshold", default=0.5, type=float)
    return parser


def _validate(args: argparse.Namespace) -> None:
    if args.offset < 0 or args.count < 0:
        raise ValueError("--offset and --count must be non-negative.")
    if args.page_size <= 0 or args.max_duration <= 0 or args.beam_size <= 0:
        raise ValueError("Page size, max duration, and beam size must be positive.")
    if not 0 <= args.ctc_weight <= 1:
        raise ValueError("--ctc-weight must be between zero and one.")
    if not args.dropout_levels or any(not 0 < x < 1 for x in args.dropout_levels):
        raise ValueError("--dropout-levels must be strictly between zero and one.")
    if len(set(args.dropout_levels)) != len(args.dropout_levels):
        raise ValueError("--dropout-levels must be unique.")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be non-empty and unique.")
    if args.dropout_intervals <= 0:
        raise ValueError("--dropout-intervals must be positive.")
    if not 0 <= args.audio_only_coverage_threshold <= 1:
        raise ValueError("Automatic routing threshold must be between zero and one.")


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("viavsr.visual_dropout_benchmark")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.FileHandler(path, encoding="utf-8", mode="a"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _protocol(args: argparse.Namespace, config: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": {
            "repository_id": args.dataset_repository,
            "revision": args.dataset_revision,
            "config": args.dataset_config,
            "split": args.split,
            "offset": args.offset,
            "count": args.count,
        },
        "model": {
            "repository_id": config.repository_id,
            "revision": config.revision,
            "device": config.device,
            "dtype": config.dtype,
            "tokenizer_model_path": str(config.tokenizer_model_path),
            "tokenizer_units_path": str(config.tokenizer_units_path),
        },
        "decoder": {
            "name": "joint_beam_search",
            "beam_size": args.beam_size,
            "ctc_weight": args.ctc_weight,
            "attention_weight": 1 - args.ctc_weight,
        },
        "dropout": {
            "levels": args.dropout_levels,
            "seeds": args.seeds,
            "intervals": args.dropout_intervals,
            "kind": "deterministic_separated_contiguous",
            "paired": ["corrupted_av", "interval_gated_av", "automatic"],
        },
        "automatic": {
            "audio_only_when_visual_coverage_lte": args.audio_only_coverage_threshold
        },
        "max_duration_seconds": args.max_duration,
    }


def _split_size(args: argparse.Namespace) -> int:
    query = urlencode(
        {
            "dataset": args.dataset_repository,
            "config": args.dataset_config,
            "split": args.split,
            "offset": 0,
            "length": 1,
        }
    )
    value = _fetch_json(f"{DATASETS_SERVER}/rows?{query}").get("num_rows_total")
    if not isinstance(value, int) or value < 0:
        raise RuntimeError("Dataset API did not report a valid split size.")
    return value


def _variant(
    source: PreparedAVInput, availability: torch.Tensor, *, gated: bool
) -> PreparedAVInput:
    mask = availability.bool().reshape(1, -1)
    if gated:
        return replace(source, visual_availability=mask)
    videos = source.videos.clone() * mask[:, None, :, None, None]
    return replace(source, videos=videos, visual_availability=None)


def _sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in sample.items() if key != "media_path"}


def _failure(
    key: str,
    condition: str,
    sample: dict[str, Any],
    exc: Exception,
    dropout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "key": key,
        "status": "failed",
        "condition": condition,
        "sample": _sample(sample),
        "error": {"type": type(exc).__name__, "message": redact_secrets(str(exc))},
    }
    if dropout is not None:
        record["dropout"] = dropout
    return record


def _infer(
    *,
    store: BenchmarkProgressStore,
    key: str,
    condition: str,
    sample: dict[str, Any],
    prepared: PreparedAVInput,
    assets: Any,
    args: argparse.Namespace,
    preprocessing_seconds: float,
    mode: str,
    dropout: dict[str, Any] | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    if key in store.records:
        return store.records[key]
    try:
        inference = recognize_prepared_av(
            assets,
            prepared,
            decoder="joint_beam_search",
            beam_size=args.beam_size,
            ctc_weight=args.ctc_weight,
            inference_mode=mode,
        )
        result = inference.to_dict()
        result.pop("token_ids", None)
        evaluation = evaluate_transcript(
            reference=sample["reference"], prediction=inference.transcript
        )
        record: dict[str, Any] = {
            "key": key,
            "status": "passed",
            "condition": condition,
            "sample": _sample(sample),
            "media": {
                "duration_seconds": prepared.metadata.duration_seconds,
                "frame_count": int(prepared.videos.shape[2]),
            },
            "preprocessing_seconds": preprocessing_seconds,
            "result": result,
            "evaluation": evaluation.to_dict(),
        }
        if dropout is not None:
            record["dropout"] = dropout
        logger.info(
            "PASSED row=%d condition=%s WER=%.6f CER=%.6f",
            sample["row_index"],
            condition,
            evaluation.wer,
            evaluation.cer,
        )
    except (InferenceError, RuntimeError, ValueError) as exc:
        record = _failure(key, condition, sample, exc, dropout)
        logger.error("FAILED row=%d condition=%s", sample["row_index"], condition)
    return store.append(record)


def _reuse(
    store: BenchmarkProgressStore,
    *,
    key: str,
    source: dict[str, Any],
    route: str,
    dropout: dict[str, Any],
) -> dict[str, Any]:
    if key in store.records:
        return store.records[key]
    copied = {
        name: value
        for name, value in source.items()
        if name not in {"key", "record_type", "protocol_id", "condition"}
    }
    return store.append(
        {
            **copied,
            "key": key,
            "condition": "automatic",
            "automatic_route": route,
            "reused_from_key": source["key"],
            "dropout": dropout,
        }
    )


def _process(
    sample: dict[str, Any],
    *,
    store: BenchmarkProgressStore,
    assets: Any,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    row = sample["row_index"]
    expected = expected_condition_keys(
        sample_index=row,
        dropout_fractions=tuple(args.dropout_levels),
        seeds=tuple(args.seeds),
    )
    if expected.issubset(store.records):
        logger.info("SKIPPED row=%d already complete", row)
        return
    started = time.perf_counter()
    prepared = prepare_mouth_roi_media(
        sample["media_path"], max_duration_seconds=args.max_duration
    )
    preprocessing_seconds = time.perf_counter() - started
    if prepared.videos.shape[2] != sample["expected_frames"]:
        logger.warning("Frame count differs for row=%d", row)

    clean_key = condition_key(sample_index=row, condition="clean_av")
    _infer(
        store=store,
        key=clean_key,
        condition="clean_av",
        sample=sample,
        prepared=prepared,
        assets=assets,
        args=args,
        preprocessing_seconds=preprocessing_seconds,
        mode="audio_visual",
        dropout=None,
        logger=logger,
    )
    audio_key = condition_key(sample_index=row, condition="audio_only")
    audio = _infer(
        store=store,
        key=audio_key,
        condition="audio_only",
        sample=sample,
        prepared=prepared,
        assets=assets,
        args=args,
        preprocessing_seconds=preprocessing_seconds,
        mode="audio_only_experimental",
        dropout=None,
        logger=logger,
    )
    frame_count = int(prepared.videos.shape[2])
    for level in args.dropout_levels:
        for seed in args.seeds:
            mask = generate_contiguous_visual_dropout(
                frame_count=frame_count,
                dropout_fraction=level,
                base_seed=seed,
                sample_index=row,
                interval_count=args.dropout_intervals,
            )
            dropout = mask.to_dict(frame_rate=25)
            availability = torch.from_numpy(mask.availability)
            corrupted_key = condition_key(
                sample_index=row,
                condition="corrupted_av",
                dropout_fraction=level,
                seed=seed,
            )
            _infer(
                store=store,
                key=corrupted_key,
                condition="corrupted_av",
                sample=sample,
                prepared=_variant(prepared, availability, gated=False),
                assets=assets,
                args=args,
                preprocessing_seconds=preprocessing_seconds,
                mode="audio_visual",
                dropout=dropout,
                logger=logger,
            )
            gated_key = condition_key(
                sample_index=row,
                condition="interval_gated_av",
                dropout_fraction=level,
                seed=seed,
            )
            gated = _infer(
                store=store,
                key=gated_key,
                condition="interval_gated_av",
                sample=sample,
                prepared=_variant(prepared, availability, gated=True),
                assets=assets,
                args=args,
                preprocessing_seconds=preprocessing_seconds,
                mode="audio_visual_interval_gated",
                dropout=dropout,
                logger=logger,
            )
            route = select_automatic_inference_mode(
                visual_coverage=1 - mask.actual_dropout_fraction,
                audio_only_coverage_threshold=args.audio_only_coverage_threshold,
            )
            auto_key = condition_key(
                sample_index=row,
                condition="automatic",
                dropout_fraction=level,
                seed=seed,
            )
            source = audio if route == "audio_only_experimental" else gated
            automatic = _reuse(
                store,
                key=auto_key,
                source=source,
                route=route,
                dropout=dropout,
            )
            logger.info(
                "ROUTED row=%d dropout=%.2f seed=%d mode=%s status=%s",
                row,
                level,
                seed,
                route,
                automatic["status"],
            )


def _error_count(evaluation: dict[str, Any], unit: str) -> int:
    return sum(
        evaluation[f"{unit}_{operation}"]
        for operation in ("substitutions", "deletions", "insertions")
    )


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [record for record in records if record.get("status") == "passed"]
    if not passed:
        return {"passed": 0, "failed": len(records)}
    evaluations = [record["evaluation"] for record in passed]
    words = sum(item["reference_words"] for item in evaluations)
    characters = sum(item["reference_characters"] for item in evaluations)
    word_errors = sum(_error_count(item, "word") for item in evaluations)
    char_errors = sum(_error_count(item, "char") for item in evaluations)
    return {
        "passed": len(passed),
        "failed": len(records) - len(passed),
        "unique_samples": len({record["sample"]["row_index"] for record in passed}),
        "reference_words": words,
        "word_errors": word_errors,
        "corpus_wer": word_errors / words if words else 0.0,
        "macro_wer": sum(item["wer"] for item in evaluations) / len(evaluations),
        "reference_characters": characters,
        "character_errors": char_errors,
        "corpus_cer": char_errors / characters if characters else 0.0,
        "macro_cer": sum(item["cer"] for item in evaluations) / len(evaluations),
    }


def _summaries(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    paired: dict[str, list[tuple[float, float]]] = {}
    indexed = {record["key"]: record for record in records}
    for record in records:
        dropout = record.get("dropout")
        group = record["condition"]
        if dropout is not None:
            group += f"/dropout={dropout['target_dropout_fraction']:.2f}"
        groups.setdefault(group, []).append(record)
        if (
            dropout is None
            or record.get("status") != "passed"
            or record["condition"]
            not in {"corrupted_av", "interval_gated_av", "automatic"}
        ):
            continue
        clean = indexed.get(
            condition_key(
                sample_index=record["sample"]["row_index"], condition="clean_av"
            )
        )
        if clean is not None and clean.get("status") == "passed":
            paired.setdefault(group, []).append(
                (
                    record["evaluation"]["wer"] - clean["evaluation"]["wer"],
                    record["evaluation"]["cer"] - clean["evaluation"]["cer"],
                )
            )
    aggregates = {key: _aggregate(value) for key, value in sorted(groups.items())}
    deltas = {
        key: {
            "pairs": len(values),
            "mean_wer_delta_vs_clean": sum(value[0] for value in values) / len(values),
            "mean_cer_delta_vs_clean": sum(value[1] for value in values) / len(values),
        }
        for key, values in sorted(paired.items())
    }
    return aggregates, deltas


def _report(
    paths: ArtifactPaths,
    *,
    store: BenchmarkProgressStore,
    protocol: dict[str, Any],
    assets: Any,
    revision: str,
    sample_count: int,
    elapsed: float,
) -> dict[str, Any]:
    records = list(store.records.values())
    failures = [record for record in records if record.get("status") != "passed"]
    aggregates, deltas = _summaries(records)
    routes: dict[str, int] = {}
    for record in records:
        route = record.get("automatic_route")
        if route is not None:
            routes[route] = routes.get(route, 0) + 1
    payload = {
        "status": "passed" if not failures else "partial_failure",
        "scope": (
            "Paired ViCocktail robustness evaluation with deterministic contiguous "
            "visual-dropout intervals and aligned audio preserved."
        ),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "wall_seconds_this_invocation": elapsed,
        "protocol_id": store.protocol_id,
        "protocol": protocol,
        "dataset_revision_verified": revision,
        "model_and_tokenizer": assets.report.to_dict(),
        "progress": {
            "samples": sample_count,
            "stored_records": len(records),
            "passed_records": len(records) - len(failures),
            "failed_records": len(failures),
        },
        "aggregates": aggregates,
        "paired_deltas": deltas,
        "automatic_routing_counts": routes,
        "artifacts": paths.artifacts(),
    }
    write_json_report(paths.report, payload)
    return payload


def main() -> int:
    args = build_parser().parse_args()
    paths = ArtifactPaths.build(args.output_dir)
    paths.prepare()
    logger = _logger(paths.log)
    started = time.perf_counter()
    try:
        _validate(args)
        revision = _fetch_json(f"{HUGGING_FACE_API}/{args.dataset_repository}").get(
            "sha"
        )
        if revision != args.dataset_revision:
            raise RuntimeError(
                "Official dataset revision mismatch: expected "
                f"{args.dataset_revision}, got {revision}."
            )
        available = max(0, _split_size(args) - args.offset)
        sample_count = available if args.count == 0 else args.count
        if sample_count > available:
            raise RuntimeError(
                f"Requested {sample_count} rows but only {available} remain."
            )
        config = load_model_assets_config(args.config)
        protocol = _protocol(args, config)
        store = BenchmarkProgressStore(paths.results, protocol=protocol)
        logger.info(
            "START protocol=%s existing_records=%d samples=%d",
            store.protocol_id,
            len(store.records),
            sample_count,
        )
        assets = load_vietnamese_avsr_assets(config)
        logger.info(
            "ASSETS model=%s@%s device=%s",
            assets.report.repository_id,
            assets.report.model_revision,
            assets.report.device,
        )
        consumed = 0
        while consumed < sample_count:
            page_count = min(args.page_size, sample_count - consumed)
            samples = _download_samples(
                repository=args.dataset_repository,
                config=args.dataset_config,
                split=args.split,
                offset=args.offset + consumed,
                count=page_count,
                media_dir=paths.media,
            )
            for sample in samples:
                try:
                    _process(
                        sample, store=store, assets=assets, args=args, logger=logger
                    )
                except (
                    MediaInputError,
                    InferenceError,
                    OSError,
                    RuntimeError,
                    ValueError,
                ) as exc:
                    key = condition_key(
                        sample_index=sample["row_index"],
                        condition="sample_preparation",
                    )
                    if key not in store.records:
                        store.append(_failure(key, "sample_preparation", sample, exc))
                    logger.error("FAILED row=%d preparation", sample["row_index"])
                sample["media_path"].unlink(missing_ok=True)
            consumed += len(samples)
        report = _report(
            paths,
            store=store,
            protocol=protocol,
            assets=assets,
            revision=revision,
            sample_count=sample_count,
            elapsed=time.perf_counter() - started,
        )
        logger.info(
            "COMPLETE status=%s records=%d failures=%d",
            report["status"],
            report["progress"]["stored_records"],
            report["progress"]["failed_records"],
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "protocol_id": store.protocol_id,
                    "progress": report["progress"],
                    "aggregates": report["aggregates"],
                    "artifacts": paths.artifacts(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if report["status"] == "passed" else 1
    except (ModelAssetsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        message = redact_secrets(str(exc))
        logger.error("Benchmark aborted: %s", message)
        write_json_report(
            paths.report,
            {
                "status": "failed",
                "error": {"type": type(exc).__name__, "message": message},
                "artifacts": paths.artifacts(),
            },
        )
        return 1
    finally:
        paths.clean_work()


if __name__ == "__main__":
    raise SystemExit(main())
