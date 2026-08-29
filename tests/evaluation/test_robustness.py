from __future__ import annotations

import json

import numpy as np
import pytest

from viavsr.evaluation.robustness import (
    BenchmarkProgressStore,
    condition_key,
    expected_condition_keys,
    generate_contiguous_visual_dropout,
    select_automatic_inference_mode,
)


def test_dropout_mask_is_deterministic_exact_and_separated() -> None:
    first = generate_contiguous_visual_dropout(
        frame_count=100,
        dropout_fraction=0.3,
        base_seed=17,
        sample_index=9,
        interval_count=2,
    )
    second = generate_contiguous_visual_dropout(
        frame_count=100,
        dropout_fraction=0.3,
        base_seed=17,
        sample_index=9,
        interval_count=2,
    )

    assert first.availability.dtype == np.bool_
    assert np.array_equal(first.availability, second.availability)
    assert int((~first.availability).sum()) == 30
    assert first.actual_dropout_fraction == pytest.approx(0.3)
    assert len(first.intervals) == 2
    assert first.intervals[0].end_frame_exclusive < first.intervals[1].start_frame
    assert first.intervals[0].start_frame > 0
    assert first.intervals[-1].end_frame_exclusive < 100


def test_dropout_seed_depends_on_sample_and_base_seed() -> None:
    common = {"frame_count": 80, "dropout_fraction": 0.3, "interval_count": 2}
    first = generate_contiguous_visual_dropout(**common, base_seed=17, sample_index=1)
    other_sample = generate_contiguous_visual_dropout(
        **common, base_seed=17, sample_index=2
    )
    other_seed = generate_contiguous_visual_dropout(
        **common, base_seed=29, sample_index=1
    )

    assert first.effective_seed != other_sample.effective_seed
    assert first.effective_seed != other_seed.effective_seed


@pytest.mark.parametrize(
    ("frame_count", "fraction", "interval_count"),
    [(1, 0.3, 2), (10, 0.0, 2), (10, 1.0, 2), (10, 0.3, 0)],
)
def test_dropout_rejects_invalid_arguments(
    frame_count: int, fraction: float, interval_count: int
) -> None:
    with pytest.raises(ValueError):
        generate_contiguous_visual_dropout(
            frame_count=frame_count,
            dropout_fraction=fraction,
            base_seed=1,
            sample_index=1,
            interval_count=interval_count,
        )


def test_automatic_routing_threshold_is_inclusive() -> None:
    assert (
        select_automatic_inference_mode(
            visual_coverage=0.7, audio_only_coverage_threshold=0.5
        )
        == "audio_visual_interval_gated"
    )
    assert (
        select_automatic_inference_mode(
            visual_coverage=0.5, audio_only_coverage_threshold=0.5
        )
        == "audio_only_experimental"
    )


def test_expected_condition_matrix_has_all_records() -> None:
    keys = expected_condition_keys(
        sample_index=7,
        dropout_fractions=(0.1, 0.3, 0.5),
        seeds=(17, 29, 43),
    )

    assert len(keys) == 29
    assert condition_key(sample_index=7, condition="clean_av") in keys
    assert (
        condition_key(
            sample_index=7,
            condition="automatic",
            dropout_fraction=0.5,
            seed=43,
        )
        in keys
    )


def test_progress_store_resumes_and_rejects_protocol_mismatch(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    protocol = {"dataset": "test", "seeds": [17]}
    first = BenchmarkProgressStore(path, protocol=protocol)
    first.append({"key": "sample=1:condition=clean_av", "status": "passed"})

    resumed = BenchmarkProgressStore(path, protocol=protocol)
    assert set(resumed.records) == {"sample=1:condition=clean_av"}
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["record_type"] == "protocol"
    assert json.loads(lines[1])["record_type"] == "result"

    with pytest.raises(ValueError, match="different benchmark protocol"):
        BenchmarkProgressStore(path, protocol={"dataset": "changed"})
