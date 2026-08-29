from __future__ import annotations

from pathlib import Path

from scripts.run_official_benchmark import BenchmarkArtifactPaths, build_parser


def test_benchmark_artifacts_keep_only_report_and_log_by_default(
    tmp_path: Path,
) -> None:
    paths = BenchmarkArtifactPaths.for_output(tmp_path / "benchmark")
    paths.reset_work_directory()
    paths.media_directory.mkdir()
    paths.predictions_directory.mkdir()
    (paths.media_directory / "sample.mp4").write_bytes(b"media")
    (paths.predictions_directory / "sample.json").write_text(
        "{}",
        encoding="utf-8",
    )

    assert set(paths.to_dict(include_intermediates=False)) == {
        "report",
        "execution_log",
    }

    paths.cleanup_intermediates()

    assert not paths.work_directory.exists()


def test_benchmark_debug_artifacts_are_explicit(tmp_path: Path) -> None:
    paths = BenchmarkArtifactPaths.for_output(tmp_path / "benchmark")

    assert set(paths.to_dict(include_intermediates=True)) == {
        "report",
        "execution_log",
        "work_directory",
        "predictions_directory",
        "media_directory",
    }


def test_benchmark_cli_does_not_keep_intermediates_by_default() -> None:
    args = build_parser().parse_args(["--config", "config.yaml"])

    assert args.keep_intermediates is False
