from __future__ import annotations

import argparse
from pathlib import Path

from viavsr.inference import load_model_assets_config
from viavsr.inference.tokenizer_assets import fetch_tokenizer_assets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and verify the pinned Vietnamese unigram2048 tokenizer."
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main() -> None:
    config = load_model_assets_config(build_parser().parse_args().config)
    downloads = fetch_tokenizer_assets(
        config.tokenizer_model_path, config.tokenizer_units_path
    )
    for result in downloads:
        action = "downloaded" if result.downloaded else "already verified"
        print(f"{result.path}: {action} ({result.sha256})")


if __name__ == "__main__":
    main()
