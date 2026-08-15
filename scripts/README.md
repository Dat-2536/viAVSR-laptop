# scripts

Human-facing command-line entrypoints live here.

Sprint 1 expected scripts:
- `evaluate_transcripts.py` — VIASVR-6
- environment smoke test — VIASVR-3
- `fetch_tokenizer_assets.py` — download and verify pinned Vietnamese tokenizer files
- `check_model_assets.py` — load and validate Vietnamese AV-HuBERT assets (VIASVR-4)
- official sample inference — VIASVR-5

Reusable logic should live under `src/viavsr/`, not directly in scripts.
