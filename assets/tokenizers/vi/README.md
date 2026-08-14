# Vietnamese tokenizer assets

The Vietnamese tokenizer is pinned to
`nguyenvulebinh/viCocktail@ad644a77e8e3177aa7422510302c11de5282fa26`.
Download and verify it with:

```bash
python scripts/fetch_tokenizer_assets.py --config configs/vietnamese_avsr.yaml
```

The command writes these files atomically:

```text
unigram2048.model
unigram2048_units.txt
```

The actual tokenizer binaries are ignored by Git. Their source URLs and SHA-256
checksums are versioned in `manifest.json`.
