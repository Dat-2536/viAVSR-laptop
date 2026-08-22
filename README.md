# Vietnamese Audio-Visual Speech Recognition Using AV-HuBERT with Joint CTC/Attention Decoding for Laptop-Based Record Transcription

## VIAVSR-4 model asset check

Install the project in the `viavsr` Conda environment, download the pinned
Vietnamese tokenizer, then validate it with the released checkpoint:

```bash
conda activate viavsr
pip install -e ".[dev]"
python scripts/fetch_tokenizer_assets.py --config configs/vietnamese_avsr.yaml
python scripts/check_model_assets.py --config configs/vietnamese_avsr.yaml
```

The checkpoint is gated. Accept its Hugging Face access conditions and provide
credentials through `HF_TOKEN`; credentials must not be written into YAML.
`configs/vietnamese_avsr.yaml` requests CUDA explicitly and fails clearly when
CUDA is unavailable. Change `model.device` to `cpu` for an intentional CPU run.

Reports and logs are generated under `outputs/model_assets/`. Model caches,
tokenizer binaries, reports, and logs are ignored by Git.

## VIAVSR-6 transcript evaluation

Evaluate a raw Vietnamese AVSR prediction against its reference transcript:

```bash
python scripts/evaluate_transcripts.py \
  --reference-text "hôm nay trời đẹp" \
  --prediction-text "hôm nay trời lạnh" \
  --output outputs/evaluation/metrics.json
```

The command prints a JSON result and optionally writes the same payload to
`--output`. It reports WER, CER, word/character edit counts, and normalized
reference lengths.

Sprint 1 normalization:

- Unicode NFC normalization
- lowercase
- selected punctuation replaced by spaces
- repeated whitespace collapsed
- Vietnamese diacritics preserved

An empty normalized reference and prediction have zero error. A non-empty
prediction for an empty reference is reported as insertion errors.
