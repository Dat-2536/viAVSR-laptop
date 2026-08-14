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
