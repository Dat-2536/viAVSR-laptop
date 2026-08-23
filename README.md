# viAVSR-laptop

Vietnamese audio-visual speech recognition for short laptop webcam recordings.

The project combines synchronized speech audio and a tracked mouth region, encodes both modalities with a released Vietnamese AV-HuBERT model, and decodes Vietnamese text with either CTC greedy decoding or joint CTC/attention beam search.

## Overview

```text
webcam video with embedded audio
    → media validation
    → face detection and 68-point landmark tracking
    → affine face alignment
    → 96×96 mouth-region extraction
    → synchronized audio/video feature preparation
    → AV-HuBERT encoding
    → CTC greedy or joint CTC/attention decoding
    → Vietnamese transcript
    → optional WER/CER evaluation
```

The implementation targets Linux. Windows users should run it inside WSL2 Ubuntu so FFmpeg, Conda, CUDA, and POSIX paths behave consistently.

## Features

- Vietnamese AV-HuBERT checkpoint loading from Hugging Face.
- Verified Vietnamese SentencePiece `unigram2048` tokenizer.
- Configurable CPU or CUDA model placement.
- Model/tokenizer vocabulary compatibility validation.
- Raw webcam media inspection.
- RetinaFace detection and FAN 68-point facial landmark tracking.
- Temporal landmark interpolation and smoothing.
- Affine face alignment and 96×96 mouth-ROI export.
- Synchronized 25 fps video and 16 kHz mono audio preprocessing.
- CTC greedy and joint CTC/attention beam-search decoding.
- Vietnamese-aware WER and CER that preserve diacritics.
- JSON reports for model assets, preprocessing, inference, and evaluation.

## Model and tokenizer

| Asset | Pinned source |
| --- | --- |
| Model | [`nguyenvulebinh/AV-HuBERT-CTC-Attention-VI`](https://huggingface.co/nguyenvulebinh/AV-HuBERT-CTC-Attention-VI) |
| Model revision | `b8a1fa5d6079701b3f8f791bfd601057fbd23de3` |
| Dataset | [`nguyenvulebinh/ViCocktail`](https://huggingface.co/datasets/nguyenvulebinh/ViCocktail) |
| Tokenizer repository | `nguyenvulebinh/viCocktail` |
| Tokenizer revision | `ad644a77e8e3177aa7422510302c11de5282fa26` |
| SentencePiece model | `unigram2048.model` |
| Token units | `unigram2048_units.txt` |

Expected tokenizer checksums:

| File | SHA-256 |
| --- | --- |
| `unigram2048.model` | `21ca39e799b64044d75edccd9016fac0315e64f89bdd43fbd3089607dceb9d64` |
| `unigram2048_units.txt` | `ea7b25e67a302305ffdb59909419c08822b3607a6b03871adef2bcb9f6ebec25` |

Tokenizer binaries, the checkpoint, and Hugging Face caches are downloaded locally and are not committed.

## Requirements

- Linux or WSL2 Ubuntu.
- Conda or Miniconda.
- Python 3.11.
- FFmpeg and FFprobe.
- A Hugging Face account with access to the released checkpoint.
- An NVIDIA GPU is recommended for full inference.

The pipeline has been validated on a GeForce GTX 1660 Ti with 6 GB of VRAM. CPU placement is supported but considerably slower.

## Installation

```bash
git clone <repository-url>
cd viAVSR-laptop

conda env create -f environment/environment.yml
conda activate viavsr
pip install -e ".[dev]"
```

For an existing environment:

```bash
conda env update -f environment/environment.yml --prune
conda activate viavsr
pip install -e ".[dev]"
```

Confirm the package and CUDA runtime:

```bash
python -c 'import torch, viavsr; print("viavsr:", viavsr.__file__); print("torch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'
```

## Configuration

The default runtime configuration is [`configs/config.yaml`](configs/config.yaml).

```yaml
model:
  repository_id: nguyenvulebinh/AV-HuBERT-CTC-Attention-VI
  revision: b8a1fa5d6079701b3f8f791bfd601057fbd23de3
  cache_dir: .cache/models
  device: cuda
  dtype: float32

tokenizer:
  model_path: assets/tokenizers/vi/unigram2048.model
  units_path: assets/tokenizers/vi/unigram2048_units.txt
```

Set `model.device` to `cpu` for an intentional CPU run. The loader does not silently fall back from CUDA to CPU. Do not put credentials or machine-specific absolute paths in YAML.

## Hugging Face authentication

Accept the model repository access conditions, then export a personal read-only token without writing it into project files:

```bash
read -rsp "Hugging Face token: " HF_TOKEN
export HF_TOKEN
echo
```

Confirm that the variable exists without printing its value:

```bash
python -c 'import os; print("HF_TOKEN configured:", bool(os.environ.get("HF_TOKEN")))'
```

Each team member should use their own token. Never commit tokens, `.env` files, or commands containing a real credential.

## Download and validate model assets

Download and checksum the Vietnamese tokenizer:

```bash
python scripts/fetch_tokenizer_assets.py \
  --config configs/config.yaml
```

Load the checkpoint, validate model/tokenizer vocabulary dimensions, and run Vietnamese tokenizer round trips:

```bash
python scripts/check_model_assets.py \
  --config configs/config.yaml
```

Expected successful output includes:

```text
Model loaded: PASSED
Tokenizer loaded: PASSED
SentencePiece vocabulary: 2048
ASR tokenizer vocabulary: 2057
Model output vocabulary: 2057
Vocabulary compatibility: PASSED
Vietnamese round-trip: PASSED
Device: cuda
```

Metadata and logs are generated under `outputs/model_assets/`.

## Recording input

Record one Vietnamese utterance per file.

- Prefer 3–5 seconds; the default maximum is 15 seconds.
- Keep the speaker frontal and well lit.
- Keep the full face visible, including chin and forehead.
- Avoid mouth occlusion, rapid head turns, and strong backlighting.
- Keep the mouth sufficiently large in the frame.
- Store audio and video in the same container.
- Prefer MP4 or WebM.
- Save the exact spoken sentence if WER/CER will be calculated.

FFmpeg normalizes source frame rate, sample rate, and channel count later.

Local recordings belong under `samples/webcam/` or `recordings/` and are ignored:

```bash
mkdir -p samples/webcam
```

Suggested local naming:

```text
samples/webcam/webcam_001.mp4
samples/webcam/webcam_001_reference.txt
```

## End-to-end webcam inference

### 1. Inspect the raw recording

```bash
python scripts/check_media_input.py \
  --media samples/webcam/webcam_001.mp4 \
  --output outputs/inference/webcam_001_preflight.json
```

Typical `next_stage` values:

- `face_alignment_and_mouth_roi_extraction_required` for ordinary webcam video.
- `ready_for_inference` for a prepared 96×96 mouth video.
- `split_or_record_a_shorter_clip` when the file exceeds the duration limit.

A raw full-frame video normally reports `has_mouth_roi_resolution: false`. This describes the current file and is not a tracking failure.

### 2. Track faces and landmarks

One recording:

```bash
python scripts/track_webcam_faces.py \
  --device auto \
  --media samples/webcam/webcam_001.mp4
```

Several recordings:

```bash
python scripts/track_webcam_faces.py \
  --device auto \
  --media \
    samples/webcam/webcam_001.mp4 \
    samples/webcam/webcam_002.mp4 \
    samples/webcam/webcam_003.mp4 \
    samples/webcam/webcam_004.mp4
```

Generated artifacts:

```text
outputs/preprocessing/face_tracks/<stem>_face_track.npz
outputs/preprocessing/face_tracks/<stem>_face_track.json
```

The NPZ contains landmarks, face boxes, confidence scores, and detection masks. Load it with `numpy.load(path, allow_pickle=False)`. Use `--device cuda` to require CUDA or `--device cpu` to force CPU tracking.

### 3. Align faces and extract mouth regions

```bash
python scripts/prepare_webcam_mouth_roi.py \
  --media \
    samples/webcam/webcam_001.mp4 \
    samples/webcam/webcam_002.mp4 \
    samples/webcam/webcam_003.mp4 \
    samples/webcam/webcam_004.mp4
```

Generated artifacts:

```text
outputs/preprocessing/mouth_roi/<stem>_mouth96.mp4
outputs/preprocessing/mouth_roi/<stem>_mouth96.json
```

Each MP4 contains a 96×96 aligned grayscale mouth track at 25 fps and synchronized 16 kHz mono audio.

### 4. Run model inference

Joint CTC/attention decoding:

```bash
python scripts/run_media_inference.py \
  --config configs/config.yaml \
  --media outputs/preprocessing/mouth_roi/webcam_001_mouth96.mp4 \
  --decoder joint_beam_search \
  --beam-size 3 \
  --ctc-weight 0.1 \
  --reference-text "hôm nay thời tiết bên ngoài rất đẹp" \
  --output outputs/inference/webcam_001_joint_beam.json
```

Omit `--reference-text` when no reference is available. The transcript and token IDs are still produced, but WER/CER are omitted.

CTC greedy decoding:

```bash
python scripts/run_media_inference.py \
  --config configs/config.yaml \
  --media outputs/preprocessing/mouth_roi/webcam_001_mouth96.mp4 \
  --decoder ctc_greedy \
  --output outputs/inference/webcam_001_greedy.json
```

Released joint-decoder settings:

- Beam size: 3.
- Attention weight: 0.9.
- CTC-prefix weight: 0.1.
- Language-model weight: 0.
- Length-bonus weight: 0.

The inference report includes media metadata, tensor shapes, token IDs, transcript, decoder parameters, device, dtype, timing, and optional evaluation.

### 5. Evaluate an existing transcript

```bash
python scripts/evaluate_transcripts.py \
  --reference-text "hôm nay trời đẹp" \
  --prediction-text "hôm nay trời lạnh" \
  --output outputs/evaluation/metrics.json
```

Normalization applies Unicode NFC, lowercase, a consistent punctuation policy, and whitespace collapse while preserving Vietnamese diacritics. Raw model predictions are evaluated without LLM rewriting.

## Preprocessing contract

| Stage | Contract |
| --- | --- |
| Prepared video | 25 fps, grayscale, 96×96 mouth frames |
| Video transform | divide by 255, center-crop to 88×88, normalize with mean 0.421 and standard deviation 0.165 |
| Video tensor | `[1, 1, T, 88, 88]` |
| Prepared audio | mono, 16 kHz |
| Synchronization | trim or pad to 640 audio samples per video frame |
| Audio transform | 26-bin log filterbanks, stack four frames, layer normalization |
| Audio tensor | `[1, 104, T]` |

Inference currently processes one utterance per call.

## Webcam evaluation results

Four local webcam recordings were processed from raw full-frame video through tracking, mouth extraction, inference, and WER/CER evaluation.

### Joint-decoder transcripts

| Sample | Duration | Reference | Prediction | WER | CER |
| --- | ---: | --- | --- | ---: | ---: |
| `webcam_001` | 4.40 s | hôm nay thời tiết bên ngoài rất đẹp | hôm nay thời tiết bên ngoài rất đẹp | 0.00% | 0.00% |
| `webcam_002` | 4.04 s | ở đây mở nhạc to quá không nghe rõ | ở đây mở nhạc to quá không nghe rõ | 0.00% | 0.00% |
| `webcam_003` | 5.04 s | mình đi ăn tối thôi | mình đi anh tối thôi | 20.00% | 10.53% |
| `webcam_004` | 7.20 s | một ông sao sáng hai ông sáng sao ba ông sao sáng | một ông sáu sáng hai ông sáng sáu ba ông sáu sáng | 25.00% | 12.24% |

### Decoder comparison

| Decoder | Word errors / words | Corpus WER | Character errors / characters | Corpus CER |
| --- | ---: | ---: | ---: | ---: |
| CTC greedy | 6 / 34 | 17.65% | 13 / 137 | 9.49% |
| Joint CTC/attention | 4 / 34 | 11.76% | 8 / 137 | 5.84% |

The joint decoder made `webcam_001` and `webcam_002` exact, matched the greedy result for `webcam_003`, and increased the repeated `sao`/`sáu` confusion in `webcam_004`.

These four files are a local integration set, not a statistically representative model benchmark.

A separate ten-sample compatibility run on the official ViCocktail clean test split produced:

| Metric | Result |
| --- | ---: |
| Successful samples | 10 / 10 |
| Evaluated duration | 69.50 s |
| Corpus WER | 5.71% |
| Corpus CER | 3.20% |
| Inference real-time factor | 0.994 |

This deterministic prefix is also a smoke check rather than a full benchmark. Reproduce it with:

```bash
python scripts/run_official_benchmark.py \
  --config configs/config.yaml \
  --split test \
  --offset 0 \
  --count 10 \
  --beam-size 3 \
  --ctc-weight 0.1
```

Artifacts are written under `outputs/official_benchmark/` and ignored by Git.

## Project structure

```text
assets/                   tokenizer manifest and downloaded local assets
configs/                  model and runtime configuration
environment/              Conda environment definition
samples/                  local media or lightweight sample metadata
scripts/                  command-line entry points
src/viavsr/
  evaluation/             Vietnamese normalization and WER/CER
  inference/              model loading, decoding, and recognition
  preprocessing/          media probing, face tracking, and mouth extraction
  utils/                  small shared utilities
tests/                    automated tests
outputs/                  generated reports, media, and predictions
```

Reusable implementation belongs under `src/viavsr/`. Scripts should primarily parse arguments and orchestrate library functions.

## Testing

```bash
python -m pytest
```

Current validated result:

```text
86 passed
```

When Ruff is installed:

```bash
ruff check scripts src tests
ruff format --check scripts src tests
```

The vendored AVSRCocktail implementation under `src/viavsr/inference/vendor/` is excluded from Ruff to preserve upstream behavior.

## Generated files and Git policy

Keep these files local:

- Raw or processed audio/video.
- Downloaded datasets.
- Checkpoints and Hugging Face caches.
- Tokenizer binaries.
- Face-track NPZ files.
- Generated reports, predictions, and logs.
- Conda or virtual environments.
- Experiment-tracking directories.
- Credentials and `.env` files.

Small source files, tests, YAML configuration, tokenizer manifests, and lightweight sample metadata may be committed.

Before committing:

```bash
git status --short
git diff --cached
```

Do not force-add ignored media or model artifacts.

## Remote GPU usage

The same workflow applies to a lab server or rented Linux GPU:

1. Clone the repository on the remote Linux filesystem.
2. Create the `viavsr` Conda environment.
3. Export a personal `HF_TOKEN`.
4. Keep `model.device: cuda`.
5. Run tokenizer and model-asset validation.
6. Copy only required input media to the server.
7. Run preprocessing and inference.
8. Copy back small JSON reports and transcripts.

Use persistent storage for `.cache/models/` when available so the checkpoint is not downloaded for every session.

## Troubleshooting

### `has_mouth_roi_resolution` is false

This is expected for ordinary webcam video. Run tracking and mouth extraction before inference.

### CUDA was requested but is unavailable

Check `nvidia-smi`, the CUDA-enabled PyTorch installation, and WSL GPU support. Set `model.device: cpu` only for an intentional CPU run.

### Model access fails

Accept the model terms and verify `HF_TOKEN`. Never put the token in `configs/config.yaml`.

### The prepared mouth video does not exist

Run `track_webcam_faces.py`, then `prepare_webcam_mouth_roi.py`. Inference consumes `<stem>_mouth96.mp4`, not the raw recording.

### The shell reports unrecognized arguments or command not found

A line-continuation backslash must be the final character on its line. Do not put spaces after it.

### Face detection is unstable

Improve frontal pose, lighting, face size, and mouth visibility. Inspect the tracking JSON for detection rate, interpolated frames, confidence, and maximum missing runs.

## Current limitations

- The checkpoint is Vietnamese-focused and unreliable for English–Vietnamese code switching.
- Numeric text may produce unknown tokenizer pieces.
- Webcam accuracy depends on alignment, visibility, synchronization, and recording quality.
- The CLI processes one utterance at a time.
- Long recordings should be segmented before inference.
- Beam parameters have not been tuned on a large local development set.
- No language model is included in the released joint decoder.
- The repository provides command-line workflows rather than a recording UI.

## Third-party implementation

Model and decoder reproduction includes code derived from AVSRCocktail. See `src/viavsr/inference/vendor/avsrcocktail/LICENSE` and `src/viavsr/inference/vendor/avsrcocktail/NOTICE.md` for third-party terms and provenance.
