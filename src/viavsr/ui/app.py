import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CONFIG = ROOT / "configs" / "config.yaml"
UPLOAD_DIR = ROOT / "uploads"
SAMPLE_DIR = ROOT / "samples" / "webcam"
OUTPUT_ROOT = ROOT / "outputs" / "demo"
DEFAULT_CLIP_SECONDS = 8
FAST_CLIP_SECONDS = 5
# FFmpeg/ffprobe often report a few frames over the trim length (e.g. 8.08s for -t 8).
DURATION_SLACK = 0.5
ON_CLOUD = Path("/mount/src").is_dir()

st.set_page_config(page_title="viAVSR", layout="wide")

RECORDER = components.declare_component(
    "webcam_recorder",
    path=str(Path(__file__).parent / "webcam_recorder"),
)

st.title("viAVSR")
if ON_CLOUD:
    st.info(
        "Streamlit Cloud dùng **CPU yếu**. Một lần chạy thường **5–15 phút** "
        "(lần đầu còn tải model ~1.7GB). Để nhanh hơn: bật **Fast mode** hoặc chạy local."
    )
else:
    st.caption("Clips dài hơn giới hạn sẽ được cắt. CPU local thường mất vài phút.")


def _tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    prefix = Path(os.environ.get("CONDA_PREFIX", ""))
    for candidate in (
        prefix / "Library" / "bin" / f"{name}.exe",
        Path.home() / "miniconda3" / "envs" / "viavsr" / "Library" / "bin" / f"{name}.exe",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _save_upload(name: str, data: bytes) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / name
    path.write_bytes(data)
    return path


def _data_url_to_webm(data_url: str) -> Path:
    marker = "base64,"
    encoded = (
        data_url[data_url.find(marker) + len(marker) :]
        if marker in data_url
        else data_url.rsplit(",", 1)[-1]
    )
    encoded = encoded.strip().replace("\n", "").replace("\r", "").replace(" ", "+")
    encoded += "=" * ((4 - len(encoded) % 4) % 4)
    return _save_upload("webcam_record.webm", base64.b64decode(encoded))


def _duration_seconds(src: Path) -> float | None:
    ffprobe = _tool("ffprobe")
    if ffprobe is None:
        return None
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(src),
        ],
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        return None
    try:
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return duration


@st.cache_resource(show_spinner=False)
def _ensure_tokenizer_assets() -> None:
    """Download pinned tokenizer files on first run (not committed to git)."""
    from viavsr.inference import load_model_assets_config
    from viavsr.inference.tokenizer_assets import fetch_tokenizer_assets

    config = load_model_assets_config(CONFIG)
    fetch_tokenizer_assets(config.tokenizer_model_path, config.tokenizer_units_path)


def _prepare_media(src: Path, clip_seconds: float) -> Path:
    ffmpeg = _tool("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg was not found. Activate the viavsr conda env and retry.")
    dest = UPLOAD_DIR / f"{src.stem}_prep.mp4"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-ss",
            "0",
            "-t",
            str(clip_seconds),
            "-i",
            str(src),
            "-vf",
            "scale='min(480,iw)':-2",
            "-r",
            "25",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(dest),
        ],
        capture_output=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0 or not dest.is_file():
        detail = result.stderr.decode("utf-8", errors="replace")[-400:]
        raise RuntimeError(f"Could not trim/scale the video.\n{detail}")
    return dest


source = st.radio("Input", ("Upload", "Webcam", "Sample"), horizontal=True)
media_path: Path | None = st.session_state.get("media_path")

if source == "Upload":
    uploaded = st.file_uploader("Video with audio", type=["mp4", "webm", "mov"])
    if uploaded is not None:
        media_path = _save_upload(uploaded.name, uploaded.getvalue())
        st.session_state.media_path = media_path

elif source == "Webcam":
    recorded = RECORDER(key="webcam")
    if recorded:
        try:
            media_path = _data_url_to_webm(recorded)
            st.session_state.media_path = media_path
        except Exception as exc:
            st.error(f"Could not read the webcam recording: {exc}")
            media_path = None

else:
    samples = sorted(SAMPLE_DIR.glob("vi_*.mp4"))
    if not samples:
        st.info("No Vietnamese samples yet. Use Upload or Webcam.")
    else:
        choice = st.selectbox("Vietnamese sample", samples, format_func=lambda p: p.name)
        if choice is not None:
            media_path = choice
            st.session_state.media_path = media_path

if media_path and media_path.is_file():
    fast_mode = st.checkbox(
        "Fast mode (clip ngắn hơn, face detection nhẹ hơn — nhanh ~2× trên Cloud)",
        value=ON_CLOUD,
    )
    clip_seconds = float(
        st.slider(
            "Clip length (seconds)",
            min_value=4,
            max_value=8,
            value=FAST_CLIP_SECONDS if fast_mode else DEFAULT_CLIP_SECONDS,
            disabled=fast_mode,
        )
    )
    max_detection = 256 if fast_mode else 320

    duration = _duration_seconds(media_path)
    if duration is not None and duration > clip_seconds:
        st.warning(
            f"This file is {duration:.1f}s. Only the first {clip_seconds:.0f}s will be used."
        )
    st.video(str(media_path))

    decoder = st.selectbox(
        "Decoder",
        ("ctc_greedy", "joint_beam_search"),
        index=0,
        help="ctc_greedy nhanh hơn; joint_beam_search chậm hơn nhưng đôi khi chính xác hơn.",
    )
    run = st.button("Run inference", type="primary")
    if run:
        try:
            if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
                st.warning(
                    "Set **HF_TOKEN** in Streamlit Cloud secrets so the model can download from Hugging Face."
                )
            with st.status("Working…", expanded=True) as status:
                status.write("0/3 — Tải tokenizer (lần đầu ~300KB)…")
                _ensure_tokenizer_assets()
                status.write("1/3 — Cắt và scale video…")
                prepared = _prepare_media(media_path, clip_seconds)
                status.write(
                    "2/3 — Face tracking + tải model AV-HuBERT (~1.7GB). "
                    "Cloud CPU có thể **10–25 phút**; giữ tab mở."
                )
                from viavsr.demo import run_end_to_end_demo

                result = run_end_to_end_demo(
                    config_path=CONFIG,
                    media_path=prepared,
                    output_root=OUTPUT_ROOT,
                    tracking_device="cpu",
                    decoder=decoder,
                    max_duration_seconds=float(clip_seconds + DURATION_SLACK),
                    max_detection_size=max_detection,
                    visual_fallback_policy="whole_utterance",
                )
                status.write("3/3 — Hoàn tất.")
                status.update(label="Finished", state="complete")
            st.session_state.result = result
        except Exception as exc:
            st.error(str(exc))

if "result" in st.session_state:
    result = st.session_state.result
    inner = result.get("result") or {}
    modality = result.get("modality_decision") or {}
    artifacts = result.get("artifacts") or {}

    st.subheader("Transcript")
    st.write(inner.get("transcript") or "(empty)")

    cols = st.columns(3)
    cols[0].metric("Status", result.get("status", ""))
    cols[1].metric("Modality", modality.get("selected_mode", ""))
    cols[2].metric("Time (s)", f"{result.get('timings_seconds', {}).get('total', 0):.1f}")

    timings = result.get("timings_seconds") or {}
    if timings:
        parts = []
        for key in (
            "face_tracking",
            "model_loading",
            "inference",
            "mouth_roi",
        ):
            if key in timings and timings[key]:
                parts.append(f"{key}: {timings[key]:.1f}s")
        if parts:
            st.caption("Thời gian từng bước: " + " · ".join(parts))

    mouth = artifacts.get("mouth_roi")
    if mouth and Path(mouth).is_file():
        st.caption("Processed mouth video")
        st.video(mouth)

    warnings = result.get("warnings") or []
    if warnings:
        st.warning("\n".join(str(item) for item in warnings))
    if result.get("status") != "passed" and result.get("error"):
        st.error(result["error"].get("message", "Inference failed"))

    with st.expander("Full report"):
        st.json(result)
