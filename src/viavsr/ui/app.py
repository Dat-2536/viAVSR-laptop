import base64
import json
import os
import shutil
import subprocess
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from viavsr.demo import run_end_to_end_demo

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs" / "config.yaml"
UPLOAD_DIR = ROOT / "uploads"
SAMPLE_DIR = ROOT / "samples" / "webcam"
OUTPUT_ROOT = ROOT / "outputs" / "demo"
CLIP_SECONDS = 8
RECORDER = components.declare_component(
    "webcam_recorder",
    path=str(Path(__file__).parent / "webcam_recorder"),
)

st.set_page_config(page_title="viAVSR", layout="wide")
st.title("viAVSR")


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


def _prepare_media(src: Path) -> Path:
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
            str(CLIP_SECONDS),
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
    duration = _duration_seconds(media_path)
    if duration is not None and duration > CLIP_SECONDS:
        st.warning(
            f"This file is {duration:.1f}s. Only the first {CLIP_SECONDS}s will be used."
        )
    st.video(str(media_path))

    decoder = st.selectbox("Decoder", ("ctc_greedy", "joint_beam_search"))
    run = st.button("Run inference", type="primary")
    if run:
        try:
            with st.status("Working…", expanded=True) as status:
                status.write("Trimming and scaling the clip…")
                prepared = _prepare_media(media_path)
                status.write(
                    "Face tracking + inference on CPU. This often takes 2–5 minutes; leave the tab open."
                )
                result = run_end_to_end_demo(
                    config_path=CONFIG,
                    media_path=prepared,
                    output_root=OUTPUT_ROOT,
                    tracking_device="cpu",
                    decoder=decoder,
                    max_duration_seconds=float(CLIP_SECONDS),
                    max_detection_size=320,
                    visual_fallback_policy="whole_utterance",
                )
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
