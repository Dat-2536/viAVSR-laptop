import base64
import gc
import html
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
DURATION_SLACK = 0.5
ON_CLOUD = Path("/mount/src").is_dir()
CLOUD_MAX_WIDTH = 360
LOCAL_MAX_WIDTH = 480

if ON_CLOUD:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

st.set_page_config(
    page_title="viAVSR",
    layout="wide",
    initial_sidebar_state="expanded",
)

RECORDER = components.declare_component(
    "webcam_recorder",
    path=str(Path(__file__).parent / "webcam_recorder"),
)

st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    .block-container { padding-top: 2rem; max-width: 960px; }
    html, body, [class*="css"] { font-family: "IBM Plex Sans", sans-serif; color: #1a1a1a; }
    div[data-testid="stSidebar"] {
        background: #fafaf8;
        border-right: 1px solid #e8e8e4;
    }
    div[data-testid="stSidebar"] .stMarkdown h3 {
        font-family: "Instrument Serif", serif;
        font-weight: 400;
        letter-spacing: -0.02em;
    }
    .masthead {
        border-bottom: 2px solid #1a1a1a;
        padding-bottom: 1.5rem;
        margin-bottom: 2rem;
    }
    .masthead-kicker {
        display: block;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #6b6b68;
        margin-bottom: 0.5rem;
    }
    .masthead h1 {
        font-family: "Instrument Serif", serif;
        font-size: 3.2rem;
        font-weight: 400;
        line-height: 1;
        margin: 0 0 0.6rem 0;
        letter-spacing: -0.03em;
        color: #1a1a1a;
    }
    .masthead-lede {
        margin: 0;
        font-size: 1.05rem;
        color: #4a4a47;
        max-width: 36rem;
        line-height: 1.55;
    }
    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #6b6b68;
        margin-bottom: 0.75rem;
    }
    .viavsr-card {
        background: #fafaf8;
        border: 1px dashed #d4d4d0;
        padding: 2.5rem 1.5rem;
        text-align: center;
        color: #8a8a86;
        font-size: 0.92rem;
    }
    .viavsr-transcript {
        font-family: "Instrument Serif", serif;
        font-size: 1.75rem;
        line-height: 1.5;
        color: #1a1a1a;
        padding: 1.5rem 0 1.5rem 1.25rem;
        margin: 1rem 0 1.5rem;
        border-left: 3px solid #1a1a1a;
        background: #fafaf8;
    }
    .viavsr-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem 1.25rem;
        margin-bottom: 0.5rem;
        font-size: 0.8rem;
        color: #6b6b68;
    }
    .viavsr-meta span { white-space: nowrap; }
    .viavsr-meta strong { color: #1a1a1a; font-weight: 600; }
    div[data-testid="stMetric"] {
        background: #fafaf8;
        border: 1px solid #e8e8e4;
        padding: 0.75rem 1rem;
    }
    div[data-testid="stMetric"] label { font-size: 0.72rem !important; letter-spacing: 0.06em; text-transform: uppercase; }
    .stButton > button[kind="primary"] {
        background: #1a1a1a !important;
        color: #fff !important;
        border: none !important;
        border-radius: 0 !important;
        font-weight: 500 !important;
        letter-spacing: 0.04em !important;
        padding: 0.65rem 2rem !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #333 !important;
    }
    hr { border-color: #e8e8e4; margin: 2rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="masthead">
  <span class="masthead-kicker">Nhận dạng lời nói tiếng Việt</span>
  <h1>viAVSR</h1>
  <p class="masthead-lede">Upload video, quay webcam, hoặc chọn mẫu — hệ thống trích xuất và phiên âm câu nói.</p>
</div>
""",
    unsafe_allow_html=True,
)

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


@st.cache_resource(show_spinner="Đang tải model…")
def _load_cached_model_assets():
    """Keep one model instance in memory across reruns (saves ~1.7 GB per inference)."""
    from viavsr.inference import load_model_assets_config, load_vietnamese_avsr_assets

    _ensure_tokenizer_assets()
    config = load_model_assets_config(CONFIG)
    assets = load_vietnamese_avsr_assets(config)
    if ON_CLOUD:
        import torch

        torch.set_num_threads(1)
    return assets


def _prepare_media(src: Path, clip_seconds: float, *, max_width: int) -> Path:
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
            f"scale='min({max_width},iw)':-2",
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


with st.sidebar:
    st.markdown("### Tuỳ chọn")
    audio_only = st.toggle(
        "Chỉ audio",
        value=ON_CLOUD,
        help="Không dùng hình miệng, chỉ phân tích âm thanh.",
    )
    fast_mode = st.toggle(
        "Rút gọn",
        value=ON_CLOUD and not audio_only,
        disabled=audio_only,
        help="Clip 5 giây.",
    )
    clip_seconds = float(
        st.slider(
            "Độ dài clip",
            min_value=4,
            max_value=8,
            value=FAST_CLIP_SECONDS if fast_mode else DEFAULT_CLIP_SECONDS,
            disabled=fast_mode,
        )
    )
    max_detection = 256 if fast_mode else 320
    max_width = CLOUD_MAX_WIDTH if ON_CLOUD else LOCAL_MAX_WIDTH
    decoder = st.selectbox(
        "Decoder",
        ("ctc_greedy", "joint_beam_search"),
        index=0,
    )
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        st.warning("Cần HF_TOKEN trong Secrets.")

input_col, preview_col = st.columns([1, 1], gap="large")

with input_col:
    st.markdown('<p class="section-label">Nguồn</p>', unsafe_allow_html=True)
    source = st.radio(
        "Chọn nguồn",
        ("Upload", "Webcam", "Sample"),
        horizontal=True,
        label_visibility="collapsed",
    )
    media_path: Path | None = st.session_state.get("media_path")

    if source == "Upload":
        uploaded = st.file_uploader(
            "Video có âm thanh",
            type=["mp4", "webm", "mov"],
            label_visibility="collapsed",
        )
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
                st.error(f"Không đọc được bản ghi webcam: {exc}")
                media_path = None

    else:
        samples = sorted(SAMPLE_DIR.glob("vi_*.mp4"))
        if not samples:
            st.info("Chưa có mẫu tiếng Việt. Dùng Upload hoặc Webcam.")
        else:
            choice = st.selectbox(
                "Mẫu tiếng Việt",
                samples,
                format_func=lambda p: p.name,
                label_visibility="collapsed",
            )
            if choice is not None:
                media_path = choice
                st.session_state.media_path = media_path

with preview_col:
    st.markdown('<p class="section-label">Xem trước</p>', unsafe_allow_html=True)
    if media_path and media_path.is_file():
        duration = _duration_seconds(media_path)
        if duration is not None and duration > clip_seconds:
            st.warning(
                f"File dài {duration:.1f}s — chỉ dùng {clip_seconds:.0f}s đầu."
            )
        st.video(str(media_path))
    else:
        st.markdown(
            '<div class="viavsr-card">Chưa có video.</div>',
            unsafe_allow_html=True,
        )

run = st.button("Chạy nhận dạng", type="primary", disabled=not (media_path and media_path.is_file()))

if run and media_path and media_path.is_file():
    try:
        with st.status("Đang xử lý…", expanded=True) as status:
            status.write("Chuẩn bị tokenizer…")
            _ensure_tokenizer_assets()
            status.write("Xử lý video…")
            prepared = _prepare_media(
                media_path,
                clip_seconds,
                max_width=max_width,
            )
            status.write(
                "Nhận dạng…"
                if audio_only
                else "Theo dõi khuôn mặt và nhận dạng…"
            )
            assets = _load_cached_model_assets()
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
                skip_face_tracking=audio_only,
                preloaded_assets=assets,
            )
            status.write("Xong.")
            status.update(label="Hoàn tất", state="complete")
        st.session_state.result = result
        gc.collect()
    except Exception as exc:
        st.error(str(exc))

if "result" in st.session_state:
    result = st.session_state.result
    inner = result.get("result") or {}
    modality = result.get("modality_decision") or {}
    artifacts = result.get("artifacts") or {}
    timings = result.get("timings_seconds") or {}
    status = result.get("status", "")
    transcript = inner.get("transcript") or "(trống)"

    st.markdown('<p class="section-label">Kết quả</p>', unsafe_allow_html=True)

    mode = modality.get("selected_mode", "—")
    total_s = timings.get("total", 0)

    st.markdown(
        f'<div class="viavsr-meta">'
        f'<span>Trạng thái: <strong>{html.escape(status)}</strong></span>'
        f'<span>Chế độ: <strong>{html.escape(mode.replace("_", " "))}</strong></span>'
        f'<span>Thời gian: <strong>{total_s:.1f}s</strong></span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="viavsr-transcript">{html.escape(transcript)}</div>',
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Tổng", f"{total_s:.1f}s")
    metric_cols[1].metric("Modality", mode.replace("_", " "))
    metric_cols[2].metric("Model", f"{timings.get('model_loading', 0):.1f}s")
    metric_cols[3].metric("Inference", f"{timings.get('inference', 0):.1f}s")

    step_keys = ("face_tracking", "mouth_roi", "model_loading", "inference")
    step_parts = [
        f"{key.replace('_', ' ')}: {timings[key]:.1f}s"
        for key in step_keys
        if key in timings and timings[key]
    ]
    if step_parts:
        st.caption(" · ".join(step_parts))

    mouth = artifacts.get("mouth_roi")
    if mouth and Path(mouth).is_file():
        with st.expander("Video miệng"):
            st.video(mouth)

    warnings = result.get("warnings") or []
    if warnings:
        st.warning("\n".join(str(item) for item in warnings))
    if status != "passed" and result.get("error"):
        st.error(result["error"].get("message", "Inference failed"))

    with st.expander("Báo cáo JSON"):
        st.json(result)
