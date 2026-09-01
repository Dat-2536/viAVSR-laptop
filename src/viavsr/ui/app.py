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
    page_title="viAVSR — Vietnamese AVSR",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RECORDER = components.declare_component(
    "webcam_recorder",
    path=str(Path(__file__).parent / "webcam_recorder"),
)

st.markdown(
    """
<style>
    .block-container { padding-top: 1.5rem; max-width: 1100px; }
    .viavsr-hero {
        background: linear-gradient(135deg, #312e81 0%, #4338ca 45%, #6366f1 100%);
        border-radius: 16px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.25rem;
        color: #f8fafc;
        box-shadow: 0 12px 40px rgba(49, 46, 129, 0.35);
    }
    .viavsr-hero h1 { margin: 0 0 0.35rem 0; font-size: 1.85rem; font-weight: 700; color: #fff; }
    .viavsr-hero p { margin: 0; opacity: 0.92; font-size: 0.95rem; line-height: 1.5; }
    .viavsr-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .viavsr-transcript {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-left: 4px solid #818cf8;
        border-radius: 0 12px 12px 0;
        padding: 1.25rem 1.5rem;
        font-size: 1.35rem;
        line-height: 1.65;
        color: #f1f5f9;
        min-height: 3rem;
    }
    .viavsr-badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-right: 0.35rem;
    }
    .viavsr-badge-ok { background: rgba(34, 197, 94, 0.2); color: #86efac; }
    .viavsr-badge-warn { background: rgba(251, 191, 36, 0.2); color: #fde68a; }
    .viavsr-badge-info { background: rgba(129, 140, 248, 0.25); color: #c7d2fe; }
    div[data-testid="stSidebar"] { background: #0f172a; }
    div[data-testid="stSidebar"] .stMarkdown { color: #cbd5e1; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="viavsr-hero">
  <h1>viAVSR</h1>
  <p>Nhận dạng giọng nói tiếng Việt từ video — AV-HuBERT · upload, webcam hoặc mẫu có sẵn.</p>
</div>
""",
    unsafe_allow_html=True,
)

if ON_CLOUD:
    st.info(
        "**Streamlit Cloud:** CPU giới hạn, có thể **bị throttle** sau thời gian dài. "
        "Đang chạy inference thì **đừng push code** lên branch `deploy`. "
        "Bật **Chỉ audio** trong sidebar để tiết kiệm RAM và thời gian."
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


@st.cache_resource(show_spinner="Đang tải model AV-HuBERT (~1.7 GB, chỉ lần đầu)…")
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
    st.markdown("### Cài đặt")
    audio_only = st.toggle(
        "Chỉ audio",
        value=ON_CLOUD,
        help="Bỏ face tracking — nhanh hơn, ít RAM hơn. Khuyên dùng trên Cloud.",
    )
    fast_mode = st.toggle(
        "Fast mode",
        value=ON_CLOUD and not audio_only,
        disabled=audio_only,
        help="Clip 5 giây, detection nhẹ hơn.",
    )
    clip_seconds = float(
        st.slider(
            "Độ dài clip (giây)",
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
        help="ctc_greedy nhanh hơn; joint_beam_search đôi khi chính xác hơn.",
    )
    st.divider()
    st.caption(
        f"{'☁️ Cloud' if ON_CLOUD else '💻 Local'} · "
        f"scale ≤{max_width}px · clip {clip_seconds:.0f}s"
    )
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        st.warning("Thiếu **HF_TOKEN** trong Secrets — model không tải được từ Hugging Face.")

input_col, preview_col = st.columns([1, 1], gap="large")

with input_col:
    st.markdown("#### Nguồn video")
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
    st.markdown("#### Xem trước")
    if media_path and media_path.is_file():
        duration = _duration_seconds(media_path)
        if duration is not None and duration > clip_seconds:
            st.warning(
                f"File dài {duration:.1f}s — chỉ dùng {clip_seconds:.0f}s đầu."
            )
        st.video(str(media_path))
    else:
        st.markdown(
            '<div class="viavsr-card" style="text-align:center;color:#94a3b8;">'
            "Chọn hoặc quay video để xem trước tại đây."
            "</div>",
            unsafe_allow_html=True,
        )

run = st.button("▶ Chạy nhận dạng", type="primary", disabled=not (media_path and media_path.is_file()))

if run and media_path and media_path.is_file():
    try:
        with st.status("Đang xử lý…", expanded=True) as status:
            status.write("① Tải tokenizer (lần đầu ~300 KB)…")
            _ensure_tokenizer_assets()
            status.write("② Cắt và scale video…")
            prepared = _prepare_media(
                media_path,
                clip_seconds,
                max_width=max_width,
            )
            status.write(
                "③ "
                + (
                    "Tải model (lần đầu ~1.7 GB) + inference audio-only…"
                    if audio_only
                    else "Face tracking + model + inference…"
                )
                + " **Giữ tab mở**, đừng push git."
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
            status.write("④ Hoàn tất.")
            status.update(label="Xong", state="complete")
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

    st.markdown("---")
    st.markdown("#### Kết quả")

    badge_class = "viavsr-badge-ok" if status == "passed" else "viavsr-badge-warn"
    mode = modality.get("selected_mode", "—")
    total_s = timings.get("total", 0)

    st.markdown(
        f'<span class="viavsr-badge {badge_class}">{status.upper()}</span>'
        f'<span class="viavsr-badge viavsr-badge-info">{mode}</span>'
        f'<span class="viavsr-badge viavsr-badge-info">{total_s:.1f}s</span>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="viavsr-transcript">{html.escape(transcript)}</div>',
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Thời gian", f"{total_s:.1f}s")
    metric_cols[1].metric("Modality", mode.replace("_", " "))
    metric_cols[2].metric(
        "Model load",
        f"{timings.get('model_loading', 0):.1f}s",
        help="0s nếu model đã cache từ lần chạy trước.",
    )
    metric_cols[3].metric("Inference", f"{timings.get('inference', 0):.1f}s")

    step_keys = ("face_tracking", "mouth_roi", "model_loading", "inference")
    step_parts = [
        f"{key.replace('_', ' ')}: {timings[key]:.1f}s"
        for key in step_keys
        if key in timings and timings[key]
    ]
    if step_parts:
        st.caption("Chi tiết: " + " · ".join(step_parts))

    mouth = artifacts.get("mouth_roi")
    if mouth and Path(mouth).is_file():
        with st.expander("Video miệng đã xử lý"):
            st.video(mouth)

    warnings = result.get("warnings") or []
    if warnings:
        st.warning("\n".join(str(item) for item in warnings))
    if status != "passed" and result.get("error"):
        st.error(result["error"].get("message", "Inference failed"))

    with st.expander("Báo cáo đầy đủ (JSON)"):
        st.json(result)
