import streamlit as st
from pathlib import Path

from viavsr.demo import run_end_to_end_demo


st.set_page_config(
    page_title="viAVSR",
    page_icon="🎙️",
)

st.title("Vietnamese Audio-Visual Speech Recognition")


# ============================================================
# INPUT
# ============================================================

st.header("Input")

uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "webm", "mov"],
)

if uploaded_file:

    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    video_path = upload_dir / uploaded_file.name

    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(str(video_path))


# ============================================================
# RUN
# ============================================================

if uploaded_file:

    if st.button("Run inference", type="primary"):

        with st.spinner("Processing video..."):

            result = run_end_to_end_demo(
                config_path="configs/config.yaml",
                media_path=video_path,
                output_root="outputs/inference",
                tracking_device="auto",
                decoder="joint_beam_search",
                beam_size=3,
                ctc_weight=0.1,
                visual_fallback_policy="whole_utterance",
            )

        st.session_state.result = result


# ============================================================
# RESULT
# ============================================================

if "result" in st.session_state:

    result = st.session_state.result

    st.header("Result")

    st.subheader("Transcript")

    st.write(
        result
        .get("result", {})
        .get("transcript", "")
    )

    st.subheader("Details")

    st.json(result)
