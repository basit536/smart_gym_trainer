import streamlit as st
import numpy as np
import json
import time
import os
import random
from pathlib import Path
from utils.pose_estimation import extract_keypoints, create_annotated_video_opencv, detect_exercise_type_with_confidence, count_reps
from utils.compare import compare_pose
from utils.feedback import FeedbackGenerator
import plotly.graph_objects as go
from PIL import Image

# ------------------------- CONFIG -------------------------
CONFIG = {
    "reference_dir": Path("reference_data"),
    "uploads_dir": Path("uploads"),
    "temp_dir": Path("temp"),
    "keypoint_indices": [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
    "exercise_thresholds": {
        "squat": {"base": 0.15, "min_ratio": 0.3},
        "press": {"base": 0.15, "min_ratio": 0.06},
        "deadlift": {"base": 0.6, "min_ratio": 0.5},
        "pushup": {"base": 0.5, "min_ratio": 0.6}
    }
}

for d in [CONFIG["uploads_dir"], CONFIG["temp_dir"], CONFIG["reference_dir"]]:
    d.mkdir(exist_ok=True)

quotes = [
    "Be hard when it gets hard.",
    "Discipline equals freedom.",
    "Everybody pities the weak; jealousy you have to earn.",
    "You will not always be motivated. You must learn to be disciplined.",
    "If you want to be a champion, you cannot have any kind of an outside negative influence."
]
highlight_words = ["hard", "freedom", "disciplined", "champion", "motivated", "jealousy"]

bg_image_url = "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
logo_path = "logo.jpg"

# ------------------------- CSS THEME -------------------------
def apply_theme():
    css = """
    <style>
        html, body, .stApp {
            font-family: 'Segoe UI', sans-serif;
            background: url("https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D") no-repeat center center fixed;
            background-size: cover;
            color: #ffffff;
        }

        .block-container {
            background-color: rgba(0, 0, 0, 0.72);
            padding: 2rem 2.5rem;
            border-radius: 16px;
            animation: fadeSlide 1s ease-out;
        }

        @keyframes fadeSlide {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        h1, h2, h3, p, label, .stMarkdown {
            color: white !important;
            text-shadow: 1px 1px 6px rgba(0, 0, 0, 0.8);
        }

        .hero-quote {
            text-align: center;
            margin-top: 1rem;
            margin-bottom: 2.5rem;
        }

        .hero-quote h2 {
            font-size: 2.3rem;
            font-weight: 700;
            color: #f9d86b;
            text-shadow:
                0 0 6px rgba(255, 215, 0, 0.4),
                0 0 12px rgba(255, 215, 0, 0.6),
                0 0 24px rgba(255, 215, 0, 0.7);
            animation: fadeGlow 5s ease-in-out infinite alternate;
            transition: all 0.4s ease-in-out;
        }

        @keyframes fadeGlow {
            0% {
                text-shadow:
                    0 0 6px rgba(255, 215, 0, 0.3),
                    0 0 12px rgba(255, 215, 0, 0.5);
            }
            100% {
                text-shadow:
                    0 0 12px rgba(255, 215, 0, 0.6),
                    0 0 28px rgba(255, 215, 0, 0.9);
            }
        }

        .glow-word {
            display: inline-block;
            color: #f9d86b;
            font-weight: bold;
            text-shadow:
                0 0 4px rgba(255, 215, 0, 0.5),
                0 0 8px rgba(255, 215, 0, 0.6),
                0 0 14px rgba(255, 215, 0, 0.7);
            transition: all 0.3s ease-in-out;
        }

        .glow-word:hover {
            animation: bounceGlow 1.5s infinite;
        }

        @keyframes bounceGlow {
            0%   { transform: scale(1); text-shadow: 0 0 6px rgba(255, 215, 0, 0.4); }
            50%  { transform: scale(1.15); text-shadow: 0 0 18px rgba(255, 215, 0, 1); }
            100% { transform: scale(1); text-shadow: 0 0 6px rgba(255, 215, 0, 0.4); }
        }

        .top-logo {
            display: flex;
            justify-content: center;
            margin-top: -1rem;
            margin-bottom: 0.5rem;
        }

        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            border: none;
            animation: pulseGlow 2s infinite;
        }

        .stButton>button:hover {
            background-color: #e63636;
            transform: scale(1.05);
            box-shadow: 0 0 12px rgba(255, 75, 75, 0.6);
        }

        @keyframes pulseGlow {
            0% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.3); }
            70% { box-shadow: 0 0 10px 10px rgba(255, 75, 75, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
        }

        /* Circle loader */
        .loader {
            border: 6px solid rgba(255, 255, 255, 0.15);
            border-top: 6px solid #FFD700;
            border-radius: 50%;
            width: 48px;
            height: 48px;
            animation: spin 1s linear infinite;
            margin: 1.2rem auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Dropdown styling and animated arrow */
        .stSelectbox > div {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 0.4rem 0.8rem;
            color: #fff;
            transition: all 0.3s ease-in-out;
            backdrop-filter: blur(10px);
        }

        .stSelectbox > div:hover {
            background-color: rgba(255, 255, 255, 0.08);
            transform: scale(1.02);
        }

        .stSelectbox > div:focus-within {
            box-shadow: 0 0 8px rgba(255, 215, 0, 0.6);
        }

        .stSelectbox > div::after {
            content: "⌄";
            float: right;
            margin-left: 10px;
            transition: transform 0.3s ease;
            color: #f9d86b;
            font-size: 1.1rem;
        }

        .stSelectbox > div:focus-within::after {
            transform: rotate(180deg);
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ------------------------- SIDEBAR -------------------------
def render_sidebar():
    with st.sidebar:
        img = Image.open(logo_path)
        st.image(img, width=100)
        st.markdown("<h2 style='color:#ff6347;'>🎛️ Controls</h2>", unsafe_allow_html=True)

        if st.button("🔄 Reset"):
            for k in st.session_state.keys():
                del st.session_state[k]
            st.write("🔄 Resetting...")
            st.stop()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#1f77b4;'>✅ Supported Exercises</h4>", unsafe_allow_html=True)
        for ex in CONFIG["exercise_thresholds"]:
            status = "✅" if (CONFIG["reference_dir"] / f"{ex}_correct.json").exists() else "❌"
            color = "green" if status == "✅" else "red"
            st.markdown(f"<span style='color:{color}'>• {ex.title()} {status}</span>", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#f8c518;'>📌 Rules</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Upload clear front-facing videos  
        - Select the correct exercise  
        - Make sure full body is visible  
        - Use a stable camera angle  
        - Max upload size: 200MB  
        """)

# ------------------------- MAIN -------------------------
def main():
    st.set_page_config(page_title="Smart Form Coach", layout="wide")
    render_sidebar()
    apply_theme()

    st.markdown('<div class="top-logo">', unsafe_allow_html=True)
    img = Image.open(logo_path)
    st.image(img, width=130)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<h1 style='text-align:center;'>💪 Smart Form Coach</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>AI-powered posture correction and rep analysis for gym workouts</p>", unsafe_allow_html=True)

    # QUOTES: 1-min switch + animated hover
    quote_slot = st.empty()
    if "quote_index" not in st.session_state:
        st.session_state["quote_index"] = random.randint(0, len(quotes) - 1)
        st.session_state["quote_time"] = time.time()

    if time.time() - st.session_state["quote_time"] > 60:
        st.session_state["quote_index"] = (st.session_state["quote_index"] + 1) % len(quotes)
        st.session_state["quote_time"] = time.time()

    quote = quotes[st.session_state["quote_index"]]
    for word in highlight_words:
        quote = quote.replace(word, f"<span class='glow-word'>{word}</span>")

    quote_slot.markdown(f"""
    <div class='hero-quote'>
        <h2>“{quote}”</h2>
    </div>
    """, unsafe_allow_html=True)


    # VIDEO + ANALYSIS
    uploaded_file = st.file_uploader("📤 Upload Your Workout Video", type=["mp4", "mov", "avi"])
    selected_exercise = st.selectbox("🏋️ Select Exercise Type", list(CONFIG["exercise_thresholds"].keys()))

    if uploaded_file:
        temp_path = CONFIG["uploads_dir"] / f"temp_{int(time.time())}.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(str(temp_path))

        if st.button("🚀 Analyze Form"):
            loader_area = st.empty()
            loader_area.markdown('<div class="loader"></div><p style="text-align:center;">Analyzing your workout... 🔍</p>', unsafe_allow_html=True)

            keypoints = extract_keypoints(str(temp_path), warmup_trim=60)
            loader_area.empty()

            if not keypoints:
                st.error("❌ Failed to extract keypoints.")
                return

            detected_type, confidence = detect_exercise_type_with_confidence(keypoints)
            mismatch = detected_type != selected_exercise and confidence > 0.6

            ref_path = CONFIG["reference_dir"] / f"{selected_exercise}_correct.json"
            if not ref_path.exists():
                st.error("❌ Reference data missing.")
                return

            with open(ref_path) as f:
                ref_data = json.load(f)

            diff_frames, joint_errors, threshold = compare_pose(
                keypoints,
                ref_data["keypoints"],
                CONFIG["keypoint_indices"],
                CONFIG["exercise_thresholds"][selected_exercise]["base"],
                selected_exercise
            )

            if len(diff_frames) <= 2 and confidence > 0.95:
                diff_frames = []

            feedback = FeedbackGenerator().generate_feedback(
                diff_frames,
                len(keypoints),
                CONFIG["exercise_thresholds"][selected_exercise]["min_ratio"],
                joint_errors,
                mismatch,
                detected_type
            )

            reps = count_reps(keypoints, selected_exercise)

            st.markdown("### 📊 Feedback Summary")
            grade = feedback['performance_grade']
            grade_color = {
                "A": "🟢", "B+": "🟡", "C": "🟠", "D": "🔴", "F": "❌"
            }.get(grade, "❓")
            st.markdown(f"### Performance Grade: {grade_color} `{grade}`")

            if grade in ["A", "B+"]:
                st.balloons()

            for item in feedback["priority_feedback"]:
                st.markdown(f"- **{item['message']}**")

            from utils.feedback import JOINT_NAMES

            if diff_frames:
                with st.expander("📋 Frame Deviation Log", expanded=False):
                    scroll_log = ""
                    for i, (frame_idx, total_error, _) in enumerate(diff_frames):
                        if frame_idx < len(joint_errors):
                            joint_error_arr = joint_errors[frame_idx]
                            max_error_idx = int(np.argmax(joint_error_arr))
                            joint_name = JOINT_NAMES[max_error_idx] if max_error_idx < len(JOINT_NAMES) else f"joint {max_error_idx}"
                            max_error = joint_error_arr[max_error_idx]
                            scroll_log += f"Frame {{{frame_idx}}} -- MAX Error at {joint_name} = {max_error:.2f} -- Total Error = {total_error:.2f}\n"
                    st.code(scroll_log.strip(), language="markdown")


                frame_ids = [f[0] for f in diff_frames]
                errors = [f[1] for f in diff_frames]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=frame_ids, y=errors, mode='lines+markers',
                                         marker=dict(color='crimson')))
                fig.update_layout(title='📉 Frame-wise Form Deviation',
                                  xaxis_title='Frame Index',
                                  yaxis_title='Deviation Score')
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🎬 Annotated Video")
            out_path = CONFIG["temp_dir"] / f"annotated_{int(time.time())}.mp4"
            result = create_annotated_video_opencv(str(temp_path), str(out_path), diff_frames, reps)
            if result and os.path.exists(out_path):
                st.video(str(out_path))
                with open(out_path, "rb") as f:
                    st.download_button("⬇️ Download Annotated Video", f.read(), file_name="annotated_output.mp4", mime="video/mp4")
            else:
                st.warning("⚠️ Annotated video could not be generated.")

if __name__ == "__main__":
    main()
