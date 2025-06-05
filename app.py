import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import os

from utils.pose_estimation import extract_keypoints, draw_pose_on_video
from utils.compare import compare_pose
from utils.feedback import generate_feedback

# Keypoint constants
KEYPOINT_NAMES = [
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]

KEYPOINT_INDICES = list(range(len(KEYPOINT_NAMES)))

EXERCISE_PARAMS = {
    "pushup": {"threshold": 0.5, "min_violations": 3},
    "press": {"threshold": 0.3, "min_violations": 3}
}

# --- Streamlit UI ---
st.set_page_config(page_title="Smart Gym Trainer", layout="wide")
st.title("🏋️ Smart Gym Trainer - Posture Correction")

if st.sidebar.button("🔄 Reset Session (Debug)"):
    st.session_state.clear()
    st.experimental_rerun()

video_file = st.file_uploader("Upload your exercise video", type=["mp4", "mpeg4"])
exercise_type = st.selectbox("Choose exercise", list(EXERCISE_PARAMS.keys()))

if video_file and exercise_type:
    ext = os.path.splitext(video_file.name)[1].lower()
    if ext not in [".mp4", ".mpeg4"]:
        st.error(f"❌ Invalid file type: {ext}")
        st.stop()

   
    user_video_path = "uploads/user_video.mp4"
    with open(user_video_path, "wb") as f:
        f.write(video_file.read())
    st.video(user_video_path)

    # Extract keypoints
    user_kps = extract_keypoints(user_video_path)

    # Load reference keypoints
    ref_kps_dict = {
        "pushup": json.load(open("reference_data/pushup_correct.json")),
        "press": json.load(open("reference_data/press_correct.json"))
    }

    # --- Classify uploaded video ---
    def classify_exercise(user_kps, ref_kps_dict, threshold=0.4):
        similarities = {}
        for ex, ref_kps in ref_kps_dict.items():
            min_len = min(len(user_kps), len(ref_kps))
            total_diff, count = 0, 0
            for i in range(min_len):
                u = np.array(user_kps[i])[KEYPOINT_INDICES]
                r = np.array(ref_kps[i])[KEYPOINT_INDICES]
                if u.shape == r.shape:
                    total_diff += np.mean(np.linalg.norm(u - r, axis=1))
                    count += 1
            avg_diff = total_diff / count if count > 0 else float('inf')
            similarities[ex] = avg_diff
        best_match = min(similarities, key=similarities.get)
        return best_match, similarities[best_match]

    best_match, match_score = classify_exercise(user_kps, ref_kps_dict)
    if best_match != exercise_type:
        st.warning(f"⚠️ The uploaded video appears more like a **{best_match}** than a **{exercise_type}**.")

    # Proceed with comparison
    ref_kps = ref_kps_dict[exercise_type]
    # Set optimized threshold for normalized coords
    threshold = 1  # You can try between 0.05 to 0.1 depending on strictness
    min_violations = EXERCISE_PARAMS[exercise_type]["min_violations"]

    # Compare poses
    diff_list, joint_errors = compare_pose(user_kps, ref_kps, KEYPOINT_INDICES, threshold=threshold)

    # Feedback
    feedback = generate_feedback(diff_list, KEYPOINT_NAMES, KEYPOINT_INDICES,
                                 threshold=threshold, min_violations=min_violations)

    if joint_errors:
        joint_errors = np.array(joint_errors)

    feedback = generate_feedback(diff_list, KEYPOINT_NAMES, KEYPOINT_INDICES,
                                  threshold=threshold, min_violations=min_violations)

    # --- Feedback ---
    st.markdown("---")
    st.subheader("📋 Posture Feedback")
    if feedback:
        st.error("❌ Issues Detected:")
        for msg in feedback:
            st.write("- ", msg)
    else:
        st.success("✅ Great form! No corrections needed.")

    # --- Deviation Chart ---
    if joint_errors is not None and joint_errors.size > 0:
        st.markdown("---")
        st.subheader("📊 Average Deviation per Joint")
        avg_errors = np.mean(joint_errors, axis=0)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(KEYPOINT_NAMES, avg_errors, color='tomato')
        ax.set_ylabel("Avg Distance from Reference")
        plt.xticks(rotation=90)
        plt.tight_layout()
        st.pyplot(fig)

       # --- Annotated Video ---
    st.markdown("---")
    st.subheader("📹 Your Exercise (with Pose Overlay)")

    diff_frames = [f[0] for f in diff_list] if diff_list else []

    with st.spinner("Generating annotated video..."):
        # Get either video bytes or temp file path from your function
        video_output = draw_pose_on_video(user_video_path, diff_frames)
        
        # If the function returns a file path
        if isinstance(video_output, str) and os.path.exists(video_output):
            with open(video_output, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
        # If the function returns bytes directly
        elif isinstance(video_output, bytes):
            st.video(video_output)
        else:
            st.error("Failed to generate video - invalid output format")
