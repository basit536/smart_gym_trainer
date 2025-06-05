import cv2
import mediapipe as mp
import tempfile
import os
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    pose = mp_pose.Pose()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            frame_keypoints = []
            for lm in results.pose_landmarks.landmark:
                # Extract x, y, z coordinates
                frame_keypoints.append([lm.x, lm.y, lm.z])
            keypoints.append(frame_keypoints)

    cap.release()
    return keypoints


def draw_pose_on_video(input_path, diff_frames=None, threshold=0.1):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create temporary video file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_path = temp_file.name
    temp_file.close()

    out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    pose = mp_pose.Pose()

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            color = (0, 0, 255) if diff_frames and frame_idx in diff_frames else (0, 255, 0)
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
            )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Ensure file writing is complete (especially on Windows)
    time.sleep(0.5)

    # Read video bytes for Streamlit display
    with open(temp_path, "rb") as f:
        video_bytes = f.read()

    # Delete temporary file (optional)
    try:
        os.remove(temp_path)
    except PermissionError:
        pass

    return video_bytes
