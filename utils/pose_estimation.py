# ✅ File: pose_estimation.py (Smart rep quality + joint-only + stable overlay)

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Optional, Tuple

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(video_path: str, warmup_trim: int = 15) -> List[List[List[float]]]:
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    keypoints = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            keypoints.append([[p.x, p.y, p.z] for p in results.pose_landmarks.landmark])
    cap.release()
    pose.close()
    return keypoints[warmup_trim:] if len(keypoints) > warmup_trim else keypoints

def detect_exercise_type_with_confidence(keypoints: List[List[List[float]]]) -> Tuple[Optional[str], float]:
    if not keypoints or len(keypoints) < 10:
        return None, 0.0
    kp = np.array(keypoints)
    shoulder = np.mean(np.abs(np.diff(kp[:, 11:13, 1], axis=0)))
    hip = np.mean(np.abs(np.diff(kp[:, 23:25, 1], axis=0)))
    wrist = np.mean(np.abs(np.diff(kp[:, 15:17, 1], axis=0)))
    if hip > 0.2:
        return "squat", 0.9
    elif shoulder > 0.15 and wrist > 0.1:
        return "press", 0.85
    elif hip > 0.1 and shoulder > 0.1:
        return "deadlift", 0.8
    elif shoulder > 0.15 and wrist < 0.03:
        return "pushup", 0.75
    return "unknown", 0.4

def count_reps(keypoints: List[List[List[float]]], exercise: str) -> List[Tuple[int, str]]:
    reps = []
    state = None
    count = 0
    rep_start = None
    kp = np.array(keypoints)
    y_vals = {
        "squat": kp[:, 24, 1],
        "press": kp[:, 15, 1],
        "deadlift": kp[:, 24, 1],
        "pushup": kp[:, 11, 1]
    }.get(exercise, None)
    if y_vals is None:
        return [(0, "")] * len(keypoints)
    y_vals = np.convolve(y_vals, np.ones(3)/3, mode='same')
    threshold_down = np.percentile(y_vals, 70)
    threshold_up = np.percentile(y_vals, 30)
    for i, y in enumerate(y_vals):
        if state is None and y > threshold_down:
            state = "down"
            rep_start = i
        elif state == "down" and y < threshold_up:
            state = "up"
        elif state == "up" and y > threshold_down:
            rep_duration = i - (rep_start or 0)
            rom = np.max(y_vals[rep_start:i+1]) - np.min(y_vals[rep_start:i+1])
            if rom < 0.05:
                quality = "Bad"
            elif rep_duration < 8:
                quality = "Partial"
            else:
                quality = "Good"
            count += 1
            state = "down"
        reps.append((count, quality if count > 0 else ""))
    return reps

def create_annotated_video_opencv(input_path: str, output_path: str, diff_frames=None, reps: Optional[List[Tuple[int, str]]] = None):
    cap = cv2.VideoCapture(input_path)
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )
    pose = mp_pose.Pose()
    frame_id = 0
    diff_frame_ids = set([f[0] for f in diff_frames]) if diff_frames else set()

    valid_joints = {11,12,13,14,15,16,23,24,25,26,27,28}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w = frame.shape[:2]
            color = (0, 0, 255) if frame_id in diff_frame_ids else (0, 255, 0)

            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx in valid_joints and end_idx in valid_joints:
                    start = landmarks[start_idx]
                    end = landmarks[end_idx]
                    start_coords = int(start.x * w), int(start.y * h)
                    end_coords = int(end.x * w), int(end.y * h)
                    cv2.line(frame, start_coords, end_coords, color, 2)
                    cv2.circle(frame, start_coords, 4, color, -1)
                    cv2.circle(frame, end_coords, 4, color, -1)

            if frame_id in diff_frame_ids:
                cv2.putText(frame, "FormError", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if reps and frame_id < len(reps):
                rep_num, quality = reps[frame_id]
                text_color = (0, 255, 0) if quality == "Good" else (0, 165, 255) if quality == "Partial" else (0, 0, 255)
                cv2.putText(frame, f"Rep {rep_num}: {quality}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, text_color, 3)

        out.write(frame)
        frame_id += 1
    cap.release()
    out.release()
    pose.close()
    return output_path

