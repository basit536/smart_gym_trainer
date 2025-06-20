import json
import os
from pathlib import Path
from utils.pose_estimation import extract_keypoints

# Define only major body joints (exclude face, eyes, ears, nose)
KEYPOINT_INDICES = [
    11, 12,   # shoulders
    13, 14,   # elbows
    15, 16,   # wrists
    23, 24,   # hips
    25, 26,   # knees
    27, 28    # ankles
]

def generate_reference_json(video_path: str, output_path: str):
    keypoints = extract_keypoints(video_path, warmup_trim=60)
    if not keypoints:
        print("❌ No keypoints found.")
        return

    trimmed_keypoints = [[frame[i] for i in KEYPOINT_INDICES] for frame in keypoints]
    data = {
        "keypoints": trimmed_keypoints,
        "num_frames": len(trimmed_keypoints)
    }
    with open(output_path, "w") as f:
        json.dump(data, f)
    print(f"✅ Reference JSON saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="input.mp4", help="Path to reference video")
    parser.add_argument("--out", default="reference.json", help="Path to save reference JSON")
    args = parser.parse_args()

    generate_reference_json(args.video, args.out)
