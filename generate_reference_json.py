import os
import json
from utils.pose_estimation import extract_keypoints

def generate_reference_json(video_path, exercise_name):
    if not os.path.exists(video_path):
        print(f"❌ Video not found at: {video_path}")
        return

    print(f"📹 Extracting keypoints from: {video_path}")
    keypoints = extract_keypoints(video_path)

    if not keypoints:
        print(f"⚠️ No keypoints extracted for {exercise_name}. Check the video format or content.")
        return

    os.makedirs("reference_data", exist_ok=True)
    output_path = f"reference_data/{exercise_name}_correct.json"
    
    with open(output_path, "w") as f:
        json.dump(keypoints, f)

    print(f"✅ Reference keypoints saved to: {output_path}")


if __name__ == "__main__":
    # ✅ Add all your reference videos here
    reference_videos = {
        "press": "reference_videos/press_correct.mp4",
        "pushup": "reference_videos/pushup_correct.mp4"  # ⬅️ Add pushup video here
    }

    for exercise, path in reference_videos.items():
        generate_reference_json(path, exercise)

