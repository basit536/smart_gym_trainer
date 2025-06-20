# ✅ File: feedback.py (with joint focus)

from typing import List, Optional, Dict
import numpy as np

JOINT_NAMES = [
    "nose", "left eye", "right eye", "left ear", "right ear",
    "left shoulder", "right shoulder", "left elbow", "right elbow",
    "left wrist", "right wrist", "left hip", "right hip",
    "left knee", "right knee", "left ankle", "right ankle"
]

class FeedbackGenerator:
    def generate_feedback(self, diff_frames: List[tuple], total_frames: int,
                          min_ratio: float, joint_errors: List[np.ndarray],
                          mismatch: bool = False, detected_type: Optional[str] = None) -> Dict:
        if mismatch:
            return {
                "priority_feedback": [{
                    "message": f"⚠️ This exercise resembles **{detected_type.title()}** more than **the selected type**.",
                    "severity": "high",
                    "error_magnitude": 1.0
                }],
                "performance_grade": "F"
            }

        error_rate = len(diff_frames) / max(1, total_frames)
        grade = "A" if error_rate < 0.05 else "B+" if error_rate < 0.15 else "C" if error_rate < 0.25 else "D"

        feedback = []
        if diff_frames:
            joint_array = np.array(joint_errors)
            if joint_array.shape[0] > 5:
                joint_deviation = np.mean(joint_array, axis=0)
                top_joints = np.argsort(joint_deviation)[-3:][::-1]
                joint_names = [JOINT_NAMES[i] if i < len(JOINT_NAMES) else f'joint {i}' for i in top_joints]
                joint_text = ', '.join(joint_names)
                feedback.append({
                    "message": f"👁️ Most affected joints: **{joint_text}**",
                    "severity": "medium",
                    "error_magnitude": float(np.mean(joint_deviation))
                })

        #for f in diff_frames:
            #feedback.append({
                #"message": f"Form deviation detected at frame {f[0]} (error = {f[1]:.2f})",
                #"severity": f[2],
                #"error_magnitude": f[1]
            #})

        return {
            "priority_feedback": feedback,
            "performance_grade": grade
        }
