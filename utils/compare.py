# ✅ File: compare.py (Body-normalized, speed-tolerant, clean joints)

import numpy as np
from typing import List, Tuple

def normalize_pose_proportional(pose: List[List[float]]) -> List[List[float]]:
    pose = np.array(pose)

    def dist(a, b):
        return np.linalg.norm(pose[a][:2] - pose[b][:2])

    shoulder_width = dist(11, 12)
    hip_width = dist(23, 24)
    torso = dist(11, 23)
    leg_length = dist(23, 27)

    scale = np.mean([shoulder_width, hip_width, torso, leg_length]) + 1e-5
    return (pose / scale).tolist()

def compare_pose(user_kps: List, ref_kps: List, indices: List[int], base_thresh: float, exercise: str, window: int = 4) -> Tuple[List, List, float]:
    user_norm = [normalize_pose_proportional(f) for f in user_kps]
    ref_norm = [normalize_pose_proportional(f) for f in ref_kps]

    user = np.array(user_norm)[:, indices]
    ref = np.array(ref_norm)[:, indices]

    min_len = min(len(user), len(ref))
    deviations = []
    joint_errors = []
    early_ignore = 50

    for i in range(min_len):
        best_error = float('inf')
        best_diff = None
        best_joints = None

        for offset in range(-window, window + 1):
            j = i + offset
            if 0 <= j < len(ref):
                diff = np.linalg.norm(user[i] - ref[j], axis=1)
                avg_error = np.mean(diff)
                if avg_error < best_error:
                    best_error = avg_error
                    best_diff = (i, avg_error)
                    best_joints = diff

        joint_errors.append(best_joints if best_joints is not None else np.zeros(len(indices)))

        if i < early_ignore:
            continue
        if best_error < 0.01:
            continue
        if best_error > base_thresh:
            deviations.append((i, best_error, "high" if best_error > base_thresh * 1.5 else "medium"))

    return deviations, joint_errors, base_thresh
