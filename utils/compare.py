import numpy as np

def compare_pose(user_kps, reference_kps, keypoint_indices, threshold=0.9):
    """
    Compares user's pose with reference and returns deviations per joint.

    Parameters:
    - threshold: Recommended range 0.05 to 0.15 (as MediaPipe coords are normalized)
    """
    feedback = []
    joint_errors = []

    num_frames = min(len(user_kps), len(reference_kps))

    for frame_idx in range(num_frames):
        user_frame = np.array(user_kps[frame_idx])
        ref_frame = np.array(reference_kps[frame_idx])

        if user_frame.shape != ref_frame.shape:
            continue

        user_selected = user_frame[keypoint_indices]
        ref_selected = ref_frame[keypoint_indices]

        diff = np.linalg.norm(user_selected - ref_selected, axis=1)
        joint_errors.append(diff.tolist())

        if np.any(diff > threshold):
            feedback.append((frame_idx, diff.tolist()))

    return feedback, joint_errors
