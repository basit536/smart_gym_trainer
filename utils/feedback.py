def generate_feedback(diff_list, keypoint_names, keypoint_indices, threshold=1, min_violations=25, verbose=False):
    """
    Generates feedback based on the number of times each joint's deviation exceeds a threshold.

    Parameters:
    - diff_list: list of (frame_index, [joint_diffs]) from compare_pose
    - keypoint_names: list of keypoint names (e.g. ["left_elbow", "right_knee"])
    - keypoint_indices: indices of the keypoints used (should match names)
    - threshold: deviation threshold to count a violation
    - min_violations: minimum number of violating frames before generating feedback
    - verbose: if True, prints debug info per joint (optional)

    Returns:
    - messages: list of feedback strings for joints exceeding min_violations
    """
    joint_violations = [0] * len(keypoint_names)

    for frame_idx, diffs in diff_list:
        for i in range(min(len(diffs), len(joint_violations))):
            if diffs[i] > threshold:
                joint_violations[i] += 1

    messages = []
    for i, count in enumerate(joint_violations):
        if count >= min_violations:
            messages.append(f"{keypoint_names[i]} deviates in {count} frames. Try correcting it.")
            if verbose:
                print(f"🟠 {keypoint_names[i]} exceeded threshold in {count} frames")

    return messages
