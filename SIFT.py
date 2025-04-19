import cv2
import numpy as np
import time


def detect_and_match_features(query_img_path, target_img_path):
    """
    Detect and match features between query and target images using SIFT.
    Returns matched keypoints image and homography matrix.
    """
    # Read images
    query_img = cv2.imread(query_img_path, cv2.IMREAD_GRAYSCALE)
    target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)

    if query_img is None or target_img is None:
        raise ValueError("Could not load one or both images.")

    # Initialize SIFT detector and FLANN matcher
    detector = cv2.SIFT_create()
    matcher = cv2.FlannBasedMatcher({"algorithm": 0, "trees": 5}, {"checks": 50})

    # Detect keypoints and compute descriptors
    start_time = time.time()
    kp1, des1 = detector.detectAndCompute(query_img, None)
    kp2, des2 = detector.detectAndCompute(target_img, None)
    detection_time = time.time() - start_time

    if des1 is None or des2 is None:
        return None, None, 0, 0, detection_time

    # Match descriptors
    start_time = time.time()
    matches = matcher.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    matching_time = time.time() - start_time

    # Draw matches
    match_img = cv2.drawMatches(
        query_img,
        kp1,
        target_img,
        kp2,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Find homography if enough matches
    H = None
    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return match_img, H, len(kp1), len(good_matches), detection_time + matching_time


def main():
    query_img_path = "query.jpg"  # Image of object alone
    target_img_path = "target.jpg"  # Image containing object among others

    print("Running SIFT for image matching...")
    sift_img, sift_H, sift_kp_count, sift_match_count, sift_time = (
        detect_and_match_features(query_img_path, target_img_path)
    )

    if sift_img is not None:
        cv2.imwrite("output/sift_matches.jpg", sift_img)
        print(
            f"SIFT: {sift_kp_count} keypoints, {sift_match_count} good matches, {sift_time:.3f} seconds"
        )


if __name__ == "__main__":
    main()
