"""Step 3: align a handheld photo to the empty-board photo with SIFT and a RANSAC homography."""
import cv2
import numpy as np

# Created on first use, because creating SIFT takes a moment and not every import needs it.
_sift = None
_matcher = None


class AlignmentError(RuntimeError):
    """Raised when a photo cannot be aligned to the reference photo."""


def _detector():
    """Return the shared SIFT detector and brute-force matcher, creating them on first use."""
    global _sift, _matcher
    if _sift is None:
        # cv2.SIFT_create: creates a SIFT detector; it finds keypoints and describes each with 128 numbers
        # that survive scaling and rotation. 4000 = keep only the 4000 strongest keypoints per image.
        _sift = cv2.SIFT_create(4000)
        # cv2.BFMatcher: brute-force matcher that compares every descriptor of one image with every one of the other.
        _matcher = cv2.BFMatcher()
    return _sift, _matcher


def align(reference: np.ndarray, img: np.ndarray, ratio: float = 0.75,
          ransac_px: float = 5.0) -> tuple[np.ndarray, int, int]:
    """Return the homography that maps img onto reference, plus the RANSAC inlier and match counts."""
    sift, matcher = _detector()
    # cv2.cvtColor: converts an image from one colour space to another (here BGR to gray).
    # detectAndCompute: finds the keypoints and computes their descriptors in one call (None = no mask).
    kp_ref, des_ref = sift.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None)
    kp_img, des_img = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    if des_ref is None or des_img is None:
        raise AlignmentError("no SIFT keypoints found")
    # knnMatch: returns the k best matches per descriptor, here the best and the second best (k=2).
    # Ratio test: keep a match only if it is clearly better than the second best, which drops ambiguous matches.
    good = [m for m, n in matcher.knnMatch(des_img, des_ref, k=2) if m.distance < ratio * n.distance]
    # A homography needs at least 4 point pairs.
    if len(good) < 4:
        raise AlignmentError(f"only {len(good)} SIFT matches")
    # Pixel coordinates of the matched keypoints in both images.
    src = np.float32([kp_img[m.queryIdx].pt for m in good])
    dst = np.float32([kp_ref[m.trainIdx].pt for m in good])
    # cv2.findHomography: estimates a 3x3 homography from point pairs; RANSAC ignores pairs that miss by more than
    # ransac_px pixels.
    H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_px)
    if H is None:
        raise AlignmentError("no homography found")
    return H, int(inlier_mask.sum()), len(good)


def rotation_degrees(H: np.ndarray) -> float:
    """Return the rotation in degrees contained in a homography, from its linear part."""
    return float(np.degrees(np.arctan2(H[0, 1], H[0, 0])))
