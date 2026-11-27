"""Step 1: find the board outline and warp the board to a top-down square."""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np


def canny_mask(img: np.ndarray) -> np.ndarray:
    """Return a filled mask of the largest outline found by Canny edge detection."""
    # cv2.GaussianBlur: smooths the image with a Gaussian kernel (here 5x5) to suppress noise and fine texture.
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    # cv2.Canny: finds edges; gradients above 150 count as edges, those between 50 and 150 only next to a strong edge.
    # cv2.dilate: grows white areas by the kernel size (here 5x5), which makes the edge lines thicker.
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((5, 5), np.uint8))

    # cv2.findContours: traces the outlines of white areas; RETR_EXTERNAL keeps only the outermost outlines.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(edges.shape, np.uint8)

    # cv2.contourArea: area inside a contour in pixels, used here to pick the largest contour.
    # cv2.drawContours: draws contours into an image; thickness -1 fills the inside instead of drawing the line.
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return mask


def saturation_mask(img: np.ndarray) -> np.ndarray:
    """Return a mask of the grey, low-saturation board area, separated from the colourful table."""
    # Channel 1 of HSV is the saturation: how colourful a pixel is, independent of its brightness.
    sat = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1], (9, 9), 0)

    # cv2.threshold: sets pixels above a cut-off to white and the rest to black; THRESH_OTSU picks the cut-off itself.
    # THRESH_BINARY_INV inverts the result, so the low-saturation board becomes white.
    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # cv2.getStructuringElement: creates the kernel shape for morphology, here a 31x31 ellipse.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))

    # cv2.morphologyEx: runs a morphology operation; OPEN = erode then dilate, CLOSE = dilate then erode.
    # Opening removes table spots, closing fills the dark pieces and green squares.
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)


# Mask functions by the name used in config.BoardSet.mask.
MASKS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "canny": canny_mask,
    "saturation": saturation_mask,
}


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Sort 4 corner points into top-left, top-right, bottom-right, bottom-left in image coordinates."""
    pts = pts.reshape(-1, 2).astype(np.float32)
    # x + y is smallest at the top-left corner and largest at the bottom-right corner.
    # y - x is smallest at the top-right corner and largest at the bottom-left corner.
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)


def board_corners(img: np.ndarray, mask_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Find the 4 outer corners of the board, using the given mask function."""
    contours, _ = cv2.findContours(mask_fn(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.convexHull: smallest convex polygon around a contour, which removes dents (e.g. from pieces on the edge).
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))

    # Simplify the hull with growing tolerance until exactly 4 corners remain.
    # eps is a fraction of the perimeter, so the tolerance scales with the board size.
    for eps in (0.01, 0.02, 0.03, 0.05):
        # cv2.arcLength: perimeter of a contour; True means the contour is closed.
        # cv2.approxPolyDP: simplifies a contour to fewer points, each point may move at most epsilon pixels.
        approx = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(approx) == 4:
            return order_corners(approx)
    # Fallback when no tolerance gives 4 corners: use the smallest rotated rectangle around the hull.
    # cv2.minAreaRect: smallest rotated rectangle that encloses the points, as (centre, size, angle).
    # cv2.boxPoints: converts such a rotated rectangle into its 4 corner points.
    return order_corners(cv2.boxPoints(cv2.minAreaRect(hull)))


def board_transform(corners: np.ndarray, size: int) -> np.ndarray:
    """Return the 3x3 homography that maps the 4 board corners onto the corners of a size x size square."""
    # Target corners in the same order as order_corners: top-left, top-right, bottom-right, bottom-left.
    target = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    # cv2.getPerspectiveTransform: computes the 3x3 homography that maps 4 source points onto 4 target points.
    return cv2.getPerspectiveTransform(corners.astype(np.float32), target)


def warp_board(img: np.ndarray, transform: np.ndarray, size: int) -> np.ndarray:
    """Apply a homography to the image and return the size x size top-down board."""
    # cv2.warpPerspective: applies a 3x3 homography to the whole image; the output size is (width, height).
    return cv2.warpPerspective(img, transform, (size, size))
