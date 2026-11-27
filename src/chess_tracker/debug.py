"""Debug images: board outline, grid and the recognised pieces."""
from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from chess_tracker.types import BLACK, WHITE, Calibration, Observation
from chess_tracker.vision.orientation import to_image_grid


def show(images: list[np.ndarray], titles: list[str] | None = None, size: int = 4) -> None:
    """Plot several images side by side in one matplotlib row."""
    fig, axes = plt.subplots(1, len(images), figsize=(size * len(images), size))
    # With a single image, subplots returns one axis instead of an array; atleast_1d makes it iterable.
    axes = np.atleast_1d(axes)
    for i, (ax, img) in enumerate(zip(axes, images)):
        # Grayscale images (2D) are shown as they are, colour images are converted from BGR to RGB for matplotlib.
        # cv2.cvtColor: converts an image from one colour space to another (BGR, RGB, gray, HSV, ...).
        ax.imshow(img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap="gray")
        ax.set_title(titles[i] if titles else "")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def draw_outline(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Return a copy of the image with the board outline drawn in red."""
    out = img.copy()
    # cv2.polylines: draws lines through the points; True closes the shape. (0, 0, 255) is red in BGR.
    cv2.polylines(out, [corners.astype(int)], True, (0, 0, 255), 4)
    return out


def draw_grid(board: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Return a copy of the warped board with the grid lines drawn in red."""
    out = board.copy()
    for x in xs:
        # cv2.line: draws a straight line between two points with the given colour and thickness.
        cv2.line(out, (int(x), 0), (int(x), out.shape[0]), (0, 0, 255), 2)
    for y in ys:
        cv2.line(out, (0, int(y)), (out.shape[1], int(y)), (0, 0, 255), 2)
    return out


def draw_observation(obs: Observation, calib: Calibration) -> np.ndarray:
    """Return the warped board with grid, a1 marker and one circle per recognised piece (green = white, red = black)."""
    out = draw_grid(obs.board, calib.xs, calib.ys)
    color = to_image_grid(obs.color, calib.rotation)
    # Mark a1 in board coordinates, then find where it lies on the warped image.
    a1 = np.zeros((8, 8), bool)
    a1[0, 0] = True
    a1 = to_image_grid(a1, calib.rotation)
    for row in range(8):
        for col in range(8):
            center = (int((calib.xs[col] + calib.xs[col + 1]) / 2), int((calib.ys[row] + calib.ys[row + 1]) / 2))
            radius = int((calib.xs[col + 1] - calib.xs[col]) / 4)
            if color[row, col] == WHITE:
                # cv2.circle: draws a circle with the given centre, radius, colour and line thickness.
                cv2.circle(out, center, radius, (0, 255, 0), 3)
            elif color[row, col] == BLACK:
                cv2.circle(out, center, radius, (0, 0, 255), 3)
            if a1[row, col]:
                # cv2.putText: writes text at a position, here the label of square a1.
                cv2.putText(out, "a1", (center[0] - radius, center[1] + radius), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 0, 0), 2)
    return out
