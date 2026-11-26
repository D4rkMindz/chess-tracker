"""Step 2: find the 9 x 9 grid lines of the 64 squares on the warped board."""
from __future__ import annotations

import cv2
import numpy as np


def grid_lines(profile: np.ndarray, n_lines: int = 9) -> np.ndarray:
    """Return the positions of n equally spaced lines whose summed profile values are the largest."""
    size = len(profile)
    best_score, best = -1.0, None
    # Try every square width from size/12 (wide border) to size/7.5 (almost no border), in half-pixel steps.
    for step in np.arange(size / 12, size / 7.5, 0.5):
        # Try every start position where all lines still fit inside the image.
        for offset in np.arange(0, size - (n_lines - 1) * step, 1.0):
            idx = (offset + step * np.arange(n_lines)).astype(int)
            # Score = sum of the profile at the line positions; clip guards against indices past the end.
            score = profile[np.clip(idx, 0, size - 1)].sum()
            if score > best_score:
                best_score, best = score, idx
    return best


def gradient_profiles(board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the absolute mean signed gradient per column (x) and per row (y) of the warped board."""
    # cv2.GaussianBlur: smooths the image with a Gaussian kernel (here 5x5) to suppress noise and fine texture.
    gray = cv2.GaussianBlur(cv2.cvtColor(board, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)

    # cv2.Sobel: computes the image gradient (brightness change); dx=1, dy=0 is the change in x direction.
    # Mean of the signed gradient per column (per row for y), then absolute value:
    # lines whose gradient keeps one sign add up, mixed signs from texture cancel out.
    profile_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3).mean(axis=0))
    profile_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3).mean(axis=1))
    return profile_x, profile_y


def find_grid(board: np.ndarray, edge_margin: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Find the 9 vertical (xs) and 9 horizontal (ys) grid line positions of a warped board."""
    profile_x, profile_y = gradient_profiles(board)
    # The outer board edge gives the strongest gradient but is not a grid line on boards with a border,
    # so the outermost edge_margin share of each profile is ignored.
    margin = int(edge_margin * len(profile_x))
    for profile in (profile_x, profile_y):
        profile[:margin] = 0
        profile[-margin:] = 0
    return grid_lines(profile_x), grid_lines(profile_y)
