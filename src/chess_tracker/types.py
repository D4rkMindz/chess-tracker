"""Data passed between the vision and the game package.

All 8x8 board arrays use the index convention board[rank][file] with a1 = [0][0], h1 = [0][7], a8 = [7][0].
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Values in Observation.color.
EMPTY, WHITE, BLACK = 0, 1, -1

LETTERS = "abcdefgh"


def square_name(rank: int, file: int) -> str:
    """Return the chess name of a board index, e.g. (0, 0) -> "a1"."""
    return f"{LETTERS[file]}{rank + 1}"


@dataclass(frozen=True)
class Calibration:
    """Everything computed once per set from the empty-board photo and the start-position photo."""
    set_name: str
    # Empty-board photo at work size. Every later photo is aligned to it.
    empty: np.ndarray
    # 3x3 homography from empty-photo coordinates to the warped top-down board.
    board_transform: np.ndarray
    # Warped empty board (warp_size x warp_size), the reference for the difference images.
    empty_board: np.ndarray
    # 9 x and 9 y positions of the grid lines on the warped board, in pixels.
    xs: np.ndarray
    ys: np.ndarray
    # Quarter turns for np.rot90 that bring a1 to the bottom-left corner of the warped board.
    rotation: int
    # Whether a1 is a dark square on this board, as it should be. False points to a wrong orientation or board.
    a1_dark: bool
    # Brightness between the white and the black pieces of the start position.
    color_threshold: float


@dataclass(frozen=True)
class Observation:
    """What the vision package sees on one photo: the input for the game tracker."""
    photo: Path
    # 8x8 bool, True where a piece stands.
    occupancy: np.ndarray
    # 8x8 int8 with EMPTY, WHITE or BLACK.
    color: np.ndarray
    # 8x8 float, mean difference to the empty board per square. Occupancy is decided on this value.
    score: np.ndarray
    # RANSAC inliers of the alignment to the empty board. Low values mean an unreliable photo.
    inliers: int
    # Warped top-down board of this photo, for debug images. Not rotated to board coordinates.
    board: np.ndarray
