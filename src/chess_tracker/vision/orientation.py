"""Step 4: find where a1 lies, and convert between the image grid and board coordinates.

Image grid: 8x8 array in the order of the warped board image, row 0 = top row of the image.
Board coordinates: board[rank][file] with a1 = [0][0] (see chess_tracker.types).
"""
from __future__ import annotations

import numpy as np


def find_rotation(piece_brightness: np.ndarray) -> int:
    """Return the quarter turns for np.rot90 that put the white pieces of the start position at the bottom."""
    scores = []
    for k in range(4):
        pieces = np.rot90(piece_brightness, k)
        # In the start position the white pieces fill the bottom two rows, the black pieces the top two.
        scores.append(np.nanmean(pieces[6:]) - np.nanmean(pieces[:2]))
    return int(np.nanargmax(scores))


def a1_is_dark(square_brightness: np.ndarray, rotation: int) -> bool:
    """Check that a1 is darker than its neighbours b1 and a2, as the chess rules require."""
    # Only a plausibility check: the pieces decide the orientation, the square colours confirm it.
    squares = to_board(square_brightness, rotation)
    return bool(squares[0, 0] < min(squares[0, 1], squares[1, 0]))


def to_board(grid: np.ndarray, rotation: int) -> np.ndarray:
    """Convert an 8x8 image-grid array into board coordinates board[rank][file]."""
    # After rot90, a1 is bottom-left (row 7, column 0); flipud moves it to [0][0].
    return np.flipud(np.rot90(grid, rotation)).copy()


def to_image_grid(board: np.ndarray, rotation: int) -> np.ndarray:
    """Convert an 8x8 board-coordinate array back into the image grid of the warped board."""
    return np.rot90(np.flipud(board), -rotation).copy()
