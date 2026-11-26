"""Steps 1 to 6 combined: calibrate once per set, then observe every photo."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from chess_tracker.config import BOARD_SETS, BoardSet, settings
from chess_tracker.io import load
from chess_tracker.types import BLACK, EMPTY, WHITE, Calibration, Observation
from chess_tracker.vision.align import align
from chess_tracker.vision.board import MASKS, board_corners, board_transform, order_corners, warp_board
from chess_tracker.vision.grid import find_grid
from chess_tracker.vision.orientation import a1_is_dark, find_rotation, to_board
from chess_tracker.vision.squares import (changed_mask, difference_image, lightness, occupancy_threshold,
                                          square_means)


def calibrate(set_name: str, empty_path: Path, start_path: Path, board_set: BoardSet | None = None) -> Calibration:
    """Compute board outline, grid, orientation and colour threshold from the empty and the start photo of a set."""
    board_set = board_set or BOARD_SETS[set_name]
    size = settings.warp_size
    mask_fn = MASKS[board_set.mask]
    empty, start = load(empty_path), load(start_path)

    # Step 3 for the start photo: map it onto the empty photo.
    H_start, _, _ = align(empty, start)

    # Step 1: board outline in empty-photo coordinates.
    if board_set.outline_from == "start":
        # Find the outline on the start photo, then move the corners into empty-photo coordinates.
        # cv2.perspectiveTransform: applies a homography to a list of points (not to an image).
        corners = cv2.perspectiveTransform(board_corners(start, mask_fn).reshape(-1, 1, 2), H_start)
        corners = order_corners(corners)
    else:
        corners = board_corners(empty, mask_fn)
    transform = board_transform(corners, size)
    empty_board = warp_board(empty, transform, size)
    # One combined homography (start -> empty -> board), so the start photo is resampled only once.
    start_board = warp_board(start, transform @ H_start, size)

    # Step 2: grid lines on the photo the outline came from.
    xs, ys = find_grid(start_board if board_set.outline_from == "start" else empty_board)

    # Step 4: orientation from the piece colours of the start photo, checked against the square colours.
    changed = changed_mask(difference_image(start_board, empty_board))
    piece_brightness = square_means(lightness(start_board), xs, ys, mask=changed)
    rotation = find_rotation(piece_brightness)
    a1_dark = a1_is_dark(square_means(lightness(empty_board), xs, ys), rotation)

    # Step 6 threshold: halfway between the mean brightness of the white (ranks 1-2) and black (ranks 7-8) pieces.
    pieces = to_board(piece_brightness, rotation)
    color_threshold = float((np.nanmean(pieces[:2]) + np.nanmean(pieces[6:])) / 2)

    return Calibration(set_name=set_name, empty=empty, board_transform=transform, empty_board=empty_board,
                       xs=xs, ys=ys, rotation=rotation, a1_dark=a1_dark, color_threshold=color_threshold)


def observe(path: Path, calib: Calibration) -> Observation:
    """Align one photo to the empty board and return occupancy and piece colour of all 64 squares."""
    img = load(path)
    # Step 3: align to the empty photo, then warp to the top-down board in one step.
    H, inliers, _ = align(calib.empty, img)
    board = warp_board(img, calib.board_transform @ H, calib.empty_board.shape[0])

    # Step 5: occupancy = squares whose mean difference to the empty board is above the threshold of this photo.
    diff = difference_image(board, calib.empty_board)
    score = square_means(diff, calib.xs, calib.ys)
    occupied = score > occupancy_threshold(score)

    # Step 6: colour = mean lightness of the changed pixels on an occupied square, against the start-photo threshold.
    brightness = square_means(lightness(board), calib.xs, calib.ys, mask=changed_mask(diff))
    color = np.where(occupied, np.where(brightness > calib.color_threshold, WHITE, BLACK), EMPTY).astype(np.int8)

    return Observation(photo=Path(path), occupancy=to_board(occupied, calib.rotation),
                       color=to_board(color, calib.rotation), score=to_board(score, calib.rotation),
                       inliers=inliers, board=board)
