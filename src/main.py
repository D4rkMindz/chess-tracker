# Main call. Calls all other methods as needed.
# Configuration through cli arguments. (Board Type. )
# Run from the repository root: python src/main.py --set crystal
from __future__ import annotations

import argparse

from chess_tracker.config import BOARD_SETS
from chess_tracker.io import set_photos
from chess_tracker.types import BLACK, WHITE, Observation
from chess_tracker.vision import AlignmentError, calibrate, observe

# Photos with fewer RANSAC inliers are flagged as unreliable (see docs/03_method-plan.md, step 3).
MIN_INLIERS = 80


def board_text(obs: Observation) -> str:
    """Return the observed board as 8 text lines, rank 8 at the top: W = white piece, B = black piece, . = empty."""
    symbols = {WHITE: "W", BLACK: "B"}
    return "\n".join(
        f"{rank + 1} " + " ".join(symbols.get(int(c), ".") for c in obs.color[rank]) for rank in range(7, -1, -1)
    ) + "\n  a b c d e f g h"


def main() -> None:
    """Calibrate on photos 0 and 1 of the chosen set, then print the observed board of every photo from 1 on."""
    parser = argparse.ArgumentParser(description="Recognise the board state on the photos of one chess set.")
    parser.add_argument("--set", required=True, choices=sorted(BOARD_SETS), help="folder name in data/chess")
    parser.add_argument("--show", action="store_true", help="show a debug image per photo")
    args = parser.parse_args()

    photos = set_photos(args.set)
    calib = calibrate(args.set, photos[0], photos[1])
    print(f"{args.set}: rotation {calib.rotation * 90}°, colour threshold {calib.color_threshold:.0f}")
    if not calib.a1_dark:
        print("warning: a1 is not a dark square - check the orientation or the board")
    print()

    # TODO (game package): tracker = game.Tracker()  # start position
    # Photo 1 (start position) is included as a check: it must show ranks 1-2 white and 7-8 black.
    for photo in photos[1:]:
        try:
            obs = observe(photo, calib)
        except AlignmentError as err:
            print(f"{photo.name}: skipped, {err}\n")
            continue
        flag = "  UNRELIABLE" if obs.inliers < MIN_INLIERS else ""
        print(f"{photo.name}: {int(obs.occupancy.sum())} pieces, {obs.inliers} inliers{flag}")
        print(board_text(obs) + "\n")
        if args.show:
            # Imported here, so matplotlib only loads when debug images are wanted.
            from chess_tracker.debug import draw_observation, show
            show([draw_observation(obs, calib)], [photo.name], size=6)
        # TODO (game package): tracker.update(obs)  # Observation -> move
    # TODO (game package): tracker.move_table().to_csv("moves.csv")


if __name__ == "__main__":
    main()
