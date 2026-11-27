"""Loading the photos of a chess set."""
from pathlib import Path

import cv2
import numpy as np

from chess_tracker.config import settings

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def photo_paths(folder: Path) -> list[Path]:
    """Return the photos of one set in sequence order (0 = empty board, 1 = start position, ...)."""
    def key(p: Path) -> tuple[int, str]:
        """Turn a file name like "16-obstructed.jpeg" into the sort key (16, "obstructed"), so 10 sorts after 9."""
        number, _, suffix = p.stem.partition("-")
        return int(number), suffix
    return sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES), key=key)


def set_photos(set_name: str) -> list[Path]:
    """Return the photos of the set with the given folder name in sequence order."""
    return photo_paths(settings.data_dir / set_name)


def load(path: Path, work_size: int | None = None) -> np.ndarray:
    """Read an image from disk and scale it so its longest side is work_size pixels."""
    work_size = work_size or settings.work_size
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    scale = work_size / max(img.shape[:2])
    # cv2.resize: scales an image; INTER_AREA averages neighbouring pixels, which looks cleanest when shrinking.
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
