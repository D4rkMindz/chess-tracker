"""Steps 5 and 6: occupancy and piece colour per square, from the difference to the empty board.

All 8x8 results here are in image-grid order (row 0 = top row of the warped board). orientation.to_board converts
them to board coordinates.
"""
import cv2
import numpy as np

# Share of the square side that is measured. Tall pieces lean into the neighbouring square, so only the centre counts.
INNER = 0.6
# Weights for the L, a, b channels of the difference. Lightness is weighted down, because shadows and lighting
# changes mostly affect L, while pieces also change the colour channels a and b.
LAB_WEIGHTS = np.array([0.5, 1.0, 1.0], np.float32)


def to_lab(img: np.ndarray) -> np.ndarray:
    """Return the blurred image in Lab colour space as float32 (L = lightness, a and b = colour)."""
    # cv2.GaussianBlur: smooths the image with a Gaussian kernel (here 7x7), which hides small misalignments.
    return cv2.cvtColor(cv2.GaussianBlur(img, (7, 7), 0), cv2.COLOR_BGR2LAB).astype(np.float32)


def difference_image(board: np.ndarray, empty_board: np.ndarray) -> np.ndarray:
    """Return the weighted per-pixel Lab difference between a warped photo and the warped empty board."""
    return (np.abs(to_lab(board) - to_lab(empty_board)) * LAB_WEIGHTS).sum(axis=2)


def changed_mask(diff: np.ndarray) -> np.ndarray:
    """Return a bool mask of the pixels that changed clearly against the empty board."""
    # cv2.normalize: rescales values; NORM_MINMAX stretches them to the full 0..255 range.
    scaled = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # cv2.threshold: sets pixels above a cut-off to white and the rest to black; THRESH_OTSU picks the cut-off itself.
    _, mask = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask > 0


def square_means(values: np.ndarray, xs: np.ndarray, ys: np.ndarray, mask: np.ndarray | None = None,
                 inner: float = INNER) -> np.ndarray:
    """Return the 8x8 mean of values per square centre, optionally only over pixels where mask is True."""
    means = np.full((8, 8), np.nan, np.float32)
    for row in range(8):
        for col in range(8):
            # Cut away (1 - inner) / 2 of the square side on every edge.
            my = (ys[row + 1] - ys[row]) * (1 - inner) / 2
            mx = (xs[col + 1] - xs[col]) * (1 - inner) / 2
            rows = slice(int(ys[row] + my), int(ys[row + 1] - my))
            cols = slice(int(xs[col] + mx), int(xs[col + 1] - mx))
            cell = values[rows, cols]
            if mask is not None:
                cell = cell[mask[rows, cols]]
            # NaN stays when no pixel is selected (e.g. no changed pixels on an empty square).
            if cell.size:
                means[row, col] = cell.mean()
    return means


def otsu_threshold(values: np.ndarray) -> float:
    """Return the Otsu threshold of a 1D sample: the cut that maximises the variance between the two groups. Keyword Histogram."""
    v = np.sort(values[~np.isnan(values)].ravel())
    best_t, best_var = float(v.mean()), -1.0
    # Try a cut between every pair of neighbouring values.
    for i in range(1, len(v)):
        low, high = v[:i], v[i:]
        between_var = len(low) * len(high) * (low.mean() - high.mean()) ** 2
        if between_var > best_var:
            best_t, best_var = float((v[i - 1] + v[i]) / 2), between_var
    return best_t


def occupancy_threshold(scores: np.ndarray) -> float:
    """Return the score above which a square counts as occupied: the Otsu threshold of the log scores."""
    # Occupied scores spread widely (a dark piece on a light square differs much more than on a dark square).
    # Plain Otsu then splits the occupied squares into two groups; on the log scale empty and occupied separate.
    return float(np.exp(otsu_threshold(np.log(np.maximum(scores, 1e-3)))))


def lightness(board: np.ndarray) -> np.ndarray:
    """Return the L channel (lightness 0..255) of the blurred warped board."""
    return to_lab(board)[:, :, 0]
