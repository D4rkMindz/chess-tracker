# Method Plan: Classical CV Pipeline

This plan lists one primary method and one fallback per pipeline step, from the raw photo to a labelled board. Each method comes from the lecture topics listed in [`01_brainstorming.md`](01_brainstorming.md): color spaces/HSV, point operators and thresholds, morphology, smoothing, convolution/Sobel, Canny, Hough, SIFT matching.

Steps 1 to 3 were prototyped on our photos (23.11.2026, OpenCV 4, images downscaled to 1000 px on the long side). The results below come from those runs. Steps 4 to 8 are untested proposals.

## Overview

| # | Step | Primary method | Fallback | Status |
|---|---|---|---|---|
| 0 | Preprocess | Downscale to ~1000 px, Gaussian blur | | trivial |
| 1 | Board outline | Canny + largest contour (wood), HSV saturation + Otsu + morphology (crystal) | Grayscale Otsu | tested, works |
| 2 | 64 squares | Warp to square, Sobel projection profile | Canny + Hough lines; `findChessboardCornersSB` | tested, 2 of 3 sets |
| 3 | Align each photo | SIFT + ratio test + RANSAC homography to the empty board | Repeat step 1 per photo | tested, works on all photos |
| 4 | Orientation (A1) | Square color + side of the white pieces in the start photo | | proposal |
| 5 | Occupancy | Per-square difference to the empty board + Otsu over the 64 values | Edge density per square | proposal |
| 6 | Piece color | Brightness of the changed pixels, 2-cluster threshold | | proposal |
| 7 | Piece type | Move logic from the known start position | Shape features + kNN on start-photo examples | proposal |
| 8 | Output | 8×8 labels / FEN, move table | | proposal |

## 0. Preprocess

The photos come in three sizes (crystal 3060×4080, wood-small 2252×4000, wood-large 1280×960). Downscale all to ~1000 px on the long side before any processing. That equalises the kernel sizes and thresholds across sets and makes SIFT fast. Keep the scale factor if full-resolution crops are needed later.

## 1. Restrict the image to the board

Goal: four corner points of the board, then `cv2.warpPerspective` to a square image (e.g. 800×800).

**A. Canny + largest contour** (lecture: smoothing, Canny, contours)
Grayscale → `GaussianBlur(5×5)` → `Canny(50, 150)` → `dilate(5×5)` to close gaps → `findContours(RETR_EXTERNAL)` → largest contour → `convexHull` → `approxPolyDP` until 4 points remain.

- wood-large: correct outline on all tested photos (white table, dark board edge).
- wood-small: correct on photos 1 to 3 (light board on dark table).
- crystal: fails. The wood grain of the table produces as many edges as the board.

**B. HSV saturation threshold + morphology** (lecture: color spaces, Otsu, opening/closing)
BGR → HSV → S channel → blur → `threshold(OTSU, inverted)` → opening then closing with a ~31 px ellipse → largest contour → 4 corners as above.

- crystal: correct outline on all tested photos. The grey marble has low saturation, the orange table high saturation.
- This is the method for any board that differs from its background in color more than in brightness.

**C. Grayscale Otsu** works on wood-small (bright board on dark table) but picks up the white table on wood-large. Keep it as a fallback only.

**Choosing per set:** either fix the method per set in a small config, or run A and B and keep the quad that is closest to a square and covers the most area. The config is simpler and honest for three sets.

The outline found here is the board's **outer edge, including the border with the letters and numbers**. Step 2 finds the playing area inside it.

## 2. Find the 64 squares

Goal: 9×9 grid points in the coordinates of the empty-board photo.

**A. Sobel projection profile** (lecture: convolution, Sobel)
On the warped board: `Sobel` in x and y → take the mean of the **signed** gradient along each column (for x) and each row (for y) → absolute value. At a square edge the gradient has the same sign along the whole line, so it adds up. Texture inside the squares (marble, wood carvings) has mixed signs and cancels out. Then search for 9 equally spaced peaks (brute force over spacing and offset).

- crystal and wood-large: all 18 lines on the square edges.
- wood-small: wrong, because the board in `wood-small/0.jpg` runs off the left edge of the photo, so step 1 gives a skewed quad. See [Data issues](#data-issues).

**B. Canny + Hough lines** (lecture: Hough)
`Canny` → `HoughLines` → split into two groups by angle → merge close lines → keep the 9 lines per direction with the most regular spacing. More work than A, but a direct lecture method and independent of step 1.

**C. `cv2.findChessboardCornersSB(gray, (7, 7))`** finds the 49 inner corners directly. It worked only on the empty crystal board. Both wooden boards fail (carvings, hinge line, frame lines), also after warping. Use it as a quick check, not as the main method.

## 3. Align each photo to the empty board

Goal: a homography H per photo that maps it onto the empty-board photo, so the grid from step 2 applies to every photo.

**SIFT + ratio test + RANSAC** (lecture: SIFT matching, exercise 7)
`SIFT_create(4000)` → `BFMatcher.knnMatch(k=2)` → ratio test 0.75 → `findHomography(RANSAC, 5 px)` → `warpPerspective`.

RANSAC inliers per photo (photo k vs. empty board 0, from the first prototype run; the notebook matches in the other direction, so its counts differ by up to ~20%):

| Set | Inliers |
|---|---|
| crystal | 119 to 300 |
| wood-large | 503 to 783; 213 for `16-obstructed` (hand over the board) |
| wood-small | 151 to 194 |

The difference image after alignment shows the pieces and nothing of the board pattern. This works for every photo, including the one with the hand.

SIFT also recovers rotations that a contour method would get wrong:

| Set | Rotation of photo k vs. empty board |
|---|---|
| crystal | ~92° for every photo: the empty board was photographed turned by 90° |
| wood-large | ~0° |
| wood-small | 180° for every photo except **photo 5** (~0°): photo 5 was taken from the opposite side |

A plain checkerboard is symmetric under 180° rotation, so SIFT could in principle match the wrong way round. On our boards the letters, numbers and marble/wood texture break that symmetry. Keep the inlier count as a quality check and flag photos below ~80 inliers.

## 4. Orientation: where is A1?

Because of the rotations above, the orientation cannot come from the camera frame. Derive it once from the **start position photo (1)** aligned to the empty board:

1. **Square color:** A1 is a dark square. Of the 4 board corners, two are dark: 2 candidate orientations remain.
2. **White pieces:** in the start photo, white pieces fill ranks 1 and 2. Compare the brightness of the changed pixels (step 5) in the two-rank band on each of the two remaining candidate sides.

Store this as a rotation of the grid (0/90/180/270°). Every other photo is already aligned to the empty board, so the same labelling holds for all of them.

## 5. Occupancy

For each square of the aligned photo:

1. **Difference to the empty board:** `absdiff` of the blurred aligned photo and the blurred empty board. Use LAB (or HSV) and weight the chroma channels: brightness changes from lighting and shadows mostly affect L, pieces also change a/b.
2. **Measure only the inner ~60% of the square.** Tall pieces lean into the neighbouring square, especially at the image edges, because the camera is not exactly above every square.
3. **Decide:** one value per square (mean difference, or share of pixels above a threshold) → Otsu over the 64 values of the photo. Photo 1 has 32 known occupied squares and gives a fixed threshold to check against.

**Fallback / second feature:** edge density per square (`Canny` pixel count). The crystal pieces are translucent and may change the color less than the wooden ones, but their edges stay visible.

**Lighting differences** (open point in the README): try, in this order, (a) the chroma channels only, (b) histogram matching of the photo to the empty board, (c) comparing with the **previous** photo instead of the empty board. Lighting changes less between two consecutive photos.

**Obstruction:** skip a photo if SIFT inliers drop sharply or more than ~6 squares change against the previous photo (one move changes at most 4 squares, castling). This catches `16-obstructed`.

## 6. Piece color

On each occupied square, take the pixels that changed (mask from step 5) and compute their mean brightness (L channel). Split into white/black with Otsu, or with the two means from the start photo (16 known white, 16 known black). Check this per set: the crystal "white" pieces are clear/cream, and the wood-small black pieces are mid brown.

## 7. Piece type

**Primary: move logic.** The start position fixes all 32 pieces. Between two consecutive photos:

- A normal move changes 2 squares: one becomes empty, one becomes occupied.
- A capture changes 2 squares: one becomes empty, the other changes color.
- Castling changes 4 squares, en passant 3.

Generate the legal moves of the current position (the `python-chess` library does this; it is game logic, not CV) and pick the move whose changed squares best match the occupancy and color changes. The piece types follow from the move. Visual recognition is then only needed to recover from errors and for promotion.

**Fallback / evaluation of the visual part:** top-down shape features of the changed-pixel mask per square: area, circularity, Hu moments (`cv2.HuMoments`), HSV histogram. Train a kNN on the 32 labelled pieces of the start photo per set. Expect weak results for pawn/bishop/rook from above. That is itself a finding for the presentation (visual only vs. visual + move logic).

King vs. queen in the start position: the queen stands on her own color (white queen on D1, black queen on D8).

## 8. Output and evaluation

- Per photo: 8×8 labels, written as a FEN string. FEN is also the fastest format for the hand-labelled ground truth (one line per photo).
- Move table in standard notation (SAN), which `python-chess` produces from the chosen move.
- Metrics from the README: per-square accuracy with confusion matrix (occupancy and type separately), full-board accuracy.

## Data issues

Found while prototyping:

- **`wood-small/0.jpg`** (empty board): the board is cut off at the left image edge. Step 1 then gives a skewed outline and step 2 fails. Re-take the empty photo, or find the grid on photo 1 (start position) for this set.
- **`crystal/0.jpg`**: the board is turned 90° compared to all other photos of the set. SIFT handles it, but keep it in mind when comparing photos by eye.
- **`wood-small/5.jpg`**: taken from the opposite side of the board (180°). SIFT handles it too.

## Suggested order of work

1. Steps 0 to 3 in one notebook per set: board outline, grid, alignment, with a debug image per photo. The prototype code exists and only needs moving into the repo.
2. Step 5 occupancy with the evaluation on the hand-labelled photos. This is the first result worth presenting.
3. Steps 4, 6, 7 on top.
