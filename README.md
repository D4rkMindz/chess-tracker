# CVAI HS26 Homework: Chess Board State from a God Cam

Group homework project for the module **Computer Vision & AI (I.BA_CVAI)**, autumn semester 2026, at the Lucerne University of Applied Sciences and Arts (HSLU).

The project is the larger homework assignment of the module. It ends with a presentation in the last semester week and earns bonus points for the exam.

## Quickstart

First, copy ``/.env.example`` to ``/.env``
Then adjust the values in it according to the documentation inside

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.13 or newer.

```bash
uv sync
```

Then open a notebook in `notebooks/` with the `.venv` kernel. Run notebooks from the repository root so the relative `data/chess/...` paths resolve.

## Run the pipeline

Run the commands from the repository root. Each command calibrates on photo `0` (empty board) and photo `1` (start position) of one set, then prints the recognised board of every photo from `1` on.

```bash
uv run python src/main.py --set crystal
uv run python src/main.py --set wood-large
uv run python src/main.py --set wood-small
```

The printed board shows rank 8 at the top: `W` = white piece, `B` = black piece, `.` = empty. Photos with fewer than 80 SIFT inliers are marked `UNRELIABLE`.

### Debug images

Add `--show` to open one debug image per photo: the warped board with the grid, the `a1` label and one circle per recognised piece (green = white, red = black). Close the window to continue with the next photo.

```bash
uv run python src/main.py --set crystal --show
uv run python src/main.py --set wood-large --show
uv run python src/main.py --set wood-small --show
```

## Question

How reliably can classical computer vision recognise the board state (occupancy and piece type) from handheld top-down photos of a chess game?

## Goal

We read top-down ("god cam") photos of a chess board. For each of the 64 squares (A1–H8) the algorithm outputs:

1. whether a piece stands on the square (**occupancy**)
2. which **piece type** it is (king, queen, rook, bishop, knight, pawn; white or black)

**Stretch goals:** derive the move between two consecutive photos and log the game in standard chess notation; process a video instead of single photos.

The full project description is in [`docs/02_project-idea.md`](docs/02_project-idea.md).

## Approach

Classical computer vision only (Python + OpenCV in Jupyter), based on the lecture material. No deep learning.

1. **Find the board and squares** on the empty-board photo (contrast detection or SIFT).
2. **Align** each handheld photo to the empty board.
3. **Background subtraction:** subtract the empty board from each photo with pieces.
4. **Occupancy:** decide which squares hold a piece.
5. **Piece type:** combine the top-down look of each piece (the start position gives labelled examples per set) with move logic (each photo differs from the previous one by a single move).

## Data

Our own photos, taken by hand above the board with varied lighting. Each set follows the same sequence: `0` = empty board, `1` = start position, then one photo after each move.

| Set | Folder | Photos |
|---|---|---|
| Small wooden set | [`data/chess/wood-small/`](data/chess/wood-small/) | 11 |
| Large wooden set | [`data/chess/wood-large/`](data/chess/wood-large/) | 19 |
| Crystal set | [`data/chess/crystal-noletter/`](data/chess/crystal-noletter/) | 15 |

## Evaluation

We label the photos by hand and compare the recognised board state with the true one:

- **Per-square accuracy** (main metric) for occupancy and piece type, with a confusion matrix
- **Full-board accuracy** (secondary): share of photos with all 64 squares correct

## Status

🟡 **First experiments.** The topic is decided and the photos for all three sets are taken. We are now working on board and square detection ([`notebooks/classification/contrast.ipynb`](notebooks/classification/contrast.ipynb)).

Open points: how to align the handheld photos (contrast detection or SIFT feature matching) and how to handle different lighting between the empty board and the later photos.

## Timeline

| Date | Milestone |
|---|---|
| 17./18.12.2026 (SW14) | Homework presentations |

## Project structure

```
.
├── data/chess/               # Our photos, one folder per chess set
│   ├── crystal-noletter/
│   ├── wood-large/
│   └── wood-small/
├── docs/
│   ├── 01_brainstorming.md   # Candidate topics and comparison
│   └── 02_project-idea.md    # Chosen topic: question, method, evaluation
├── notebooks/
│   └── classification/       # Experiment notebooks
│       └── contrast.ipynb    # First experiments with contrast detection
├── src/chess_tracker/        # Reusable image analysis code (in progress)
├── Effort.md                 # Time log
├── pyproject.toml
└── README.md
```

## Authors

- Natalie Sumbo Filipe (NSF)
- Silas Schuler (SSC)
- Björn Pfoster (BPF)

## Acknowledgments

- **Joachim Wirth** – project coach
- Module instructors: Thomas Koller, Joachim Wirth, Aljosa Smolic, Jan Svoboda

## License

This project is licensed under the [MIT License](LICENSE).
