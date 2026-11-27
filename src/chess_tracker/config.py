from dataclasses import dataclass
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    """ All settings that can be defined in .env or have a default value. """
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        extra="ignore"
    )

    # Folder with one subfolder of photos per chess set.
    data_dir: Path = PROJECT_ROOT / "data" / "chess"
    # Longest image side in pixels after loading. Kernel sizes and thresholds are tuned for this size.
    work_size: int = 1024
    # Side length in pixels of the warped, top-down board.
    warp_size: int = 820


settings = Settings()


@dataclass(frozen=True)
class BoardSet:
    """Detection settings that differ between the chess sets."""
    # "canny" for boards that differ from the table in brightness, "saturation" for boards that differ in colour.
    mask: str
    # Photo the board outline and grid are detected on: "empty", or "start" if the empty photo cuts the board off.
    outline_from: str = "empty"


# Settings per set, key = folder name in data_dir.
BOARD_SETS = {
    "crystal": BoardSet(mask="saturation"),
    "wood-large": BoardSet(mask="canny"),
    # wood-small/0.jpg cuts the board off at the left edge, so the outline comes from the start photo.
    "wood-small": BoardSet(mask="canny", outline_from="start"),
}
