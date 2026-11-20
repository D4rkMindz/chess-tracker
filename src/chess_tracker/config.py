from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    """ All settings that can be defined in .env or have a default value. """
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        # .env is shared with docker compose and the AWS/S3 clients (S3_*, AWS_*), so keys that
        # are not settings here are ignored instead of rejected.
        extra="ignore"
    )