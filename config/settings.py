from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class ConfigError(RuntimeError):
    pass


@dataclass
class Settings:
    bot_token: str
    ocr_languages: List[str] = field(default_factory=lambda: ["ru", "en"])
    use_gpu: bool = False
    max_file_size_mb: int = 20
    keep_uploads: bool = False
    uploads_dir: Path = DATA_DIR / "uploads"
    results_dir: Path = DATA_DIR / "results"


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _parse_languages(value: str | None) -> List[str]:
    if not value:
        return ["ru", "en"]
    langs = [item.strip().lower() for item in value.split(",") if item.strip()]
    return langs or ["ru", "en"]


def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        raise ConfigError(
            "Не задан BOT_TOKEN. Создайте файл .env (см. .env.example) "
            "и укажите токен, полученный у @BotFather."
        )

    try:
        max_size = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
    except ValueError as exc:
        raise ConfigError("MAX_FILE_SIZE_MB должен быть целым числом.") from exc
    if max_size <= 0:
        raise ConfigError("MAX_FILE_SIZE_MB должен быть больше нуля.")

    settings = Settings(
        bot_token=token,
        ocr_languages=_parse_languages(os.getenv("OCR_LANGUAGES")),
        use_gpu=_parse_bool(os.getenv("USE_GPU"), default=False),
        max_file_size_mb=max_size,
        keep_uploads=_parse_bool(os.getenv("KEEP_UPLOADS"), default=False),
    )

    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    return settings
