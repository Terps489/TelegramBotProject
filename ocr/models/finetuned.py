from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Iterable, List, Optional

import numpy as np

from ocr.models.base import RawDetection

log = logging.getLogger(__name__)


class FinetunedEasyOCRModel:
    def __init__(
        self,
        recog_network: str,
        model_storage_directory: Path,
        user_network_directory: Path,
        languages: Iterable[str] = ("ru", "en"),
        gpu: bool = False,
    ) -> None:
        self.recog_network = recog_network
        self.model_storage_directory = Path(model_storage_directory)
        self.user_network_directory = Path(user_network_directory)
        self.languages: List[str] = list(languages)
        self.gpu = gpu
        self._reader = None
        self._lock = Lock()

    def warmup(self) -> None:
        self._ensure_reader()

    def recognize(self, image: np.ndarray) -> List[RawDetection]:
        reader = self._ensure_reader()
        return reader.readtext(image, detail=1, paragraph=False)

    def _ensure_reader(self):
        if self._reader is not None:
            return self._reader
        with self._lock:
            if self._reader is None:
                import easyocr

                self._validate_files()
                log.info(
                    "Загружаю кастомную модель EasyOCR '%s' (gpu=%s)...",
                    self.recog_network,
                    self.gpu,
                )
                self._reader = easyocr.Reader(
                    self.languages,
                    gpu=self.gpu,
                    verbose=False,
                    recog_network=self.recog_network,
                    model_storage_directory=str(self.model_storage_directory),
                    user_network_directory=str(self.user_network_directory),
                )
        return self._reader

    def _validate_files(self) -> None:
        weights = self.model_storage_directory / f"{self.recog_network}.pth"
        yaml = self.user_network_directory / f"{self.recog_network}.yaml"
        py = self.user_network_directory / f"{self.recog_network}.py"
        missing = [p for p in (weights, yaml, py) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Не найдены файлы кастомной модели: " + ", ".join(str(p) for p in missing)
            )
