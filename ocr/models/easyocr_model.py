from __future__ import annotations

import logging
from threading import Lock
from typing import Iterable, List, Optional

import numpy as np

from ocr.models.base import RawDetection

log = logging.getLogger(__name__)


class EasyOCRModel:
    def __init__(
        self,
        languages: Iterable[str] = ("ru", "en"),
        gpu: bool = False,
        model_storage_directory: Optional[str] = None,
    ) -> None:
        self.languages: List[str] = list(languages)
        self.gpu = gpu
        self.model_storage_directory = model_storage_directory
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

                log.info("Загружаю EasyOCR (%s, gpu=%s)...", self.languages, self.gpu)
                kwargs = {"gpu": self.gpu, "verbose": False}
                if self.model_storage_directory:
                    kwargs["model_storage_directory"] = self.model_storage_directory
                self._reader = easyocr.Reader(self.languages, **kwargs)
        return self._reader
