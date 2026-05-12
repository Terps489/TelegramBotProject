from __future__ import annotations

import logging
import time
from pathlib import Path
from threading import Lock
from typing import Iterable, List, Union

import numpy as np

from ocr.preprocessing import preprocess
from ocr.result import OCRBox, OCRResult

log = logging.getLogger(__name__)

PathLike = Union[str, Path]


class OCRRecognizer:
    def __init__(self, languages: Iterable[str] = ("ru", "en"), gpu: bool = False) -> None:
        self.languages: List[str] = list(languages)
        self.gpu = gpu
        self._reader = None
        self._lock = Lock()

    def warmup(self) -> None:
        self._ensure_reader()

    def _ensure_reader(self):
        if self._reader is not None:
            return self._reader
        with self._lock:
            if self._reader is None:
                import easyocr

                log.info("Загружаю модели EasyOCR (%s, gpu=%s)...", self.languages, self.gpu)
                self._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
        return self._reader

    def recognize(self, image_path: PathLike) -> OCRResult:
        start = time.perf_counter()
        processed = preprocess(image_path)
        reader = self._ensure_reader()
        raw = reader.readtext(processed, detail=1, paragraph=False)
        elapsed = time.perf_counter() - start

        boxes = [_to_box(item) for item in raw]
        return OCRResult(boxes=boxes, elapsed_seconds=elapsed)


def _to_box(raw_item) -> OCRBox:
    bbox, text, confidence = raw_item
    normalized_bbox = tuple((int(x), int(y)) for x, y in bbox)
    return OCRBox(
        text=str(text).strip(),
        confidence=float(confidence) if confidence is not None else 0.0,
        bbox=normalized_bbox,
    )


def recognize_array(recognizer: OCRRecognizer, array: np.ndarray) -> OCRResult:
    start = time.perf_counter()
    reader = recognizer._ensure_reader()
    raw = reader.readtext(array, detail=1, paragraph=False)
    elapsed = time.perf_counter() - start
    return OCRResult(boxes=[_to_box(i) for i in raw], elapsed_seconds=elapsed)
