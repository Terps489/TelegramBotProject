from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable, List, Optional, Union

import numpy as np

from ocr.models import EasyOCRModel, OCRModel, RawDetection
from ocr.preprocessing import PreprocessingOptions, preprocess
from ocr.result import OCRBox, OCRResult

log = logging.getLogger(__name__)

PathLike = Union[str, Path]


class OCRRecognizer:
    def __init__(
        self,
        model: Optional[OCRModel] = None,
        languages: Iterable[str] = ("ru", "en"),
        gpu: bool = False,
        preprocess_options: Optional[PreprocessingOptions] = None,
    ) -> None:
        self.model: OCRModel = model if model is not None else EasyOCRModel(
            languages=languages, gpu=gpu
        )
        self.preprocess_options = preprocess_options or PreprocessingOptions()

    def warmup(self) -> None:
        self.model.warmup()

    def recognize(self, image_path: PathLike) -> OCRResult:
        start = time.perf_counter()
        processed = preprocess(image_path, self.preprocess_options, model=self.model)
        raw = self.model.recognize(processed)
        elapsed = time.perf_counter() - start
        return OCRResult(boxes=[_to_box(item) for item in raw], elapsed_seconds=elapsed)

    def recognize_array(self, array: np.ndarray) -> OCRResult:
        start = time.perf_counter()
        raw = self.model.recognize(array)
        elapsed = time.perf_counter() - start
        return OCRResult(boxes=[_to_box(item) for item in raw], elapsed_seconds=elapsed)


def _to_box(raw_item: RawDetection) -> OCRBox:
    bbox, text, confidence = raw_item
    normalized_bbox = tuple((int(x), int(y)) for x, y in bbox)
    return OCRBox(
        text=str(text).strip(),
        confidence=float(confidence) if confidence is not None else 0.0,
        bbox=normalized_bbox,
    )
