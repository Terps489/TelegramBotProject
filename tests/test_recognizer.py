from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from ocr.recognizer import OCRRecognizer, _to_box
from ocr.result import OCRBox, OCRResult


def _write_blank_image(path: Path) -> None:
    img = np.full((400, 800, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_to_box_parses_easyocr_tuple():
    raw = ([[0, 0], [10, 0], [10, 5], [0, 5]], "Hello", 0.9123)
    box = _to_box(raw)
    assert isinstance(box, OCRBox)
    assert box.text == "Hello"
    assert box.confidence == pytest.approx(0.9123)
    assert box.bbox == ((0, 0), (10, 0), (10, 5), (0, 5))


def test_to_box_strips_whitespace():
    raw = ([[0, 0], [1, 0], [1, 1], [0, 1]], "  spaced  ", 0.5)
    assert _to_box(raw).text == "spaced"


def test_ocr_result_aggregates_text_and_confidence():
    result = OCRResult(
        boxes=[
            OCRBox(text="Привет", confidence=0.9),
            OCRBox(text="мир", confidence=0.7),
        ],
        elapsed_seconds=0.5,
    )
    assert result.text == "Привет\nмир"
    assert result.is_empty is False
    assert result.average_confidence == pytest.approx(0.8)


def test_ocr_result_is_empty_when_no_text():
    result = OCRResult(boxes=[], elapsed_seconds=0.1)
    assert result.is_empty
    assert result.text == ""
    assert result.average_confidence is None


def test_recognizer_uses_injected_reader(tmp_path: Path):
    img_path = tmp_path / "blank.png"
    _write_blank_image(img_path)

    fake_reader = MagicMock()
    fake_reader.readtext.return_value = [
        ([[0, 0], [50, 0], [50, 20], [0, 20]], "Hello", 0.95),
        ([[0, 30], [60, 30], [60, 50], [0, 50]], "world", 0.85),
    ]

    recognizer = OCRRecognizer(languages=("en",))
    recognizer._reader = fake_reader

    result = recognizer.recognize(img_path)

    fake_reader.readtext.assert_called_once()
    assert result.text == "Hello\nworld"
    assert result.average_confidence == pytest.approx(0.9)
    assert result.elapsed_seconds >= 0


def test_recognizer_empty_when_reader_returns_nothing(tmp_path: Path):
    img_path = tmp_path / "blank.png"
    _write_blank_image(img_path)

    fake_reader = MagicMock()
    fake_reader.readtext.return_value = []

    recognizer = OCRRecognizer(languages=("en",))
    recognizer._reader = fake_reader

    result = recognizer.recognize(img_path)
    assert result.is_empty
    assert result.average_confidence is None
