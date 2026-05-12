"""Юнит-тесты для модуля предобработки изображений.

Тесты намеренно не требуют реальных файлов — синтетические массивы
генерируются на лету через numpy/OpenCV, чтобы их можно было прогнать в CI
без подгрузки моделей и без интернет-соединения.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from ocr import preprocessing


def _synthetic_text_image(width: int = 800, height: int = 200) -> np.ndarray:
    """Белый фон с чёрной строкой текста — годится для базовой проверки."""
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(
        img,
        "Hello, OCR!",
        org=(40, 130),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=2.5,
        color=(0, 0, 0),
        thickness=4,
        lineType=cv2.LINE_AA,
    )
    return img


def test_to_grayscale_returns_single_channel():
    img = _synthetic_text_image()
    gray = preprocessing.to_grayscale(img)
    assert gray.ndim == 2
    assert gray.shape == img.shape[:2]


def test_to_grayscale_passes_single_channel_through():
    gray = np.full((100, 100), 128, dtype=np.uint8)
    assert preprocessing.to_grayscale(gray) is gray or preprocessing.to_grayscale(gray).ndim == 2


def test_binarize_produces_only_two_values():
    gray = preprocessing.to_grayscale(_synthetic_text_image())
    binary = preprocessing.binarize(gray)
    unique = np.unique(binary)
    assert set(unique.tolist()).issubset({0, 255})


def test_denoise_keeps_shape_and_dtype():
    gray = preprocessing.to_grayscale(_synthetic_text_image())
    cleaned = preprocessing.denoise(gray)
    assert cleaned.shape == gray.shape
    assert cleaned.dtype == gray.dtype


def test_resize_upscales_small_image():
    small = np.full((100, 150, 3), 255, dtype=np.uint8)
    resized = preprocessing.resize_if_needed(small)
    longest = max(resized.shape[:2])
    assert longest >= preprocessing.MIN_SIDE_PX


def test_resize_downscales_huge_image():
    huge = np.full((3000, 4000, 3), 255, dtype=np.uint8)
    resized = preprocessing.resize_if_needed(huge)
    longest = max(resized.shape[:2])
    assert longest <= preprocessing.MAX_SIDE_PX


def test_resize_keeps_image_in_range():
    img = np.full((1000, 1200, 3), 255, dtype=np.uint8)
    resized = preprocessing.resize_if_needed(img)
    assert resized.shape == img.shape


def test_read_image_raises_for_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        preprocessing.read_image(tmp_path / "no-such-file.png")


def test_read_image_raises_for_corrupt_file(tmp_path: Path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"this is not an image")
    with pytest.raises(ValueError):
        preprocessing.read_image(bad)


def test_preprocess_full_pipeline_returns_binary(tmp_path: Path):
    img_path = tmp_path / "sample.png"
    cv2.imwrite(str(img_path), _synthetic_text_image())

    result = preprocessing.preprocess(img_path)
    assert result.ndim == 2
    assert result.dtype == np.uint8
    assert set(np.unique(result).tolist()).issubset({0, 255})
