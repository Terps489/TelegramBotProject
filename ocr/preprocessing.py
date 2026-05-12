"""Предобработка изображений для повышения качества OCR.

Шаги конвейера:
    1. чтение изображения из файла;
    2. перевод в grayscale;
    3. шумоподавление;
    4. бинаризация (адаптивный или Otsu thresholding);
    5. опциональное изменение размера, если изображение слишком маленькое.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

PathLike = Union[str, Path]

MIN_SIDE_PX = 600
MAX_SIDE_PX = 2000


def read_image(path: PathLike) -> np.ndarray:
    """Читает изображение с диска. Корректно работает с не-ASCII путями (Windows)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    # cv2.imread на Windows может падать с кириллическими путями,
    # поэтому читаем через numpy.fromfile + cv2.imdecode.
    raw = np.fromfile(str(path), dtype=np.uint8)
    if raw.size == 0:
        raise ValueError(f"Файл пуст или не читается: {path}")
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV не смог декодировать изображение: {path}")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(image: np.ndarray) -> np.ndarray:
    """Шумоподавление с сохранением резкости границ символов."""
    return cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)


def binarize(image: np.ndarray) -> np.ndarray:
    """Бинаризация: Otsu — устойчиво для большинства сканов и фото текста."""
    _, binary = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def resize_if_needed(image: np.ndarray) -> np.ndarray:
    """Масштабирует изображение, если оно слишком маленькое/огромное.

    Слишком мелкие картинки EasyOCR распознаёт плохо; слишком крупные —
    долго и без выигрыша в качестве.
    """
    h, w = image.shape[:2]
    longest = max(h, w)

    if longest < MIN_SIDE_PX:
        scale = MIN_SIDE_PX / longest
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

    if longest > MAX_SIDE_PX:
        scale = MAX_SIDE_PX / longest
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    return image


def preprocess(path: PathLike) -> np.ndarray:
    """Полный конвейер предобработки: путь → готовое к OCR изображение."""
    image = read_image(path)
    image = resize_if_needed(image)
    gray = to_grayscale(image)
    cleaned = denoise(gray)
    binary = binarize(cleaned)
    return binary
