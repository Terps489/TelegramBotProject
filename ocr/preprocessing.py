from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

PathLike = Union[str, Path]

MIN_SIDE_PX = 600
MAX_SIDE_PX = 2000
DESKEW_MAX_ANGLE = 15.0


@dataclass
class PreprocessingOptions:
    deskew: bool = True
    orientation_correction: bool = False
    denoise: bool = True
    binarize: bool = True


def read_image(path: PathLike) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    # cv2.imread на Windows ломается на не-ASCII путях: читаем через imdecode.
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
    return cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)


def binarize(image: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def resize_if_needed(image: np.ndarray) -> np.ndarray:
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


def estimate_skew_angle(binary: np.ndarray) -> float:
    inverted = cv2.bitwise_not(binary) if binary.mean() > 127 else binary
    coords = np.column_stack(np.where(inverted > 0))
    if coords.size == 0:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    if abs(angle) > DESKEW_MAX_ANGLE:
        return 0.0
    return float(angle)


def rotate(image: np.ndarray, angle: float, border_value: int = 255) -> np.ndarray:
    if abs(angle) < 0.05:
        return image
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def deskew(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    angle = estimate_skew_angle(binary)
    if angle == 0.0:
        return image
    border = 255 if image.ndim == 2 else (255, 255, 255)
    return rotate(image, angle, border_value=border)


def correct_orientation(image: np.ndarray, model) -> np.ndarray:
    rotations = {
        0: image,
        90: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(image, cv2.ROTATE_180),
        270: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }
    best_angle, best_score = 0, -1.0
    for angle, candidate in rotations.items():
        score = _orientation_score(model.recognize(candidate))
        if score > best_score:
            best_score, best_angle = score, angle
    return rotations[best_angle]


def _orientation_score(detections) -> float:
    total = 0.0
    for item in detections:
        _, text, conf = item
        if text and text.strip():
            total += float(conf) * len(text.strip())
    return total


def preprocess(
    path: PathLike,
    options: Optional[PreprocessingOptions] = None,
    model=None,
) -> np.ndarray:
    options = options or PreprocessingOptions()
    image = read_image(path)
    image = resize_if_needed(image)

    if options.orientation_correction and model is not None:
        image = correct_orientation(image, model)

    if options.deskew:
        image = deskew(image)

    gray = to_grayscale(image)
    cleaned = denoise(gray) if options.denoise else gray
    return binarize(cleaned) if options.binarize else cleaned
