from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from ocr import preprocessing
from ocr.preprocessing import PreprocessingOptions


def _imwrite(path: Path, image: np.ndarray) -> None:
    # cv2.imwrite ломается на не-ASCII путях под Windows
    ok, buf = cv2.imencode(path.suffix, image)
    assert ok
    path.write_bytes(buf.tobytes())


def _synthetic_text_image(width: int = 800, height: int = 200) -> np.ndarray:
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
    result = preprocessing.to_grayscale(gray)
    assert result.ndim == 2


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
    assert max(resized.shape[:2]) >= preprocessing.MIN_SIDE_PX


def test_resize_downscales_huge_image():
    huge = np.full((3000, 4000, 3), 255, dtype=np.uint8)
    resized = preprocessing.resize_if_needed(huge)
    assert max(resized.shape[:2]) <= preprocessing.MAX_SIDE_PX


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
    _imwrite(img_path, _synthetic_text_image())

    result = preprocessing.preprocess(img_path)
    assert result.ndim == 2
    assert result.dtype == np.uint8
    assert set(np.unique(result).tolist()).issubset({0, 255})


def test_deskew_detects_and_corrects_rotation():
    img = _synthetic_text_image()
    rotated = preprocessing.rotate(img, angle=7.0)
    fixed = preprocessing.deskew(rotated)

    gray = preprocessing.to_grayscale(fixed)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    residual = preprocessing.estimate_skew_angle(binary)
    assert abs(residual) < 2.0


def test_deskew_no_op_for_straight_image():
    img = _synthetic_text_image()
    gray = preprocessing.to_grayscale(img)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    assert abs(preprocessing.estimate_skew_angle(binary)) < 1.0


def test_estimate_skew_returns_zero_for_blank():
    blank = np.full((100, 100), 255, dtype=np.uint8)
    assert preprocessing.estimate_skew_angle(blank) == 0.0


def test_estimate_skew_ignores_extreme_angles():
    img = _synthetic_text_image()
    rotated = preprocessing.rotate(img, angle=40.0)
    gray = preprocessing.to_grayscale(rotated)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    assert preprocessing.estimate_skew_angle(binary) == 0.0


def test_correct_orientation_picks_best_rotation():
    img = _synthetic_text_image()
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    model = MagicMock()

    def fake_recognize(arr: np.ndarray):
        if arr.shape == img.shape:
            return [([[0, 0], [1, 0], [1, 1], [0, 1]], "Hello", 0.95)]
        return [([[0, 0], [1, 0], [1, 1], [0, 1]], "x", 0.1)]

    model.recognize.side_effect = fake_recognize
    fixed = preprocessing.correct_orientation(rotated, model)
    assert fixed.shape == img.shape


def test_preprocess_calls_orientation_only_when_enabled(tmp_path: Path):
    img_path = tmp_path / "sample.png"
    _imwrite(img_path, _synthetic_text_image())

    model = MagicMock()
    model.recognize.return_value = [([[0, 0], [1, 0], [1, 1], [0, 1]], "x", 0.5)]

    preprocessing.preprocess(img_path, PreprocessingOptions(orientation_correction=False), model=model)
    model.recognize.assert_not_called()

    preprocessing.preprocess(img_path, PreprocessingOptions(orientation_correction=True), model=model)
    assert model.recognize.call_count == 4
