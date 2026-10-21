"""Scientifically conservative monitor-based normalization for detector sums.

Each selected frame contributes its recorded detector counts to the numerator.
The result is (sum of counts) / (sum of matched monitor readings).  These are
monitor-normalized *rates per monitor unit*, not per-frame normalized means.

Beam ROI readings require an explicitly supplied saturation threshold: we never
infer unsaturated beam readings from a bright blob or silently drop saturated
pixels, which could bias the monitor low.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


class NormalizationError(ValueError):
    """Invalid, missing, ambiguous or non-positive normalization data."""


def parse_beam_roi(value: str, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """An image ROI in x,y,width,height, 0-based, with positive extent."""
    try:
        parts = [int(part.strip()) for part in value.split(",")]
    except (AttributeError, TypeError, ValueError) as exc:
        raise NormalizationError("Beam ROI must be x,y,width,height (0-based integers)") from exc
    if len(parts) != 4:
        raise NormalizationError("Beam ROI must be x,y,width,height (0-based integers)")
    x, y, width, height = parts
    h, w = shape
    if x < 0 or y < 0 or width < 1 or height < 1 or x + width > w or y + height > h:
        raise NormalizationError(f"Beam ROI {value!r} falls outside frame {shape}")
    return x, y, width, height


def validate_saturation(value: float | None) -> float:
    if value is None or not math.isfinite(float(value)) or float(value) <= 0:
        raise NormalizationError(
            "Direct beam normalization requires --beam-saturation: the detector's "
            "known first saturated count (not an inferred maximum)"
        )
    return float(value)


def load_beam_mask(path: str | Path | None, shape: tuple[int, int]) -> np.ndarray | None:
    if path is None or not str(path).strip():
        return None
    try:
        mask = np.load(Path(path), allow_pickle=False)
    except Exception as exc:
        raise NormalizationError(f"Cannot read Beam ROI exclusion mask: {exc}") from exc
    if mask.shape != shape or mask.dtype != np.bool_:
        raise NormalizationError(f"Beam mask must be bool .npy with shape {shape}; got {mask.dtype} {mask.shape}")
    return mask


def monitor_scalar(value: object, *, label: str) -> float:
    try:
        data = np.asarray(value)
        if data.size != 1 or data.dtype.kind not in "biuf":
            raise ValueError("expected one numeric value")
        number = float(data.reshape(-1)[0])
    except (TypeError, ValueError, OverflowError) as exc:
        raise NormalizationError(f"{label}: expected one scalar monitor reading per frame") from exc
    if not math.isfinite(number) or number <= 0:
        raise NormalizationError(f"{label}: monitor must be finite and > 0; got {number}")
    return number


def beam_roi_intensity(
    frame: np.ndarray, roi: tuple[int, int, int, int], *,
    saturation: float, exclude_mask: np.ndarray | None = None,
    min_valid_fraction: float = 0.95,
) -> float:
    """Sum a known unsaturated beam ROI; fail closed on saturated data.

    Negative detector sentinels and nonfinite values are never valid. Masked
    pixels are deliberately excluded. Require sufficient ROI coverage to make
    a single badly specified beam mask / detector gap obvious.
    """
    data = np.asarray(frame)
    if data.ndim != 2:
        raise NormalizationError("Direct beam ROI requires 2D detector frames")
    x, y, width, height = roi
    crop = data[y:y + height, x:x + width]
    if crop.shape != (height, width):
        raise NormalizationError("Beam ROI is outside detector frame")
    valid = np.isfinite(crop) & (crop >= 0)
    if exclude_mask is not None:
        if exclude_mask.shape != data.shape or exclude_mask.dtype != np.bool_:
            raise NormalizationError("Beam ROI exclusion mask shape/dtype mismatch")
        valid &= ~exclude_mask[y:y + height, x:x + width]
    if np.count_nonzero(valid) < max(1, math.ceil(crop.size * min_valid_fraction)):
        raise NormalizationError("Beam ROI has too many invalid/masked pixels; choose a clean ROI")
    if np.any(crop[valid] >= validate_saturation(saturation)):
        raise NormalizationError("Beam ROI contains saturated pixels; use an unsaturated reference or I0 monitor")
    result = float(np.sum(crop[valid], dtype=np.float64))
    if not math.isfinite(result) or result <= 0:
        raise NormalizationError("Beam ROI has no positive, finite summed intensity")
    return result


def validate_normalization_options(
    operation: str, *, i0_path: str = "", beam_roi: str = "",
    beam_saturation: float | None = None, frame_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if operation == "i0-normalized":
        if not i0_path.strip().startswith("/"):
            raise NormalizationError("I0 normalization requires an absolute HDF5 Dataset path (--i0-path)")
    if operation == "beam-normalized":
        roi = parse_beam_roi(beam_roi, frame_shape)
        validate_saturation(beam_saturation)
        return roi
    return None


def normalized_sum(accumulator: np.ndarray, monitor_total: float, *, label: str) -> np.ndarray:
    if not math.isfinite(monitor_total) or monitor_total <= 0:
        raise NormalizationError(f"{label} monitor sum must be finite and > 0")
    return np.asarray(accumulator, dtype=np.float64) / monitor_total
