"""Non-destructive detector bright-spot detection for HDXF Viewer.

Returns detector-pixel bounding boxes (x0, y0, x1, y1) with exclusive
right/bottom bounds. No source image or user mask is modified. This is a
visual anomaly finder, NOT a scientific Bragg-peak indexing algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class BrightSpotOptions:
    sensitivity: float = 6.0  # multiplier of robust local-residual noise
    minimum_pixels: int = 2
    max_regions: int = 250
    background_sigma_px: float = 7.0


def detect_bright_spots(
    image: np.ndarray,
    options: BrightSpotOptions = BrightSpotOptions(),
    *,
    excluded_mask: np.ndarray | None = None,
) -> list[tuple[int, int, int, int]]:
    """Find connected regions significantly brighter than local background.

    Negative/nonfinite detector pixels (typical detector gaps/bad pixels) and
    user-masked pixels are excluded. The detector's broad illumination is
    estimated using a Gaussian low-pass background; spot significance is
    measured against a robust MAD estimate of residual noise. A minimum
    fractional contrast avoids selecting ordinary variations on bright areas.
    """
    pixels = np.asarray(image)
    if pixels.ndim != 2:
        raise ValueError("bright-spot detection expects a 2D detector frame")
    if not (1.0 <= options.sensitivity <= 50.0):
        raise ValueError("sensitivity must be within 1..50")
    if not (1 <= options.minimum_pixels <= 1000000):
        raise ValueError("minimum_pixels must be >= 1")
    if options.max_regions < 1 or options.background_sigma_px <= 0:
        raise ValueError("invalid bright-spot detection options")

    valid = np.isfinite(pixels) & (pixels >= 0)
    if excluded_mask is not None:
        if excluded_mask.shape != pixels.shape:
            raise ValueError("excluded mask shape differs from frame")
        valid &= ~np.asarray(excluded_mask, dtype=bool)
    if np.count_nonzero(valid) < 16:
        return []

    values = np.asarray(pixels, dtype=np.float32)
    sample = values[valid]
    cap = float(np.percentile(sample, 98.5))
    # Clipping for background *only* prevents a small number of very hot pixels
    # from raising their own background. Retain original values for detection.
    blurred_values = np.where(valid, np.minimum(values, cap), 0.0)
    sigma = float(options.background_sigma_px)
    weights = ndi.gaussian_filter(valid.astype(np.float32), sigma=sigma, mode="nearest")
    background = ndi.gaussian_filter(blurred_values, sigma=sigma, mode="nearest")
    background /= np.maximum(weights, 1e-5)
    residual = values - background

    # Robust global noise estimate; sample sparsely on large detector frames.
    noise_sample = residual[::4, ::4][valid[::4, ::4]]
    center = float(np.median(noise_sample))
    mad = float(np.median(np.abs(noise_sample - center)))
    noise = max(1.0, 1.4826 * mad)
    # Use both absolute significance and relative contrast, so uniformly high
    # illumination isn't mistaken for a local bright feature.
    candidate = valid & (residual > options.sensitivity * noise) & (
        values > background + 1.0
    ) & (values > np.maximum(0.0, background) * 1.30)
    if not np.any(candidate):
        return []

    labels, count = ndi.label(candidate, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return []
    boxes = ndi.find_objects(labels)
    areas = np.bincount(labels[candidate], minlength=count + 1)
    hits: list[tuple[float, tuple[int, int, int, int]]] = []
    for index, area_slice in enumerate(boxes, start=1):
        if area_slice is None or areas[index] < options.minimum_pixels:
            continue
        ys, xs = area_slice
        # Ignore spurious giant regions; flag smaller spatially-local anomalies.
        if ((ys.stop - ys.start) * (xs.stop - xs.start)) > pixels.size * 0.15:
            continue
        patch = residual[area_slice]
        local_labels = labels[area_slice] == index
        score = float(np.max(patch[local_labels]))
        hits.append((score, (int(xs.start), int(ys.start), int(xs.stop), int(ys.stop))))
    hits.sort(key=lambda entry: entry[0], reverse=True)
    return [box for _score, box in hits[:options.max_regions]]
