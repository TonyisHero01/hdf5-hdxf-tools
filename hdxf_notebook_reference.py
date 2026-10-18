"""Reproducible 1D reduction based on 2D_1D_Plots (2).ipynb, cell 6/14.

This is deliberately NOT pyFAI integration. It reproduces notebook's pixel-radius
binning, beam fit, gap/saturation/hot masks, and its *non-cumulative* MAD clip.
Coordinates are NumPy array indices (x=column, y=row). Angles are 2theta deg.
An external NPY input is required for bit-for-bit comparison to the notebook;
streamed averaging from the viewer may differ from the precomputed NPY input.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NotebookOptions:
    distance_mm: float = 65.0
    pixel_size_mm: float = 0.075
    beam_x: float | None = None
    beam_y: float | None = None
    fit_beam: bool = True
    bin_width_px: float = 2.0
    exclude_radius_px: float = 60.0
    gap_positions: tuple[int, ...] = (500, 1100)
    gap_half_width: int = 35
    edge_buffer: int = 10
    saturation_threshold: float = 32765.0
    clip_sigma: float = 3.0
    clip_iterations: int = 3


def gaussian_beam_center(image: np.ndarray) -> tuple[float, float, dict]:
    """Cell 6's saturation-first, hot-spot-excluded 2D Gaussian fit."""
    from scipy.ndimage import median_filter, gaussian_filter
    from scipy.optimize import curve_fit

    img = np.asarray(image, dtype=np.float64)
    beam_mask = (~np.isfinite(img)) | (img < 0) | (img >= 32765.0)
    local_med = median_filter(img, size=7)
    beam_mask |= (img > local_med + 10.0) & (img > 2.0 * (local_med + 1.0))
    smooth = gaussian_filter(np.where(beam_mask, 0.0, img), sigma=8)
    approx_y, approx_x = np.unravel_index(np.nanargmax(smooth), smooth.shape)
    x0, x1 = max(0, approx_x - 80), min(img.shape[1], approx_x + 80)
    y0, y1 = max(0, approx_y - 80), min(img.shape[0], approx_y + 80)
    roi = img[y0:y1, x0:x1]
    roi_mask = beam_mask[y0:y1, x0:x1]
    yy, xx = np.indices(roi.shape)
    valid = ~roi_mask & np.isfinite(roi)
    xs = xx[valid].ravel().astype(np.float64)
    ys = yy[valid].ravel().astype(np.float64)
    zs = roi[valid].ravel().astype(np.float64)
    if zs.size < 50:
        raise ValueError('Too few valid pixels for Notebook Gaussian beam fit; use manual Beam X/Y.')
    lo, hi = np.percentile(zs, [1, 99.5])
    good = (zs >= lo) & (zs <= hi)
    xs, ys, zs = xs[good], ys[good], zs[good]
    if zs.size < 50:
        raise ValueError('Notebook beam fit has fewer than 50 valid samples')

    def model(coords, amp, xcen, ycen, sx, sy, offset):
        x, y = coords
        return offset + amp * np.exp(-((x-xcen)**2 / (2*sx**2) + (y-ycen)**2 / (2*sy**2)))

    p0 = [np.nanmax(zs)-np.nanmedian(zs), roi.shape[1]/2, roi.shape[0]/2,
          20, 20, np.nanmedian(zs)]
    bounds = ([0, 0, 0, 1, 1, -np.inf],
              [np.inf, roi.shape[1], roi.shape[0], 200, 200, np.inf])
    popt, _ = curve_fit(model, (xs, ys), zs, p0=p0, bounds=bounds, maxfev=20000)
    return (float(x0+popt[1]), float(y0+popt[2]),
            {'approx_x': int(approx_x), 'approx_y': int(approx_y),
             'sigma_x': float(popt[3]), 'sigma_y': float(popt[4])})


def _radial_average(img: np.ndarray, radius: np.ndarray, mask: np.ndarray,
                    bin_width: float, distance_mm: float, pixel_size_mm: float) -> dict:
    valid = ~mask & np.isfinite(img)
    if not np.any(valid):
        raise ValueError('Notebook mask has no valid pixels')
    r_valid = radius[valid]
    values = img[valid]
    bins = np.arange(0.0, float(np.max(r_valid)) + bin_width, bin_width)
    if bins.size < 2:
        bins = np.array([0.0, bin_width], dtype=np.float64)
    weighted_sum, edges = np.histogram(r_valid, bins=bins, weights=values)
    counts, _ = np.histogram(r_valid, bins=bins)
    centers = (edges[:-1] + edges[1:]) * 0.5
    # Notebook intentionally divides empty bins by 1, displaying 0 rather than NaN.
    return {
        'radius_px': centers,
        'two_theta_deg': np.degrees(np.arctan(centers * pixel_size_mm / distance_mm)),
        'intensity': weighted_sum / np.maximum(counts, 1),
        'count': counts,
        'sum': weighted_sum,
    }


def _reference_sigma_clip(img: np.ndarray, radius: np.ndarray, base_mask: np.ndarray,
                          bin_width: float, nsigma: float, iterations: int) -> np.ndarray:
    """Same behavior as cell 14, but groups ring pixels once instead of rescanning images.

    IMPORTANT: the notebook *replaces* rather than accumulates new clips every
    iteration. We retain this explicitly for reproducibility, not as a new
    recommended sigma-clipping algorithm. n_phi_bins is ignored in the notebook.
    """
    rmax = float(np.max(radius[~base_mask]))
    edges = np.arange(0.0, rmax + bin_width, bin_width)
    if len(edges) < 2:
        return np.zeros(img.shape, dtype=bool)
    bins = np.searchsorted(edges, radius.ravel(), side='right') - 1
    order = np.argsort(bins, kind='stable')
    sorted_bins = bins[order]
    stops = np.searchsorted(sorted_bins, np.arange(len(edges)), side='left')
    start_positions = stops[:-1]
    stop_positions = stops[1:]
    flat_img = img.ravel()
    flat_base = base_mask.ravel()
    finite = np.isfinite(flat_img)
    working_mask = flat_base.copy()
    last_clip = np.zeros(flat_base.shape, dtype=bool)
    for _ in range(iterations):
        new_clip = np.zeros(flat_base.shape, dtype=bool)
        for start, stop in zip(start_positions, stop_positions):
            indices = order[start:stop]
            if indices.size < 6:
                continue
            good = ~working_mask[indices] & finite[indices]
            if np.count_nonzero(good) < 6:
                continue
            vals = flat_img[indices[good]]
            med = np.median(vals)
            mad = np.median(np.abs(vals - med))
            std = mad * 1.4826
            if std < 1e-10:
                continue
            bad = good & (np.abs(flat_img[indices] - med) > nsigma * std)
            new_clip[indices[bad]] = True
        working_mask = flat_base | new_clip  # Faithful to the notebook.
        last_clip = new_clip
    return last_clip.reshape(img.shape)


def calculate_notebook_reference(image: np.ndarray, options: NotebookOptions,
                                 *, extra_mask: np.ndarray | None = None) -> dict:
    """Return three notebook 1D curves and diagnostics, matching cells 6/14."""
    from scipy.ndimage import median_filter, binary_dilation

    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f'Notebook reference requires 2D image, got {img.shape}')
    if options.distance_mm <= 0 or options.pixel_size_mm <= 0 or options.bin_width_px <= 0:
        raise ValueError('Distance, square pixel size and bin width must be positive')
    if options.fit_beam:
        beam_x, beam_y, fit_info = gaussian_beam_center(img)
    else:
        if options.beam_x is None or options.beam_y is None:
            raise ValueError('Beam X/Y required when Gaussian fit is disabled')
        beam_x, beam_y = float(options.beam_x), float(options.beam_y)
        fit_info = {}
    yy, xx = np.indices(img.shape)
    radius = np.hypot(xx - beam_x, yy - beam_y)
    gap_mask = np.zeros(img.shape, dtype=bool)
    for gy in options.gap_positions:
        low = max(0, int(gy) - options.gap_half_width - options.edge_buffer)
        high = min(img.shape[0], int(gy) + options.gap_half_width + options.edge_buffer + 1)
        gap_mask[low:high, :] = True
    base = (~np.isfinite(img)) | (img < 0) | (img >= options.saturation_threshold) | gap_mask
    if extra_mask is not None:
        if np.shape(extra_mask) != np.shape(img):
            raise ValueError('Additional detector mask has incorrect dimensions')
        base |= np.asarray(extra_mask, dtype=bool)
    if not np.any(~base):
        raise ValueError('No valid pixels remain after Notebook base mask')
    # Notebook cell 6: replace masked pixel values with global valid median,
    # then take 15x15 median and dilate candidates by 10 iterations.
    filled = np.where(base, np.nanmedian(img[~base]), img)
    local_med = median_filter(filled, size=15)
    hot = ((filled > local_med + 15.0) & (filled > 2.0 * (local_med + 1.0)))
    hot &= radius > options.exclude_radius_px
    hot = binary_dilation(hot, iterations=10)
    spot_mask = base | hot
    sigma = _reference_sigma_clip(img, radius, base, options.bin_width_px,
                                  options.clip_sigma, options.clip_iterations)
    powder_mask = base | sigma
    included = _radial_average(img, radius, base, options.bin_width_px,
                               options.distance_mm, options.pixel_size_mm)
    spots = _radial_average(img, radius, spot_mask, options.bin_width_px,
                            options.distance_mm, options.pixel_size_mm)
    powder = _radial_average(img, radius, powder_mask, options.bin_width_px,
                             options.distance_mm, options.pixel_size_mm)
    return {
        'included': included, 'spots_masked': spots, 'sigma_clipped': powder,
        'beam_x': beam_x, 'beam_y': beam_y, 'fit_info': fit_info,
        'base_mask_count': int(base.sum()),
        'spot_mask_count': int(hot.sum()),
        'sigma_mask_count': int(sigma.sum()),
        'base_mask': base, 'hot_mask': hot, 'sigma_mask': sigma,
        'min_radius_px': options.exclude_radius_px,
        'shape': img.shape,
    }
