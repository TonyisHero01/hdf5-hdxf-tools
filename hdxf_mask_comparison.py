"""Numeric before/after manual-mask comparison for Notebook Reference 1D.

Both reductions must use the same averaged detector image, geometry and beam
center. Before = optional source detector mask only; After = source mask OR
user mask. Notebook's own gap/saturation/hot-pixel clipping remains unchanged.
"""
from __future__ import annotations

import numpy as np

TRACKS = (
    ("included", "Spots included"),
    ("spots_masked", "Hot/Bragg spots masked"),
    ("sigma_clipped", "Ring MAD σ-clipped (notebook)"),
)


def make_mask_comparison(before: dict, after: dict, *, excluded_radius: float) -> dict:
    """Align notebook radial bins; distinguish empty bins from zero intensity.

    The notebook uses a fixed radial bin width, but masks may shorten the
    highest-radius edge. Pad missing bins with NaN rather than extrapolating.
    """
    tracks: dict[str, dict] = {}
    for key, _label in TRACKS:
        old, new = before[key], after[key]
        n_old = len(old["radius_px"])
        n_new = len(new["radius_px"])
        n = max(n_old, n_new)
        common = min(n_old, n_new)
        if common and not np.allclose(old["radius_px"][:common], new["radius_px"][:common], rtol=0, atol=1e-8):
            raise ValueError(f"Cannot align Notebook bins for {key}: different radial coordinates")
        longer = old if n_old >= n_new else new
        radius = np.asarray(longer["radius_px"], dtype=np.float64).copy()
        theta = np.asarray(longer["two_theta_deg"], dtype=np.float64).copy()
        before_y = np.full(n, np.nan)
        after_y = np.full(n, np.nan)
        before_count = np.zeros(n, dtype=np.int64)
        after_count = np.zeros(n, dtype=np.int64)
        before_y[:n_old] = old["intensity"]
        after_y[:n_new] = new["intensity"]
        before_count[:n_old] = old["count"]
        after_count[:n_new] = new["count"]
        before_y[before_count == 0] = np.nan
        after_y[after_count == 0] = np.nan
        delta = after_y - before_y
        within = (radius > excluded_radius) & np.isfinite(theta)
        comparable = within & np.isfinite(delta)
        changed = comparable & ~np.isclose(before_y, after_y, atol=1e-12, rtol=1e-10)
        newly_empty = within & (before_count > 0) & (after_count == 0)
        abs_change = np.abs(delta[comparable])
        tracks[key] = {
            "radius_px": radius, "two_theta_deg": theta,
            "before": before_y, "after": after_y, "delta": delta,
            "before_count": before_count, "after_count": after_count,
            "compared_bins": int(np.count_nonzero(comparable)),
            "changed_bins": int(np.count_nonzero(changed)),
            "newly_empty_bins": int(np.count_nonzero(newly_empty)),
            "max_abs_delta": float(np.max(abs_change)) if abs_change.size else 0.0,
            "mean_abs_delta": float(np.mean(abs_change)) if abs_change.size else 0.0,
        }
    return {"tracks": tracks}
