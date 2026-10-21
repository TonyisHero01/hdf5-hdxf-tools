#!/usr/bin/env python3
"""Geometry-aware 1D pyFAI axis handling and non-destructive geometry preview.

PONI specifies the point of normal incidence, not the direct beam intersection
when the detector is tilted. pyFAI handles geometry for native 2theta/q units.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

CORRECTED_THETA = "Corrected θ (rad)"
CORRECTED_TWOTHETA = "Corrected 2θ (deg)"
CORRECTED_Q = "Corrected q (Å⁻¹)"
LEGACY_THETA = "Legacy θ = ½atan(R/L) (rad)"
ONE_D_AXES = (CORRECTED_THETA, CORRECTED_TWOTHETA, CORRECTED_Q, LEGACY_THETA)


@dataclass(frozen=True)
class AxisSpec:
    integration_unit: str
    display_axis: str


def get_1d_axis_spec(option: str, rot1: float, rot2: float, rot3: float) -> AxisSpec:
    """Choose native pyFAI binning. The legacy radius conversion disallows tilt."""
    if option == CORRECTED_THETA:
        return AxisSpec("2th_rad", "theta_corrected_rad")
    if option == CORRECTED_TWOTHETA:
        return AxisSpec("2th_deg", "2th_deg")
    if option == CORRECTED_Q:
        return AxisSpec("q_A^-1", "q_A^-1")
    if option == LEGACY_THETA:
        if any(abs(float(angle)) > 1e-12 for angle in (rot1, rot2, rot3)):
            raise ValueError("Legacy radius-to-theta conversion is not valid for tilted detectors. Choose a corrected 1D axis.")
        return AxisSpec("r_mm", "theta_custom_rad")
    raise ValueError(f"Unsupported 1D axis choice: {option!r}")


def format_1d_axis(radial: np.ndarray, *, display_axis: str, distance_m: float) -> np.ndarray:
    """Transform bin-center coordinates without altering integrated intensities."""
    bins = np.asarray(radial, dtype=np.float64)
    if bins.ndim != 1:
        raise ValueError("Expected a 1D radial axis")
    if display_axis == "theta_corrected_rad":
        return np.ascontiguousarray(bins * 0.5)
    if display_axis == "theta_custom_rad":
        distance = float(distance_m)
        if distance <= 0 or not np.isfinite(distance):
            raise ValueError("Distance must be positive")
        return np.ascontiguousarray(0.5 * np.arctan(bins * 1e-3 / distance))
    if display_axis in ("2th_deg", "q_A^-1"):
        return np.ascontiguousarray(bins)
    raise ValueError(f"Unsupported display axis: {display_axis}")


def two_theta_degrees(ai, shape: tuple[int, int]) -> np.ndarray:
    """Ask pyFAI for its true per-pixel scattering angles, with no image warp."""
    geometry = np.asarray(ai.twoThetaArray(shape), dtype=np.float64)
    if geometry.shape != tuple(shape):
        raise ValueError("pyFAI returned an incompatible geometry map")
    if not np.all(np.isfinite(geometry)):
        raise ValueError("pyFAI produced invalid two-theta coordinates")
    return np.degrees(geometry)


def color_geometry_map(values: np.ndarray, *, delta: bool = False) -> tuple[np.ndarray, float, float]:
    """Stable RGB depiction for a geometry map (not a detector image)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("Geometry map must contain finite 2D values")
    if delta:
        limit = max(float(np.max(np.abs(arr))), 1e-12)
        scaled = np.clip(arr / limit, -1., 1.)
        magnitude = np.abs(scaled)
        rgb = np.empty((*arr.shape, 3), dtype=np.uint8)
        rgb[..., 0] = np.where(scaled >= 0, 235, 230*(1-magnitude)+45*magnitude).astype(np.uint8)
        rgb[..., 1] = (238*(1-magnitude)+65*magnitude).astype(np.uint8)
        rgb[..., 2] = np.where(scaled < 0, 250, 235*(1-magnitude)+55*magnitude).astype(np.uint8)
        return rgb, -limit, limit
    lo, hi = float(arr.min()), float(arr.max())
    normalized = np.clip((arr-lo)/max(hi-lo, 1e-12), 0, 1)
    channel = (normalized * 225 + 20).astype(np.uint8)
    return np.stack([channel, channel, channel], axis=2), lo, hi
