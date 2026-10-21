#!/usr/bin/env python3
"""Scientific integration *chart* export (not detector-frame or raw-data export).

This module never modifies scientific result arrays or source HDF5/HDXF data.
Matplotlib's noninteractive Figure/Agg API is used so saving charts does not
require a screenshot, an exposed window, or a screen-resolution-dependent grab.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

PLOT_FORMATS: dict[str, str] = {
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".svg": "svg",
    ".pdf": "pdf",
}

BG = "#030711"
FG = "#EAF2FF"
MUTED = "#8EA4C5"
ACCENT = "#38BDF8"
BLUE = "#60A5FA"
PINK = "#F472B6"
ORANGE = "#FB923C"
GREEN = "#34D399"
YELLOW = "#FBBF24"

TRACK_LABELS = {
    "included": "Spots included",
    "spots_masked": "Hot/Bragg spots masked",
    "sigma_clipped": "Ring MAD sigma-clipped (notebook)",
}


def _axis_label(unit: str) -> str:
    return {
        "theta_custom_rad": "Legacy theta = 1/2 atan(R/L) (rad)",
        "theta_corrected_rad": "theta (rad; geometry corrected)",
        "q_A^-1": "q (1/Angstrom)",
        "q_nm^-1": "q (1/nm)",
        "2th_deg": "2theta (degrees)",
        "2th_rad": "2theta (rad)",
        "r_mm": "Radius (mm)",
    }.get(str(unit), str(unit))


def _new_figure():
    # Import on demand: normal Viewer startup stays fast and independent of a
    # matplotlib GUI backend. Figure.savefig uses the noninteractive Agg canvas.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure = Figure(figsize=(11.5, 7.1), facecolor=BG)
    FigureCanvasAgg(figure)
    ax = figure.add_axes([0.11, 0.16, 0.83, 0.68], facecolor=BG)
    ax.tick_params(axis="both", colors=MUTED, labelsize=10, length=3)
    for spine in ax.spines.values():
        spine.set_color("#344862")
    ax.grid(True, color="#263854", alpha=0.38, linewidth=0.5)
    ax.set_axisbelow(True)
    return figure, ax


def _fig_titles(figure, title: str, subtitle: str = ""):
    figure.text(0.055, 0.955, title, ha="left", va="top", color=FG,
                fontsize=14, fontweight="bold")
    if subtitle:
        figure.text(0.055, 0.918, subtitle, ha="left", va="top", color=MUTED,
                    fontsize=9)


def _finish_1d_axes(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, color=MUTED, labelpad=12, fontsize=11)
    ax.set_ylabel(ylabel, color=MUTED, labelpad=10, fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(FG)


def _plot_line(ax, x: np.ndarray, y: np.ndarray, *, color: str, label: str,
               radius: np.ndarray | None = None, exclude_radius: float = -math.inf) -> bool:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("Curve x and intensity arrays must be equal-length 1D arrays")
    valid_x = np.isfinite(x)
    if radius is not None:
        radius = np.asarray(radius, dtype=np.float64)
        if radius.shape != x.shape:
            raise ValueError("Notebook radial coordinate shape mismatch")
        valid_x &= np.isfinite(radius) & (radius > float(exclude_radius))
    # NaNs intentionally produce breaks: fully masked bins are not zero-valued.
    plot_y = np.where(valid_x & np.isfinite(y), y, np.nan)
    if np.count_nonzero(np.isfinite(plot_y)) == 0:
        return False
    ax.plot(x, plot_y, color=color, linewidth=1.25, label=label)
    return True


def _notebook_robust_ylim(ax, radius: np.ndarray, exclude: float,
                          curves: tuple[np.ndarray, ...]) -> None:
    """Match the Notebook result window's robust (99th pct) top limit.

    Rare high spike intensities should not flatten all other curves when a
    publication chart is generated from the same result arrays.
    """
    radius = np.asarray(radius, dtype=np.float64)
    vals = []
    for arr in curves:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape != radius.shape:
            raise ValueError("Notebook curve / radius shape mismatch")
        keep = np.isfinite(radius) & (radius > exclude) & np.isfinite(arr)
        if np.any(keep):
            vals.append(arr[keep])
    if not vals:
        return
    joined = np.concatenate(vals)
    ymin = min(0.0, float(np.min(joined)))
    ymax = max(1e-9, float(np.percentile(joined, 99.0)) * 1.4)
    if ymax <= ymin:
        ymax = ymin + 1.0
    ax.set_ylim(ymin, ymax)


def _draw_notebook(ax, result: dict[str, Any], mode: str, track: str) -> tuple[str, str, str]:
    profiles = result["profiles"]
    cfg = result["config"]
    exclude = float(cfg.get("exclude_radius_px", 0.0))
    comparison = result.get("mask_comparison")
    position = f"({float(profiles.get('beam_x', 0.0)):.2f}, {float(profiles.get('beam_y', 0.0)):.2f})"
    fit_label = "Gaussian-fit beam" if cfg.get("fit_beam") else "Manual beam"
    subtitle = f"{fit_label}: {position}  |  Center exclusion: {exclude:g} pixels"

    if comparison and mode in ("Before / After", "Difference Δ"):
        choice = track if track in TRACK_LABELS else "included"
        data = comparison["tracks"][choice]
        theta = np.asarray(data["two_theta_deg"], dtype=np.float64)
        radius = np.asarray(data["radius_px"], dtype=np.float64)
        if mode == "Difference Δ":
            drawn = _plot_line(ax, theta, data["delta"], radius=radius,
                               exclude_radius=exclude, color=YELLOW,
                               label="Delta = After - Before")
            ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
            ylabel = "Delta intensity (after - before)"
            # Keep the difference range symmetric to make signed changes clear.
            keep = np.isfinite(data["delta"]) & np.isfinite(theta) & (radius > exclude)
            if np.any(keep):
                amp = max(1e-12, float(np.max(np.abs(np.asarray(data["delta"])[keep]))) * 1.10)
                ax.set_ylim(-amp, amp)
        else:
            drawn = _plot_line(ax, theta, data["before"], radius=radius,
                               exclude_radius=exclude, color=BLUE, label="Before manual Mask")
            drawn = _plot_line(ax, theta, data["after"], radius=radius,
                               exclude_radius=exclude, color=PINK, label="After manual Mask") or drawn
            ylabel = "Mean intensity per ring"
        if mode != "Difference Δ":
            _notebook_robust_ylim(ax, radius, exclude, (data["before"], data["after"]))
        subtitle += (f"  |  Manual Mask: {int(comparison['manual_pixels']):,} pixels"
                     f"  |  Changed bins: {int(data['changed_bins']):,}/{int(data['compared_bins']):,}"
                     f"  |  Max |Delta|: {float(data['max_abs_delta']):.6g}")
        if not drawn:
            ax.text(0.5, 0.5, "No valid radial bins outside center exclusion",
                    transform=ax.transAxes, ha="center", va="center", color=YELLOW)
        return "Notebook Mask comparison - " + TRACK_LABELS[choice], subtitle, ylabel

    # Original three independent notebook reductions (unchanged numeric data).
    series = (("included", BLUE), ("spots_masked", ORANGE), ("sigma_clipped", GREEN))
    drawn = False
    for key, color in series:
        entry = profiles[key]
        drawn = _plot_line(ax, entry["two_theta_deg"], entry["intensity"],
                           radius=entry["radius_px"], exclude_radius=exclude,
                           color=color, label=TRACK_LABELS[key]) or drawn
    if not drawn:
        ax.text(0.5, 0.5, "No valid radial bins", transform=ax.transAxes,
                ha="center", va="center", color=YELLOW)
    else:
        _notebook_robust_ylim(ax, profiles["included"]["radius_px"], exclude,
                              tuple(profiles[key]["intensity"] for key, _color in series))
    return "Notebook 1D Reference", subtitle, "Mean intensity per ring"


def _draw_1d(ax, result: dict[str, Any]) -> tuple[str, str]:
    x = np.asarray(result["radial"], dtype=np.float64)
    y = np.asarray(result["intensity"], dtype=np.float64)
    if not _plot_line(ax, x, y, color=ACCENT, label="Integrated average intensity"):
        ax.text(0.5, 0.5, "No finite integrated points", color=YELLOW,
                ha="center", va="center", transform=ax.transAxes)
    src = result["source"]
    n = int(src.get("frame_count_averaged", 1))
    return f"All {n:,} frames averaged", "Average intensity"


def _draw_2d(ax, result: dict[str, Any]) -> tuple[str, str]:
    data = np.asarray(result["intensity"], dtype=np.float64)
    radial = np.asarray(result["radial"], dtype=np.float64)
    azimuthal = np.asarray(result["azimuthal"], dtype=np.float64)
    if data.ndim != 2 or radial.size != data.shape[1] or azimuthal.size != data.shape[0]:
        raise ValueError("2D intensity/radial/azimuthal dimensions do not match")
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        ax.text(0.5, 0.5, "No finite integrated intensity", transform=ax.transAxes,
                ha="center", va="center", color=YELLOW)
        return "pyFAI 2D Cake", "No finite data"
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    norm = np.log1p(norm * 30.0) / math.log(31.0)
    gray = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    # Row 0 remains visually at the *top*, as in the Tk Viewer. Assign each
    # row its true azimuthal coordinate: top=azimuthal[0], bottom=azimuthal[-1].
    # This intentionally corrects the historic inverted Tk tick-end labels.
    extent = (float(radial[0]), float(radial[-1]),
              float(azimuthal[-1]), float(azimuthal[0]))
    ax.imshow(gray, cmap="gray", vmin=0.0, vmax=1.0,
              origin="upper", interpolation="nearest", aspect="auto", extent=extent)
    ax.set_ylabel("Azimuth chi (degrees)", color=MUTED, labelpad=10, fontsize=11)
    ax.grid(False)
    return "pyFAI 2D Cake", f"display {lo:.4g} ... {hi:.4g} (same log/percentile contrast as Viewer)"


def build_integration_figure(result: dict[str, Any], *, comparison_mode: str = "Notebook 3 curves",
                             comparison_track: str = "included"):
    """Return a standalone Matplotlib Figure; no Tk objects/screenshots needed."""
    figure, ax = _new_figure()
    kind = str(result.get("kind", ""))
    source = result["source"]
    name = str(source.get("filename") or "Detector data")
    if kind == "notebook_1d":
        heading, info, ylabel = _draw_notebook(ax, result, comparison_mode, comparison_track)
        _finish_1d_axes(ax, "2theta (degrees)", ylabel)
    elif kind == "1d":
        heading, info = _draw_1d(ax, result)
        _finish_1d_axes(ax, _axis_label(result["config"].get("one_d_axis", "theta_custom_rad")), "Average intensity")
    elif kind == "2d":
        heading, info = _draw_2d(ax, result)
        ax.set_xlabel(_axis_label(result["config"]["unit"]), color=MUTED,
                      fontsize=11, labelpad=12)
    else:
        raise ValueError(f"Unsupported integration chart kind: {kind!r}")
    if kind == "notebook_1d":
        scope = str(source.get("average_scope") or "")
        info = (scope + "  |  " + info).strip(" |")
    elif kind == "2d":
        info = f"Frame {int(source['frame_index'])+1}  |  {info}"
    _fig_titles(figure, f"{heading}  -  {name}", info)
    return figure


def export_integration_chart(result: dict[str, Any], destination: str | Path,
                             *, comparison_mode: str = "Notebook 3 curves",
                             comparison_track: str = "included", dpi: int = 180) -> Path:
    """Atomically export currently selected chart to PNG/JPEG/SVG/PDF.

    Chart is derived from integration result arrays; raw data exports stay in
    the Viewer's separate Export Data action. A failed write leaves an existing
    destination file intact, including failures inside Matplotlib.
    """
    path = Path(destination)
    file_format = PLOT_FORMATS.get(path.suffix.lower())
    if file_format is None:
        raise ValueError("Chart extension must be .png, .jpg, .jpeg, .svg or .pdf")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Chart export directory not found: {path.parent}")
    if not (72 <= int(dpi) <= 600):
        raise ValueError("Chart DPI must be between 72 and 600")
    figure = build_integration_figure(result,
                                     comparison_mode=comparison_mode,
                                     comparison_track=comparison_track)
    fd = None
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent)
        os.close(fd)
        fd = None
        figure.savefig(temp_path, format=file_format, dpi=int(dpi), facecolor=BG,
                       edgecolor=BG, bbox_inches="tight", pad_inches=0.2)
        os.replace(temp_path, path)
        temp_path = None
        return path
    finally:
        if fd is not None:
            os.close(fd)
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        figure.clear()
