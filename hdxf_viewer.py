#!/usr/bin/env python3
"""Desktop viewer for HDXF and HDF5 detector-frame data.

Supported HDXF frame encodings:
  - HDXFB v1: whole-block Raw and Delta
  - HDXFB v2: tile-adaptive constant/repeat/static/sparse/bitpack/
    Jungfrau split/Raw
  - HDXFB v3/v4: chained Delta, shared static models, zero-RLE bitpack

Features:
  - Open HDXF and HDF5 detector files from dialogs or Windows Explorer drag-and-drop
  - Frame navigation
  - Mouse-wheel zoom centered under the cursor
  - Mouse-drag pan
  - Temporary manual beam-center anchor: left click to set, right click to clear
  - Multi-shape ROI: Rectangle, Ellipse, Circle, Annulus, Polygon and Freehand using the right mouse button
  - Beam-center anchor stays fixed while changing frames and never writes source files
  - Fit-to-window and 1:1 display
  - Automatic or manual grayscale range
  - Raw pixel value under the cursor
  - High-contrast two-tone cursor overlay
  - Pixel-value overlay at high zoom
  - Up to four internal views in one application window
  - Display parameters synchronized across all internal views
  - Small LRU block cache: large frame counts do not load into RAM at once
"""

from __future__ import annotations

import argparse
import base64
import io
import bisect
import hashlib
import json
import math
import os
import queue
import re
import shlex
import struct
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

# Register common detector HDF5 filters before direct HDF5 reads.
# Without this import, Jungfrau/Eiger-style datasets can expose shape/dtype
# successfully but fail when the Viewer actually requests pixel values.
try:
    import hdf5plugin  # type: ignore  # noqa: F401
    HDF5PLUGIN_AVAILABLE = True
    HDF5PLUGIN_IMPORT_ERROR = None
except Exception as exc:
    hdf5plugin = None  # type: ignore
    HDF5PLUGIN_AVAILABLE = False
    HDF5PLUGIN_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

from hdxf_codec import HDXFBError, decode_hdxfb as decode_hdxfb_payload
from hdxf_index import HDXFIError, unpack_frame_index

try:
    import blosc2
except ImportError as exc:
    raise SystemExit(
        "blosc2 is required. Install dependencies with:\n"
        "  python -m pip install -r requirements.txt"
    ) from exc

try:
    from PIL import Image, ImageTk
except ImportError as exc:
    raise SystemExit(
        "Pillow is required. Install dependencies with:\n"
        "  python -m pip install -r requirements.txt"
    ) from exc


def enable_windows_high_dpi_awareness() -> str:
    """Enable crisp native rendering before the first Tk window is created.

    Windows otherwise treats a plain Python/Tk process as DPI-unaware on some
    installations and bitmap-scales the whole application.  That makes both
    Tk widgets and the native Open-file dialog visibly blurry.  Per-monitor V2
    is preferred, with older Windows APIs retained as safe fallbacks.
    """
    if sys.platform != "win32":
        return "not-windows"

    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        process_mode: str | None = None

        setter = getattr(user32, "SetProcessDpiAwarenessContext", None)
        if setter is not None:
            setter.argtypes = [wintypes.HANDLE]
            setter.restype = wintypes.BOOL
            # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == (HANDLE)-4
            if setter(ctypes.c_void_p(-4)):
                process_mode = "per-monitor-v2"
            elif ctypes.get_last_error() == 5:
                # Python or a launcher may already have selected a process DPI
                # mode.  A thread-level V2 context below can still upgrade the
                # Tk GUI thread and native file dialogs.
                process_mode = "process-already-configured"

        thread_setter = getattr(user32, "SetThreadDpiAwarenessContext", None)
        if thread_setter is not None:
            thread_setter.argtypes = [wintypes.HANDLE]
            thread_setter.restype = wintypes.HANDLE
            previous = thread_setter(ctypes.c_void_p(-4))
            if previous:
                return "per-monitor-v2"

        if process_mode is not None:
            return process_mode

        try:
            shcore = ctypes.WinDLL("shcore", use_last_error=True)
            legacy_setter = shcore.SetProcessDpiAwareness
            legacy_setter.argtypes = [ctypes.c_int]
            legacy_setter.restype = ctypes.c_long
            result = int(legacy_setter(2))  # PROCESS_PER_MONITOR_DPI_AWARE
            if result in (0, -2147024891):  # S_OK or E_ACCESSDENIED
                return "per-monitor-v1" if result == 0 else "already-configured"
        except (AttributeError, OSError):
            pass

        fallback = getattr(user32, "SetProcessDPIAware", None)
        if fallback is not None:
            fallback.restype = wintypes.BOOL
            if fallback():
                return "system-aware"
            if ctypes.get_last_error() == 5:
                return "already-configured"
    except Exception:
        # High-DPI support is an enhancement.  Never prevent the viewer from
        # starting on unusual Python/Windows combinations.
        pass
    return "unavailable"


import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont


FORMAT_NAME = "HDF5-derived Detector eXchange Format"
FORMAT_ID = "hdxf"
FORMAT_VERSION = "0.4.8.0"
SUPPORTED_FORMAT_VERSIONS = {
    "0.3.0", "0.4.0", "0.4.2", "0.4.3", "0.4.4", "0.4.5",
    "0.4.6", "0.4.7", "0.4.8"
}
HDXFB_FORMAT = "hdxfb"
HDXFB_VERSIONS = (1, 2, 3, 4)
HDXFB_MAGIC = b"HDXFB\x00\x01"
MANIFEST_PATH = "manifest.json"


def _default_legacy_template_dir() -> Path:
    """Use the project's current template directory, accepting old spelling."""
    base = Path(__file__).resolve().parent
    for candidate in (
        base / "legacy_template",
        base / "legacy-template",
    ):
        if candidate.is_dir():
            return candidate
    return base / "legacy_template"


class SafeFileDropController:
    """Enable Explorer file drops through TkDND's ordinary Tk event path.

    The root window must be created as ``TkinterDnD.Tk``.  Registering only the
    root is not sufficient for this application because the visible image
    canvases are child windows and become the actual native drop targets under
    the pointer.  This controller therefore registers the root, the image
    workspace and every internal View/Canvas explicitly.
    """

    def __init__(self, root: tk.Tk, callback) -> None:
        self.root = root
        self.callback = callback
        self.enabled = False
        self.status = "not-installed"
        self._dnd_files: str | None = None
        self._copy_action: str = "copy"
        self._targets: dict[str, tk.Misc] = {}

    def _on_drop(self, event):
        try:
            paths = tuple(str(item) for item in self.root.tk.splitlist(event.data))
            screen_x = int(getattr(event, "x_root", self.root.winfo_pointerx()))
            screen_y = int(getattr(event, "y_root", self.root.winfo_pointery()))
            if paths:
                self.callback(paths, screen_x, screen_y)
        except (tk.TclError, ValueError, TypeError):
            return "refuse_drop"
        return self._copy_action

    def register_targets(self, widgets) -> int:
        """Register newly created visible widgets as DND_FILES targets."""
        if not self.enabled or self._dnd_files is None:
            return 0
        registered = 0
        for widget in widgets:
            if widget is None:
                continue
            try:
                key = str(widget)
                if key in self._targets or not widget.winfo_exists():
                    continue
                if not hasattr(widget, "drop_target_register") or not hasattr(widget, "dnd_bind"):
                    continue
                widget.drop_target_register(self._dnd_files)
                widget.dnd_bind("<<Drop>>", self._on_drop, add="+")
                self._targets[key] = widget
                registered += 1
            except tk.TclError:
                continue
        self.status = f"tkdnd: {len(self._targets)} targets"
        return registered

    def install(self, widgets=()) -> str:
        if self.enabled:
            self.register_targets(widgets)
            return self.status
        try:
            from tkinterdnd2 import DND_FILES, COPY

            if getattr(self.root, "_hdxf_dnd_backend", "") != "tkinterdnd2":
                raise RuntimeError("main window was not created with TkinterDnD.Tk")
            if not hasattr(self.root, "drop_target_register") or not hasattr(self.root, "dnd_bind"):
                raise RuntimeError("TkDND widget methods are unavailable")
            self._dnd_files = DND_FILES
            self._copy_action = COPY
            self.enabled = True
            self.register_targets((self.root, *tuple(widgets)))
            if not self._targets:
                raise RuntimeError("no visible drop target could be registered")
        except ImportError:
            self.status = "unavailable: install tkinterdnd2"
            self.enabled = False
        except Exception as exc:
            self.status = f"unavailable: {type(exc).__name__}: {exc}"
            self.enabled = False
            self._dnd_files = None
            self._targets.clear()
        return self.status

    def close(self) -> None:
        for widget in list(self._targets.values()):
            try:
                if widget.winfo_exists() and hasattr(widget, "drop_target_unregister"):
                    widget.drop_target_unregister()
            except tk.TclError:
                pass
        self._targets.clear()
        self.enabled = False
        self._dnd_files = None


# Modern scientific dark theme.  The palette deliberately uses a restrained
# blue/cyan accent so dense detector information remains readable for long
# sessions without looking like a generic gray Tk application.
UI_BG = "#070B14"
UI_TOPBAR = "#0B1220"
UI_SURFACE = "#101A2C"
UI_SURFACE_RAISED = "#152238"
UI_CARD = "#111D31"
UI_CARD_ALT = "#17263D"
UI_BORDER = "#263854"
UI_BORDER_SOFT = "#1C2A41"
UI_TEXT = "#EAF2FF"
UI_TEXT_MUTED = "#8EA4C5"
UI_TEXT_DIM = "#647A9B"
UI_ACCENT = "#38BDF8"
UI_ACCENT_STRONG = "#2563EB"
UI_ACCENT_HOVER = "#60A5FA"
UI_ACCENT_DARK = "#123B5A"
UI_SUCCESS = "#34D399"
UI_WARNING = "#F59E0B"
UI_DANGER = "#F87171"
UI_CANVAS = "#030711"
UI_HISTOGRAM = "#0C1424"
UI_WHITE = "#FFFFFF"


class WindowsDPIController:
    """Keep Tk point scaling aligned with the monitor hosting the main window."""

    POLL_MS = 600

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.dpi = 96
        self.ui_scale = 1.0
        self._after_id: str | None = None
        self._closed = False
        self.refresh(force=True)

    def _query_window_dpi(self) -> int:
        if sys.platform != "win32":
            try:
                return max(72, int(round(float(self.root.winfo_fpixels("1i")))))
            except Exception:
                return 96
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.WinDLL("user32", use_last_error=True)
            getter = getattr(user32, "GetDpiForWindow", None)
            if getter is not None:
                getter.argtypes = [wintypes.HWND]
                getter.restype = wintypes.UINT
                dpi = int(getter(int(self.root.winfo_id())))
                if dpi > 0:
                    return dpi

            system_getter = getattr(user32, "GetDpiForSystem", None)
            if system_getter is not None:
                system_getter.restype = wintypes.UINT
                dpi = int(system_getter())
                if dpi > 0:
                    return dpi

            hdc = user32.GetDC(0)
            if hdc:
                try:
                    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
                    dpi = int(gdi32.GetDeviceCaps(hdc, 88))  # LOGPIXELSX
                    if dpi > 0:
                        return dpi
                finally:
                    user32.ReleaseDC(0, hdc)
        except Exception:
            pass
        return 96

    def refresh(self, *, force: bool = False) -> bool:
        try:
            self.root.update_idletasks()
            dpi = self._query_window_dpi()
            if not force and dpi == self.dpi:
                return False
            self.dpi = dpi
            self.ui_scale = max(0.75, min(4.0, dpi / 96.0))
            # Tk font sizes are specified in points.  One point is 1/72 inch.
            self.root.tk.call("tk", "scaling", dpi / 72.0)
            setattr(self.root, "_hdxf_dpi", dpi)
            setattr(self.root, "_hdxf_ui_scale", self.ui_scale)
            if not force:
                self.root.event_generate("<<HDXFDPIChanged>>", when="tail")
            return True
        except tk.TclError:
            return False

    def start(self) -> None:
        if self._after_id is None and not self._closed:
            self._after_id = self.root.after(self.POLL_MS, self._poll)

    def _poll(self) -> None:
        self._after_id = None
        if self._closed:
            return
        self.refresh()
        self.start()

    def close(self) -> None:
        self._closed = True
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None


class HDXFViewerError(RuntimeError):
    """Raised for unsupported, unreadable, or damaged detector data."""


def _decode_dtype(value: Any) -> np.dtype[Any]:
    """Decode the simple NumPy dtype descriptors used by detector frames."""
    if isinstance(value, str):
        return np.dtype(value)

    # NumPy structured dtype descriptions may be JSON lists. Detector frames
    # normally use a scalar dtype, but accepting this form costs little.
    if isinstance(value, list):
        def tuples(item: Any) -> Any:
            if isinstance(item, list):
                return tuple(tuples(x) for x in item)
            return item
        return np.dtype([tuples(item) for item in value])

    if isinstance(value, dict) and value.get("$type") == "bytes":
        import base64
        return np.dtype(base64.b64decode(value["base64"]).decode("ascii"))

    raise HDXFViewerError(f"Unsupported dtype descriptor: {value!r}")


def _format_value(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.6g}"
    return str(value)


def _format_overlay_value(value: Any) -> str:
    """Format only the high-zoom text drawn inside image pixels.

    Floating-point values are rounded to two decimal places so neighbouring
    labels remain readable.  Integer detector counts remain integers.  This
    formatter is deliberately not used by the pointer readout, Image Info,
    Histogram controls, metadata, or archive data.
    """
    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.integer):
            return str(int(value))
        if np.issubdtype(value.dtype, np.floating):
            value = float(value)
        else:
            value = value.item()

    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        rounded = round(value, 2)
        if rounded == 0.0:
            rounded = 0.0  # avoid displaying -0.00
        return f"{rounded:.2f}"
    return _format_value(value)


def _decode_tagged_value(value: Any) -> Any:
    """Decode the strict JSON tags used for generic HDF5 payload values."""
    if isinstance(value, list):
        return [_decode_tagged_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    tag = value.get("$type")
    if tag == "bytes":
        return base64.b64decode(str(value.get("base64", "")))
    if tag == "float":
        token = str(value.get("value", "nan"))
        return {"nan": float("nan"), "+inf": float("inf"), "-inf": float("-inf")}.get(token, float("nan"))
    if tag == "complex":
        return complex(_decode_tagged_value(value.get("real")), _decode_tagged_value(value.get("imag")))
    if tag == "ndarray_npy":
        raw = base64.b64decode(str(value.get("base64", "")))
        return np.load(io.BytesIO(raw), allow_pickle=False)
    if tag == "object_array":
        shape = tuple(int(x) for x in value.get("shape", []))
        items = [_decode_tagged_value(item) for item in value.get("items", [])]
        return np.asarray(items, dtype=object).reshape(shape)
    return {str(k): _decode_tagged_value(v) for k, v in value.items()}


def _scalar_text(value: Any) -> str:
    """Return a readable scalar/string representation from an HDF5 value."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.size == 1:
            value = value.reshape(-1)[0]
        else:
            return ", ".join(_scalar_text(item) for item in value.reshape(-1)[:8])
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.rstrip(b"\x00").decode("utf-8", errors="replace")
    return _format_value(value)


def _numeric_scalar(value: Any, *, index: int | None = None) -> float | None:
    try:
        array = np.asarray(value)
        if array.size == 0:
            return None
        if index is not None and array.ndim >= 1 and array.shape[0] > 1:
            selected = array[min(max(0, int(index)), int(array.shape[0]) - 1)]
            array = np.asarray(selected)
        scalar = array.reshape(-1)[0]
        result = float(scalar)
        return result if math.isfinite(result) else None
    except Exception:
        return None


AUTO_CONTRAST_VISIBLE_SPARSE = "Visible detail — low counts"
AUTO_CONTRAST_FRAME_SPARSE = "Whole frame — sparse 99.9%"
AUTO_CONTRAST_FRAME_ROBUST = "Whole frame — robust wide range"
AUTO_CONTRAST_MODES = (
    AUTO_CONTRAST_VISIBLE_SPARSE,
    AUTO_CONTRAST_FRAME_SPARSE,
    AUTO_CONTRAST_FRAME_ROBUST,
)

ROI_SHAPES = (
    "Rectangle",
    "Ellipse",
    "Circle",
    "Annulus",
    "Polygon",
    "Freehand",
)



def _detector_values(array: np.ndarray, *, max_samples: int = 4_000_000) -> np.ndarray:
    """Return finite detector values with integer minimum sentinels removed.

    A deterministic stride is used for unusually large arrays so that contrast
    calculation stays responsive without introducing frame-to-frame randomness.
    """
    values = np.asarray(array).reshape(-1)
    if values.size > max_samples:
        stride = max(1, int(math.ceil(values.size / max_samples)))
        values = values[::stride]

    if array.dtype.kind == "f":
        values = values[np.isfinite(values)]
    elif array.dtype.kind == "i":
        info = np.iinfo(array.dtype)
        values = values[values != info.min]
    return values


def _expand_degenerate_range(low: float, high: float) -> tuple[float, float]:
    if not math.isfinite(low) or not math.isfinite(high):
        return 0.0, 1.0
    if high > low:
        return float(low), float(high)
    padding = max(1.0, abs(float(low)) * 0.01)
    return float(low) - padding, float(high) + padding


def _integer_detail_ceiling(positive: np.ndarray) -> tuple[float, dict[str, float]]:
    """Find the low-count cluster in an integer detector signal.

    Sparse detector frames often contain a scientifically useful population at
    1, 2, 3, ... plus a tiny tail of strong Bragg peaks or saturated pixels.
    A conventional percentile can still land in that bright tail. This helper
    deliberately identifies the low-count cluster instead.
    """
    data = positive.astype(np.float64, copy=False)
    p10, p25, p50, p75, p90, p95, p99 = np.percentile(
        data, [10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0]
    )

    # Build a ceiling around the lower population. For counts near 1..3 this
    # typically produces a cluster limit of 8..16, excluding isolated peaks at
    # thousands without hiding the low-count structure.
    base = max(1.0, float(p25), min(float(p50), 64.0))
    cluster_limit = min(512.0, max(8.0, base * 6.0))
    cluster = data[data <= cluster_limit]

    # Require a meaningful low-count population, not just one accidental pixel.
    required = max(4, int(math.ceil(data.size * 0.01)))
    if cluster.size >= required:
        high = float(np.percentile(cluster, 99.0))
        high = float(max(1.0, math.ceil(high + 0.25)))
        source = 1.0  # low-count cluster
    else:
        high = float(p90)
        source = 0.0  # percentile fallback

    return high, {
        "positive_p10": float(p10),
        "positive_p25": float(p25),
        "positive_p50": float(p50),
        "positive_p75": float(p75),
        "positive_p90": float(p90),
        "positive_p95": float(p95),
        "positive_p99": float(p99),
        "detail_cluster_limit": float(cluster_limit),
        "detail_cluster_samples": float(cluster.size),
        "detail_cluster_used": source,
    }


def _sparse_frame_ceiling(positive: np.ndarray, *, integer: bool) -> tuple[float, dict[str, float]]:
    """Choose a broader whole-frame sparse-signal ceiling.

    This mode intentionally includes more of the medium-strength signal than
    the visible-detail mode, while rejecting the extreme hot-pixel/saturation
    tail through a robust log-domain limit.
    """
    data = positive.astype(np.float64, copy=False)
    p50, p90, p95, p98, p99, p995, p999 = np.percentile(
        data, [50.0, 90.0, 95.0, 98.0, 99.0, 99.5, 99.9]
    )
    logs = np.log1p(data)
    log_med = float(np.median(logs))
    log_mad = float(np.median(np.abs(logs - log_med)))
    if log_mad > 0.0:
        robust_limit = float(np.expm1(log_med + 8.0 * 1.4826 * log_mad))
        # Whole-frame sparse mode is intentionally broader than visible
        # detail. Keep up to the 99.9th percentile of non-zero signal, while
        # using the log-domain limit to avoid a single saturated pixel.
        high = min(float(p999), max(float(p995), robust_limit))
    else:
        high = float(p999)

    if integer and high <= 64.0:
        high = float(max(1.0, math.ceil(high + 0.5)))

    return high, {
        "positive_p50": float(p50),
        "positive_p90": float(p90),
        "positive_p95": float(p95),
        "positive_p98": float(p98),
        "positive_p99": float(p99),
        "positive_p995": float(p995),
        "positive_p999": float(p999),
        "log_median": log_med,
        "log_mad": log_mad,
        "sparse_robust_limit": float(robust_limit if log_mad > 0.0 else p999),
    }


def calculate_auto_contrast(
    array: np.ndarray,
    *,
    mode: str,
) -> tuple[float, float, dict[str, float]]:
    """Calculate one of three deliberately distinct display ranges."""
    values = _detector_values(array)
    if values.size == 0:
        return 0.0, 1.0, {"samples": 0.0, "zero_fraction": 0.0}

    zero_fraction = float(np.count_nonzero(values == 0) / values.size)
    positive = values[values > 0]
    negative = values[values < 0]
    stats: dict[str, float] = {
        "samples": float(values.size),
        "zero_fraction": zero_fraction,
        "positive_samples": float(positive.size),
        "negative_samples": float(negative.size),
    }

    if mode == AUTO_CONTRAST_VISIBLE_SPARSE:
        if zero_fraction >= 0.01:
            low = 0.0
        elif negative.size:
            low = float(np.percentile(negative.astype(np.float64, copy=False), 5.0))
        else:
            low = float(np.min(values))

        if positive.size:
            if np.issubdtype(array.dtype, np.integer):
                high, detail_stats = _integer_detail_ceiling(positive)
                stats.update(detail_stats)
            else:
                high = float(np.percentile(positive.astype(np.float64, copy=False), 90.0))
        else:
            low, high = np.percentile(
                values.astype(np.float64, copy=False), [1.0, 99.0]
            )
            low, high = float(low), float(high)

    elif mode == AUTO_CONTRAST_FRAME_SPARSE:
        low = 0.0 if zero_fraction >= 0.01 else float(
            np.percentile(values.astype(np.float64, copy=False), 1.0)
        )
        if positive.size:
            high, sparse_stats = _sparse_frame_ceiling(
                positive,
                integer=np.issubdtype(array.dtype, np.integer),
            )
            stats.update(sparse_stats)
        else:
            low, high = np.percentile(
                values.astype(np.float64, copy=False), [0.5, 99.5]
            )
            low, high = float(low), float(high)

    elif mode == AUTO_CONTRAST_FRAME_ROBUST:
        # This is intentionally the broadest automatic mode. On a
        # zero-dominated detector frame, percentiles of *all* pixels collapse
        # to zero, so calculate the bright limit from non-zero signal instead.
        if zero_fraction >= 0.50 and positive.size:
            low = 0.0
            high = float(np.percentile(
                positive.astype(np.float64, copy=False), 99.95
            ))
            if negative.size:
                low = float(min(0.0, np.percentile(
                    negative.astype(np.float64, copy=False), 0.5
                )))
        else:
            low, high = np.percentile(
                values.astype(np.float64, copy=False), [0.1, 99.9]
            )
            low, high = float(low), float(high)
        if high <= low and positive.size:
            low = 0.0 if zero_fraction else float(np.min(values))
            high = float(np.max(positive))
    else:
        raise ValueError(f"Unknown auto contrast mode: {mode!r}")

    low, high = _expand_degenerate_range(float(low), float(high))
    stats["low"] = low
    stats["high"] = high
    return low, high, stats


class HDXFArchive:
    """Random-access reader for the frame section of an HDXF archive."""

    file_format = "HDXF"

    def __init__(self, path: Path, *, block_cache_size: int = 8) -> None:
        self.path = path.resolve()
        self._zip = zipfile.ZipFile(self.path, "r")
        self._cache_size = max(1, int(block_cache_size))
        self._block_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._static_model_cache: dict[str, dict[str, np.ndarray]] = {}
        self._dataset_value_cache: dict[str, Any] = {}
        self._path_objects: dict[str, list[dict[str, Any]]] = {}

        try:
            manifest = json.loads(self._zip.read(MANIFEST_PATH))
        except KeyError as exc:
            self.close()
            raise HDXFViewerError("manifest.json is missing") from exc
        except Exception as exc:
            self.close()
            raise HDXFViewerError(f"Cannot read manifest.json: {exc}") from exc

        identity = manifest.get("hdxf", {})
        if identity.get("format_id") != FORMAT_ID:
            self.close()
            raise HDXFViewerError(
                f"Not an HDXF archive: format_id={identity.get('format_id')!r}"
            )
        if identity.get("version") not in SUPPORTED_FORMAT_VERSIONS:
            self.close()
            raise HDXFViewerError(
                f"Unsupported HDXF version: {identity.get('version')!r}. "
                f"This Viewer supports: {', '.join(sorted(SUPPORTED_FORMAT_VERSIONS))}"
            )
        if identity.get("format") != FORMAT_NAME:
            self.close()
            raise HDXFViewerError(
                f"Unexpected HDXF format name: {identity.get('format')!r}"
            )
        profile = identity.get("profile")
        detector = manifest.get("detector_archive")
        if not isinstance(detector, dict):
            self.close()
            raise HDXFViewerError(
                "This file does not contain detector_archive frame data."
            )

        frames = detector.get("frames")
        if not isinstance(frames, dict):
            self.close()
            raise HDXFViewerError("detector_archive.frames is missing")

        blocks = frames.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            index_doc = frames.get("block_index")
            if not isinstance(index_doc, dict):
                self.close()
                raise HDXFViewerError("No frame blocks or binary frame index were found")
            index_path = str(index_doc.get("path", ""))
            try:
                index_raw = self._zip.read(index_path)
            except KeyError as exc:
                self.close()
                raise HDXFViewerError(f"Missing frame index: {index_path}") from exc
            expected_size = index_doc.get("size_bytes")
            expected_sha = index_doc.get("sha256")
            if isinstance(expected_size, int) and len(index_raw) != expected_size:
                self.close()
                raise HDXFViewerError("Frame index size mismatch")
            if isinstance(expected_sha, str) and hashlib.sha256(index_raw).hexdigest() != expected_sha:
                self.close()
                raise HDXFViewerError("Frame index checksum mismatch")
            try:
                blocks = unpack_frame_index(
                    index_raw,
                    datasets=list(frames.get("datasets", [])),
                    frame_shape=list(frames.get("frame_shape") or []),
                )
            except HDXFIError as exc:
                self.close()
                raise HDXFViewerError(f"Invalid binary frame index: {exc}") from exc

        self.manifest = manifest
        objects_root = manifest.get("objects", {})
        if isinstance(objects_root, dict):
            for object_doc in objects_root.values():
                if not isinstance(object_doc, dict):
                    continue
                source_path = str(object_doc.get("source_path") or "")
                if source_path:
                    self._path_objects.setdefault(source_path, []).append(object_doc)
        self.profile = str(profile or "")
        self.frame_count = int(frames.get("frame_count", 0))
        self.frame_shape = tuple(int(x) for x in frames.get("frame_shape") or [])
        self.dtype = _decode_dtype(frames.get("dtype"))
        self.blocks = sorted(
            blocks,
            key=lambda item: int(item.get("global_start_frame", 0)),
        )
        self._starts = [
            int(item.get("global_start_frame", 0)) for item in self.blocks
        ]
        self._path_to_position = {str(item.get("path")): i for i, item in enumerate(self.blocks)}
        self._start_to_position = {int(item.get("global_start_frame", 0)): i for i, item in enumerate(self.blocks)}
        self._static_model_docs = {
            str(item.get("id")): item
            for item in frames.get("static_models", [])
            if isinstance(item, dict) and item.get("id")
        }

        if self.frame_count <= 0:
            self.close()
            raise HDXFViewerError("Frame count is zero")
        if len(self.frame_shape) != 2:
            self.close()
            raise HDXFViewerError(
                f"The simple viewer requires 2D frames; got {self.frame_shape}"
            )

    def close(self) -> None:
        self._block_cache.clear()
        self._static_model_cache.clear()
        self._dataset_value_cache.clear()
        self._path_objects.clear()
        if getattr(self, "_zip", None) is not None:
            self._zip.close()

    def __enter__(self) -> "HDXFArchive":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _find_block(self, frame_index: int) -> dict[str, Any]:
        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError(frame_index)

        pos = bisect.bisect_right(self._starts, frame_index) - 1
        if pos < 0:
            raise HDXFViewerError(f"No block covers frame {frame_index}")

        block = self.blocks[pos]
        start = int(block.get("global_start_frame", 0))
        count = int(block.get("frame_count", 0))
        if not (start <= frame_index < start + count):
            raise HDXFViewerError(f"No block covers frame {frame_index}")
        return block

    def get_frame(self, frame_index: int) -> np.ndarray:
        block_doc = self._find_block(frame_index)
        member_path = str(block_doc["path"])
        block = self._load_block_doc(block_doc)
        local_index = frame_index - int(block_doc["global_start_frame"])
        if local_index < 0 or local_index >= block.shape[0]:
            raise HDXFViewerError(
                f"Block {member_path} has no local frame {local_index}"
            )
        return block[local_index]

    def _cache_block(self, member_path: str, block: np.ndarray) -> None:
        self._block_cache[member_path] = block
        self._block_cache.move_to_end(member_path)
        while len(self._block_cache) > self._cache_size:
            self._block_cache.popitem(last=False)

    def _load_block_doc(self, block_doc: dict[str, Any]) -> np.ndarray:
        member_path = str(block_doc["path"])
        cached = self._block_cache.get(member_path)
        if cached is not None:
            self._block_cache.move_to_end(member_path)
            return cached

        target_pos = self._path_to_position.get(member_path)
        if target_pos is None:
            raise HDXFViewerError(f"Unknown frame payload: {member_path}")
        requires_previous = bool(block_doc.get("requires_previous_frame", False))
        if requires_previous:
            anchor_start = int(block_doc.get("chain_anchor_global_start_frame", -1))
            anchor_pos = self._start_to_position.get(anchor_start)
            if anchor_pos is None or anchor_pos > target_pos:
                raise HDXFViewerError(f"Invalid chain anchor for {member_path}")
        else:
            anchor_pos = target_pos

        previous_frame: np.ndarray | None = None
        result: np.ndarray | None = None
        for pos in range(anchor_pos, target_pos + 1):
            doc = self.blocks[pos]
            path = str(doc["path"])
            cached_part = self._block_cache.get(path)
            if cached_part is not None:
                part = cached_part
                self._block_cache.move_to_end(path)
            else:
                try:
                    payload = self._zip.read(path)
                except KeyError as exc:
                    raise HDXFViewerError(f"Missing payload: {path}") from exc
                part = self._decode_hdxfb(
                    payload, path, previous_frame=previous_frame,
                    static_model_id=doc.get("static_model_id"),
                )
                self._cache_block(path, part)
            previous_frame = np.ascontiguousarray(part[-1])
            result = part
        assert result is not None
        return result

    def _load_static_model(self, model_id: str | None) -> dict[str, np.ndarray] | None:
        if not model_id:
            return None
        model_id = str(model_id)
        cached = self._static_model_cache.get(model_id)
        if cached is not None:
            return cached
        doc = self._static_model_docs.get(model_id)
        if doc is None:
            raise HDXFViewerError(f"Missing shared static model: {model_id}")
        try:
            mask_doc = doc["mask"]
            values_doc = doc["static_values"]
            mask_payload = self._zip.read(str(mask_doc["path"]))
            values_payload = self._zip.read(str(values_doc["path"]))
            packed_mask, _ = decode_hdxfb_payload(
                mask_payload, member_path=str(mask_doc["path"]), verify_hash=True
            )
            static_values, _ = decode_hdxfb_payload(
                values_payload, member_path=str(values_doc["path"]), verify_hash=True
            )
        except (KeyError, HDXFBError) as exc:
            raise HDXFViewerError(f"Cannot load static model {model_id}: {exc}") from exc
        pixel_count = int(doc.get("pixel_count", np.prod(self.frame_shape)))
        bits = np.unpackbits(np.asarray(packed_mask, dtype=np.uint8).reshape(-1), bitorder="little")
        if bits.size < pixel_count:
            raise HDXFViewerError(f"Truncated static model mask: {model_id}")
        mask = bits[:pixel_count].astype(bool).reshape(self.frame_shape)
        values = np.ascontiguousarray(static_values).reshape(-1)
        if values.size != int(np.count_nonzero(mask)):
            raise HDXFViewerError(f"Static model value count mismatch: {model_id}")
        model = {"mask": mask, "static_values": values}
        self._static_model_cache[model_id] = model
        return model

    def _decode_hdxfb(
        self,
        payload: bytes,
        member_path: str,
        *,
        previous_frame: np.ndarray | None = None,
        static_model_id: str | None = None,
    ) -> np.ndarray:
        try:
            block, header = decode_hdxfb_payload(
                payload, member_path=member_path, verify_hash=True,
                previous_frame=previous_frame,
                static_model=self._load_static_model(static_model_id),
            )
        except HDXFBError as exc:
            raise HDXFViewerError(str(exc)) from exc
        if header.get("kind") != "frame-block":
            raise HDXFViewerError(
                f"Unsupported HDXFB kind {header.get('kind')!r}: {member_path}"
            )
        return block


    def _dataset_object(self, path: str) -> dict[str, Any] | None:
        docs = self._path_objects.get(path, [])
        for doc in docs:
            if doc.get("type") == "dataset":
                return doc
        return None

    def _decode_generic_member(self, member: dict[str, Any]) -> Any:
        member_path = str(member.get("path") or "")
        encoding = str(member.get("encoding") or "")
        raw = self._zip.read(member_path)
        expected = member.get("sha256")
        if isinstance(expected, str) and hashlib.sha256(raw).hexdigest() != expected:
            raise HDXFViewerError(f"Dataset payload checksum mismatch: {member_path}")
        if encoding == "npy":
            return np.load(io.BytesIO(raw), allow_pickle=False)
        if encoding == "json":
            doc = json.loads(raw.decode("utf-8"))
            return _decode_tagged_value(doc.get("data"))
        if encoding == "array-block-v1":
            array, _header = decode_hdxfb_payload(
                raw, member_path=member_path, verify_hash=True
            )
            return array
        raise HDXFViewerError(f"Unsupported metadata payload encoding: {encoding}")

    def read_hdf5_dataset(self, path: str) -> Any | None:
        """Read one preserved non-frame HDF5 Dataset by its original path."""
        if path in self._dataset_value_cache:
            return self._dataset_value_cache[path]
        object_doc = self._dataset_object(path)
        if object_doc is None:
            return None
        payload = object_doc.get("payload")
        if not isinstance(payload, dict) or payload.get("state") not in (None, "present"):
            return None
        encoding = str(payload.get("encoding") or "")
        if encoding in ("frame-sequence", "frame-sequence-indexed", "calibration-raw-chunks-v1"):
            return None
        members = payload.get("members")
        if not isinstance(members, list) or not members:
            return None
        parts: list[Any] = []
        try:
            for member in members:
                if isinstance(member, dict):
                    parts.append(self._decode_generic_member(member))
        except Exception:
            return None
        if not parts:
            return None
        if len(parts) == 1:
            value = parts[0]
        else:
            try:
                arrays = [np.asarray(part) for part in parts]
                value = np.concatenate(arrays, axis=0)
            except Exception:
                value = parts
        try:
            nbytes = int(np.asarray(value).nbytes)
        except Exception:
            nbytes = 0
        if nbytes <= 32 * 1024 * 1024:
            self._dataset_value_cache[path] = value
        return value

    def hdf5_dataset_attributes(self, path: str) -> dict[str, Any]:
        doc = self._dataset_object(path)
        if not isinstance(doc, dict):
            return {}
        attrs = doc.get("attributes")
        return dict(attrs) if isinstance(attrs, dict) else {}

    def find_hdf5_value(self, candidates: tuple[str, ...], *, suffixes: tuple[str, ...] = ()) -> tuple[Any | None, str | None]:
        for path in candidates:
            value = self.read_hdf5_dataset(path)
            if value is not None:
                return value, path
        if suffixes:
            lower_suffixes = tuple(item.lower() for item in suffixes)
            for path in sorted(self._path_objects):
                lower = path.lower()
                if any(lower.endswith(suffix) for suffix in lower_suffixes):
                    value = self.read_hdf5_dataset(path)
                    if value is not None:
                        return value, path
        return None, None

    def image_info(self, frame_index: int, frame: np.ndarray) -> dict[str, list[tuple[str, str]]]:
        """Build an Albula-style summary from original HDF5 values and current-frame statistics."""
        height, width = frame.shape

        def find(candidates: tuple[str, ...], suffixes: tuple[str, ...] = ()) -> tuple[Any | None, str | None]:
            return self.find_hdf5_value(candidates, suffixes=suffixes)

        def text_value(candidates: tuple[str, ...], suffixes: tuple[str, ...] = ()) -> str | None:
            value, _path = find(candidates, suffixes)
            if value is None:
                return None
            text = _scalar_text(value).strip()
            return text or None

        def number_value(candidates: tuple[str, ...], suffixes: tuple[str, ...] = (), *, per_frame: bool = False) -> tuple[float | None, str | None]:
            value, path = find(candidates, suffixes)
            return _numeric_scalar(value, index=frame_index if per_frame else None), path

        def units_for(path: str | None) -> str:
            if not path:
                return ""
            attrs = self.hdf5_dataset_attributes(path)
            units = attrs.get("units") or attrs.get("unit")
            return _scalar_text(_decode_tagged_value(units)).strip() if units is not None else ""

        def convert(value: float | None, units: str, target: str) -> float | None:
            if value is None:
                return None
            u = units.strip().lower().replace("ångström", "angstrom")
            if target == "m":
                factors = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "µm": 1e-6, "nm": 1e-9}
                return value * factors.get(u, 1.0)
            if target == "s":
                factors = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9}
                return value * factors.get(u, 1.0)
            if target == "deg":
                return math.degrees(value) if u in ("rad", "radian", "radians") else value
            if target == "angstrom":
                factors = {"m": 1e10, "nm": 10.0, "angstrom": 1.0, "a": 1.0, "å": 1.0}
                return value * factors.get(u, 1.0)
            if target == "ev":
                if u == "kev":
                    return value * 1e3
                if u in ("j", "joule", "joules"):
                    return value / 1.602176634e-19
                return value
            return value

        detector_name = text_value((
            "/entry/instrument/detector/description",
            "/entry/instrument/detector/detector_name",
            "/entry/instrument/detector/name",
            "/entry/instrument/detector/type",
        ), ("/detector/description", "/detector/detector_name"))
        x_pixel, x_path = number_value((
            "/entry/instrument/detector/x_pixel_size",
            "/entry/instrument/detector/detectorSpecific/x_pixel_size",
        ), ("/x_pixel_size",))
        y_pixel, y_path = number_value((
            "/entry/instrument/detector/y_pixel_size",
            "/entry/instrument/detector/detectorSpecific/y_pixel_size",
        ), ("/y_pixel_size",))
        thickness, thickness_path = number_value((
            "/entry/instrument/detector/sensor_thickness",
        ), ("/sensor_thickness",))

        defective: int | None = None
        mask_value, _mask_path = find((
            "/entry/instrument/detector/detectorSpecific/pixel_mask",
            "/entry/instrument/detector/pixel_mask",
        ), ("/pixel_mask",))
        if mask_value is not None:
            try:
                defective = int(np.count_nonzero(np.asarray(mask_value)))
            except Exception:
                defective = None

        detector_rows: list[tuple[str, str]] = []
        if detector_name:
            detector_rows.append(("Detector", detector_name))
        detector_rows.extend([
            ("Number of pixels in X-direction", str(width)),
            ("Number of pixels in Y-direction", str(height)),
        ])
        if x_pixel is not None:
            detector_rows.append(("Pixel size X (m)", f"{convert(x_pixel, units_for(x_path), 'm'):.6g}"))
        if y_pixel is not None and (x_pixel is None or not math.isclose(y_pixel, x_pixel)):
            detector_rows.append(("Pixel size Y (m)", f"{convert(y_pixel, units_for(y_path), 'm'):.6g}"))
        if thickness is not None:
            detector_rows.append(("Sensor thickness (m)", f"{convert(thickness, units_for(thickness_path), 'm'):.6g}"))
        if defective is not None:
            detector_rows.append(("Number of defective", str(defective)))

        integer_frame = np.asarray(frame)
        total_intensity = int(np.sum(integer_frame, dtype=np.int64)) if integer_frame.dtype.kind in "biu" else float(np.sum(integer_frame, dtype=np.float64))
        max_intensity = float(np.max(integer_frame))
        zero_count = int(np.count_nonzero(integer_frame == 0))
        saturation, _sat_path = number_value((
            "/entry/instrument/detector/saturation_value",
            "/entry/instrument/detector/detectorSpecific/saturation_value",
        ), ("/saturation_value",))
        if saturation is None and integer_frame.dtype.kind in "iu":
            saturation = float(np.iinfo(integer_frame.dtype).max)
        saturated_count = int(np.count_nonzero(integer_frame >= saturation)) if saturation is not None else 0
        image_rows = [
            ("Image number", str(frame_index + 1)),
            ("Total intensity", f"{total_intensity:.6e}"),
            ("Maximum intensity", f"{max_intensity:.6e}"),
            ("# zeros", str(zero_count)),
            ("# saturated pixels", str(saturated_count)),
        ]

        beam_rows: list[tuple[str, str]] = []
        specs = [
            ("Start angle (deg)", (
                "/entry/sample/goniometer/omega_start",
                "/entry/sample/goniometer/phi_start",
                "/entry/sample/transformations/omega",
            ), ("/omega_start", "/phi_start"), "deg", True),
            ("Oscillation range (deg)", (
                "/entry/sample/goniometer/omega_increment",
                "/entry/sample/goniometer/omega_range_average",
                "/entry/sample/goniometer/phi_increment",
            ), ("/omega_increment", "/omega_range_average", "/phi_increment"), "deg", False),
            ("Beam centre X (pixel)", (
                "/entry/instrument/detector/beam_center_x",
                "/entry/instrument/detector/beam_centre_x",
            ), ("/beam_center_x", "/beam_centre_x"), "", False),
            ("Beam centre Y (pixel)", (
                "/entry/instrument/detector/beam_center_y",
                "/entry/instrument/detector/beam_centre_y",
            ), ("/beam_center_y", "/beam_centre_y"), "", False),
            ("Exposure time (s)", (
                "/entry/instrument/detector/count_time",
                "/entry/instrument/detector/exposure_time",
            ), ("/count_time", "/exposure_time"), "s", False),
            ("Exposure period (s)", (
                "/entry/instrument/detector/frame_time",
                "/entry/instrument/detector/exposure_period",
            ), ("/frame_time", "/exposure_period"), "s", False),
            ("Wavelength (Å)", (
                "/entry/instrument/beam/incident_wavelength",
                "/entry/instrument/beam/wavelength",
            ), ("/incident_wavelength", "/beam/wavelength"), "angstrom", False),
            ("Beam energy (eV)", (
                "/entry/instrument/beam/incident_energy",
                "/entry/instrument/beam/energy",
            ), ("/incident_energy", "/beam/energy"), "ev", False),
            ("Distance (m)", (
                "/entry/instrument/detector/detector_distance",
                "/entry/instrument/detector/distance",
            ), ("/detector_distance",), "m", False),
            ("Filter transmission", (
                "/entry/instrument/attenuator/transmission",
                "/entry/instrument/beam/filter_transmission",
            ), ("/filter_transmission", "/attenuator/transmission"), "", False),
        ]
        beam_values: dict[str, float] = {}
        for label, candidates, suffixes, target, per_frame in specs:
            value, path = number_value(candidates, suffixes, per_frame=per_frame)
            if value is None:
                continue
            converted = convert(value, units_for(path), target) if target else value
            if converted is None:
                continue
            beam_values[label] = converted
        bx = beam_values.pop("Beam centre X (pixel)", None)
        by = beam_values.pop("Beam centre Y (pixel)", None)
        for label, _candidates, _suffixes, _target, _per_frame in specs:
            if label.startswith("Beam centre"):
                continue
            if label in beam_values:
                beam_rows.append((label, f"{beam_values[label]:.6g}"))
            if label == "Oscillation range (deg)" and bx is not None and by is not None:
                beam_rows.append(("Beam centre (pixel)", f"({bx:.6g}, {by:.6g})"))
        if bx is not None and by is not None and not any(row[0] == "Beam centre (pixel)" for row in beam_rows):
            beam_rows.insert(2, ("Beam centre (pixel)", f"({bx:.6g}, {by:.6g})"))

        result = {"Detector": detector_rows, "Image": image_rows}
        if beam_rows:
            result["Beamline"] = beam_rows
        return result


    def describe_frame(self, frame_index: int) -> dict[str, Any]:
        """Return only metadata preserved from the original HDF5 sources.

        HDXF container identity, codec details, conversion software, execution
        device and timing information are intentionally excluded from this
        user-facing metadata view.
        """
        block = dict(self._find_block(frame_index))
        object_id = str(block.get("source_object") or "")
        objects = self.manifest.get("objects", {})
        object_doc = objects.get(object_id, {}) if isinstance(objects, dict) else {}

        source_id = str(object_doc.get("source_file") or "") if isinstance(object_doc, dict) else ""
        source_doc: dict[str, Any] = {}
        source_root = self.manifest.get("source", {})
        if isinstance(source_root, dict):
            for item in source_root.get("files", []):
                if isinstance(item, dict) and str(item.get("id")) == source_id:
                    source_doc = dict(item)
                    break

        dataset_path = str(object_doc.get("source_path") or "") if isinstance(object_doc, dict) else ""
        storage = object_doc.get("hdf5_storage") if isinstance(object_doc, dict) else None
        attributes = object_doc.get("attributes") if isinstance(object_doc, dict) else None

        dataset_metadata: dict[str, Any] = {"path": dataset_path}
        if isinstance(storage, dict):
            dataset_metadata.update(storage)
        if attributes not in (None, {}, []):
            dataset_metadata["attributes"] = attributes

        # Preserve parent-group attributes exactly as stored from HDF5.  Only
        # groups on the current Dataset path and in the same physical source
        # file are relevant to the selected frame.
        group_metadata: dict[str, Any] = {}
        if isinstance(objects, dict) and dataset_path:
            candidates: list[tuple[str, dict[str, Any]]] = []
            for candidate in objects.values():
                if not isinstance(candidate, dict) or candidate.get("type") != "group":
                    continue
                if str(candidate.get("source_file") or "") != source_id:
                    continue
                group_path = str(candidate.get("source_path") or "")
                if not group_path:
                    continue
                prefix = group_path.rstrip("/") + "/" if group_path != "/" else "/"
                if dataset_path == group_path or dataset_path.startswith(prefix):
                    candidates.append((group_path, candidate))
            for group_path, candidate in sorted(candidates, key=lambda item: (item[0].count("/"), item[0])):
                group_attrs = candidate.get("attributes")
                if group_attrs not in (None, {}, []):
                    group_metadata[group_path] = {"attributes": group_attrs}

        # Link fields shown here are limited to properties that existed in the
        # original HDF5 graph.  HDXF resolution/cache fields are not exposed.
        original_links: list[dict[str, Any]] = []
        for link in self.manifest.get("links", []):
            if not isinstance(link, dict):
                continue
            link_path = str(link.get("path") or "")
            if link_path != dataset_path:
                continue
            filtered = {
                key: link[key]
                for key in ("path", "type", "target_path", "target_file")
                if key in link
            }
            if filtered:
                original_links.append(filtered)

        document: dict[str, Any] = {
            "HDF5 Source": {"filename": source_doc.get("filename")},
            "HDF5 Dataset": dataset_metadata,
        }
        if group_metadata:
            document["HDF5 Parent Groups"] = group_metadata
        if original_links:
            document["HDF5 Links"] = original_links
        return document



class HDF5Archive:
    """Fault-tolerant direct reader for detector frames stored in HDF5.

    The viewer intentionally does not require one exact vendor schema.  It
    first tries the common /entry/data layout, follows ExternalLinks, then
    falls back to other readable numeric image datasets.  Missing metadata is
    never treated as a reason to reject an otherwise readable image file.
    """

    file_format = "HDF5"
    profile = "hdf5-direct"
    _METADATA_CACHE_LIMIT = 32 * 1024 * 1024
    _OTHER_INFO_LIMIT = 80

    def __init__(self, path: Path, *, block_cache_size: int = 8) -> None:
        del block_cache_size
        if h5py is None:
            raise HDXFViewerError(
                "HDF5 support requires h5py. Install it with: python -m pip install h5py"
            )

        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise HDXFViewerError(f"File does not exist: {self.path}")
        try:
            if not h5py.is_hdf5(str(self.path)):
                raise HDXFViewerError(f"Not an HDF5 file: {self.path.name}")
        except HDXFViewerError:
            raise
        except Exception as exc:
            raise HDXFViewerError(f"Cannot identify HDF5 file: {exc}") from exc

        self._h5 = None
        self._extra_h5: list[Any] = []
        self._dataset_value_cache: dict[str, Any] = {}
        self._warnings: list[str] = []
        self._catalog_dataset_paths: set[str] = set()
        self._catalog_group_paths: set[str] = set()
        self._frame_sources: list[dict[str, Any]] = []
        self._starts: list[int] = []
        self._source_by_path: dict[str, dict[str, Any]] = {}

        # Discovery diagnostics.  If every Dataset/ExternalLink declared under
        # /entry/data resolves and yields a real multi-frame sequence, that
        # sequence is authoritative and we deliberately avoid a generic HDF5
        # tree scan.  This is both safer and much faster for detector masters.
        self._entry_data_declared_sources = 0
        self._entry_data_failed_sources = 0

        try:
            self._h5 = h5py.File(self.path, "r")
        except Exception as exc:
            raise HDXFViewerError(f"Cannot open HDF5 file: {type(exc).__name__}: {exc}") from exc

        try:
            self._catalog_current_file()
            self._discover_frame_sources()
            if not self._frame_sources:
                details = "; ".join(self._warnings[-6:]) if self._warnings else "no readable 2D/3D numeric image dataset was found"
                raise HDXFViewerError(
                    "No readable detector image dataset was found. "
                    f"Fallback discovery was attempted. Details: {details}"
                )

            self.frame_count = int(sum(int(item["frame_count"]) for item in self._frame_sources))
            first = self._frame_sources[0]
            self.frame_shape = tuple(int(x) for x in first["frame_shape"])
            self.dtype = np.dtype(first["dtype"])
            self._starts = [int(item["global_start"]) for item in self._frame_sources]
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self._dataset_value_cache.clear()
        for handle in list(getattr(self, "_extra_h5", [])):
            try:
                handle.close()
            except Exception:
                pass
        self._extra_h5 = []
        handle = getattr(self, "_h5", None)
        self._h5 = None
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    def __enter__(self) -> "HDF5Archive":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _is_numeric_image_dtype(dtype: Any) -> bool:
        try:
            return np.dtype(dtype).kind in "biuf"
        except Exception:
            return False

    @staticmethod
    def _safe_storage_size(dataset: Any) -> int | None:
        try:
            return int(dataset.id.get_storage_size())
        except Exception:
            return None

    @staticmethod
    def _safe_metadata_value(value: Any, *, max_items: int = 16) -> Any:
        """Convert arbitrary HDF5 metadata to a small, display-safe value."""
        try:
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, bytes):
                return value.rstrip(b"\x00").decode("utf-8", errors="replace")
            if isinstance(value, str) or value is None or isinstance(value, (bool, int, float, complex)):
                return value
            if h5py is not None and isinstance(value, (h5py.Reference, h5py.RegionReference)):
                return "<HDF5 reference>"
            if isinstance(value, np.ndarray):
                if value.size == 0:
                    return []
                if value.size <= max_items:
                    return [HDF5Archive._safe_metadata_value(item) for item in value.reshape(-1)]
                preview = [HDF5Archive._safe_metadata_value(item) for item in value.reshape(-1)[:max_items]]
                return {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "preview": preview,
                    "truncated": int(value.size - max_items),
                }
            if isinstance(value, (list, tuple)):
                data = list(value)
                if len(data) <= max_items:
                    return [HDF5Archive._safe_metadata_value(item) for item in data]
                return [HDF5Archive._safe_metadata_value(item) for item in data[:max_items]] + [f"… {len(data)-max_items} more"]
            return repr(value)
        except Exception:
            try:
                return repr(value)
            except Exception:
                return "<unreadable metadata>"

    def _safe_attrs(self, obj: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        try:
            keys = list(obj.attrs.keys())
        except Exception as exc:
            return {"<attributes>": f"unavailable: {type(exc).__name__}: {exc}"}
        for key in keys:
            try:
                result[str(key)] = self._safe_metadata_value(obj.attrs[key])
            except Exception as exc:
                result[str(key)] = f"<unreadable: {type(exc).__name__}: {exc}>"
        return result

    def _catalog_current_file(self) -> None:
        if self._h5 is None:
            return
        self._catalog_group_paths.add("/")
        try:
            def visitor(name: str, obj: Any) -> None:
                path = "/" + name.lstrip("/") if name else "/"
                try:
                    if isinstance(obj, h5py.Dataset):
                        self._catalog_dataset_paths.add(path)
                    elif isinstance(obj, h5py.Group):
                        self._catalog_group_paths.add(path)
                except Exception:
                    pass
            self._h5.visititems(visitor)
        except Exception as exc:
            self._warnings.append(f"HDF5 tree catalog incomplete: {type(exc).__name__}: {exc}")

    def _open_external_link_fallback(self, link: Any) -> Any | None:
        """Open a broken relative ExternalLink using conservative path fallbacks."""
        if h5py is None or not isinstance(link, h5py.ExternalLink):
            return None
        raw_name = os.fsdecode(link.filename)
        candidates: list[Path] = []
        linked = Path(raw_name)
        if linked.is_absolute():
            candidates.append(linked)
        else:
            candidates.append(self.path.parent / linked)
            candidates.append(self.path.parent / linked.name)
        seen: set[str] = set()
        for candidate in candidates:
            key = os.path.normcase(str(candidate.resolve())) if candidate.exists() else os.path.normcase(str(candidate))
            if key in seen:
                continue
            seen.add(key)
            if not candidate.is_file():
                continue
            try:
                handle = h5py.File(candidate, "r")
                target = handle[str(link.path)]
                self._extra_h5.append(handle)
                return target
            except Exception as exc:
                try:
                    handle.close()
                except Exception:
                    pass
                self._warnings.append(
                    f"ExternalLink fallback failed for {candidate.name}:{link.path}: {type(exc).__name__}: {exc}"
                )
        return None

    @staticmethod
    def _frame_layout(dataset: Any) -> tuple[int, tuple[int, int]] | None:
        try:
            shape = tuple(int(x) for x in dataset.shape)
            if len(shape) < 2 or any(dim <= 0 for dim in shape[-2:]):
                return None
            if not HDF5Archive._is_numeric_image_dtype(dataset.dtype):
                return None
            if len(shape) == 2:
                return 1, (shape[0], shape[1])
            count = int(np.prod(shape[:-2], dtype=np.int64))
            if count <= 0:
                return None
            return count, (shape[-2], shape[-1])
        except Exception:
            return None

    @staticmethod
    def _read_dataset_frame(dataset: Any, local_index: int) -> np.ndarray:
        shape = tuple(int(x) for x in dataset.shape)
        if len(shape) == 2:
            if local_index != 0:
                raise IndexError(local_index)
            data = dataset[...]
        else:
            leading = shape[:-2]
            coords = np.unravel_index(int(local_index), leading)
            selection = tuple(int(x) for x in coords) + (slice(None), slice(None))
            data = dataset[selection]
        array = np.asarray(data)
        if array.ndim != 2:
            array = np.asarray(array).reshape(shape[-2], shape[-1])
        if array.dtype.kind not in "biuf":
            raise HDXFViewerError(f"Unsupported image dtype: {array.dtype}")
        return np.ascontiguousarray(array)

    def _candidate(
        self,
        source_path: str,
        dataset: Any,
        *,
        priority: int,
        link_doc: dict[str, Any] | None = None,
        role: str = "generic",
    ) -> dict[str, Any] | None:
        layout = self._frame_layout(dataset)
        if layout is None:
            return None
        count, frame_shape = layout
        try:
            # Validate one actual frame.  Metadata problems are ignored, but a
            # frame source must at least be readable before it can become the
            # primary image source.
            self._read_dataset_frame(dataset, 0)
        except Exception as exc:
            self._warnings.append(
                f"Skipped unreadable frame candidate {source_path}: {type(exc).__name__}: {exc}"
            )
            return None
        try:
            physical_file = str(Path(dataset.file.filename).resolve())
        except Exception:
            physical_file = str(getattr(dataset.file, "filename", self.path))
        return {
            "source_path": str(source_path),
            "dataset": dataset,
            "dataset_path": str(getattr(dataset, "name", source_path)),
            "physical_file": physical_file,
            "frame_count": int(count),
            "frame_shape": tuple(frame_shape),
            "dtype": np.dtype(dataset.dtype),
            "priority": int(priority),
            "link": link_doc,
            "role": str(role),
        }

    def _entry_data_candidates(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        self._entry_data_declared_sources = 0
        self._entry_data_failed_sources = 0

        if self._h5 is None:
            return result

        try:
            obj = self._h5.get("/entry/data")
        except Exception as exc:
            self._warnings.append(
                f"Cannot access /entry/data: {type(exc).__name__}: {exc}"
            )
            obj = None

        if isinstance(obj, h5py.Dataset):
            self._entry_data_declared_sources = 1
            candidate = self._candidate(
                "/entry/data",
                obj,
                priority=120,
                role="entry-data",
            )
            if candidate:
                result.append(candidate)
            else:
                self._entry_data_failed_sources = 1
            return result

        if not isinstance(obj, h5py.Group):
            return result

        try:
            names = sorted(obj.keys())
        except Exception as exc:
            self._warnings.append(
                f"Cannot list /entry/data: {type(exc).__name__}: {exc}"
            )
            return result

        for name in names:
            source_path = f"/entry/data/{name}"
            link_doc = None
            try:
                link = obj.get(name, getlink=True)
            except Exception:
                link = None

            if isinstance(link, h5py.ExternalLink):
                # An ExternalLink under /entry/data is a declared detector
                # source even when its target later proves unreadable.
                self._entry_data_declared_sources += 1
                link_doc = {
                    "type": "ExternalLink",
                    "target_file": os.fsdecode(link.filename),
                    "target_path": str(link.path),
                }

            try:
                dataset = obj[name]
            except Exception as exc:
                dataset = self._open_external_link_fallback(link)
                if dataset is None:
                    if isinstance(link, h5py.ExternalLink):
                        self._entry_data_failed_sources += 1
                    self._warnings.append(
                        f"Cannot open {source_path}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue

            if not isinstance(dataset, h5py.Dataset):
                # Non-Dataset children of /entry/data are not detector-frame
                # sources and therefore do not affect completeness.
                continue

            if not isinstance(link, h5py.ExternalLink):
                self._entry_data_declared_sources += 1

            candidate = self._candidate(
                source_path,
                dataset,
                priority=130 if isinstance(link, h5py.ExternalLink) else 125,
                link_doc=link_doc,
                role="entry-data",
            )
            if candidate:
                result.append(candidate)
            else:
                self._entry_data_failed_sources += 1

        return result


    def _sibling_data_candidates(self) -> list[dict[str, Any]]:
        """Recover detector data files next to a master HDF5.

        For a ``*_master.h5`` file, only files with the same exact experiment
        prefix are considered, e.g.::

            scan_master.h5
            scan_data_000001.h5
            scan_data_000002.h5

        This recovery is intentionally strict for master files so an unrelated
        run in the same directory is never merged into the current sequence.
        For a non-master file, sibling discovery is only used as a last resort
        by ``_discover_frame_sources``.
        """
        result: list[dict[str, Any]] = []
        stem = self.path.stem

        if stem.endswith("_master"):
            prefix = stem[:-7]
            patterns = (
                f"{prefix}_data_*.h5",
                f"{prefix}_data_*.hdf5",
                f"{prefix}_data_*.nxs",
            )
        else:
            # Keep the broad legacy fallback only for non-master files.  It is
            # never preferred over a directly readable Dataset.
            patterns = ("*_data_*.h5", "*_data_*.hdf5", "*_data_*.nxs")

        siblings: list[Path] = []
        for pattern in patterns:
            siblings.extend(sorted(self.path.parent.glob(pattern)))

        seen_files: set[str] = set()

        for sibling in siblings:
            try:
                resolved = sibling.resolve()
            except Exception:
                resolved = sibling

            key = os.path.normcase(str(resolved))
            if key in seen_files:
                continue
            if resolved == self.path:
                continue
            seen_files.add(key)

            try:
                handle = h5py.File(sibling, "r")
            except Exception as exc:
                self._warnings.append(
                    f"Cannot open sibling detector file {sibling.name}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            selected: tuple[str, Any] | None = None

            # Strong detector-data paths first.
            for candidate_path in (
                "/entry/data/data",
                "/entry/data",
                "/entry/instrument/detector/data",
                "/data",
            ):
                try:
                    obj = handle.get(candidate_path)
                except Exception:
                    obj = None
                if isinstance(obj, h5py.Dataset) and self._frame_layout(obj) is not None:
                    selected = (candidate_path, obj)
                    break

            # Last resort inside this one data file: scan only metadata while
            # HDF5 is inside visititems().  Calling dataset[...] from an HDF5
            # traversal callback can re-enter the HDF5/Python boundary and has
            # triggered a fatal GIL/thread-state crash on Windows+h5py.
            if selected is None:
                found: list[tuple[int, int, int, str]] = []
                try:
                    def visitor(name: str, obj: Any) -> None:
                        if not isinstance(obj, h5py.Dataset):
                            return
                        layout = self._frame_layout(obj)
                        if layout is None:
                            return
                        count, frame_shape = layout
                        path_in_file = "/" + name.lstrip("/")
                        lower = path_in_file.lower()
                        score = 0
                        if lower.endswith("/data") or "/data/" in lower:
                            score += 40
                        if (
                            "detector" in lower
                            or "image" in lower
                            or "frame" in lower
                        ):
                            score += 20
                        area = int(frame_shape[0]) * int(frame_shape[1])
                        found.append(
                            (int(count), score, area, path_in_file)
                        )

                    handle.visititems(visitor)
                except Exception as exc:
                    self._warnings.append(
                        f"Could not scan sibling detector file {sibling.name}: "
                        f"{type(exc).__name__}: {exc}"
                    )

                if found:
                    found.sort(reverse=True)
                    _count, _score, _area, path_in_file = found[0]
                    try:
                        dataset = handle[path_in_file]
                    except Exception:
                        dataset = None
                    if isinstance(dataset, h5py.Dataset):
                        selected = (path_in_file, dataset)

            if selected is None:
                try:
                    handle.close()
                except Exception:
                    pass
                continue

            path_in_file, dataset = selected
            source_path = f"<sibling:{sibling.name}>{path_in_file}"
            candidate = self._candidate(
                source_path,
                dataset,
                priority=90,
                role="sibling-data",
            )
            if candidate:
                self._extra_h5.append(handle)
                result.append(candidate)
            else:
                try:
                    handle.close()
                except Exception:
                    pass

        return result


    def _fallback_candidates(self) -> list[dict[str, Any]]:
        """Find fallback image Datasets without Dataset I/O in visititems().

        h5py/HDF5 traversal callbacks must remain metadata-only.  On Windows,
        reading ``dataset[...]`` from inside a ``visititems`` callback can
        re-enter the HDF5/Python boundary while HDF5 is still executing the
        callback and can terminate the interpreter with
        ``PyEval_RestoreThread ... GIL ... thread state is NULL``.

        We therefore collect only path/shape/name scores during traversal and
        validate a small ranked set *after* visititems has completely returned.
        """
        result: list[dict[str, Any]] = []
        if self._h5 is None:
            return result

        preferred_paths = (
            "/entry/instrument/detector/data",
            "/entry/data/data",
            "/data",
        )
        seen_ids: set[tuple[str, str]] = set()

        # Known strong paths can be validated directly because this code is not
        # running inside an HDF5 traversal callback.
        for path in preferred_paths:
            try:
                obj = self._h5.get(path)
            except Exception:
                obj = None
            if isinstance(obj, h5py.Dataset):
                candidate = self._candidate(
                    path,
                    obj,
                    priority=105,
                    role="preferred-data",
                )
                if candidate:
                    result.append(candidate)
                    seen_ids.add(
                        (
                            candidate["physical_file"],
                            candidate["dataset_path"],
                        )
                    )

        metadata_candidates: list[
            tuple[int, int, int, str]
        ] = []

        try:
            def visitor(name: str, obj: Any) -> None:
                if not isinstance(obj, h5py.Dataset):
                    return

                layout = self._frame_layout(obj)
                if layout is None:
                    return

                count, frame_shape = layout
                path = "/" + name.lstrip("/")
                lower = path.lower()

                # Calibration, MX tables and azimuthal integration are valid
                # scientific datasets, but they are poor generic candidates
                # for a 2-D detector image source.
                if (
                    "/calibration/" in lower
                    or "/entry/mx/" in lower
                    or "/entry/azint/" in lower
                ):
                    return

                height, width = (
                    int(frame_shape[0]),
                    int(frame_shape[1]),
                )
                area = height * width

                # Reject obvious curve/table shapes before any pixel read.
                if min(height, width) < 8 or area < 4096:
                    return

                priority = 40
                if lower.endswith("/data") or "/data/" in lower:
                    priority += 25
                if (
                    "detector" in lower
                    or "image" in lower
                    or "frame" in lower
                ):
                    priority += 15

                metadata_candidates.append(
                    (
                        priority,
                        int(count),
                        area,
                        path,
                    )
                )

            self._h5.visititems(visitor)
        except Exception as exc:
            self._warnings.append(
                f"Fallback HDF5 scan incomplete: "
                f"{type(exc).__name__}: {exc}"
            )

        # Rank using metadata only, then validate outside the callback.  A small
        # limit keeps pathological HDF5 files with thousands of numeric tables
        # from causing hundreds/thousands of full image reads.
        metadata_candidates.sort(reverse=True)
        for priority, _count, _area, path in metadata_candidates[:32]:
            try:
                obj = self._h5.get(path)
            except Exception:
                obj = None
            if not isinstance(obj, h5py.Dataset):
                continue

            candidate = self._candidate(
                path,
                obj,
                priority=priority,
                role="generic",
            )
            if not candidate:
                continue

            identity = (
                candidate["physical_file"],
                candidate["dataset_path"],
            )
            if identity in seen_ids:
                continue
            seen_ids.add(identity)
            result.append(candidate)

        return result


    @staticmethod
    def _natural_source_key(item: dict[str, Any]) -> tuple[Any, ...]:
        """Natural sort key so data_2 comes before data_10."""
        text = str(item.get("source_path") or "")
        parts = re.split(r"(\d+)", text)
        return tuple(int(part) if part.isdigit() else part.lower() for part in parts)

    @staticmethod
    def _deduplicate_candidates(
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Collapse the same physical Dataset discovered through multiple routes."""
        best: dict[tuple[str, str], dict[str, Any]] = {}
        for item in candidates:
            identity = (
                os.path.normcase(str(item.get("physical_file") or "")),
                str(item.get("dataset_path") or ""),
            )
            current = best.get(identity)
            if current is None:
                best[identity] = item
                continue

            current_priority = int(current.get("priority", 0))
            new_priority = int(item.get("priority", 0))

            # Prefer the direct master/ExternalLink representation so metadata
            # and original source paths remain nicer in the inspector.
            if new_priority > current_priority:
                best[identity] = item

        return list(best.values())

    @staticmethod
    def _select_compatible_sources(
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Select the dominant real detector sequence, not merely one 2-D Dataset.

        The previous HDF5 fallback selected only the highest-priority schema
        tier.  That is dangerous for a master file: one readable 2-D Dataset
        (for example a mask or preview) can outrank 60 sibling detector data
        files and make the Viewer report exactly one image.

        Selection is now sequence-aware:
          * identical physical Datasets are deduplicated;
          * explicit /entry/data and sibling detector-data candidates are
            preferred over generic metadata arrays;
          * candidates are grouped by 2-D frame shape;
          * the group with the largest real frame sequence wins;
          * isolated one-frame outliers are removed when a real multi-frame
            sequence exists.
        """
        if not candidates:
            return []

        candidates = HDF5Archive._deduplicate_candidates(candidates)

        strong_roles = {"entry-data", "sibling-data", "preferred-data"}
        strong = [
            item
            for item in candidates
            if str(item.get("role")) in strong_roles
        ]
        pool = strong if strong else candidates

        groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for item in pool:
            groups.setdefault(tuple(item["frame_shape"]), []).append(item)

        if not groups:
            return []

        def group_score(pair: tuple[tuple[int, int], list[dict[str, Any]]]) -> tuple[int, int, int, int]:
            shape, items = pair
            total_frames = sum(int(item["frame_count"]) for item in items)
            multi_frame_sources = sum(
                1 for item in items if int(item["frame_count"]) > 1
            )
            source_files = len({
                os.path.normcase(str(item.get("physical_file") or ""))
                for item in items
            })
            max_priority = max(int(item.get("priority", 0)) for item in items)
            # Frame count is the strongest signal.  File/source count comes
            # next because detector masters commonly span many data files.
            return (
                total_frames,
                source_files,
                multi_frame_sources,
                max_priority,
            )

        _shape, selected = max(groups.items(), key=group_score)
        selected = list(selected)

        total_frames = sum(int(item["frame_count"]) for item in selected)
        multi_frame = [
            item for item in selected
            if int(item["frame_count"]) > 1
        ]

        # If the winning group clearly contains a true sequence, suppress
        # isolated one-frame preview/mask candidates that happen to share the
        # detector's 2-D shape.
        if multi_frame and total_frames > 1:
            sequence_roles = {"entry-data", "sibling-data", "preferred-data"}
            cleaned = [
                item for item in selected
                if int(item["frame_count"]) > 1
                or str(item.get("role")) == "sibling-data"
                or (
                    str(item.get("role")) == "entry-data"
                    and isinstance(item.get("link"), dict)
                    and str(item["link"].get("type")) == "ExternalLink"
                )
            ]
            if cleaned:
                selected = cleaned

        selected.sort(key=HDF5Archive._natural_source_key)
        return selected


    def _discover_frame_sources(self) -> None:
        """Discover the detector sequence without unsafe generic rescans.

        Order:
          1. /entry/data (authoritative when complete and multi-frame)
          2. exact-prefix sibling data files for *_master
          3. generic HDF5 fallback only when the first two routes are
             insufficient

        This avoids scanning/reading thousands of MX/azint/calibration datasets
        when the master already provides the real detector sequence.
        """
        entry_candidates = self._entry_data_candidates()
        entry_selected = self._select_compatible_sources(
            entry_candidates
        )
        entry_total = sum(
            int(item["frame_count"])
            for item in entry_selected
        )

        entry_complete = (
            self._entry_data_declared_sources > 0
            and self._entry_data_failed_sources == 0
            and len(entry_candidates)
            == self._entry_data_declared_sources
        )

        # A complete /entry/data sequence is the strongest possible evidence.
        # For the production detector master this means 60 ExternalLinks ×
        # 1000 frames and lets us avoid _fallback_candidates entirely.
        if entry_complete and entry_total > 1:
            selected = entry_selected
            self._warnings.append(
                "HDF5 frame discovery used authoritative /entry/data "
                f"sequence: {entry_total} images from "
                f"{len(selected)} Dataset source(s)"
            )
        else:
            candidates: list[dict[str, Any]] = []
            candidates.extend(entry_candidates)

            # For a master, exact-prefix sibling files are much safer than a
            # generic tree scan.  They also recover missing/broken ExternalLinks.
            if self.path.stem.endswith("_master"):
                sibling_candidates = (
                    self._sibling_data_candidates()
                )
                candidates.extend(sibling_candidates)

            selected = self._select_compatible_sources(
                candidates
            )
            selected_total = sum(
                int(item["frame_count"])
                for item in selected
            )

            # Only now use generic discovery.  The implementation is two-pass
            # and never performs pixel reads inside visititems callbacks.
            if selected_total <= 1:
                fallback_candidates = (
                    self._fallback_candidates()
                )
                candidates.extend(fallback_candidates)

                if (
                    not self.path.stem.endswith("_master")
                    and not candidates
                ):
                    candidates.extend(
                        self._sibling_data_candidates()
                    )

                selected = self._select_compatible_sources(
                    candidates
                )

        global_start = 0
        for item in selected:
            item = dict(item)
            item["global_start"] = global_start
            global_start += int(item["frame_count"])
            self._frame_sources.append(item)
            self._source_by_path[
                str(item["source_path"])
            ] = item

        if selected:
            role_counts = Counter(
                str(item.get("role") or "unknown")
                for item in selected
            )
            physical_files = {
                os.path.normcase(
                    str(item.get("physical_file") or "")
                )
                for item in selected
            }
            self._warnings.append(
                "HDF5 frame discovery: "
                f"{global_start} images from {len(selected)} "
                f"Dataset source(s), "
                f"{len(physical_files)} physical data file(s), "
                f"roles={dict(role_counts)}"
            )

        if (
            self.path.stem.endswith("_master")
            and global_start <= 1
        ):
            self._warnings.append(
                "Master HDF5 resolved to only one image after "
                "/entry/data and exact-prefix sibling-data recovery. "
                "Check the corresponding *_data_*.h5 files and "
                "ExternalLinks."
            )


    def _find_source(self, frame_index: int) -> dict[str, Any]:
        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError(frame_index)
        pos = bisect.bisect_right(self._starts, int(frame_index)) - 1
        if pos < 0:
            raise HDXFViewerError(f"No HDF5 Dataset covers image {frame_index + 1}")
        source = self._frame_sources[pos]
        start = int(source["global_start"])
        count = int(source["frame_count"])
        if not (start <= frame_index < start + count):
            raise HDXFViewerError(f"No HDF5 Dataset covers image {frame_index + 1}")
        return source

    def get_frame(self, frame_index: int) -> np.ndarray:
        source = self._find_source(int(frame_index))
        local_index = int(frame_index) - int(source["global_start"])
        try:
            frame = self._read_dataset_frame(source["dataset"], local_index)
        except Exception as exc:
            raise HDXFViewerError(
                f"Cannot read HDF5 image {frame_index + 1} from {source['source_path']}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if tuple(frame.shape) != tuple(self.frame_shape):
            raise HDXFViewerError(
                f"HDF5 frame shape changed unexpectedly: {frame.shape} != {self.frame_shape}"
            )
        return frame

    def _open_object(self, path: str) -> Any | None:
        if self._h5 is None:
            return None
        try:
            return self._h5[path]
        except Exception:
            source = self._source_by_path.get(path)
            return source.get("dataset") if source else None

    def read_hdf5_dataset(self, path: str) -> Any | None:
        if path in self._dataset_value_cache:
            return self._dataset_value_cache[path]
        obj = self._open_object(path)
        if not isinstance(obj, h5py.Dataset):
            return None
        try:
            if obj.shape is not None and len(obj.shape) > 0:
                estimate = int(np.prod(obj.shape, dtype=np.int64)) * max(1, int(np.dtype(obj.dtype).itemsize))
                if estimate > self._METADATA_CACHE_LIMIT:
                    return None
            value = obj[()]
        except Exception:
            return None
        try:
            nbytes = int(np.asarray(value).nbytes)
        except Exception:
            nbytes = 0
        if nbytes <= self._METADATA_CACHE_LIMIT:
            self._dataset_value_cache[path] = value
        return value

    def hdf5_dataset_attributes(self, path: str) -> dict[str, Any]:
        obj = self._open_object(path)
        if obj is None:
            return {}
        return self._safe_attrs(obj)

    def find_hdf5_value(
        self,
        candidates: tuple[str, ...],
        *,
        suffixes: tuple[str, ...] = (),
    ) -> tuple[Any | None, str | None]:
        for path in candidates:
            value = self.read_hdf5_dataset(path)
            if value is not None:
                return value, path
        if suffixes:
            lower_suffixes = tuple(str(item).lower() for item in suffixes)
            for path in sorted(self._catalog_dataset_paths):
                lower = path.lower()
                if any(lower.endswith(suffix) for suffix in lower_suffixes):
                    value = self.read_hdf5_dataset(path)
                    if value is not None:
                        return value, path
        return None, None

    def _fallback_detector_name(self) -> tuple[str, str]:
        for group_path in ("/entry/instrument/detector", "/entry/instrument", "/entry", "/"):
            obj = self._open_object(group_path)
            if obj is None:
                continue
            attrs = self._safe_attrs(obj)
            for key in ("description", "detector_name", "name", "type", "NX_class"):
                if key in attrs:
                    text = _scalar_text(attrs[key]).strip()
                    if text:
                        return text, f"attribute {group_path}@{key}"
        return self.path.stem, "filename fallback"

    def _extra_information_rows(self, frame_index: int, *, max_rows: int | None = None) -> list[tuple[str, str]]:
        max_rows = self._OTHER_INFO_LIMIT if max_rows is None else max(1, int(max_rows))
        rows: list[tuple[str, str]] = []
        source = self._find_source(frame_index)
        local_index = frame_index - int(source["global_start"])
        rows.extend([
            ("File format", "HDF5"),
            ("Opened file", self.path.name),
            ("Frame source", str(source.get("source_path") or "-")),
            ("Physical data file", Path(str(source.get("physical_file") or self.path)).name),
            ("Local frame index", str(local_index)),
        ])
        dataset = source.get("dataset")
        if isinstance(dataset, h5py.Dataset):
            try:
                rows.append(("HDF5 chunks", str(dataset.chunks)))
            except Exception:
                pass
            try:
                rows.append(("HDF5 compression", str(dataset.compression or "none")))
            except Exception:
                pass
            storage = self._safe_storage_size(dataset)
            if storage is not None:
                rows.append(("Dataset storage bytes", str(storage)))

        if self._warnings:
            rows.append(("Fallback warnings", " | ".join(self._warnings[:4])))

        known_suffixes = (
            "/description", "/detector_name", "/x_pixel_size", "/y_pixel_size",
            "/sensor_thickness", "/pixel_mask", "/saturation_value", "/omega_start",
            "/omega_increment", "/omega_range_average", "/phi_start", "/phi_increment",
            "/beam_center_x", "/beam_center_y", "/beam_centre_x", "/beam_centre_y",
            "/count_time", "/exposure_time", "/frame_time", "/exposure_period",
            "/incident_wavelength", "/wavelength", "/incident_energy", "/energy",
            "/detector_distance", "/filter_transmission", "/transmission",
        )
        frame_paths = {str(item.get("source_path")) for item in self._frame_sources}
        for path in sorted(self._catalog_dataset_paths):
            if len(rows) >= max_rows:
                break
            lower = path.lower()
            if path in frame_paths or any(lower.endswith(suffix) for suffix in known_suffixes):
                continue
            if "/calibration/" in lower:
                continue
            obj = self._open_object(path)
            if not isinstance(obj, h5py.Dataset):
                continue
            try:
                size = int(obj.size)
            except Exception:
                continue
            if size > 16:
                continue
            value = self.read_hdf5_dataset(path)
            if value is None:
                continue
            text = _scalar_text(value).strip()
            if not text:
                continue
            if len(text) > 120:
                text = text[:117] + "…"
            rows.append((path, text))

        # Group/file attributes are useful extra metadata even when the file
        # has no NeXus-standard Dataset for the same concept.
        for group_path in ("/", "/entry", "/entry/instrument", "/entry/instrument/detector", "/entry/instrument/beam", "/entry/sample"):
            if len(rows) >= max_rows:
                break
            obj = self._open_object(group_path)
            if obj is None:
                continue
            for key, value in self._safe_attrs(obj).items():
                if len(rows) >= max_rows:
                    break
                text = _scalar_text(value).strip()
                if len(text) > 120:
                    text = text[:117] + "…"
                rows.append((f"{group_path}@{key}", text))
        return rows

    def image_info(self, frame_index: int, frame: np.ndarray) -> dict[str, list[tuple[str, str]]]:
        """Albula-style summary with explicit metadata fallbacks and extra fields."""
        try:
            sections = HDXFArchive.image_info(self, frame_index, frame)
        except Exception as exc:
            # Metadata must never make an otherwise readable HDF5 image fail.
            height, width = frame.shape
            sections = {
                "Detector": [
                    ("Detector", self.path.stem),
                    ("Number of pixels in X-direction", str(width)),
                    ("Number of pixels in Y-direction", str(height)),
                ],
                "Image": [
                    ("Image number", str(frame_index + 1)),
                    ("Total intensity", f"{float(np.sum(frame, dtype=np.float64)):.6e}"),
                    ("Maximum intensity", f"{float(np.max(frame)):.6e}"),
                    ("# zeros", str(int(np.count_nonzero(frame == 0)))),
                    ("# saturated pixels", "N/A"),
                ],
                "Other Information": [
                    ("Metadata fallback", f"Standard metadata parser failed safely: {type(exc).__name__}: {exc}"),
                ],
            }

        detector_rows = list(sections.get("Detector", []))
        detector_labels = {label for label, _ in detector_rows}
        fallback_notes: list[str] = []
        if "Detector" not in detector_labels:
            name, reason = self._fallback_detector_name()
            detector_rows.insert(0, ("Detector", name))
            fallback_notes.append(f"Detector name: {reason}")
        for label in (
            "Pixel size X (m)",
            "Pixel size Y (m)",
            "Sensor thickness (m)",
            "Number of defective",
        ):
            if label not in {key for key, _ in detector_rows}:
                detector_rows.append((label, "N/A (metadata missing)"))
        sections["Detector"] = detector_rows

        beam_rows = list(sections.get("Beamline", []))
        beam_map = {label: value for label, value in beam_rows}
        # Safe derived fallback: photon energy and wavelength are physically
        # equivalent.  Only derive one when the other was explicitly present.
        try:
            if "Wavelength (Å)" not in beam_map and "Beam energy (eV)" in beam_map:
                energy = float(beam_map["Beam energy (eV)"])
                if energy > 0:
                    beam_rows.append(("Wavelength (Å)", f"{12398.419843320026 / energy:.6g} (derived from energy)"))
                    fallback_notes.append("Wavelength derived from beam energy")
            elif "Beam energy (eV)" not in beam_map and "Wavelength (Å)" in beam_map:
                wavelength = float(beam_map["Wavelength (Å)"])
                if wavelength > 0:
                    beam_rows.append(("Beam energy (eV)", f"{12398.419843320026 / wavelength:.6g} (derived from wavelength)"))
                    fallback_notes.append("Beam energy derived from wavelength")
        except Exception:
            pass
        beam_labels = {label for label, _ in beam_rows}
        for label in (
            "Start angle (deg)",
            "Oscillation range (deg)",
            "Beam centre (pixel)",
            "Exposure time (s)",
            "Exposure period (s)",
            "Wavelength (Å)",
            "Beam energy (eV)",
            "Distance (m)",
            "Filter transmission",
        ):
            if label not in beam_labels:
                beam_rows.append((label, "N/A (metadata missing)"))
        sections["Beamline"] = beam_rows

        other_rows = list(sections.get("Other Information", []))
        if fallback_notes:
            other_rows.append(("Fallbacks used", "; ".join(fallback_notes)))
        try:
            other_rows.extend(self._extra_information_rows(frame_index))
        except Exception as exc:
            other_rows.append(("Other metadata", f"Partially unavailable: {type(exc).__name__}: {exc}"))
        sections["Other Information"] = other_rows[: self._OTHER_INFO_LIMIT]
        return sections

    def describe_frame(self, frame_index: int) -> dict[str, Any]:
        """Return a best-effort HDF5 metadata document for the metadata tree."""
        source = self._find_source(frame_index)
        dataset = source.get("dataset")
        local_index = frame_index - int(source["global_start"])

        source_doc: dict[str, Any] = {
            "format": "HDF5",
            "opened_filename": self.path.name,
            "opened_path": str(self.path),
            "physical_data_file": str(source.get("physical_file") or self.path),
        }
        if source.get("link"):
            source_doc["link"] = dict(source["link"])

        dataset_doc: dict[str, Any] = {
            "path": str(source.get("source_path") or ""),
            "physical_dataset_path": str(source.get("dataset_path") or ""),
            "current_local_frame": int(local_index),
            "shape": list(getattr(dataset, "shape", ())),
            "dtype": str(getattr(dataset, "dtype", "unknown")),
        }
        if isinstance(dataset, h5py.Dataset):
            for key, getter in (
                ("chunks", lambda: dataset.chunks),
                ("compression", lambda: dataset.compression),
                ("compression_opts", lambda: dataset.compression_opts),
                ("shuffle", lambda: dataset.shuffle),
                ("fletcher32", lambda: dataset.fletcher32),
                ("scaleoffset", lambda: dataset.scaleoffset),
                ("maxshape", lambda: dataset.maxshape),
            ):
                try:
                    value = getter()
                    if value is not None:
                        dataset_doc[key] = self._safe_metadata_value(value)
                except Exception:
                    pass
            storage = self._safe_storage_size(dataset)
            if storage is not None:
                dataset_doc["storage_bytes"] = storage
            attrs = self._safe_attrs(dataset)
            if attrs:
                dataset_doc["attributes"] = attrs

        parent_groups: dict[str, Any] = {}
        source_path = str(source.get("source_path") or "")
        if source_path.startswith("/"):
            parts = [part for part in source_path.split("/") if part]
            for length in range(0, len(parts)):
                group_path = "/" if length == 0 else "/" + "/".join(parts[:length])
                obj = self._open_object(group_path)
                if isinstance(obj, h5py.Group):
                    attrs = self._safe_attrs(obj)
                    if attrs:
                        parent_groups[group_path] = {"attributes": attrs}

        role_counts = Counter(
            str(item.get("role") or "unknown")
            for item in self._frame_sources
        )
        physical_files = {
            os.path.normcase(str(item.get("physical_file") or ""))
            for item in self._frame_sources
        }
        try:
            filter_32008_available = (
                bool(h5py.h5z.filter_avail(32008))
                if h5py is not None
                else None
            )
        except Exception:
            filter_32008_available = None

        other: dict[str, Any] = {
            "reader": "direct HDF5 detector reader",
            "frame_sources": len(self._frame_sources),
            "physical_frame_files": len(physical_files),
            "discovered_frame_count": int(self.frame_count),
            "source_roles": dict(role_counts),
            "hdf5plugin_available": HDF5PLUGIN_AVAILABLE,
            "hdf5_filter_32008_available": filter_32008_available,
        }
        if HDF5PLUGIN_IMPORT_ERROR:
            other["hdf5plugin_import_error"] = HDF5PLUGIN_IMPORT_ERROR
        if self._warnings:
            other["warnings"] = list(self._warnings[:20])
        try:
            extras = self._extra_information_rows(frame_index, max_rows=40)
            if extras:
                other["additional_metadata"] = {label: value for label, value in extras}
        except Exception as exc:
            other["additional_metadata"] = f"unavailable: {type(exc).__name__}: {exc}"

        document: dict[str, Any] = {
            "HDF5 Source": source_doc,
            "HDF5 Dataset": dataset_doc,
        }
        if parent_groups:
            document["HDF5 Parent Groups"] = parent_groups
        document["Other Information"] = other
        return document

def _histogram_domain_for_values(
    values: np.ndarray,
    *,
    display_low: float,
    display_high: float,
    integer: bool,
) -> tuple[float, float]:
    """Return the exact finite pixel-value interval for the current frame.

    Histogram interaction is deliberately bounded by the true minimum and
    maximum detector values. Percentile clipping, low-count-only viewports and
    automatic padding are not used. ``display_low`` and ``display_high`` are
    accepted for API compatibility but do not enlarge the pixel domain.
    """
    del display_low, display_high, integer
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return 0.0, 1.0
    low = float(np.min(data))
    high = float(np.max(data))
    # A constant-valued image has no adjustable interval. Keep a small plotting
    # span so the single histogram bar and its value remain visible.
    return _expand_degenerate_range(low, high)


@dataclass(frozen=True)
class HistogramIntensityAxis:
    """Smooth full-range coordinate used by the interactive histogram.

    A detector frame can contain millions of low-count pixels and only a few
    extremely bright reflections.  Mapping mouse x linearly to the raw
    minimum/maximum makes one pixel of mouse movement jump by thousands or
    millions of counts.  The asinh coordinate keeps the exact extrema while
    expanding the populated low-count region and compressing the sparse tail.
    """

    raw_low: float
    raw_high: float
    origin: float
    scale: float
    axis_low: float
    axis_high: float

    @classmethod
    def from_values(
        cls,
        values: np.ndarray,
        *,
        raw_low: float,
        raw_high: float,
        integer: bool,
    ) -> "HistogramIntensityAxis":
        data = np.asarray(values, dtype=np.float64).reshape(-1)
        data = data[np.isfinite(data)]
        span = max(float(raw_high) - float(raw_low), np.finfo(np.float64).eps)

        if raw_low <= 0.0 <= raw_high:
            origin = 0.0
        elif raw_low > 0.0:
            origin = float(raw_low)
        else:
            origin = float(raw_high)

        if integer:
            # Quarter-count softening gives integer thresholds such as 0, 1,
            # 2 and 3 a generous control area even when a rare reflection is
            # many orders of magnitude brighter.
            scale = 0.25
        else:
            distances = np.abs(data - origin)
            distances = distances[(distances > 0.0) & np.isfinite(distances)]
            if distances.size:
                # A lower quartile gives useful precision around the dense
                # signal without letting a handful of hot pixels set the scale.
                q25 = float(np.percentile(distances, 25.0))
                scale = q25 / 4.0
            else:
                scale = span / 256.0
            lower_bound = max(
                span / 1_000_000_000_000_000.0,
                abs(origin) * np.finfo(np.float64).eps * 8.0,
                np.finfo(np.float64).eps,
            )
            upper_bound = max(lower_bound, span / 8.0)
            scale = min(upper_bound, max(lower_bound, scale))

        scale = max(float(scale), np.finfo(np.float64).eps)
        axis_low = math.asinh((float(raw_low) - origin) / scale)
        axis_high = math.asinh((float(raw_high) - origin) / scale)
        if not math.isfinite(axis_low) or not math.isfinite(axis_high) or axis_high <= axis_low:
            axis_low, axis_high = 0.0, 1.0
        return cls(
            raw_low=float(raw_low),
            raw_high=float(raw_high),
            origin=origin,
            scale=scale,
            axis_low=float(axis_low),
            axis_high=float(axis_high),
        )

    @property
    def axis_span(self) -> float:
        return max(self.axis_high - self.axis_low, np.finfo(np.float64).eps)

    def raw_to_axis(self, value: float) -> float:
        value = min(self.raw_high, max(self.raw_low, float(value)))
        return float(math.asinh((value - self.origin) / self.scale))

    def axis_to_raw(self, value: float) -> float:
        value = min(self.axis_high, max(self.axis_low, float(value)))
        raw = self.origin + self.scale * math.sinh(value)
        return float(min(self.raw_high, max(self.raw_low, raw)))

    def raw_to_fraction(self, value: float) -> float:
        return min(1.0, max(0.0, (self.raw_to_axis(value) - self.axis_low) / self.axis_span))

    def fraction_to_raw(self, fraction: float) -> float:
        fraction = min(1.0, max(0.0, float(fraction)))
        return self.axis_to_raw(self.axis_low + fraction * self.axis_span)

    def transform_array(self, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=np.float64)
        data = np.clip(data, self.raw_low, self.raw_high)
        return np.arcsinh((data - self.origin) / self.scale)


class ImageCanvas(tk.Canvas):
    """Canvas with source-coordinate zooming, panning and pixel overlays."""

    def __init__(
        self,
        master: tk.Misc,
        status_callback,
        view_callback=None,
        beam_center_callback=None,
        roi_callback=None,
    ) -> None:
        super().__init__(
            master,
            background=UI_CANVAS,
            highlightthickness=0,
            # The native Tk crosshair can disappear on mid-gray detector
            # pixels. Hide it and draw a two-tone overlay instead.
            cursor="none",
        )
        self.status_callback = status_callback
        self.view_callback = view_callback
        self.beam_center_callback = beam_center_callback
        self.roi_callback = roi_callback
        self.frame: np.ndarray | None = None
        self.display: np.ndarray | None = None
        self.zoom = 1.0
        self.center_x = 0.0
        self.center_y = 0.0
        self._photo: ImageTk.PhotoImage | None = None
        self._render_after: str | None = None
        self._pan_start: tuple[int, int, float, float] | None = None
        self._pan_button: int | None = None
        self._pan_dragged = False
        self._pan_threshold_px = 4.0
        # Viewer-only beam center. This coordinate is never written to HDF5/HDXF.
        # It intentionally lives on the persistent canvas so it survives frame changes.
        self.manual_beam_center: tuple[int, int] | None = None
        # Viewer-only geometry used to render resolution rings.  Values are
        # copied from preserved HDF5/HDXF metadata and are never written back.
        self.resolution_geometry: dict[str, float] | None = None
        self.show_resolution_rings = True

        # Viewer-only Region of Interest (ROI).  Every shape keeps an integer
        # bounding box (x, y, width, height) so the sidebar can edit it
        # numerically. Polygon/freehand shapes additionally preserve their
        # integer detector-pixel vertices. Nothing is written to HDF5/HDXF.
        self.roi: tuple[int, int, int, int] | None = None
        self.roi_shape = "Rectangle"
        self.roi_draw_shape = "Rectangle"
        self.roi_points: list[tuple[int, int]] = []
        # Annulus/Ring ROI: diameter of the excluded concentric inner circle,
        # in detector pixels. It is Viewer-only and never written to source data.
        self.roi_inner_diameter = 1
        self.roi_draw_mode = False
        self._roi_drag_start: tuple[int, int] | None = None
        self._roi_drag_current: tuple[int, int] | None = None
        self._roi_drag_points: list[tuple[int, int]] = []
        self._roi_polygon_points: list[tuple[int, int]] = []
        self._roi_polygon_preview: tuple[int, int] | None = None
        # Hover/move state for existing ROI outlines. When the pointer is on
        # an ROI outline, the cursor changes to a move cursor and a left-drag
        # translates the ROI without resizing it.
        self._roi_hover_move = False
        self._roi_move_start_src: tuple[float, float] | None = None
        self._roi_move_start_screen: tuple[float, float] | None = None
        self._roi_move_origin: tuple[int, int, int, int] | None = None
        self._roi_move_origin_points: list[tuple[int, int]] = []
        self._roi_move_dragged = False
        # Existing ROI outlines are resize handles. The pale-blue selection fill is
        # a move handle. Resize state is kept separate so line dragging never
        # turns into image panning or Beam Center anchoring.
        self._roi_hover_resize_mode: str | None = None
        self._roi_resize_mode: str | None = None
        self._roi_resize_start_src: tuple[float, float] | None = None
        self._roi_resize_start_screen: tuple[float, float] | None = None
        self._roi_resize_origin: tuple[int, int, int, int] | None = None
        self._roi_resize_origin_points: list[tuple[int, int]] = []
        self._roi_resize_origin_inner_diameter = 0
        self._roi_resize_dragged = False
        self._show_values = True
        self._font_cache: dict[int, tkfont.Font] = {}
        self._pointer_xy: tuple[float, float] | None = None
        self._pointer_inside = False

        self.bind("<Configure>", self._on_resize)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", self._on_leave)
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1.25))
        self.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 0.8))

        for button in (1, 2, 3):
            self.bind(f"<ButtonPress-{button}>", self._pan_begin)
            self.bind(f"<B{button}-Motion>", self._pan_move)
            self.bind(f"<ButtonRelease-{button}>", self._pan_end)

    def _view_state(self) -> dict[str, float] | None:
        if self.frame is None:
            return None
        image_h, image_w = self.frame.shape
        if image_w <= 0 or image_h <= 0:
            return None
        return {
            "zoom": float(self.zoom),
            "center_x_fraction": float(self.center_x / image_w),
            "center_y_fraction": float(self.center_y / image_h),
        }

    def _notify_view_changed(self) -> None:
        if self.view_callback is None:
            return
        state = self._view_state()
        if state is not None:
            self.view_callback(state)

    def apply_synced_view(self, state: dict[str, float]) -> None:
        """Apply a view broadcast without rebroadcasting it."""
        if self.frame is None:
            return
        image_h, image_w = self.frame.shape
        self.zoom = max(0.01, min(80.0, float(state.get("zoom", self.zoom))))
        self.center_x = float(state.get("center_x_fraction", 0.5)) * image_w
        self.center_y = float(state.get("center_y_fraction", 0.5)) * image_h
        self._clamp_center()
        self.schedule_render()

    def set_frame(self, frame: np.ndarray, display: np.ndarray) -> None:
        old_shape = None if self.frame is None else self.frame.shape
        self.frame = frame
        self.display = display
        if old_shape != frame.shape:
            self.fit_to_window(notify=False)
        else:
            self.schedule_render()

    def set_display(self, display: np.ndarray) -> None:
        self.display = display
        self.schedule_render()

    def set_show_values(self, value: bool) -> None:
        self._show_values = bool(value)
        self.schedule_render()

    def fit_to_window(self, *, notify: bool = True) -> None:
        if self.frame is None:
            return
        self.update_idletasks()
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        image_h, image_w = self.frame.shape
        self.zoom = max(0.01, min(width / image_w, height / image_h))
        self.center_x = image_w / 2.0
        self.center_y = image_h / 2.0
        self._clamp_center()
        self.schedule_render()
        if notify:
            self._notify_view_changed()

    def one_to_one(self, *, notify: bool = True) -> None:
        if self.frame is None:
            return
        self.zoom = 1.0
        self._clamp_center()
        self.schedule_render()
        if notify:
            self._notify_view_changed()

    def set_zoom(self, zoom: float, *, notify: bool = True) -> None:
        if self.frame is None:
            return
        self.zoom = max(0.01, min(80.0, float(zoom)))
        self._clamp_center()
        self.schedule_render()
        if notify:
            self._notify_view_changed()

    def zoom_in(self) -> None:
        self._zoom_at(self.winfo_width() // 2, self.winfo_height() // 2, 1.25)

    def zoom_out(self) -> None:
        self._zoom_at(self.winfo_width() // 2, self.winfo_height() // 2, 0.8)

    def schedule_render(self) -> None:
        if self._render_after is not None:
            self.after_cancel(self._render_after)
        self._render_after = self.after_idle(self.render)

    def render(self) -> None:
        self._render_after = None
        self.delete("all")
        self._photo = None

        if self.frame is None or self.display is None:
            width = max(1, self.winfo_width())
            height = max(1, self.winfo_height())
            card_w = min(430, max(260, width - 80))
            card_h = 178
            x0 = (width - card_w) / 2
            y0 = (height - card_h) / 2
            x1 = x0 + card_w
            y1 = y0 + card_h
            self.create_rectangle(
                x0, y0, x1, y1,
                fill=UI_SURFACE,
                outline=UI_BORDER,
                width=1,
            )
            self.create_rectangle(
                x0, y0, x0 + 4, y1,
                fill=UI_ACCENT,
                outline="",
            )
            self.create_text(
                width / 2,
                y0 + 45,
                text="HDXF",
                fill=UI_ACCENT,
                font=("Segoe UI", 24, "bold"),
            )
            self.create_text(
                width / 2,
                y0 + 87,
                text="Open a detector archive",
                fill=UI_TEXT,
                font=("Segoe UI", 13, "bold"),
            )
            self.create_text(
                width / 2,
                y0 + 119,
                text="Use Open, press Ctrl+O, or drop .hdxf / .h5 / .hdf5",
                fill=UI_TEXT_MUTED,
                font=("Segoe UI", 10),
            )
            self.create_text(
                width / 2,
                y0 + 146,
                text="HDXF and HDF5 can be compared side-by-side in up to four synchronized views",
                fill=UI_TEXT_DIM,
                font=("Segoe UI", 9),
            )
            return

        canvas_w = max(1, self.winfo_width())
        canvas_h = max(1, self.winfo_height())
        image_h, image_w = self.frame.shape
        left = canvas_w / 2.0 - self.center_x * self.zoom
        top = canvas_h / 2.0 - self.center_y * self.zoom

        sx0 = max(0, int(math.floor((0.0 - left) / self.zoom)))
        sy0 = max(0, int(math.floor((0.0 - top) / self.zoom)))
        sx1 = min(image_w, int(math.ceil((canvas_w - left) / self.zoom)))
        sy1 = min(image_h, int(math.ceil((canvas_h - top) / self.zoom)))

        if sx1 <= sx0 or sy1 <= sy0:
            return

        crop = self._apply_roi_selection_shade(
            self.display[sy0:sy1, sx0:sx1],
            sx0, sy0, sx1, sy1,
        )
        target_w = max(1, int(round((sx1 - sx0) * self.zoom)))
        target_h = max(1, int(round((sy1 - sy0) * self.zoom)))
        resample = Image.Resampling.NEAREST if self.zoom >= 1 else Image.Resampling.BILINEAR
        image = Image.fromarray(crop, mode="RGB")
        if image.size != (target_w, target_h):
            image = image.resize((target_w, target_h), resample=resample)

        self._photo = ImageTk.PhotoImage(image)
        screen_x = left + sx0 * self.zoom
        screen_y = top + sy0 * self.zoom
        self.create_image(
            screen_x,
            screen_y,
            anchor="nw",
            image=self._photo,
            tags=("image",),
        )
        self.create_rectangle(
            left,
            top,
            left + image_w * self.zoom,
            top + image_h * self.zoom,
            outline=UI_BORDER,
            width=1,
        )

        visible_count = (sx1 - sx0) * (sy1 - sy0)
        if self._show_values and self.zoom >= 14.0 and visible_count <= 2500:
            font_size = max(8, min(18, int(self.zoom * 0.34)))
            font = self._font_cache.get(font_size)
            if font is None:
                font = tkfont.Font(family="Consolas", size=font_size)
                self._font_cache[font_size] = font

            for y in range(sy0, sy1):
                for x in range(sx0, sx1):
                    px = left + (x + 0.5) * self.zoom
                    py = top + (y + 0.5) * self.zoom
                    gray = int(self.display[y, x])
                    color = "#ffffff" if gray < 118 else "#000000"
                    self.create_text(
                        px,
                        py,
                        text=_format_overlay_value(self.frame[y, x]),
                        fill=color,
                        font=font,
                        anchor="center",
                        tags=("pixel-value",),
                    )

        self._draw_resolution_ring_overlay()
        self._draw_roi_overlay()
        self._draw_beam_center_overlay()
        self._draw_pointer_overlay()
        self.status_callback(self._current_pointer_info())

    def set_resolution_geometry(self, geometry: dict[str, float] | None) -> None:
        """Set Viewer-only detector geometry used by resolution rings.

        Expected keys are ``wavelength_angstrom``, ``distance_m``,
        ``pixel_size_x_m`` and ``pixel_size_y_m``.  The geometry is only held
        in memory and is never written to HDF5 or HDXF.
        """
        if geometry is None:
            self.resolution_geometry = None
        else:
            try:
                clean = {
                    "wavelength_angstrom": float(geometry["wavelength_angstrom"]),
                    "distance_m": float(geometry["distance_m"]),
                    "pixel_size_x_m": float(geometry["pixel_size_x_m"]),
                    "pixel_size_y_m": float(geometry["pixel_size_y_m"]),
                }
                if not all(math.isfinite(value) and value > 0.0 for value in clean.values()):
                    clean = None
                self.resolution_geometry = clean
            except Exception:
                self.resolution_geometry = None
        self.schedule_render()

    @staticmethod
    def _resolution_ring_candidates() -> tuple[float, ...]:
        """Common crystallographic d-spacings in Angstrom, coarse to fine."""
        return (
            50.0, 40.0, 30.0, 25.0, 20.0, 15.0, 12.0, 10.0,
            8.0, 7.0, 6.0, 5.0, 4.5, 4.0, 3.5, 3.0,
            2.5, 2.2, 2.0, 1.8, 1.6, 1.5, 1.4, 1.3,
            1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5,
        )

    def _draw_resolution_ring_overlay(self) -> None:
        """Draw d-spacing rings around the manually anchored beam center.

        Geometry assumes the detector plane is perpendicular to the direct
        beam.  For each d-spacing, Bragg's law gives theta and the detector
        radius is ``distance * tan(2*theta)``.  Unequal X/Y pixel sizes are
        represented as an ellipse.  No source metadata is modified.
        """
        self.delete("resolution-ring")
        if (
            not self.show_resolution_rings
            or self.frame is None
            or self.manual_beam_center is None
            or self.resolution_geometry is None
        ):
            return

        geometry = self.resolution_geometry
        wavelength = float(geometry["wavelength_angstrom"])
        distance = float(geometry["distance_m"])
        pixel_x = float(geometry["pixel_size_x_m"])
        pixel_y = float(geometry["pixel_size_y_m"])
        if wavelength <= 0.0 or distance <= 0.0 or pixel_x <= 0.0 or pixel_y <= 0.0:
            return

        image_h, image_w = self.frame.shape
        beam_x, beam_y = self.manual_beam_center
        source_cx = beam_x + 0.5
        source_cy = beam_y + 0.5
        screen_cx, screen_cy = self._source_to_screen(source_cx, source_cy)
        canvas_w = max(1, self.winfo_width())
        canvas_h = max(1, self.winfo_height())

        # Maximum physical radius needed to reach any detector corner.  Rings
        # beyond this cannot intersect the detector image and are omitted.
        corner_radii_m = []
        for edge_x in (0.0, float(image_w)):
            for edge_y in (0.0, float(image_h)):
                dx_m = (edge_x - source_cx) * pixel_x
                dy_m = (edge_y - source_cy) * pixel_y
                corner_radii_m.append(math.hypot(dx_m, dy_m))
        detector_radius_m = max(corner_radii_m, default=0.0)
        if detector_radius_m <= 0.0:
            return

        rings: list[tuple[float, float, float, float]] = []
        for d_spacing in self._resolution_ring_candidates():
            ratio = wavelength / (2.0 * d_spacing)
            if ratio <= 0.0 or ratio >= 1.0:
                continue
            theta = math.asin(ratio)
            two_theta = 2.0 * theta
            radius_m = distance * math.tan(two_theta)
            if not math.isfinite(radius_m) or radius_m <= 0.0:
                continue
            if radius_m > detector_radius_m * 1.06:
                continue
            radius_x_px = radius_m / pixel_x
            radius_y_px = radius_m / pixel_y
            # Skip tiny rings that would be hidden by the beam-center marker.
            if max(radius_x_px, radius_y_px) * self.zoom < 18.0:
                continue
            rings.append((d_spacing, radius_m, radius_x_px, radius_y_px))

        if not rings:
            return

        # Keep labels legible: select at most ten rings, spaced across the
        # detector radius instead of drawing every candidate.
        selected: list[tuple[float, float, float, float]] = []
        minimum_spacing_m = detector_radius_m * 0.075
        for ring in sorted(rings, key=lambda item: item[1]):
            if not selected or ring[1] - selected[-1][1] >= minimum_spacing_m:
                selected.append(ring)
        if rings[-1] not in selected:
            selected.append(rings[-1])
        if len(selected) > 10:
            indices = np.linspace(0, len(selected) - 1, 10).round().astype(int)
            selected = [selected[int(i)] for i in sorted(set(indices.tolist()))]

        ring_color = "#38BDF8"
        label_font = self._font_cache.get(-12)
        if label_font is None:
            label_font = tkfont.Font(family="Segoe UI", size=8, weight="bold")
            self._font_cache[-12] = label_font

        for d_spacing, _radius_m, radius_x_px, radius_y_px in selected:
            rx = radius_x_px * self.zoom
            ry = radius_y_px * self.zoom
            if rx > 2_000_000 or ry > 2_000_000:
                continue
            x0 = screen_cx - rx
            y0 = screen_cy - ry
            x1 = screen_cx + rx
            y1 = screen_cy + ry
            if x1 < 0 or y1 < 0 or x0 > canvas_w or y0 > canvas_h:
                continue
            self.create_oval(
                x0, y0, x1, y1,
                outline=ring_color,
                width=1,
                dash=(5, 4),
                tags=("resolution-ring",),
            )

            # Put the label near the upper-right quadrant of the ring so it
            # remains easy to associate with the corresponding circle/ellipse.
            angle = math.radians(-38.0)
            label_x = screen_cx + rx * math.cos(angle)
            label_y = screen_cy + ry * math.sin(angle)
            if -80 <= label_x <= canvas_w + 80 and -30 <= label_y <= canvas_h + 30:
                label = f"{d_spacing:g} Å"
                self.create_text(
                    label_x + 4, label_y - 3,
                    text=label,
                    fill=ring_color,
                    font=label_font,
                    anchor="sw",
                    tags=("resolution-ring",),
                )

        self.tag_raise("resolution-ring")

    def set_roi_draw_shape(self, shape: str, *, notify: bool = True) -> bool:
        """Select the ROI shape used for drawing and numeric editing."""
        shape = str(shape).strip().title()
        if shape not in ROI_SHAPES:
            return False
        previous_shape = self.roi_shape
        self.roi_draw_shape = shape

        # When an ROI already exists, switching the selector changes its shape
        # while preserving the same integer bounding box. For Polygon/Freehand,
        # initialise a simple four-corner contour if no contour exists yet.
        if self.roi is not None:
            self.roi_shape = shape
            if shape in ("Polygon", "Freehand") and len(self.roi_points) < 3:
                x, y, width, height = self.roi
                self.roi_points = [
                    (x, y),
                    (x + width - 1, y),
                    (x + width - 1, y + height - 1),
                    (x, y + height - 1),
                ]
            elif shape not in ("Polygon", "Freehand"):
                self.roi_points = []
            if shape in ("Circle", "Annulus"):
                x, y, width, height = self.roi
                size = max(1, min(width, height))
                self.roi = (x, y, size, size)
                if shape == "Annulus":
                    # Start with a clearly visible 50% inner exclusion region
                    # when converting another ROI type into an annulus. If an
                    # existing annulus is merely being reselected, preserve its
                    # user-edited inner diameter.
                    max_inner = max(0, size - 1)
                    if previous_shape != "Annulus" or not (0 < int(self.roi_inner_diameter) <= max_inner):
                        self.roi_inner_diameter = max(1, size // 2) if size > 1 else 0
            self.schedule_render()
            if notify and self.roi_callback is not None:
                self.roi_callback(self.roi)
        return True

    def get_roi_shape(self) -> str:
        return self.roi_shape if self.roi is not None else self.roi_draw_shape

    def set_roi_draw_mode(self, enabled: bool) -> bool:
        """Enable/disable Viewer-only ROI drawing with the right mouse button."""
        enabled = bool(enabled) and self.frame is not None
        self.roi_draw_mode = enabled
        self._roi_drag_start = None
        self._roi_drag_current = None
        self._roi_drag_points = []
        self._roi_polygon_points = []
        self._roi_polygon_preview = None
        self._roi_hover_move = False
        self._roi_move_start_src = None
        self._roi_move_start_screen = None
        self._roi_move_origin = None
        self._roi_move_origin_points = []
        self._roi_move_dragged = False
        self._roi_hover_resize_mode = None
        self._roi_resize_mode = None
        self._roi_resize_start_src = None
        self._roi_resize_start_screen = None
        self._roi_resize_origin = None
        self._roi_resize_origin_points = []
        self._roi_resize_origin_inner_diameter = 0
        self._roi_resize_dragged = False
        self.configure(cursor="crosshair" if enabled else "none")
        self.schedule_render()
        return self.roi_draw_mode

    @staticmethod
    def _roi_bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
        if not points:
            return None
        xs = [int(point[0]) for point in points]
        ys = [int(point[1]) for point in points]
        left = min(xs)
        top = min(ys)
        return left, top, max(xs) - left + 1, max(ys) - top + 1

    def _clamp_roi_bbox(self, x: int, y: int, width: int, height: int) -> tuple[int, int, int, int] | None:
        if self.frame is None:
            return None
        image_h, image_w = self.frame.shape
        if image_w <= 0 or image_h <= 0:
            return None
        x = min(max(0, int(x)), image_w - 1)
        y = min(max(0, int(y)), image_h - 1)
        width = max(1, int(width))
        height = max(1, int(height))
        width = min(width, image_w - x)
        height = min(height, image_h - y)
        return x, y, width, height

    def _scale_roi_points_to_bbox(
        self,
        old_bbox: tuple[int, int, int, int],
        new_bbox: tuple[int, int, int, int],
    ) -> list[tuple[int, int]]:
        """Translate/scale polygon or freehand vertices to a numerically edited bbox."""
        if not self.roi_points:
            return []
        ox, oy, ow, oh = old_bbox
        nx, ny, nw, nh = new_bbox
        old_dx = max(1, ow - 1)
        old_dy = max(1, oh - 1)
        new_dx = max(0, nw - 1)
        new_dy = max(0, nh - 1)
        points: list[tuple[int, int]] = []
        for px, py in self.roi_points:
            fx = (px - ox) / old_dx if ow > 1 else 0.0
            fy = (py - oy) / old_dy if oh > 1 else 0.0
            tx = nx + int(round(fx * new_dx))
            ty = ny + int(round(fy * new_dy))
            points.append((tx, ty))
        return points

    def set_roi(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        notify: bool = True,
        shape: str | None = None,
        points: list[tuple[int, int]] | None = None,
        inner_diameter: int | None = None,
    ) -> bool:
        """Set a Viewer-only ROI and keep all coordinates on detector pixels."""
        if self.frame is None:
            return False
        try:
            x = int(x)
            y = int(y)
            width = int(width)
            height = int(height)
        except (TypeError, ValueError):
            return False

        target_shape = str(shape or (self.roi_shape if self.roi is not None else self.roi_draw_shape)).title()
        if target_shape not in ROI_SHAPES:
            target_shape = "Rectangle"

        if target_shape in ("Circle", "Annulus"):
            # Circle/Annulus use a square integer outer bounding box. The
            # caller/sidebar keeps both Width and Height synchronized.
            size = max(1, min(width, height))
            width = size
            height = size

        clamped = self._clamp_roi_bbox(x, y, width, height)
        if clamped is None:
            return False
        x, y, width, height = clamped

        old_bbox = self.roi
        old_shape = self.roi_shape
        self.roi = (x, y, width, height)
        self.roi_shape = target_shape
        self.roi_draw_shape = target_shape

        if target_shape == "Annulus":
            outer = max(1, min(width, height))
            max_inner = max(0, outer - 1)
            if inner_diameter is None:
                candidate_inner = int(self.roi_inner_diameter)
                if not (0 < candidate_inner <= max_inner):
                    candidate_inner = max(1, outer // 2) if outer > 1 else 0
            else:
                try:
                    candidate_inner = int(inner_diameter)
                except (TypeError, ValueError):
                    candidate_inner = max(1, outer // 2) if outer > 1 else 0
            self.roi_inner_diameter = min(max_inner, max(0 if outer <= 1 else 1, candidate_inner))

        if target_shape in ("Polygon", "Freehand"):
            if points is not None:
                image_h, image_w = self.frame.shape
                cleaned: list[tuple[int, int]] = []
                for px, py in points:
                    px = min(image_w - 1, max(0, int(px)))
                    py = min(image_h - 1, max(0, int(py)))
                    if not cleaned or cleaned[-1] != (px, py):
                        cleaned.append((px, py))
                self.roi_points = cleaned
                bbox = self._roi_bbox_from_points(self.roi_points)
                if bbox is not None:
                    self.roi = bbox
            elif old_bbox is not None and old_shape in ("Polygon", "Freehand") and self.roi_points:
                self.roi_points = self._scale_roi_points_to_bbox(old_bbox, self.roi)
            elif len(self.roi_points) < 3:
                self.roi_points = [
                    (x, y),
                    (x + width - 1, y),
                    (x + width - 1, y + height - 1),
                    (x, y + height - 1),
                ]
        else:
            self.roi_points = []

        self.roi_draw_mode = False
        self._roi_drag_start = None
        self._roi_drag_current = None
        self._roi_drag_points = []
        self._roi_polygon_points = []
        self._roi_polygon_preview = None
        self.configure(cursor="none")
        self.schedule_render()
        if notify and self.roi_callback is not None:
            self.roi_callback(self.roi)
        return True

    def clear_roi(self, *, notify: bool = True) -> None:
        """Clear the Viewer-only ROI without modifying HDF5/HDXF source data."""
        had_roi = self.roi is not None or self.roi_draw_mode
        self.roi = None
        self.roi_points = []
        self.roi_inner_diameter = 1
        self.roi_draw_mode = False
        self._roi_drag_start = None
        self._roi_drag_current = None
        self._roi_drag_points = []
        self._roi_polygon_points = []
        self._roi_polygon_preview = None
        self._roi_hover_move = False
        self._roi_move_start_src = None
        self._roi_move_start_screen = None
        self._roi_move_origin = None
        self._roi_move_origin_points = []
        self._roi_move_dragged = False
        self._roi_hover_resize_mode = None
        self._roi_resize_mode = None
        self._roi_resize_start_src = None
        self._roi_resize_start_screen = None
        self._roi_resize_origin = None
        self._roi_resize_origin_points = []
        self._roi_resize_origin_inner_diameter = 0
        self._roi_resize_dragged = False
        self.configure(cursor="none")
        self.schedule_render()
        if notify and had_roi and self.roi_callback is not None:
            self.roi_callback(None)

    def get_roi(self) -> tuple[int, int, int, int] | None:
        return self.roi

    def get_roi_points(self) -> tuple[tuple[int, int], ...]:
        return tuple(self.roi_points)

    def get_roi_inner_diameter(self) -> int:
        """Return the excluded inner-circle diameter for an Annulus ROI."""
        return int(self.roi_inner_diameter)

    def set_roi_inner_diameter(self, diameter: int, *, notify: bool = True) -> bool:
        """Set the Annulus inner diameter in integer detector pixels."""
        if self.roi is None or self.roi_shape != "Annulus":
            return False
        try:
            diameter = int(diameter)
        except (TypeError, ValueError):
            return False
        outer = max(1, min(int(self.roi[2]), int(self.roi[3])))
        # Diameter 0 is permitted only for the degenerate one-pixel case.
        max_inner = max(0, outer - 1)
        diameter = min(max_inner, max(0 if outer <= 1 else 1, diameter))
        self.roi_inner_diameter = diameter
        self.schedule_render()
        if notify and self.roi_callback is not None:
            self.roi_callback(self.roi)
        return True

    def _source_pixel_at_screen(self, x: float, y: float) -> tuple[int, int] | None:
        """Return the integer detector pixel under a canvas coordinate."""
        if self.frame is None:
            return None
        sx, sy = self._screen_to_source(x, y)
        image_h, image_w = self.frame.shape
        if sx < 0.0 or sy < 0.0 or sx >= image_w or sy >= image_h:
            return None
        return int(math.floor(sx)), int(math.floor(sy))

    def _roi_bbox_from_drag(
        self,
        start: tuple[int, int],
        current: tuple[int, int],
        shape: str,
    ) -> tuple[int, int, int, int]:
        x0, y0 = start
        x1, y1 = current
        if shape not in ("Circle", "Annulus"):
            left = min(x0, x1)
            top = min(y0, y1)
            return left, top, abs(x1 - x0) + 1, abs(y1 - y0) + 1

        # Circle/Annulus: use the larger drag distance but keep the square inside the
        # detector. The initial click acts as one corner of the square.
        assert self.frame is not None
        image_h, image_w = self.frame.shape
        sx = 1 if x1 >= x0 else -1
        sy = 1 if y1 >= y0 else -1
        side = max(abs(x1 - x0), abs(y1 - y0)) + 1
        max_x_side = (image_w - x0) if sx > 0 else (x0 + 1)
        max_y_side = (image_h - y0) if sy > 0 else (y0 + 1)
        side = max(1, min(side, max_x_side, max_y_side))
        end_x = x0 + sx * (side - 1)
        end_y = y0 + sy * (side - 1)
        return min(x0, end_x), min(y0, end_y), side, side

    def _current_roi_for_drawing(self) -> tuple[str, tuple[int, int, int, int] | None, list[tuple[int, int]]]:
        shape = self.roi_draw_shape if self.roi_draw_mode else self.roi_shape
        if shape in ("Polygon", "Freehand"):
            points = self._roi_polygon_points if shape == "Polygon" else self._roi_drag_points
            if points:
                return shape, self._roi_bbox_from_points(points), list(points)
        if self._roi_drag_start is not None and self._roi_drag_current is not None:
            bbox = self._roi_bbox_from_drag(self._roi_drag_start, self._roi_drag_current, shape)
            return shape, bbox, []
        return self.roi_shape, self.roi, list(self.roi_points)

    def _roi_polygon_mask(
        self,
        points: list[tuple[int, int]],
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Return a point-in-polygon mask evaluated at detector pixel centres."""
        x, y, width, height = bbox
        if len(points) < 3:
            return np.zeros((height, width), dtype=bool)
        xx, yy = np.meshgrid(
            x + np.arange(width, dtype=np.float64) + 0.5,
            y + np.arange(height, dtype=np.float64) + 0.5,
        )
        inside = np.zeros((height, width), dtype=bool)
        vertices = [(float(px) + 0.5, float(py) + 0.5) for px, py in points]
        j = len(vertices) - 1
        eps = np.finfo(np.float64).eps
        for i in range(len(vertices)):
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            crosses = ((yi > yy) != (yj > yy))
            x_cross = (xj - xi) * (yy - yi) / ((yj - yi) + eps) + xi
            inside ^= crosses & (xx < x_cross)
            j = i
        return inside

    def roi_mask(self) -> np.ndarray | None:
        """Return a boolean mask for the current shape within its bounding box."""
        if self.roi is None:
            return None
        x, y, width, height = self.roi
        shape = self.roi_shape
        if shape == "Rectangle":
            return np.ones((height, width), dtype=bool)
        if shape in ("Ellipse", "Circle", "Annulus"):
            yy, xx = np.ogrid[:height, :width]
            cx = width / 2.0
            cy = height / 2.0
            rx = max(width / 2.0, 0.5)
            ry = max(height / 2.0, 0.5)
            outer = (((xx + 0.5 - cx) / rx) ** 2 + ((yy + 0.5 - cy) / ry) ** 2) <= 1.0
            if shape != "Annulus":
                return outer

            # The inner circle is an exclusion mask. Pixels whose centres lie
            # inside the inner circle are deliberately omitted from all ROI
            # statistics, exactly leaving the ring between the two circles.
            inner_d = max(0, min(int(self.roi_inner_diameter), min(width, height) - 1))
            if inner_d <= 0:
                return outer
            inner_r = max(inner_d / 2.0, 0.5)
            inner = ((xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2) <= inner_r ** 2
            return outer & ~inner
        if shape in ("Polygon", "Freehand"):
            return self._roi_polygon_mask(list(self.roi_points), self.roi)
        return np.ones((height, width), dtype=bool)

    def roi_values(self, frame: np.ndarray) -> np.ndarray:
        """Return only pixels actually inside the current ROI shape."""
        if self.roi is None:
            return np.asarray([], dtype=np.asarray(frame).dtype)
        x, y, width, height = self.roi
        region = np.asarray(frame[y:y + height, x:x + width])
        mask = self.roi_mask()
        if mask is None or mask.shape != region.shape:
            return region.reshape(-1)
        return region[mask]

    def _apply_roi_selection_shade(
        self,
        crop: np.ndarray,
        sx0: int,
        sy0: int,
        sx1: int,
        sy1: int,
    ) -> np.ndarray:
        """Blend a pale-blue semi-transparent-looking layer over the selected ROI.

        The source detector data and ``self.display`` are never modified. The
        blend is applied only to the temporary visible crop, while the green
        vector outline is drawn afterwards and therefore stays crisp.
        """
        gray = np.array(crop, dtype=np.uint8, copy=True)
        result = np.repeat(gray[:, :, None], 3, axis=2)
        if self.roi is None:
            return result
        x, y, width, height = self.roi
        ix0 = max(int(sx0), int(x))
        iy0 = max(int(sy0), int(y))
        ix1 = min(int(sx1), int(x + width))
        iy1 = min(int(sy1), int(y + height))
        if ix1 <= ix0 or iy1 <= iy0:
            return result
        mask = self.roi_mask()
        if mask is None:
            return result
        mask_part = mask[iy0 - y:iy1 - y, ix0 - x:ix1 - x]
        if mask_part.size == 0 or not np.any(mask_part):
            return result
        target = result[iy0 - sy0:iy1 - sy0, ix0 - sx0:ix1 - sx0]
        # 50 % deeper blue + 50 % original grayscale gives a stronger selected
        # region while preserving diffraction detail.
        overlay_rgb = np.array([96.0, 182.0, 236.0], dtype=np.float32)
        alpha = 0.50
        original = target[mask_part].astype(np.float32, copy=False)
        blended = original * (1.0 - alpha) + overlay_rgb * alpha
        target[mask_part] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
        return result

    @staticmethod
    def _distance_point_to_segment_screen(
        px: float,
        py: float,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> float:
        """Return the shortest screen-space distance from a point to a segment."""
        dx = x1 - x0
        dy = y1 - y0
        if dx == 0.0 and dy == 0.0:
            return math.hypot(px - x0, py - y0)
        t = ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy)
        t = min(1.0, max(0.0, t))
        qx = x0 + t * dx
        qy = y0 + t * dy
        return math.hypot(px - qx, py - qy)

    def _point_near_ellipse_outline(
        self,
        px: float,
        py: float,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        tolerance_px: float,
    ) -> bool:
        """Return True if a screen-space point is near an ellipse boundary."""
        left = min(x0, x1)
        right = max(x0, x1)
        top = min(y0, y1)
        bottom = max(y0, y1)
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        rx = max(0.5, (right - left) / 2.0)
        ry = max(0.5, (bottom - top) / 2.0)
        vx = px - cx
        vy = py - cy
        denom = (vx / rx) ** 2 + (vy / ry) ** 2
        if denom <= 0.0:
            return False
        scale = 1.0 / math.sqrt(denom)
        bx = cx + vx * scale
        by = cy + vy * scale
        return math.hypot(px - bx, py - by) <= tolerance_px

    def _roi_fill_hit_test(self, screen_x: float, screen_y: float) -> bool:
        """Return True only for pixels covered by the pale-blue ROI selection fill."""
        if self.frame is None or self.roi is None or self.roi_draw_mode:
            return False
        sx, sy = self._screen_to_source(screen_x, screen_y)
        ix = int(math.floor(sx))
        iy = int(math.floor(sy))
        x, y, width, height = self.roi
        if ix < x or iy < y or ix >= x + width or iy >= y + height:
            return False
        mask = self.roi_mask()
        if mask is None:
            return False
        mx = ix - x
        my = iy - y
        return bool(0 <= my < mask.shape[0] and 0 <= mx < mask.shape[1] and mask[my, mx])

    def _roi_resize_hit_test(
        self,
        screen_x: float,
        screen_y: float,
        tolerance_px: float = 7.0,
    ) -> str | None:
        """Return the resize mode for a pointer near the ROI outline.

        Outline hit-testing has priority over the gray-fill move hit-test.
        Annulus inner and outer circles are intentionally independent: the
        inner line changes only the excluded inner diameter, while the outer
        line changes the outer circle.
        """
        if self.frame is None or self.roi is None or self.roi_draw_mode:
            return None
        shape = self.roi_shape
        x, y, width, height = self.roi
        x0, y0 = self._source_to_screen(float(x), float(y))
        x1, y1 = self._source_to_screen(float(x + width), float(y + height))

        if shape in ("Polygon", "Freehand") and len(self.roi_points) >= 2:
            points = list(self.roi_points)
            if len(points) >= 3:
                points.append(points[0])
            last: tuple[float, float] | None = None
            for px_src, py_src in points:
                sx, sy = self._source_to_screen(px_src + 0.5, py_src + 0.5)
                if last is not None and self._distance_point_to_segment_screen(
                    screen_x, screen_y, last[0], last[1], sx, sy
                ) <= tolerance_px:
                    return "scale"
                last = (sx, sy)
            return None

        if shape in ("Circle", "Annulus"):
            if shape == "Annulus":
                outer_size = min(width, height)
                inner_d = max(0, min(int(self.roi_inner_diameter), outer_size - 1))
                if inner_d > 0:
                    cx_src = x + width / 2.0
                    cy_src = y + height / 2.0
                    half = inner_d / 2.0
                    ix0, iy0 = self._source_to_screen(cx_src - half, cy_src - half)
                    ix1, iy1 = self._source_to_screen(cx_src + half, cy_src + half)
                    if self._point_near_ellipse_outline(
                        screen_x, screen_y, ix0, iy0, ix1, iy1, tolerance_px
                    ):
                        return "annulus-inner"
            if self._point_near_ellipse_outline(screen_x, screen_y, x0, y0, x1, y1, tolerance_px):
                return "radial"
            return None

        if shape == "Ellipse":
            if not self._point_near_ellipse_outline(screen_x, screen_y, x0, y0, x1, y1, tolerance_px):
                return None
            left, right = min(x0, x1), max(x0, x1)
            top, bottom = min(y0, y1), max(y0, y1)
            cx = (left + right) / 2.0
            cy = (top + bottom) / 2.0
            rx = max(1.0, (right - left) / 2.0)
            ry = max(1.0, (bottom - top) / 2.0)
            nx = (screen_x - cx) / rx
            ny = (screen_y - cy) / ry
            ax, ay = abs(nx), abs(ny)
            horizontal = "right" if nx >= 0 else "left"
            vertical = "bottom" if ny >= 0 else "top"
            if ax >= 0.55 and ay >= 0.55:
                return f"{vertical}_{horizontal}"
            return horizontal if ax >= ay else vertical

        # Rectangle: corners win over single edges.
        left, right = min(x0, x1), max(x0, x1)
        top, bottom = min(y0, y1), max(y0, y1)
        within_x = left - tolerance_px <= screen_x <= right + tolerance_px
        within_y = top - tolerance_px <= screen_y <= bottom + tolerance_px
        near_left = within_y and abs(screen_x - left) <= tolerance_px
        near_right = within_y and abs(screen_x - right) <= tolerance_px
        near_top = within_x and abs(screen_y - top) <= tolerance_px
        near_bottom = within_x and abs(screen_y - bottom) <= tolerance_px
        if near_top and near_left:
            return "top_left"
        if near_top and near_right:
            return "top_right"
        if near_bottom and near_left:
            return "bottom_left"
        if near_bottom and near_right:
            return "bottom_right"
        if near_left:
            return "left"
        if near_right:
            return "right"
        if near_top:
            return "top"
        if near_bottom:
            return "bottom"
        return None

    def _roi_outline_hit_test(self, screen_x: float, screen_y: float, tolerance_px: float = 7.0) -> bool:
        """Compatibility helper: True when any resizeable ROI line is hit."""
        return self._roi_resize_hit_test(screen_x, screen_y, tolerance_px) is not None

    @staticmethod
    def _resize_cursor_name(mode: str | None) -> str:
        if mode in ("left", "right"):
            return "sb_h_double_arrow"
        if mode in ("top", "bottom"):
            return "sb_v_double_arrow"
        if mode == "top_left":
            return "top_left_corner"
        if mode == "top_right":
            return "top_right_corner"
        if mode == "bottom_left":
            return "bottom_left_corner"
        if mode == "bottom_right":
            return "bottom_right_corner"
        return "sizing"

    def _set_cursor_safe(self, cursor: str) -> None:
        try:
            self.configure(cursor=cursor)
        except tk.TclError:
            try:
                self.configure(cursor="fleur" if cursor == "sizing" else "crosshair")
            except tk.TclError:
                pass

    def _refresh_hover_cursor(self) -> None:
        """Choose resize on ROI lines, move on pale-blue fill, otherwise normal cursor."""
        if not self._pointer_inside:
            self._set_cursor_safe("")
            return
        resize_mode = self._roi_resize_mode or self._roi_hover_resize_mode
        if resize_mode is not None:
            self._set_cursor_safe(self._resize_cursor_name(resize_mode))
        elif self._roi_move_start_src is not None or self._roi_hover_move:
            self._set_cursor_safe("fleur")
        elif self.roi_draw_mode:
            self._set_cursor_safe("crosshair")
        elif self._pan_start is not None and self._pan_dragged:
            self._set_cursor_safe("fleur")
        else:
            self._set_cursor_safe("none")

    def _update_roi_hover_state(self, screen_x: float, screen_y: float) -> None:
        """Update hover modes with outline resize taking priority over fill move."""
        if self.roi_draw_mode or self._roi_move_start_src is not None or self._roi_resize_mode is not None:
            return
        resize_mode = self._roi_resize_hit_test(screen_x, screen_y)
        self._roi_hover_resize_mode = resize_mode
        self._roi_hover_move = resize_mode is None and self._roi_fill_hit_test(screen_x, screen_y)

    def _translate_current_roi_from_origin(self, delta_x: int, delta_y: int, *, notify: bool = True) -> bool:
        """Move the current ROI by integer pixels without resizing it."""
        if self.frame is None or self._roi_move_origin is None or self.roi is None:
            return False
        origin_x, origin_y, width, height = self._roi_move_origin
        image_h, image_w = self.frame.shape
        min_dx = -origin_x
        max_dx = image_w - (origin_x + width)
        min_dy = -origin_y
        max_dy = image_h - (origin_y + height)
        actual_dx = min(max(int(delta_x), min_dx), max_dx)
        actual_dy = min(max(int(delta_y), min_dy), max_dy)
        self.roi = (origin_x + actual_dx, origin_y + actual_dy, width, height)
        if self.roi_shape in ("Polygon", "Freehand") and self._roi_move_origin_points:
            self.roi_points = [
                (px + actual_dx, py + actual_dy)
                for px, py in self._roi_move_origin_points
            ]
        self.schedule_render()
        if notify and self.roi_callback is not None:
            self.roi_callback(self.roi)
        return True

    def _resize_current_roi_from_origin(self, current_sx: float, current_sy: float, *, notify: bool = True) -> bool:
        """Resize the active ROI from its original mouse-down geometry."""
        if (
            self.frame is None
            or self._roi_resize_mode is None
            or self._roi_resize_origin is None
            or self._roi_resize_start_src is None
        ):
            return False
        mode = self._roi_resize_mode
        x, y, width, height = self._roi_resize_origin
        image_h, image_w = self.frame.shape
        start_sx, start_sy = self._roi_resize_start_src
        dx = int(round(current_sx - start_sx))
        dy = int(round(current_sy - start_sy))

        if mode == "annulus-inner":
            cx = x + width / 2.0
            cy = y + height / 2.0
            diameter = int(round(2.0 * math.hypot(current_sx - cx, current_sy - cy)))
            max_inner = max(0, min(width, height) - 1)
            diameter = min(max_inner, max(0 if max_inner == 0 else 1, diameter))
            self.roi_inner_diameter = diameter
        elif mode == "radial":
            cx = x + width / 2.0
            cy = y + height / 2.0
            size = max(1, int(round(2.0 * math.hypot(current_sx - cx, current_sy - cy))))
            max_size = max(1, int(math.floor(2.0 * min(cx, image_w - cx, cy, image_h - cy))))
            size = min(size, max_size)
            nx = int(round(cx - size / 2.0))
            ny = int(round(cy - size / 2.0))
            clamped = self._clamp_roi_bbox(nx, ny, size, size)
            if clamped is None:
                return False
            self.roi = clamped
            if self.roi_shape == "Annulus":
                outer = min(clamped[2], clamped[3])
                self.roi_inner_diameter = min(
                    max(0, outer - 1),
                    max(0 if outer <= 1 else 1, self._roi_resize_origin_inner_diameter),
                )
        elif mode == "scale":
            cx = x + width / 2.0
            cy = y + height / 2.0
            start_dist = max(0.5, math.hypot(start_sx - cx, start_sy - cy))
            current_dist = max(0.5, math.hypot(current_sx - cx, current_sy - cy))
            scale = current_dist / start_dist
            new_w = max(1, int(round(width * scale)))
            new_h = max(1, int(round(height * scale)))
            nx = int(round(cx - new_w / 2.0))
            ny = int(round(cy - new_h / 2.0))
            clamped = self._clamp_roi_bbox(nx, ny, new_w, new_h)
            if clamped is None:
                return False
            self.roi = clamped
            if self.roi_shape in ("Polygon", "Freehand") and self._roi_resize_origin_points:
                ox, oy, ow, oh = self._roi_resize_origin
                nx, ny, nw, nh = clamped
                old_dx = max(1, ow - 1)
                old_dy = max(1, oh - 1)
                new_dx = max(0, nw - 1)
                new_dy = max(0, nh - 1)
                self.roi_points = [
                    (
                        nx + int(round(((px - ox) / old_dx if ow > 1 else 0.0) * new_dx)),
                        ny + int(round(((py - oy) / old_dy if oh > 1 else 0.0) * new_dy)),
                    )
                    for px, py in self._roi_resize_origin_points
                ]
        else:
            left = x
            right = x + width
            top = y
            bottom = y + height
            if "left" in mode:
                left = min(right - 1, max(0, x + dx))
            if "right" in mode:
                right = max(left + 1, min(image_w, x + width + dx))
            if "top" in mode:
                top = min(bottom - 1, max(0, y + dy))
            if "bottom" in mode:
                bottom = max(top + 1, min(image_h, y + height + dy))
            new_bbox = (int(left), int(top), int(right - left), int(bottom - top))
            self.roi = new_bbox

        self.schedule_render()
        if notify and self.roi_callback is not None:
            self.roi_callback(self.roi)
        return True

    def _draw_roi_overlay(self) -> None:
        """Draw the current ROI shape as a thin green Viewer-only overlay."""
        self.delete("roi-overlay")
        if self.frame is None:
            return

        shape, bbox, points = self._current_roi_for_drawing()
        color = UI_SUCCESS

        if shape == "Polygon" and self._roi_polygon_points:
            points = list(self._roi_polygon_points)
            screen_points: list[float] = []
            for px, py in points:
                sx, sy = self._source_to_screen(px + 0.5, py + 0.5)
                screen_points.extend((sx, sy))
            if len(screen_points) >= 4:
                self.create_line(*screen_points, fill=color, width=1, tags=("roi-overlay",))
            if self._roi_polygon_preview is not None and points:
                sx0, sy0 = self._source_to_screen(points[-1][0] + 0.5, points[-1][1] + 0.5)
                sx1, sy1 = self._source_to_screen(self._roi_polygon_preview[0] + 0.5, self._roi_polygon_preview[1] + 0.5)
                self.create_line(sx0, sy0, sx1, sy1, fill=color, width=1, dash=(3, 3), tags=("roi-overlay",))
            radius = 2.5
            for px, py in points:
                sx, sy = self._source_to_screen(px + 0.5, py + 0.5)
                self.create_oval(sx - radius, sy - radius, sx + radius, sy + radius, outline=color, width=1, tags=("roi-overlay",))
            self.tag_raise("roi-overlay")
            return

        if shape == "Freehand" and self._roi_drag_points:
            points = list(self._roi_drag_points)
            screen_points: list[float] = []
            for px, py in points:
                sx, sy = self._source_to_screen(px + 0.5, py + 0.5)
                screen_points.extend((sx, sy))
            if len(screen_points) >= 4:
                self.create_line(*screen_points, fill=color, width=1, tags=("roi-overlay",))
            self.tag_raise("roi-overlay")
            return

        if bbox is None:
            return
        x, y, width, height = bbox
        x0, y0 = self._source_to_screen(float(x), float(y))
        x1, y1 = self._source_to_screen(float(x + width), float(y + height))

        if shape in ("Ellipse", "Circle", "Annulus"):
            self.create_oval(x0, y0, x1, y1, outline=color, width=1, tags=("roi-overlay",))
            if shape == "Annulus":
                outer_size = min(width, height)
                if self.roi_draw_mode and self._roi_drag_start is not None:
                    # During the initial mouse drag, preview the same 50% inner
                    # diameter that will be committed on button release.
                    inner_d = max(1, outer_size // 2) if outer_size > 1 else 0
                else:
                    inner_d = max(0, min(int(self.roi_inner_diameter), outer_size - 1))
                if inner_d > 0:
                    cx_src = x + width / 2.0
                    cy_src = y + height / 2.0
                    half = inner_d / 2.0
                    ix0, iy0 = self._source_to_screen(cx_src - half, cy_src - half)
                    ix1, iy1 = self._source_to_screen(cx_src + half, cy_src + half)
                    self.create_oval(ix0, iy0, ix1, iy1, outline=color, width=1, tags=("roi-overlay",))
        elif shape in ("Polygon", "Freehand") and points:
            screen_points: list[float] = []
            for px, py in points:
                sx, sy = self._source_to_screen(px + 0.5, py + 0.5)
                screen_points.extend((sx, sy))
            if len(screen_points) >= 6:
                screen_points.extend(screen_points[:2])
                self.create_line(*screen_points, fill=color, width=1, tags=("roi-overlay",))
        else:
            self.create_rectangle(x0, y0, x1, y1, outline=color, width=1, tags=("roi-overlay",))

        self.tag_raise("roi-overlay")

    def _finish_polygon_roi(self, *, notify: bool = True) -> bool:
        if self.frame is None or len(self._roi_polygon_points) < 3:
            return False
        points = list(self._roi_polygon_points)
        bbox = self._roi_bbox_from_points(points)
        if bbox is None:
            return False
        x, y, width, height = bbox
        return self.set_roi(x, y, width, height, notify=notify, shape="Polygon", points=points)

    def set_manual_beam_center(
        self,
        x: float,
        y: float,
        *,
        notify: bool = True,
    ) -> bool:
        """Anchor the temporary beam center to one detector pixel.

        The stored coordinate is the integer pixel index ``(x, y)``.  The
        marker itself is rendered at the visual centre of that pixel
        (``x + 0.5``, ``y + 0.5`` in source-image coordinates).  This is
        strictly Viewer-only state: no HDF5/HDXF metadata is modified and no
        source file is opened for writing.
        """
        if self.frame is None:
            return False
        image_h, image_w = self.frame.shape
        x = float(x)
        y = float(y)
        if not (math.isfinite(x) and math.isfinite(y)):
            return False
        if x < 0.0 or y < 0.0 or x >= image_w or y >= image_h:
            return False

        # A click anywhere inside a pixel selects that pixel.  Keep only the
        # integer detector-pixel indices; sub-pixel click position is discarded.
        pixel_x = int(math.floor(x))
        pixel_y = int(math.floor(y))
        self.manual_beam_center = (pixel_x, pixel_y)
        self.schedule_render()
        if notify and self.beam_center_callback is not None:
            self.beam_center_callback(self.manual_beam_center)
        return True

    def clear_manual_beam_center(self, *, notify: bool = True) -> None:
        """Clear the Viewer-only beam-center anchor without touching source data."""
        had_center = self.manual_beam_center is not None
        self.manual_beam_center = None
        self.schedule_render()
        if notify and had_center and self.beam_center_callback is not None:
            self.beam_center_callback(None)

    def get_manual_beam_center(self) -> tuple[int, int] | None:
        """Return the current Viewer-only beam center, if one is anchored."""
        return self.manual_beam_center

    def _source_to_screen(self, x: float, y: float) -> tuple[float, float]:
        canvas_w = max(1, self.winfo_width())
        canvas_h = max(1, self.winfo_height())
        screen_x = canvas_w / 2.0 + (float(x) - self.center_x) * self.zoom
        screen_y = canvas_h / 2.0 + (float(y) - self.center_y) * self.zoom
        return screen_x, screen_y

    def _draw_beam_center_overlay(self) -> None:
        """Draw the persistent temporary beam-center marker above the image."""
        self.delete("beam-center")
        if self.frame is None or self.manual_beam_center is None:
            return

        source_x, source_y = self.manual_beam_center
        # Integer coordinates identify a detector pixel; draw the marker at
        # the exact visual centre of that selected pixel.
        x, y = self._source_to_screen(source_x + 0.5, source_y + 0.5)
        canvas_w = max(1, self.winfo_width())
        canvas_h = max(1, self.winfo_height())
        if x < -30 or y < -30 or x > canvas_w + 30 or y > canvas_h + 30:
            return

        # Temporary beam-center marker (Viewer-only).
        # IMPORTANT: no center circle is drawn here.  Use four long, thin,
        # pure-red arms so the anchor remains obvious over detector pixels.
        arm = 64.0
        gap = 3.0
        segments = (
            (x - arm, y, x - gap, y),
            (x + gap, y, x + arm, y),
            (x, y - arm, x, y - gap),
            (x, y + gap, x, y + arm),
        )
        beam_color = "#FF0000"
        for x0, y0, x1, y1 in segments:
            self.create_line(
                x0, y0, x1, y1,
                fill=beam_color,
                width=1,
                capstyle=tk.BUTT,
                tags=("beam-center",),
            )

        label = f"TEMP BEAM  X {source_x}  Y {source_y}"
        font = self._font_cache.get(-11)
        if font is None:
            font = tkfont.Font(family="Segoe UI", size=9, weight="bold")
            self._font_cache[-11] = font
        label_x = x + 15.0
        label_y = y - 24.0
        label_w = font.measure(label) + 12
        label_h = font.metrics("linespace") + 8
        if label_x + label_w > canvas_w - 4:
            label_x = x - 15.0 - label_w
        if label_y < 4:
            label_y = y + 16.0
        label_x = min(max(4.0, label_x), max(4.0, canvas_w - label_w - 4.0))
        label_y = min(max(4.0, label_y), max(4.0, canvas_h - label_h - 4.0))
        self.create_rectangle(
            label_x, label_y, label_x + label_w, label_y + label_h,
            fill=UI_BG,
            outline=beam_color,
            width=1,
            tags=("beam-center",),
        )
        self.create_text(
            label_x + 6, label_y + 4,
            text=label,
            fill=beam_color,
            font=font,
            anchor="nw",
            tags=("beam-center",),
        )
        self.tag_raise("beam-center")

    def _draw_pointer_overlay(self) -> None:
        """Draw an always-visible cursor and an Albula-style pixel readout."""
        self.delete("mouse-cursor")
        self.delete("mouse-readout")
        if (
            not self._pointer_inside
            or self._pointer_xy is None
            or self._pan_start is not None
            or self.roi_draw_mode
            or self._roi_hover_move
            or self._roi_hover_resize_mode is not None
            or self._roi_move_start_src is not None
            or self._roi_resize_mode is not None
        ):
            return

        x, y = self._pointer_xy
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        if x < 0 or y < 0 or x >= width or y >= height:
            return

        arm = 10
        gap = 3
        segments = (
            (x - arm, y, x - gap, y),
            (x + gap, y, x + arm, y),
            (x, y - arm, x, y - gap),
            (x, y + gap, x, y + arm),
        )
        # Black outside stroke remains visible on white/light gray; the white
        # inner stroke remains visible on black/dark gray.
        for line_width, color in ((5, "#000000"), (2, "#ffffff")):
            for x0, y0, x1, y1 in segments:
                self.create_line(
                    x0, y0, x1, y1,
                    fill=color,
                    width=line_width,
                    capstyle=tk.ROUND,
                    tags=("mouse-cursor",),
                )
        self.create_oval(
            x - 2, y - 2, x + 2, y + 2,
            outline="#000000",
            fill="#ffffff",
            width=1,
            tags=("mouse-cursor",),
        )

        info = self._pixel_info(x, y)
        if info is not None and "x" in info:
            # Full precision is retained here.  Only the high-zoom text drawn
            # inside each pixel uses the compact two-decimal formatter.
            coord_text = f"X {info['x']}   Y {info['y']}"
            value_text = f"Value  {_format_value(info['value'])}"
            font_coord = self._font_cache.get(-9)
            if font_coord is None:
                font_coord = tkfont.Font(family="Segoe UI", size=9)
                self._font_cache[-9] = font_coord
            font_value = self._font_cache.get(-10)
            if font_value is None:
                font_value = tkfont.Font(family="Segoe UI", size=9, weight="bold")
                self._font_cache[-10] = font_value

            pad_x = 8
            pad_y = 6
            gap_y = 2
            coord_w = font_coord.measure(coord_text)
            value_w = font_value.measure(value_text)
            line_h = max(font_coord.metrics("linespace"), font_value.metrics("linespace"))
            box_w = max(coord_w, value_w) + 2 * pad_x
            box_h = line_h * 2 + gap_y + 2 * pad_y

            box_x = x + 18
            box_y = y + 18
            if box_x + box_w > width - 4:
                box_x = x - 18 - box_w
            if box_y + box_h > height - 4:
                box_y = y - 18 - box_h
            box_x = min(max(4.0, box_x), max(4.0, width - box_w - 4.0))
            box_y = min(max(4.0, box_y), max(4.0, height - box_h - 4.0))

            self.create_rectangle(
                box_x + 2,
                box_y + 2,
                box_x + box_w + 2,
                box_y + box_h + 2,
                fill="#000000",
                outline="",
                stipple="gray50",
                tags=("mouse-readout",),
            )
            self.create_rectangle(
                box_x,
                box_y,
                box_x + box_w,
                box_y + box_h,
                fill=UI_BG,
                outline=UI_ACCENT,
                width=1,
                tags=("mouse-readout",),
            )
            self.create_text(
                box_x + pad_x,
                box_y + pad_y,
                text=coord_text,
                fill=UI_TEXT_MUTED,
                font=font_coord,
                anchor="nw",
                tags=("mouse-readout",),
            )
            self.create_text(
                box_x + pad_x,
                box_y + pad_y + line_h + gap_y,
                text=value_text,
                fill=UI_WHITE,
                font=font_value,
                anchor="nw",
                tags=("mouse-readout",),
            )

        self.tag_raise("mouse-readout")
        self.tag_raise("mouse-cursor")

    def visible_source_bounds(self) -> tuple[int, int, int, int] | None:
        """Return the currently visible source-pixel rectangle."""
        if self.frame is None:
            return None
        canvas_w = max(1, self.winfo_width())
        canvas_h = max(1, self.winfo_height())
        image_h, image_w = self.frame.shape
        left = canvas_w / 2.0 - self.center_x * self.zoom
        top = canvas_h / 2.0 - self.center_y * self.zoom
        sx0 = max(0, int(math.floor((0.0 - left) / self.zoom)))
        sy0 = max(0, int(math.floor((0.0 - top) / self.zoom)))
        sx1 = min(image_w, int(math.ceil((canvas_w - left) / self.zoom)))
        sy1 = min(image_h, int(math.ceil((canvas_h - top) / self.zoom)))
        if sx1 <= sx0 or sy1 <= sy0:
            return None
        return sx0, sy0, sx1, sy1

    def _screen_to_source(self, x: float, y: float) -> tuple[float, float]:
        canvas_w = max(1, self.winfo_width())
        canvas_h = max(1, self.winfo_height())
        sx = self.center_x + (x - canvas_w / 2.0) / self.zoom
        sy = self.center_y + (y - canvas_h / 2.0) / self.zoom
        return sx, sy

    def _current_pointer_info(self) -> dict[str, Any] | None:
        x = self.winfo_pointerx() - self.winfo_rootx()
        y = self.winfo_pointery() - self.winfo_rooty()
        return self._pixel_info(x, y)

    def _pixel_info(self, x: float, y: float) -> dict[str, Any] | None:
        if self.frame is None:
            return None
        sx, sy = self._screen_to_source(x, y)
        ix = int(math.floor(sx))
        iy = int(math.floor(sy))
        image_h, image_w = self.frame.shape
        if ix < 0 or iy < 0 or ix >= image_w or iy >= image_h:
            return {"zoom": self.zoom}
        return {
            "x": ix,
            "y": iy,
            "value": self.frame[iy, ix],
            "zoom": self.zoom,
        }

    def _on_enter(self, event: tk.Event) -> None:
        self._pointer_inside = True
        self._pointer_xy = (float(event.x), float(event.y))
        self._update_roi_hover_state(event.x, event.y)
        if self._pan_start is None and self._roi_move_start_src is None and self._roi_resize_mode is None:
            self._refresh_hover_cursor()
            self._draw_pointer_overlay()

    def _on_leave(self, _event: tk.Event) -> None:
        self._pointer_inside = False
        self._pointer_xy = None
        self._roi_hover_move = False
        self._roi_hover_resize_mode = None
        self.delete("mouse-cursor")
        self.delete("mouse-readout")
        self._refresh_hover_cursor()
        self.status_callback(None)

    def _on_motion(self, event: tk.Event) -> None:
        self._pointer_inside = True
        self._pointer_xy = (float(event.x), float(event.y))
        if self.roi_draw_mode and self.roi_draw_shape == "Polygon" and self._roi_polygon_points:
            pixel = self._source_pixel_at_screen(event.x, event.y)
            self._roi_polygon_preview = pixel
            self.schedule_render()
        if self._pan_start is None and self._roi_move_start_src is None and self._roi_resize_mode is None:
            self._update_roi_hover_state(event.x, event.y)
            self._refresh_hover_cursor()
            self._draw_pointer_overlay()
            self.status_callback(self._pixel_info(event.x, event.y))

    def _on_resize(self, _event: tk.Event) -> None:
        self._clamp_center()
        self.schedule_render()

    def _on_wheel(self, event: tk.Event) -> None:
        factor = 1.25 if event.delta > 0 else 0.8
        self._zoom_at(event.x, event.y, factor)

    def _zoom_at(self, x: float, y: float, factor: float) -> None:
        if self.frame is None:
            return
        source_x, source_y = self._screen_to_source(x, y)
        new_zoom = max(0.01, min(80.0, self.zoom * factor))
        canvas_w = max(1, self.winfo_width())
        canvas_h = max(1, self.winfo_height())
        self.zoom = new_zoom
        self.center_x = source_x - (x - canvas_w / 2.0) / new_zoom
        self.center_y = source_y - (y - canvas_h / 2.0) / new_zoom
        self._clamp_center()
        self.schedule_render()
        self._notify_view_changed()

    def _pan_begin(self, event: tk.Event) -> None:
        if self.frame is None:
            return
        button = int(getattr(event, "num", 0) or 0)

        if button == 1 and not self.roi_draw_mode and self.roi is not None:
            resize_mode = self._roi_resize_hit_test(event.x, event.y)
            if resize_mode is not None:
                sx, sy = self._screen_to_source(event.x, event.y)
                self._roi_hover_resize_mode = resize_mode
                self._roi_hover_move = False
                self._roi_resize_mode = resize_mode
                self._roi_resize_start_src = (sx, sy)
                self._roi_resize_start_screen = (float(event.x), float(event.y))
                self._roi_resize_origin = self.roi
                self._roi_resize_origin_points = list(self.roi_points)
                self._roi_resize_origin_inner_diameter = int(self.roi_inner_diameter)
                self._roi_resize_dragged = False
                self._pan_start = None
                self._pan_button = None
                self._pan_dragged = False
                self.delete("mouse-cursor")
                self.delete("mouse-readout")
                self._refresh_hover_cursor()
                return
            if self._roi_fill_hit_test(event.x, event.y):
                sx, sy = self._screen_to_source(event.x, event.y)
                self._roi_hover_move = True
                self._roi_hover_resize_mode = None
                self._roi_move_start_src = (sx, sy)
                self._roi_move_start_screen = (float(event.x), float(event.y))
                self._roi_move_origin = self.roi
                self._roi_move_origin_points = list(self.roi_points)
                self._roi_move_dragged = False
                self._pan_start = None
                self._pan_button = None
                self._pan_dragged = False
                self.delete("mouse-cursor")
                self.delete("mouse-readout")
                self._refresh_hover_cursor()
                return

        # ROI drawing deliberately reserves only the RIGHT mouse button. The
        # left mouse button keeps its existing click-to-anchor / drag-to-pan
        # behaviour even while ROI draw mode is armed.
        if self.roi_draw_mode and button == 3:
            pixel = self._source_pixel_at_screen(event.x, event.y)
            if pixel is None:
                return
            shape = self.roi_draw_shape
            self._pan_start = None
            self._pan_button = None
            self._pan_dragged = False
            self._refresh_hover_cursor()

            if shape == "Polygon":
                # Repeated right clicks create vertices. Clicking the first
                # vertex again (within a small screen-space tolerance) closes
                # the polygon after at least three vertices.
                if self._roi_polygon_points and len(self._roi_polygon_points) >= 3:
                    first = self._roi_polygon_points[0]
                    fx, fy = self._source_to_screen(first[0] + 0.5, first[1] + 0.5)
                    if math.hypot(float(event.x) - fx, float(event.y) - fy) <= 9.0:
                        self._finish_polygon_roi(notify=True)
                        return
                if not self._roi_polygon_points or self._roi_polygon_points[-1] != pixel:
                    self._roi_polygon_points.append(pixel)
                self._roi_polygon_preview = pixel
                self.schedule_render()
                return

            if shape == "Freehand":
                self._roi_drag_points = [pixel]
                self._roi_drag_start = pixel
                self._roi_drag_current = pixel
                self.schedule_render()
                return

            self._roi_drag_start = pixel
            self._roi_drag_current = pixel
            self.schedule_render()
            return

        self._pan_start = (
            event.x,
            event.y,
            self.center_x,
            self.center_y,
        )
        self._pan_button = button
        self._pan_dragged = False

    def _pan_move(self, event: tk.Event) -> None:
        if self.roi_draw_mode and self.roi_draw_shape == "Polygon" and self._roi_polygon_points:
            pixel = self._source_pixel_at_screen(event.x, event.y)
            if pixel is not None:
                self._roi_polygon_preview = pixel
                self.schedule_render()
            return

        if self._roi_resize_mode is not None and self._roi_resize_origin is not None:
            start_screen = self._roi_resize_start_screen or (float(event.x), float(event.y))
            screen_dx = float(event.x) - float(start_screen[0])
            screen_dy = float(event.y) - float(start_screen[1])
            if not self._roi_resize_dragged and math.hypot(screen_dx, screen_dy) < self._pan_threshold_px:
                return
            self._roi_resize_dragged = True
            sx, sy = self._screen_to_source(event.x, event.y)
            self._resize_current_roi_from_origin(sx, sy, notify=True)
            self._refresh_hover_cursor()
            return

        if self._roi_move_start_src is not None and self._roi_move_origin is not None:
            start_screen = self._roi_move_start_screen or (float(event.x), float(event.y))
            screen_dx = float(event.x) - float(start_screen[0])
            screen_dy = float(event.y) - float(start_screen[1])
            if not self._roi_move_dragged and math.hypot(screen_dx, screen_dy) < self._pan_threshold_px:
                return
            self._roi_move_dragged = True
            sx, sy = self._screen_to_source(event.x, event.y)
            start_sx, start_sy = self._roi_move_start_src
            delta_x = int(round(sx - start_sx))
            delta_y = int(round(sy - start_sy))
            self._translate_current_roi_from_origin(delta_x, delta_y, notify=True)
            self._refresh_hover_cursor()
            return

        if self._roi_drag_start is not None:
            if self.frame is None:
                return
            sx, sy = self._screen_to_source(event.x, event.y)
            image_h, image_w = self.frame.shape
            px = min(image_w - 1, max(0, int(math.floor(sx))))
            py = min(image_h - 1, max(0, int(math.floor(sy))))
            self._roi_drag_current = (px, py)
            if self.roi_draw_shape == "Freehand":
                point = (px, py)
                if not self._roi_drag_points or self._roi_drag_points[-1] != point:
                    self._roi_drag_points.append(point)
            self.schedule_render()
            return

        if self._pan_start is None:
            return
        start_x, start_y, center_x, center_y = self._pan_start
        dx = float(event.x - start_x)
        dy = float(event.y - start_y)

        if not self._pan_dragged:
            if math.hypot(dx, dy) < self._pan_threshold_px:
                return
            self._pan_dragged = True
            self.delete("mouse-cursor")
            self.delete("mouse-readout")
            self._refresh_hover_cursor()

        self.center_x = center_x - dx / self.zoom
        self.center_y = center_y - dy / self.zoom
        self._clamp_center()
        self.schedule_render()
        self._notify_view_changed()

    def _pan_end(self, event: tk.Event) -> None:
        if self._roi_resize_mode is not None:
            self._pointer_inside = True
            self._pointer_xy = (float(event.x), float(event.y))
            self._roi_resize_mode = None
            self._roi_resize_start_src = None
            self._roi_resize_start_screen = None
            self._roi_resize_origin = None
            self._roi_resize_origin_points = []
            self._roi_resize_origin_inner_diameter = 0
            self._roi_resize_dragged = False
            self._roi_hover_resize_mode = self._roi_resize_hit_test(event.x, event.y)
            self._roi_hover_move = (
                self._roi_hover_resize_mode is None
                and self._roi_fill_hit_test(event.x, event.y)
            )
            self._refresh_hover_cursor()
            self._draw_pointer_overlay()
            self.status_callback(self._pixel_info(event.x, event.y))
            return

        if self._roi_move_start_src is not None:
            self._pointer_inside = True
            self._pointer_xy = (float(event.x), float(event.y))
            self._roi_move_start_src = None
            self._roi_move_start_screen = None
            self._roi_move_origin = None
            self._roi_move_origin_points = []
            self._roi_move_dragged = False
            self._roi_hover_resize_mode = self._roi_resize_hit_test(event.x, event.y)
            self._roi_hover_move = (
                self._roi_hover_resize_mode is None
                and self._roi_fill_hit_test(event.x, event.y)
            )
            self._refresh_hover_cursor()
            self._draw_pointer_overlay()
            self.status_callback(self._pixel_info(event.x, event.y))
            return

        # Polygon uses repeated right clicks; ButtonRelease must not fall
        # through to the ordinary right-click beam-centre clear action.
        if self.roi_draw_mode and self.roi_draw_shape == "Polygon" and int(getattr(event, "num", 0) or 0) == 3:
            self._pointer_inside = True
            self._pointer_xy = (float(event.x), float(event.y))
            self._refresh_hover_cursor()
            self.schedule_render()
            return

        if self._roi_drag_start is not None:
            start = self._roi_drag_start
            current = self._roi_drag_current or start
            if self.frame is not None:
                sx, sy = self._screen_to_source(event.x, event.y)
                image_h, image_w = self.frame.shape
                current = (
                    min(image_w - 1, max(0, int(math.floor(sx)))),
                    min(image_h - 1, max(0, int(math.floor(sy)))),
                )
            shape = self.roi_draw_shape
            points = list(self._roi_drag_points)
            self._roi_drag_start = None
            self._roi_drag_current = None
            self._roi_drag_points = []

            if shape == "Freehand":
                if points and points[-1] != current:
                    points.append(current)
                # Remove consecutive duplicates and require a real area.
                cleaned: list[tuple[int, int]] = []
                for point in points:
                    if not cleaned or cleaned[-1] != point:
                        cleaned.append(point)
                if len(cleaned) >= 3:
                    bbox = self._roi_bbox_from_points(cleaned)
                    if bbox is not None:
                        x, y, width, height = bbox
                        self.set_roi(x, y, width, height, notify=True, shape="Freehand", points=cleaned)
                else:
                    self.set_roi_draw_mode(False)
            else:
                left, top, width, height = self._roi_bbox_from_drag(start, current, shape)
                if shape == "Annulus":
                    outer = max(1, min(width, height))
                    default_inner = max(1, outer // 2) if outer > 1 else 0
                    self.set_roi(
                        left, top, width, height, notify=True, shape=shape,
                        inner_diameter=default_inner,
                    )
                else:
                    self.set_roi(left, top, width, height, notify=True, shape=shape)

            self._pointer_inside = True
            self._pointer_xy = (float(event.x), float(event.y))
            self._roi_hover_resize_mode = self._roi_resize_hit_test(event.x, event.y)
            self._roi_hover_move = (
                self._roi_hover_resize_mode is None
                and self._roi_fill_hit_test(event.x, event.y)
            )
            self._refresh_hover_cursor()
            return

        if self._pan_start is None:
            return

        button = self._pan_button
        dragged = self._pan_dragged
        self._pan_start = None
        self._pan_button = None
        self._pan_dragged = False
        self._pointer_inside = True
        self._pointer_xy = (float(event.x), float(event.y))
        self._roi_hover_resize_mode = self._roi_resize_hit_test(event.x, event.y)
        self._roi_hover_move = (
            self._roi_hover_resize_mode is None
            and self._roi_fill_hit_test(event.x, event.y)
        )
        self._refresh_hover_cursor()

        if not dragged:
            if button == 1:
                source_x, source_y = self._screen_to_source(event.x, event.y)
                self.set_manual_beam_center(source_x, source_y, notify=True)
            elif button == 3:
                self.clear_manual_beam_center(notify=True)
        else:
            self._notify_view_changed()

        self.schedule_render()

    def _clamp_center(self) -> None:
        if self.frame is None:
            return
        canvas_w = max(1, self.winfo_width())
        canvas_h = max(1, self.winfo_height())
        image_h, image_w = self.frame.shape

        visible_w = canvas_w / self.zoom
        visible_h = canvas_h / self.zoom

        if visible_w >= image_w:
            self.center_x = image_w / 2.0
        else:
            half = visible_w / 2.0
            self.center_x = min(image_w - half, max(half, self.center_x))

        if visible_h >= image_h:
            self.center_y = image_h / 2.0
        else:
            half = visible_h / 2.0
            self.center_y = min(image_h - half, max(half, self.center_y))


@dataclass
class InternalViewState:
    """One independently opened detector image inside the main UI."""

    view_id: int
    host: tk.Frame
    title_var: tk.StringVar
    canvas: ImageCanvas
    title_bar: tk.Frame | None = None
    archive: HDXFArchive | HDF5Archive | None = None
    current_frame_index: int = 0
    current_frame: np.ndarray | None = None
    current_display: np.ndarray | None = None
    display_min: float = 0.0
    display_max: float = 1.0
    contrast_mode: str = AUTO_CONTRAST_VISIBLE_SPARSE
    invert: bool = False
    show_values: bool = True

    def label(self) -> str:
        filename = self.archive.path.name if self.archive is not None else "No file"
        return f"View {self.view_id} — {filename}"


class ViewerWorkspaceManager:
    """Coordinate up to four image panes inside one application window."""

    MAX_VIEWS = 4

    def __init__(self, root: tk.Tk, initial_path: Path | None = None) -> None:
        self.root = root
        self.sync_pan_zoom_var = tk.BooleanVar(master=root, value=True)
        self.sync_frames_var = tk.BooleanVar(master=root, value=False)
        self._view_after_id: str | None = None
        self._pending_view: tuple[InternalViewState, dict[str, float]] | None = None
        self._frame_after_id: str | None = None
        self._pending_frame: tuple[InternalViewState, int] | None = None
        self.app = HDXFViewerApp(root, self, initial_path)

    def create_window(self, *, initial_path: Path | None = None, use_root: bool = False) -> InternalViewState | None:
        # Kept under the old method name for compatibility with toolbar/menu
        # callbacks. No operating-system window is created.
        del use_root
        return self.app.add_view(initial_path=initial_path)

    def close_window(self, app: "HDXFViewerApp") -> None:
        app._shutdown_resources()
        self.root.destroy()

    def request_view_sync(self, source: InternalViewState, state: dict[str, float]) -> None:
        if not self.sync_pan_zoom_var.get() or len(self.app.views) < 2:
            return
        self._pending_view = (source, dict(state))
        if self._view_after_id is None:
            self._view_after_id = self.root.after(12, self._flush_view_sync)

    def _flush_view_sync(self) -> None:
        self._view_after_id = None
        pending = self._pending_view
        self._pending_view = None
        if pending is None or not self.sync_pan_zoom_var.get():
            return
        source, state = pending
        for view in list(self.app.views):
            if view is source or view.current_frame is None:
                continue
            try:
                view.canvas.apply_synced_view(state)
            except tk.TclError:
                pass

    def request_frame_sync(self, source: InternalViewState, index: int) -> None:
        if not self.sync_frames_var.get() or len(self.app.views) < 2:
            return
        self._pending_frame = (source, int(index))
        if self._frame_after_id is None:
            self._frame_after_id = self.root.after_idle(self._flush_frame_sync)

    def _flush_frame_sync(self) -> None:
        self._frame_after_id = None
        pending = self._pending_frame
        self._pending_frame = None
        if pending is None or not self.sync_frames_var.get():
            return
        source, index = pending
        active = self.app.active_view
        for view in list(self.app.views):
            if view is source or view.archive is None:
                continue
            target = min(max(0, index), view.archive.frame_count - 1)
            self.app.load_frame_for_view(
                view,
                target,
                recompute_contrast=False,
                playback=False,
            )
        if active in self.app.views:
            self.app.activate_view(active, refresh=True)

    def align_to(self, source: InternalViewState | None = None) -> None:
        source = source or self.app.active_view
        if source is None:
            return
        state = source.canvas._view_state()
        if state is None:
            return
        for view in list(self.app.views):
            if view is source or view.current_frame is None:
                continue
            view.canvas.apply_synced_view(state)

    def tile_windows(self) -> None:
        self.app.relayout_views()

    def cascade_windows(self) -> None:
        # Internal panes always use the automatic 1 / 2 / 2x2 arrangement.
        self.app.relayout_views()


class HDXFViewerApp:
    """Albula-inspired HDXF desktop viewer with internal multi-view panes."""

    @property
    def active_view(self) -> InternalViewState | None:
        if not getattr(self, "views", None):
            return None
        index = min(max(0, self.active_view_index), len(self.views) - 1)
        return self.views[index]

    @property
    def canvas(self) -> ImageCanvas:
        view = self.active_view
        if view is None:
            raise HDXFViewerError("no active image view")
        return view.canvas

    @property
    def archive(self) -> HDXFArchive | HDF5Archive | None:
        view = self.active_view
        return None if view is None else view.archive

    @archive.setter
    def archive(self, value: HDXFArchive | HDF5Archive | None) -> None:
        view = self.active_view
        if view is not None:
            view.archive = value

    @property
    def current_frame_index(self) -> int:
        view = self.active_view
        return 0 if view is None else view.current_frame_index

    @current_frame_index.setter
    def current_frame_index(self, value: int) -> None:
        view = self.active_view
        if view is not None:
            view.current_frame_index = int(value)

    @property
    def current_frame(self) -> np.ndarray | None:
        view = self.active_view
        return None if view is None else view.current_frame

    @current_frame.setter
    def current_frame(self, value: np.ndarray | None) -> None:
        view = self.active_view
        if view is not None:
            view.current_frame = value

    @property
    def current_display(self) -> np.ndarray | None:
        view = self.active_view
        return None if view is None else view.current_display

    @current_display.setter
    def current_display(self, value: np.ndarray | None) -> None:
        view = self.active_view
        if view is not None:
            view.current_display = value

    @property
    def display_min(self) -> float:
        view = self.active_view
        return 0.0 if view is None else view.display_min

    @display_min.setter
    def display_min(self, value: float) -> None:
        view = self.active_view
        if view is not None:
            view.display_min = float(value)

    @property
    def display_max(self) -> float:
        view = self.active_view
        return 1.0 if view is None else view.display_max

    @display_max.setter
    def display_max(self, value: float) -> None:
        view = self.active_view
        if view is not None:
            view.display_max = float(value)

    def _render_view_display(self, view: InternalViewState) -> None:
        """Rebuild one pane from its stored global display settings."""
        frame = view.current_frame
        if frame is None:
            return
        low = float(view.display_min)
        high = float(view.display_max)
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            low, high = 0.0, 1.0
            view.display_min = low
            view.display_max = high
        scale = 255.0 / (high - low)
        work = np.asarray(frame, dtype=np.float32)
        display = np.clip((work - low) * scale, 0.0, 255.0).astype(np.uint8)
        if view.invert:
            display = 255 - display
        view.current_display = display
        view.canvas.set_frame(frame, display)
        view.canvas.set_show_values(view.show_values)

    def _apply_display_settings_to_all(
        self,
        *,
        low: float | None = None,
        high: float | None = None,
        mode: str | None = None,
        invert: bool | None = None,
        show_values: bool | None = None,
        rebuild: bool = True,
    ) -> None:
        """Broadcast right-side display controls to every internal view."""
        for view in self.views:
            if low is not None:
                view.display_min = float(low)
            if high is not None:
                view.display_max = float(high)
            if mode is not None:
                view.contrast_mode = str(mode)
            if invert is not None:
                view.invert = bool(invert)
            if show_values is not None:
                view.show_values = bool(show_values)
                view.canvas.set_show_values(view.show_values)
            if rebuild and view.current_frame is not None:
                self._render_view_display(view)

    def __init__(self, root: tk.Misc, manager: ViewerWorkspaceManager, initial_path: Path | None = None) -> None:
        self.root = root
        self.manager = manager
        self.root.title("HDXF Viewer 0.5.2.0 · HDF5 · ROI DEEPER BLUE SHADE · MOVE + RESIZE")
        ui_scale = float(getattr(self.root, "_hdxf_ui_scale", 1.0))
        screen_w = max(1, int(self.root.winfo_screenwidth()))
        screen_h = max(1, int(self.root.winfo_screenheight()))
        target_w = min(int(round(1580 * ui_scale)), max(900, screen_w - int(round(48 * ui_scale))))
        target_h = min(int(round(960 * ui_scale)), max(620, screen_h - int(round(80 * ui_scale))))
        self.root.geometry(f"{target_w}x{target_h}")
        self.root.minsize(
            min(int(round(1040 * ui_scale)), target_w),
            min(int(round(660 * ui_scale)), target_h),
        )

        self.views: list[InternalViewState] = []
        self.active_view_index = -1
        self._next_view_id = 1

        self.playing = False
        self._play_after_id: str | None = None
        self._play_generation = 0
        self._playback_rendering = False

        self.frame_var = tk.StringVar(value="1")
        self.frame_label_var = tk.StringVar(value="Image 0 / 0")
        self.fps_var = tk.StringVar(value="5")
        self.loop_var = tk.BooleanVar(value=True)
        self.min_var = tk.StringVar(value="0")
        self.max_var = tk.StringVar(value="1")
        self.contrast_mode_var = tk.StringVar(value=AUTO_CONTRAST_VISIBLE_SPARSE)
        self.invert_var = tk.BooleanVar(value=False)
        self.show_values_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")
        self.pixel_summary_var = tk.StringVar(value="Pixel: —")
        self.file_summary_var = tk.StringVar(value="No file open")
        self.active_view_var = tk.StringVar(value="VIEW 01 · NO FILE")
        self.view_count_var = tk.StringVar(value="1 / 4 VIEWS")

        # Shared right-sidebar controls edit the ROI of the currently active
        # internal view. ROI itself remains per-view and Viewer-only.
        self.roi_shape_var = tk.StringVar(value="Rectangle")
        self.roi_x_var = tk.StringVar(value="0")
        self.roi_y_var = tk.StringVar(value="0")
        self.roi_width_var = tk.StringVar(value="1")
        self.roi_height_var = tk.StringVar(value="1")
        self.roi_inner_diameter_var = tk.StringVar(value="—")
        self.roi_x2_var = tk.StringVar(value="—")
        self.roi_y2_var = tk.StringVar(value="—")
        self.roi_pixels_var = tk.StringVar(value="—")
        self.roi_min_var = tk.StringVar(value="—")
        self.roi_max_var = tk.StringVar(value="—")
        self.roi_mean_var = tk.StringVar(value="—")
        self.roi_sum_var = tk.StringVar(value="—")
        self.roi_std_var = tk.StringVar(value="—")
        self.roi_zeros_var = tk.StringVar(value="—")
        self.roi_draw_text_var = tk.StringVar(value="Draw ROI (Right)")

        self._thumbnail_photo: ImageTk.PhotoImage | None = None
        self._metadata_item_count = 0
        self._histogram_domain: tuple[float, float] | None = None
        self._histogram_axis: HistogramIntensityAxis | None = None
        self._histogram_plot_rect: tuple[float, float, float, float] | None = None
        self._histogram_handle_positions: dict[str, float] = {}
        self._histogram_drag_mode: str | None = None
        self._histogram_drag_start_x = 0.0
        self._histogram_drag_start_range: tuple[float, float] | None = None
        self._histogram_hover_x: float | None = None
        self._histogram_pending_range: tuple[float, float] | None = None
        self._histogram_render_after_id: str | None = None
        self._histogram_values_cache: np.ndarray | None = None
        self._histogram_cache_key: tuple[Any, ...] | None = None
        self._histogram_logs_cache: np.ndarray | None = None

        # Native Windows Explorer file drop support.  It is installed after
        # the first internal view exists so a drop can be routed immediately.
        self._file_drop_controller = SafeFileDropController(
            self.root, self._handle_dropped_files
        )

        # External converter tool state.  The converter remains a separate
        # process and source file; this viewer only provides a small launcher
        # and progress-oriented launcher around hdf5_to_hdxf.py in the same directory.
        self._converter_panel: tk.Frame | None = None
        self._converter_process: subprocess.Popen[str] | None = None
        self._converter_reader: threading.Thread | None = None
        self._converter_queue: queue.Queue[tuple[int, str, Any]] = queue.Queue()
        self._converter_run_serial = 0
        self._converter_poll_after_id: str | None = None
        self._converter_output_path: Path | None = None

        # Reverse conversion launcher.  Like the forward converter, it only
        # starts a sibling script as a subprocess; HDF5 restoration logic stays
        # outside the Viewer.
        self._restore_panel: tk.Frame | None = None
        self._restore_process: subprocess.Popen[str] | None = None
        self._restore_reader: threading.Thread | None = None
        self._restore_queue: queue.Queue[tuple[int, str, Any]] = queue.Queue()
        self._restore_run_serial = 0
        self._restore_poll_after_id: str | None = None
        self._restore_log_file = None
        self._restore_start_time: float | None = None
        self._restore_last_progress = 0.0

        # HDXF processing launcher shared by hdxf_sum.py, hdxf_subtract.py and hdxf_pixelop.py.
        # The processing algorithms stay in sibling scripts; Viewer only builds
        # validated command lines and presents progress/log output.
        self._processing_panel: tk.Frame | None = None
        self._processing_process: subprocess.Popen[str] | None = None
        self._processing_reader: threading.Thread | None = None
        self._processing_queue: queue.Queue[tuple[int, str, Any]] = queue.Queue()
        self._processing_run_serial = 0
        self._processing_poll_after_id: str | None = None
        self._processing_kind: str | None = None
        self._processing_output_path: Path | None = None
        self._processing_start_time: float | None = None
        self._processing_stop_requested = False
        self._processing_last_progress = 0.0
        self._processing_output_auto = True

        self._build_ui()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        first_view = self.add_view()
        self.root.after_idle(self._install_file_drop_support)

        if initial_path is not None and first_view is not None:
            self.root.after(100, lambda: self.open_path(initial_path))

    def _make_card(
        self,
        parent: tk.Misc,
        title: str,
        subtitle: str | None = None,
        *,
        pady: tuple[int, int] = (0, 10),
    ) -> ttk.Frame:
        """Create a modern card and return its content frame."""
        card = ttk.Frame(parent, style="Card.TFrame", padding=(12, 10))
        card.pack(fill="x", pady=pady)
        heading = ttk.Frame(card, style="Card.TFrame")
        heading.pack(fill="x", pady=(0, 8 if subtitle else 6))
        ttk.Label(heading, text=title, style="CardTitle.TLabel").pack(side="left", anchor="w")
        if subtitle:
            ttk.Label(
                card,
                text=subtitle,
                style="CardSubtitle.TLabel",
                wraplength=330,
                justify="left",
            ).pack(fill="x", anchor="w", pady=(0, 8))
        body = ttk.Frame(card, style="Card.TFrame")
        body.pack(fill="x")
        return body

    def _build_ui(self) -> None:
        self.root.configure(background=UI_BG)
        self._build_menu()

        app_shell = ttk.Frame(self.root, style="App.TFrame")
        app_shell.pack(fill="both", expand=True)

        topbar = ttk.Frame(app_shell, style="Topbar.TFrame", padding=(14, 9))
        topbar.pack(side="top", fill="x")

        brand = ttk.Frame(topbar, style="Topbar.TFrame")
        brand.pack(side="left", padx=(0, 16), fill="y")
        tk.Frame(brand, background=UI_ACCENT, width=4, height=48).pack(side="left", fill="y", padx=(0, 10))
        brand_text = ttk.Frame(brand, style="Topbar.TFrame")
        brand_text.pack(side="left", anchor="center")
        ttk.Label(brand_text, text="HDXF VIEWER", style="Brand.TLabel").pack(anchor="w")
        ttk.Label(brand_text, text="DETECTOR WORKSPACE  ·  0.5.2.0 · HDF5 · ROI DEEPER BLUE SHADE · MOVE + RESIZE", style="BrandSub.TLabel").pack(anchor="w")

        def toolbar_group(title: str) -> ttk.Frame:
            outer = ttk.Frame(topbar, style="ToolbarGroup.TFrame", padding=(9, 5))
            outer.pack(side="left", padx=(0, 7), fill="y")
            ttk.Label(outer, text=title, style="ToolbarGroupTitle.TLabel").pack(anchor="w", pady=(0, 3))
            row = ttk.Frame(outer, style="ToolbarGroup.TFrame")
            row.pack(anchor="w")
            return row

        workspace = toolbar_group("WORKSPACE")
        ttk.Button(workspace, text="＋ New View", style="Accent.TButton", command=self.new_window).pack(side="left")
        ttk.Button(workspace, text="Open", style="Toolbar.TButton", command=self.open_dialog).pack(side="left", padx=(5, 0))

        processing_tools = toolbar_group("HDXF PROCESS")
        ttk.Button(
            processing_tools, text="Σ Sum / Mean", style="Toolbar.TButton",
            command=self.open_hdxf_sum_tool,
        ).pack(side="left")
        ttk.Button(
            processing_tools, text="− Subtract", style="Toolbar.TButton",
            command=self.open_hdxf_subtract_tool,
        ).pack(side="left", padx=(5, 0))
        ttk.Button(
            processing_tools, text="± Pixel Op", style="Toolbar.TButton",
            command=self.open_hdxf_pixelop_tool,
        ).pack(side="left", padx=(5, 0))

        viewport = toolbar_group("VIEWPORT")
        ttk.Button(viewport, text="Fit", style="Toolbar.TButton", width=5, command=self._fit).pack(side="left")
        ttk.Button(viewport, text="1:1", style="Toolbar.TButton", width=4, command=self._one_to_one).pack(side="left", padx=(4, 0))
        ttk.Button(viewport, text="−", style="Icon.TButton", width=3, command=self._zoom_out).pack(side="left", padx=(7, 0))
        ttk.Button(viewport, text="+", style="Icon.TButton", width=3, command=self._zoom_in).pack(side="left", padx=(3, 0))

        playback = toolbar_group("PLAYBACK")
        ttk.Button(playback, text="|◀", style="Icon.TButton", width=3, command=lambda: self.load_frame(0)).pack(side="left")
        ttk.Button(playback, text="◀", style="Icon.TButton", width=3, command=lambda: self.step_frame(-1)).pack(side="left", padx=(3, 0))
        self.play_button = ttk.Button(playback, text="▶  PLAY", style="Play.TButton", width=9, command=self.toggle_playback)
        self.play_button.pack(side="left", padx=5)
        ttk.Button(playback, text="▶", style="Icon.TButton", width=3, command=lambda: self.step_frame(1)).pack(side="left")
        ttk.Button(
            playback,
            text="▶|",
            style="Icon.TButton",
            width=3,
            command=lambda: self.load_frame(self.archive.frame_count - 1 if self.archive else 0),
        ).pack(side="left", padx=(3, 0))

        navigation = toolbar_group("IMAGE")
        self.frame_spin = ttk.Spinbox(
            navigation,
            from_=1,
            to=1,
            width=8,
            textvariable=self.frame_var,
            command=self._load_frame_from_entry,
            style="Modern.TSpinbox",
        )
        self.frame_spin.pack(side="left")
        self.frame_spin.bind("<Return>", self._load_frame_from_entry)
        self.frame_spin.bind("<FocusOut>", self._load_frame_from_entry)
        ttk.Label(navigation, textvariable=self.frame_label_var, style="ToolbarValue.TLabel").pack(side="left", padx=(7, 10))
        ttk.Label(navigation, text="FPS", style="ToolbarMuted.TLabel").pack(side="left")
        self.fps_combo = ttk.Combobox(
            navigation,
            width=4,
            state="readonly",
            textvariable=self.fps_var,
            values=("1", "2", "5", "10", "15", "20"),
            style="Modern.TCombobox",
        )
        self.fps_combo.pack(side="left", padx=(4, 7))
        ttk.Checkbutton(navigation, text="Loop", variable=self.loop_var, style="Toolbar.TCheckbutton").pack(side="left")

        ttk.Label(topbar, textvariable=self.view_count_var, style="TopbarChip.TLabel").pack(side="right", padx=(8, 0), anchor="center")

        statusbar = ttk.Frame(app_shell, style="Statusbar.TFrame", padding=(12, 5))
        statusbar.pack(side="bottom", fill="x")
        tk.Label(statusbar, text="●", background=UI_SURFACE, foreground=UI_SUCCESS, font=("Segoe UI", 9)).pack(side="left", padx=(0, 7))
        ttk.Label(statusbar, textvariable=self.status_var, style="Status.TLabel", anchor="w").pack(side="left", fill="x", expand=True)
        ttk.Label(statusbar, text="PAN / ZOOM SYNC", style="StatusChip.TLabel").pack(side="right")

        main = ttk.Panedwindow(app_shell, orient="horizontal", style="Modern.TPanedwindow")
        main.pack(side="top", fill="both", expand=True)

        image_host = ttk.Frame(main, style="ImageHost.TFrame", padding=(8, 8))
        side_host = ttk.Frame(main, width=390, style="Side.TFrame")
        main.add(image_host, weight=1)
        main.add(side_host, weight=0)

        self.view_grid = ttk.Frame(image_host, style="ImageHost.TFrame")
        self.view_grid.pack(fill="both", expand=True)

        self._build_sidebar(side_host)

    def _build_menu(self) -> None:
        menu_options = {
            "tearoff": False,
            "background": UI_SURFACE,
            "foreground": UI_TEXT,
            "activebackground": UI_ACCENT_STRONG,
            "activeforeground": UI_WHITE,
            "selectcolor": UI_ACCENT,
            "borderwidth": 0,
            "relief": "flat",
            "font": ("Segoe UI", 9),
        }
        menu = tk.Menu(
            self.root,
            background=UI_TOPBAR,
            foreground=UI_TEXT,
            activebackground=UI_ACCENT_STRONG,
            activeforeground=UI_WHITE,
            borderwidth=0,
            relief="flat",
        )
        file_menu = tk.Menu(menu, **menu_options)
        file_menu.add_command(label="New View", command=self.new_window, accelerator="Ctrl+N")
        file_menu.add_command(label="Open…", command=self.open_dialog, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Close View", command=self.close_active_view, accelerator="Ctrl+W")
        file_menu.add_command(label="Exit", command=self.close)
        menu.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menu, **menu_options)
        view_menu.add_command(label="Fit to window", command=self._fit, accelerator="F")
        view_menu.add_command(label="Actual size", command=self._one_to_one)
        view_menu.add_separator()
        view_menu.add_command(label="Auto contrast", command=self.auto_contrast)
        view_menu.add_command(label="Full range", command=self.full_contrast)
        menu.add_cascade(label="View", menu=view_menu)

        playback_menu = tk.Menu(menu, **menu_options)
        playback_menu.add_command(label="Play / Pause", command=self.toggle_playback, accelerator="Space")
        playback_menu.add_command(label="Previous frame", command=lambda: self.step_frame(-1))
        playback_menu.add_command(label="Next frame", command=lambda: self.step_frame(1))
        menu.add_cascade(label="Playback", menu=playback_menu)

        sync_menu = tk.Menu(menu, **menu_options)
        sync_menu.add_checkbutton(label="Synchronize Pan and Zoom", variable=self.manager.sync_pan_zoom_var)
        sync_menu.add_checkbutton(label="Synchronize Image Number", variable=self.manager.sync_frames_var)
        sync_menu.add_separator()
        sync_menu.add_command(label="Align Other Views to Current View", command=lambda: self.manager.align_to(self.active_view))
        menu.add_cascade(label="Synchronize", menu=sync_menu)

        tools_menu = tk.Menu(menu, **menu_options)
        # Keep the two HDXF processing tools directly visible in Tools.
        # This avoids burying them in a submenu and makes it immediately
        # obvious whether the running Viewer contains the processing build.
        tools_menu.add_command(
            label="HDXF Sum / Mean…",
            command=self.open_hdxf_sum_tool,
            accelerator="Ctrl+Alt+S",
        )
        tools_menu.add_command(
            label="HDXF Subtract / Filter…",
            command=self.open_hdxf_subtract_tool,
            accelerator="Ctrl+Alt+D",
        )
        tools_menu.add_command(
            label="HDXF Pixel Operation…",
            command=self.open_hdxf_pixelop_tool,
            accelerator="Ctrl+Alt+P",
        )
        tools_menu.add_separator()
        tools_menu.add_command(
            label="HDF5 → HDXF Converter…",
            command=self.open_hdf5_to_hdxf_tool,
            accelerator="Ctrl+Shift+C",
        )
        tools_menu.add_command(
            label="HDXF → HDF5 Converter…",
            command=self.open_hdxf_to_hdf5_tool,
            accelerator="Ctrl+Shift+R",
        )
        menu.add_cascade(label="Tools", menu=tools_menu)

        window_menu = tk.Menu(menu, **menu_options)
        window_menu.add_command(label="New View", command=self.new_window, accelerator="Ctrl+N")
        window_menu.add_command(label="Close Current View", command=self.close_active_view, accelerator="Ctrl+W")
        window_menu.add_separator()
        window_menu.add_command(label="Arrange Views", command=self.relayout_views)
        menu.add_cascade(label="Views", menu=window_menu)
        self.root.configure(menu=menu)

    def _converter_script_path(self) -> Path:
        """Return the sibling converter script required by the Tools menu."""
        return Path(__file__).resolve().with_name("hdf5_to_hdxf.py")

    def open_hdf5_to_hdxf_tool(self) -> None:
        """Show a friendly in-window launcher for the external converter."""
        script = self._converter_script_path()
        if not script.is_file():
            messagebox.showerror(
                "Converter not found",
                "hdf5_to_hdxf.py was not found next to hdxf_viewer.py.\n\n"
                f"Expected location:\n{script}",
                parent=self.root,
            )
            return

        if self._converter_panel is not None and self._converter_panel.winfo_exists():
            self._converter_panel.lift()
            return

        overlay = tk.Frame(self.root, background="#02050B", highlightthickness=0, borderwidth=0)
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        overlay.lift()
        self._converter_panel = overlay
        self._converter_advanced_panel: tk.Frame | None = None
        self._converter_log_file = None
        self._converter_log_path_auto = True
        self._converter_details_visible = False
        self._converter_start_time: float | None = None
        self._converter_stop_requested = False
        self._converter_total_frames = 0
        self._converter_total_blocks = 0
        self._converter_completed_dataset_frames = 0
        self._converter_completed_dataset_blocks = 0
        self._converter_current_dataset_frames = 0
        self._converter_current_dataset_blocks = 0
        self._converter_current_block = 0
        self._converter_last_progress = 0.0

        shell = ttk.Frame(overlay, style="Card.TFrame", padding=(20, 17))
        shell.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.84, relheight=0.68)
        self.converter_shell = shell

        header = ttk.Frame(shell, style="Card.TFrame")
        header.pack(fill="x", pady=(0, 14))
        title_area = ttk.Frame(header, style="Card.TFrame")
        title_area.pack(side="left", fill="x", expand=True)
        ttk.Label(title_area, text="HDF5 → HDXF CONVERTER", style="InspectorTitle.TLabel").pack(anchor="w")
        ttk.Label(
            title_area,
            text="Simple conversion mode · advanced settings are optional",
            style="CardSubtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))
        ttk.Button(
            header,
            text="Close",
            style="Secondary.TButton",
            command=self._close_converter_tool,
        ).pack(side="right")

        form = ttk.Frame(shell, style="Card.TFrame")
        form.pack(fill="x")
        form.grid_columnconfigure(1, weight=1)

        self.converter_input_var = tk.StringVar()
        self.converter_output_var = tk.StringVar()
        self.converter_preset_var = tk.StringVar(value="Recommended Delta")
        self.converter_backend_var = tk.StringVar(value="cuda")
        self.converter_overwrite_var = tk.BooleanVar(value=True)
        self.converter_open_after_var = tk.BooleanVar(value=False)
        self.converter_save_log_var = tk.BooleanVar(value=False)
        self.converter_log_path_var = tk.StringVar()
        self.converter_status_var = tk.StringVar(value="Ready")
        self.converter_stage_var = tk.StringVar(value="Ready to convert")
        self.converter_activity_var = tk.StringVar(
            value="Select an HDF5 master file and an output location, then press Start Conversion."
        )
        self.converter_progress_var = tk.DoubleVar(value=0.0)
        self.converter_progress_text_var = tk.StringVar(value="0%")
        self.converter_elapsed_var = tk.StringVar(value="Elapsed  —")
        self.converter_work_var = tk.StringVar(value="Frames  —")
        self.converter_output_size_var = tk.StringVar(value="Output  —")
        self.converter_options_summary_var = tk.StringVar()

        # Advanced values.  These are deliberately represented by constrained
        # controls instead of exposing a free-form command line.
        self.converter_frame_transform_var = tk.StringVar(value="delta")
        self.converter_block_frames_var = tk.StringVar(value="32")
        self.converter_keyframe_interval_var = tk.StringVar(value="4")
        self.converter_delta_stream_var = tk.StringVar(value="zero-rle-bitpack")
        self.converter_delta_filter_var = tk.StringVar(value="learn")
        self.converter_zstd_level_var = tk.StringVar(value="3")
        self.converter_blosc_threads_var = tk.StringVar(value="2")
        self.converter_global_static_var = tk.StringVar(value="off")
        self.converter_calibration_mode_var = tk.StringVar(value="embedded")
        self.converter_calibration_read_var = tk.StringVar(value="auto")
        self.converter_frame_index_var = tk.StringVar(value="binary")
        self.converter_manifest_layout_var = tk.StringVar(value="compact")
        self.converter_cuda_device_var = tk.StringVar(value="0")
        self.converter_gpu_memory_var = tk.StringVar(value="0")
        self.converter_source_checksum_var = tk.BooleanVar(value=False)
        self.converter_allow_missing_external_var = tk.BooleanVar(value=False)

        def add_path_row(row: int, label: str, variable: tk.StringVar, command) -> None:
            ttk.Label(form, text=label, style="FieldLabel.TLabel").grid(
                row=row, column=0, sticky="w", padx=(0, 10), pady=5
            )
            ttk.Entry(form, textvariable=variable, style="Modern.TEntry").grid(
                row=row, column=1, sticky="ew", pady=5
            )
            ttk.Button(form, text="Browse…", style="Secondary.TButton", command=command).grid(
                row=row, column=2, padx=(8, 0), pady=5
            )

        add_path_row(0, "INPUT HDF5", self.converter_input_var, self._browse_converter_input)
        add_path_row(1, "OUTPUT HDXF", self.converter_output_var, self._browse_converter_output)

        ttk.Label(form, text="CONVERSION MODE", style="FieldLabel.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 10), pady=5
        )
        mode_row = ttk.Frame(form, style="Card.TFrame")
        mode_row.grid(row=2, column=1, columnspan=2, sticky="ew", pady=5)
        preset = ttk.Combobox(
            mode_row,
            state="readonly",
            textvariable=self.converter_preset_var,
            values=("Recommended Delta", "Converter defaults"),
            style="Modern.TCombobox",
            width=23,
        )
        preset.pack(side="left")
        preset.bind("<<ComboboxSelected>>", self._on_converter_preset_selected)
        ttk.Label(mode_row, text="BACKEND", style="FieldLabel.TLabel").pack(side="left", padx=(18, 6))
        backend = ttk.Combobox(
            mode_row,
            state="readonly",
            textvariable=self.converter_backend_var,
            values=("cuda", "cpu", "cpu-vectorized", "auto"),
            style="Modern.TCombobox",
            width=13,
        )
        backend.pack(side="left")
        backend.bind("<<ComboboxSelected>>", lambda _e: self._update_converter_options_summary())
        self.converter_advanced_button = ttk.Button(
            mode_row,
            text="Advanced Options…",
            style="Secondary.TButton",
            command=self._open_converter_advanced_options,
        )
        self.converter_advanced_button.pack(side="right")

        ttk.Label(form, text="CURRENT SETTINGS", style="FieldLabel.TLabel").grid(
            row=3, column=0, sticky="nw", padx=(0, 10), pady=5
        )
        ttk.Label(
            form,
            textvariable=self.converter_options_summary_var,
            style="CardSubtitle.TLabel",
            justify="left",
            wraplength=820,
        ).grid(row=3, column=1, columnspan=2, sticky="w", pady=5)

        flags = ttk.Frame(form, style="Card.TFrame")
        flags.grid(row=4, column=1, columnspan=2, sticky="w", pady=(5, 4))
        ttk.Checkbutton(
            flags,
            text="Overwrite existing output",
            variable=self.converter_overwrite_var,
            style="Modern.TCheckbutton",
        ).pack(side="left")
        ttk.Checkbutton(
            flags,
            text="Open output after success",
            variable=self.converter_open_after_var,
            style="Modern.TCheckbutton",
        ).pack(side="left", padx=(18, 0))
        self.converter_save_log_check = ttk.Checkbutton(
            flags,
            text="Save detailed log",
            variable=self.converter_save_log_var,
            style="Modern.TCheckbutton",
            command=self._toggle_converter_log_controls,
        )
        self.converter_save_log_check.pack(side="left", padx=(18, 0))

        ttk.Label(form, text="LOG FILE", style="FieldLabel.TLabel").grid(
            row=5, column=0, sticky="w", padx=(0, 10), pady=5
        )
        self.converter_log_path_entry = ttk.Entry(
            form, textvariable=self.converter_log_path_var, style="Modern.TEntry"
        )
        self.converter_log_path_entry.grid(row=5, column=1, sticky="ew", pady=5)
        self.converter_log_path_entry.bind(
            "<KeyRelease>", lambda _event: setattr(self, "_converter_log_path_auto", False)
        )
        self.converter_log_browse_button = ttk.Button(
            form,
            text="Browse…",
            style="Secondary.TButton",
            command=self._browse_converter_log,
        )
        self.converter_log_browse_button.grid(row=5, column=2, padx=(8, 0), pady=5)

        progress_card = ttk.Frame(shell, style="Card.TFrame", padding=(15, 13))
        progress_card.pack(fill="x", pady=(14, 10))
        progress_header = ttk.Frame(progress_card, style="Card.TFrame")
        progress_header.pack(fill="x")
        ttk.Label(progress_header, textvariable=self.converter_stage_var, style="CardTitle.TLabel").pack(side="left")
        ttk.Label(
            progress_header,
            textvariable=self.converter_progress_text_var,
            style="InspectorChip.TLabel",
        ).pack(side="right")
        ttk.Label(
            progress_card,
            textvariable=self.converter_activity_var,
            style="CardSubtitle.TLabel",
            anchor="w",
            justify="left",
            wraplength=1120,
        ).pack(fill="x", pady=(5, 10))
        self.converter_progressbar = ttk.Progressbar(
            progress_card,
            orient="horizontal",
            mode="determinate",
            maximum=100.0,
            variable=self.converter_progress_var,
            style="Converter.Horizontal.TProgressbar",
        )
        self.converter_progressbar.pack(fill="x", ipady=4)

        metrics = ttk.Frame(progress_card, style="Card.TFrame")
        metrics.pack(fill="x", pady=(11, 0))
        for column in range(3):
            metrics.grid_columnconfigure(column, weight=1)
        ttk.Label(metrics, textvariable=self.converter_work_var, style="ReadoutMuted.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(metrics, textvariable=self.converter_elapsed_var, style="ReadoutMuted.TLabel").grid(
            row=0, column=1
        )
        ttk.Label(metrics, textvariable=self.converter_output_size_var, style="ReadoutMuted.TLabel").grid(
            row=0, column=2, sticky="e"
        )

        actions = ttk.Frame(shell, style="Card.TFrame")
        actions.pack(fill="x", pady=(0, 4))
        self.converter_start_button = ttk.Button(
            actions,
            text="▶  Start Conversion",
            style="Accent.TButton",
            command=self._start_converter_process,
        )
        self.converter_start_button.pack(side="left")
        self.converter_stop_button = ttk.Button(
            actions,
            text="■  Stop",
            style="Secondary.TButton",
            command=self._stop_converter_process,
            state="disabled",
        )
        self.converter_stop_button.pack(side="left", padx=(8, 0))
        self.converter_details_button = ttk.Button(
            actions,
            text="Show details",
            style="Secondary.TButton",
            command=self._toggle_converter_details,
        )
        self.converter_details_button.pack(side="left", padx=(8, 0))
        ttk.Label(actions, textvariable=self.converter_status_var, style="ReadoutMuted.TLabel").pack(side="right")

        self.converter_details_shell = ttk.Frame(shell, style="Card.TFrame")
        ttk.Separator(self.converter_details_shell, orient="horizontal", style="Card.TSeparator").pack(
            fill="x", pady=(4, 9)
        )
        details_header = ttk.Frame(self.converter_details_shell, style="Card.TFrame")
        details_header.pack(fill="x", pady=(0, 5))
        ttk.Label(details_header, text="TECHNICAL DETAILS", style="FieldLabel.TLabel").pack(side="left")
        ttk.Label(
            details_header,
            text="Only needed for troubleshooting",
            style="CardSubtitle.TLabel",
        ).pack(side="left", padx=(9, 0))
        log_frame = tk.Frame(self.converter_details_shell, background=UI_CARD_ALT)
        log_frame.pack(fill="both", expand=True)
        self.converter_log_text = tk.Text(
            log_frame,
            height=11,
            wrap="none",
            background="#060A12",
            foreground="#C7D7EF",
            insertbackground=UI_TEXT,
            selectbackground=UI_ACCENT_DARK,
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=8,
            font=("Consolas", 9),
        )
        log_scroll = ttk.Scrollbar(
            log_frame,
            orient="vertical",
            command=self.converter_log_text.yview,
            style="Modern.Vertical.TScrollbar",
        )
        self.converter_log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")
        self.converter_log_text.pack(side="left", fill="both", expand=True)
        self.converter_log_text.insert("end", f"Converter script: {script}\n")
        self.converter_log_text.configure(state="disabled")

        self.converter_input_var.trace_add("write", lambda *_args: self._on_converter_input_changed())
        self.converter_output_var.trace_add("write", lambda *_args: self._on_converter_output_changed())
        self._update_converter_options_summary()
        self._toggle_converter_log_controls()

    def _on_converter_preset_selected(self, _event: tk.Event | None = None) -> None:
        # tkinter Variable objects (StringVar/BooleanVar/etc.) are not hashable,
        # so they must not be used as dict keys.  Keep the preset as an ordered
        # sequence of (variable, value) pairs instead.
        if self.converter_preset_var.get() == "Recommended Delta":
            values = (
                (self.converter_frame_transform_var, "delta"),
                (self.converter_block_frames_var, "32"),
                (self.converter_keyframe_interval_var, "4"),
                (self.converter_delta_stream_var, "zero-rle-bitpack"),
                (self.converter_delta_filter_var, "learn"),
                (self.converter_zstd_level_var, "3"),
                (self.converter_blosc_threads_var, "2"),
                (self.converter_global_static_var, "off"),
                (self.converter_calibration_mode_var, "embedded"),
                (self.converter_calibration_read_var, "auto"),
                (self.converter_frame_index_var, "binary"),
                (self.converter_manifest_layout_var, "compact"),
            )
        else:
            values = (
                (self.converter_frame_transform_var, "adaptive"),
                (self.converter_block_frames_var, "64"),
                (self.converter_keyframe_interval_var, "4"),
                (self.converter_delta_stream_var, "auto"),
                (self.converter_delta_filter_var, "learn"),
                (self.converter_zstd_level_var, "7"),
                (self.converter_blosc_threads_var, "1"),
                (self.converter_global_static_var, "off"),
                (self.converter_calibration_mode_var, "embedded"),
                (self.converter_calibration_read_var, "auto"),
                (self.converter_frame_index_var, "binary"),
                (self.converter_manifest_layout_var, "compact"),
            )
        for variable, value in values:
            variable.set(value)
        self._update_converter_options_summary()

    def _update_converter_options_summary(self) -> None:
        if not hasattr(self, "converter_options_summary_var"):
            return
        transform = self.converter_frame_transform_var.get().title()
        backend = self.converter_backend_var.get().upper()
        text = (
            f"{transform} encoding · {self.converter_block_frames_var.get()} frames per block · "
            f"{backend} · Zstd level {self.converter_zstd_level_var.get()} · "
            f"{self.converter_calibration_mode_var.get().title()} calibration · "
            f"{self.converter_frame_index_var.get().title()} frame index"
        )
        self.converter_options_summary_var.set(text)

    def _open_converter_advanced_options(self) -> None:
        if self._converter_panel is None or not self._converter_panel.winfo_exists():
            return
        panel = getattr(self, "_converter_advanced_panel", None)
        if panel is not None and panel.winfo_exists():
            panel.lift()
            return

        modal = tk.Frame(self._converter_panel, background="#01040A", highlightthickness=0)
        modal.place(relx=0, rely=0, relwidth=1, relheight=1)
        modal.lift()
        self._converter_advanced_panel = modal
        shell = ttk.Frame(modal, style="Card.TFrame", padding=(20, 17))
        shell.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.72, relheight=0.78)

        header = ttk.Frame(shell, style="Card.TFrame")
        header.pack(fill="x", pady=(0, 14))
        title = ttk.Frame(header, style="Card.TFrame")
        title.pack(side="left", fill="x", expand=True)
        ttk.Label(title, text="ADVANCED CONVERSION OPTIONS", style="InspectorTitle.TLabel").pack(anchor="w")
        ttk.Label(
            title,
            text="All fields use safe predefined choices; no command-line knowledge is required.",
            style="CardSubtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))
        ttk.Button(header, text="Done", style="Accent.TButton", command=self._close_converter_advanced_options).pack(side="right")

        # Keep the header and footer fixed while the complete option form can
        # scroll vertically.  This is important on laptops and at 125–200 %
        # Windows display scaling, where the safe option descriptions are
        # intentionally more valuable than squeezing every row into one page.
        scroll_host = ttk.Frame(shell, style="Card.TFrame")
        scroll_host.pack(fill="both", expand=True)

        options_canvas = tk.Canvas(
            scroll_host,
            background=UI_CARD,
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            takefocus=True,
        )
        options_scrollbar = ttk.Scrollbar(
            scroll_host,
            orient="vertical",
            command=options_canvas.yview,
            style="Modern.Vertical.TScrollbar",
        )
        options_canvas.configure(yscrollcommand=options_scrollbar.set)
        self._converter_advanced_canvas = options_canvas
        self._converter_advanced_scrollbar = options_scrollbar
        options_canvas.pack(side="left", fill="both", expand=True)
        options_scrollbar.pack(side="right", fill="y", padx=(10, 0))

        body = ttk.Frame(options_canvas, style="Card.TFrame")
        body_window = options_canvas.create_window((0, 0), window=body, anchor="nw")
        body.grid_columnconfigure(0, weight=1, uniform="advanced")
        body.grid_columnconfigure(1, weight=1, uniform="advanced")

        def update_scroll_region(_event: tk.Event | None = None) -> None:
            options_canvas.configure(scrollregion=options_canvas.bbox("all"))

        def resize_scrolled_body(event: tk.Event) -> None:
            options_canvas.itemconfigure(body_window, width=max(1, event.width))
            update_scroll_region()

        body.bind("<Configure>", update_scroll_region, add="+")
        options_canvas.bind("<Configure>", resize_scrolled_body, add="+")

        def scroll_advanced_options(event: tk.Event) -> str:
            if getattr(event, "num", None) == 4:
                units = -3
            elif getattr(event, "num", None) == 5:
                units = 3
            else:
                delta = int(getattr(event, "delta", 0))
                if delta == 0:
                    return "break"
                magnitude = max(1, abs(delta) // 120)
                units = -3 * magnitude if delta > 0 else 3 * magnitude
            options_canvas.yview_scroll(units, "units")
            return "break"

        def bind_scroll_recursively(widget: tk.Misc) -> None:
            # A readonly Combobox receives the same page-scroll binding while
            # closed.  Its transient drop-down Listbox is not a child of this
            # modal and therefore keeps its own native wheel behaviour.
            if not isinstance(widget, ttk.Scrollbar):
                widget.bind("<MouseWheel>", scroll_advanced_options, add="+")
                widget.bind("<Button-4>", scroll_advanced_options, add="+")
                widget.bind("<Button-5>", scroll_advanced_options, add="+")
            for child in widget.winfo_children():
                bind_scroll_recursively(child)

        left = ttk.Frame(body, style="Card.TFrame", padding=(0, 0, 14, 0))
        right = ttk.Frame(body, style="Card.TFrame", padding=(14, 0, 0, 0))
        left.grid(row=0, column=0, sticky="nsew")
        right.grid(row=0, column=1, sticky="nsew")

        def section(parent: ttk.Frame, title_text: str, subtitle: str) -> ttk.Frame:
            card = ttk.Frame(parent, style="Card.TFrame")
            card.pack(fill="x", pady=(0, 14))
            ttk.Label(card, text=title_text, style="CardTitle.TLabel").pack(anchor="w")
            ttk.Label(
                card, text=subtitle, style="CardSubtitle.TLabel", wraplength=420, justify="left"
            ).pack(anchor="w", pady=(2, 8))
            return card

        def combo_row(parent: ttk.Frame, label: str, variable: tk.StringVar, values: tuple[str, ...], help_text: str) -> None:
            row = ttk.Frame(parent, style="Card.TFrame")
            row.pack(fill="x", pady=(0, 8))
            ttk.Label(row, text=label, style="FieldLabel.TLabel").pack(anchor="w")
            combo = ttk.Combobox(
                row,
                state="readonly",
                textvariable=variable,
                values=values,
                style="Modern.TCombobox",
            )
            combo.pack(fill="x", pady=(3, 1))
            combo.bind("<<ComboboxSelected>>", lambda _e: self._update_converter_options_summary())
            ttk.Label(
                row,
                text=help_text,
                style="CardSubtitle.TLabel",
                wraplength=420,
                justify="left",
            ).pack(anchor="w")

        encoding = section(left, "Frame encoding", "Controls compression ratio, speed and random-access cost.")
        combo_row(
            encoding,
            "Encoding method",
            self.converter_frame_transform_var,
            ("delta", "adaptive", "auto", "raw"),
            "Delta is recommended for a time sequence; Raw prioritizes simplicity over size.",
        )
        combo_row(
            encoding,
            "Frames per block",
            self.converter_block_frames_var,
            ("16", "32", "64", "128"),
            "32 is the balanced setting. Smaller blocks improve random access but add overhead.",
        )
        combo_row(
            encoding,
            "Keyframe interval (blocks)",
            self.converter_keyframe_interval_var,
            ("1", "2", "4", "8"),
            "4 provides strong compression while limiting the Delta decode chain.",
        )
        combo_row(
            encoding,
            "Delta stream",
            self.converter_delta_stream_var,
            ("zero-rle-bitpack", "auto", "zstd"),
            "Zero-RLE bit-packing is best for sparse detector differences.",
        )
        combo_row(
            encoding,
            "Delta filter",
            self.converter_delta_filter_var,
            ("learn", "auto", "none", "shuffle", "bitshuffle"),
            "Learn tests safe filters on the first block and reuses the best result.",
        )

        performance = section(right, "Performance", "Adjust CPU/GPU use and compression effort.")
        combo_row(
            performance,
            "Zstd compression level",
            self.converter_zstd_level_var,
            tuple(str(value) for value in range(10)),
            "Level 3 is fast. Higher levels may save a little space but take longer.",
        )
        combo_row(
            performance,
            "Blosc worker threads",
            self.converter_blosc_threads_var,
            ("1", "2", "4", "8"),
            "2 is conservative on Windows. More threads can increase memory and contention.",
        )
        combo_row(
            performance,
            "CUDA device",
            self.converter_cuda_device_var,
            ("0", "1", "2", "3"),
            "Use 0 unless the computer has more than one CUDA GPU.",
        )
        combo_row(
            performance,
            "GPU memory limit (MiB)",
            self.converter_gpu_memory_var,
            ("0", "2048", "3072", "4096", "5120"),
            "0 lets CuPy manage the pool. Set a limit only when sharing GPU memory.",
        )
        combo_row(
            performance,
            "Static-pixel scan",
            self.converter_global_static_var,
            ("off", "dataset", "experiment"),
            "Off is recommended for this detector data and avoids a second complete read pass.",
        )

        archive = section(left, "Archive contents", "Controls how calibration and indexes are stored.")
        combo_row(
            archive,
            "Calibration storage",
            self.converter_calibration_mode_var,
            ("embedded", "referenced", "hybrid"),
            "Embedded creates one self-contained HDXF file.",
        )
        combo_row(
            archive,
            "Calibration read mode",
            self.converter_calibration_read_var,
            ("auto", "raw-chunks", "logical"),
            "Auto safely preserves filtered HDF5 chunks when logical decoding is risky.",
        )
        combo_row(
            archive,
            "Frame index",
            self.converter_frame_index_var,
            ("binary", "json"),
            "Binary is smaller and recommended.",
        )
        combo_row(
            archive,
            "Manifest layout",
            self.converter_manifest_layout_var,
            ("compact", "pretty"),
            "Compact saves space; Pretty is intended for manual inspection.",
        )

        checks = section(right, "Optional checks", "Extra safety checks can increase conversion time.")
        ttk.Checkbutton(
            checks,
            text="Calculate source-file SHA-256 checksums",
            variable=self.converter_source_checksum_var,
            style="Modern.TCheckbutton",
        ).pack(anchor="w", pady=(0, 6))
        ttk.Label(
            checks,
            text="Reads every source file again to record a checksum.",
            style="CardSubtitle.TLabel",
            wraplength=420,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))
        ttk.Checkbutton(
            checks,
            text="Allow missing external HDF5 files",
            variable=self.converter_allow_missing_external_var,
            style="Modern.TCheckbutton",
        ).pack(anchor="w", pady=(0, 6))
        ttk.Label(
            checks,
            text="Leave disabled for a complete detector archive. Enable only for deliberate partial preservation.",
            style="CardSubtitle.TLabel",
            wraplength=420,
            justify="left",
        ).pack(anchor="w")

        footer = ttk.Frame(shell, style="Card.TFrame")
        footer.pack(fill="x", pady=(12, 0))
        ttk.Button(
            footer,
            text="Restore selected preset",
            style="Secondary.TButton",
            command=self._on_converter_preset_selected,
        ).pack(side="left")
        ttk.Button(
            footer,
            text="Apply and close",
            style="Accent.TButton",
            command=self._close_converter_advanced_options,
        ).pack(side="right")

        modal.update_idletasks()
        update_scroll_region()
        bind_scroll_recursively(modal)
        options_canvas.focus_set()

    def _close_converter_advanced_options(self) -> None:
        self._update_converter_options_summary()
        panel = getattr(self, "_converter_advanced_panel", None)
        self._converter_advanced_panel = None
        self._converter_advanced_canvas = None
        self._converter_advanced_scrollbar = None
        if panel is not None:
            try:
                if panel.winfo_exists():
                    panel.destroy()
            except tk.TclError:
                pass

    def _browse_converter_input(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self.root,
            title="Select HDF5 master file",
            filetypes=[
                ("HDF5 master files", "*_master.h5"),
                ("HDF5 files", "*.h5 *.hdf5"),
                ("All files", "*.*"),
            ],
        )
        if not filename:
            return
        # converter_input_var has a write trace; changing the source therefore
        # refreshes the next output filename automatically.
        self.converter_input_var.set(str(Path(filename)))

    def _browse_converter_output(self) -> None:
        current = self.converter_output_var.get().strip()
        initial = Path(current) if current else None
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save HDXF archive",
            defaultextension=".hdxf",
            initialdir=str(initial.parent) if initial is not None else None,
            initialfile=initial.name if initial is not None else None,
            filetypes=[("HDXF detector archive", "*.hdxf"), ("All files", "*.*")],
        )
        if filename:
            self.converter_output_var.set(str(Path(filename)))

    def _browse_converter_log(self) -> None:
        current = self.converter_log_path_var.get().strip()
        initial = Path(current) if current else None
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save detailed conversion log",
            defaultextension=".log",
            initialdir=str(initial.parent) if initial is not None else None,
            initialfile=initial.name if initial is not None else None,
            filetypes=[("Log file", "*.log"), ("Text file", "*.txt"), ("All files", "*.*")],
        )
        if filename:
            self._converter_log_path_auto = False
            self.converter_log_path_var.set(str(Path(filename)))

    def _on_converter_input_changed(self) -> None:
        """Synchronize OUTPUT HDXF with the newly selected INPUT HDF5.

        When an output path already exists, keep its directory and replace only
        the filename.  This makes batch-like repeated conversions convenient:
        choose an output folder once, then select BEST/WORST/other masters in
        succession without accidentally overwriting the previous .hdxf file.
        """
        input_text = self.converter_input_var.get().strip()
        if not input_text:
            return

        source = Path(input_text)
        output_name = source.with_suffix(".hdxf").name
        current_output = self.converter_output_var.get().strip()

        if current_output:
            # Preserve the current output directory, including a directory the
            # user selected manually via Browse..., and update only the name.
            target = Path(current_output).with_name(output_name)
        else:
            target = source.with_suffix(".hdxf")

        target_text = str(target)
        if current_output != target_text:
            self.converter_output_var.set(target_text)

        # A new source starts a new job. Clear the visual state left by the
        # previous completed conversion so the panel no longer shows 100%.
        process = getattr(self, "_converter_process", None)
        if process is None or process.poll() is not None:
            self.converter_progress_var.set(0.0)
            self.converter_progress_text_var.set("0%")
            self.converter_stage_var.set("Ready to convert")
            self.converter_activity_var.set(
                "Input changed. Review the automatically updated output path, then press Start Conversion."
            )
            self.converter_status_var.set("Ready")
            self.converter_elapsed_var.set("Elapsed  —")
            self.converter_work_var.set("Frames  —")
            self.converter_output_size_var.set("Output  —")
            self._converter_last_progress = 0.0

    def _on_converter_output_changed(self) -> None:
        if self._converter_log_path_auto:
            output = self.converter_output_var.get().strip()
            if output:
                path = Path(output)
                self.converter_log_path_var.set(str(path.with_name(path.stem + ".conversion.log")))
            else:
                self.converter_log_path_var.set("")

    def _toggle_converter_log_controls(self) -> None:
        enabled = bool(self.converter_save_log_var.get())
        state = "normal" if enabled else "disabled"
        if enabled and not self.converter_log_path_var.get().strip():
            self._on_converter_output_changed()
        for widget_name in ("converter_log_path_entry", "converter_log_browse_button"):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                try:
                    widget.configure(state=state)
                except tk.TclError:
                    pass

    def _toggle_converter_details(self) -> None:
        if self._converter_details_visible:
            self.converter_details_shell.pack_forget()
            self.converter_details_button.configure(text="Show details")
            try:
                self.converter_shell.place_configure(relheight=0.68)
            except tk.TclError:
                pass
            self._converter_details_visible = False
        else:
            try:
                self.converter_shell.place_configure(relheight=0.88)
            except tk.TclError:
                pass
            self.converter_details_shell.pack(fill="both", expand=True, pady=(6, 0))
            self.converter_details_button.configure(text="Hide details")
            self._converter_details_visible = True

    def _build_converter_command(self, *, validate: bool) -> list[str]:
        script = self._converter_script_path()
        input_text = self.converter_input_var.get().strip()
        output_text = self.converter_output_var.get().strip()
        if validate:
            if not script.is_file():
                raise HDXFViewerError(f"converter script not found: {script}")
            if not input_text:
                raise HDXFViewerError("select an input HDF5 master file")
            input_path = Path(input_text)
            if not input_path.is_file():
                raise HDXFViewerError(f"input HDF5 file does not exist: {input_path}")
            if not output_text:
                raise HDXFViewerError("select an output .hdxf file")
            if Path(output_text).resolve() == input_path.resolve():
                raise HDXFViewerError("input and output paths must be different")
            if self.converter_save_log_var.get() and not self.converter_log_path_var.get().strip():
                raise HDXFViewerError("select a detailed log file or turn off Save detailed log")

        cmd = [sys.executable, "-u", str(script)]
        if input_text:
            cmd.append(input_text)
        if output_text:
            cmd.extend(["-o", output_text])

        cmd.extend([
            "--profile", "detector-frame-archive",
            "--block-frames", self.converter_block_frames_var.get(),
            "--frame-transform", self.converter_frame_transform_var.get(),
            "--compute-backend", self.converter_backend_var.get(),
            "--cuda-device", self.converter_cuda_device_var.get(),
            "--gpu-memory-limit-mib", self.converter_gpu_memory_var.get(),
            "--keyframe-interval-blocks", self.converter_keyframe_interval_var.get(),
            "--delta-stream", self.converter_delta_stream_var.get(),
            "--global-static-scan", self.converter_global_static_var.get(),
            "--frame-index-layout", self.converter_frame_index_var.get(),
            "--delta-filter", self.converter_delta_filter_var.get(),
            "--zstd-level", self.converter_zstd_level_var.get(),
            "--blosc-threads", self.converter_blosc_threads_var.get(),
            "--gc-every", "0",
            "--manifest-layout", self.converter_manifest_layout_var.get(),
            "--calibration-mode", self.converter_calibration_mode_var.get(),
            "--calibration-read-mode", self.converter_calibration_read_var.get(),
            "--progress-every", "1",
        ])
        if self.converter_source_checksum_var.get():
            cmd.append("--source-checksum")
        if self.converter_allow_missing_external_var.get():
            cmd.append("--allow-missing-external")
        if self.converter_overwrite_var.get():
            cmd.append("--overwrite")
        return cmd

    @staticmethod
    def _format_command_for_display(cmd: list[str]) -> str:
        if sys.platform == "win32":
            return subprocess.list2cmdline(cmd)
        return shlex.join(cmd)

    @staticmethod
    def _estimate_hdf5_work(input_path: Path, block_frames: int) -> tuple[int, int]:
        """Estimate detector frames/blocks from HDF5 metadata only."""
        try:
            import h5py
        except ImportError:
            return 0, 0

        total_frames = 0
        total_blocks = 0
        block_frames = max(1, int(block_frames))
        seen: set[tuple[str, str]] = set()
        with h5py.File(input_path, "r") as handle:
            data_group = handle.get("/entry/data")
            if isinstance(data_group, h5py.Group):
                for name in sorted(data_group.keys()):
                    try:
                        dataset = data_group[name]
                        if not isinstance(dataset, h5py.Dataset) or dataset.shape is None or len(dataset.shape) < 3:
                            continue
                        identity = (str(Path(dataset.file.filename).resolve()), dataset.name)
                        if identity in seen:
                            continue
                        seen.add(identity)
                        frames = int(dataset.shape[0])
                        total_frames += frames
                        total_blocks += int(math.ceil(frames / block_frames))
                    except Exception:
                        continue
            if total_frames == 0:
                def visitor(_name: str, obj: Any) -> None:
                    nonlocal total_frames, total_blocks
                    if not isinstance(obj, h5py.Dataset) or obj.shape is None or len(obj.shape) < 3:
                        return
                    identity = (str(Path(obj.file.filename).resolve()), obj.name)
                    if identity in seen:
                        return
                    seen.add(identity)
                    frames = int(obj.shape[0])
                    total_frames += frames
                    total_blocks += int(math.ceil(frames / block_frames))
                handle.visititems(visitor)
        return total_frames, total_blocks

    def _append_converter_log(self, text: str) -> None:
        log_file = getattr(self, "_converter_log_file", None)
        if log_file is not None:
            try:
                log_file.write(text)
                log_file.flush()
            except Exception:
                pass
        if not hasattr(self, "converter_log_text"):
            return
        try:
            if not self.converter_log_text.winfo_exists():
                return
        except tk.TclError:
            return
        self.converter_log_text.configure(state="normal")
        self.converter_log_text.insert("end", text)
        self.converter_log_text.see("end")
        self.converter_log_text.configure(state="disabled")

    def _set_converter_progress(self, value: float, *, text: str | None = None) -> None:
        value = max(self._converter_last_progress, min(100.0, float(value)))
        self._converter_last_progress = value
        try:
            self.converter_progressbar.stop()
            self.converter_progressbar.configure(mode="determinate")
        except tk.TclError:
            pass
        self.converter_progress_var.set(value)
        self.converter_progress_text_var.set(text if text is not None else f"{value:.0f}%")

    def _set_converter_indeterminate(self, activity: str) -> None:
        self.converter_activity_var.set(activity)
        try:
            self.converter_progressbar.configure(mode="indeterminate")
            self.converter_progressbar.start(12)
        except tk.TclError:
            pass
        self.converter_progress_text_var.set("Working…")

    def _update_converter_elapsed(self) -> None:
        started = self._converter_start_time
        if started is None:
            return
        elapsed = max(0.0, time.monotonic() - started)
        if elapsed < 60:
            label = f"Elapsed  {elapsed:.0f} s"
        else:
            minutes, seconds = divmod(int(elapsed), 60)
            label = f"Elapsed  {minutes:d}m {seconds:02d}s"
        self.converter_elapsed_var.set(label)

    def _handle_converter_output_line(self, raw_line: str) -> None:
        line = re.sub(r"^\[\d{2}:\d{2}:\d{2}\]\s*", "", raw_line.strip())
        if not line:
            return

        dataset_match = re.search(r"frame dataset\s+(.+?):\s+frames=(\d+).*?blocks=(\d+)", line)
        if dataset_match:
            if self._converter_current_dataset_blocks:
                self._converter_completed_dataset_blocks += self._converter_current_dataset_blocks
                self._converter_completed_dataset_frames += self._converter_current_dataset_frames
            self._converter_current_dataset_frames = int(dataset_match.group(2))
            self._converter_current_dataset_blocks = int(dataset_match.group(3))
            self._converter_current_block = 0
            self.converter_stage_var.set("Compressing detector frames")
            self.converter_activity_var.set(
                f"Preparing {self._converter_current_dataset_frames:,} frames from the next detector dataset."
            )
            return

        block_match = re.search(r"encoding frame block\s+(\d+)/(\d+)", line)
        if block_match:
            block_number = int(block_match.group(1))
            block_total = int(block_match.group(2))
            self._converter_current_block = block_number
            self._converter_current_dataset_blocks = max(self._converter_current_dataset_blocks, block_total)
            done_blocks = self._converter_completed_dataset_blocks + block_number
            block_frames = max(1, int(self.converter_block_frames_var.get()))
            done_frames = self._converter_completed_dataset_frames + min(
                self._converter_current_dataset_frames, block_number * block_frames
            )
            if self._converter_total_blocks > 0:
                fraction = min(1.0, done_blocks / self._converter_total_blocks)
                percent = 5.0 + 87.0 * fraction
                self._set_converter_progress(percent)
            self.converter_stage_var.set("Compressing detector frames")
            if self._converter_total_frames > 0:
                self.converter_activity_var.set(
                    f"Processed approximately {min(done_frames, self._converter_total_frames):,} of "
                    f"{self._converter_total_frames:,} frames."
                )
                self.converter_work_var.set(
                    f"Frames  {min(done_frames, self._converter_total_frames):,} / {self._converter_total_frames:,}"
                )
            else:
                self.converter_activity_var.set(f"Encoding detector block {block_number} of {block_total}.")
                self.converter_work_var.set(f"Block  {block_number} / {block_total}")
            return

        reading_match = re.search(r"reading frame block\s+(\d+)/(\d+)", line)
        if reading_match:
            self.converter_stage_var.set("Reading detector frames")
            self.converter_activity_var.set(
                f"Reading source block {reading_match.group(1)} of {reading_match.group(2)}."
            )
            return

        lower = line.lower()
        if "starting conversion" in lower:
            self.converter_stage_var.set("Preparing conversion")
            self.converter_activity_var.set("Opening the HDF5 master file and preparing the output archive.")
            self._set_converter_progress(1.0)
        elif "scanning detector metadata" in lower or "resolving calibration" in lower:
            self.converter_stage_var.set("Reading detector metadata")
            self.converter_activity_var.set("Collecting detector, beamline and calibration metadata.")
            self._set_converter_progress(2.0)
        elif "walking hdf5 object tree" in lower:
            self.converter_stage_var.set("Scanning HDF5 contents")
            self.converter_activity_var.set("Discovering datasets and external detector data files.")
            self._set_converter_progress(4.0)
        elif "pre-scanning" in lower and "static" in lower:
            self.converter_stage_var.set("Analysing static pixels")
            self.converter_activity_var.set("Checking whether unchanged detector pixels can be shared.")
        elif "calibration" in lower and ("embedding" in lower or "preserving" in lower or "writing" in lower):
            self.converter_stage_var.set("Embedding calibration")
            self.converter_activity_var.set("Adding detector calibration data to the HDXF archive.")
            self._set_converter_progress(94.0)
        elif "building manifest" in lower:
            self.converter_stage_var.set("Finalising archive metadata")
            self.converter_activity_var.set("Building the compact HDF5 object and frame index.")
            self._set_converter_progress(97.0)
        elif "published archive" in lower or line.startswith("created:"):
            self.converter_stage_var.set("Finishing output file")
            self.converter_activity_var.set("The archive has been written; final checks are completing.")
            self._set_converter_progress(99.0)
        elif "space breakdown:" in lower:
            match = re.search(r"archive=([0-9.]+)\s*MiB", line, re.IGNORECASE)
            if match:
                self.converter_output_size_var.set(f"Output  {match.group(1)} MiB")
        elif line.startswith("frame payload:"):
            match = re.search(r"ratio:\s*([0-9.]+)x", line)
            if match:
                self.converter_output_size_var.set(f"Compression  {match.group(1)}×")
        elif lower.startswith("error:") or "traceback (most recent call last)" in lower:
            self.converter_stage_var.set("Conversion error")
            self.converter_activity_var.set("The converter reported an error. Open details or the saved log for diagnostics.")

    def _start_converter_process(self) -> None:
        if self._converter_process is not None and self._converter_process.poll() is None:
            return
        try:
            cmd = self._build_converter_command(validate=True)
        except Exception as exc:
            messagebox.showerror("Cannot start conversion", str(exc), parent=self.root)
            return

        output_text = self.converter_output_var.get().strip()
        self._converter_output_path = Path(output_text) if output_text else None
        self._converter_stop_requested = False
        self._converter_last_progress = 0.0
        self.converter_progress_var.set(0.0)
        self.converter_progress_text_var.set("0%")
        self.converter_stage_var.set("Preparing conversion")
        self.converter_activity_var.set("Inspecting the HDF5 frame layout before conversion starts.")
        self.converter_status_var.set("Starting…")
        self.converter_elapsed_var.set("Elapsed  0 s")
        self.converter_work_var.set("Frames  scanning…")
        self.converter_output_size_var.set("Output  —")
        self._converter_start_time = time.monotonic()
        self._converter_completed_dataset_frames = 0
        self._converter_completed_dataset_blocks = 0
        self._converter_current_dataset_frames = 0
        self._converter_current_dataset_blocks = 0
        self._converter_current_block = 0

        self.converter_log_text.configure(state="normal")
        self.converter_log_text.delete("1.0", "end")
        self.converter_log_text.configure(state="disabled")

        self._close_converter_log_file()
        if self.converter_save_log_var.get():
            try:
                log_path = Path(self.converter_log_path_var.get().strip())
                log_path.parent.mkdir(parents=True, exist_ok=True)
                self._converter_log_file = log_path.open("w", encoding="utf-8", newline="")
            except Exception as exc:
                messagebox.showerror("Cannot create log file", f"{type(exc).__name__}: {exc}", parent=self.root)
                return

        self._append_converter_log(f"$ {self._format_command_for_display(cmd)}\n\n")

        try:
            block_frames = max(1, int(self.converter_block_frames_var.get()))
            self._converter_total_frames, self._converter_total_blocks = self._estimate_hdf5_work(
                Path(self.converter_input_var.get().strip()), block_frames
            )
        except Exception as exc:
            self._converter_total_frames = 0
            self._converter_total_blocks = 0
            self._append_converter_log(f"[progress estimate unavailable] {exc}\n")
        if self._converter_total_frames > 0:
            self.converter_work_var.set(f"Frames  0 / {self._converter_total_frames:,}")
            self.converter_activity_var.set(
                f"Detected {self._converter_total_frames:,} detector frames in "
                f"{self._converter_total_blocks:,} compression blocks."
            )
            self._set_converter_progress(0.0)
        else:
            self._set_converter_indeterminate("Starting conversion; frame count will appear when available.")

        creationflags = 0
        if sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            creationflags = int(subprocess.CREATE_NO_WINDOW)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(self._converter_script_path().parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=creationflags,
            )
        except Exception as exc:
            self._close_converter_log_file()
            messagebox.showerror("Conversion launch failed", f"{type(exc).__name__}: {exc}", parent=self.root)
            return

        self._converter_run_serial += 1
        run_serial = self._converter_run_serial
        while True:
            try:
                self._converter_queue.get_nowait()
            except queue.Empty:
                break

        self._converter_process = process
        self.converter_start_button.configure(state="disabled")
        self.converter_stop_button.configure(state="normal")
        self.converter_status_var.set("Running")
        self.status_var.set(f"HDF5 → HDXF conversion running · PID {process.pid}")

        def reader() -> None:
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    self._converter_queue.put((run_serial, "line", line))
            except Exception as exc:
                self._converter_queue.put((run_serial, "line", f"\n[log reader error] {exc}\n"))
            finally:
                return_code = process.wait()
                self._converter_queue.put((run_serial, "done", return_code))

        self._converter_reader = threading.Thread(target=reader, name="hdxf-converter-log", daemon=True)
        self._converter_reader.start()
        self._schedule_converter_poll()

    def _schedule_converter_poll(self) -> None:
        if self._converter_poll_after_id is None:
            self._converter_poll_after_id = self.root.after(80, self._poll_converter_queue)

    def _poll_converter_queue(self) -> None:
        self._converter_poll_after_id = None
        done_code: int | None = None
        while True:
            try:
                run_serial, kind, payload = self._converter_queue.get_nowait()
            except queue.Empty:
                break
            if run_serial != self._converter_run_serial:
                continue
            if kind == "line":
                line = str(payload)
                self._append_converter_log(line)
                self._handle_converter_output_line(line)
            elif kind == "done":
                done_code = int(payload)

        self._update_converter_elapsed()
        output = self._converter_output_path
        if output is not None and output.is_file():
            try:
                size_mib = output.stat().st_size / (1024 * 1024)
                self.converter_output_size_var.set(f"Output  {size_mib:.2f} MiB")
            except OSError:
                pass

        if done_code is not None:
            self._converter_process = None
            try:
                self.converter_progressbar.stop()
            except tk.TclError:
                pass
            if hasattr(self, "converter_start_button"):
                try:
                    if self.converter_start_button.winfo_exists():
                        self.converter_start_button.configure(state="normal")
                        self.converter_stop_button.configure(state="disabled")
                except tk.TclError:
                    pass
            if done_code == 0:
                self._set_converter_progress(100.0, text="100%")
                self.converter_stage_var.set("Conversion completed")
                self.converter_activity_var.set("The HDXF archive was created successfully and is ready to open.")
                self.converter_status_var.set("Completed successfully")
                self.status_var.set("HDF5 → HDXF conversion completed")
                self._append_converter_log("\n[completed successfully]\n")
                if (
                    bool(self.converter_open_after_var.get())
                    and output is not None
                    and output.is_file()
                ):
                    self.open_path(output)
            elif self._converter_stop_requested:
                self.converter_stage_var.set("Conversion stopped")
                self.converter_activity_var.set("Conversion was stopped by the user. A partial output file may remain.")
                self.converter_status_var.set("Stopped")
                self.status_var.set("HDF5 → HDXF conversion stopped")
                self._append_converter_log(f"\n[stopped; process exited with code {done_code}]\n")
            else:
                self.converter_stage_var.set("Conversion failed")
                self.converter_activity_var.set(
                    "The HDXF file could not be completed. Open details or the saved log for the technical error."
                )
                self.converter_status_var.set("Failed")
                self.status_var.set(f"HDF5 → HDXF conversion failed · exit code {done_code}")
                self._append_converter_log(f"\n[process exited with code {done_code}]\n")
                if not self._converter_details_visible:
                    self._toggle_converter_details()
            self._close_converter_log_file()
            return

        process = self._converter_process
        if process is not None and process.poll() is None:
            self._schedule_converter_poll()

    def _stop_converter_process(self) -> None:
        process = self._converter_process
        if process is None or process.poll() is not None:
            return
        self._converter_stop_requested = True
        self.converter_status_var.set("Stopping…")
        self.converter_stage_var.set("Stopping conversion")
        self.converter_activity_var.set("Waiting for the converter process to stop safely.")
        self._append_converter_log("\n[stop requested]\n")
        try:
            process.terminate()
        except Exception as exc:
            self._append_converter_log(f"[terminate failed] {exc}\n")

    def _close_converter_log_file(self) -> None:
        log_file = getattr(self, "_converter_log_file", None)
        self._converter_log_file = None
        if log_file is not None:
            try:
                log_file.flush()
                log_file.close()
            except Exception:
                pass

    def _close_converter_tool(self) -> None:
        process = self._converter_process
        if process is not None and process.poll() is None:
            close = messagebox.askyesno(
                "Conversion is running",
                "Stop the running conversion and close the tool?",
                parent=self.root,
            )
            if not close:
                return
            self._stop_converter_process()
            self._converter_run_serial += 1
            self._converter_process = None
        if self._converter_poll_after_id is not None:
            try:
                self.root.after_cancel(self._converter_poll_after_id)
            except tk.TclError:
                pass
            self._converter_poll_after_id = None
        self._close_converter_log_file()
        self._close_converter_advanced_options()
        panel = self._converter_panel
        self._converter_panel = None
        if panel is not None and panel.winfo_exists():
            panel.destroy()

    def _restore_script_path(self) -> Path:
        """Return the fixed sibling HDXF-to-HDF5 converter."""
        return Path(__file__).resolve().with_name("hdxf_to_hdf5.py")

    def _legacy_template_dir(self) -> Path:
        """Return the template directory used by exact HDF5 restoration."""
        return _default_legacy_template_dir()

    def open_hdxf_to_hdf5_tool(self) -> None:
        """Show an in-window launcher for the external reverse converter."""
        script = self._restore_script_path()
        if not script.is_file():
            messagebox.showerror(
                "Converter not found",
                "hdxf_to_hdf5.py was not found next to hdxf_viewer.py.\n\n"
                f"Expected location:\n{script}",
                parent=self.root,
            )
            return
        if self._restore_panel is not None and self._restore_panel.winfo_exists():
            self._restore_panel.lift()
            return

        overlay = tk.Frame(self.root, background="#02050B", highlightthickness=0, borderwidth=0)
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        overlay.lift()
        self._restore_panel = overlay
        self._restore_log_file = None
        self._restore_details_visible = False
        self._restore_stop_requested = False
        self._restore_total_frames = 0
        self._restore_total_blocks = 0
        self._restore_done_frames = 0
        self._restore_last_progress = 0.0

        shell = ttk.Frame(overlay, style="Card.TFrame", padding=(20, 18))
        shell.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.80, relheight=0.78)
        self.restore_shell = shell

        header = ttk.Frame(shell, style="Card.TFrame")
        header.pack(fill="x", pady=(0, 14))
        title_area = ttk.Frame(header, style="Card.TFrame")
        title_area.pack(side="left", fill="x", expand=True)
        ttk.Label(title_area, text="HDXF → HDF5 CONVERTER", style="InspectorTitle.TLabel").pack(anchor="w")
        ttk.Label(
            title_area,
            text="Restore the original master file and external HDF5 data-file layout",
            style="ReadoutMuted.TLabel",
        ).pack(anchor="w", pady=(2, 0))
        ttk.Button(
            header, text="Close", style="Secondary.TButton",
            command=self._close_restore_tool,
        ).pack(side="right")

        form = ttk.Frame(shell, style="Card.TFrame")
        form.pack(fill="x")
        form.grid_columnconfigure(1, weight=1)

        self.restore_input_var = tk.StringVar()
        self.restore_output_var = tk.StringVar()
        self.restore_overwrite_var = tk.BooleanVar(value=True)
        self.restore_verify_var = tk.BooleanVar(value=True)
        self.restore_allow_missing_calibration_var = tk.BooleanVar(value=False)
        self.restore_save_log_var = tk.BooleanVar(value=False)
        self.restore_log_path_var = tk.StringVar()
        self.restore_status_var = tk.StringVar(value="Ready")
        self.restore_stage_var = tk.StringVar(value="Ready to restore")
        self.restore_activity_var = tk.StringVar(
            value="Select an HDXF archive and output folder, then press Start Restoration."
        )
        self.restore_progress_var = tk.DoubleVar(value=0.0)
        self.restore_progress_text_var = tk.StringVar(value="0%")
        self.restore_elapsed_var = tk.StringVar(value="Elapsed  —")
        self.restore_work_var = tk.StringVar(value="Frames  —")
        self.restore_output_info_var = tk.StringVar(value="Files  —")

        def add_path_row(row: int, label: str, variable: tk.StringVar, command) -> None:
            ttk.Label(form, text=label, style="FieldLabel.TLabel").grid(
                row=row, column=0, sticky="w", padx=(0, 10), pady=5
            )
            ttk.Entry(form, textvariable=variable, style="Modern.TEntry").grid(
                row=row, column=1, sticky="ew", pady=5
            )
            ttk.Button(form, text="Browse…", style="Secondary.TButton", command=command).grid(
                row=row, column=2, padx=(8, 0), pady=5
            )

        add_path_row(0, "INPUT HDXF", self.restore_input_var, self._browse_restore_input)
        add_path_row(1, "OUTPUT FOLDER", self.restore_output_var, self._browse_restore_output)

        template_dir = self._legacy_template_dir()
        ttk.Label(form, text="MASTER TEMPLATES", style="FieldLabel.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 10), pady=5
        )
        ttk.Label(
            form,
            text=str(template_dir),
            style="ReadoutMuted.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=2, column=1, columnspan=2, sticky="w", pady=5)
        ttk.Label(
            form,
            text=(
                "Exact restore reads the master filename and SHA-256 from HDXF, "
                "finds that original master in this template directory, copies it "
                "byte-for-byte, and rebuilds the external HDF5 files. No Albula "
                "compatibility rewrite is applied."
            ),
            style="ReadoutMuted.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=3, column=1, columnspan=2, sticky="w", pady=(0, 5))

        options = ttk.Frame(form, style="Card.TFrame")
        options.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        ttk.Checkbutton(
            options, text="Overwrite existing restored files",
            variable=self.restore_overwrite_var, style="Modern.TCheckbutton",
        ).pack(side="left")
        ttk.Checkbutton(
            options, text="Verify every restored frame",
            variable=self.restore_verify_var, style="Modern.TCheckbutton",
        ).pack(side="left", padx=(18, 0))
        ttk.Label(
            options,
            text="Exact mode requires complete external Dataset coverage",
            style="ReadoutMuted.TLabel",
        ).pack(side="left", padx=(18, 0))

        log_options = ttk.Frame(form, style="Card.TFrame")
        log_options.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(2, 2))
        self.restore_save_log_check = ttk.Checkbutton(
            log_options, text="Save detailed log",
            variable=self.restore_save_log_var,
            style="Modern.TCheckbutton",
            command=self._toggle_restore_log_controls,
        )
        self.restore_save_log_check.pack(side="left")

        ttk.Label(form, text="LOG FILE", style="FieldLabel.TLabel").grid(
            row=6, column=0, sticky="w", padx=(0, 10), pady=5
        )
        self.restore_log_path_entry = ttk.Entry(
            form, textvariable=self.restore_log_path_var, style="Modern.TEntry"
        )
        self.restore_log_path_entry.grid(row=6, column=1, sticky="ew", pady=5)
        self.restore_log_browse_button = ttk.Button(
            form, text="Browse…", style="Secondary.TButton",
            command=self._browse_restore_log,
        )
        self.restore_log_browse_button.grid(row=6, column=2, padx=(8, 0), pady=5)

        progress_card = ttk.Frame(shell, style="CardAlt.TFrame", padding=(14, 12))
        progress_card.pack(fill="x", pady=(16, 10))
        progress_header = ttk.Frame(progress_card, style="CardAlt.TFrame")
        progress_header.pack(fill="x")
        ttk.Label(progress_header, textvariable=self.restore_stage_var, style="CardTitle.TLabel").pack(side="left")
        ttk.Label(
            progress_header, textvariable=self.restore_progress_text_var,
            style="ReadoutValue.TLabel",
        ).pack(side="right")
        ttk.Label(
            progress_card, textvariable=self.restore_activity_var,
            style="ReadoutMuted.TLabel", wraplength=900, justify="left",
        ).pack(fill="x", pady=(5, 10))
        self.restore_progressbar = ttk.Progressbar(
            progress_card, orient="horizontal", mode="determinate",
            maximum=100.0, variable=self.restore_progress_var,
            style="Converter.Horizontal.TProgressbar",
        )
        self.restore_progressbar.pack(fill="x", ipady=4)

        metrics = ttk.Frame(progress_card, style="CardAlt.TFrame")
        metrics.pack(fill="x", pady=(9, 0))
        metrics.grid_columnconfigure((0, 1, 2), weight=1)
        ttk.Label(metrics, textvariable=self.restore_work_var, style="ReadoutMuted.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(metrics, textvariable=self.restore_elapsed_var, style="ReadoutMuted.TLabel").grid(
            row=0, column=1
        )
        ttk.Label(metrics, textvariable=self.restore_output_info_var, style="ReadoutMuted.TLabel").grid(
            row=0, column=2, sticky="e"
        )

        actions = ttk.Frame(shell, style="Card.TFrame")
        actions.pack(fill="x", pady=(2, 0))
        self.restore_start_button = ttk.Button(
            actions, text="Start Restoration", style="Accent.TButton",
            command=self._start_restore_process,
        )
        self.restore_start_button.pack(side="left")
        self.restore_stop_button = ttk.Button(
            actions, text="Stop", style="Secondary.TButton",
            command=self._stop_restore_process, state="disabled",
        )
        self.restore_stop_button.pack(side="left", padx=(8, 0))
        self.restore_details_button = ttk.Button(
            actions, text="Show details", style="Secondary.TButton",
            command=self._toggle_restore_details,
        )
        self.restore_details_button.pack(side="left", padx=(8, 0))
        ttk.Label(actions, textvariable=self.restore_status_var, style="ReadoutMuted.TLabel").pack(side="right")

        self.restore_details_shell = ttk.Frame(shell, style="Card.TFrame")
        ttk.Separator(self.restore_details_shell, orient="horizontal", style="Card.TSeparator").pack(
            fill="x", pady=(8, 8)
        )
        details_header = ttk.Frame(self.restore_details_shell, style="Card.TFrame")
        details_header.pack(fill="x")
        ttk.Label(details_header, text="DETAILED LOG", style="FieldLabel.TLabel").pack(side="left")
        ttk.Label(
            details_header,
            text="Normally hidden; useful for diagnosing filter or filesystem errors",
            style="ReadoutMuted.TLabel",
        ).pack(side="right")
        log_frame = tk.Frame(self.restore_details_shell, background=UI_CARD_ALT)
        log_frame.pack(fill="both", expand=True, pady=(6, 0))
        self.restore_log_text = tk.Text(
            log_frame,
            background=UI_CARD_ALT,
            foreground=UI_TEXT,
            insertbackground=UI_TEXT,
            selectbackground=UI_ACCENT_DARK,
            selectforeground=UI_WHITE,
            relief="flat",
            borderwidth=0,
            wrap="none",
            font=("Consolas", 9),
        )
        log_scroll = ttk.Scrollbar(
            log_frame, orient="vertical", command=self.restore_log_text.yview,
            style="Modern.Vertical.TScrollbar",
        )
        self.restore_log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")
        self.restore_log_text.pack(side="left", fill="both", expand=True)
        self.restore_log_text.insert("end", f"Converter script: {script}\n")
        self.restore_log_text.configure(state="disabled")

        self.restore_input_var.trace_add("write", lambda *_args: self._on_restore_input_changed())
        self.restore_output_var.trace_add("write", lambda *_args: self._on_restore_output_changed())
        self._toggle_restore_log_controls()

    def _browse_restore_input(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self.root,
            title="Select HDXF archive",
            filetypes=[("HDXF archives", "*.hdxf"), ("All files", "*.*")],
        )
        if filename:
            self.restore_input_var.set(str(Path(filename)))

    def _browse_restore_output(self) -> None:
        current = self.restore_output_var.get().strip()
        initial = Path(current) if current else Path.cwd()
        folder = filedialog.askdirectory(
            parent=self.root,
            title="Select restored HDF5 output folder",
            initialdir=str(initial if initial.is_dir() else initial.parent),
            mustexist=False,
        )
        if folder:
            self.restore_output_var.set(str(Path(folder)))

    def _browse_restore_log(self) -> None:
        current = self.restore_log_path_var.get().strip()
        initial = Path(current) if current else Path.cwd() / "hdxf_to_hdf5.restore.log"
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save detailed restoration log",
            initialdir=str(initial.parent),
            initialfile=initial.name,
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
        )
        if filename:
            self.restore_log_path_var.set(str(Path(filename)))

    def _on_restore_input_changed(self) -> None:
        input_text = self.restore_input_var.get().strip()
        if input_text and not self.restore_output_var.get().strip():
            source = Path(input_text)
            self.restore_output_var.set(str(source.with_name(source.stem + "_restored_hdf5")))

    def _on_restore_output_changed(self) -> None:
        if not self.restore_save_log_var.get():
            return
        output_text = self.restore_output_var.get().strip()
        if output_text and not self.restore_log_path_var.get().strip():
            output = Path(output_text)
            self.restore_log_path_var.set(str(output.with_name(output.name + ".restore.log")))

    def _toggle_restore_log_controls(self) -> None:
        enabled = bool(self.restore_save_log_var.get())
        if enabled and not self.restore_log_path_var.get().strip():
            self._on_restore_output_changed()
        state = "normal" if enabled else "disabled"
        for widget_name in ("restore_log_path_entry", "restore_log_browse_button"):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                try:
                    widget.configure(state=state)
                except tk.TclError:
                    pass

    def _toggle_restore_details(self) -> None:
        if self._restore_details_visible:
            self.restore_details_shell.pack_forget()
            self.restore_details_button.configure(text="Show details")
            try:
                self.restore_shell.place_configure(relheight=0.78)
            except tk.TclError:
                pass
            self._restore_details_visible = False
        else:
            try:
                self.restore_shell.place_configure(relheight=0.90)
            except tk.TclError:
                pass
            self.restore_details_shell.pack(fill="both", expand=True, pady=(6, 0))
            self.restore_details_button.configure(text="Hide details")
            self._restore_details_visible = True

    def _build_restore_command(self, *, validate: bool) -> list[str]:
        script = self._restore_script_path()
        input_text = self.restore_input_var.get().strip()
        output_text = self.restore_output_var.get().strip()
        template_dir = self._legacy_template_dir()

        if validate:
            if not script.is_file():
                raise HDXFViewerError(f"converter script not found: {script}")
            if not input_text:
                raise HDXFViewerError("select an input HDXF archive")
            input_path = Path(input_text)
            if not input_path.is_file():
                raise HDXFViewerError(
                    f"input HDXF file does not exist: {input_path}"
                )
            if not output_text:
                raise HDXFViewerError("select an output folder")
            if not template_dir.is_dir():
                raise HDXFViewerError(
                    "exact HDF5 restoration needs the legacy template directory:\n"
                    f"{template_dir}\n\n"
                    "Put the original master HDF5 in that directory using its "
                    "original filename."
                )
            if (
                self.restore_save_log_var.get()
                and not self.restore_log_path_var.get().strip()
            ):
                raise HDXFViewerError(
                    "select a log file or turn off Save detailed log"
                )

        cmd = [sys.executable, "-u", str(script)]
        if input_text:
            cmd.append(input_text)
        if output_text:
            cmd.extend(["-o", output_text])

        cmd.extend([
            "--restore-mode", "exact",
            "--legacy-template-dir", str(template_dir),
            "--progress-every", "1",
        ])

        if self.restore_overwrite_var.get():
            cmd.append("--overwrite")
        if self.restore_verify_var.get():
            cmd.append("--verify")
        return cmd

    @staticmethod
    def _estimate_hdxf_restore_work(input_path: Path) -> tuple[int, int, int]:
        try:
            with zipfile.ZipFile(input_path, "r") as archive:
                manifest = json.loads(archive.read(MANIFEST_PATH))
                frames = manifest.get("detector_archive", {}).get("frames", {})
                total_frames = int(frames.get("frame_count", 0)) if isinstance(frames, dict) else 0
                total_blocks = 0
                if isinstance(frames, dict):
                    blocks = frames.get("blocks")
                    if isinstance(blocks, list):
                        total_blocks = len(blocks)
                    else:
                        index = frames.get("block_index")
                        if isinstance(index, dict):
                            total_blocks = int(index.get("record_count", 0))
                legacy = manifest.get("legacy_hdf5")
                if isinstance(legacy, dict):
                    external_files = legacy.get("external_files")
                    file_count = (
                        1 + len(external_files)
                        if isinstance(external_files, list)
                        else 1
                    )
                else:
                    source_files = manifest.get("source", {}).get("files", [])
                    file_count = (
                        len(source_files)
                        if isinstance(source_files, list)
                        else 0
                    )
                return total_frames, total_blocks, file_count
        except Exception:
            return 0, 0, 0

    def _append_restore_log(self, text: str) -> None:
        log_file = getattr(self, "_restore_log_file", None)
        if log_file is not None:
            try:
                log_file.write(text)
                log_file.flush()
            except Exception:
                pass
        widget = getattr(self, "restore_log_text", None)
        if widget is None:
            return
        try:
            if not widget.winfo_exists():
                return
            widget.configure(state="normal")
            widget.insert("end", text)
            widget.see("end")
            widget.configure(state="disabled")
        except tk.TclError:
            pass

    def _set_restore_progress(self, value: float, *, text: str | None = None) -> None:
        value = max(self._restore_last_progress, min(100.0, float(value)))
        self._restore_last_progress = value
        try:
            self.restore_progressbar.stop()
            self.restore_progressbar.configure(mode="determinate")
        except tk.TclError:
            pass
        self.restore_progress_var.set(value)
        self.restore_progress_text_var.set(text if text is not None else f"{value:.0f}%")

    def _update_restore_elapsed(self) -> None:
        started = self._restore_start_time
        if started is None:
            return
        elapsed = max(0.0, time.monotonic() - started)
        if elapsed < 60:
            label = f"Elapsed  {elapsed:.0f} s"
        else:
            minutes, seconds = divmod(int(elapsed), 60)
            label = f"Elapsed  {minutes:d}m {seconds:02d}s"
        self.restore_elapsed_var.set(label)

    def _handle_restore_output_line(self, raw_line: str) -> None:
        line = raw_line.strip()
        if not line:
            return
        block_match = re.search(
            r"\[RESTORE\]\s+frame block\s+(\d+)/(\d+)\s+\((\d+)/(\d+)\s+frames,\s*([0-9.]+)%\)",
            line,
        )
        if block_match:
            block_no = int(block_match.group(1))
            block_total = int(block_match.group(2))
            done_frames = int(block_match.group(3))
            total_frames = int(block_match.group(4))
            percent = float(block_match.group(5))
            # Reserve 5% for structure creation and 5% for final links/checks.
            self._set_restore_progress(5.0 + 90.0 * percent / 100.0)
            self.restore_stage_var.set("Restoring detector frames")
            self.restore_activity_var.set(
                f"Decoded and wrote {done_frames:,} of {total_frames:,} frames "
                f"from block {block_no:,} of {block_total:,}."
            )
            self.restore_work_var.set(f"Frames  {done_frames:,} / {total_frames:,}")
            return

        lower = line.lower()
        if line.startswith("[START]"):
            self.restore_stage_var.set("Opening HDXF archive")
            self.restore_activity_var.set("Reading the manifest, frame index and source-file layout.")
            self._set_restore_progress(1.0)
        elif "[stage] creating hdf5 files" in lower:
            self.restore_stage_var.set("Creating HDF5 structure")
            self.restore_activity_var.set("Creating the master file, external files, Groups and Datasets.")
            self._set_restore_progress(3.0)
        elif "[stage] restoring preserved hdf5 datasets" in lower:
            self.restore_stage_var.set("Restoring metadata datasets")
            self.restore_activity_var.set("Writing preserved detector metadata and Calibration datasets.")
            self._set_restore_progress(5.0)
        elif "[stage] decoding detector frame blocks" in lower:
            self.restore_stage_var.set("Restoring detector frames")
            self.restore_activity_var.set("Decoding HDXFB blocks and writing HDF5 frame chunks.")
            self._set_restore_progress(6.0)
        elif "[stage] restoring hdf5 links" in lower:
            self.restore_stage_var.set("Restoring HDF5 links")
            self.restore_activity_var.set("Recreating ExternalLink, SoftLink and HardLink entries.")
            self._set_restore_progress(96.0)
        elif "[stage] resolving hdf5 references" in lower:
            self.restore_stage_var.set("Resolving HDF5 references")
            self.restore_activity_var.set("Connecting preserved object and region references.")
            self._set_restore_progress(97.0)
        elif "[stage] verifying restored hdf5 structure" in lower:
            self.restore_stage_var.set("Verifying restored files")
            self.restore_activity_var.set("Reading back the restored HDF5 structure and links.")
            self._set_restore_progress(98.0)
        elif line.startswith("[INFO] output files:"):
            match = re.search(r"output files:\s*(\d+)", line, flags=re.IGNORECASE)
            if match:
                self.restore_output_info_var.set(f"Files  {int(match.group(1))}")
        elif line.startswith("[OUT master]"):
            self.restore_output_info_var.set("Master restored")
            self._set_restore_progress(99.0)
        elif line.startswith("[SUMMARY]"):
            match = re.search(r"frames=(\d+).*?warnings=(\d+)", line)
            if match:
                self.restore_work_var.set(f"Frames  {int(match.group(1)):,}")
                self.restore_output_info_var.set(f"Warnings  {int(match.group(2))}")
        elif "[exact] copying byte-identical master template" in lower:
            self.restore_stage_var.set("Copying original master")
            self.restore_activity_var.set(
                "The SHA-validated original master is being copied byte-for-byte."
            )
            self._set_restore_progress(3.0)
        elif "[exact] rebuilding external hdf5 structure" in lower:
            self.restore_stage_var.set("Rebuilding external HDF5 structure")
            self.restore_activity_var.set(
                "Recreating the original Groups, Datasets, attributes, links and storage layout."
            )
            self._set_restore_progress(max(self._restore_last_progress, 5.0))
        elif "[stage] restoring exact external dataset values" in lower:
            self.restore_stage_var.set("Restoring auxiliary Dataset values")
            self.restore_activity_var.set(
                "Writing MX, azimuthal-integration, diagnostic and other external Dataset values."
            )
            self._set_restore_progress(max(self._restore_last_progress, 8.0))
        elif "[stage] decoding detector frame blocks" in lower:
            self.restore_stage_var.set("Restoring detector frames")
            self.restore_activity_var.set(
                "Decoding HDXF frame blocks into their original physical HDF5 Datasets."
            )
            self._set_restore_progress(max(self._restore_last_progress, 10.0))
        elif "[stage] resolving exact external hdf5 references" in lower:
            self.restore_stage_var.set("Resolving HDF5 references")
            self.restore_activity_var.set(
                "Restoring deferred HDF5 object and region references."
            )
            self._set_restore_progress(max(self._restore_last_progress, 96.0))
        elif "[stage] verifying reconstructed hdf5 structure" in lower:
            self.restore_stage_var.set("Verifying exact HDF5 structure")
            self.restore_activity_var.set(
                "Checking master SHA-256, external links, Dataset shapes, dtypes, chunks, filters and attributes."
            )
            self._set_restore_progress(max(self._restore_last_progress, 98.0))
        elif "[result] pass" in lower:
            self.restore_stage_var.set("Exact restoration completed")
            self.restore_activity_var.set(
                "The original master and external HDF5 structure were restored and verified."
            )
            self._set_restore_progress(100.0, text="100%")
        elif "[result] fail" in lower or "traceback" in lower:
            self.restore_stage_var.set("Restoration error")
            self.restore_activity_var.set("The reverse converter reported an error. Open details for diagnostics.")

    def _start_restore_process(self) -> None:
        if self._restore_process is not None and self._restore_process.poll() is None:
            return
        try:
            cmd = self._build_restore_command(validate=True)
        except Exception as exc:
            messagebox.showerror("Cannot start restoration", str(exc), parent=self.root)
            return

        self._restore_stop_requested = False
        self._restore_last_progress = 0.0
        self.restore_progress_var.set(0.0)
        self.restore_progress_text_var.set("0%")
        self.restore_stage_var.set("Preparing restoration")
        self.restore_activity_var.set("Inspecting the HDXF frame index and source-file layout.")
        self.restore_status_var.set("Starting…")
        self.restore_elapsed_var.set("Elapsed  0 s")
        self.restore_work_var.set("Frames  scanning…")
        self.restore_output_info_var.set("Files  scanning…")
        self._restore_start_time = time.monotonic()

        self.restore_log_text.configure(state="normal")
        self.restore_log_text.delete("1.0", "end")
        self.restore_log_text.configure(state="disabled")

        self._close_restore_log_file()
        if self.restore_save_log_var.get():
            try:
                log_path = Path(self.restore_log_path_var.get().strip())
                log_path.parent.mkdir(parents=True, exist_ok=True)
                self._restore_log_file = log_path.open("w", encoding="utf-8", newline="")
            except Exception as exc:
                messagebox.showerror("Cannot create log file", f"{type(exc).__name__}: {exc}", parent=self.root)
                return

        input_path = Path(self.restore_input_var.get().strip())
        total_frames, total_blocks, file_count = self._estimate_hdxf_restore_work(input_path)
        self._restore_total_frames = total_frames
        self._restore_total_blocks = total_blocks
        self.restore_work_var.set(f"Frames  {total_frames:,}" if total_frames else "Frames  —")
        self.restore_output_info_var.set(f"Files  {file_count}" if file_count else "Files  —")

        self._append_restore_log("Command:\n" + self._format_command_for_display(cmd) + "\n\n")
        restore_env = os.environ.copy()
        restore_env["PYTHONUNBUFFERED"] = "1"
        restore_env["PYTHONUTF8"] = "1"
        restore_env["PYTHONIOENCODING"] = "utf-8:replace"
        popen_kwargs: dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
            "cwd": str(self._restore_script_path().parent),
            "env": restore_env,
        }
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(cmd, **popen_kwargs)
        except Exception as exc:
            self._close_restore_log_file()
            messagebox.showerror("Cannot start restoration", f"{type(exc).__name__}: {exc}", parent=self.root)
            return

        self._restore_process = process
        self._restore_run_serial += 1
        serial = self._restore_run_serial
        self.restore_start_button.configure(state="disabled")
        self.restore_stop_button.configure(state="normal")
        self.restore_status_var.set("Running")

        def reader() -> None:
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    self._restore_queue.put((serial, "line", line))
                return_code = process.wait()
                self._restore_queue.put((serial, "done", return_code))
            except Exception as exc:
                self._restore_queue.put((serial, "reader_error", exc))

        self._restore_reader = threading.Thread(target=reader, daemon=True)
        self._restore_reader.start()
        self._poll_restore_queue()

    def _poll_restore_queue(self) -> None:
        self._restore_poll_after_id = None
        serial = self._restore_run_serial
        try:
            while True:
                item_serial, kind, payload = self._restore_queue.get_nowait()
                if item_serial != serial:
                    continue
                if kind == "line":
                    text = str(payload)
                    self._append_restore_log(text)
                    self._handle_restore_output_line(text)
                elif kind == "done":
                    self._finish_restore_process(int(payload))
                    return
                elif kind == "reader_error":
                    self._append_restore_log(f"\nReader error: {payload}\n")
                    self._finish_restore_process(1)
                    return
        except queue.Empty:
            pass
        self._update_restore_elapsed()
        process = self._restore_process
        if process is not None and process.poll() is None and self._restore_panel is not None:
            try:
                if self._restore_panel.winfo_exists():
                    self._restore_poll_after_id = self.root.after(80, self._poll_restore_queue)
            except tk.TclError:
                pass

    def _finish_restore_process(self, return_code: int) -> None:
        self._restore_process = None
        self.restore_start_button.configure(state="normal")
        self.restore_stop_button.configure(state="disabled")
        self._update_restore_elapsed()
        self._close_restore_log_file()
        if self._restore_stop_requested:
            self.restore_stage_var.set("Restoration stopped")
            self.restore_activity_var.set("The restoration process was stopped before completion.")
            self.restore_status_var.set("Stopped")
            return
        if return_code == 0:
            self._set_restore_progress(100.0, text="100%")
            self.restore_stage_var.set("Restoration complete")
            self.restore_activity_var.set("The HDF5 master and external files were restored successfully.")
            self.restore_status_var.set("Completed")
        else:
            self.restore_stage_var.set("Restoration failed")
            self.restore_activity_var.set("The converter exited with an error. Review the detailed log.")
            self.restore_status_var.set(f"Failed ({return_code})")
            if not self._restore_details_visible:
                self._toggle_restore_details()

    def _stop_restore_process(self) -> None:
        process = self._restore_process
        if process is None or process.poll() is not None:
            return
        self._restore_stop_requested = True
        self.restore_status_var.set("Stopping…")
        self.restore_activity_var.set("Stopping the reverse converter; partially restored files may remain.")
        try:
            process.terminate()
        except Exception:
            pass

    def _close_restore_log_file(self) -> None:
        log_file = getattr(self, "_restore_log_file", None)
        self._restore_log_file = None
        if log_file is not None:
            try:
                log_file.close()
            except Exception:
                pass

    def _close_restore_tool(self) -> None:
        process = self._restore_process
        if process is not None and process.poll() is None:
            close = messagebox.askyesno(
                "Restoration is running",
                "Stop the running restoration and close the tool?",
                parent=self.root,
            )
            if not close:
                return
            self._stop_restore_process()
            self._restore_run_serial += 1
            self._restore_process = None
        if self._restore_poll_after_id is not None:
            try:
                self.root.after_cancel(self._restore_poll_after_id)
            except tk.TclError:
                pass
            self._restore_poll_after_id = None
        self._close_restore_log_file()
        panel = self._restore_panel
        self._restore_panel = None
        if panel is not None and panel.winfo_exists():
            panel.destroy()


    def _sum_script_path(self) -> Path:
        """Return the sibling HDXF frame-combine script."""
        return Path(__file__).resolve().with_name("hdxf_sum.py")

    def _subtract_script_path(self) -> Path:
        """Return the sibling HDXF subtract/filter script."""
        return Path(__file__).resolve().with_name("hdxf_subtract.py")

    def _pixelop_script_path(self) -> Path:
        """Return the sibling HDXF scalar pixel-operation script."""
        return Path(__file__).resolve().with_name("hdxf_pixelop.py")

    def open_hdxf_sum_tool(self) -> None:
        self._open_hdxf_processing_tool("sum")

    def open_hdxf_subtract_tool(self) -> None:
        self._open_hdxf_processing_tool("subtract")

    def open_hdxf_pixelop_tool(self) -> None:
        self._open_hdxf_processing_tool("pixelop")

    def _processing_script_path(self, kind: str | None = None) -> Path:
        selected = kind or self._processing_kind or "sum"
        if selected == "subtract":
            return self._subtract_script_path()
        if selected == "pixelop":
            return self._pixelop_script_path()
        return self._sum_script_path()

    def _active_hdxf_path(self) -> Path | None:
        view = self.active_view
        if view is None or view.archive is None:
            return None
        if getattr(view.archive, "file_format", "") != "HDXF":
            return None
        return view.archive.path

    def _open_hdxf_processing_tool(self, kind: str) -> None:
        if kind not in ("sum", "subtract", "pixelop"):
            raise ValueError(kind)
        script = self._processing_script_path(kind)
        if not script.is_file():
            expected = {"sum": "hdxf_sum.py", "subtract": "hdxf_subtract.py", "pixelop": "hdxf_pixelop.py"}[kind]
            messagebox.showerror(
                "Processing tool not found",
                f"{expected} was not found next to hdxf_viewer.py.\n\nExpected location:\n{script}",
                parent=self.root,
            )
            return

        if self._processing_panel is not None and self._processing_panel.winfo_exists():
            process = self._processing_process
            if self._processing_kind == kind:
                self._processing_panel.lift()
                return
            if process is not None and process.poll() is None:
                messagebox.showinfo(
                    "HDXF processing is running",
                    "Stop the current processing job before switching tools.",
                    parent=self.root,
                )
                self._processing_panel.lift()
                return
            self._close_processing_tool(force=True)

        self._processing_kind = kind
        self._processing_output_path = None
        self._processing_output_auto = True
        self._processing_stop_requested = False
        self._processing_last_progress = 0.0

        overlay = tk.Frame(self.root, background="#02050B", highlightthickness=0, borderwidth=0)
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        overlay.lift()
        self._processing_panel = overlay

        shell = ttk.Frame(overlay, style="Card.TFrame", padding=(20, 17))
        shell.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.84, relheight=0.84)
        self.processing_shell = shell

        header = ttk.Frame(shell, style="Card.TFrame")
        header.pack(fill="x", pady=(0, 12))
        title_area = ttk.Frame(header, style="Card.TFrame")
        title_area.pack(side="left", fill="x", expand=True)
        title = {
            "sum": "HDXF SUM / MEAN",
            "subtract": "HDXF SUBTRACT / FILTER",
            "pixelop": "HDXF PIXEL OPERATION",
        }[kind]
        subtitle = {
            "sum": "Combine selected HDXF frames into one derived frame",
            "subtract": "Apply background subtraction, thresholds and ROI protection directly to HDXF frames",
            "pixelop": "Add, subtract, multiply or divide selected pixels directly in HDXF frames",
        }[kind]
        ttk.Label(title_area, text=title, style="InspectorTitle.TLabel").pack(anchor="w")
        ttk.Label(title_area, text=subtitle, style="CardSubtitle.TLabel").pack(anchor="w", pady=(2, 0))

        switch = ttk.Frame(header, style="Card.TFrame")
        switch.pack(side="right", padx=(12, 10))
        ttk.Button(
            switch, text="Sum / Mean", style="Accent.TButton" if kind == "sum" else "Secondary.TButton",
            command=self.open_hdxf_sum_tool,
        ).pack(side="left")
        ttk.Button(
            switch, text="Subtract / Filter", style="Accent.TButton" if kind == "subtract" else "Secondary.TButton",
            command=self.open_hdxf_subtract_tool,
        ).pack(side="left", padx=(6, 0))
        ttk.Button(
            switch, text="Pixel Op", style="Accent.TButton" if kind == "pixelop" else "Secondary.TButton",
            command=self.open_hdxf_pixelop_tool,
        ).pack(side="left", padx=(6, 0))
        ttk.Button(header, text="Close", style="Secondary.TButton", command=self._close_processing_tool).pack(side="right")

        self.processing_input_var = tk.StringVar()
        self.processing_output_var = tk.StringVar()
        self.processing_frame_from_var = tk.StringVar()
        self.processing_frame_to_var = tk.StringVar()
        self.processing_select_var = tk.StringVar()
        self.processing_overwrite_var = tk.BooleanVar(value=True)
        self.processing_open_after_var = tk.BooleanVar(value=True)
        self.processing_status_var = tk.StringVar(value="Ready")
        self.processing_stage_var = tk.StringVar(value="Ready")
        self.processing_activity_var = tk.StringVar(value="Choose processing settings, then press Start.")
        self.processing_progress_var = tk.DoubleVar(value=0.0)
        self.processing_progress_text_var = tk.StringVar(value="0%")
        self.processing_elapsed_var = tk.StringVar(value="Elapsed  —")
        self.processing_work_var = tk.StringVar(value="Frames  —")

        # Sum/Mean controls.
        self.processing_operation_var = tk.StringVar(value="mean")
        self.processing_dtype_var = tk.StringVar(value="auto")
        self.processing_keep_azint_var = tk.BooleanVar(value=False)

        # Subtract/Filter controls.
        self.processing_mode_var = tk.StringVar(value="Background subtraction")
        self.processing_threshold_var = tk.StringVar()
        self.processing_background_var = tk.StringVar()
        self.processing_background_frame_var = tk.StringVar(value="1")
        self.processing_remove_low_var = tk.StringVar()
        self.processing_remove_high_var = tk.StringVar()
        self.processing_upper_var = tk.StringVar()
        self.processing_fill_var = tk.StringVar(value="0")
        self.processing_protect_var = tk.StringVar()
        self.processing_frame_transform_var = tk.StringVar(value="delta")
        self.processing_block_frames_var = tk.StringVar(value="8")
        self.processing_zstd_level_var = tk.StringVar(value="3")
        self.processing_verify_var = tk.BooleanVar(value=True)

        # Scalar Pixel Operation controls. Keep the CLI spelling used by
        # hdxf_pixelop.py (including --multiple / --devide) in the command
        # builder while presenting normal English labels in the UI.
        self.processing_pixelop_operation_var = tk.StringVar(value="Add")
        self.processing_pixelop_value_var = tk.StringVar(value="0")
        self.processing_pixelop_roi_var = tk.StringVar()

        form = ttk.Frame(shell, style="Card.TFrame")
        form.pack(fill="x")
        form.grid_columnconfigure(1, weight=1)

        def path_row(row: int, label: str, variable: tk.StringVar, command, *, output: bool = False) -> ttk.Entry:
            ttk.Label(form, text=label, style="FieldLabel.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=5)
            entry = ttk.Entry(form, textvariable=variable, style="Modern.TEntry")
            entry.grid(row=row, column=1, sticky="ew", pady=5)
            ttk.Button(form, text="Browse…", style="Secondary.TButton", command=command).grid(row=row, column=2, padx=(8, 0), pady=5)
            if output:
                entry.bind("<KeyRelease>", lambda _e: setattr(self, "_processing_output_auto", False))
            return entry

        path_row(0, "INPUT HDXF", self.processing_input_var, self._browse_processing_input)
        path_row(
            1,
            "OUTPUT HDXF" if kind == "sum" else "OUTPUT FOLDER",
            self.processing_output_var,
            self._browse_processing_output,
            output=True,
        )

        select_card = ttk.Frame(form, style="CardAlt.TFrame", padding=(10, 8))
        select_card.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 6))
        select_card.grid_columnconfigure(5, weight=1)
        ttk.Label(select_card, text="FRAME SELECTION", style="FieldLabel.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Label(select_card, text="From", style="ReadoutMuted.TLabel").grid(row=0, column=1, padx=(0, 4))
        ttk.Entry(select_card, textvariable=self.processing_frame_from_var, width=8, style="Modern.TEntry").grid(row=0, column=2, padx=(0, 8))
        ttk.Label(select_card, text="To", style="ReadoutMuted.TLabel").grid(row=0, column=3, padx=(0, 4))
        ttk.Entry(select_card, textvariable=self.processing_frame_to_var, width=8, style="Modern.TEntry").grid(row=0, column=4, padx=(0, 12))
        ttk.Entry(select_card, textvariable=self.processing_select_var, style="Modern.TEntry").grid(row=0, column=5, sticky="ew")
        ttk.Label(
            select_card,
            text="Comma-separated frame numbers override From/To; leave all blank for every frame.",
            style="CardSubtitle.TLabel",
        ).grid(row=1, column=1, columnspan=5, sticky="w", pady=(5, 0))

        if kind == "sum":
            options = ttk.Frame(form, style="Card.TFrame")
            options.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(5, 4))
            options.grid_columnconfigure(1, weight=1)
            options.grid_columnconfigure(3, weight=1)
            ttk.Label(options, text="OPERATION", style="FieldLabel.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
            op = ttk.Combobox(
                options, state="readonly", textvariable=self.processing_operation_var,
                values=("sum", "mean", "exposure-normalized"), style="Modern.TCombobox", width=22,
            )
            op.grid(row=0, column=1, sticky="ew", padx=(0, 18))
            op.bind("<<ComboboxSelected>>", self._processing_sum_operation_changed)
            ttk.Label(options, text="OUTPUT DTYPE", style="FieldLabel.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 8))
            ttk.Combobox(
                options, state="readonly", textvariable=self.processing_dtype_var,
                values=("auto", "source", "int16", "int32", "int64", "uint16", "uint32", "uint64", "float32", "float64"),
                style="Modern.TCombobox", width=14,
            ).grid(row=0, column=3, sticky="ew")
            ttk.Checkbutton(
                options, text="Keep /entry/azint metadata", variable=self.processing_keep_azint_var,
                style="Modern.TCheckbutton",
            ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))
        elif kind == "subtract":
            options = ttk.Frame(form, style="Card.TFrame")
            options.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(5, 4))
            for column in (1, 3):
                options.grid_columnconfigure(column, weight=1)

            ttk.Label(options, text="MODE", style="FieldLabel.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
            mode = ttk.Combobox(
                options, state="readonly", textvariable=self.processing_mode_var,
                values=("Background subtraction", "Lower threshold", "Remove inclusive range", "Copy only"),
                style="Modern.TCombobox",
            )
            mode.grid(row=0, column=1, sticky="ew", padx=(0, 18), pady=4)
            mode.bind("<<ComboboxSelected>>", self._processing_mode_changed)

            ttk.Label(options, text="FILL VALUE", style="FieldLabel.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 8), pady=4)
            ttk.Entry(options, textvariable=self.processing_fill_var, style="Modern.TEntry").grid(row=0, column=3, sticky="ew", pady=4)

            ttk.Label(options, text="LOWER THRESHOLD", style="FieldLabel.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
            self.processing_threshold_entry = ttk.Entry(options, textvariable=self.processing_threshold_var, style="Modern.TEntry")
            self.processing_threshold_entry.grid(row=1, column=1, sticky="ew", padx=(0, 18), pady=4)

            ttk.Label(options, text="UPPER THRESHOLD", style="FieldLabel.TLabel").grid(row=1, column=2, sticky="w", padx=(0, 8), pady=4)
            ttk.Entry(options, textvariable=self.processing_upper_var, style="Modern.TEntry").grid(row=1, column=3, sticky="ew", pady=4)

            ttk.Label(options, text="BACKGROUND HDXF", style="FieldLabel.TLabel").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
            background_row = ttk.Frame(options, style="Card.TFrame")
            background_row.grid(row=2, column=1, columnspan=3, sticky="ew", pady=4)
            background_row.grid_columnconfigure(0, weight=1)
            self.processing_background_entry = ttk.Entry(background_row, textvariable=self.processing_background_var, style="Modern.TEntry")
            self.processing_background_entry.grid(row=0, column=0, sticky="ew")
            self.processing_background_button = ttk.Button(background_row, text="Browse…", style="Secondary.TButton", command=self._browse_processing_background)
            self.processing_background_button.grid(row=0, column=1, padx=(7, 12))
            ttk.Label(background_row, text="Frame", style="ReadoutMuted.TLabel").grid(row=0, column=2, padx=(0, 5))
            self.processing_background_frame_entry = ttk.Entry(background_row, textvariable=self.processing_background_frame_var, width=7, style="Modern.TEntry")
            self.processing_background_frame_entry.grid(row=0, column=3)

            ttk.Label(options, text="REMOVE RANGE", style="FieldLabel.TLabel").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
            range_row = ttk.Frame(options, style="Card.TFrame")
            range_row.grid(row=3, column=1, sticky="ew", padx=(0, 18), pady=4)
            range_row.grid_columnconfigure((0, 2), weight=1)
            self.processing_remove_low_entry = ttk.Entry(range_row, textvariable=self.processing_remove_low_var, style="Modern.TEntry")
            self.processing_remove_low_entry.grid(row=0, column=0, sticky="ew")
            ttk.Label(range_row, text="to", style="ReadoutMuted.TLabel").grid(row=0, column=1, padx=5)
            self.processing_remove_high_entry = ttk.Entry(range_row, textvariable=self.processing_remove_high_var, style="Modern.TEntry")
            self.processing_remove_high_entry.grid(row=0, column=2, sticky="ew")

            ttk.Label(options, text="PROTECT ROI", style="FieldLabel.TLabel").grid(row=3, column=2, sticky="w", padx=(0, 8), pady=4)
            ttk.Entry(options, textvariable=self.processing_protect_var, style="Modern.TEntry").grid(row=3, column=3, sticky="ew", pady=4)
            ttk.Label(options, text="Format: x0,y0,x1,y1", style="CardSubtitle.TLabel").grid(row=4, column=3, sticky="w", pady=(0, 4))

            encode = ttk.Frame(options, style="CardAlt.TFrame", padding=(9, 7))
            encode.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8, 0))
            encode.grid_columnconfigure((1, 3, 5), weight=1)
            ttk.Label(encode, text="ENCODING", style="FieldLabel.TLabel").grid(row=0, column=0, padx=(0, 7))
            ttk.Combobox(
                encode, state="readonly", textvariable=self.processing_frame_transform_var,
                values=("delta", "adaptive", "auto", "raw"), style="Modern.TCombobox", width=11,
            ).grid(row=0, column=1, sticky="ew", padx=(0, 12))
            ttk.Label(encode, text="BLOCK FRAMES", style="FieldLabel.TLabel").grid(row=0, column=2, padx=(0, 7))
            ttk.Combobox(
                encode, state="readonly", textvariable=self.processing_block_frames_var,
                values=("4", "8", "16", "32", "64"), style="Modern.TCombobox", width=8,
            ).grid(row=0, column=3, sticky="ew", padx=(0, 12))
            ttk.Label(encode, text="ZSTD", style="FieldLabel.TLabel").grid(row=0, column=4, padx=(0, 7))
            ttk.Combobox(
                encode, state="readonly", textvariable=self.processing_zstd_level_var,
                values=tuple(str(i) for i in range(10)), style="Modern.TCombobox", width=6,
            ).grid(row=0, column=5, sticky="ew")
            self._processing_mode_changed()

        else:  # pixelop
            options = ttk.Frame(form, style="Card.TFrame")
            options.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(5, 4))
            options.grid_columnconfigure(1, weight=1)
            options.grid_columnconfigure(3, weight=1)

            ttk.Label(options, text="OPERATION", style="FieldLabel.TLabel").grid(
                row=0, column=0, sticky="w", padx=(0, 8), pady=4
            )
            ttk.Combobox(
                options, state="readonly", textvariable=self.processing_pixelop_operation_var,
                values=("Add", "Subtract", "Multiply", "Divide"),
                style="Modern.TCombobox", width=16,
            ).grid(row=0, column=1, sticky="ew", padx=(0, 18), pady=4)

            ttk.Label(options, text="VALUE", style="FieldLabel.TLabel").grid(
                row=0, column=2, sticky="w", padx=(0, 8), pady=4
            )
            ttk.Entry(
                options, textvariable=self.processing_pixelop_value_var, style="Modern.TEntry"
            ).grid(row=0, column=3, sticky="ew", pady=4)

            ttk.Label(options, text="ROI", style="FieldLabel.TLabel").grid(
                row=1, column=0, sticky="w", padx=(0, 8), pady=4
            )
            ttk.Entry(
                options, textvariable=self.processing_pixelop_roi_var, style="Modern.TEntry"
            ).grid(row=1, column=1, columnspan=3, sticky="ew", pady=4)
            ttk.Label(
                options,
                text="Optional: x1,y1,x2,y2 · 0-based inclusive. Pixels outside ROI are copied unchanged.",
                style="CardSubtitle.TLabel",
            ).grid(row=2, column=1, columnspan=3, sticky="w", pady=(0, 5))

            encode = ttk.Frame(options, style="CardAlt.TFrame", padding=(9, 7))
            encode.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(8, 0))
            encode.grid_columnconfigure((1, 3, 5), weight=1)
            ttk.Label(encode, text="ENCODING", style="FieldLabel.TLabel").grid(row=0, column=0, padx=(0, 7))
            ttk.Combobox(
                encode, state="readonly", textvariable=self.processing_frame_transform_var,
                values=("delta", "adaptive", "auto", "raw"), style="Modern.TCombobox", width=11,
            ).grid(row=0, column=1, sticky="ew", padx=(0, 12))
            ttk.Label(encode, text="BLOCK FRAMES", style="FieldLabel.TLabel").grid(row=0, column=2, padx=(0, 7))
            ttk.Combobox(
                encode, state="readonly", textvariable=self.processing_block_frames_var,
                values=("4", "8", "16", "32", "64"), style="Modern.TCombobox", width=8,
            ).grid(row=0, column=3, sticky="ew", padx=(0, 12))
            ttk.Label(encode, text="ZSTD", style="FieldLabel.TLabel").grid(row=0, column=4, padx=(0, 7))
            ttk.Combobox(
                encode, state="readonly", textvariable=self.processing_zstd_level_var,
                values=tuple(str(i) for i in range(10)), style="Modern.TCombobox", width=6,
            ).grid(row=0, column=5, sticky="ew")

        flags = ttk.Frame(form, style="Card.TFrame")
        flags.grid(row=4, column=0, columnspan=3, sticky="w", pady=(9, 4))
        ttk.Checkbutton(
            flags, text="Overwrite existing output", variable=self.processing_overwrite_var,
            style="Modern.TCheckbutton",
        ).pack(side="left")
        ttk.Checkbutton(
            flags, text="Open output after success", variable=self.processing_open_after_var,
            style="Modern.TCheckbutton",
        ).pack(side="left", padx=(18, 0))
        if kind in ("subtract", "pixelop"):
            ttk.Checkbutton(
                flags, text="Verify every output frame", variable=self.processing_verify_var,
                style="Modern.TCheckbutton",
            ).pack(side="left", padx=(18, 0))

        progress_card = ttk.Frame(shell, style="CardAlt.TFrame", padding=(14, 12))
        progress_card.pack(fill="x", pady=(13, 9))
        progress_head = ttk.Frame(progress_card, style="CardAlt.TFrame")
        progress_head.pack(fill="x")
        ttk.Label(progress_head, textvariable=self.processing_stage_var, style="CardTitle.TLabel").pack(side="left")
        ttk.Label(progress_head, textvariable=self.processing_progress_text_var, style="ReadoutValue.TLabel").pack(side="right")
        ttk.Label(
            progress_card, textvariable=self.processing_activity_var, style="ReadoutMuted.TLabel",
            wraplength=980, justify="left",
        ).pack(fill="x", pady=(4, 8))
        self.processing_progressbar = ttk.Progressbar(
            progress_card, orient="horizontal", mode="determinate", maximum=100.0,
            variable=self.processing_progress_var, style="Converter.Horizontal.TProgressbar",
        )
        self.processing_progressbar.pack(fill="x", ipady=4)
        metric = ttk.Frame(progress_card, style="CardAlt.TFrame")
        metric.pack(fill="x", pady=(8, 0))
        ttk.Label(metric, textvariable=self.processing_work_var, style="ReadoutMuted.TLabel").pack(side="left")
        ttk.Label(metric, textvariable=self.processing_elapsed_var, style="ReadoutMuted.TLabel").pack(side="right")

        actions = ttk.Frame(shell, style="Card.TFrame")
        actions.pack(fill="x", pady=(0, 6))
        self.processing_start_button = ttk.Button(
            actions, text="▶  Start Processing", style="Accent.TButton", command=self._start_processing_process,
        )
        self.processing_start_button.pack(side="left")
        self.processing_stop_button = ttk.Button(
            actions, text="■  Stop", style="Secondary.TButton", command=self._stop_processing_process, state="disabled",
        )
        self.processing_stop_button.pack(side="left", padx=(8, 0))
        ttk.Label(actions, textvariable=self.processing_status_var, style="ReadoutMuted.TLabel").pack(side="right")

        log_frame = tk.Frame(shell, background=UI_CARD_ALT)
        log_frame.pack(fill="both", expand=True, pady=(4, 0))
        self.processing_log_text = tk.Text(
            log_frame, height=10, wrap="none", background="#060A12", foreground="#C7D7EF",
            insertbackground=UI_TEXT, selectbackground=UI_ACCENT_DARK, relief="flat", borderwidth=0,
            padx=10, pady=8, font=("Consolas", 9),
        )
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.processing_log_text.yview, style="Modern.Vertical.TScrollbar")
        self.processing_log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")
        self.processing_log_text.pack(side="left", fill="both", expand=True)
        self.processing_log_text.insert("end", f"Processing script: {script}\n")
        self.processing_log_text.configure(state="disabled")

        active = self._active_hdxf_path()
        if active is not None:
            self.processing_input_var.set(str(active))
            self._processing_set_default_output(force=True)
        self.processing_input_var.trace_add("write", lambda *_args: self._processing_input_changed())
        if kind == "sum":
            self.processing_select_var.trace_add("write", lambda *_args: self._processing_set_default_output())
            self.processing_frame_from_var.trace_add("write", lambda *_args: self._processing_set_default_output())
            self.processing_frame_to_var.trace_add("write", lambda *_args: self._processing_set_default_output())

    def _processing_set_default_output(self, *, force: bool = False) -> None:
        if not force and not self._processing_output_auto:
            return
        input_text = getattr(self, "processing_input_var", tk.StringVar(master=self.root)).get().strip()
        if not input_text:
            return
        source = Path(input_text)
        if self._processing_kind == "sum":
            operation = self.processing_operation_var.get() if hasattr(self, "processing_operation_var") else "mean"
            tag = {"sum": "SUM", "mean": "MEAN", "exposure-normalized": "RATE"}.get(operation, "SUM")
            selection = "ALL"
            selected = self.processing_select_var.get().strip() if hasattr(self, "processing_select_var") else ""
            start = self.processing_frame_from_var.get().strip() if hasattr(self, "processing_frame_from_var") else ""
            end = self.processing_frame_to_var.get().strip() if hasattr(self, "processing_frame_to_var") else ""
            if selected:
                selection = "SEL"
            elif start or end:
                selection = f"{start or '1'}-{end or 'end'}"
            self.processing_output_var.set(str(source.with_name(f"{source.stem}_{tag}_{selection}.hdxf")))
        else:
            self.processing_output_var.set(str(source.parent / "hdxf-output"))
        self._processing_output_auto = True

    def _processing_input_changed(self) -> None:
        if self._processing_output_auto:
            self._processing_set_default_output(force=True)

    def _processing_sum_operation_changed(self, _event: tk.Event | None = None) -> None:
        self._processing_set_default_output(force=True)

    def _processing_mode_changed(self, _event: tk.Event | None = None) -> None:
        if self._processing_kind != "subtract" or not hasattr(self, "processing_threshold_entry"):
            return
        mode = self.processing_mode_var.get()
        threshold_enabled = mode == "Lower threshold"
        background_enabled = mode == "Background subtraction"
        range_enabled = mode == "Remove inclusive range"
        for widget in (self.processing_threshold_entry,):
            widget.configure(state="normal" if threshold_enabled else "disabled")
        for widget in (self.processing_background_entry, self.processing_background_button, self.processing_background_frame_entry):
            widget.configure(state="normal" if background_enabled else "disabled")
        for widget in (self.processing_remove_low_entry, self.processing_remove_high_entry):
            widget.configure(state="normal" if range_enabled else "disabled")

    def _browse_processing_input(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self.root, title="Select input HDXF",
            filetypes=[("HDXF detector archive", "*.hdxf"), ("All files", "*.*")],
        )
        if filename:
            self.processing_input_var.set(str(Path(filename)))
            self._processing_output_auto = True
            self._processing_set_default_output(force=True)

    def _browse_processing_output(self) -> None:
        if self._processing_kind == "sum":
            current = self.processing_output_var.get().strip()
            initial = Path(current) if current else None
            filename = filedialog.asksaveasfilename(
                parent=self.root, title="Save processed HDXF", defaultextension=".hdxf",
                initialdir=str(initial.parent) if initial is not None else None,
                initialfile=initial.name if initial is not None else None,
                filetypes=[("HDXF detector archive", "*.hdxf"), ("All files", "*.*")],
            )
            if filename:
                self.processing_output_var.set(str(Path(filename)))
                self._processing_output_auto = False
        else:
            current = self.processing_output_var.get().strip()
            initial = Path(current) if current else Path.cwd()
            folder = filedialog.askdirectory(
                parent=self.root, title="Select HDXF output folder",
                initialdir=str(initial if initial.is_dir() else initial.parent), mustexist=False,
            )
            if folder:
                self.processing_output_var.set(str(Path(folder)))
                self._processing_output_auto = False

    def _browse_processing_background(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self.root, title="Select background HDXF",
            filetypes=[("HDXF detector archive", "*.hdxf"), ("All files", "*.*")],
        )
        if filename:
            self.processing_background_var.set(str(Path(filename)))

    @staticmethod
    def _processing_optional_float(text: str, label: str) -> str | None:
        value = text.strip()
        if not value:
            return None
        try:
            float(value)
        except ValueError as exc:
            raise HDXFViewerError(f"{label} must be a number") from exc
        return value

    @staticmethod
    def _processing_optional_int(text: str, label: str, *, minimum: int = 1) -> str | None:
        value = text.strip()
        if not value:
            return None
        try:
            parsed = int(value)
        except ValueError as exc:
            raise HDXFViewerError(f"{label} must be an integer") from exc
        if parsed < minimum:
            raise HDXFViewerError(f"{label} must be >= {minimum}")
        return str(parsed)

    def _append_processing_selection(self, cmd: list[str]) -> None:
        selected = self.processing_select_var.get().strip()
        if selected:
            try:
                values = [int(item.strip()) for item in selected.split(",") if item.strip()]
            except ValueError as exc:
                raise HDXFViewerError("Selected frames must be comma-separated integers") from exc
            if not values or any(value < 1 for value in values):
                raise HDXFViewerError("Selected frame numbers must be >= 1")
            cmd.extend(["--select-frame", ",".join(str(value) for value in values)])
            return
        start = self._processing_optional_int(self.processing_frame_from_var.get(), "Frame From")
        end = self._processing_optional_int(self.processing_frame_to_var.get(), "Frame To")
        if start is not None and end is not None and int(start) > int(end):
            raise HDXFViewerError("Frame From cannot be greater than Frame To")
        if start is not None:
            cmd.extend(["--frame-from", start])
        if end is not None:
            cmd.extend(["--frame-to", end])

    def _build_processing_command(self, *, validate: bool = True) -> list[str]:
        kind = self._processing_kind or "sum"
        script = self._processing_script_path(kind)
        input_text = self.processing_input_var.get().strip()
        output_text = self.processing_output_var.get().strip()
        if validate:
            if not script.is_file():
                raise HDXFViewerError(f"processing script not found: {script}")
            if not input_text:
                raise HDXFViewerError("select an input HDXF archive")
            input_path = Path(input_text)
            if not input_path.is_file():
                raise HDXFViewerError(f"input HDXF does not exist: {input_path}")
            if input_path.suffix.lower() != ".hdxf":
                raise HDXFViewerError("input must be an .hdxf archive")
            if not output_text:
                raise HDXFViewerError("select an output path")

        cmd = [sys.executable, "-u", str(script), "-i", input_text, "-o", output_text]
        self._append_processing_selection(cmd)

        if kind == "sum":
            output_path = Path(output_text)
            if output_path.suffix.lower() != ".hdxf":
                raise HDXFViewerError("Sum / Mean output must be an .hdxf file")
            if input_text and output_path.resolve() == Path(input_text).resolve():
                raise HDXFViewerError("input and output HDXF paths must be different")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--operation", self.processing_operation_var.get()])
            cmd.extend(["--out-dtype", self.processing_dtype_var.get()])
            cmd.extend(["--progress-step", "2"])
            if self.processing_keep_azint_var.get():
                cmd.append("--keep-azint")
            if self.processing_overwrite_var.get():
                cmd.append("--overwrite")
            self._processing_output_path = output_path
        elif kind == "subtract":
            output_dir = Path(output_text)
            output_dir.mkdir(parents=True, exist_ok=True)
            mode = self.processing_mode_var.get()
            if mode == "Lower threshold":
                value = self._processing_optional_float(self.processing_threshold_var.get(), "Lower threshold")
                if value is None:
                    raise HDXFViewerError("enter a lower threshold")
                cmd.extend(["--threshold", value])
            elif mode == "Background subtraction":
                background = self.processing_background_var.get().strip()
                if not background:
                    raise HDXFViewerError("select a background HDXF archive")
                background_path = Path(background)
                if not background_path.is_file():
                    raise HDXFViewerError(f"background HDXF does not exist: {background_path}")
                frame = self._processing_optional_int(self.processing_background_frame_var.get(), "Background frame") or "1"
                cmd.extend(["--background", background, "--background-frame", frame])
            elif mode == "Remove inclusive range":
                low = self._processing_optional_float(self.processing_remove_low_var.get(), "Remove range low")
                high = self._processing_optional_float(self.processing_remove_high_var.get(), "Remove range high")
                if low is None or high is None:
                    raise HDXFViewerError("enter both low and high values for Remove range")
                if float(low) > float(high):
                    raise HDXFViewerError("Remove range low cannot be greater than high")
                cmd.extend(["--remove-threshold", f"{low},{high}"])

            upper = self._processing_optional_float(self.processing_upper_var.get(), "Upper threshold")
            if upper is not None:
                cmd.extend(["--upper-threshold", upper])
            fill = self._processing_optional_float(self.processing_fill_var.get(), "Fill value") or "0"
            cmd.extend(["--fill", fill])
            protect = self.processing_protect_var.get().strip()
            if protect:
                parts = [item.strip() for item in protect.split(",")]
                if len(parts) != 4:
                    raise HDXFViewerError("Protect ROI must be x0,y0,x1,y1")
                try:
                    [int(item) for item in parts]
                except ValueError as exc:
                    raise HDXFViewerError("Protect ROI coordinates must be integers") from exc
                cmd.extend(["--protect", ",".join(parts)])

            block_frames = self._processing_optional_int(self.processing_block_frames_var.get(), "Block frames") or "8"
            zstd = self._processing_optional_int(self.processing_zstd_level_var.get(), "Zstd level", minimum=0) or "3"
            if int(zstd) > 9:
                raise HDXFViewerError("Zstd level must be between 0 and 9")
            cmd.extend([
                "--frame-transform", self.processing_frame_transform_var.get(),
                "--block-frames", block_frames,
                "--zstd-level", zstd,
                "--progress-step", "2",
            ])
            if self.processing_verify_var.get():
                cmd.append("--verify")
            cmd.append("--overwrite" if self.processing_overwrite_var.get() else "--no-overwrite")
            self._processing_output_path = None  # learned from [OUT] line
        else:  # pixelop
            output_dir = Path(output_text)
            output_dir.mkdir(parents=True, exist_ok=True)

            operation = self.processing_pixelop_operation_var.get()
            flag = {
                "Add": "--add",
                "Subtract": "--subtract",
                "Multiply": "--multiple",
                "Divide": "--devide",
            }.get(operation)
            if flag is None:
                raise HDXFViewerError(f"unknown Pixel Op operation: {operation}")
            value = self._processing_optional_float(
                self.processing_pixelop_value_var.get(), "Pixel Op value"
            )
            if value is None:
                raise HDXFViewerError("enter a Pixel Op value")
            if operation == "Divide" and float(value) == 0.0:
                raise HDXFViewerError("Pixel Op divide value must be non-zero")
            cmd.extend([flag, value])

            roi = self.processing_pixelop_roi_var.get().strip()
            if roi:
                parts = [item.strip() for item in roi.split(",")]
                if len(parts) != 4:
                    raise HDXFViewerError("Pixel Op ROI must be x1,y1,x2,y2")
                try:
                    coords = [int(item) for item in parts]
                except ValueError as exc:
                    raise HDXFViewerError("Pixel Op ROI coordinates must be integers") from exc
                x1, y1, x2, y2 = coords
                if x1 < 0 or y1 < 0 or x2 < x1 or y2 < y1:
                    raise HDXFViewerError(
                        "Pixel Op ROI requires 0 <= x1 <= x2 and 0 <= y1 <= y2"
                    )
                cmd.extend(["--roi", ",".join(str(v) for v in coords)])

            block_frames = self._processing_optional_int(
                self.processing_block_frames_var.get(), "Block frames"
            ) or "8"
            zstd = self._processing_optional_int(
                self.processing_zstd_level_var.get(), "Zstd level", minimum=0
            ) or "3"
            if int(zstd) > 9:
                raise HDXFViewerError("Zstd level must be between 0 and 9")
            cmd.extend([
                "--frame-transform", self.processing_frame_transform_var.get(),
                "--block-frames", block_frames,
                "--zstd-level", zstd,
                "--progress-step", "2",
            ])
            if self.processing_verify_var.get():
                cmd.append("--verify")
            cmd.append("--overwrite" if self.processing_overwrite_var.get() else "--no-overwrite")
            self._processing_output_path = None  # learned from [OUT] line
        return cmd

    def _append_processing_log(self, text: str) -> None:
        widget = getattr(self, "processing_log_text", None)
        if widget is None:
            return
        try:
            if not widget.winfo_exists():
                return
            widget.configure(state="normal")
            widget.insert("end", text)
            widget.see("end")
            widget.configure(state="disabled")
        except tk.TclError:
            pass

    def _set_processing_progress(self, value: float, *, text: str | None = None) -> None:
        value = max(self._processing_last_progress, min(100.0, float(value)))
        self._processing_last_progress = value
        self.processing_progress_var.set(value)
        self.processing_progress_text_var.set(text if text is not None else f"{value:.0f}%")

    def _update_processing_elapsed(self) -> None:
        started = self._processing_start_time
        if started is None:
            return
        elapsed = max(0.0, time.monotonic() - started)
        if elapsed < 60:
            self.processing_elapsed_var.set(f"Elapsed  {elapsed:.0f} s")
        else:
            minutes, seconds = divmod(int(elapsed), 60)
            self.processing_elapsed_var.set(f"Elapsed  {minutes:d}m {seconds:02d}s")

    def _handle_processing_output_line(self, raw_line: str) -> None:
        line = raw_line.strip()
        if not line:
            return

        final_output_match = re.match(
            r"\[OUT HDXF\]\s+(.+\.hdxf)\s*$", line, flags=re.IGNORECASE
        )
        if final_output_match:
            raw = final_output_match.group(1).strip().strip('"')
            self._processing_output_path = Path(raw)
            self.processing_stage_var.set("Finalising HDXF archive")
            self.processing_activity_var.set(f"Output written: {raw}")
            self._set_processing_progress(100.0, text="100%")
            return

        early_output_match = re.match(
            r"\[OUT\]\s+(.+\.hdxf)\s*$", line, flags=re.IGNORECASE
        )
        if early_output_match:
            raw = early_output_match.group(1).strip().strip('"')
            self._processing_output_path = Path(raw)
            self.processing_activity_var.set(f"Output destination: {raw}")
            return

        proc_match = re.search(
            r"\[PROC\]\s*(\d+)/(\d+)\s*\(([0-9]+(?:\.[0-9]+)?)%\)",
            line, flags=re.IGNORECASE,
        )
        if proc_match:
            done = int(proc_match.group(1))
            total = int(proc_match.group(2))
            percent = max(0.0, min(100.0, float(proc_match.group(3))))
            self._set_processing_progress(
                percent,
                text=(f"{percent:.1f}%" if percent < 10.0 else f"{percent:.0f}%"),
            )
            self.processing_work_var.set(f"Frames  {done:,} / {total:,}")
            self.processing_stage_var.set("Processing detector frames")
            self.processing_activity_var.set(line)
            return

        verify_progress = re.search(
            r"\[VERIFY\]\s*(\d+)/(\d+)\s*\(([0-9]+(?:\.[0-9]+)?)%\)",
            line, flags=re.IGNORECASE,
        )
        if verify_progress:
            done = int(verify_progress.group(1))
            total = int(verify_progress.group(2))
            percent = max(0.0, min(100.0, float(verify_progress.group(3))))
            if not self.processing_stage_var.get().startswith("Verifying"):
                self._processing_last_progress = 0.0
                self.processing_progress_var.set(0.0)
            self.processing_stage_var.set("Verifying output")
            self.processing_activity_var.set(line)
            self.processing_work_var.set(f"Verify  {done:,} / {total:,}")
            self._set_processing_progress(
                percent,
                text=(f"{percent:.1f}%" if percent < 10.0 else f"{percent:.0f}%"),
            )
            return

        lower = line.lower()
        if line.startswith("[START]"):
            self.processing_stage_var.set("Opening HDXF archive")
            self.processing_activity_var.set(line)
            self._set_processing_progress(0.0, text="0%")
        elif line.startswith("[DISCOVER]") or line.startswith("[INPUT]"):
            self.processing_stage_var.set("Reading detector archive")
            self.processing_activity_var.set(line)
            self._set_processing_progress(0.0, text="0%")
        elif line.startswith("[SELECT"):
            self.processing_stage_var.set("Selecting detector frames")
            self.processing_activity_var.set(line)
            match = re.search(r"\]\s+(\d+)\s+frames", line)
            if match:
                self.processing_work_var.set(f"Frames  {int(match.group(1)):,} selected")
            self._set_processing_progress(0.0, text="0%")
        elif line.startswith("[DTYPE]") or line.startswith("[CFG]") or line.startswith("[OP]"):
            self.processing_stage_var.set("Preparing processing")
            self.processing_activity_var.set(line)
            self._set_processing_progress(0.0, text="0%")
        elif line.startswith("[VERIFY]"):
            self.processing_stage_var.set("Verifying output")
            self.processing_activity_var.set(line)
            if "pass" in lower:
                self._set_processing_progress(100.0, text="100%")
            else:
                # The output archive has already reached 100%; verification is
                # a second phase, so start a fresh phase-local progress bar.
                self._processing_last_progress = 0.0
                self.processing_progress_var.set(0.0)
                self.processing_progress_text_var.set("0%")
        elif "[result] pass" in lower:
            self.processing_stage_var.set("Processing completed")
            self.processing_activity_var.set(line)
            self._set_processing_progress(100.0, text="100%")
        elif "[result] fail" in lower or "traceback" in lower or lower.startswith("error:"):
            self.processing_stage_var.set("Processing error")
            self.processing_activity_var.set(line)

    def _start_processing_process(self) -> None:
        if self._processing_process is not None and self._processing_process.poll() is None:
            return
        try:
            cmd = self._build_processing_command(validate=True)
        except Exception as exc:
            messagebox.showerror("Cannot start HDXF processing", str(exc), parent=self.root)
            return

        self._processing_stop_requested = False
        self._processing_last_progress = 0.0
        self.processing_progress_var.set(0.0)
        self.processing_progress_text_var.set("0%")
        self.processing_stage_var.set("Starting processing")
        self.processing_activity_var.set("Launching the external HDXF processing script.")
        self.processing_status_var.set("Starting…")
        self.processing_elapsed_var.set("Elapsed  0 s")
        self.processing_work_var.set("Frames  —")
        self._processing_start_time = time.monotonic()

        self.processing_log_text.configure(state="normal")
        self.processing_log_text.delete("1.0", "end")
        self.processing_log_text.insert("end", "Command:\n" + self._format_command_for_display(cmd) + "\n\n")
        self.processing_log_text.configure(state="disabled")

        creationflags = 0
        if sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            creationflags = int(subprocess.CREATE_NO_WINDOW)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8:replace"
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(self._processing_script_path().parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=creationflags,
            )
        except Exception as exc:
            messagebox.showerror("HDXF processing launch failed", f"{type(exc).__name__}: {exc}", parent=self.root)
            return

        self._processing_run_serial += 1
        serial = self._processing_run_serial
        while True:
            try:
                self._processing_queue.get_nowait()
            except queue.Empty:
                break
        self._processing_process = process
        self.processing_start_button.configure(state="disabled")
        self.processing_stop_button.configure(state="normal")
        self.processing_status_var.set("Running")
        label = {
            "sum": "Sum / Mean",
            "subtract": "Subtract / Filter",
            "pixelop": "Pixel Operation",
        }.get(self._processing_kind, "Processing")
        self.status_var.set(f"HDXF {label} running · PID {process.pid}")

        def reader() -> None:
            try:
                assert process.stdout is not None
                for text in process.stdout:
                    self._processing_queue.put((serial, "line", text))
            except Exception as exc:
                self._processing_queue.put((serial, "line", f"\n[log reader error] {exc}\n"))
            finally:
                self._processing_queue.put((serial, "done", process.wait()))

        self._processing_reader = threading.Thread(target=reader, name="hdxf-processing-log", daemon=True)
        self._processing_reader.start()
        self._schedule_processing_poll()

    def _schedule_processing_poll(self) -> None:
        if self._processing_poll_after_id is None:
            self._processing_poll_after_id = self.root.after(80, self._poll_processing_queue)

    def _poll_processing_queue(self) -> None:
        self._processing_poll_after_id = None
        done_code: int | None = None
        while True:
            try:
                serial, kind, payload = self._processing_queue.get_nowait()
            except queue.Empty:
                break
            if serial != self._processing_run_serial:
                continue
            if kind == "line":
                text = str(payload)
                self._append_processing_log(text)
                self._handle_processing_output_line(text)
            elif kind == "done":
                done_code = int(payload)
        self._update_processing_elapsed()
        if done_code is not None:
            self._finish_processing_process(done_code)
            return
        process = self._processing_process
        if process is not None and process.poll() is None and self._processing_panel is not None:
            try:
                if self._processing_panel.winfo_exists():
                    self._schedule_processing_poll()
            except tk.TclError:
                pass

    def _finish_processing_process(self, return_code: int) -> None:
        self._processing_process = None
        try:
            self.processing_start_button.configure(state="normal")
            self.processing_stop_button.configure(state="disabled")
        except tk.TclError:
            pass
        self._update_processing_elapsed()
        if self._processing_stop_requested:
            self.processing_stage_var.set("Processing stopped")
            self.processing_activity_var.set("The processing job was stopped before completion.")
            self.processing_status_var.set("Stopped")
            self.status_var.set("HDXF processing stopped")
            return
        if return_code == 0:
            self._set_processing_progress(100.0, text="100%")
            self.processing_stage_var.set("Processing complete")
            self.processing_activity_var.set("The derived HDXF archive was created successfully.")
            self.processing_status_var.set("Completed")
            self.status_var.set("HDXF processing completed")
            output = self._processing_output_path
            if self.processing_open_after_var.get() and output is not None and output.is_file():
                self.open_path(output)
        else:
            self.processing_stage_var.set("Processing failed")
            self.processing_activity_var.set("The processing script exited with an error. Review the log below.")
            self.processing_status_var.set(f"Failed ({return_code})")
            self.status_var.set(f"HDXF processing failed · exit code {return_code}")

    def _stop_processing_process(self) -> None:
        process = self._processing_process
        if process is None or process.poll() is not None:
            return
        self._processing_stop_requested = True
        self.processing_status_var.set("Stopping…")
        self.processing_stage_var.set("Stopping processing")
        self.processing_activity_var.set("Stopping the external HDXF processing process.")
        self._append_processing_log("\n[stop requested]\n")
        try:
            process.terminate()
        except Exception as exc:
            self._append_processing_log(f"[terminate failed] {exc}\n")

    def _close_processing_tool(self, *, force: bool = False) -> None:
        process = self._processing_process
        if process is not None and process.poll() is None:
            if not force:
                close = messagebox.askyesno(
                    "HDXF processing is running",
                    "Stop the running processing job and close the tool?",
                    parent=self.root,
                )
                if not close:
                    return
            self._stop_processing_process()
            self._processing_run_serial += 1
            self._processing_process = None
        if self._processing_poll_after_id is not None:
            try:
                self.root.after_cancel(self._processing_poll_after_id)
            except tk.TclError:
                pass
            self._processing_poll_after_id = None
        panel = self._processing_panel
        self._processing_panel = None
        self._processing_kind = None
        if panel is not None:
            try:
                if panel.winfo_exists():
                    panel.destroy()
            except tk.TclError:
                pass



    def _build_sidebar(self, parent: ttk.Frame) -> None:
        # The complete inspector is scrollable; each section is a separate
        # visual card so dense scientific information stays easy to scan.
        scroll_host = ttk.Frame(parent, style="Side.TFrame")
        scroll_host.pack(fill="both", expand=True)

        self.sidebar_canvas = tk.Canvas(
            scroll_host,
            background=UI_BG,
            highlightthickness=0,
            borderwidth=0,
            takefocus=False,
        )
        self.sidebar_scrollbar = ttk.Scrollbar(
            scroll_host,
            orient="vertical",
            command=self.sidebar_canvas.yview,
            style="Modern.Vertical.TScrollbar",
        )
        self.sidebar_canvas.configure(yscrollcommand=self.sidebar_scrollbar.set)
        self.sidebar_scrollbar.pack(side="right", fill="y")
        self.sidebar_canvas.pack(side="left", fill="both", expand=True)

        sidebar = ttk.Frame(self.sidebar_canvas, style="Side.TFrame", padding=(12, 12))
        self._sidebar_window = self.sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")
        sidebar.bind("<Configure>", self._on_sidebar_content_configure)
        self.sidebar_canvas.bind("<Configure>", self._on_sidebar_canvas_configure)
        scroll_host.bind("<Enter>", self._enable_sidebar_mousewheel)
        scroll_host.bind("<Leave>", self._disable_sidebar_mousewheel)

        inspector_header = ttk.Frame(sidebar, style="Side.TFrame")
        inspector_header.pack(fill="x", pady=(0, 10))
        header_text = ttk.Frame(inspector_header, style="Side.TFrame")
        header_text.pack(side="left", fill="x", expand=True)
        ttk.Label(header_text, text="INSPECTOR", style="InspectorTitle.TLabel").pack(anchor="w")
        ttk.Label(header_text, text="ACTIVE DETECTOR VIEW", style="InspectorSub.TLabel").pack(anchor="w")
        ttk.Label(inspector_header, textvariable=self.active_view_var, style="InspectorChip.TLabel").pack(side="right", anchor="center")

        preview = self._make_card(sidebar, "Navigator", "Current frame overview and detector centre")
        self.thumbnail_canvas = tk.Canvas(
            preview,
            width=340,
            height=185,
            background=UI_CANVAS,
            highlightthickness=0,
            borderwidth=0,
        )
        self.thumbnail_canvas.pack(fill="x")

        histogram = self._make_card(
            sidebar,
            "Histogram",
            "Edges resize · centre moves · wheel zoom · double-click Auto · right-click Full.",
        )
        self.histogram_canvas = tk.Canvas(
            histogram,
            width=330,
            height=210,
            background=UI_HISTOGRAM,
            highlightthickness=0,
            borderwidth=0,
            cursor="crosshair",
        )
        self.histogram_canvas.pack(fill="x")
        self.histogram_canvas.bind("<ButtonPress-1>", self._histogram_press)
        self.histogram_canvas.bind("<B1-Motion>", self._histogram_drag)
        self.histogram_canvas.bind("<ButtonRelease-1>", self._histogram_release)
        self.histogram_canvas.bind("<Motion>", self._histogram_motion)
        self.histogram_canvas.bind("<Leave>", self._histogram_leave)
        self.histogram_canvas.bind("<MouseWheel>", self._histogram_wheel)
        self.histogram_canvas.bind("<Button-4>", self._histogram_wheel)
        self.histogram_canvas.bind("<Button-5>", self._histogram_wheel)
        self.histogram_canvas.bind("<Double-Button-1>", self._histogram_auto)
        self.histogram_canvas.bind("<Button-3>", self._histogram_full)
        self.histogram_canvas.bind("<Configure>", lambda _event: self._update_histogram())

        display = self._make_card(
            sidebar,
            "Display",
            "Shared by every open view for direct visual comparison.",
        )
        self.contrast_mode = ttk.Combobox(
            display,
            state="readonly",
            textvariable=self.contrast_mode_var,
            values=AUTO_CONTRAST_MODES,
            style="Modern.TCombobox",
        )
        self.contrast_mode.pack(fill="x", pady=(0, 8))
        self.contrast_mode.bind("<<ComboboxSelected>>", lambda _event: self.auto_contrast())

        buttons = ttk.Frame(display, style="Card.TFrame")
        buttons.pack(fill="x", pady=(0, 9))
        ttk.Button(buttons, text="Auto Contrast", style="Accent.TButton", command=self.auto_contrast).pack(side="left", fill="x", expand=True)
        ttk.Button(buttons, text="Full Range", style="Secondary.TButton", command=self.full_contrast).pack(side="left", fill="x", expand=True, padx=(6, 0))

        range_row = ttk.Frame(display, style="Card.TFrame")
        range_row.pack(fill="x", pady=(0, 7))
        ttk.Label(range_row, text="MIN", style="FieldLabel.TLabel").grid(row=0, column=0, sticky="w")
        min_entry = ttk.Entry(range_row, textvariable=self.min_var, width=11, style="Modern.TEntry")
        min_entry.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        ttk.Label(range_row, text="MAX", style="FieldLabel.TLabel").grid(row=0, column=1, sticky="w")
        max_entry = ttk.Entry(range_row, textvariable=self.max_var, width=11, style="Modern.TEntry")
        max_entry.grid(row=1, column=1, sticky="ew")
        range_row.columnconfigure(0, weight=1)
        range_row.columnconfigure(1, weight=1)
        min_entry.bind("<Return>", lambda _e: self.apply_contrast())
        max_entry.bind("<Return>", lambda _e: self.apply_contrast())
        ttk.Button(display, text="Apply Shared Range", style="Secondary.TButton", command=self.apply_contrast).pack(fill="x", pady=(0, 8))

        options = ttk.Frame(display, style="Card.TFrame")
        options.pack(fill="x")
        ttk.Checkbutton(
            options,
            text="Invert grayscale",
            variable=self.invert_var,
            command=self._toggle_invert,
            style="Modern.TCheckbutton",
        ).pack(anchor="w", pady=(0, 4))
        ttk.Checkbutton(
            options,
            text="Show pixel values at high zoom",
            variable=self.show_values_var,
            command=self._toggle_show_values,
            style="Modern.TCheckbutton",
        ).pack(anchor="w")

        roi_card = self._make_card(
            sidebar,
            "Region of Interest",
            "Choose a shape. Right-drag draws Rectangle / Ellipse / Circle / Annulus / Freehand; Polygon uses right-click vertices. Annulus statistics exclude the inner circle.",
        )

        shape_row = ttk.Frame(roi_card, style="Card.TFrame")
        shape_row.pack(fill="x", pady=(0, 8))
        ttk.Label(shape_row, text="SHAPE", style="FieldLabel.TLabel").pack(side="left", padx=(0, 8))
        self.roi_shape_combo = ttk.Combobox(
            shape_row,
            state="readonly",
            textvariable=self.roi_shape_var,
            values=ROI_SHAPES,
            style="Modern.TCombobox",
            width=13,
        )
        self.roi_shape_combo.pack(side="left", fill="x", expand=True)
        self.roi_shape_combo.bind("<<ComboboxSelected>>", self._on_roi_shape_selected)

        roi_actions = ttk.Frame(roi_card, style="Card.TFrame")
        roi_actions.pack(fill="x", pady=(0, 9))
        self.roi_draw_button = ttk.Button(
            roi_actions,
            textvariable=self.roi_draw_text_var,
            style="Accent.TButton",
            command=self._toggle_roi_draw_mode,
        )
        self.roi_draw_button.pack(side="left", fill="x", expand=True)
        self.roi_clear_button = ttk.Button(
            roi_actions,
            text="Clear ROI",
            style="Secondary.TButton",
            command=self._clear_active_roi,
        )
        self.roi_clear_button.pack(side="left", fill="x", expand=True, padx=(6, 0))

        roi_grid = ttk.Frame(roi_card, style="Card.TFrame")
        roi_grid.pack(fill="x")
        for col in range(2):
            roi_grid.columnconfigure(col, weight=1)

        def roi_spin(row: int, col: int, label: str, variable: tk.StringVar, *, minimum: int) -> ttk.Spinbox:
            box = ttk.Frame(roi_grid, style="Card.TFrame")
            box.grid(row=row, column=col, sticky="ew", padx=(0 if col == 0 else 4, 4 if col == 0 else 0), pady=(0, 7))
            ttk.Label(box, text=label, style="FieldLabel.TLabel").pack(anchor="w")
            spin = ttk.Spinbox(
                box,
                from_=minimum,
                to=999999,
                increment=1,
                textvariable=variable,
                style="Modern.TSpinbox",
                command=lambda: self._apply_roi_numeric(create_if_missing=True),
            )
            spin.pack(fill="x", pady=(2, 0))
            spin.bind("<Return>", lambda _e: self._apply_roi_numeric(create_if_missing=True))
            spin.bind("<FocusOut>", lambda _e: self._apply_roi_numeric(create_if_missing=False))
            return spin

        self.roi_x_spin = roi_spin(0, 0, "X", self.roi_x_var, minimum=0)
        self.roi_y_spin = roi_spin(0, 1, "Y", self.roi_y_var, minimum=0)
        self.roi_width_spin = roi_spin(1, 0, "WIDTH", self.roi_width_var, minimum=1)
        self.roi_height_spin = roi_spin(1, 1, "HEIGHT", self.roi_height_var, minimum=1)
        self.roi_inner_diameter_spin = roi_spin(2, 0, "INNER DIAMETER", self.roi_inner_diameter_var, minimum=1)
        ttk.Label(
            roi_grid,
            text="Excluded for Annulus",
            style="CardSubtitle.TLabel",
            justify="left",
        ).grid(row=2, column=1, sticky="w", padx=(4, 0), pady=(18, 7))

        ttk.Label(
            roi_card,
            text="For Polygon/Freehand, X/Y/Width/Height translate and scale the contour. Circle and Annulus keep Width = Height. Annulus INNER DIAMETER is ignored during statistics.",
            style="CardSubtitle.TLabel",
            wraplength=330,
            justify="left",
        ).pack(fill="x", pady=(0, 7))

        bounds = ttk.Frame(roi_card, style="CardAlt.TFrame", padding=(9, 7))
        bounds.pack(fill="x", pady=(1, 8))
        ttk.Label(bounds, text="X2", style="FieldLabel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(bounds, textvariable=self.roi_x2_var, style="ReadoutValue.TLabel").grid(row=0, column=1, sticky="e")
        ttk.Label(bounds, text="Y2", style="FieldLabel.TLabel").grid(row=1, column=0, sticky="w", pady=(3, 0))
        ttk.Label(bounds, textvariable=self.roi_y2_var, style="ReadoutValue.TLabel").grid(row=1, column=1, sticky="e", pady=(3, 0))
        bounds.columnconfigure(1, weight=1)

        ttk.Label(roi_card, text="CURRENT FRAME STATISTICS", style="FieldLabel.TLabel").pack(anchor="w", pady=(0, 5))
        stats = ttk.Frame(roi_card, style="CardAlt.TFrame", padding=(9, 7))
        stats.pack(fill="x")
        roi_stats = (
            ("Pixels", self.roi_pixels_var),
            ("Min", self.roi_min_var),
            ("Max", self.roi_max_var),
            ("Mean", self.roi_mean_var),
            ("Sum", self.roi_sum_var),
            ("Std Dev", self.roi_std_var),
            ("Zeros", self.roi_zeros_var),
        )
        for row, (label, variable) in enumerate(roi_stats):
            ttk.Label(stats, text=label, style="ReadoutMuted.TLabel").grid(row=row, column=0, sticky="w", pady=(2 if row else 0, 0))
            ttk.Label(stats, textvariable=variable, style="ReadoutValue.TLabel").grid(row=row, column=1, sticky="e", pady=(2 if row else 0, 0))
        stats.columnconfigure(1, weight=1)

        image_info = self._make_card(sidebar, "Image Info", "Detector, acquisition and current-frame statistics")
        self.image_info_text = tk.Text(
            image_info,
            height=25,
            width=38,
            wrap="word",
            background=UI_CARD_ALT,
            foreground=UI_TEXT,
            selectbackground=UI_ACCENT_STRONG,
            selectforeground=UI_WHITE,
            insertbackground=UI_TEXT,
            relief="flat",
            borderwidth=0,
            padx=9,
            pady=8,
            font=("Segoe UI", 9),
            state="disabled",
        )
        self.image_info_text.tag_configure("section", font=("Segoe UI", 10, "bold"), foreground=UI_ACCENT, spacing1=8, spacing3=3)
        self.image_info_text.tag_configure("label", font=("Segoe UI", 9, "bold"), foreground=UI_TEXT_MUTED)
        self.image_info_text.pack(fill="x", expand=False)

        metadata = self._make_card(sidebar, "HDF5 Metadata", "Best-effort original HDF5 fields; extra/unclassified values appear under Other Information", pady=(0, 4))
        tree_host = ttk.Frame(metadata, style="Card.TFrame")
        tree_host.pack(fill="both", expand=True)
        self.metadata_tree = ttk.Treeview(
            tree_host,
            columns=("value",),
            show="tree headings",
            height=28,
            style="Metadata.Treeview",
        )
        self.metadata_tree.heading("#0", text="FIELD")
        self.metadata_tree.heading("value", text="VALUE")
        self.metadata_tree.column("#0", width=145, stretch=True)
        self.metadata_tree.column("value", width=180, stretch=True)
        metadata_scroll = ttk.Scrollbar(
            tree_host,
            orient="vertical",
            command=self.metadata_tree.yview,
            style="Modern.Vertical.TScrollbar",
        )
        self.metadata_tree.configure(yscrollcommand=metadata_scroll.set)
        self.metadata_tree.pack(side="left", fill="both", expand=True)
        metadata_scroll.pack(side="right", fill="y")

    def _on_sidebar_content_configure(self, _event: tk.Event | None = None) -> None:
        if not hasattr(self, "sidebar_canvas"):
            return
        self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

    def _on_sidebar_canvas_configure(self, event: tk.Event) -> None:
        if hasattr(self, "_sidebar_window"):
            self.sidebar_canvas.itemconfigure(self._sidebar_window, width=max(1, event.width))
        self._on_sidebar_content_configure()

    def _enable_sidebar_mousewheel(self, _event: tk.Event | None = None) -> None:
        self.root.bind_all("<MouseWheel>", self._on_sidebar_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_sidebar_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._on_sidebar_mousewheel, add="+")

    def _disable_sidebar_mousewheel(self, _event: tk.Event | None = None) -> None:
        # Remove only the temporary sidebar-wide bindings. The image canvas has
        # its own widget-local zoom bindings and is unaffected.
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

    def _on_sidebar_mousewheel(self, event: tk.Event) -> str | None:
        # Let the Metadata Treeview consume the wheel itself when the pointer is
        # directly over it; elsewhere, scroll the complete right-side panel.
        widget = self.root.winfo_containing(self.root.winfo_pointerx(), self.root.winfo_pointery())
        if widget is self.metadata_tree:
            return None
        if getattr(event, "num", None) == 4:
            units = -3
        elif getattr(event, "num", None) == 5:
            units = 3
        else:
            delta = int(getattr(event, "delta", 0))
            units = -max(1, abs(delta) // 120) if delta > 0 else max(1, abs(delta) // 120)
        self.sidebar_canvas.yview_scroll(units, "units")
        return "break"

    def _set_roi_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for name in (
            "roi_x_spin", "roi_y_spin", "roi_width_spin", "roi_height_spin",
            "roi_inner_diameter_spin", "roi_draw_button", "roi_clear_button",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                try:
                    widget.configure(state=state)
                except tk.TclError:
                    pass
        combo = getattr(self, "roi_shape_combo", None)
        if combo is not None:
            try:
                combo.configure(state="readonly" if enabled else "disabled")
            except tk.TclError:
                pass

    def _roi_draw_button_text(self, view: InternalViewState | None) -> str:
        if view is None:
            return "Draw ROI (Right)"
        shape = view.canvas.roi_draw_shape
        if view.canvas.roi_draw_mode:
            return f"Drawing {shape}…"
        return f"Draw {shape}"

    def _sync_roi_sidebar_from_view(self, view: InternalViewState | None = None) -> None:
        view = view or self.active_view
        has_frame = view is not None and view.current_frame is not None
        self._set_roi_controls_enabled(has_frame)
        if not has_frame or view is None:
            self.roi_shape_var.set("Rectangle")
            self.roi_x_var.set("0")
            self.roi_y_var.set("0")
            self.roi_width_var.set("1")
            self.roi_height_var.set("1")
            self.roi_inner_diameter_var.set("—")
            self.roi_x2_var.set("—")
            self.roi_y2_var.set("—")
            self.roi_draw_text_var.set("Draw ROI (Right)")
            self._update_roi_statistics(view)
            return

        roi = view.canvas.get_roi()
        self.roi_shape_var.set(view.canvas.get_roi_shape())
        self.roi_draw_text_var.set(self._roi_draw_button_text(view))
        image_h, image_w = view.current_frame.shape
        try:
            self.roi_x_spin.configure(to=max(0, image_w - 1))
            self.roi_y_spin.configure(to=max(0, image_h - 1))
            self.roi_width_spin.configure(to=max(1, image_w))
            self.roi_height_spin.configure(to=max(1, image_h))
            self.roi_inner_diameter_spin.configure(to=max(1, min(image_w, image_h)))
        except tk.TclError:
            pass

        if roi is None:
            self.roi_inner_diameter_var.set("—")
            self.roi_x2_var.set("—")
            self.roi_y2_var.set("—")
        else:
            x, y, width, height = roi
            self.roi_x_var.set(str(x))
            self.roi_y_var.set(str(y))
            self.roi_width_var.set(str(width))
            self.roi_height_var.set(str(height))
            self.roi_x2_var.set(str(x + width - 1))
            self.roi_y2_var.set(str(y + height - 1))
            if view.canvas.get_roi_shape() == "Annulus":
                self.roi_inner_diameter_var.set(str(view.canvas.get_roi_inner_diameter()))
            else:
                self.roi_inner_diameter_var.set("—")

        # Inner diameter is meaningful only for Annulus. Keep the control
        # visible so the layout is stable, but disable it for other shapes.
        try:
            self.roi_inner_diameter_spin.configure(
                state="normal" if view.canvas.get_roi_shape() == "Annulus" else "disabled"
            )
        except tk.TclError:
            pass
        self._update_roi_statistics(view)

    def _on_roi_shape_selected(self, _event: tk.Event | None = None) -> None:
        view = self.active_view
        if view is None or view.current_frame is None:
            return
        shape = self.roi_shape_var.get().strip().title()
        if shape not in ROI_SHAPES:
            self._sync_roi_sidebar_from_view(view)
            return
        view.canvas.set_roi_draw_shape(shape, notify=True)
        self._sync_roi_sidebar_from_view(view)
        self.roi_draw_text_var.set(self._roi_draw_button_text(view))
        self.status_var.set(
            f"ROI shape: {shape} · source HDF5/HDXF unchanged"
        )

    def _toggle_roi_draw_mode(self) -> None:
        view = self.active_view
        if view is None or view.current_frame is None:
            self.status_var.set("Open a detector image before drawing an ROI")
            return
        enabled = not view.canvas.roi_draw_mode
        view.canvas.set_roi_draw_shape(self.roi_shape_var.get(), notify=False)
        view.canvas.set_roi_draw_mode(enabled)
        self.roi_draw_text_var.set(self._roi_draw_button_text(view))
        shape = view.canvas.roi_draw_shape
        if enabled:
            if shape == "Polygon":
                self.status_var.set(
                    "Polygon ROI · RIGHT-click vertices · click the first vertex again to close · left drag still pans"
                )
            elif shape == "Freehand":
                self.status_var.set(
                    "Freehand ROI · hold RIGHT button and trace the contour · release to finish · left drag still pans"
                )
            elif shape == "Annulus":
                self.status_var.set(
                    "Annulus ROI · RIGHT-drag the outer circle · inner diameter defaults to 50% and can be edited numerically · inner circle is excluded from statistics"
                )
            else:
                self.status_var.set(
                    f"{shape} ROI · RIGHT-drag over detector pixels · release to finish · left drag still pans"
                )
        else:
            self.status_var.set("ROI draw mode cancelled")

    def _clear_active_roi(self) -> None:
        view = self.active_view
        if view is None:
            return
        view.canvas.clear_roi(notify=True)

    def _apply_roi_numeric(self, *, create_if_missing: bool) -> None:
        view = self.active_view
        if view is None or view.current_frame is None:
            return
        if view.canvas.get_roi() is None and not create_if_missing:
            return
        try:
            x = int(self.roi_x_var.get().strip())
            y = int(self.roi_y_var.get().strip())
            width = int(self.roi_width_var.get().strip())
            height = int(self.roi_height_var.get().strip())
        except (TypeError, ValueError):
            self._sync_roi_sidebar_from_view(view)
            self.status_var.set("ROI values must be integers")
            return
        if width < 1 or height < 1:
            self._sync_roi_sidebar_from_view(view)
            self.status_var.set("ROI Width and Height must be at least 1 pixel")
            return

        shape = self.roi_shape_var.get().strip().title()
        current = view.canvas.get_roi()
        if shape in ("Circle", "Annulus"):
            # Make whichever single dimension the user changed authoritative.
            if current is not None:
                old_width, old_height = current[2], current[3]
                if width != old_width and height == old_height:
                    size = width
                elif height != old_height and width == old_width:
                    size = height
                else:
                    size = max(width, height)
            else:
                size = max(width, height)
            width = height = max(1, size)

        inner_diameter: int | None = None
        if shape == "Annulus":
            try:
                inner_diameter = int(self.roi_inner_diameter_var.get().strip())
            except (TypeError, ValueError):
                # A newly selected Annulus may not have had its sidebar value
                # populated yet. Use the canvas default rather than rejecting it.
                inner_diameter = view.canvas.get_roi_inner_diameter()
            max_inner = max(0, width - 1)
            if width > 1:
                inner_diameter = min(max_inner, max(1, inner_diameter))
            else:
                inner_diameter = 0

        if view.canvas.set_roi(
            x, y, width, height, notify=True, shape=shape,
            inner_diameter=inner_diameter,
        ):
            return
        self._sync_roi_sidebar_from_view(view)

    @staticmethod
    def _format_roi_stat(value: Any) -> str:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        try:
            number = float(value)
        except Exception:
            return str(value)
        if not math.isfinite(number):
            return str(number)
        if number == 0.0:
            return "0"
        magnitude = abs(number)
        if magnitude >= 1e7 or magnitude < 1e-4:
            return f"{number:.6e}"
        return f"{number:.6g}"

    def _update_roi_statistics(self, view: InternalViewState | None = None) -> None:
        view = view or self.active_view
        empty_vars = (
            self.roi_pixels_var, self.roi_min_var, self.roi_max_var,
            self.roi_mean_var, self.roi_sum_var, self.roi_std_var,
            self.roi_zeros_var,
        )
        if view is None or view.current_frame is None or view.canvas.get_roi() is None:
            for variable in empty_vars:
                variable.set("—")
            return

        values = np.asarray(view.canvas.roi_values(view.current_frame)).reshape(-1)
        if values.size == 0:
            for variable in empty_vars:
                variable.set("—")
            return

        self.roi_pixels_var.set(f"{int(values.size):,}")
        self.roi_zeros_var.set(f"{int(np.count_nonzero(values == 0)):,}")

        if values.dtype.kind == "f":
            finite_values = values[np.isfinite(values)]
        else:
            finite_values = values
        if finite_values.size == 0:
            for variable in (self.roi_min_var, self.roi_max_var, self.roi_mean_var, self.roi_sum_var, self.roi_std_var):
                variable.set("N/A")
            return

        work = finite_values.astype(np.float64, copy=False)
        self.roi_min_var.set(self._format_roi_stat(np.min(finite_values)))
        self.roi_max_var.set(self._format_roi_stat(np.max(finite_values)))
        self.roi_mean_var.set(self._format_roi_stat(np.mean(work, dtype=np.float64)))
        self.roi_sum_var.set(self._format_roi_stat(np.sum(work, dtype=np.float64)))
        self.roi_std_var.set(self._format_roi_stat(np.std(work, dtype=np.float64)))

    def _bind_keys(self) -> None:
        self.root.bind("<Left>", lambda _e: self.step_frame(-1))
        self.root.bind("<Right>", lambda _e: self.step_frame(1))
        self.root.bind("<Prior>", lambda _e: self.step_frame(-10))
        self.root.bind("<Next>", lambda _e: self.step_frame(10))
        self.root.bind("<Home>", lambda _e: self.load_frame(0))
        self.root.bind("<End>", lambda _e: self.load_frame(self.archive.frame_count - 1 if self.archive else 0))
        self.root.bind("<space>", lambda _e: self.toggle_playback())
        self.root.bind("<plus>", lambda _e: self._zoom_in())
        self.root.bind("<equal>", lambda _e: self._zoom_in())
        self.root.bind("<minus>", lambda _e: self._zoom_out())
        self.root.bind("<Control-o>", lambda _e: self.open_dialog())
        self.root.bind("<Control-n>", lambda _e: self.new_window())
        self.root.bind("<Control-w>", lambda _e: self.close_active_view())
        self.root.bind("<Control-Shift-C>", lambda _e: self.open_hdf5_to_hdxf_tool())
        self.root.bind("<Control-Shift-R>", lambda _e: self.open_hdxf_to_hdf5_tool())
        self.root.bind("<Control-Alt-s>", lambda _e: self.open_hdxf_sum_tool())
        self.root.bind("<Control-Alt-S>", lambda _e: self.open_hdxf_sum_tool())
        self.root.bind("<Control-Alt-d>", lambda _e: self.open_hdxf_subtract_tool())
        self.root.bind("<Control-Alt-D>", lambda _e: self.open_hdxf_subtract_tool())
        self.root.bind("<Control-Alt-p>", lambda _e: self.open_hdxf_pixelop_tool())
        self.root.bind("<Control-Alt-P>", lambda _e: self.open_hdxf_pixelop_tool())
        self.root.bind("<Key-f>", lambda _e: self._fit())

    def add_view(self, initial_path: Path | None = None) -> InternalViewState | None:
        if len(self.views) >= self.manager.MAX_VIEWS:
            messagebox.showinfo(
                "New View",
                f"A maximum of {self.manager.MAX_VIEWS} image views is supported.",
                parent=self.root,
            )
            return None

        view_id = self._next_view_id
        self._next_view_id += 1
        host = tk.Frame(
            self.view_grid,
            background=UI_CANVAS,
            highlightthickness=2,
            highlightbackground=UI_BORDER,
            highlightcolor=UI_BORDER,
            borderwidth=0,
        )
        title_bar = tk.Frame(host, background=UI_SURFACE, height=32)
        title_bar.pack(side="top", fill="x")
        title_var = tk.StringVar(value=f"VIEW {view_id:02d}   ·   NO FILE")
        title_label = tk.Label(
            title_bar,
            textvariable=title_var,
            anchor="w",
            background=UI_SURFACE,
            foreground=UI_TEXT_MUTED,
            padx=10,
            pady=5,
            font=("Segoe UI", 9, "bold"),
        )
        title_label.pack(side="left", fill="x", expand=True)

        holder: dict[str, InternalViewState] = {}

        def status_callback(info: dict[str, Any] | None) -> None:
            view = holder.get("view")
            if view is not None and view is self.active_view:
                self._update_pointer_status(info)

        def view_callback(sync_state: dict[str, float]) -> None:
            view = holder.get("view")
            if view is not None:
                self.manager.request_view_sync(view, sync_state)

        def beam_center_callback(center: tuple[int, int] | None) -> None:
            view = holder.get("view")
            if view is None or view is not self.active_view:
                return
            if center is None:
                self.status_var.set(
                    "Temporary beam center cleared · resolution rings hidden · source HDF5/HDXF unchanged"
                )
            else:
                x, y = center
                geometry = view.canvas.resolution_geometry
                if geometry is None:
                    self.status_var.set(
                        f"Temporary beam center anchored at pixel X={x}, Y={y} · "
                        "resolution rings unavailable: wavelength / detector distance / pixel size metadata missing · "
                        "source HDF5/HDXF unchanged"
                    )
                else:
                    self.status_var.set(
                        f"Temporary beam center anchored at pixel X={x}, Y={y} · "
                        f"resolution rings shown (λ={geometry['wavelength_angstrom']:.5g} Å, "
                        f"distance={geometry['distance_m']:.5g} m) · source HDF5/HDXF unchanged"
                    )

        def roi_callback(roi: tuple[int, int, int, int] | None) -> None:
            view = holder.get("view")
            if view is None or view is not self.active_view:
                return
            self._sync_roi_sidebar_from_view(view)
            if roi is None:
                self.status_var.set("ROI cleared · source HDF5/HDXF unchanged")
            else:
                x, y, width, height = roi
                shape = view.canvas.get_roi_shape()
                if shape == "Annulus":
                    self.status_var.set(
                        f"Annulus ROI · X={x}, Y={y}, Outer={width}px, "
                        f"Inner={view.canvas.get_roi_inner_diameter()}px excluded · "
                        f"source HDF5/HDXF unchanged"
                    )
                else:
                    self.status_var.set(
                        f"{shape} ROI · X={x}, Y={y}, Width={width}, Height={height} · "
                        f"X2={x + width - 1}, Y2={y + height - 1} · source HDF5/HDXF unchanged"
                    )

        canvas = ImageCanvas(
            host,
            status_callback,
            view_callback,
            beam_center_callback,
            roi_callback,
        )
        canvas.pack(side="top", fill="both", expand=True)
        template = self.active_view
        if template is None:
            try:
                inherited_min = float(self.min_var.get())
            except (TypeError, ValueError):
                inherited_min = 0.0
            try:
                inherited_max = float(self.max_var.get())
            except (TypeError, ValueError):
                inherited_max = 1.0
            if not math.isfinite(inherited_min) or not math.isfinite(inherited_max) or inherited_max <= inherited_min:
                inherited_min, inherited_max = 0.0, 1.0
        else:
            inherited_min = template.display_min
            inherited_max = template.display_max
        view = InternalViewState(
            view_id=view_id,
            host=host,
            title_var=title_var,
            canvas=canvas,
            title_bar=title_bar,
            display_min=inherited_min,
            display_max=inherited_max,
            contrast_mode=(template.contrast_mode if template is not None else self.contrast_mode_var.get()),
            invert=(template.invert if template is not None else bool(self.invert_var.get())),
            show_values=(template.show_values if template is not None else bool(self.show_values_var.get())),
        )
        canvas.set_show_values(view.show_values)
        holder["view"] = view
        self.views.append(view)

        def activate(_event: tk.Event | None = None, selected: InternalViewState = view) -> None:
            self.activate_view(selected, refresh=True)

        for widget in (host, title_bar, title_label, canvas):
            widget.bind("<ButtonPress-1>", activate, add="+")
            widget.bind("<ButtonPress-2>", activate, add="+")
            widget.bind("<ButtonPress-3>", activate, add="+")
        if self._file_drop_controller.enabled:
            self._file_drop_controller.register_targets((host, title_bar, canvas))
        self.relayout_views()
        self.activate_view(view, refresh=True)
        if initial_path is not None:
            self.root.after(50, lambda v=view, path=initial_path: self.open_path_in_view(v, path))
        return view

    def new_window(self) -> None:
        self.manager.create_window()

    def close_active_view(self) -> None:
        view = self.active_view
        if view is None:
            return
        self.stop_playback()
        if view.archive is not None:
            view.archive.close()
            view.archive = None
        try:
            index = self.views.index(view)
        except ValueError:
            return
        view.host.destroy()
        self.views.remove(view)
        if not self.views:
            self.active_view_index = -1
            replacement = self.add_view()
            if replacement is None:
                return
        else:
            self.active_view_index = min(index, len(self.views) - 1)
            self.relayout_views()
            self.activate_view(self.views[self.active_view_index], refresh=True)

    def relayout_views(self) -> None:
        if not hasattr(self, "view_grid"):
            return
        for child in self.views:
            child.host.grid_forget()
        for row in range(2):
            self.view_grid.grid_rowconfigure(row, weight=0)
        for col in range(2):
            self.view_grid.grid_columnconfigure(col, weight=0)

        count = len(self.views)
        self.view_count_var.set(f"{count} / {self.manager.MAX_VIEWS} VIEWS")
        if count <= 1:
            positions = [(0, 0, 1)]
            rows, cols = 1, 1
        elif count == 2:
            positions = [(0, 0, 1), (0, 1, 1)]
            rows, cols = 1, 2
        elif count == 3:
            positions = [(0, 0, 1), (0, 1, 1), (1, 0, 2)]
            rows, cols = 2, 2
        else:
            positions = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
            rows, cols = 2, 2

        for row in range(rows):
            self.view_grid.grid_rowconfigure(row, weight=1, uniform="view-row")
        for col in range(cols):
            self.view_grid.grid_columnconfigure(col, weight=1, uniform="view-col")
        for view, (row, col, colspan) in zip(self.views, positions):
            view.host.grid(
                row=row,
                column=col,
                columnspan=colspan,
                sticky="nsew",
                padx=3,
                pady=3,
            )
        self.root.after_idle(self._fit_empty_or_new_views)

    def _fit_empty_or_new_views(self) -> None:
        for view in self.views:
            if view.current_frame is not None and view.canvas.zoom <= 0.011:
                view.canvas.fit_to_window(notify=False)

    def activate_view(self, view: InternalViewState | None, *, refresh: bool = True) -> None:
        if view is None or view not in self.views:
            return
        if self.playing and view is not self.active_view:
            self.stop_playback()
        self.active_view_index = self.views.index(view)
        for item in self.views:
            active = item is view
            border = UI_ACCENT if active else UI_BORDER
            bar = UI_ACCENT_DARK if active else UI_SURFACE
            item.host.configure(highlightbackground=border, highlightcolor=border)
            if item.title_bar is not None:
                item.title_bar.configure(background=bar)
                for child in item.title_bar.winfo_children():
                    try:
                        child.configure(
                            background=bar,
                            foreground=UI_WHITE if active else UI_TEXT_MUTED,
                        )
                    except tk.TclError:
                        pass
        active_name = view.archive.path.name if view.archive is not None else "NO FILE"
        if len(active_name) > 26:
            active_name = f"{active_name[:11]}…{active_name[-12:]}"
        self.active_view_var.set(f"VIEW {view.view_id:02d} · {active_name}")
        self.contrast_mode_var.set(view.contrast_mode)
        self.invert_var.set(view.invert)
        self.show_values_var.set(view.show_values)
        self.min_var.set(f"{view.display_min:.8g}")
        self.max_var.set(f"{view.display_max:.8g}")
        view.canvas.set_show_values(view.show_values)
        self._sync_roi_sidebar_from_view(view)
        self._histogram_domain = None
        self._histogram_axis = None
        self._histogram_values_cache = None
        self._histogram_cache_key = None
        self._histogram_logs_cache = None
        if view.archive is None:
            self.frame_spin.configure(from_=1, to=1)
            self.frame_var.set("1")
            self.frame_label_var.set("Image 0 / 0")
            self.file_summary_var.set(f"View {view.view_id}: No file open")
            self.status_var.set(f"View {view.view_id} selected")
        else:
            self.frame_spin.configure(from_=1, to=max(1, view.archive.frame_count))
            self._set_frame_controls(view.current_frame_index)
            format_label = getattr(view.archive, "file_format", "DATA")
            self.file_summary_var.set(
                f"{view.archive.path.name} | {format_label} | {view.archive.frame_count} images | "
                f"{view.archive.frame_shape} | {view.archive.dtype}"
            )
            self._update_pointer_status(None)
        if refresh:
            self._update_thumbnail()
            self._update_histogram()
            self._update_image_info()
            self._update_metadata()
        self._refresh_title()

    def _toggle_show_values(self) -> None:
        value = bool(self.show_values_var.get())
        self._apply_display_settings_to_all(show_values=value, rebuild=False)
        self.status_var.set(
            f"Pixel-value overlay {'enabled' if value else 'disabled'} for all {len(self.views)} views"
        )

    def _toggle_invert(self) -> None:
        value = bool(self.invert_var.get())
        self._apply_display_settings_to_all(invert=value, rebuild=True)
        self._refresh_active_side_panels()
        self.status_var.set(
            f"Grayscale inversion {'enabled' if value else 'disabled'} for all {len(self.views)} views"
        )

    def _refresh_active_side_panels(self) -> None:
        """Refresh shared controls for the currently selected pane only."""
        self._update_thumbnail()
        if not self._playback_rendering:
            self._update_histogram()
        self._update_pointer_status(None)
        self._update_roi_statistics(self.active_view)

    def _on_canvas_view_changed(self, state: dict[str, float]) -> None:
        view = self.active_view
        if view is not None:
            self.manager.request_view_sync(view, state)

    def _refresh_title(self) -> None:
        view = self.active_view
        suffix = ""
        if view is not None:
            suffix = f" — {view.label()}"
        self.root.title(f"HDXF Viewer 0.5.2.0 · HDF5 · ROI DEEPER BLUE SHADE · MOVE + RESIZE{suffix}")

    def _drop_target_widgets(self) -> tuple[tk.Misc, ...]:
        targets: list[tk.Misc] = []
        if hasattr(self, "view_grid"):
            targets.append(self.view_grid)
        for view in self.views:
            targets.extend((view.host, view.title_bar, view.canvas))
        return tuple(target for target in targets if target is not None)

    def _install_file_drop_support(self) -> None:
        """Enable Explorer drops after all first-view widgets are mapped."""
        status = self._file_drop_controller.install(self._drop_target_widgets())
        if status.startswith("tkdnd"):
            self.status_var.set(f"Ready · drag-and-drop enabled · {status}")
            return

        detail = str(getattr(self.root, "_hdxf_dnd_error", "")).strip()
        if detail and "not installed" not in detail.lower():
            self.status_var.set(f"Ready · drag-and-drop disabled · {detail}")
        else:
            self.status_var.set(
                "Ready · drag-and-drop disabled; run: python -m pip install tkinterdnd2"
            )

    @staticmethod
    def _widget_is_inside(widget: tk.Misc | None, ancestor: tk.Misc) -> bool:
        current = widget
        while current is not None:
            if current is ancestor:
                return True
            current = getattr(current, "master", None)
        return False

    def _view_at_screen_point(self, screen_x: int, screen_y: int) -> InternalViewState | None:
        try:
            widget = self.root.winfo_containing(int(screen_x), int(screen_y))
        except tk.TclError:
            return None
        for view in self.views:
            if self._widget_is_inside(widget, view.host):
                return view
        return None

    @staticmethod
    def _path_is_supported_detector_file(path: Path) -> bool:
        if not path.is_file():
            return False
        suffix = path.suffix.lower()
        if suffix == ".hdxf":
            return True
        if suffix in (".h5", ".hdf5", ".nxs"):
            return True
        if h5py is not None:
            try:
                return bool(h5py.is_hdf5(str(path)))
            except Exception:
                pass
        try:
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path, "r") as archive:
                    return MANIFEST_PATH in archive.namelist()
        except Exception:
            pass
        return False

    def _handle_dropped_files(
        self,
        raw_paths: tuple[str, ...] | list[str],
        screen_x: int | None = None,
        screen_y: int | None = None,
    ) -> None:
        """Open one or more dropped HDXF/HDF5 files in internal image views."""
        accepted: list[Path] = []
        rejected: list[str] = []
        seen: set[str] = set()
        for raw in raw_paths:
            path = Path(str(raw).strip().strip('"'))
            key = os.path.normcase(os.path.abspath(str(path)))
            if key in seen:
                continue
            seen.add(key)
            if not self._path_is_supported_detector_file(path):
                rejected.append(str(path))
                continue
            accepted.append(path)

        if not accepted:
            if rejected:
                messagebox.showwarning(
                    "Unsupported drop",
                    "Open an existing HDXF or HDF5 detector file (.hdxf, .h5, .hdf5).",
                    parent=self.root,
                )
            return

        target: InternalViewState | None = None
        if screen_x is not None and screen_y is not None:
            target = self._view_at_screen_point(screen_x, screen_y)
        target = target or self.active_view
        if target is None:
            target = self.add_view()
        if target is None:
            return

        opened = 0
        skipped: list[Path] = []
        used_views: set[int] = set()
        for index, path in enumerate(accepted):
            if index == 0:
                destination = target
            else:
                destination = next(
                    (
                        view
                        for view in self.views
                        if view.view_id not in used_views and view.archive is None
                    ),
                    None,
                )
                if destination is None and len(self.views) < self.manager.MAX_VIEWS:
                    destination = self.add_view()
                if destination is None:
                    skipped.extend(accepted[index:])
                    break

            used_views.add(destination.view_id)
            before = destination.archive
            self.open_path_in_view(destination, path)
            if destination.archive is not None and destination.archive is not before:
                opened += 1

        if opened:
            self.status_var.set(
                f"Opened {opened} dropped detector file{'s' if opened != 1 else ''}"
            )
        messages: list[str] = []
        if skipped:
            messages.append(
                f"{len(skipped)} file(s) were not opened because the four-view limit was reached."
            )
        if rejected:
            messages.append(
                f"{len(rejected)} item(s) were ignored because they were not readable HDXF/HDF5 files."
            )
        if messages:
            messagebox.showinfo("Drag-and-drop result", "\n".join(messages), parent=self.root)

    def open_dialog(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self.root,
            title="Open detector data",
            filetypes=[
                ("Detector data (HDXF/HDF5)", "*.hdxf *.h5 *.hdf5 *.nxs"),
                ("HDXF detector archive", "*.hdxf"),
                ("HDF5 detector data", "*.h5 *.hdf5 *.nxs"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            self.open_path(Path(filename))

    def open_path_in_view(self, view: InternalViewState, path: Path) -> None:
        if view not in self.views:
            return
        self.activate_view(view, refresh=False)
        self.open_path(path)

    def _open_detector_source(self, path: Path) -> HDXFArchive | HDF5Archive:
        path = Path(path).resolve()
        if not path.is_file():
            raise HDXFViewerError(f"File does not exist: {path}")

        suffix = path.suffix.lower()
        hdf5_signature = False
        if h5py is not None:
            try:
                hdf5_signature = bool(h5py.is_hdf5(str(path)))
            except Exception:
                hdf5_signature = False

        # Signature wins over extension. This also supports HDF5 files with a
        # non-standard .nxs or vendor-specific extension.
        if hdf5_signature:
            return HDF5Archive(path, block_cache_size=8)

        if suffix == ".hdxf" or zipfile.is_zipfile(path):
            try:
                return HDXFArchive(path, block_cache_size=8)
            except Exception as hdxf_exc:
                # A mislabeled HDF5 file can still recover here if h5py's
                # signature probe failed for an unusual platform/build.
                if h5py is not None:
                    try:
                        return HDF5Archive(path, block_cache_size=8)
                    except Exception:
                        pass
                raise hdxf_exc

        if h5py is not None:
            try:
                return HDF5Archive(path, block_cache_size=8)
            except Exception as hdf5_exc:
                try:
                    return HDXFArchive(path, block_cache_size=8)
                except Exception:
                    raise hdf5_exc

        return HDXFArchive(path, block_cache_size=8)

    def _resolution_geometry_for_view(
        self,
        view: InternalViewState,
    ) -> dict[str, float] | None:
        """Read detector geometry needed for resolution rings.

        This is a read-only metadata lookup for both direct HDF5 and HDXF.
        The manually anchored beam center intentionally overrides any beam
        center stored in the source file, so beam-center metadata is not read
        here.
        """
        archive = view.archive
        if archive is None:
            return None

        def find_number(
            candidates: tuple[str, ...],
            suffixes: tuple[str, ...],
            *,
            per_frame: bool = False,
        ) -> tuple[float | None, str | None, str]:
            try:
                value, path = archive.find_hdf5_value(candidates, suffixes=suffixes)
            except Exception:
                return None, None, ""
            if value is None:
                return None, path, ""
            number = _numeric_scalar(
                value,
                index=view.current_frame_index if per_frame else None,
            )
            units = ""
            if path:
                try:
                    attrs = archive.hdf5_dataset_attributes(path)
                    raw_units = attrs.get("units") or attrs.get("unit")
                    if raw_units is not None:
                        units = _scalar_text(_decode_tagged_value(raw_units)).strip()
                except Exception:
                    units = ""
            return number, path, units

        def length_to_m(value: float | None, units: str) -> float | None:
            if value is None:
                return None
            unit = units.strip().lower().replace("μ", "µ")
            factors = {
                "m": 1.0,
                "meter": 1.0,
                "metre": 1.0,
                "mm": 1e-3,
                "cm": 1e-2,
                "um": 1e-6,
                "µm": 1e-6,
                "nm": 1e-9,
            }
            # NeXus detector geometry normally stores SI metres.  Preserve the
            # existing Viewer convention when units are absent.
            return float(value) * factors.get(unit, 1.0)

        def wavelength_to_angstrom(value: float | None, units: str) -> float | None:
            if value is None:
                return None
            unit = units.strip().lower().replace("ångström", "angstrom")
            factors = {
                "m": 1e10,
                "nm": 10.0,
                "pm": 0.01,
                "angstrom": 1.0,
                "angstroms": 1.0,
                "a": 1.0,
                "å": 1.0,
            }
            # NeXus incident_wavelength is commonly stored in angstrom when a
            # units attribute is present; if units are absent, values below
            # 1e-6 are overwhelmingly likely to be metres.
            if not unit:
                return float(value) * 1e10 if abs(float(value)) < 1e-6 else float(value)
            return float(value) * factors.get(unit, 1.0)

        x_pixel, _path, x_units = find_number(
            (
                "/entry/instrument/detector/x_pixel_size",
                "/entry/instrument/detector/detectorSpecific/x_pixel_size",
            ),
            ("/x_pixel_size",),
        )
        y_pixel, _path, y_units = find_number(
            (
                "/entry/instrument/detector/y_pixel_size",
                "/entry/instrument/detector/detectorSpecific/y_pixel_size",
            ),
            ("/y_pixel_size",),
        )
        distance, _path, distance_units = find_number(
            (
                "/entry/instrument/detector/detector_distance",
                "/entry/instrument/detector/distance",
            ),
            ("/detector_distance", "/detector/distance"),
            per_frame=True,
        )
        wavelength, _path, wavelength_units = find_number(
            (
                "/entry/instrument/beam/incident_wavelength",
                "/entry/instrument/beam/wavelength",
            ),
            ("/incident_wavelength", "/beam/wavelength"),
            per_frame=True,
        )

        pixel_x_m = length_to_m(x_pixel, x_units)
        pixel_y_m = length_to_m(y_pixel, y_units)
        distance_m = length_to_m(distance, distance_units)
        wavelength_angstrom = wavelength_to_angstrom(wavelength, wavelength_units)

        if wavelength_angstrom is None:
            energy, _path, energy_units = find_number(
                (
                    "/entry/instrument/beam/incident_energy",
                    "/entry/instrument/beam/energy",
                ),
                ("/incident_energy", "/beam/energy"),
                per_frame=True,
            )
            if energy is not None:
                unit = energy_units.strip().lower()
                energy_ev = float(energy) * (1000.0 if unit == "kev" else 1.0)
                if energy_ev > 0.0:
                    wavelength_angstrom = 12398.419843320026 / energy_ev

        values = (pixel_x_m, pixel_y_m, distance_m, wavelength_angstrom)
        if any(value is None or not math.isfinite(float(value)) or float(value) <= 0.0 for value in values):
            return None
        return {
            "pixel_size_x_m": float(pixel_x_m),
            "pixel_size_y_m": float(pixel_y_m),
            "distance_m": float(distance_m),
            "wavelength_angstrom": float(wavelength_angstrom),
        }

    def _refresh_resolution_geometry(self, view: InternalViewState) -> None:
        geometry = self._resolution_geometry_for_view(view)
        view.canvas.set_resolution_geometry(geometry)

    def open_path(self, path: Path) -> None:
        view = self.active_view
        if view is None:
            view = self.add_view()
            if view is None:
                return
        self.stop_playback()
        path = Path(path).resolve()
        self.status_var.set(f"Opening {path} …")
        self.root.update_idletasks()
        new_archive = None
        try:
            new_archive = self._open_detector_source(path)
            frame = new_archive.get_frame(0)
        except Exception as exc:
            if new_archive is not None:
                try:
                    new_archive.close()
                except Exception:
                    pass
            messagebox.showerror(
                "Open failed",
                f"{type(exc).__name__}: {exc}\n\n"
                "The viewer tried format detection and safe HDF5 fallbacks. "
                "Missing metadata alone does not prevent opening; this error means no readable image source could be obtained.",
                parent=self.root,
            )
            self.status_var.set("Open failed")
            return

        if view.archive is not None:
            view.archive.close()
        # Beam-center anchors belong only to the currently opened source.
        # Clear after the new file has opened successfully; a failed open keeps
        # the previous file and its temporary anchor untouched.
        view.canvas.clear_manual_beam_center(notify=False)
        view.canvas.clear_roi(notify=False)
        view.archive = new_archive
        view.current_frame_index = 0
        view.current_frame = frame
        view.current_display = None
        view.display_min = 0.0
        view.display_max = 1.0
        view.title_var.set(f"VIEW {view.view_id:02d}   ·   {path.name}")
        chip_name = path.name if len(path.name) <= 26 else f"{path.name[:11]}…{path.name[-12:]}"
        self.active_view_var.set(f"VIEW {view.view_id:02d} · {chip_name}")
        self._histogram_domain = None
        self._histogram_axis = None
        self._histogram_values_cache = None
        self._histogram_cache_key = None
        self._histogram_logs_cache = None
        self.frame_spin.configure(from_=1, to=max(1, view.archive.frame_count))
        self._set_frame_controls(0)
        self._refresh_resolution_geometry(view)
        self._sync_roi_sidebar_from_view(view)
        self.auto_contrast()
        view.canvas.fit_to_window(notify=False)
        # Side panels are deliberately best-effort. A malformed metadata field
        # must never undo a successful image open.
        try:
            self._update_image_info()
        except Exception:
            pass
        try:
            self._update_metadata()
        except Exception:
            pass
        self._refresh_title()
        format_label = getattr(view.archive, "file_format", "DATA")
        self.file_summary_var.set(
            f"{path.name} | {format_label} | {view.archive.frame_count} images | "
            f"{view.archive.frame_shape} | {view.archive.dtype}"
        )
        self.status_var.set(
            f"{path.name} | {format_label} | frames={view.archive.frame_count} | "
            f"shape={view.archive.frame_shape} | dtype={view.archive.dtype}"
        )

    def _shutdown_resources(self) -> None:
        self.stop_playback()
        self._file_drop_controller.close()
        process = self._converter_process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass
        self._converter_run_serial += 1
        self._converter_process = None
        restore_process = self._restore_process
        if restore_process is not None and restore_process.poll() is None:
            try:
                restore_process.terminate()
            except Exception:
                pass
        self._restore_run_serial += 1
        self._restore_process = None
        processing_process = self._processing_process
        if processing_process is not None and processing_process.poll() is None:
            try:
                processing_process.terminate()
            except Exception:
                pass
        self._processing_run_serial += 1
        self._processing_process = None
        if self._processing_poll_after_id is not None:
            try:
                self.root.after_cancel(self._processing_poll_after_id)
            except tk.TclError:
                pass
            self._processing_poll_after_id = None
        restore_log = getattr(self, "_restore_log_file", None)
        if restore_log is not None:
            try:
                restore_log.close()
            except Exception:
                pass
            self._restore_log_file = None
        for view in list(self.views):
            if view.archive is not None:
                view.archive.close()
                view.archive = None

    def close(self) -> None:
        self.manager.close_window(self)

    def load_frame(
        self,
        index: int,
        *,
        recompute_contrast: bool = True,
        sync_controls: bool = True,
        playback: bool = False,
        notify_sync: bool = True,
    ) -> bool:
        if self.archive is None:
            return False
        index = max(0, min(self.archive.frame_count - 1, int(index)))
        self.status_var.set(f"Reading image {index + 1} …")
        self.root.update_idletasks()
        try:
            frame = self.archive.get_frame(index)
        except Exception as exc:
            self.stop_playback()
            messagebox.showerror("Read failed", f"{type(exc).__name__}: {exc}")
            self.status_var.set(f"Failed to read image {index + 1}")
            return False

        self.current_frame_index = index
        self.current_frame = frame
        active_view = self.active_view
        if active_view is not None:
            self._refresh_resolution_geometry(active_view)
        self._histogram_domain = None
        self._histogram_axis = None
        self._histogram_values_cache = None
        self._histogram_cache_key = None
        self._histogram_logs_cache = None
        if sync_controls:
            self._set_frame_controls(index)
        self._playback_rendering = bool(playback)
        try:
            if recompute_contrast:
                self.auto_contrast()
            else:
                self._rebuild_display()
        finally:
            self._playback_rendering = False
        self._update_image_info()
        self._update_roi_statistics(self.active_view)
        if not playback or index % 5 == 0:
            self._update_metadata()
        if playback:
            self.status_var.set(
                f"Playing | Image {index + 1} / {self.archive.frame_count} | {self.fps_var.get()} FPS"
            )
        if notify_sync:
            source_view = self.active_view
            if source_view is not None:
                self.manager.request_frame_sync(source_view, index)
        return True

    def load_frame_for_view(
        self,
        view: InternalViewState,
        index: int,
        *,
        recompute_contrast: bool = False,
        playback: bool = False,
    ) -> bool:
        """Load an inactive view without moving the shared sidebar focus."""
        if view not in self.views or view.archive is None:
            return False
        if view is self.active_view:
            return self.load_frame(
                index,
                recompute_contrast=recompute_contrast,
                sync_controls=True,
                playback=playback,
                notify_sync=False,
            )
        index = max(0, min(view.archive.frame_count - 1, int(index)))
        try:
            frame = view.archive.get_frame(index)
        except Exception:
            return False
        view.current_frame_index = index
        view.current_frame = frame
        self._refresh_resolution_geometry(view)
        if recompute_contrast:
            source = frame
            if view.contrast_mode == AUTO_CONTRAST_VISIBLE_SPARSE:
                bounds = view.canvas.visible_source_bounds()
                if bounds is not None:
                    sx0, sy0, sx1, sy1 = bounds
                    crop = frame[sy0:sy1, sx0:sx1]
                    if crop.size >= 32:
                        source = crop
            low, high, _stats = calculate_auto_contrast(source, mode=view.contrast_mode)
            view.display_min = float(low)
            view.display_max = float(high)
        low = view.display_min
        high = view.display_max
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            low, high = 0.0, 1.0
        scale = 255.0 / (high - low)
        work = np.asarray(frame, dtype=np.float32)
        display = np.clip((work - low) * scale, 0.0, 255.0).astype(np.uint8)
        if view.invert:
            display = 255 - display
        view.current_display = display
        view.canvas.set_frame(frame, display)
        view.canvas.set_show_values(view.show_values)
        view.title_var.set(view.label())
        return True

    def step_frame(self, delta: int) -> None:
        if self.archive is None:
            return
        self.load_frame(self.current_frame_index + delta)

    def _load_frame_from_entry(self, _event: tk.Event | None = None) -> None:
        if self.archive is None:
            return
        try:
            image_number = int(self.frame_var.get())
        except ValueError:
            self.frame_var.set(str(self.current_frame_index + 1))
            return
        self.load_frame(image_number - 1)

    def _set_frame_controls(self, index: int) -> None:
        if self.archive is None:
            return
        image_number = index + 1
        self.frame_var.set(str(image_number))
        format_label = getattr(self.archive, "file_format", "DATA")
        self.frame_label_var.set(f"/ {self.archive.frame_count}   ({format_label} {index})")

    def toggle_playback(self) -> None:
        if self.playing:
            self.stop_playback()
        else:
            self.start_playback()

    def start_playback(self) -> None:
        if self.archive is None or self.playing:
            return
        self.playing = True
        self._play_generation += 1
        # Keep pixel-value overlays active during playback when the user has
        # enabled them. This favors inspection accuracy over maximum FPS.
        self.canvas.set_show_values(bool(self.show_values_var.get()))
        self.play_button.configure(text="Ⅱ  PAUSE")
        self._schedule_next_play_frame(delay_ms=1, generation=self._play_generation)

    def stop_playback(self) -> None:
        self.playing = False
        self._play_generation += 1
        if self._play_after_id is not None:
            try:
                self.root.after_cancel(self._play_after_id)
            except tk.TclError:
                pass
            self._play_after_id = None
        if hasattr(self, "play_button"):
            self.play_button.configure(text="▶  PLAY")
        self.canvas.set_show_values(bool(self.show_values_var.get()))
        if self.current_frame is not None and hasattr(self, "histogram_canvas"):
            self._update_histogram()
            self._update_image_info()
            self._update_metadata()

    def _schedule_next_play_frame(self, *, delay_ms: int, generation: int) -> None:
        if not self.playing or generation != self._play_generation:
            return
        self._play_after_id = self.root.after(
            max(1, int(delay_ms)),
            lambda: self._play_tick(generation),
        )

    def _play_tick(self, generation: int) -> None:
        self._play_after_id = None
        if not self.playing or self.archive is None or generation != self._play_generation:
            return
        started = time.perf_counter()
        next_index = self.current_frame_index + 1
        if next_index >= self.archive.frame_count:
            if self.loop_var.get():
                next_index = 0
            else:
                self.stop_playback()
                return
        ok = self.load_frame(
            next_index,
            recompute_contrast=False,
            sync_controls=True,
            playback=True,
        )
        if not ok or not self.playing:
            return
        try:
            fps = max(0.2, float(self.fps_var.get()))
        except ValueError:
            fps = 5.0
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        delay_ms = max(1, int(round(1000.0 / fps - elapsed_ms)))
        self._schedule_next_play_frame(delay_ms=delay_ms, generation=generation)

    def auto_contrast(self) -> None:
        if self.current_frame is None:
            return
        mode = self.contrast_mode_var.get()
        self._apply_display_settings_to_all(mode=mode, rebuild=False)
        source = self.current_frame
        source_name = "whole frame"
        if mode == AUTO_CONTRAST_VISIBLE_SPARSE:
            bounds = self.canvas.visible_source_bounds()
            if bounds is not None:
                sx0, sy0, sx1, sy1 = bounds
                crop = self.current_frame[sy0:sy1, sx0:sx1]
                if crop.size >= 32:
                    source = crop
                    source_name = f"visible {sx0}:{sx1}, {sy0}:{sy1}"
        low, high, stats = calculate_auto_contrast(source, mode=mode)
        # Recalculate the display window. The histogram axis itself remains
        # the current frame's exact pixel minimum-to-maximum interval.
        self._histogram_domain = None
        self._histogram_axis = None
        self._histogram_cache_key = None
        self._histogram_logs_cache = None
        self._set_contrast(low, high)
        zero_pct = 100.0 * float(stats.get("zero_fraction", 0.0))
        self.status_var.set(
            f"{mode} | {source_name} | range={low:.6g}…{high:.6g} | zero={zero_pct:.1f}%"
        )

    def full_contrast(self) -> None:
        if self.current_frame is None:
            return
        array = self.current_frame
        finite = array[np.isfinite(array)] if array.dtype.kind == "f" else array.reshape(-1)
        if finite.size == 0:
            return
        low = float(np.min(finite))
        high = float(np.max(finite))
        if high <= low:
            high = low + 1.0
        self._histogram_domain = None
        self._histogram_axis = None
        self._histogram_cache_key = None
        self._histogram_logs_cache = None
        self._set_contrast(low, high)

    def apply_contrast(self) -> None:
        try:
            low = float(self.min_var.get())
            high = float(self.max_var.get())
        except ValueError:
            messagebox.showerror("Display range", "Min and Max must be numeric")
            return
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            messagebox.showerror("Display range", "Max must be greater than Min")
            return
        self._set_contrast(low, high)

    def _set_contrast(self, low: float, high: float) -> None:
        low = float(low)
        high = float(high)
        self.min_var.set(f"{low:.8g}")
        self.max_var.set(f"{high:.8g}")
        self._apply_display_settings_to_all(low=low, high=high, rebuild=True)
        self._refresh_active_side_panels()

    def _rebuild_display(self) -> None:
        """Rebuild only the active pane after a frame change."""
        view = self.active_view
        if view is None or view.current_frame is None:
            return
        self._render_view_display(view)
        self._refresh_active_side_panels()

    def _update_thumbnail(self) -> None:
        canvas = self.thumbnail_canvas
        canvas.delete("all")
        canvas.update_idletasks()
        width = max(40, canvas.winfo_width() or 320)
        height = max(40, canvas.winfo_height() or 170)
        canvas.configure(background=UI_HISTOGRAM)
        if self.current_display is None:
            canvas.create_text(width / 2, height / 2, text="No image", fill=UI_TEXT_MUTED, font=("Segoe UI", 10, "bold"))
            return

        image = Image.fromarray(self.current_display, mode="L")
        # Leave enough room for a permanent white outline that is independent
        # of the grayscale inversion state.
        image.thumbnail((width - 18, height - 18), Image.Resampling.BILINEAR)
        image_w, image_h = image.size
        center_x = width / 2.0
        center_y = height / 2.0
        x0 = center_x - image_w / 2.0
        y0 = center_y - image_h / 2.0
        x1 = x0 + image_w
        y1 = y0 + image_h

        # A dark outer shadow and fixed white inner border remain visible for
        # both black and white detector images.
        canvas.create_rectangle(x0 - 4, y0 - 4, x1 + 4, y1 + 4, outline="#000000", width=2)
        canvas.create_rectangle(x0 - 2, y0 - 2, x1 + 2, y1 + 2, outline="#ffffff", width=2)
        self._thumbnail_photo = ImageTk.PhotoImage(image)
        canvas.create_image(center_x, center_y, image=self._thumbnail_photo, anchor="center")

        # Draw a two-tone crosshair so it stays visible on either polarity.
        for offset, color, line_width in ((1, "#000000", 3), (0, "#ffffff", 1)):
            canvas.create_line(
                center_x - 7 + offset,
                center_y + offset,
                center_x + 7 + offset,
                center_y + offset,
                fill=color,
                width=line_width,
            )
            canvas.create_line(
                center_x + offset,
                center_y - 7 + offset,
                center_x + offset,
                center_y + 7 + offset,
                fill=color,
                width=line_width,
            )

    def _ensure_histogram_domain(self, values: np.ndarray) -> tuple[float, float]:
        integer = self.current_frame is not None and self.current_frame.dtype.kind in "iu"
        if self._histogram_domain is None:
            self._histogram_domain = _histogram_domain_for_values(
                values,
                display_low=self.display_min,
                display_high=self.display_max,
                integer=integer,
            )
        if self._histogram_axis is None:
            self._histogram_axis = HistogramIntensityAxis.from_values(
                values,
                raw_low=self._histogram_domain[0],
                raw_high=self._histogram_domain[1],
                integer=integer,
            )
        # Exact pixel extrema remain the hard limits; only the mouse coordinate
        # is non-linear so low-count detector values remain controllable.
        return self._histogram_domain

    def _histogram_value_to_x(self, value: float) -> float:
        if self._histogram_axis is None or self._histogram_plot_rect is None:
            return 0.0
        x0, _y0, x1, _y1 = self._histogram_plot_rect
        fraction = self._histogram_axis.raw_to_fraction(float(value))
        return x0 + fraction * (x1 - x0)

    def _histogram_x_to_axis(self, x: float) -> float:
        if self._histogram_axis is None or self._histogram_plot_rect is None:
            return 0.0
        x0, _y0, x1, _y1 = self._histogram_plot_rect
        if x1 <= x0:
            return self._histogram_axis.axis_low
        fraction = min(1.0, max(0.0, (float(x) - x0) / (x1 - x0)))
        return self._histogram_axis.axis_low + fraction * self._histogram_axis.axis_span

    def _histogram_x_to_value(self, x: float) -> float:
        if self._histogram_axis is None:
            return self.display_min
        return self._histogram_axis.axis_to_raw(self._histogram_x_to_axis(x))

    def _histogram_minimum_axis_gap(self) -> float:
        if self._histogram_axis is None:
            return 1e-9
        # Around one quarter of a screen pixel at normal histogram widths.
        # Unlike a raw-domain gap this does not become hundreds of thousands
        # of counts when one hot pixel stretches the frame maximum.
        return max(self._histogram_axis.axis_span / 4096.0, 1e-12)

    def _histogram_minimum_gap(self) -> float:
        if self._histogram_axis is None:
            return 1e-9
        centre_axis = 0.5 * (
            self._histogram_axis.raw_to_axis(self.display_min)
            + self._histogram_axis.raw_to_axis(self.display_max)
        )
        half = self._histogram_minimum_axis_gap() / 2.0
        low = self._histogram_axis.axis_to_raw(centre_axis - half)
        high = self._histogram_axis.axis_to_raw(centre_axis + half)
        return max(high - low, 1e-12)

    def _histogram_normalize_range(self, low: float, high: float) -> tuple[float, float]:
        axis = self._histogram_axis
        if axis is None:
            return float(low), float(high)
        low_axis = axis.raw_to_axis(float(low))
        high_axis = axis.raw_to_axis(float(high))
        if high_axis < low_axis:
            low_axis, high_axis = high_axis, low_axis
        gap = self._histogram_minimum_axis_gap()
        if high_axis - low_axis < gap:
            centre = (low_axis + high_axis) / 2.0
            low_axis = centre - gap / 2.0
            high_axis = centre + gap / 2.0
        span = min(axis.axis_span, max(gap, high_axis - low_axis))
        low_axis = max(axis.axis_low, min(low_axis, axis.axis_high - span))
        high_axis = low_axis + span
        return axis.axis_to_raw(low_axis), axis.axis_to_raw(high_axis)

    def _histogram_flush_pending_range(self) -> None:
        self._histogram_render_after_id = None
        pending = self._histogram_pending_range
        self._histogram_pending_range = None
        if pending is None:
            return
        for view in self.views:
            if view.current_frame is not None:
                self._render_view_display(view)
        self._update_thumbnail()

    def _histogram_apply_range(self, low: float, high: float, *, final: bool) -> None:
        low, high = self._histogram_normalize_range(low, high)
        self.min_var.set(f"{low:.8g}")
        self.max_var.set(f"{high:.8g}")
        self._apply_display_settings_to_all(low=low, high=high, rebuild=False)
        self._histogram_pending_range = (low, high)
        self._update_histogram()
        if final:
            if self._histogram_render_after_id is not None:
                try:
                    self.root.after_cancel(self._histogram_render_after_id)
                except tk.TclError:
                    pass
                self._histogram_render_after_id = None
            self._histogram_flush_pending_range()
            self._update_pointer_status(None)
        elif self._histogram_render_after_id is None:
            # At most ~30 display rebuilds/s while dragging.  The histogram
            # itself remains immediate, so the handles stay attached to the mouse.
            self._histogram_render_after_id = self.root.after(
                33, self._histogram_flush_pending_range
            )

    def _histogram_set_handle(self, handle: str, x: float, *, final: bool = False) -> None:
        if self.current_frame is None or self._histogram_domain is None:
            return
        value = self._histogram_x_to_value(x)
        gap = self._histogram_minimum_gap()
        if handle == "min":
            low = min(value, self.display_max - gap)
            high = self.display_max
        else:
            low = self.display_min
            high = max(value, self.display_min + gap)
        self._histogram_apply_range(low, high, final=final)
        self.status_var.set(
            f"Histogram {handle.title()} | {self.display_min:.6g} … {self.display_max:.6g}"
        )

    def _histogram_press(self, event: tk.Event) -> str:
        if self.current_frame is None:
            return "break"
        if not self._histogram_handle_positions:
            self._update_histogram()
        min_x = self._histogram_handle_positions.get("min", float(event.x))
        max_x = self._histogram_handle_positions.get("max", float(event.x))
        hit = 13.0
        x = float(event.x)
        selection_width = max(0.0, max_x - min_x)
        centre = (min_x + max_x) / 2.0
        # When the display window is very narrow, the two generous edge hit
        # zones overlap.  Reserve the visual centre for moving the complete
        # window so users are never locked out of the Albula-style pan action.
        narrow_centre_hit = max(3.0, min(7.0, selection_width * 0.30))
        if min_x <= x <= max_x and selection_width <= hit * 2.0 and abs(x - centre) <= narrow_centre_hit:
            mode = "window"
        elif abs(x - min_x) <= hit:
            mode = "min"
        elif abs(x - max_x) <= hit:
            mode = "max"
        elif min_x + hit < x < max_x - hit:
            mode = "window"
        else:
            mode = "min" if abs(x - min_x) <= abs(x - max_x) else "max"
        self._histogram_drag_mode = mode
        self._histogram_drag_start_x = x
        self._histogram_drag_start_range = (self.display_min, self.display_max)
        self.histogram_canvas.configure(
            cursor="fleur" if mode == "window" else "sb_h_double_arrow"
        )
        if mode in {"min", "max"} and not (
            abs(x - min_x) <= hit or abs(x - max_x) <= hit
        ):
            self._histogram_set_handle(mode, x, final=False)
        return "break"

    def _histogram_drag(self, event: tk.Event) -> str:
        mode = self._histogram_drag_mode
        if mode is None or self.current_frame is None:
            return "break"
        if mode in {"min", "max"}:
            self._histogram_set_handle(mode, float(event.x), final=False)
            return "break"
        if mode == "window" and self._histogram_drag_start_range is not None:
            start_low, start_high = self._histogram_drag_start_range
            axis = self._histogram_axis
            if axis is None:
                return "break"
            delta_axis = (
                self._histogram_x_to_axis(float(event.x))
                - self._histogram_x_to_axis(self._histogram_drag_start_x)
            )
            low = axis.axis_to_raw(axis.raw_to_axis(start_low) + delta_axis)
            high = axis.axis_to_raw(axis.raw_to_axis(start_high) + delta_axis)
            low, high = self._histogram_normalize_range(low, high)
            self._histogram_apply_range(low, high, final=False)
            self.status_var.set(
                f"Histogram window | {low:.6g} … {high:.6g} | width={high-low:.6g}"
            )
        return "break"

    def _histogram_release(self, event: tk.Event) -> str:
        mode = self._histogram_drag_mode
        if mode in {"min", "max"}:
            self._histogram_set_handle(mode, float(event.x), final=True)
        elif mode == "window" and self._histogram_drag_start_range is not None:
            start_low, start_high = self._histogram_drag_start_range
            axis = self._histogram_axis
            if axis is None:
                return "break"
            delta_axis = (
                self._histogram_x_to_axis(float(event.x))
                - self._histogram_x_to_axis(self._histogram_drag_start_x)
            )
            low = axis.axis_to_raw(axis.raw_to_axis(start_low) + delta_axis)
            high = axis.axis_to_raw(axis.raw_to_axis(start_high) + delta_axis)
            low, high = self._histogram_normalize_range(low, high)
            self._histogram_apply_range(low, high, final=True)
        self._histogram_drag_mode = None
        self._histogram_drag_start_range = None
        self._histogram_motion(event)
        return "break"

    def _histogram_motion(self, event: tk.Event) -> str:
        if self.current_frame is None or self._histogram_plot_rect is None:
            return "break"
        if self._histogram_drag_mode is not None:
            return "break"
        x0, y0, x1, y1 = self._histogram_plot_rect
        x = min(x1, max(x0, float(event.x)))
        self._histogram_hover_x = x
        min_x = self._histogram_handle_positions.get("min", x0)
        max_x = self._histogram_handle_positions.get("max", x1)
        hit = 13.0
        if abs(x - min_x) <= hit or abs(x - max_x) <= hit:
            cursor = "sb_h_double_arrow"
        elif min_x + hit < x < max_x - hit:
            cursor = "fleur"
        else:
            cursor = "crosshair"
        self.histogram_canvas.configure(cursor=cursor)
        canvas = self.histogram_canvas
        canvas.delete("hist-hover")
        value = self._histogram_x_to_value(x)
        canvas.create_line(
            x, y0, x, y1,
            fill=UI_WHITE,
            width=1,
            dash=(3, 3),
            tags=("hist-hover",),
        )
        label = f"{value:.6g}"
        text_id = canvas.create_text(
            x, y0 + 7,
            text=label,
            fill=UI_WHITE,
            anchor="n",
            font=("Segoe UI", 8, "bold"),
            tags=("hist-hover",),
        )
        box = canvas.bbox(text_id)
        if box is not None:
            pad = 3
            rect = canvas.create_rectangle(
                box[0] - pad, box[1] - 1, box[2] + pad, box[3] + 1,
                fill=UI_BG,
                outline=UI_BORDER,
                tags=("hist-hover",),
            )
            canvas.tag_lower(rect, text_id)
        return "break"

    def _histogram_leave(self, _event: tk.Event) -> str:
        self._histogram_hover_x = None
        self.histogram_canvas.delete("hist-hover")
        if self._histogram_drag_mode is None:
            self.histogram_canvas.configure(cursor="crosshair")
        return "break"

    def _histogram_wheel(self, _event: tk.Event) -> str:
        """Block wheel input over the Histogram without changing contrast.

        Histogram contrast changes are intentionally drag-only.  Returning
        ``"break"`` also prevents the scrollable Inspector from moving while
        the pointer is over the plot, so a wheel gesture can never alter a
        threshold or unexpectedly move the panel.
        """
        return "break"

    def _histogram_auto(self, _event: tk.Event) -> str:
        self.auto_contrast()
        return "break"

    def _histogram_full(self, _event: tk.Event) -> str:
        self.full_contrast()
        return "break"

    def _draw_histogram_handle(
        self,
        canvas: tk.Canvas,
        *,
        name: str,
        x: float,
        top: float,
        bottom: float,
        color: str,
    ) -> None:
        canvas.create_line(
            x, top, x, bottom,
            fill=color,
            width=3,
            tags=("hist-handle", name),
        )
        canvas.create_polygon(
            x - 8, top,
            x + 8, top,
            x + (8 if name == "min" else -8), top + 11,
            x, top + 16,
            x + (-8 if name == "min" else 8), top + 11,
            fill=color,
            outline=UI_BG,
            width=1,
            tags=("hist-handle", name),
        )
        canvas.create_rectangle(
            x - 8, bottom - 15, x + 8, bottom + 2,
            fill=color,
            outline=UI_BG,
            width=1,
            tags=("hist-handle", name),
        )
        canvas.create_text(
            x, bottom - 7,
            text="◀" if name == "min" else "▶",
            fill=UI_BG,
            font=("Segoe UI Symbol", 8, "bold"),
            tags=("hist-handle", name),
        )

    def _draw_histogram_tone_ramp(
        self,
        canvas: tk.Canvas,
        *,
        left: float,
        right: float,
        top: float,
        bottom: float,
    ) -> None:
        steps = 128
        span = max(1.0, right - left)
        window = max(self._histogram_minimum_gap(), self.display_max - self.display_min)
        for i in range(steps):
            x0 = left + span * i / steps
            x1 = left + span * (i + 1) / steps + 1
            value = self._histogram_x_to_value((x0 + x1) / 2.0)
            level = min(1.0, max(0.0, (value - self.display_min) / window))
            if self.invert_var.get():
                level = 1.0 - level
            gray = int(round(level * 255.0))
            color = f"#{gray:02x}{gray:02x}{gray:02x}"
            canvas.create_rectangle(x0, top, x1, bottom, fill=color, outline="")
        canvas.create_rectangle(left, top, right, bottom, outline=UI_BORDER, width=1)

    def _update_histogram(self) -> None:
        canvas = self.histogram_canvas
        canvas.delete("all")
        self._histogram_handle_positions = {}
        if self.current_frame is None:
            return
        width = max(160, canvas.winfo_width() or 330)
        height = max(150, canvas.winfo_height() or 210)
        if self._histogram_values_cache is None:
            # Use every valid pixel so the displayed minimum and maximum are
            # exact, not estimates from a stride sample.
            self._histogram_values_cache = _detector_values(
                self.current_frame,
                max_samples=max(1, int(self.current_frame.size)),
            ).astype(np.float64, copy=False)
        values = self._histogram_values_cache
        if values.size == 0:
            return

        domain_low, domain_high = self._ensure_histogram_domain(values)
        if domain_high <= domain_low:
            return

        margin_x = 18
        top = 28
        bottom = height - 78
        left = float(margin_x)
        right = float(width - margin_x)
        self._histogram_plot_rect = (left, float(top), right, float(bottom))
        plot_w = max(1.0, right - left)
        plot_h = max(1.0, bottom - top)

        axis = self._histogram_axis
        if axis is None:
            return
        cache_key = (
            id(self.current_frame),
            float(domain_low),
            float(domain_high),
            float(axis.origin),
            float(axis.scale),
        )
        if self._histogram_cache_key != cache_key or self._histogram_logs_cache is None:
            in_range = values[(values >= domain_low) & (values <= domain_high)]
            if in_range.size == 0:
                in_range = np.asarray([domain_low, domain_high], dtype=np.float64)
            transformed = axis.transform_array(in_range)
            hist, _ = np.histogram(
                transformed,
                bins=192,
                range=(axis.axis_low, axis.axis_high),
            )
            self._histogram_logs_cache = np.log1p(hist.astype(np.float64))
            self._histogram_cache_key = cache_key
        logs = self._histogram_logs_cache
        peak = float(np.max(logs))

        min_x = self._histogram_value_to_x(self.display_min)
        max_x = self._histogram_value_to_x(self.display_max)
        self._histogram_handle_positions = {"min": min_x, "max": max_x}

        # A visible active window makes it clear that the central area itself
        # can be dragged, not only the two thin marker lines.
        canvas.create_rectangle(
            min_x, top, max_x, bottom,
            fill="#173A56",
            stipple="gray50",
            outline="",
        )
        for fraction in (0.25, 0.5, 0.75):
            grid_y = bottom - plot_h * fraction
            canvas.create_line(left, grid_y, right, grid_y, fill=UI_BORDER_SOFT, width=1)
        canvas.create_rectangle(left, top, right, bottom, outline=UI_BORDER, width=1)
        if peak > 0:
            step = plot_w / len(logs)
            for i, count in enumerate(logs):
                bar_h = (count / peak) * plot_h
                x0 = left + i * step
                x1 = left + (i + 1) * step + 1
                center = (x0 + x1) / 2.0
                selected = min_x <= center <= max_x
                fill = "#A9D9F5" if selected else "#54708F"
                canvas.create_rectangle(x0, bottom - bar_h, x1, bottom, fill=fill, outline="")

        if min_x > left:
            canvas.create_rectangle(left, top, min_x, bottom, fill=UI_BG, stipple="gray25", outline="")
        if max_x < right:
            canvas.create_rectangle(max_x, top, right, bottom, fill=UI_BG, stipple="gray25", outline="")

        # Raw-value labels make the compressed coordinate explicit: the axis
        # is non-linear, but every label and threshold remains a true detector
        # intensity.
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            tick_x = left + plot_w * fraction
            tick_value = axis.fraction_to_raw(fraction)
            canvas.create_line(
                tick_x, bottom - 4, tick_x, bottom,
                fill=UI_TEXT_MUTED, width=1,
            )
            anchor = "w" if fraction == 0.0 else ("e" if fraction == 1.0 else "s")
            text_x = tick_x if fraction in (0.0, 1.0) else tick_x
            canvas.create_text(
                text_x, bottom + 23,
                text=_format_value(tick_value),
                fill=UI_TEXT_MUTED,
                anchor=anchor if fraction in (0.0, 1.0) else "n",
                font=("Segoe UI", 7),
            )

        self._draw_histogram_handle(
            canvas,
            name="min",
            x=min_x,
            top=top,
            bottom=bottom,
            color=UI_ACCENT,
        )
        self._draw_histogram_handle(
            canvas,
            name="max",
            x=max_x,
            top=top,
            bottom=bottom,
            color=UI_WARNING,
        )

        # The histogram axis is the exact pixel interval, so no values are
        # hidden as a clipped tail.
        window_width = self.display_max - self.display_min
        canvas.create_text(
            left,
            7,
            text=f"MIN  {self.display_min:.6g}",
            fill=UI_ACCENT,
            anchor="nw",
            font=("Segoe UI", 8, "bold"),
        )
        canvas.create_text(
            right,
            7,
            text=f"MAX  {self.display_max:.6g}",
            fill=UI_WARNING,
            anchor="ne",
            font=("Segoe UI", 8, "bold"),
        )
        canvas.create_text(
            width / 2,
            7,
            text=f"WINDOW  {window_width:.6g}",
            fill=UI_TEXT,
            anchor="n",
            font=("Segoe UI", 8, "bold"),
        )

        ramp_top = bottom + 38
        ramp_bottom = bottom + 49
        self._draw_histogram_tone_ramp(
            canvas,
            left=left,
            right=right,
            top=ramp_top,
            bottom=ramp_bottom,
        )
        # Report the real frame extrema explicitly. For a constant frame the
        # plotting domain is expanded only visually, while the label still shows
        # the exact single pixel value.
        finite_values = values[np.isfinite(values)]
        pixel_low = float(np.min(finite_values)) if finite_values.size else 0.0
        pixel_high = float(np.max(finite_values)) if finite_values.size else 1.0
        canvas.create_text(
            width / 2,
            height - 7,
            text=(
                f"DATA  {_format_value(pixel_low)} … {_format_value(pixel_high)}"
                "   ·   ADAPTIVE INTENSITY AXIS   ·   LOG PIXEL COUNT"
            ),
            fill=UI_TEXT_MUTED,
            anchor="s",
            font=("Segoe UI", 7, "bold"),
        )

    def _update_image_info(self) -> None:
        if not hasattr(self, "image_info_text"):
            return
        widget = self.image_info_text
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        if self.archive is None or self.current_frame is None:
            widget.insert("end", "No image\n")
            widget.configure(state="disabled")
            return
        try:
            sections = self.archive.image_info(self.current_frame_index, self.current_frame)
        except Exception as exc:
            # Last-resort display from the image itself. This deliberately
            # avoids turning a metadata problem into an image-open failure.
            frame = np.asarray(self.current_frame)
            height, width = frame.shape
            try:
                total = float(np.sum(frame, dtype=np.float64))
            except Exception:
                total = float("nan")
            try:
                maximum = float(np.max(frame))
            except Exception:
                maximum = float("nan")
            sections = {
                "Detector": [
                    ("Detector", "Unknown detector (metadata unavailable)"),
                    ("Number of pixels in X-direction", str(width)),
                    ("Number of pixels in Y-direction", str(height)),
                ],
                "Image": [
                    ("Image number", str(self.current_frame_index + 1)),
                    ("Total intensity", f"{total:.6e}"),
                    ("Maximum intensity", f"{maximum:.6e}"),
                    ("# zeros", str(int(np.count_nonzero(frame == 0)))),
                ],
                "Other Information": [
                    ("Metadata warning", f"{type(exc).__name__}: {exc}"),
                    ("Fallback", "Image remains viewable using shape/dtype/pixel statistics."),
                ],
            }
        for section, rows in sections.items():
            widget.insert("end", f"{section}\n", "section")
            for label, value in rows:
                try:
                    safe_label = str(label)
                    safe_value = _scalar_text(value) if not isinstance(value, str) else value
                except Exception:
                    safe_label = str(label)
                    safe_value = repr(value)
                if safe_label == "Detector" and section == "Detector":
                    widget.insert("end", f"{safe_value}\n")
                else:
                    widget.insert("end", f"{safe_label}: ", "label")
                    widget.insert("end", f"{safe_value}\n")
            widget.insert("end", "\n")
        widget.configure(state="disabled")

    def _update_metadata(self) -> None:
        tree = self.metadata_tree
        tree.delete(*tree.get_children())
        self._metadata_item_count = 0
        if self.archive is None:
            return
        try:
            document = self.archive.describe_frame(self.current_frame_index)
            if not isinstance(document, dict):
                document = {"Other Information": {"metadata": self._metadata_text(document)}}
        except Exception as exc:
            document = {
                "HDF5 Source": {
                    "filename": getattr(getattr(self.archive, "path", None), "name", "unknown"),
                    "format": getattr(self.archive, "file_format", "unknown"),
                },
                "Other Information": {
                    "metadata_status": "partial / fallback",
                    "warning": f"{type(exc).__name__}: {exc}",
                    "note": "Image data is readable; only the metadata inspector was incomplete.",
                },
            }
        for key, value in document.items():
            try:
                node = tree.insert(
                    "",
                    "end",
                    text=str(key),
                    values=("",),
                    open=key in ("HDF5 Source", "HDF5 Dataset", "Other Information"),
                )
                self._insert_metadata_value(node, value, depth=0)
            except Exception as exc:
                tree.insert("", "end", text="Other Information", values=(f"metadata display warning: {exc}",), open=True)

    def _insert_metadata_value(self, parent: str, value: Any, *, depth: int) -> None:
        if self._metadata_item_count >= 350:
            if not self.metadata_tree.get_children(parent):
                self.metadata_tree.insert(parent, "end", text="…", values=("truncated",))
            return
        if depth >= 5:
            self.metadata_tree.insert(parent, "end", text="value", values=(self._metadata_text(value),))
            self._metadata_item_count += 1
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if item is None or item == {} or item == []:
                    continue
                if isinstance(item, (dict, list, tuple)):
                    node = self.metadata_tree.insert(parent, "end", text=str(key), values=("",), open=False)
                    self._metadata_item_count += 1
                    self._insert_metadata_value(node, item, depth=depth + 1)
                else:
                    self.metadata_tree.insert(parent, "end", text=str(key), values=(self._metadata_text(item),))
                    self._metadata_item_count += 1
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value[:80]):
                if isinstance(item, (dict, list, tuple)):
                    node = self.metadata_tree.insert(parent, "end", text=f"[{index}]", values=("",), open=False)
                    self._metadata_item_count += 1
                    self._insert_metadata_value(node, item, depth=depth + 1)
                else:
                    self.metadata_tree.insert(parent, "end", text=f"[{index}]", values=(self._metadata_text(item),))
                    self._metadata_item_count += 1
            if len(value) > 80:
                self.metadata_tree.insert(parent, "end", text="…", values=(f"{len(value) - 80} more",))
        else:
            self.metadata_tree.insert(parent, "end", text="value", values=(self._metadata_text(value),))
            self._metadata_item_count += 1

    @staticmethod
    def _metadata_text(value: Any) -> str:
        try:
            if isinstance(value, (dict, list, tuple)):
                text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            else:
                text = _format_value(value)
        except Exception:
            text = repr(value)
        return text if len(text) <= 180 else text[:177] + "…"

    def _fit(self) -> None:
        for view in self.views:
            if view.current_frame is not None:
                view.canvas.fit_to_window(notify=False)

    def _one_to_one(self) -> None:
        for view in self.views:
            if view.current_frame is not None:
                view.canvas.one_to_one(notify=False)

    def _set_shared_zoom_factor(self, factor: float) -> None:
        source = self.active_view
        if source is None or source.current_frame is None:
            return
        source.canvas.set_zoom(source.canvas.zoom * float(factor), notify=False)
        state = source.canvas._view_state()
        if state is None:
            return
        for view in self.views:
            if view is source or view.current_frame is None:
                continue
            view.canvas.apply_synced_view(state)

    def _zoom_in(self) -> None:
        self._set_shared_zoom_factor(1.25)

    def _zoom_out(self) -> None:
        self._set_shared_zoom_factor(0.8)

    def _update_pointer_status(self, info: dict[str, Any] | None) -> None:
        if self.archive is None:
            self.status_var.set("Ready")
            self.pixel_summary_var.set("Pixel: —")
            return
        base = (
            f"{self.archive.path.name} | Image {self.current_frame_index + 1} "
            f"(HDXF {self.current_frame_index}) | "
            f"range {self.display_min:.6g}…{self.display_max:.6g}"
        )
        if not info:
            self.status_var.set(base)
            self.pixel_summary_var.set("Pixel: —")
            return
        zoom = float(info.get("zoom", self.canvas.zoom))
        if "x" in info:
            pixel_text = f"Pixel X={info['x']}  Y={info['y']}  Value={_format_value(info['value'])}"
            self.pixel_summary_var.set(pixel_text)
            self.status_var.set(f"{base} | {pixel_text} | zoom={zoom:.2f}×")
        else:
            self.pixel_summary_var.set("Pixel: outside image")
            self.status_var.set(f"{base} | zoom={zoom:.2f}×")

def configure_modern_theme(root: tk.Tk, *, ui_scale: float = 1.0) -> None:
    """Configure a coherent modern dark ttk theme without external assets."""
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    ui_scale = max(0.75, min(4.0, float(ui_scale)))

    def px(value: int) -> int:
        return max(1, int(round(value * ui_scale)))

    default_font = ("Segoe UI", 9)
    semibold = ("Segoe UI", 9, "bold")
    style.configure(".", font=default_font, background=UI_BG, foreground=UI_TEXT)

    style.configure("App.TFrame", background=UI_BG)
    style.configure("Topbar.TFrame", background=UI_TOPBAR)
    style.configure("ToolbarGroup.TFrame", background=UI_SURFACE, relief="flat")
    style.configure("ImageHost.TFrame", background=UI_BG)
    style.configure("Side.TFrame", background=UI_BG)
    style.configure("Card.TFrame", background=UI_CARD, relief="flat")
    style.configure("Statusbar.TFrame", background=UI_SURFACE)
    style.configure("Modern.TPanedwindow", background=UI_BORDER_SOFT, sashwidth=px(5))

    style.configure("Brand.TLabel", background=UI_TOPBAR, foreground=UI_TEXT, font=("Segoe UI", 14, "bold"))
    style.configure("BrandSub.TLabel", background=UI_TOPBAR, foreground=UI_TEXT_DIM, font=("Segoe UI", 8, "bold"))
    style.configure("ToolbarGroupTitle.TLabel", background=UI_SURFACE, foreground=UI_TEXT_DIM, font=("Segoe UI", 7, "bold"))
    style.configure("ToolbarValue.TLabel", background=UI_SURFACE, foreground=UI_TEXT, font=semibold)
    style.configure("ToolbarMuted.TLabel", background=UI_SURFACE, foreground=UI_TEXT_MUTED)
    style.configure("TopbarChip.TLabel", background=UI_ACCENT_DARK, foreground=UI_ACCENT, padding=(px(10), px(5)), font=("Segoe UI", 8, "bold"))
    style.configure("InspectorTitle.TLabel", background=UI_BG, foreground=UI_TEXT, font=("Segoe UI", 12, "bold"))
    style.configure("InspectorSub.TLabel", background=UI_BG, foreground=UI_TEXT_DIM, font=("Segoe UI", 8, "bold"))
    style.configure("InspectorChip.TLabel", background=UI_ACCENT_DARK, foreground=UI_ACCENT, padding=(px(8), px(5)), font=("Segoe UI", 8, "bold"))
    style.configure("CardTitle.TLabel", background=UI_CARD, foreground=UI_TEXT, font=("Segoe UI", 10, "bold"))
    style.configure("CardSubtitle.TLabel", background=UI_CARD, foreground=UI_TEXT_MUTED, font=("Segoe UI", 8))
    style.configure("FieldLabel.TLabel", background=UI_CARD, foreground=UI_TEXT_DIM, font=("Segoe UI", 8, "bold"))
    style.configure("Readout.TLabel", background=UI_CARD, foreground=UI_TEXT, font=("Consolas", 9, "bold"))
    style.configure("ReadoutMuted.TLabel", background=UI_CARD, foreground=UI_TEXT_MUTED, font=("Segoe UI", 8))
    style.configure("Status.TLabel", background=UI_SURFACE, foreground=UI_TEXT_MUTED)
    style.configure("StatusChip.TLabel", background=UI_ACCENT_DARK, foreground=UI_ACCENT, padding=(px(8), px(3)), font=("Segoe UI", 8, "bold"))

    button_common = dict(font=semibold, padding=(px(9), px(5)), borderwidth=0, relief="flat")
    style.configure("TButton", background=UI_SURFACE_RAISED, foreground=UI_TEXT, **button_common)
    style.map("TButton", background=[("pressed", UI_ACCENT_DARK), ("active", UI_BORDER)], foreground=[("disabled", UI_TEXT_DIM)])
    style.configure("Toolbar.TButton", background=UI_SURFACE_RAISED, foreground=UI_TEXT, **button_common)
    style.map("Toolbar.TButton", background=[("pressed", UI_ACCENT_DARK), ("active", UI_BORDER)])
    style.configure("Icon.TButton", background=UI_SURFACE_RAISED, foreground=UI_TEXT, font=("Segoe UI Symbol", 10, "bold"), padding=(px(6), px(5)), borderwidth=0)
    style.map("Icon.TButton", background=[("pressed", UI_ACCENT_DARK), ("active", UI_BORDER)], foreground=[("active", UI_WHITE)])
    style.configure("Accent.TButton", background=UI_ACCENT_STRONG, foreground=UI_WHITE, **button_common)
    style.map("Accent.TButton", background=[("pressed", "#1D4ED8"), ("active", UI_ACCENT_HOVER)], foreground=[("active", UI_WHITE)])
    style.configure("Play.TButton", background=UI_ACCENT, foreground=UI_BG, font=("Segoe UI", 9, "bold"), padding=(px(10), px(5)), borderwidth=0)
    style.map("Play.TButton", background=[("pressed", UI_ACCENT_STRONG), ("active", "#7DD3FC")], foreground=[("active", UI_BG)])
    style.configure("Secondary.TButton", background=UI_CARD_ALT, foreground=UI_TEXT, **button_common)
    style.map("Secondary.TButton", background=[("pressed", UI_ACCENT_DARK), ("active", UI_BORDER)])
    style.configure("Converter.Horizontal.TProgressbar", troughcolor=UI_CARD_ALT, background=UI_ACCENT, bordercolor=UI_CARD_ALT, lightcolor=UI_ACCENT, darkcolor=UI_ACCENT, thickness=px(16))

    style.configure("TLabel", background=UI_BG, foreground=UI_TEXT)
    style.configure("TFrame", background=UI_BG)
    style.configure("TSeparator", background=UI_BORDER_SOFT)
    style.configure("Card.TSeparator", background=UI_BORDER_SOFT)

    style.configure("Modern.TEntry", fieldbackground=UI_CARD_ALT, background=UI_CARD_ALT, foreground=UI_TEXT, insertcolor=UI_TEXT, bordercolor=UI_BORDER, lightcolor=UI_BORDER, darkcolor=UI_BORDER, padding=(px(7), px(5)))
    style.map("Modern.TEntry", bordercolor=[("focus", UI_ACCENT)], lightcolor=[("focus", UI_ACCENT)], darkcolor=[("focus", UI_ACCENT)])
    style.configure("Modern.TSpinbox", fieldbackground=UI_CARD_ALT, background=UI_CARD_ALT, foreground=UI_TEXT, arrowcolor=UI_TEXT_MUTED, bordercolor=UI_BORDER, lightcolor=UI_BORDER, darkcolor=UI_BORDER, padding=(px(6), px(4)))
    style.map("Modern.TSpinbox", bordercolor=[("focus", UI_ACCENT)], arrowcolor=[("active", UI_ACCENT)])
    style.configure("Modern.TCombobox", fieldbackground=UI_CARD_ALT, background=UI_CARD_ALT, foreground=UI_TEXT, arrowcolor=UI_TEXT_MUTED, bordercolor=UI_BORDER, lightcolor=UI_BORDER, darkcolor=UI_BORDER, padding=(px(6), px(4)))
    style.map("Modern.TCombobox", fieldbackground=[("readonly", UI_CARD_ALT)], background=[("readonly", UI_CARD_ALT)], foreground=[("readonly", UI_TEXT)], arrowcolor=[("active", UI_ACCENT)])

    style.configure("Modern.TCheckbutton", background=UI_CARD, foreground=UI_TEXT, indicatorbackground=UI_CARD_ALT, indicatorforeground=UI_ACCENT, padding=(0, px(3)))
    style.map("Modern.TCheckbutton", background=[("active", UI_CARD)], foreground=[("active", UI_WHITE)], indicatorbackground=[("selected", UI_ACCENT_STRONG), ("active", UI_BORDER)])
    style.configure("Toolbar.TCheckbutton", background=UI_SURFACE, foreground=UI_TEXT_MUTED, indicatorbackground=UI_CARD_ALT, padding=(0, px(2)))
    style.map("Toolbar.TCheckbutton", background=[("active", UI_SURFACE)], foreground=[("active", UI_TEXT)], indicatorbackground=[("selected", UI_ACCENT_STRONG)])

    style.configure("Metadata.Treeview", background=UI_CARD_ALT, fieldbackground=UI_CARD_ALT, foreground=UI_TEXT, rowheight=px(24), borderwidth=0, relief="flat")
    style.configure("Metadata.Treeview.Heading", background=UI_SURFACE_RAISED, foreground=UI_TEXT_MUTED, font=("Segoe UI", 8, "bold"), relief="flat", padding=(px(5), px(5)))
    style.map("Metadata.Treeview", background=[("selected", UI_ACCENT_DARK)], foreground=[("selected", UI_WHITE)])
    style.map("Metadata.Treeview.Heading", background=[("active", UI_BORDER)])

    style.configure("Modern.Vertical.TScrollbar", background=UI_SURFACE_RAISED, troughcolor=UI_BG, bordercolor=UI_BG, arrowcolor=UI_TEXT_DIM, gripcount=0, width=px(11))
    style.map("Modern.Vertical.TScrollbar", background=[("active", UI_BORDER), ("pressed", UI_ACCENT_DARK)])

    # Keep readonly combobox pop-down lists consistent with the dark shell.
    root.option_add("*TCombobox*Listbox.background", UI_CARD_ALT)
    root.option_add("*TCombobox*Listbox.foreground", UI_TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", UI_ACCENT_STRONG)
    root.option_add("*TCombobox*Listbox.selectForeground", UI_WHITE)


def create_application_root() -> tk.Tk:
    """Create the one and only Tk root, using TkDND when it is installed.

    ``tkinterdnd2`` must own root creation; loading it after ``tk.Tk()`` is not
    enough on all Windows/Tk builds.  Failure remains non-fatal so File > Open
    continues to work without the optional dependency.
    """
    try:
        from tkinterdnd2 import TkinterDnD
    except ImportError:
        root = tk.Tk()
        root._hdxf_dnd_backend = "not-installed"  # type: ignore[attr-defined]
        root._hdxf_dnd_error = "tkinterdnd2 is not installed"  # type: ignore[attr-defined]
        return root

    try:
        root = TkinterDnD.Tk()
        root._hdxf_dnd_backend = "tkinterdnd2"  # type: ignore[attr-defined]
        root._hdxf_dnd_error = ""  # type: ignore[attr-defined]
        return root
    except Exception as exc:
        # A broken Tcl/TkDND binary must not make the Viewer unusable.
        root = tk.Tk()
        root._hdxf_dnd_backend = "failed"  # type: ignore[attr-defined]
        root._hdxf_dnd_error = f"{type(exc).__name__}: {exc}"  # type: ignore[attr-defined]
        return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Albula-inspired HDXF/HDF5 desktop viewer v0.4.8.0 HDF5 + SUM/SUB/PIXEL build")
    parser.add_argument("hdxf", nargs="?", type=Path, help="Optional .hdxf file")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dpi_mode = enable_windows_high_dpi_awareness()
    root = create_application_root()
    dpi_controller = WindowsDPIController(root)

    try:
        configure_modern_theme(root, ui_scale=dpi_controller.ui_scale)
    except Exception:
        # The viewer remains usable with the platform default theme if a
        # particular Tk build rejects an optional style property.
        pass

    manager = ViewerWorkspaceManager(root, args.hdxf)
    manager.app.status_var.set(
        f"Ready · DPI {dpi_controller.dpi} ({dpi_mode})"
        if sys.platform == "win32" else "Ready"
    )

    def refresh_theme_for_monitor(_event: tk.Event | None = None) -> None:
        try:
            configure_modern_theme(root, ui_scale=dpi_controller.ui_scale)
            manager.app.relayout_views()
            root.update_idletasks()
        except tk.TclError:
            pass

    root.bind("<<HDXFDPIChanged>>", refresh_theme_for_monitor, add="+")
    dpi_controller.start()

    def report_callback_exception(exc_type, exc_value, exc_tb):
        details = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        messagebox.showerror("Viewer error", details, parent=root)

    root.report_callback_exception = report_callback_exception
    try:
        root.mainloop()
    finally:
        dpi_controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
