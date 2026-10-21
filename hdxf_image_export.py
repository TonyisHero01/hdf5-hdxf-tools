"""Non-destructive frame exports for HDXF Viewer.

Display PNG/JPEG use the current 8-bit contrast mapping at native detector
resolution. NPY retains array values/dtype (including unsaved live edits).
16-bit PNG is available only for uint16 input, with no display annotations.
No export mode writes an HDF5 or HDXF archive.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
from PIL import Image, ImageDraw


PNG_DISPLAY = "PNG (display, 8-bit)"
JPG_DISPLAY = "JPEG (display, 8-bit)"
NPY_RAW = "NPY (pixel values, lossless)"
PNG_UINT16 = "PNG 16-bit (uint16 only)"
EXPORT_FORMATS = (PNG_DISPLAY, JPG_DISPLAY, NPY_RAW, PNG_UINT16)


@dataclass(frozen=True)
class OverlayState:
    """Snapshot of the current view's non-destructive annotation state."""

    manual_mask: np.ndarray | None = None
    roi_bbox: tuple[int, int, int, int] | None = None
    roi_shape: str = "Rectangle"
    roi_mask: np.ndarray | None = None
    roi_points: tuple[tuple[int, int], ...] = ()
    roi_inner_diameter: int = 0
    beam_center: tuple[int, int] | None = None
    resolution_geometry: dict[str, float] | None = None
    show_resolution_rings: bool = True
    bright_spot_boxes: tuple[tuple[int, int, int, int], ...] = ()


def file_extension(kind: str) -> str:
    return {
        PNG_DISPLAY: ".png", JPG_DISPLAY: ".jpg",
        NPY_RAW: ".npy", PNG_UINT16: ".png",
    }[kind]


def frame_filename(source_stem: str, number: int, kind: str) -> str:
    """Human-readable, Windows-friendly, deterministic 1-based filename."""
    import re
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(source_stem)).rstrip(" .")[:85] or "detector"
    return f"{stem}_frame_{int(number):06d}{file_extension(kind)}"


def frame_indices(start: int, stop: int, step: int, count: int) -> range:
    """Validate GUI 1-based inclusive range without allocating all indices."""
    start, stop, step, count = int(start), int(stop), int(step), int(count)
    if count <= 0 or start < 1 or start > count or stop < start or stop > count or step <= 0:
        raise ValueError(f"Frame range must satisfy 1 <= start <= end <= {count}, with step >= 1")
    return range(start - 1, stop, step)


def display_gray(frame: np.ndarray, low: float, high: float, invert: bool) -> np.ndarray:
    """Mirror HDXFViewerApp._render_view_display's 8-bit intensity mapping."""
    if frame.ndim != 2:
        raise ValueError("Export requires one 2-D detector frame")
    if not (math.isfinite(low) and math.isfinite(high) and high > low):
        raise ValueError("Invalid display contrast range")
    mapped = np.clip((np.asarray(frame, dtype=np.float32) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    return 255 - mapped if invert else mapped


def _blend(rgb: np.ndarray, selection: np.ndarray, tint: tuple[int, int, int], alpha: float) -> None:
    if selection.size == 0 or not np.any(selection):
        return
    pixel = rgb[selection].astype(np.float32)
    rgb[selection] = np.clip(pixel * (1 - alpha) + np.array(tint, np.float32) * alpha, 0, 255).astype(np.uint8)


def _draw_resolution_rings(draw: ImageDraw.ImageDraw, overlay: OverlayState, shape: tuple[int, int]) -> None:
    if not (overlay.show_resolution_rings and overlay.beam_center is not None and overlay.resolution_geometry):
        return
    g = overlay.resolution_geometry
    try:
        wavelength = float(g["wavelength_angstrom"])
        distance = float(g["distance_m"])
        px = float(g["pixel_size_x_m"])
        py = float(g["pixel_size_y_m"])
        if min(wavelength, distance, px, py) <= 0:
            return
    except (KeyError, ValueError, TypeError):
        return
    cx, cy = (overlay.beam_center[0] + 0.5, overlay.beam_center[1] + 0.5)
    max_r = max(math.hypot((x-cx)*px, (y-cy)*py) for x in (0, shape[1]) for y in (0, shape[0]))
    # Same resolution candidates as the viewer, limited to useful separated rings.
    spacings = (50, 40, 30, 25, 20, 15, 12, 10, 8, 7, 6, 5, 4.5, 4, 3.5, 3, 2.5, 2.2,
                2, 1.8, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1, .9, .8, .7, .6, .5)
    candidates = []
    for spacing in spacings:
        ratio = wavelength / (2 * spacing)
        if not 0 < ratio < 1:
            continue
        radius = distance * math.tan(2 * math.asin(ratio))
        if 0 < radius <= 1.06 * max_r:
            candidates.append((radius, radius/px, radius/py))
    candidates.sort()
    selected = []
    for r in candidates:
        if not selected or r[0] - selected[-1][0] >= max_r * .075:
            selected.append(r)
    if candidates and candidates[-1] not in selected:
        selected.append(candidates[-1])
    if len(selected) > 10:
        selected = [selected[int(i)] for i in np.linspace(0, len(selected)-1, 10).round()]
    for _, rx, ry in selected:
        draw.ellipse((cx-rx, cy-ry, cx+rx, cy+ry), outline=(56, 189, 248), width=1)


def display_image(frame: np.ndarray, *, low: float, high: float, invert: bool = False,
                  overlays: OverlayState | None = None) -> Image.Image:
    """Export the full detector frame with current contrast and optional annotations."""
    gray = display_gray(np.asarray(frame), low, high, invert)
    if overlays is None:
        return Image.fromarray(gray)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    height, width = gray.shape
    bbox = overlays.roi_bbox
    local = overlays.roi_mask
    if bbox is not None and local is not None:
        x, y, w, h = bbox
        if 0 <= x and 0 <= y and x+w <= width and y+h <= height and local.shape == (h, w):
            _blend(rgb[y:y+h, x:x+w], np.asarray(local, dtype=bool), (96, 182, 236), 0.50)
    manual = overlays.manual_mask
    if manual is not None:
        if manual.shape != gray.shape:
            raise ValueError(f"Manual mask shape {manual.shape} does not match frame {gray.shape}")
        _blend(rgb, np.asarray(manual, dtype=bool), (250, 72, 126), 0.65)

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    if bbox is not None:
        x, y, w, h = bbox
        shape = overlays.roi_shape
        outline = (52, 211, 153)
        box = (x, y, x+w-1, y+h-1)
        if shape in ("Ellipse", "Circle", "Annulus"):
            draw.ellipse(box, outline=outline, width=1)
            if shape == "Annulus" and overlays.roi_inner_diameter > 0:
                radius = overlays.roi_inner_diameter / 2.0
                cx, cy = x+w/2.0, y+h/2.0
                draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius), outline=outline, width=1)
        elif shape in ("Polygon", "Freehand") and len(overlays.roi_points) >= 3:
            draw.line(list(overlays.roi_points) + [overlays.roi_points[0]], fill=outline, width=1)
        else:
            draw.rectangle(box, outline=outline, width=1)
    _draw_resolution_rings(draw, overlays, gray.shape)
    for x0, y0, x1, y1 in overlays.bright_spot_boxes:
        draw.rectangle((max(0,x0-1), max(0,y0-1), min(width-1,x1), min(height-1,y1)),
                       outline=(255,48,48), width=1)
    if overlays.beam_center is not None:
        x, y = overlays.beam_center
        if 0 <= x < width and 0 <= y < height:
            c = (255, 184, 77)
            draw.line((x-8,y+.5,x+9,y+.5), fill=c, width=1)
            draw.line((x+.5,y-8,x+.5,y+9), fill=c, width=1)
            draw.ellipse((x-3,y-3,x+4,y+4), outline=c, width=1)
    return image


def export_frame(frame: np.ndarray, destination: str | Path, kind: str, *, low: float = 0,
                 high: float = 1, invert: bool = False, overlays: OverlayState | None = None) -> Path:
    """Write one frame in the specified representation. Never change input."""
    if kind not in EXPORT_FORMATS:
        raise ValueError(f"Unsupported export format: {kind}")
    destination = Path(destination)
    if destination.suffix.lower() != file_extension(kind):
        raise ValueError(f"Output extension for {kind} must be {file_extension(kind)}")
    data = np.asarray(frame)
    if data.ndim != 2:
        raise ValueError("Expected one 2-D detector frame")
    if kind == NPY_RAW:
        with destination.open("wb") as stream:
            np.save(stream, data, allow_pickle=False)
        return destination
    if kind == PNG_UINT16:
        if data.dtype != np.uint16:
            raise ValueError("16-bit PNG requires uint16 detector pixels; use NPY for other dtypes")
        Image.fromarray(data).save(destination, format="PNG")
        return destination
    image = display_image(data, low=low, high=high, invert=invert, overlays=overlays)
    if kind == PNG_DISPLAY:
        image.save(destination, format="PNG")
    else:
        image.convert("RGB").save(destination, format="JPEG", quality=95, subsampling=0)
    return destination
