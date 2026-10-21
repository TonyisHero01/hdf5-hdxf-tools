"""Non-destructive module-gap masks for detector frames.

A gap is established only from explicit NeXus NXdetector_module pixel slabs
or user-supplied row/column pixel ranges. Intensity zero is never evidence.
Masks use NumPy detector image indexing [row, column], True == excluded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np


class GapMaskError(ValueError):
    """Missing, ambiguous, or invalid module geometry."""


@dataclass(frozen=True)
class GapProposal:
    mask: np.ndarray
    description: str
    modules: int = 0


def parse_pixel_ranges(text: str, length: int, axis: str) -> tuple[tuple[int, int], ...]:
    """Parse 0-based *inclusive* 12, 20:24, 30-35 detector pixel intervals."""
    text = str(text).strip()
    if not text:
        return ()
    ranges = []
    for token in text.split(","):
        token = token.strip()
        match = re.fullmatch(r"(\d+)(?:\s*[:\-]\s*(\d+))?", token)
        if match is None:
            raise GapMaskError(f"Invalid {axis} range {token!r}; use 10:14, 30 (inclusive, zero-based).")
        first = int(match.group(1))
        last = int(match.group(2)) if match.group(2) is not None else first
        if not (0 <= first <= last < length):
            raise GapMaskError(f"{axis} range {token!r} outside 0..{length-1} or reversed.")
        ranges.append((first, last))
    return tuple(ranges)


def _padding(value: int) -> int:
    try:
        result = int(value)
    except (ValueError, TypeError) as exc:
        raise GapMaskError("Gap padding must be an integer 0..32 px") from exc
    if not 0 <= result <= 32:
        raise GapMaskError("Gap padding must be between 0 and 32 pixels")
    return result


def _shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2 or min(shape) <= 0:
        raise GapMaskError(f"Expected a 2-D positive image shape, got {shape}")
    return int(shape[0]), int(shape[1])


def expand_gap_mask(mask: np.ndarray, padding: int) -> np.ndarray:
    """Dilate selected pixels with an axis-aligned square (no wraparound)."""
    pad = _padding(padding)
    result = np.array(mask, dtype=bool, copy=True)
    if not pad or not np.any(result):
        return result
    # Two 1-D passes: O(padding * image_pixels), no huge structuring element.
    old = result
    rows = old.copy()
    for step in range(1, pad + 1):
        rows[step:, :] |= old[:-step, :]
        rows[:-step, :] |= old[step:, :]
    out = rows.copy()
    for step in range(1, pad + 1):
        out[:, step:] |= rows[:, :-step]
        out[:, :-step] |= rows[:, step:]
    return out


def manual_gap_proposal(shape: tuple[int, int], rows: str, columns: str,
                        padding: int = 0) -> GapProposal:
    height, width = _shape(shape)
    row_ranges = parse_pixel_ranges(rows, height, "Row")
    col_ranges = parse_pixel_ranges(columns, width, "Column")
    if not row_ranges and not col_ranges:
        raise GapMaskError("Enter at least one row or column range.")
    mask = np.zeros((height, width), dtype=bool)
    for first, last in row_ranges:
        mask[first:last + 1, :] = True
    for first, last in col_ranges:
        mask[:, first:last + 1] = True
    mask = expand_gap_mask(mask, padding)
    return GapProposal(mask, f"Manual: {len(row_ranges)} row range(s), {len(col_ranges)} column range(s); padding {padding} px")


def module_gap_proposal(shape: tuple[int, int], modules: list[tuple[tuple[int, int], tuple[int, int]]],
                        padding: int = 0) -> GapProposal:
    """NXdetector_module data_origin/data_size are slow-to-fast: (row, col)."""
    height, width = _shape(shape)
    if len(modules) < 2:
        raise GapMaskError("At least two explicit NXdetector_module rectangles are required.")
    occupied = np.zeros((height, width), dtype=bool)
    minrow, mincol, maxrow, maxcol = height, width, 0, 0
    for origin, size in modules:
        if len(origin) != 2 or len(size) != 2:
            raise GapMaskError("NXdetector_module origin and size must both have two axes.")
        row, col = map(int, origin)
        nrow, ncol = map(int, size)
        if row < 0 or col < 0 or nrow <= 0 or ncol <= 0 or row + nrow > height or col + ncol > width:
            raise GapMaskError(f"Module rectangle {origin}, {size} is outside frame {shape}.")
        patch = occupied[row:row+nrow, col:col+ncol]
        if np.any(patch):
            raise GapMaskError("Module rectangles overlap: ambiguous module geometry.")
        patch[:] = True
        minrow, mincol = min(minrow, row), min(mincol, col)
        maxrow, maxcol = max(maxrow, row + nrow), max(maxcol, col + ncol)
    gap = np.zeros_like(occupied)
    # Only the rectangular envelope spanned by declared modules is considered.
    # Do not invent a gap outside the module area or from pixel intensity values.
    gap[minrow:maxrow, mincol:maxcol] = ~occupied[minrow:maxrow, mincol:maxcol]
    if not np.any(gap):
        raise GapMaskError("Module rectangles contain no uncovered gap pixels.")
    gap = expand_gap_mask(gap, padding)
    return GapProposal(gap, f"NXdetector_module: {len(modules)} modules; padding {padding} px", len(modules))


def _pixel_pair(value: Any, label: str) -> tuple[int, int]:
    arr = np.asarray(value)
    if arr.size != 2 or arr.dtype.kind not in "iu":
        raise GapMaskError(f"{label} must be two integer pixel coordinates [row, column].")
    return tuple(int(v) for v in arr.reshape(-1))


def metadata_module_proposal(archive: Any, shape: tuple[int, int], padding: int = 0) -> GapProposal:
    """Use explicit NXdetector_module data_origin+data_size only.

    Both direct HDF5 and preserved HDXF HDF5 metadata are supported. A source
    pixel_mask is deliberately *not* interpreted as physical module geometry.
    """
    _shape(shape)
    if archive is None:
        raise GapMaskError("Open a detector source first.")
    candidate_paths: set[str] = set()
    for attr in ("_catalog_dataset_paths", "_catalog_group_paths", "_path_objects"):
        values = getattr(archive, attr, ())
        candidate_paths.update(str(x) for x in values)
    detector_prefix = "/entry/instrument/detector/"
    group_paths = set()
    for path in candidate_paths:
        if not path.startswith(detector_prefix):
            continue
        group = path.rsplit("/", 1)[0] if path.endswith(("/data_origin", "/data_size")) else path
        last = group.rsplit("/", 1)[-1]
        if re.fullmatch(r"module(?:[_-]?\d+)?", last, flags=re.IGNORECASE):
            group_paths.add(group)
    modules = []
    for path in sorted(group_paths):
        try:
            origin = archive.read_hdf5_dataset(path + "/data_origin")
            size = archive.read_hdf5_dataset(path + "/data_size")
        except Exception as exc:
            raise GapMaskError(f"Cannot read NXdetector_module {path}: {exc}") from exc
        if origin is None and size is None:
            continue
        if origin is None or size is None:
            raise GapMaskError(f"Incomplete NXdetector_module at {path}: origin and size both required.")
        modules.append((_pixel_pair(origin, path + "/data_origin"),
                        _pixel_pair(size, path + "/data_size")))
    if len(modules) < 2:
        raise GapMaskError("No complete multi-module NXdetector_module geometry found. "
                           "Use Manual rows / columns instead (source pixel_mask is separate).")
    return module_gap_proposal(shape, modules, padding)


def _gap_stripe_intervals(axis_signal: np.ndarray, *, length: int,
                          min_width: int = 3) -> list[tuple[int, int]]:
    """Find *candidate* interior runs with nearly no detector counts.

    Intervals are not declared real module gaps: active rows/columns on BOTH
    sides are required, as are enough contiguous dark pixels. Never apply
    without explicit user confirmation of the preview.
    """
    if length < 2 * 12 + min_width:
        return []
    signal = np.asarray(axis_signal, dtype=np.float64).reshape(-1)
    # The previous fixed 0.5%/2% occupancy cutoffs miss actual module gaps if
    # a few stray nonzero counts occur inside the dark strip, or if the source
    # is sparse. Only RELATIVE contrast between the run and functioning rows on
    # BOTH sides constitutes a candidate. A blank frame stays unsupported.
    baseline = float(np.median(signal))
    if baseline < 0.008:
        return []
    dark_limit = min(0.12, max(0.003, baseline * 0.33))
    dark = np.asarray(signal <= dark_limit, dtype=bool)
    starts = np.flatnonzero(dark & ~np.r_[False, dark[:-1]])
    ends = np.flatnonzero(dark & ~np.r_[dark[1:], False])
    intervals = []
    for first, last in zip(starts.tolist(), ends.tolist()):
        stripe_width = last - first + 1
        if stripe_width < min_width or stripe_width > max(60, length // 8):
            continue
        if first < 12 or last + 13 > length:
            continue
        before = float(np.median(signal[first-12:first]))
        after = float(np.median(signal[last+1:last+13]))
        reference = min(before, after)
        gap_level = float(np.median(signal[first:last+1]))
        # Both flanks must contain genuine detector signal. This rejects dark
        # edges, beamstop shadows, low-count gradients and almost-empty frames.
        if reference < max(0.008, baseline * 0.35):
            continue
        if gap_level > reference * 0.33:
            continue
        intervals.append((first, last))
    return intervals


def image_stripe_gap_proposal(frame: np.ndarray, padding: int = 0) -> GapProposal:
    """Propose *visual* gap candidates using long zero/sentinel-valued stripes.

    This is deliberately NOT a proof of detector geometry; preview first.
    Reject all-zero/sparse frames and edges. It detects long internal stripes
    whose pixel occupancy is markedly lower than the normal region on BOTH
    sides. A few nonzero/sentinel/noisy pixels are tolerated, but sparse/blank
    images and unsupported patterns are rejected. Always preview; never
    silently apply a heuristic-generated mask.
    """
    array = np.asarray(frame)
    if array.ndim != 2:
        raise GapMaskError("Image stripe analysis expects a 2-D detector frame")
    height, width = _shape(tuple(array.shape))
    if height < 32 or width < 32:
        raise GapMaskError("Image stripe analysis needs an image at least 32x32 pixels")
    # Camera gaps usually read as exact zero or integer-minimum sentinels.
    # Exclude non-finite data from positive occupancy; do not manufacture a gap
    # from a bright but uniform row/column.
    valid = np.isfinite(array) & (array != 0)
    if array.dtype.kind == "i":
        valid &= array != np.iinfo(array.dtype).min
    # Positive/signed detector counts may include ordinary negative noise,
    # which remains evidence of a functioning sensor row/column.
    row_signal = np.count_nonzero(valid, axis=1) / float(width)
    col_signal = np.count_nonzero(valid, axis=0) / float(height)
    row_intervals = _gap_stripe_intervals(row_signal, length=height)
    col_intervals = _gap_stripe_intervals(col_signal, length=width)
    if not row_intervals and not col_intervals:
        raise GapMaskError(
            "No confident full-width/height dark stripe candidates in this frame. "
            "Dark gaps with noise or weak image contrast may require manual rows. "
            "This is NOT proof there are no gaps."
        )
    mask = np.zeros(array.shape, dtype=bool)
    for first, last in row_intervals:
        mask[first:last + 1, :] = True
    for first, last in col_intervals:
        mask[:, first:last + 1] = True
    mask = expand_gap_mask(mask, padding)
    def formatted(ranges):
        return ", ".join(f"{first}:{last}" for first, last in ranges) or "none"
    return GapProposal(
        mask,
        "IMAGE HEURISTIC, VERIFY BEFORE APPLY: "
        f"row candidates [{formatted(row_intervals)}], "
        f"column candidates [{formatted(col_intervals)}]; padding {padding} px"
    )
