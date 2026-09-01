#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate two extreme Albula-style HDF5 datasets for HDXF compression tests.

The script expects a ``legacy-template`` (or ``legacy_template``) directory
next to this Python file.  The directory should contain one known-good legacy
``*_master.h5`` file.  If the master references a data file that is also
present, that file is used as the physical/schema skeleton for the generated
external data files.

Two datasets are generated with identical frame count, geometry and dtype:

BEST case
    Every pixel in every frame has the same uint16 value (default: 1).
    Spatial entropy is extremely low and temporal delta after the first frame
    is exactly zero.  This is intentionally favourable to HDXF compression.

WORST case
    Every pixel of every frame is independently sampled from the complete
    uint16 range [0, 65535].  Spatial entropy is close to 16 bits/pixel and
    adjacent frames are statistically unrelated.  This is intentionally
    unfavourable to temporal-delta compression.

Default output:
    hdxf_extreme_hdf5_output/
        hdxf_BEST_10frames_master.h5
        hdxf_BEST_10frames_data_000001.h5
        hdxf_WORST_10frames_master.h5
        hdxf_WORST_10frames_data_000001.h5

Example:
    python generate_hdxf_extreme_hdf5.py --overwrite --verify

For a controlled same-range comparison rather than the absolute worst case:
    python generate_hdxf_extreme_hdf5.py \
        --worst-min 1 --worst-max 1000 --overwrite --verify
"""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

try:
    import hdf5plugin  # registers detector-oriented HDF5 filters
except Exception:
    hdf5plugin = None


DATASET_PATH = "/entry/data/data"
DEFAULT_FRAMES = 10
DEFAULT_HEIGHT = 1614
DEFAULT_WIDTH = 1030
UINT16_MAX = int(np.iinfo(np.uint16).max)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description=(
            "Generate BEST-case and WORST-case Albula-style HDF5 detector "
            "datasets for HDXF compression testing."
        )
    )
    ap.add_argument(
        "-o", "--output",
        type=Path,
        default=script_dir / "hdxf_extreme_hdf5_output",
        help="Output directory (default: ./hdxf_extreme_hdf5_output).",
    )
    ap.add_argument(
        "--template",
        type=Path,
        default=None,
        help=(
            "Explicit legacy *_master.h5. If omitted, exactly one master is "
            "auto-detected in ./legacy-template or ./legacy_template."
        ),
    )
    ap.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help=f"Frames in each dataset (default: {DEFAULT_FRAMES}).",
    )
    ap.add_argument(
        "--height",
        type=int,
        default=None,
        help="Frame height. Default: infer from legacy data; fallback 1614.",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=None,
        help="Frame width. Default: infer from legacy data; fallback 1030.",
    )
    ap.add_argument(
        "--best-value",
        type=int,
        default=1,
        help="Constant uint16 value for BEST dataset (default: 1).",
    )
    ap.add_argument(
        "--worst-min",
        type=int,
        default=0,
        help="Inclusive minimum for WORST random pixels (default: 0).",
    )
    ap.add_argument(
        "--worst-max",
        type=int,
        default=UINT16_MAX,
        help=f"Inclusive maximum for WORST random pixels (default: {UINT16_MAX}).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for WORST dataset. Randomly generated if omitted.",
    )
    ap.add_argument(
        "--compression",
        choices=("template", "bitshuffle-lz4", "gzip", "none"),
        default="template",
        help=(
            "HDF5 Dataset compression. This affects source .h5 size but not the "
            "logical pixels given to the HDXF converter. Default: template."
        ),
    )
    ap.add_argument(
        "--best-prefix",
        default="hdxf_BEST_10frames",
        help="Filename prefix for BEST dataset.",
    )
    ap.add_argument(
        "--worst-prefix",
        default="hdxf_WORST_10frames",
        help="Filename prefix for WORST dataset.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite outputs.")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Read every generated frame back and verify the two patterns.",
    )
    return ap.parse_args()


def find_template_master(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Legacy template master not found: {path}")
        return path

    script_dir = Path(__file__).resolve().parent
    roots = [script_dir / "legacy-template", script_dir / "legacy_template"]
    candidates: list[Path] = []
    for root in roots:
        if root.is_dir():
            candidates.extend(sorted(root.glob("*_master.h5")))

    unique: list[Path] = []
    seen: set[str] = set()
    for item in candidates:
        resolved = item.resolve()
        key = os.path.normcase(str(resolved))
        if resolved.is_file() and key not in seen:
            seen.add(key)
            unique.append(resolved)

    if not unique:
        raise FileNotFoundError(
            "No legacy *_master.h5 found. Put it next to this script under "
            "legacy-template/ (or legacy_template/)."
        )
    if len(unique) != 1:
        listing = "\n".join(f"  {p}" for p in unique)
        raise RuntimeError(
            "More than one legacy master found. Use --template to choose one:\n"
            + listing
        )
    return unique[0]


def first_external_data_link(master_path: Path) -> tuple[str, str, str] | None:
    with h5py.File(master_path, "r") as f:
        group = f.get("/entry/data")
        if not isinstance(group, h5py.Group):
            return None
        for name in sorted(group.keys()):
            link = group.get(name, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                return str(name), str(link.filename), str(link.path)
    return None


def resolve_template_data(master_path: Path) -> tuple[Path | None, str]:
    link = first_external_data_link(master_path)
    if link is None:
        return None, DATASET_PATH
    _name, filename, hdf5_path = link
    target = (master_path.parent / filename).resolve()
    return (target if target.is_file() else None), hdf5_path


def _read_scalar_int(handle: h5py.File, paths: tuple[str, ...]) -> int | None:
    for path in paths:
        try:
            obj = handle[path]
            if isinstance(obj, h5py.Dataset) and obj.size >= 1:
                return int(np.asarray(obj[()]).reshape(-1)[0])
        except Exception:
            pass
    return None


def infer_geometry(
    master_path: Path,
    template_data: Path | None,
    template_dataset_path: str,
) -> tuple[int, int]:
    if template_data is not None:
        try:
            with h5py.File(template_data, "r") as f:
                ds = f.get(template_dataset_path)
                if isinstance(ds, h5py.Dataset) and ds.ndim == 3:
                    return int(ds.shape[1]), int(ds.shape[2])
        except Exception:
            pass

    try:
        with h5py.File(master_path, "r") as f:
            width = _read_scalar_int(
                f,
                (
                    "/entry/instrument/detector/detectorSpecific/x_pixels_in_detector",
                    "/entry/instrument/detector/x_pixels_in_detector",
                ),
            )
            height = _read_scalar_int(
                f,
                (
                    "/entry/instrument/detector/detectorSpecific/y_pixels_in_detector",
                    "/entry/instrument/detector/y_pixels_in_detector",
                ),
            )
            if width and height and width > 0 and height > 0:
                return height, width
    except Exception:
        pass
    return DEFAULT_HEIGHT, DEFAULT_WIDTH


def capture_dataset_template(ds: h5py.Dataset | None) -> dict[str, Any]:
    if ds is None:
        return {}
    return {
        "attrs": {k: ds.attrs[k] for k in ds.attrs.keys()},
        "chunks": tuple(int(x) for x in ds.chunks) if ds.chunks else None,
        "compression": ds.compression,
        "compression_opts": ds.compression_opts,
        "shuffle": bool(ds.shuffle),
    }


def copy_attrs(attrs: dict[str, Any], dst: h5py.Dataset) -> None:
    for key, value in attrs.items():
        try:
            dst.attrs[key] = value
        except Exception:
            pass


def create_detector_dataset(
    parent: h5py.Group,
    name: str,
    *,
    frames: int,
    height: int,
    width: int,
    compression_mode: str,
    template_info: dict[str, Any],
) -> tuple[h5py.Dataset, str]:
    if name in parent:
        del parent[name]

    template_chunks = template_info.get("chunks")
    if template_chunks and len(template_chunks) == 3:
        chunks = (
            max(1, min(int(template_chunks[0]), frames)),
            max(1, min(int(template_chunks[1]), height)),
            max(1, min(int(template_chunks[2]), width)),
        )
    else:
        chunks = (1, height, width)

    base: dict[str, Any] = {
        "shape": (frames, height, width),
        "maxshape": (None, height, width),
        "dtype": np.uint16,
        "chunks": chunks,
    }

    ds: h5py.Dataset | None = None
    chosen = "none"

    if compression_mode == "template":
        comp = template_info.get("compression")
        if comp in ("gzip", "lzf", "szip"):
            kwargs = dict(base)
            kwargs["compression"] = comp
            opts = template_info.get("compression_opts")
            if opts is not None:
                kwargs["compression_opts"] = opts
            if comp == "gzip" and template_info.get("shuffle"):
                kwargs["shuffle"] = True
            try:
                ds = parent.create_dataset(name, **kwargs)
                chosen = f"template:{comp}"
            except Exception:
                ds = None
        elif comp == 32008 and hdf5plugin is not None:
            try:
                ds = parent.create_dataset(
                    name, **base, **hdf5plugin.Bitshuffle(cname="lz4")
                )
                chosen = "template:bitshuffle-lz4"
            except Exception:
                ds = None

        if ds is None and hdf5plugin is not None:
            try:
                ds = parent.create_dataset(
                    name, **base, **hdf5plugin.Bitshuffle(cname="lz4")
                )
                chosen = "bitshuffle-lz4(fallback)"
            except Exception:
                ds = None

    elif compression_mode == "bitshuffle-lz4":
        if hdf5plugin is None:
            raise RuntimeError(
                "--compression bitshuffle-lz4 requires hdf5plugin."
            )
        ds = parent.create_dataset(
            name, **base, **hdf5plugin.Bitshuffle(cname="lz4")
        )
        chosen = "bitshuffle-lz4"

    elif compression_mode == "gzip":
        ds = parent.create_dataset(
            name, **base, compression="gzip", compression_opts=4, shuffle=True
        )
        chosen = "gzip-4+shuffle"

    elif compression_mode == "none":
        ds = parent.create_dataset(name, **base)
        chosen = "none"

    if ds is None:
        ds = parent.create_dataset(name, **base)
        chosen = "none(fallback)"

    copy_attrs(template_info.get("attrs", {}), ds)
    for key, value in (
        ("image_nr_low", np.int64(1)),
        ("image_nr_high", np.int64(frames)),
        ("nimages", np.int64(frames)),
    ):
        try:
            ds.attrs[key] = value
        except Exception:
            pass

    return ds, chosen


def _set_existing_count(f: h5py.File, path: str, value: int) -> bool:
    try:
        ds = f[path]
    except KeyError:
        return False
    if not isinstance(ds, h5py.Dataset) or ds.size != 1:
        return False
    try:
        if ds.shape == ():
            ds[()] = np.asarray(value, dtype=ds.dtype).reshape(()).item()
        else:
            ds[...] = np.full(ds.shape, value, dtype=ds.dtype)
        return True
    except Exception:
        return False


def _create_scalar_count(f: h5py.File, path: str, value: int) -> None:
    parent_path, name = path.rsplit("/", 1)
    group = f.require_group(parent_path)
    if name not in group:
        group.create_dataset(name, data=np.asarray(value, dtype=np.int32))


def update_frame_counters(f: h5py.File, frames: int, create_missing: bool) -> None:
    for path in (
        "/entry/instrument/detector/detectorSpecific/nimages",
        "/entry/instrument/detector/detectorSpecific/nimages_collected",
        "/entry/instrument/detector/detectorSpecific/nimages_written",
        "/entry/instrument/detector/nimages",
        "/entry/instrument/detector/nimages_collected",
        "/entry/instrument/detector/nimages_written",
    ):
        _set_existing_count(f, path, frames)

    for path, value in (
        ("/entry/instrument/detector/detectorSpecific/nfiles", 1),
        ("/entry/instrument/detector/detectorSpecific/nimages_per_file", frames),
        ("/entry/instrument/detector/detectorSpecific/ntrigger", frames),
    ):
        if not _set_existing_count(f, path, value) and create_missing:
            _create_scalar_count(f, path, value)


def prepare_data_file(
    output_data: Path,
    template_data: Path | None,
    template_dataset_path: str,
    *,
    frames: int,
    height: int,
    width: int,
    compression_mode: str,
    writer: Callable[[h5py.Dataset], None],
) -> None:
    template_info: dict[str, Any] = {}

    if template_data is not None:
        shutil.copy2(template_data, output_data)
        with h5py.File(output_data, "r+") as f:
            old = f.get(template_dataset_path)
            if isinstance(old, h5py.Dataset):
                template_info = capture_dataset_template(old)
            parent_path, name = template_dataset_path.rsplit("/", 1)
            parent = f.require_group(parent_path)
            ds, chosen = create_detector_dataset(
                parent,
                name,
                frames=frames,
                height=height,
                width=width,
                compression_mode=compression_mode,
                template_info=template_info,
            )
            print(f"[DATA] {output_data.name}: {ds.name} shape={ds.shape} chunks={ds.chunks}")
            print(f"[DATA] HDF5 compression={chosen}")
            writer(ds)
            update_frame_counters(f, frames, create_missing=False)
            f.flush()
        return

    with h5py.File(output_data, "w") as f:
        entry = f.require_group("/entry")
        try:
            entry.attrs["NX_class"] = np.bytes_("NXentry")
        except Exception:
            pass
        data_group = f.require_group("/entry/data")
        try:
            data_group.attrs["NX_class"] = np.bytes_("NXdata")
        except Exception:
            pass
        fallback_comp = compression_mode
        if fallback_comp == "template":
            fallback_comp = "bitshuffle-lz4" if hdf5plugin is not None else "none"
        ds, chosen = create_detector_dataset(
            data_group,
            "data",
            frames=frames,
            height=height,
            width=width,
            compression_mode=fallback_comp,
            template_info={},
        )
        print(f"[DATA] {output_data.name}: {ds.name} shape={ds.shape} chunks={ds.chunks}")
        print(f"[DATA] HDF5 compression={chosen}")
        writer(ds)
        update_frame_counters(f, frames, create_missing=True)
        f.flush()


def prepare_master(
    output_master: Path,
    template_master: Path,
    output_data: Path,
    frames: int,
    label: str,
) -> None:
    shutil.copy2(template_master, output_master)
    with h5py.File(output_master, "r+") as f:
        group = f.require_group("/entry/data")
        for name in list(group.keys()):
            del group[name]
        group["data_000001"] = h5py.ExternalLink(output_data.name, DATASET_PATH)
        update_frame_counters(f, frames, create_missing=True)
        try:
            f["/"].attrs["hdxf_compression_test_case"] = np.bytes_(label)
            f["/"].attrs["generated_by"] = np.bytes_(
                "generate_hdxf_extreme_hdf5.py"
            )
        except Exception:
            pass
        f.flush()


def write_best(ds: h5py.Dataset, frames: int, value: int) -> None:
    frame = np.full((ds.shape[1], ds.shape[2]), value, dtype=np.uint16)
    for i in range(frames):
        ds[i, :, :] = frame
        print(f"[BEST WRITE] {i + 1}/{frames}")


def write_worst(
    ds: h5py.Dataset,
    frames: int,
    rng: np.random.Generator,
    low: int,
    high: int,
) -> None:
    for i in range(frames):
        frame = rng.integers(
            low,
            high + 1,
            size=(ds.shape[1], ds.shape[2]),
            dtype=np.uint16,
        )
        ds[i, :, :] = frame
        print(f"[WORST WRITE] {i + 1}/{frames}")


def verify_master_link(master_path: Path, data_path: Path, expected_shape: tuple[int, int, int]) -> None:
    with h5py.File(master_path, "r") as f:
        link = f["/entry/data"].get("data_000001", getlink=True)
        if not isinstance(link, h5py.ExternalLink):
            raise RuntimeError(f"{master_path.name}: data_000001 is not ExternalLink")
        if Path(link.filename).name != data_path.name:
            raise RuntimeError(f"{master_path.name}: ExternalLink target mismatch")
        ds = f["/entry/data/data_000001"]
        if tuple(ds.shape) != expected_shape:
            raise RuntimeError(
                f"{master_path.name}: linked shape {ds.shape} != {expected_shape}"
            )


def verify_best(data_path: Path, frames: int, height: int, width: int, value: int) -> None:
    with h5py.File(data_path, "r") as f:
        ds = f[DATASET_PATH]
        if tuple(ds.shape) != (frames, height, width):
            raise RuntimeError("BEST shape mismatch")
        for i in range(frames):
            frame = ds[i]
            if not np.all(frame == value):
                raise RuntimeError(f"BEST frame {i + 1} is not constant {value}")
    print("[VERIFY BEST] PASS - every pixel in every frame is identical")


def verify_worst(
    data_path: Path,
    frames: int,
    height: int,
    width: int,
    low: int,
    high: int,
) -> None:
    changed_total = 0
    compared_total = 0
    zero_delta_total = 0
    prev: np.ndarray | None = None

    with h5py.File(data_path, "r") as f:
        ds = f[DATASET_PATH]
        if tuple(ds.shape) != (frames, height, width):
            raise RuntimeError("WORST shape mismatch")
        for i in range(frames):
            frame = ds[i]
            fmin = int(frame.min())
            fmax = int(frame.max())
            if fmin < low or fmax > high:
                raise RuntimeError(f"WORST frame {i + 1} outside [{low},{high}]")
            if prev is not None:
                equal = np.count_nonzero(frame == prev)
                pixels = frame.size
                zero_delta_total += equal
                changed_total += pixels - equal
                compared_total += pixels
            prev = frame

    changed_pct = 100.0 * changed_total / compared_total if compared_total else 0.0
    zero_delta_pct = 100.0 * zero_delta_total / compared_total if compared_total else 0.0
    print(
        f"[VERIFY WORST] PASS - adjacent changed pixels={changed_pct:.6f}% "
        f"zero-delta={zero_delta_pct:.6f}%"
    )


def file_size_text(path: Path) -> str:
    size = path.stat().st_size
    return f"{size / (1024 ** 2):.2f} MiB"


def main() -> int:
    args = parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be > 0")
    for name, value in (
        ("--best-value", args.best_value),
        ("--worst-min", args.worst_min),
        ("--worst-max", args.worst_max),
    ):
        if value < 0 or value > UINT16_MAX:
            raise ValueError(f"{name} must be in [0,{UINT16_MAX}]")
    if args.worst_max < args.worst_min:
        raise ValueError("--worst-max must be >= --worst-min")

    template_master = find_template_master(args.template)
    template_data, template_dataset_path = resolve_template_data(template_master)
    inferred_h, inferred_w = infer_geometry(
        template_master, template_data, template_dataset_path
    )
    height = int(args.height if args.height is not None else inferred_h)
    width = int(args.width if args.width is not None else inferred_w)
    if height <= 0 or width <= 0:
        raise ValueError("height/width must be > 0")

    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    best_master = output_dir / f"{args.best_prefix}_master.h5"
    best_data = output_dir / f"{args.best_prefix}_data_000001.h5"
    worst_master = output_dir / f"{args.worst_prefix}_master.h5"
    worst_data = output_dir / f"{args.worst_prefix}_data_000001.h5"
    outputs = (best_master, best_data, worst_master, worst_data)

    for path in outputs:
        if path.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Output exists: {path}\nUse --overwrite to replace existing files."
                )
            path.unlink()

    seed = int(args.seed) if args.seed is not None else secrets.randbits(63)
    rng = np.random.default_rng(seed)
    raw_bytes = args.frames * height * width * np.dtype(np.uint16).itemsize

    print(f"[TEMPLATE MASTER] {template_master}")
    print(f"[TEMPLATE DATA]   {template_data if template_data else '<not found>'}")
    print(f"[GEOMETRY] frames={args.frames}, shape=({height},{width}), dtype=uint16")
    print(f"[LOGICAL RAW] {raw_bytes / (1024 ** 2):.2f} MiB per test dataset")
    print(f"[BEST] every pixel = {args.best_value}")
    print(f"[WORST] independent uniform random [{args.worst_min},{args.worst_max}]")
    print(f"[SEED] {seed}")
    print(f"[OUTPUT] {output_dir}")

    print("\n========== GENERATING BEST CASE ==========")
    prepare_data_file(
        best_data,
        template_data,
        template_dataset_path,
        frames=args.frames,
        height=height,
        width=width,
        compression_mode=args.compression,
        writer=lambda ds: write_best(ds, args.frames, args.best_value),
    )
    prepare_master(
        best_master,
        template_master,
        best_data,
        args.frames,
        "BEST_CONSTANT_IDENTICAL_FRAMES",
    )

    print("\n========== GENERATING WORST CASE =========")
    prepare_data_file(
        worst_data,
        template_data,
        template_dataset_path,
        frames=args.frames,
        height=height,
        width=width,
        compression_mode=args.compression,
        writer=lambda ds: write_worst(
            ds, args.frames, rng, args.worst_min, args.worst_max
        ),
    )
    prepare_master(
        worst_master,
        template_master,
        worst_data,
        args.frames,
        "WORST_INDEPENDENT_FULL_RANGE_RANDOM_FRAMES",
    )

    print("\n========== OUTPUT ==========")
    print(f"[BEST master]  {best_master}")
    print(f"[BEST data]    {best_data} ({file_size_text(best_data)})")
    print(f"[WORST master] {worst_master}")
    print(f"[WORST data]   {worst_data} ({file_size_text(worst_data)})")

    if args.verify:
        print("\n========== VERIFY ==========")
        shape = (args.frames, height, width)
        verify_master_link(best_master, best_data, shape)
        verify_master_link(worst_master, worst_data, shape)
        verify_best(best_data, args.frames, height, width, args.best_value)
        verify_worst(
            worst_data,
            args.frames,
            height,
            width,
            args.worst_min,
            args.worst_max,
        )
        print("[VERIFY] PASS - both extreme HDF5 datasets are valid")

    print("\n[HDXF TEST EXPECTATION]")
    print("  BEST : very high compression; temporal Delta after first frame is all zero")
    print("  WORST: low compression; adjacent frames are nearly independent high-entropy data")
    print("[RESULT] PASS - HDXF compression extreme test datasets generated")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[STOP] interrupted by user", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
