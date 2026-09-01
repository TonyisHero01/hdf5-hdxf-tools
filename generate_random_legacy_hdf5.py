#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate an Albula-style random HDF5 detector dataset from a legacy template.

The script expects a ``legacy-template`` directory next to this Python file.
The directory should contain one known-good legacy ``*_master.h5`` file.  If
that master references a data file that is also present in the same directory,
the data file is used as a physical/schema skeleton as well.  Otherwise a new
minimal external data file is created.

Default generated detector data:
    frames      : 10
    pixel values: random integers in [1, 1000]
    dtype       : uint16

Output layout:
    <prefix>_master.h5
    <prefix>_data_000001.h5

The master is copied from the legacy template and /entry/data is rebuilt to
point to the generated external data file.  Common detector frame counters are
updated in-place while preserving the template Dataset dtypes/shapes whenever
possible.
"""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    import hdf5plugin  # registers Bitshuffle and related HDF5 filters
except Exception:
    hdf5plugin = None


DATASET_PATH = "/entry/data/data"
DEFAULT_FRAMES = 10
DEFAULT_HEIGHT = 1614
DEFAULT_WIDTH = 1030
DEFAULT_MIN_VALUE = 1
DEFAULT_MAX_VALUE = 1000


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Generate an Albula-style legacy HDF5 master + external data file "
            "containing random detector frames."
        )
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=script_dir / "random_hdf5_output",
        help="Output directory (default: ./random_hdf5_output next to this script).",
    )
    parser.add_argument(
        "--prefix",
        default="random_10frames",
        help="Output filename prefix (default: random_10frames).",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help=(
            "Optional explicit legacy master path. If omitted, the script finds "
            "one *_master.h5 inside ./legacy-template next to this script."
        ),
    )
    parser.add_argument(
        "--frames", type=int, default=DEFAULT_FRAMES,
        help=f"Number of frames (default: {DEFAULT_FRAMES}).",
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help=(
            "Frame height. By default it is inferred from the template data file; "
            f"fallback is {DEFAULT_HEIGHT}."
        ),
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help=(
            "Frame width. By default it is inferred from the template data file; "
            f"fallback is {DEFAULT_WIDTH}."
        ),
    )
    parser.add_argument(
        "--min-value", type=int, default=DEFAULT_MIN_VALUE,
        help=f"Inclusive random minimum (default: {DEFAULT_MIN_VALUE}).",
    )
    parser.add_argument(
        "--max-value", type=int, default=DEFAULT_MAX_VALUE,
        help=f"Inclusive random maximum (default: {DEFAULT_MAX_VALUE}).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed. If omitted, a random seed is generated and printed.",
    )
    parser.add_argument(
        "--compression",
        choices=("template", "bitshuffle-lz4", "gzip", "none"),
        default="template",
        help=(
            "Compression for /entry/data/data. 'template' tries to preserve the "
            "template data Dataset's compression when possible (default)."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Read all generated frames back and verify shape/range/link integrity.",
    )
    return parser.parse_args()


def find_template_master(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Legacy template master not found: {path}")
        return path

    script_dir = Path(__file__).resolve().parent
    candidates_dirs = [
        script_dir / "legacy-template",
        script_dir / "legacy_template",  # also accept the older project spelling
    ]

    existing_dirs = [p for p in candidates_dirs if p.is_dir()]
    if not existing_dirs:
        raise FileNotFoundError(
            "Legacy template directory not found. Expected one of:\n"
            + "\n".join(f"  {p}" for p in candidates_dirs)
        )

    candidates: list[Path] = []
    for folder in existing_dirs:
        candidates.extend(sorted(folder.glob("*_master.h5")))

    # De-duplicate in case a path is reachable through aliases/symlinks.
    unique: list[Path] = []
    seen: set[str] = set()
    for item in candidates:
        resolved = item.resolve()
        key = os.path.normcase(str(resolved))
        if key not in seen and resolved.is_file():
            seen.add(key)
            unique.append(resolved)

    if not unique:
        raise FileNotFoundError(
            "No *_master.h5 was found in the legacy-template directory."
        )
    if len(unique) > 1:
        names = "\n".join(f"  {p}" for p in unique)
        raise RuntimeError(
            "More than one legacy master was found. Please choose one with --template:\n"
            + names
        )
    return unique[0]


def first_external_data_link(master_path: Path) -> tuple[str, str, str] | None:
    """Return (link_name, filename, hdf5_path) for the first /entry/data ExternalLink."""
    with h5py.File(master_path, "r") as handle:
        group = handle.get("/entry/data")
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
                value = np.asarray(obj[()]).reshape(-1)[0]
                return int(value)
        except Exception:
            continue
    return None


def infer_frame_geometry(
    master_path: Path,
    template_data_path: Path | None,
    template_dataset_path: str,
) -> tuple[int, int, np.dtype[Any] | None]:
    """Infer H, W and source dtype from the template when possible."""
    if template_data_path is not None:
        try:
            with h5py.File(template_data_path, "r") as handle:
                ds = handle.get(template_dataset_path)
                if isinstance(ds, h5py.Dataset) and ds.ndim == 3:
                    return int(ds.shape[1]), int(ds.shape[2]), np.dtype(ds.dtype)
        except Exception:
            pass

    # A few common detector dimension scalar names used by EIGER/JUNGFRAU-like files.
    try:
        with h5py.File(master_path, "r") as handle:
            width = _read_scalar_int(
                handle,
                (
                    "/entry/instrument/detector/detectorSpecific/x_pixels_in_detector",
                    "/entry/instrument/detector/x_pixels_in_detector",
                ),
            )
            height = _read_scalar_int(
                handle,
                (
                    "/entry/instrument/detector/detectorSpecific/y_pixels_in_detector",
                    "/entry/instrument/detector/y_pixels_in_detector",
                ),
            )
            if width and height and width > 0 and height > 0:
                return int(height), int(width), None
    except Exception:
        pass

    return DEFAULT_HEIGHT, DEFAULT_WIDTH, None


def copy_attrs(attrs: dict[str, Any], dst: h5py.Dataset) -> None:
    for key, value in attrs.items():
        try:
            dst.attrs[key] = value
        except Exception:
            pass


def capture_dataset_template(ds: h5py.Dataset | None) -> dict[str, Any]:
    if ds is None:
        return {}
    result: dict[str, Any] = {
        "attrs": {key: ds.attrs[key] for key in ds.attrs.keys()},
        "chunks": tuple(int(x) for x in ds.chunks) if ds.chunks else None,
        "compression": ds.compression,
        "compression_opts": ds.compression_opts,
        "shuffle": bool(ds.shuffle),
        "fletcher32": bool(ds.fletcher32),
        "scaleoffset": ds.scaleoffset,
        "fillvalue": ds.fillvalue,
        "dtype": np.dtype(ds.dtype),
    }
    return result


def create_random_dataset(
    parent: h5py.Group,
    name: str,
    *,
    frames: int,
    height: int,
    width: int,
    min_value: int,
    max_value: int,
    rng: np.random.Generator,
    compression_mode: str,
    template_info: dict[str, Any],
) -> h5py.Dataset:
    if name in parent:
        del parent[name]

    shape = (frames, height, width)
    dtype = np.dtype(np.uint16)

    template_chunks = template_info.get("chunks")
    if template_chunks and len(template_chunks) == 3:
        # A frame-aligned chunk is reliable for arbitrary output frame counts.
        chunks = (
            max(1, min(int(template_chunks[0]), frames)),
            max(1, min(int(template_chunks[1]), height)),
            max(1, min(int(template_chunks[2]), width)),
        )
    else:
        chunks = (1, height, width)

    kwargs: dict[str, Any] = {
        "shape": shape,
        "maxshape": (None, height, width),
        "dtype": dtype,
        "chunks": chunks,
    }

    def try_create(candidate_kwargs: dict[str, Any]) -> h5py.Dataset | None:
        try:
            return parent.create_dataset(name, **candidate_kwargs)
        except Exception:
            return None

    ds: h5py.Dataset | None = None
    chosen = "none"

    if compression_mode == "template":
        comp = template_info.get("compression")
        # Preserve standard HDF5 compressors directly.
        if comp in ("gzip", "lzf", "szip"):
            trial = dict(kwargs)
            trial["compression"] = comp
            opts = template_info.get("compression_opts")
            if opts is not None:
                trial["compression_opts"] = opts
            if comp == "gzip" and template_info.get("shuffle"):
                trial["shuffle"] = True
            ds = try_create(trial)
            if ds is not None:
                chosen = f"template:{comp}"
        # Filter 32008 is Bitshuffle in the detector files used by this project.
        elif comp == 32008 and hdf5plugin is not None:
            try:
                ds = parent.create_dataset(
                    name,
                    **kwargs,
                    **hdf5plugin.Bitshuffle(cname="lz4"),
                )
                chosen = "template:bitshuffle-lz4"
            except Exception:
                ds = None

        # If template compression cannot be recreated, prefer Bitshuffle/LZ4
        # when available because it matches the detector ecosystem used here.
        if ds is None and hdf5plugin is not None:
            try:
                ds = parent.create_dataset(
                    name,
                    **kwargs,
                    **hdf5plugin.Bitshuffle(cname="lz4"),
                )
                chosen = "bitshuffle-lz4(fallback)"
            except Exception:
                ds = None

    elif compression_mode == "bitshuffle-lz4":
        if hdf5plugin is None:
            raise RuntimeError(
                "--compression bitshuffle-lz4 requires hdf5plugin. "
                "Install with: python -m pip install hdf5plugin"
            )
        ds = parent.create_dataset(
            name,
            **kwargs,
            **hdf5plugin.Bitshuffle(cname="lz4"),
        )
        chosen = "bitshuffle-lz4"

    elif compression_mode == "gzip":
        trial = dict(kwargs)
        trial.update(compression="gzip", compression_opts=4, shuffle=True)
        ds = parent.create_dataset(name, **trial)
        chosen = "gzip-4+shuffle"

    elif compression_mode == "none":
        ds = parent.create_dataset(name, **kwargs)
        chosen = "none"

    if ds is None:
        ds = parent.create_dataset(name, **kwargs)
        chosen = "none(fallback)"

    copy_attrs(template_info.get("attrs", {}), ds)
    for key, value in (
        ("image_nr_low", np.int64(1)),
        ("image_nr_high", np.int64(frames)),
        ("nimages", np.int64(frames)),
    ):
        try:
            ds.attrs.modify(key, value)
        except Exception:
            try:
                ds.attrs[key] = value
            except Exception:
                pass

    print(f"[DATA] created {ds.name}: shape={ds.shape}, dtype={ds.dtype}, chunks={ds.chunks}")
    print(f"[DATA] compression: {chosen}")

    # Generate/write one frame at a time. This keeps memory low and guarantees
    # the requested inclusive range [min_value, max_value].
    for index in range(frames):
        frame = rng.integers(
            min_value,
            max_value + 1,
            size=(height, width),
            dtype=dtype,
        )
        ds[index, :, :] = frame
        print(f"[WRITE] frame {index + 1}/{frames}")

    return ds


def _set_existing_count(handle: h5py.File, path: str, value: int) -> bool:
    """Set a scalar/size-one Dataset in-place, preserving dtype and dataspace."""
    try:
        obj = handle[path]
    except KeyError:
        return False
    if not isinstance(obj, h5py.Dataset) or obj.size != 1:
        return False
    cast = np.asarray(value, dtype=obj.dtype)
    try:
        if obj.shape == ():
            obj[()] = cast.reshape(()).item()
        else:
            obj[...] = np.full(obj.shape, value, dtype=obj.dtype)
        return True
    except Exception:
        return False


def _create_scalar_count(handle: h5py.File, path: str, value: int, dtype: Any = np.int32) -> None:
    parent_path, name = path.rsplit("/", 1)
    group = handle.require_group(parent_path)
    if name in group:
        return
    group.create_dataset(name, data=np.asarray(value, dtype=dtype))


def update_common_frame_counters(handle: h5py.File, frames: int, *, create_missing: bool) -> None:
    count_paths = (
        "/entry/instrument/detector/detectorSpecific/nimages",
        "/entry/instrument/detector/detectorSpecific/nimages_collected",
        "/entry/instrument/detector/detectorSpecific/nimages_written",
        "/entry/instrument/detector/nimages",
        "/entry/instrument/detector/nimages_collected",
        "/entry/instrument/detector/nimages_written",
    )
    for path in count_paths:
        changed = _set_existing_count(handle, path, frames)
        if changed:
            print(f"[COUNT] {path} = {frames}")

    special = (
        ("/entry/instrument/detector/detectorSpecific/nfiles", 1),
        ("/entry/instrument/detector/detectorSpecific/nimages_per_file", frames),
        ("/entry/instrument/detector/detectorSpecific/ntrigger", frames),
    )
    for path, value in special:
        changed = _set_existing_count(handle, path, value)
        if changed:
            print(f"[COUNT] {path} = {value}")
        elif create_missing:
            # Legacy template uses int32 for these detector counters. Keep that
            # conservative physical type when a missing scalar must be created.
            _create_scalar_count(handle, path, value, np.int32)
            print(f"[COUNT] created {path} = {value} (int32)")


def prepare_data_file(
    output_data: Path,
    template_data: Path | None,
    template_dataset_path: str,
    *,
    frames: int,
    height: int,
    width: int,
    min_value: int,
    max_value: int,
    rng: np.random.Generator,
    compression_mode: str,
) -> None:
    template_info: dict[str, Any] = {}

    if template_data is not None:
        print(f"[TEMPLATE DATA] {template_data}")
        shutil.copy2(template_data, output_data)
        with h5py.File(output_data, "r+") as handle:
            old_ds = handle.get(template_dataset_path)
            if isinstance(old_ds, h5py.Dataset):
                template_info = capture_dataset_template(old_ds)
            parent_path, name = template_dataset_path.rsplit("/", 1)
            parent = handle.require_group(parent_path)
            create_random_dataset(
                parent,
                name,
                frames=frames,
                height=height,
                width=width,
                min_value=min_value,
                max_value=max_value,
                rng=rng,
                compression_mode=compression_mode,
                template_info=template_info,
            )
            update_common_frame_counters(handle, frames, create_missing=False)
            handle.flush()
        return

    print("[TEMPLATE DATA] referenced legacy data file is unavailable; creating a minimal data file")
    with h5py.File(output_data, "w") as handle:
        entry = handle.require_group("/entry")
        try:
            entry.attrs["NX_class"] = np.bytes_("NXentry")
        except Exception:
            pass
        data_group = handle.require_group("/entry/data")
        try:
            data_group.attrs["NX_class"] = np.bytes_("NXdata")
        except Exception:
            pass
        create_random_dataset(
            data_group,
            "data",
            frames=frames,
            height=height,
            width=width,
            min_value=min_value,
            max_value=max_value,
            rng=rng,
            compression_mode=("bitshuffle-lz4" if compression_mode == "template" and hdf5plugin is not None else compression_mode),
            template_info={},
        )
        update_common_frame_counters(handle, frames, create_missing=True)
        handle.flush()


def prepare_master(output_master: Path, template_master: Path, output_data: Path, frames: int) -> None:
    shutil.copy2(template_master, output_master)
    with h5py.File(output_master, "r+") as handle:
        data_group = handle.require_group("/entry/data")
        for name in list(data_group.keys()):
            del data_group[name]
        data_group["data_000001"] = h5py.ExternalLink(output_data.name, DATASET_PATH)
        print(
            f"[MASTER] /entry/data/data_000001 -> "
            f"{output_data.name}:{DATASET_PATH}"
        )
        update_common_frame_counters(handle, frames, create_missing=True)
        try:
            handle["/"].attrs["generated_random_test_data"] = np.bytes_(
                "generate_random_legacy_hdf5.py"
            )
        except Exception:
            pass
        handle.flush()


def verify_output(
    master_path: Path,
    data_path: Path,
    *,
    frames: int,
    height: int,
    width: int,
    min_value: int,
    max_value: int,
) -> None:
    print("[VERIFY] opening master and following ExternalLink")
    with h5py.File(master_path, "r") as master:
        link = master["/entry/data"].get("data_000001", getlink=True)
        if not isinstance(link, h5py.ExternalLink):
            raise RuntimeError("/entry/data/data_000001 is not an ExternalLink")
        if Path(link.filename).name != data_path.name:
            raise RuntimeError(
                f"ExternalLink filename mismatch: {link.filename!r} != {data_path.name!r}"
            )
        linked_ds = master["/entry/data/data_000001"]
        if linked_ds.shape != (frames, height, width):
            raise RuntimeError(f"Linked Dataset shape mismatch: {linked_ds.shape}")

    print("[VERIFY] reading every generated frame")
    global_min: int | None = None
    global_max: int | None = None
    with h5py.File(data_path, "r") as data_file:
        ds = data_file[DATASET_PATH]
        if ds.shape != (frames, height, width):
            raise RuntimeError(f"Data Dataset shape mismatch: {ds.shape}")
        if np.dtype(ds.dtype) != np.dtype(np.uint16):
            raise RuntimeError(f"Unexpected dtype: {ds.dtype}")
        for index in range(frames):
            frame = ds[index]
            fmin = int(frame.min())
            fmax = int(frame.max())
            global_min = fmin if global_min is None else min(global_min, fmin)
            global_max = fmax if global_max is None else max(global_max, fmax)
            if fmin < min_value or fmax > max_value:
                raise RuntimeError(
                    f"Frame {index + 1} contains value outside [{min_value}, {max_value}]: "
                    f"min={fmin}, max={fmax}"
                )
            print(f"[VERIFY] frame {index + 1}/{frames}: min={fmin}, max={fmax}")

    print(
        f"[VERIFY] PASS - shape=({frames},{height},{width}), dtype=uint16, "
        f"observed range=[{global_min},{global_max}]"
    )


def main() -> int:
    args = parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be > 0")
    if args.min_value < 0:
        raise ValueError("--min-value must be >= 0 for uint16 output")
    if args.max_value < args.min_value:
        raise ValueError("--max-value must be >= --min-value")
    if args.max_value > np.iinfo(np.uint16).max:
        raise ValueError(f"--max-value must be <= {np.iinfo(np.uint16).max}")

    template_master = find_template_master(args.template)
    template_data, template_dataset_path = resolve_template_data(template_master)
    inferred_h, inferred_w, inferred_dtype = infer_frame_geometry(
        template_master,
        template_data,
        template_dataset_path,
    )

    height = int(args.height if args.height is not None else inferred_h)
    width = int(args.width if args.width is not None else inferred_w)
    if height <= 0 or width <= 0:
        raise ValueError("Frame height/width must be > 0")

    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix.strip()
    if not prefix:
        raise ValueError("--prefix must not be empty")

    output_master = output_dir / f"{prefix}_master.h5"
    output_data = output_dir / f"{prefix}_data_000001.h5"
    for path in (output_master, output_data):
        if path.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Output already exists: {path}\nUse --overwrite to replace it."
                )
            path.unlink()

    seed = int(args.seed) if args.seed is not None else secrets.randbits(63)
    rng = np.random.default_rng(seed)

    print(f"[TEMPLATE MASTER] {template_master}")
    print(f"[TEMPLATE DATA]   {template_data if template_data else '<not found>'}")
    print(
        f"[CONFIG] frames={args.frames}, shape=({height},{width}), dtype=uint16, "
        f"random=[{args.min_value},{args.max_value}]"
    )
    if inferred_dtype is not None:
        print(f"[INFO] template source frame dtype={inferred_dtype}; generated dtype=uint16")
    print(f"[SEED] {seed}")
    print(f"[OUTPUT] {output_dir}")

    # Create the external data first, then install the master link. This avoids
    # leaving a master that points at an incomplete data file if generation fails.
    prepare_data_file(
        output_data,
        template_data,
        template_dataset_path,
        frames=args.frames,
        height=height,
        width=width,
        min_value=args.min_value,
        max_value=args.max_value,
        rng=rng,
        compression_mode=args.compression,
    )
    prepare_master(output_master, template_master, output_data, args.frames)

    print(f"[OUT data]   {output_data}")
    print(f"[OUT master] {output_master}")

    if args.verify:
        verify_output(
            output_master,
            output_data,
            frames=args.frames,
            height=height,
            width=width,
            min_value=args.min_value,
            max_value=args.max_value,
        )

    print("[RESULT] PASS - random Albula-style HDF5 dataset generated")
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
