#!/usr/bin/env python3
"""Structure-preserving HDF5 detector-frame processing backend.

The input is an HDF5 master file.  Pixel/filter operations clone the complete
master + external-data bundle and only patch selected detector frames.  Sum and
mean create a one-frame master/data bundle from the current file layout: the
master and the representative data file are copied first, existing groups,
datasets and attributes are retained, and frame-dependent datasets are reduced
to one record.

The one unavoidable schema change for Sum/Mean is cardinality: a multi-file
``/entry/data`` sequence becomes one ExternalLink because the result has one
frame.  The group itself, target dataset path, and the complete master/data
subtrees remain based on the source template.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

try:
    import hdf5plugin  # type: ignore  # noqa: F401
except Exception:
    hdf5plugin = None  # type: ignore

import h5py
import numpy as np


class ProcessingError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrameSource:
    link_name: str
    file_path: Path
    dataset_path: str
    frame_count: int
    frame_shape: tuple[int, int]
    dtype: np.dtype
    global_start: int


def _as_path(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().resolve()


def _external_target(master: Path, filename: str) -> Path:
    candidate = Path(os.fsdecode(filename))
    if not candidate.is_absolute():
        candidate = master.parent / candidate
    return candidate.resolve()


def discover_frame_sources(master: Path) -> list[FrameSource]:
    result: list[FrameSource] = []
    start = 0
    with h5py.File(master, "r") as handle:
        if "/entry/data" not in handle:
            raise ProcessingError("input master is missing /entry/data")
        group = handle["/entry/data"]
        if isinstance(group, h5py.Dataset):
            shape = tuple(int(v) for v in group.shape)
            if len(shape) == 2:
                count, image_shape = 1, shape
            elif len(shape) >= 3:
                count, image_shape = int(np.prod(shape[:-2])), shape[-2:]
            else:
                raise ProcessingError("/entry/data is not a 2D/3D frame dataset")
            result.append(FrameSource("data", master, "/entry/data", count, image_shape, group.dtype, 0))
            return result
        if not isinstance(group, h5py.Group):
            raise ProcessingError("/entry/data is neither a group nor a dataset")

        for name in sorted(group.keys()):
            link = group.get(name, getlink=True)
            dataset_path = f"/entry/data/{name}"
            file_path = master
            target_path = dataset_path
            if isinstance(link, h5py.ExternalLink):
                file_path = _external_target(master, os.fsdecode(link.filename))
                target_path = str(link.path)
                if not file_path.is_file():
                    raise ProcessingError(f"external data file is missing: {file_path}")
                with h5py.File(file_path, "r") as external:
                    if target_path not in external or not isinstance(external[target_path], h5py.Dataset):
                        raise ProcessingError(f"external target is not a dataset: {file_path}:{target_path}")
                    dataset = external[target_path]
                    shape = tuple(int(v) for v in dataset.shape)
                    dtype = np.dtype(dataset.dtype)
            else:
                obj = group[name]
                if not isinstance(obj, h5py.Dataset):
                    continue
                shape = tuple(int(v) for v in obj.shape)
                dtype = np.dtype(obj.dtype)
            if len(shape) == 2:
                count, image_shape = 1, shape
            elif len(shape) >= 3:
                count, image_shape = int(np.prod(shape[:-2])), shape[-2:]
            else:
                continue
            if count <= 0 or len(image_shape) != 2 or dtype.kind not in "biuf":
                continue
            source = FrameSource(name, file_path, target_path, count, image_shape, dtype, start)
            result.append(source)
            start += count

    if not result:
        raise ProcessingError("no numeric detector-frame dataset was found under /entry/data")
    expected_shape = result[0].frame_shape
    if any(item.frame_shape != expected_shape for item in result):
        raise ProcessingError("detector data links do not share one frame shape")
    return result


def parse_selection(total: int, selected: str | None, frame_from: int | None, frame_to: int | None) -> list[int]:
    if total <= 0:
        raise ProcessingError("input has no frames")
    if selected:
        try:
            one_based = [int(item.strip()) for item in selected.split(",") if item.strip()]
        except ValueError as exc:
            raise ProcessingError("--select-frame must contain comma-separated integers") from exc
        if not one_based:
            raise ProcessingError("--select-frame is empty")
        indices = sorted(set(value - 1 for value in one_based))
    else:
        lo = 1 if frame_from is None else int(frame_from)
        hi = total if frame_to is None else int(frame_to)
        if lo < 1 or hi < lo:
            raise ProcessingError("invalid inclusive frame range")
        indices = list(range(lo - 1, hi))
    if not indices or indices[0] < 0 or indices[-1] >= total:
        raise ProcessingError(f"selected frames must be within 1..{total}")
    return indices


def _selection_by_source(sources: list[FrameSource], indices: Iterable[int]) -> dict[FrameSource, list[int]]:
    result = {source: [] for source in sources}
    for index in indices:
        for source in sources:
            if source.global_start <= index < source.global_start + source.frame_count:
                result[source].append(index - source.global_start)
                break
        else:
            raise ProcessingError(f"no data source covers global frame {index + 1}")
    return result


def _copy_attrs(source: h5py.Dataset, target: h5py.Dataset) -> None:
    for key, value in source.attrs.items():
        target.attrs[key] = value


def _dataset_kwargs(source: h5py.Dataset, shape: tuple[int, ...]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if source.chunks is not None and shape:
        kwargs["chunks"] = tuple(max(1, min(int(chunk), int(dim) if int(dim) > 0 else 1)) for chunk, dim in zip(source.chunks, shape))
    if source.compression is not None:
        kwargs["compression"] = source.compression
        if source.compression_opts is not None:
            kwargs["compression_opts"] = source.compression_opts
    if source.shuffle:
        kwargs["shuffle"] = True
    if source.fletcher32:
        kwargs["fletcher32"] = True
    if source.scaleoffset is not None:
        kwargs["scaleoffset"] = source.scaleoffset
    if source.fillvalue is not None:
        kwargs["fillvalue"] = source.fillvalue
    if source.maxshape is not None and len(source.maxshape) == len(shape):
        maxshape: list[int | None] = []
        for dim, maximum in zip(shape, source.maxshape):
            if maximum is None:
                maxshape.append(None)
            else:
                maxshape.append(max(int(dim), int(maximum)))
        kwargs["maxshape"] = tuple(maxshape)
    return kwargs


def replace_dataset(dataset: h5py.Dataset, data: np.ndarray, *, dtype: np.dtype | None = None) -> h5py.Dataset:
    """Replace/resize a dataset while retaining its name, attributes and layout."""
    array = np.asarray(data)
    source_dtype = np.dtype(dataset.dtype if dtype is None else dtype)
    if array.dtype != source_dtype:
        array = array.astype(source_dtype, copy=False)
    new_shape = tuple(int(v) for v in array.shape)
    old_attrs = list(dataset.attrs.items())

    if dataset.chunks is not None and len(dataset.shape) == len(new_shape) and np.dtype(dataset.dtype) == source_dtype:
        try:
            dataset.resize(new_shape)
            dataset[...] = array
            return dataset
        except Exception:
            pass

    parent = dataset.parent
    name = dataset.name.rsplit("/", 1)[-1]
    kwargs = _dataset_kwargs(dataset, new_shape)
    del parent[name]
    try:
        new_dataset = parent.create_dataset(name, data=array, dtype=source_dtype, **kwargs)
    except Exception as exc:
        # A third-party filter may be readable but not creatable on this host.
        # Keep the dtype/chunk model and fall back to gzip rather than dropping
        # the dataset or its metadata.
        fallback: dict[str, Any] = {}
        if new_shape:
            fallback["chunks"] = tuple(1 if i == 0 else max(1, min(new_shape[i], 256)) for i in range(len(new_shape)))
            fallback.update(compression="gzip", compression_opts=4, shuffle=True)
        try:
            new_dataset = parent.create_dataset(name, data=array, dtype=source_dtype, **fallback)
        except Exception:
            raise ProcessingError(f"cannot recreate resized dataset {dataset.name}: {exc}") from exc
    for key, value in old_attrs:
        new_dataset.attrs[key] = value
    return new_dataset


def _cast_pixels(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        array = np.nan_to_num(array, nan=0.0, posinf=float(info.max), neginf=float(info.min))
        array = np.clip(array, info.min, info.max)
    return np.asarray(array).astype(dtype, copy=False)


def _read_local_frames(source: FrameSource, local_indices: list[int]) -> np.ndarray:
    with h5py.File(source.file_path, "r") as handle:
        dataset = handle[source.dataset_path]
        if dataset.ndim == 2:
            return np.asarray(dataset[...])[np.newaxis, ...]
        # Detector datasets in this project are (N,H,W).  Flattening more
        # leading dimensions keeps the same global-frame semantics as Viewer.
        if dataset.ndim == 3:
            return np.asarray(dataset[local_indices, ...])
        reshaped = np.asarray(dataset[...]).reshape((-1,) + source.frame_shape)
        return reshaped[local_indices]


def _output_dtype(operation: str, source_dtype: np.dtype, requested: str, count: int) -> np.dtype:
    if requested not in ("auto", "source"):
        return np.dtype(requested)
    if requested == "source":
        return np.dtype(source_dtype)
    if operation in ("mean", "exposure-normalized"):
        return np.dtype("float64")
    dtype = np.dtype(source_dtype)
    if np.issubdtype(dtype, np.unsignedinteger):
        return np.dtype("uint64")
    if np.issubdtype(dtype, np.integer):
        return np.dtype("int64")
    return np.dtype("float64") if dtype.itemsize > 4 else np.dtype("float32")


def _copy_external_bundle(master: Path, output_master: Path, sources: list[FrameSource], overwrite: bool) -> dict[Path, Path]:
    if master == output_master:
        raise ProcessingError("input and output master paths must be different")
    output_master.parent.mkdir(parents=True, exist_ok=True)
    if output_master.exists() and not overwrite:
        raise ProcessingError(f"output already exists: {output_master}")
    shutil.copy2(master, output_master)
    copied: dict[Path, Path] = {master: output_master}
    for source in sources:
        if source.file_path == master or source.file_path in copied:
            continue
        try:
            relative = source.file_path.relative_to(master.parent)
            if any(part == ".." for part in relative.parts):
                raise ValueError
        except ValueError:
            relative = Path(source.file_path.name)
        target = (output_master.parent / relative).resolve()
        if target == source.file_path:
            raise ProcessingError("output folder must be separate from the input bundle folder")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            raise ProcessingError(f"output data file already exists: {target}")
        shutil.copy2(source.file_path, target)
        copied[source.file_path] = target

    # Rewrite only links whose absolute target could not be preserved by a
    # relative bundle copy.  Link names and target dataset paths stay the same.
    with h5py.File(output_master, "r+") as handle:
        group = handle.get("/entry/data")
        if isinstance(group, h5py.Group):
            for source in sources:
                if source.file_path == master:
                    continue
                target = copied[source.file_path]
                relative_name = os.path.relpath(target, output_master.parent).replace(os.sep, "/")
                existing = group.get(source.link_name, getlink=True)
                if isinstance(existing, h5py.ExternalLink):
                    if os.fsdecode(existing.filename).replace("\\", "/") != relative_name:
                        del group[source.link_name]
                        group[source.link_name] = h5py.ExternalLink(relative_name, source.dataset_path)
    return copied


def _iter_datasets(handle: h5py.File) -> list[h5py.Dataset]:
    result: list[h5py.Dataset] = []
    handle.visititems(lambda _name, obj: result.append(obj) if isinstance(obj, h5py.Dataset) else None)
    return result


COUNT_NAMES = {"nimages", "nimages_collected", "nimages_written", "nframes", "frame_count", "image_count"}
TIME_NAMES = {"exposure_time", "count_time", "period", "frame_time"}


def _normalized_name(path: str) -> str:
    return path.rsplit("/", 1)[-1].lower().replace("-", "_").replace(" ", "_")


def _set_existing_counts_one(handle: h5py.File) -> None:
    for dataset in _iter_datasets(handle):
        if _normalized_name(dataset.name) not in COUNT_NAMES or dataset.shape != ():
            continue
        try:
            dataset[()] = np.asarray(1, dtype=dataset.dtype)
        except Exception:
            pass


def _selected_side_values(path: str, sources: list[FrameSource], by_source: dict[FrameSource, list[int]]) -> list[np.ndarray]:
    values: list[np.ndarray] = []
    for source in sources:
        local = by_source[source]
        if not local:
            continue
        with h5py.File(source.file_path, "r") as handle:
            obj = handle.get(path)
            if not isinstance(obj, h5py.Dataset) or obj.ndim < 1 or int(obj.shape[0]) != source.frame_count:
                continue
            values.append(np.asarray(obj[local, ...]))
    return values


def _reduce_side_dataset(path: str, values: list[np.ndarray], operation: str, dtype: np.dtype) -> np.ndarray | None:
    if not values:
        return None
    combined = np.concatenate(values, axis=0)
    if combined.shape[0] == 0:
        return None
    if combined.dtype.kind in "biufc":
        name = _normalized_name(path)
        if "/azint/" in path.lower():
            reduced = combined.sum(axis=0, dtype=np.float64) if operation == "sum" else combined.mean(axis=0, dtype=np.float64)
        elif name in TIME_NAMES:
            reduced = combined.sum(axis=0, dtype=np.float64) if operation in ("sum", "exposure-normalized") else combined.mean(axis=0, dtype=np.float64)
        else:
            reduced = combined.mean(axis=0, dtype=np.float64)
        return _cast_pixels(np.asarray(reduced)[np.newaxis, ...], dtype)
    return combined[0:1, ...].astype(dtype, copy=False)


def _reduce_template_file(
    output_file: Path,
    source_file: Path,
    sources: list[FrameSource],
    by_source: dict[FrameSource, list[int]],
    source_frame_count: int,
    operation: str,
    main_dataset_path: str | None,
    main_frame: np.ndarray | None,
) -> None:
    with h5py.File(output_file, "r+") as output, h5py.File(source_file, "r") as template:
        paths = [dataset.name for dataset in _iter_datasets(output)]
        for path in paths:
            dataset = output.get(path)
            source_dataset = template.get(path)
            if not isinstance(dataset, h5py.Dataset) or not isinstance(source_dataset, h5py.Dataset):
                continue
            if main_dataset_path is not None and path == main_dataset_path:
                assert main_frame is not None
                new_dataset = replace_dataset(
                    dataset, main_frame[np.newaxis, ...], dtype=np.dtype(main_frame.dtype)
                )
                for key in ("image_nr_low", "image_nr_high", "nimages"):
                    if key in new_dataset.attrs:
                        new_dataset.attrs.modify(key, np.asarray(1, dtype=np.asarray(new_dataset.attrs[key]).dtype))
                continue
            if source_dataset.ndim < 1 or int(source_dataset.shape[0]) != source_frame_count:
                continue
            values = _selected_side_values(path, sources, by_source)
            reduced = _reduce_side_dataset(path, values, operation, source_dataset.dtype)
            if reduced is None:
                # Path exists only in the representative file; retain the
                # representative selected record instead of inventing data.
                first_local = by_source[sources[0]][0] if by_source[sources[0]] else 0
                reduced = np.asarray(source_dataset[first_local:first_local + 1, ...])
            replace_dataset(dataset, reduced)
        _set_existing_counts_one(output)


def process_sum_mean(args: argparse.Namespace, master: Path, output_master: Path, sources: list[FrameSource], indices: list[int]) -> None:
    by_source = _selection_by_source(sources, indices)
    operation = args.operation
    dtype = _output_dtype(operation, sources[0].dtype, args.out_dtype, len(indices))
    accumulator = np.zeros(sources[0].frame_shape, dtype=np.float64 if operation != "sum" else dtype)
    done = 0
    exposure_total = 0.0
    for source in sources:
        local = by_source[source]
        if not local:
            continue
        block = _read_local_frames(source, local)
        accumulator += block.sum(axis=0, dtype=accumulator.dtype)
        done += len(local)
        percent = 100.0 * done / len(indices)
        print(f"[PROC] {done}/{len(indices)} ({percent:.1f}%)", flush=True)
        for time_path in (
            "/entry/instrument/detector/count_time",
            "/entry/instrument/detector/exposure_time",
            "/entry/detector/count_time",
            "/entry/detector/exposure_time",
        ):
            with h5py.File(source.file_path, "r") as handle:
                obj = handle.get(time_path)
                if isinstance(obj, h5py.Dataset):
                    data = np.asarray(obj[()])
                    if data.shape and data.shape[0] == source.frame_count:
                        exposure_total += float(np.asarray(data[local]).sum())
                        break
                    if data.shape == ():
                        exposure_total += float(data) * len(local)
                        break
    if operation == "mean":
        accumulator = accumulator / float(len(indices))
    elif operation == "exposure-normalized":
        if exposure_total <= 0:
            raise ProcessingError("exposure-normalized output requires positive exposure/count_time metadata")
        accumulator = accumulator / exposure_total
    result = _cast_pixels(accumulator, dtype)

    # Clone the master first, then make a representative derived data file.
    output_master.parent.mkdir(parents=True, exist_ok=True)
    if output_master.exists() and not args.overwrite:
        raise ProcessingError(f"output already exists: {output_master}")
    shutil.copy2(master, output_master)
    first = sources[0]
    prefix = output_master.stem[:-7] if output_master.stem.lower().endswith("_master") else output_master.stem
    output_data = output_master.with_name(f"{prefix}_data_000001.h5")
    if output_data.exists() and not args.overwrite:
        raise ProcessingError(f"output already exists: {output_data}")
    if first.file_path == master:
        # A direct in-master detector dataset needs no separate data file.
        output_data = output_master
    else:
        shutil.copy2(first.file_path, output_data)

    _reduce_template_file(
        output_data,
        first.file_path,
        sources,
        by_source,
        first.frame_count,
        operation,
        first.dataset_path,
        result,
    )

    with h5py.File(output_master, "r+") as output:
        entry_data = output.get("/entry/data")
        if isinstance(entry_data, h5py.Group):
            # Preserve non-frame children.  The frame sequence necessarily has
            # one link after an N->1 operation; retain the original first link
            # name and target dataset path.
            frame_names = {source.link_name for source in sources}
            for name in list(entry_data.keys()):
                if name in frame_names:
                    del entry_data[name]
            if output_data != output_master:
                entry_data[first.link_name] = h5py.ExternalLink(output_data.name, first.dataset_path)
        elif isinstance(entry_data, h5py.Dataset) and first.file_path == master:
            # Already reduced above in output_master.
            pass

        # Master-resident frame-dependent metadata uses the global frame count.
        paths = [dataset.name for dataset in _iter_datasets(output)]
        for path in paths:
            if path == "/entry/data" or path.startswith("/entry/data/"):
                continue
            dataset = output.get(path)
            if not isinstance(dataset, h5py.Dataset) or dataset.ndim < 1 or int(dataset.shape[0]) != sum(s.frame_count for s in sources):
                continue
            original_values = np.asarray(dataset[indices, ...])
            reduced = _reduce_side_dataset(path, [original_values], operation, dataset.dtype)
            if reduced is not None:
                replace_dataset(dataset, reduced)
        _set_existing_counts_one(output)

        # Existing scalar acquisition-time fields must describe the result.
        for path in (
            "/entry/instrument/detector/count_time",
            "/entry/instrument/detector/exposure_time",
            "/entry/detector/count_time",
            "/entry/detector/exposure_time",
        ):
            obj = output.get(path)
            if isinstance(obj, h5py.Dataset) and obj.shape == () and exposure_total > 0:
                value = exposure_total if operation in ("sum", "exposure-normalized") else exposure_total / len(indices)
                try:
                    obj[()] = np.asarray(value, dtype=obj.dtype)
                except Exception:
                    pass

    verify_sum_tree(master, output_master, first.file_path, output_data, {source.link_name for source in sources})
    verify_one_frame(output_master)
    print(f"[OUT DATA] {output_data}", flush=True)
    print(f"[OUT HDF5] {output_master}", flush=True)


def _map_output_source(source: FrameSource, copied: dict[Path, Path], output_master: Path) -> FrameSource:
    return FrameSource(
        source.link_name,
        copied.get(source.file_path, output_master),
        source.dataset_path,
        source.frame_count,
        source.frame_shape,
        source.dtype,
        source.global_start,
    )


def _roi_slices(text: str | None, shape: tuple[int, int]) -> tuple[slice, slice]:
    if not text:
        return slice(0, shape[0]), slice(0, shape[1])
    try:
        x1, y1, x2, y2 = (int(item.strip()) for item in text.split(","))
    except Exception as exc:
        raise ProcessingError("ROI must be x1,y1,x2,y2 (0-based inclusive)") from exc
    height, width = shape
    if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
        raise ProcessingError(f"ROI is outside image bounds 0..{width-1},0..{height-1}")
    return slice(y1, y2 + 1), slice(x1, x2 + 1)


def _background_frame(master: Path, index_one_based: int) -> np.ndarray:
    sources = discover_frame_sources(master)
    index = int(index_one_based) - 1
    if index < 0 or index >= sum(item.frame_count for item in sources):
        raise ProcessingError("background frame is outside its file")
    by_source = _selection_by_source(sources, [index])
    for source in sources:
        if by_source[source]:
            return _read_local_frames(source, by_source[source])[0].astype(np.float64)
    raise ProcessingError("background frame could not be read")


def process_patch(args: argparse.Namespace, master: Path, output_master: Path, sources: list[FrameSource], indices: list[int]) -> None:
    copied = _copy_external_bundle(master, output_master, sources, args.overwrite)
    output_sources = [_map_output_source(source, copied, output_master) for source in sources]
    selected = set(indices)
    roi_y, roi_x = _roi_slices(args.roi, sources[0].frame_shape)
    protect_y, protect_x = _roi_slices(args.protect, sources[0].frame_shape) if args.protect else (None, None)
    background = _background_frame(_as_path(args.background), args.background_frame) if args.background else None

    done = 0
    for source in output_sources:
        local_indices = [i for i in range(source.frame_count) if source.global_start + i in selected]
        if not local_indices:
            continue
        with h5py.File(source.file_path, "r+") as handle:
            dataset = handle[source.dataset_path]
            for local_index in local_indices:
                selection = (local_index, slice(None), slice(None)) if dataset.ndim == 3 else (...,)
                frame = np.asarray(dataset[selection]).astype(np.float64)
                original = frame.copy() if protect_y is not None else None
                target = frame[roi_y, roi_x]
                if args.mode == "pixelop":
                    if args.pixel_operation == "add":
                        target += args.value
                    elif args.pixel_operation == "subtract":
                        target -= args.value
                    elif args.pixel_operation == "multiply":
                        target *= args.value
                    elif args.pixel_operation == "divide":
                        if args.value == 0:
                            raise ProcessingError("divide value must be non-zero")
                        target /= args.value
                else:
                    if background is not None:
                        target -= background[roi_y, roi_x]
                    if args.lower_threshold is not None:
                        target[target < args.lower_threshold] = args.fill
                    if args.remove_low is not None and args.remove_high is not None:
                        mask = (target >= args.remove_low) & (target <= args.remove_high)
                        target[mask] = args.fill
                    if args.upper_threshold is not None:
                        target[target > args.upper_threshold] = args.fill
                if protect_y is not None and protect_x is not None and original is not None:
                    frame[protect_y, protect_x] = original[protect_y, protect_x]
                dataset[selection] = _cast_pixels(frame, dataset.dtype)
                done += 1
                print(f"[PROC] {done}/{len(indices)} ({100.0 * done / len(indices):.1f}%)", flush=True)
            handle.flush()
    verify_same_tree(master, output_master, sources, copied, selected)
    print(f"[OUT HDF5] {output_master}", flush=True)


def _link_signature(handle: h5py.File) -> dict[str, tuple[str, str, str]]:
    result: dict[str, tuple[str, str, str]] = {}
    def walk(group: h5py.Group, base: str) -> None:
        for name in group.keys():
            path = f"{base}/{name}".replace("//", "/")
            link = group.get(name, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                result[path] = ("external", "*", str(link.path))
            elif isinstance(link, h5py.SoftLink):
                result[path] = ("soft", "", str(link.path))
            else:
                obj = group.get(name)
                result[path] = ("group" if isinstance(obj, h5py.Group) else "dataset", "", "")
                if isinstance(obj, h5py.Group):
                    walk(obj, path)
    walk(handle["/"], "")
    return result


def _attrs_equal(left: Any, right: Any) -> bool:
    if set(left.attrs.keys()) != set(right.attrs.keys()):
        return False
    for key in left.attrs.keys():
        try:
            if not np.array_equal(np.asarray(left.attrs[key]), np.asarray(right.attrs[key])):
                return False
        except Exception:
            if repr(left.attrs[key]) != repr(right.attrs[key]):
                return False
    return True


def _dataset_hash(dataset: h5py.Dataset) -> str:
    digest = hashlib.sha256()
    digest.update(str(dataset.shape).encode("ascii"))
    digest.update(np.dtype(dataset.dtype).str.encode("ascii"))
    if dataset.shape == ():
        digest.update(np.ascontiguousarray(np.asarray(dataset[()])).tobytes())
        return digest.hexdigest()
    if dataset.ndim >= 1 and dataset.shape[0] > 1:
        for start in range(0, int(dataset.shape[0]), 16):
            digest.update(np.ascontiguousarray(dataset[start:start + 16, ...]).tobytes())
    else:
        digest.update(np.ascontiguousarray(dataset[...]).tobytes())
    return digest.hexdigest()


def _storage_signature(dataset: h5py.Dataset) -> tuple[Any, ...]:
    return (
        tuple(dataset.shape), np.dtype(dataset.dtype).str,
        dataset.chunks, dataset.compression, dataset.compression_opts,
        bool(dataset.shuffle), bool(dataset.fletcher32), dataset.scaleoffset,
    )


def verify_same_tree(
    master: Path,
    output_master: Path,
    sources: list[FrameSource],
    copied: dict[Path, Path],
    selected_global: set[int],
) -> None:
    pairs = [(master, output_master)] + [(source.file_path, copied[source.file_path]) for source in sources if source.file_path != master and source.file_path in copied]
    checked: set[tuple[Path, Path]] = set()
    for original, output in pairs:
        if (original, output) in checked:
            continue
        checked.add((original, output))
        with h5py.File(original, "r") as left, h5py.File(output, "r") as right:
            if _link_signature(left) != _link_signature(right):
                raise ProcessingError(f"tree verification failed for {output.name}")
            left_paths = {dataset.name for dataset in _iter_datasets(left)}
            right_paths = {dataset.name for dataset in _iter_datasets(right)}
            if left_paths != right_paths:
                raise ProcessingError(f"dataset-path verification failed for {output.name}")
            frame_sources = [source for source in sources if source.file_path == original]
            frame_paths = {source.dataset_path: source for source in frame_sources}
            for path in sorted(left_paths):
                lds, rds = left[path], right[path]
                if _storage_signature(lds) != _storage_signature(rds):
                    raise ProcessingError(f"dataset layout changed unexpectedly: {output.name}:{path}")
                if not _attrs_equal(lds, rds):
                    raise ProcessingError(f"dataset attributes changed unexpectedly: {output.name}:{path}")
                source = frame_paths.get(path)
                if source is None:
                    if _dataset_hash(lds) != _dataset_hash(rds):
                        raise ProcessingError(f"non-frame dataset content changed: {output.name}:{path}")
                    continue
                for local in range(source.frame_count):
                    if source.global_start + local in selected_global:
                        continue
                    left_frame = lds[local, ...] if lds.ndim >= 3 else lds[...]
                    right_frame = rds[local, ...] if rds.ndim >= 3 else rds[...]
                    if not np.array_equal(left_frame, right_frame):
                        raise ProcessingError(f"unselected frame changed: {output.name}:{path}[{local}]")
    print("[VERIFY] PASS: tree/layout/attributes/non-frame data and unselected frames are preserved", flush=True)


def verify_sum_tree(
    input_master: Path,
    output_master: Path,
    template_data: Path,
    output_data: Path,
    frame_link_names: set[str],
) -> None:
    with h5py.File(input_master, "r") as left, h5py.File(output_master, "r") as right:
        left_sig = _link_signature(left)
        right_sig = _link_signature(right)
        for name in frame_link_names:
            left_sig.pop(f"/entry/data/{name}", None)
            right_sig.pop(f"/entry/data/{name}", None)
        if left_sig != right_sig:
            raise ProcessingError("derived master no longer follows the input master tree")
    if template_data != input_master:
        with h5py.File(template_data, "r") as left, h5py.File(output_data, "r") as right:
            if set(_link_signature(left)) != set(_link_signature(right)):
                raise ProcessingError("derived data file no longer follows the source data tree")
    print("[VERIFY] PASS: derived master/data hierarchy follows the source HDF5 tree", flush=True)


def verify_one_frame(master: Path) -> None:
    sources = discover_frame_sources(master)
    total = sum(item.frame_count for item in sources)
    if total != 1:
        raise ProcessingError(f"derived HDF5 should expose 1 frame, found {total}")
    source = sources[0]
    _read_local_frames(source, [0])
    print("[VERIFY] PASS: derived HDF5 exposes one readable detector frame", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process detector HDF5 while preserving its source structure model")
    parser.add_argument("-i", "--input", required=True, type=Path, help="input master HDF5")
    parser.add_argument("-o", "--output", required=True, type=Path, help="output master HDF5")
    parser.add_argument("--mode", choices=("sum", "mean", "exposure-normalized", "pixelop", "filter"), required=True)
    parser.add_argument("--frame-from", type=int)
    parser.add_argument("--frame-to", type=int)
    parser.add_argument("--select-frame")
    parser.add_argument("--out-dtype", default="auto", choices=("auto", "source", "int16", "int32", "int64", "uint16", "uint32", "uint64", "float32", "float64"))
    parser.add_argument("--pixel-operation", choices=("add", "subtract", "multiply", "divide"))
    parser.add_argument("--value", type=float, default=0.0)
    parser.add_argument("--roi")
    parser.add_argument("--background")
    parser.add_argument("--background-frame", type=int, default=1)
    parser.add_argument("--lower-threshold", type=float)
    parser.add_argument("--remove-low", type=float)
    parser.add_argument("--remove-high", type=float)
    parser.add_argument("--upper-threshold", type=float)
    parser.add_argument("--fill", type=float, default=0.0)
    parser.add_argument("--protect")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    master = _as_path(args.input)
    output_master = _as_path(args.output)
    if not master.is_file() or not h5py.is_hdf5(master):
        raise ProcessingError(f"input is not an HDF5 file: {master}")
    sources = discover_frame_sources(master)
    total = sum(item.frame_count for item in sources)
    indices = parse_selection(total, args.select_frame, args.frame_from, args.frame_to)
    print(f"[START] HDF5 {args.mode}: {master}", flush=True)
    print(f"[INPUT] {len(sources)} frame datasets, {total} total frames", flush=True)
    print(f"[SELECT] {len(indices)} frames", flush=True)
    if args.mode in ("sum", "mean", "exposure-normalized"):
        args.operation = args.mode
        process_sum_mean(args, master, output_master, sources, indices)
    else:
        if args.mode == "pixelop" and not args.pixel_operation:
            raise ProcessingError("--pixel-operation is required for pixelop mode")
        process_patch(args, master, output_master, sources, indices)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProcessingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
