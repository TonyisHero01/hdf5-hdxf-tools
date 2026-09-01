#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process detector frames directly inside HDXF archives.

This is the HDXF counterpart of the original HDF5 ``subtract.py`` workflow.
It keeps the original command-line operations where they still make sense:

* lower threshold
* fixed-frame background subtraction
* inclusive value-range removal
* upper threshold
* protected ROI plus outside-ROI high-value rejection
* global frame selection/ranges
* single-file and directory batch operation

The output is a new self-contained HDXF archive.  Non-frame payloads and source
HDF5 metadata are copied from the input archive, while selected detector frames
are decoded, transformed, re-encoded as HDXFB blocks, and described by a fresh
JSON frame index in ``manifest.json``.

Required sibling modules from the HDXF project:

* hdxf_codec.py
* hdxf_index.py
* hdf5_to_hdxf.py

``hdf5_to_hdxf.py`` supplies the canonical frame encoder, so this utility stays
compatible with the project's current HDXFB version and compression policy.
"""
from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import inspect
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import tracemalloc
import zipfile
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

try:
    from hdxf_codec import HDXFBError, decode_hdxfb
    from hdxf_index import HDXFIError, unpack_frame_index
except Exception as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "HDXF project modules are required. Put hdxf_subtract.py next to "
        "hdxf_codec.py and hdxf_index.py.\n"
        f"Import error: {type(exc).__name__}: {exc}"
    ) from exc

try:
    import hdf5_to_hdxf as _hdxf_encoder
except Exception as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "hdf5_to_hdxf.py is required next to hdxf_subtract.py because it "
        "provides the canonical HDXFB frame encoder.\n"
        f"Import error: {type(exc).__name__}: {exc}"
    ) from exc


MANIFEST_PATH = "manifest.json"
FORMAT_NAME = "HDF5-derived Detector eXchange Format"
FORMAT_ID = "hdxf"
SUPPORTED_FORMAT_VERSIONS = {
    "0.3.0", "0.4.0", "0.4.2", "0.4.3", "0.4.4", "0.4.5",
}
SCRIPT_VERSION = "1.1.0-hdxf-stream-verify"
DEBUG = False


class HDXFProcessError(RuntimeError):
    """Fatal input, transform, encoding, or output error."""


def dprint(*args: Any, **kwargs: Any) -> None:
    if DEBUG:
        print(*args, **kwargs, flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json_dumps(value: Any, *, indent: int | None = None) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
        separators=(",", ":") if indent is None else None,
    ).encode("utf-8")


def npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _get_rss_mb() -> float | None:
    if psutil is None:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def print_mem(tag: str, enabled: bool) -> None:
    if not enabled:
        return
    rss = _get_rss_mb()
    if rss is not None:
        print(f"[MEM] {tag}: RSS={rss:.1f} MB", flush=True)
        return
    current, peak = tracemalloc.get_traced_memory()
    print(
        f"[MEM] {tag}: PythonHeap current={current / 1024**2:.1f} MB, "
        f"peak={peak / 1024**2:.1f} MB (install psutil for RSS)",
        flush=True,
    )


def _decode_tagged(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_tagged(item) for item in value]
    if not isinstance(value, dict):
        return value
    tag = value.get("$type")
    if tag == "bytes":
        import base64
        return base64.b64decode(str(value.get("base64", "")))
    if tag == "float":
        token = str(value.get("value", "nan"))
        return {
            "nan": float("nan"),
            "+inf": float("inf"),
            "-inf": float("-inf"),
        }.get(token, float("nan"))
    if tag == "ndarray_npy":
        import base64
        return np.load(
            io.BytesIO(base64.b64decode(str(value.get("base64", "")))),
            allow_pickle=False,
        )
    return {str(k): _decode_tagged(v) for k, v in value.items()}


def _dtype_from_descriptor(value: Any) -> np.dtype[Any]:
    value = _decode_tagged(value)
    if isinstance(value, np.dtype):
        return value
    if isinstance(value, str):
        return np.dtype(value)
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        fields: list[Any] = []
        for item in value:
            if isinstance(item, tuple):
                item = list(item)
            if not isinstance(item, list):
                return np.dtype(value)
            if len(item) == 2:
                fields.append((item[0], _dtype_from_descriptor(item[1])))
            elif len(item) == 3:
                fields.append(
                    (item[0], _dtype_from_descriptor(item[1]), tuple(item[2]))
                )
            else:
                raise HDXFProcessError(f"Invalid dtype field: {item!r}")
        return np.dtype(fields)
    raise HDXFProcessError(f"Unsupported dtype descriptor: {value!r}")


def _dtype_from_info(info: Any) -> np.dtype[Any]:
    if isinstance(info, dict) and "numpy" in info:
        return _dtype_from_descriptor(info["numpy"])
    return _dtype_from_descriptor(info)


def _normalise_json(value: Any) -> Any:
    """Convert NumPy values and tuples into strict JSON-compatible values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_normalise_json(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return [_normalise_json(item) for item in value]
    if isinstance(value, list):
        return [_normalise_json(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _normalise_json(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        # Encoder metadata should not contain non-finite values.  Preserve
        # strict JSON rather than emitting implementation-specific NaN tokens.
        return str(value)
    return value


@dataclass(frozen=True)
class FrameDatasetRecord:
    object_id: str
    source_path: str
    global_start: int
    frame_count: int

    @property
    def global_stop(self) -> int:
        return self.global_start + self.frame_count


class HDXFArchiveReader:
    """Bounded-memory sequential/random reader for HDXF detector frames."""

    def __init__(self, path: Path, *, block_cache_size: int = 4) -> None:
        self.path = path.resolve()
        try:
            self.zip = zipfile.ZipFile(self.path, "r")
        except Exception as exc:
            raise HDXFProcessError(f"Cannot open HDXF archive {self.path}: {exc}") from exc

        try:
            self.manifest = json.loads(self.zip.read(MANIFEST_PATH))
        except Exception as exc:
            self.close()
            raise HDXFProcessError(f"Cannot read manifest.json: {exc}") from exc

        identity = self.manifest.get("hdxf", {})
        if identity.get("format_id") != FORMAT_ID or identity.get("format") != FORMAT_NAME:
            self.close()
            raise HDXFProcessError(f"Not an HDXF detector archive: {self.path}")
        version = str(identity.get("version", ""))
        if version not in SUPPORTED_FORMAT_VERSIONS:
            self.close()
            raise HDXFProcessError(f"Unsupported HDXF version {version!r}")

        detector = self.manifest.get("detector_archive")
        frames = detector.get("frames") if isinstance(detector, dict) else None
        if not isinstance(frames, dict):
            self.close()
            raise HDXFProcessError("detector_archive.frames is missing")
        self.frames_doc = frames

        blocks = frames.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            index_doc = frames.get("block_index")
            if not isinstance(index_doc, dict):
                self.close()
                raise HDXFProcessError("Frame blocks and block_index are both missing")
            raw = self._read_member(index_doc)
            try:
                blocks = unpack_frame_index(
                    raw,
                    datasets=list(frames.get("datasets", [])),
                    frame_shape=list(frames.get("frame_shape", [])),
                )
            except HDXFIError as exc:
                self.close()
                raise HDXFProcessError(f"Invalid HDXF frame index: {exc}") from exc

        self.blocks = sorted(
            [dict(item) for item in blocks if isinstance(item, dict)],
            key=lambda item: int(item.get("global_start_frame", 0)),
        )
        self.frame_count = int(frames.get("frame_count", 0))
        self.frame_shape = tuple(int(x) for x in frames.get("frame_shape", []))
        self.dtype = _dtype_from_descriptor(frames.get("dtype"))
        if self.frame_count <= 0 or len(self.frame_shape) != 2:
            self.close()
            raise HDXFProcessError(
                f"Invalid detector frame geometry: count={self.frame_count}, "
                f"shape={self.frame_shape}"
            )

        self.datasets: list[FrameDatasetRecord] = []
        for item in frames.get("datasets", []):
            if not isinstance(item, dict) or not item.get("object"):
                continue
            self.datasets.append(
                FrameDatasetRecord(
                    object_id=str(item["object"]),
                    source_path=str(item.get("source_path") or ""),
                    global_start=int(item.get("global_start_frame", 0)),
                    frame_count=int(item.get("frame_count", 0)),
                )
            )
        self.datasets.sort(key=lambda item: item.global_start)
        if not self.datasets:
            self.close()
            raise HDXFProcessError("No detector frame dataset records were found")
        self._dataset_starts = [item.global_start for item in self.datasets]

        self._block_starts = [int(item.get("global_start_frame", 0)) for item in self.blocks]
        self._path_to_position = {
            str(item.get("path")): index for index, item in enumerate(self.blocks)
        }
        self._start_to_position = {
            int(item.get("global_start_frame", 0)): index
            for index, item in enumerate(self.blocks)
        }
        self._cache_size = max(1, int(block_cache_size))
        self._block_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._static_cache: dict[str, dict[str, np.ndarray]] = {}
        self._static_docs = {
            str(item.get("id")): item
            for item in frames.get("static_models", [])
            if isinstance(item, dict) and item.get("id")
        }

    def close(self) -> None:
        self._block_cache.clear()
        self._static_cache.clear()
        if getattr(self, "zip", None) is not None:
            self.zip.close()

    def __enter__(self) -> "HDXFArchiveReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _read_member(self, doc: dict[str, Any]) -> bytes:
        path = str(doc.get("path") or "")
        try:
            raw = self.zip.read(path)
        except KeyError as exc:
            raise HDXFProcessError(f"Missing archive member: {path}") from exc
        size = doc.get("size_bytes")
        if isinstance(size, int) and len(raw) != size:
            raise HDXFProcessError(f"Archive member size mismatch: {path}")
        digest = doc.get("sha256")
        if isinstance(digest, str) and sha256_bytes(raw) != digest:
            raise HDXFProcessError(f"Archive member SHA-256 mismatch: {path}")
        return raw

    def _load_static_model(self, model_id: str | None) -> dict[str, np.ndarray] | None:
        if not model_id:
            return None
        key = str(model_id)
        cached = self._static_cache.get(key)
        if cached is not None:
            return cached
        doc = self._static_docs.get(key)
        if not isinstance(doc, dict):
            raise HDXFProcessError(f"Static model is missing: {key}")
        mask_doc = doc.get("mask")
        values_doc = doc.get("static_values")
        if not isinstance(mask_doc, dict) or not isinstance(values_doc, dict):
            raise HDXFProcessError(f"Static model is incomplete: {key}")
        try:
            packed, _ = decode_hdxfb(
                self._read_member(mask_doc),
                member_path=str(mask_doc.get("path")),
                verify_hash=True,
            )
            values, _ = decode_hdxfb(
                self._read_member(values_doc),
                member_path=str(values_doc.get("path")),
                verify_hash=True,
            )
        except HDXFBError as exc:
            raise HDXFProcessError(f"Cannot decode static model {key}: {exc}") from exc
        pixel_count = int(doc.get("pixel_count", np.prod(self.frame_shape)))
        bits = np.unpackbits(np.asarray(packed, dtype=np.uint8).reshape(-1), bitorder="little")
        if bits.size < pixel_count:
            raise HDXFProcessError(f"Static model mask is truncated: {key}")
        mask = bits[:pixel_count].astype(bool).reshape(self.frame_shape)
        values = np.ascontiguousarray(values).reshape(-1)
        if values.size != int(np.count_nonzero(mask)):
            raise HDXFProcessError(f"Static model value count mismatch: {key}")
        result = {"mask": mask, "static_values": values}
        self._static_cache[key] = result
        return result

    def _decode_block(
        self,
        doc: dict[str, Any],
        previous_frame: np.ndarray | None,
    ) -> np.ndarray:
        raw = self._read_member(doc)
        try:
            block, header = decode_hdxfb(
                raw,
                member_path=str(doc.get("path")),
                verify_hash=True,
                previous_frame=previous_frame,
                static_model=self._load_static_model(doc.get("static_model_id")),
            )
        except HDXFBError as exc:
            raise HDXFProcessError(
                f"Cannot decode frame block {doc.get('path')}: {exc}"
            ) from exc
        if header.get("kind") != "frame-block":
            raise HDXFProcessError(
                f"Unexpected HDXFB kind in {doc.get('path')}: {header.get('kind')!r}"
            )
        return np.ascontiguousarray(block)

    def dataset_for_global_index(self, frame_index: int) -> FrameDatasetRecord:
        pos = bisect.bisect_right(self._dataset_starts, frame_index) - 1
        if pos < 0:
            raise HDXFProcessError(f"No frame dataset covers index {frame_index}")
        record = self.datasets[pos]
        if not (record.global_start <= frame_index < record.global_stop):
            raise HDXFProcessError(f"No frame dataset covers index {frame_index}")
        return record

    def get_frame(self, frame_index: int) -> np.ndarray:
        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError(frame_index)
        pos = bisect.bisect_right(self._block_starts, frame_index) - 1
        if pos < 0:
            raise HDXFProcessError(f"No frame block covers index {frame_index}")
        target_doc = self.blocks[pos]
        target_path = str(target_doc.get("path"))
        cached = self._block_cache.get(target_path)
        if cached is None:
            target_pos = self._path_to_position[target_path]
            if bool(target_doc.get("requires_previous_frame", False)):
                anchor = int(target_doc.get("chain_anchor_global_start_frame", -1))
                anchor_pos = self._start_to_position.get(anchor)
                if anchor_pos is None or anchor_pos > target_pos:
                    raise HDXFProcessError(f"Invalid Delta chain anchor for {target_path}")
            else:
                anchor_pos = target_pos
            previous: np.ndarray | None = None
            block: np.ndarray | None = None
            for index in range(anchor_pos, target_pos + 1):
                doc = self.blocks[index]
                path = str(doc.get("path"))
                part = self._block_cache.get(path)
                if part is None:
                    part = self._decode_block(doc, previous)
                    self._block_cache[path] = part
                    self._block_cache.move_to_end(path)
                    while len(self._block_cache) > self._cache_size:
                        self._block_cache.popitem(last=False)
                previous = np.ascontiguousarray(part[-1])
                block = part
            assert block is not None
            cached = block
        start = int(target_doc.get("global_start_frame", 0))
        local = frame_index - start
        return np.ascontiguousarray(cached[local])

    def iter_frame_blocks(
        self,
    ) -> Iterator[tuple[int, str, np.ndarray]]:
        """Sequentially decode each detector block exactly once.

        Delta chains are followed in archive order.  This method intentionally
        bypasses the random-access block cache, because sequential consumers
        (processing and verification) already retain the previous frame needed
        by chained Delta blocks.
        """
        previous_frame: np.ndarray | None = None
        previous_source: str | None = None
        for doc in self.blocks:
            block_start = int(doc.get("global_start_frame", 0))
            source_object = str(doc.get("source_object") or "")
            requires_previous = bool(doc.get("requires_previous_frame", False))
            if source_object != previous_source or not requires_previous:
                previous_frame = None
            block = self._decode_block(doc, previous_frame)
            previous_frame = np.ascontiguousarray(block[-1])
            previous_source = source_object
            yield block_start, source_object, block

    def iter_selected_blocks(
        self,
        selected_indices: Sequence[int],
    ) -> Iterator[tuple[list[int], str, np.ndarray]]:
        """Yield selected frames grouped by their already-decoded source block.

        Every physical HDXFB frame block is decoded at most once.  The returned
        index list contains the original zero-based global frame numbers for the
        rows in the returned ndarray.  This is the high-throughput path used by
        Full Verify so transform_block() can operate on several frames at once.
        """
        selected = sorted(set(int(index) for index in selected_indices))
        if not selected:
            return
        if selected[0] < 0 or selected[-1] >= self.frame_count:
            raise HDXFProcessError(
                f"Selected frame range is outside 1..{self.frame_count}"
            )

        pointer = 0
        for block_start, source_object, block in self.iter_frame_blocks():
            if pointer >= len(selected):
                break
            block_stop = block_start + int(block.shape[0])
            while pointer < len(selected) and selected[pointer] < block_start:
                raise HDXFProcessError(
                    f"Selected frame {selected[pointer] + 1} is not covered by the frame index"
                )

            local_indices: list[int] = []
            global_indices: list[int] = []
            while pointer < len(selected) and selected[pointer] < block_stop:
                global_index = selected[pointer]
                local = global_index - block_start
                if local < 0 or local >= block.shape[0]:
                    raise HDXFProcessError(
                        f"Frame index mismatch at global frame {global_index + 1}"
                    )
                local_indices.append(local)
                global_indices.append(global_index)
                pointer += 1

            if local_indices:
                rows = np.asarray(local_indices, dtype=np.intp)
                yield (
                    global_indices,
                    source_object,
                    np.ascontiguousarray(block[rows]),
                )

        if pointer != len(selected):
            raise HDXFProcessError(
                f"Only {pointer} of {len(selected)} selected frames were decoded"
            )

    def iter_selected_frames(
        self,
        selected_indices: Sequence[int],
    ) -> Iterator[tuple[int, str, np.ndarray]]:
        """Decode every source block at most once and yield selected frames."""
        for global_indices, source_object, block in self.iter_selected_blocks(
            selected_indices
        ):
            for row, global_index in enumerate(global_indices):
                yield (
                    global_index,
                    source_object,
                    np.ascontiguousarray(block[row]),
                )


@dataclass
class TransformConfig:
    mode: str
    threshold: float | None
    background: np.ndarray | None
    remove_low: float | None
    remove_high: float | None
    upper_threshold: float | None
    fill: float
    protect_roi: tuple[int, int, int, int] | None


def _cast_fill(dtype: np.dtype[Any], value: float) -> Any:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        value = float(np.clip(value, info.min, info.max))
    return np.asarray(value, dtype=dtype).item()


def transform_block(block: np.ndarray, config: TransformConfig) -> np.ndarray:
    """Apply the original subtract.py operations to one (N,H,W) block."""
    source = np.ascontiguousarray(block)
    result = source.copy()
    dtype = source.dtype
    fill_value = _cast_fill(dtype, config.fill)

    roi_max: np.ndarray | None = None
    roi_patch: np.ndarray | None = None
    if config.protect_roi is not None:
        x0, y0, x1, y1 = config.protect_roi
        roi_patch = source[:, y0:y1, x0:x1].copy()
        roi_max = source[:, y0:y1, x0:x1].max(axis=(1, 2))

    if config.mode == "threshold":
        assert config.threshold is not None
        result[result < config.threshold] = fill_value
    elif config.mode == "background":
        assert config.background is not None
        if np.issubdtype(dtype, np.integer):
            work = result.astype(np.int64, copy=False)
            work -= config.background.astype(np.int64, copy=False)
            info = np.iinfo(dtype)
            np.clip(work, info.min, info.max, out=work)
            result = work.astype(dtype, copy=False)
        else:
            work = result.astype(np.float64, copy=False)
            work -= config.background.astype(np.float64, copy=False)
            result = work.astype(dtype, copy=False)
    elif config.mode == "remove":
        assert config.remove_low is not None and config.remove_high is not None
        mask = (result >= config.remove_low) & (result <= config.remove_high)
        result[mask] = fill_value

    if config.upper_threshold is not None:
        result[result > config.upper_threshold] = fill_value

    if config.protect_roi is not None:
        assert roi_max is not None and roi_patch is not None
        # Preserve the historical behaviour: outside the protected ROI,
        # values >= the original ROI maximum are forced to literal zero.
        result[result >= roi_max[:, None, None]] = _cast_fill(dtype, 0.0)
        x0, y0, x1, y1 = config.protect_roi
        result[:, y0:y1, x0:x1] = roi_patch

    return np.ascontiguousarray(result)


def _encoder_function():
    function = getattr(_hdxf_encoder, "encode_frame_block_hdxfb", None)
    if not callable(function):
        raise HDXFProcessError(
            "hdf5_to_hdxf.py does not expose encode_frame_block_hdxfb(). "
            "Use the converter version that belongs to this HDXF project."
        )
    return function


def encode_output_block(
    block: np.ndarray,
    *,
    args: argparse.Namespace,
    previous_frame: np.ndarray | None,
    learned_delta_filter: str | None,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    """Call the installed converter's encoder across HDXF tool versions."""
    function = _encoder_function()
    parameters = inspect.signature(function).parameters
    kwargs: dict[str, Any] = {}

    values: dict[str, Any] = {
        "transform": args.frame_transform,
        "clevel": args.zstd_level,
        "tile_height": args.tile_height,
        "tile_width": args.tile_width,
        "sparse_threshold": args.sparse_threshold,
        "enable_jungfrau_split": not args.no_jungfrau_split,
        "delta_compute_backend": args.compute_backend,
        "cuda_device": args.cuda_device,
        "delta_filter": (
            learned_delta_filter
            if learned_delta_filter is not None
            else ("auto" if args.delta_filter == "learn" else args.delta_filter)
        ),
        "previous_frame": previous_frame,
        "static_mask": None,
        "static_reference": None,
        "shared_static_model_id": None,
        "delta_stream": args.delta_stream,
        "zero_rle_threshold": args.zero_rle_threshold,
    }
    for name, value in values.items():
        if name in parameters:
            kwargs[name] = value

    try:
        result = function(block, **kwargs)
    except TypeError as exc:
        raise HDXFProcessError(
            "The installed hdf5_to_hdxf.py encoder API is incompatible with "
            f"subtract_hdxf.py: {exc}"
        ) from exc
    except Exception as exc:
        raise HDXFProcessError(
            f"HDXFB frame encoding failed: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(result, tuple) or len(result) < 3:
        raise HDXFProcessError(
            "encode_frame_block_hdxfb() returned an unexpected result"
        )
    data = bytes(result[0])
    header = dict(result[1])
    candidate_sizes = dict(result[2]) if isinstance(result[2], dict) else {}
    return data, header, _normalise_json(candidate_sizes)


def _frame_encoding_name(version: int) -> str:
    if version >= 4:
        return "frame-block-v4"
    if version == 3:
        return "frame-block-v3"
    if version == 2:
        return "frame-block-v2"
    return "frame-block-v1"


def _old_frame_payload_paths(reader: HDXFArchiveReader) -> set[str]:
    paths = {
        str(item.get("path"))
        for item in reader.blocks
        if isinstance(item, dict) and item.get("path")
    }
    index_doc = reader.frames_doc.get("block_index")
    if isinstance(index_doc, dict) and index_doc.get("path"):
        paths.add(str(index_doc["path"]))
    for model in reader.frames_doc.get("static_models", []):
        if not isinstance(model, dict):
            continue
        for key in ("mask", "static_values"):
            doc = model.get(key)
            if isinstance(doc, dict) and doc.get("path"):
                paths.add(str(doc["path"]))
    return paths


def _object_source_path(obj: dict[str, Any]) -> str:
    return str(obj.get("source_path") or "")


def _frame_object_source_info(
    manifest: dict[str, Any],
    object_ids: Iterable[str],
) -> dict[str, tuple[str, str]]:
    objects = manifest.get("objects", {})
    result: dict[str, tuple[str, str]] = {}
    if not isinstance(objects, dict):
        return result
    for object_id in object_ids:
        obj = objects.get(object_id)
        if isinstance(obj, dict):
            result[object_id] = (
                str(obj.get("source_file") or ""),
                str(obj.get("source_path") or ""),
            )
    return result


def _update_frame_storage(
    obj: dict[str, Any],
    *,
    frame_count: int,
    frame_shape: tuple[int, int],
    global_low: int,
    global_high: int,
) -> None:
    storage = obj.get("hdf5_storage")
    if isinstance(storage, dict):
        storage["shape"] = [int(frame_count), int(frame_shape[0]), int(frame_shape[1])]
        maxshape = storage.get("maxshape")
        if isinstance(maxshape, list) and len(maxshape) >= 1:
            maxshape[0] = None if maxshape[0] is None else int(frame_count)
        chunks = storage.get("chunks")
        if isinstance(chunks, list) and len(chunks) >= 1:
            chunks[0] = max(1, min(int(chunks[0]), int(frame_count)))
    attrs = obj.setdefault("attributes", {})
    if isinstance(attrs, dict):
        attrs["image_nr_low"] = int(global_low)
        attrs["image_nr_high"] = int(global_high)
        attrs["nimages"] = int(frame_count)
    obj["payload"] = {
        "encoding": "frame-sequence",
        "state": "present",
        "members": [],
    }


def _replace_scalar_dataset_payload(
    obj: dict[str, Any],
    value: int,
) -> tuple[str, bytes, set[str]] | None:
    storage = obj.get("hdf5_storage")
    if not isinstance(storage, dict):
        return None
    shape_doc = storage.get("shape")
    if shape_doc is None:
        return None
    shape = tuple(int(x) for x in shape_doc)
    if shape != () and int(np.prod(shape, dtype=np.int64)) != 1:
        return None
    try:
        dtype = _dtype_from_info(storage.get("dtype"))
    except Exception:
        return None
    array = np.asarray(value, dtype=dtype)
    if shape != ():
        array = np.asarray([value], dtype=dtype).reshape(shape)
    payload_bytes = npy_bytes(array)
    digest = sha256_bytes(payload_bytes)
    member_path = f"payloads/{digest}.npy"
    old_paths: set[str] = set()
    old_payload = obj.get("payload")
    if isinstance(old_payload, dict):
        for member in old_payload.get("members", []):
            if isinstance(member, dict) and member.get("path"):
                old_paths.add(str(member["path"]))
    obj["payload"] = {
        "encoding": "generic-members",
        "state": "present",
        "members": [
            {
                "path": member_path,
                "encoding": "npy",
                "size_bytes": len(payload_bytes),
                "sha256": digest,
                "start": None,
                "count": None,
                "chunk_shape": list(array.shape),
            }
        ],
    }
    return member_path, payload_bytes, old_paths


def _update_counter_payloads(
    manifest: dict[str, Any],
    *,
    main_source_id: str,
    selected_source_counts: dict[str, int],
    total_selected: int,
) -> tuple[dict[str, bytes], set[str]]:
    payloads: dict[str, bytes] = {}
    replaced_paths: set[str] = set()
    objects = manifest.get("objects", {})
    if not isinstance(objects, dict):
        return payloads, replaced_paths

    selected_file_count = len([count for count in selected_source_counts.values() if count > 0])
    uniform_counts = set(selected_source_counts.values())
    uniform_per_file = next(iter(uniform_counts)) if len(uniform_counts) == 1 else None

    count_names = {"nimages", "nimages_collected", "nimages_written"}
    for obj in objects.values():
        if not isinstance(obj, dict) or str(obj.get("type")) != "dataset":
            continue
        leaf = PurePosixPath(_object_source_path(obj)).name
        source_id = str(obj.get("source_file") or "")
        value: int | None = None
        if leaf in count_names:
            if source_id == main_source_id:
                value = total_selected
            elif source_id in selected_source_counts:
                value = selected_source_counts[source_id]
        elif leaf == "nfiles":
            if source_id == main_source_id:
                value = selected_file_count
            elif source_id in selected_source_counts:
                value = 1
        elif leaf == "nimages_per_file":
            if source_id == main_source_id and uniform_per_file is not None:
                value = uniform_per_file
            elif source_id in selected_source_counts:
                value = selected_source_counts[source_id]
        if value is None:
            continue
        replacement = _replace_scalar_dataset_payload(obj, int(value))
        if replacement is None:
            continue
        path, data, old_paths = replacement
        payloads[path] = data
        replaced_paths.update(old_paths)
    return payloads, replaced_paths


def prepare_output_manifest(
    reader: HDXFArchiveReader,
    *,
    selected_by_object: dict[str, list[int]],
    operation_doc: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    dict[str, int],
    dict[str, int],
    dict[str, bytes],
    set[str],
]:
    """Prune dropped frame files and prepare fresh frame payload entries."""
    manifest = copy.deepcopy(reader.manifest)
    objects = manifest.get("objects")
    links = manifest.get("links")
    source = manifest.get("source")
    detector = manifest.get("detector_archive")
    if not isinstance(objects, dict) or not isinstance(links, list):
        raise HDXFProcessError("Manifest object graph is incomplete")
    if not isinstance(source, dict) or not isinstance(detector, dict):
        raise HDXFProcessError("Manifest source/detector sections are incomplete")

    original_frame_ids = [record.object_id for record in reader.datasets]
    frame_info = _frame_object_source_info(manifest, original_frame_ids)
    selected_ids = {
        object_id for object_id, indices in selected_by_object.items() if indices
    }
    selected_source_ids = {
        frame_info[object_id][0]
        for object_id in selected_ids
        if object_id in frame_info
    }
    all_frame_source_ids = {
        source_id for source_id, _ in frame_info.values() if source_id
    }
    main_source_id = str(source.get("main_file") or "")

    # Frame data sources referenced outside /entry/data may carry other useful
    # content.  Keep those physical source records even when their frame object
    # was not selected.
    protected_source_ids = {
        str(link.get("target_source_file") or "")
        for link in links
        if isinstance(link, dict)
        and not str(link.get("path") or "").startswith("/entry/data/")
        and str(link.get("path") or "") != "/entry/data"
    }
    dropped_source_ids = (
        all_frame_source_ids
        - selected_source_ids
        - protected_source_ids
        - {main_source_id}
    )

    source_files = source.get("files")
    if isinstance(source_files, list):
        source["files"] = [
            item
            for item in source_files
            if not isinstance(item, dict)
            or str(item.get("id") or "") not in dropped_source_ids
        ]

    removed_object_ids: set[str] = set()
    for object_id, obj in list(objects.items()):
        if not isinstance(obj, dict):
            continue
        source_id = str(obj.get("source_file") or "")
        if source_id in dropped_source_ids:
            removed_object_ids.add(str(object_id))
            del objects[object_id]
            continue
        if str(object_id) in original_frame_ids and str(object_id) not in selected_ids:
            removed_object_ids.add(str(object_id))
            del objects[object_id]

    selected_target_pairs = {
        frame_info[object_id]
        for object_id in selected_ids
        if object_id in frame_info
    }
    pruned_links: list[dict[str, Any]] = []
    selected_data_links: list[dict[str, Any]] = []
    for link in links:
        if not isinstance(link, dict):
            continue
        if str(link.get("target_source_file") or "") in dropped_source_ids:
            continue
        if str(link.get("target_object") or "") in removed_object_ids:
            continue
        path = str(link.get("path") or "")
        if str(link.get("type") or "") == "external" and (
            path == "/entry/data" or path.startswith("/entry/data/")
        ):
            pair = (
                str(link.get("target_source_file") or ""),
                str(link.get("target_path") or ""),
            )
            if pair not in selected_target_pairs:
                continue
            selected_data_links.append(dict(link))
            continue
        pruned_links.append(dict(link))

    selected_order = [
        record.object_id for record in reader.datasets if record.object_id in selected_ids
    ]
    pair_order = {
        frame_info[object_id]: index
        for index, object_id in enumerate(selected_order)
        if object_id in frame_info
    }
    selected_data_links.sort(
        key=lambda link: pair_order.get(
            (
                str(link.get("target_source_file") or ""),
                str(link.get("target_path") or ""),
            ),
            10**9,
        )
    )
    for sequence, link in enumerate(selected_data_links, start=1):
        old_path = str(link.get("path") or "")
        if old_path != "/entry/data":
            link["path"] = f"/entry/data/data_{sequence:06d}"
        pruned_links.append(link)
    manifest["links"] = pruned_links

    members_by_object: dict[str, list[dict[str, Any]]] = {
        object_id: [] for object_id in selected_order
    }
    global_start_by_object: dict[str, int] = {}
    count_by_object: dict[str, int] = {}
    selected_source_counts: dict[str, int] = defaultdict(int)
    global_cursor = 0
    frame_datasets: list[dict[str, Any]] = []
    for object_id in selected_order:
        count = len(selected_by_object[object_id])
        count_by_object[object_id] = count
        global_start_by_object[object_id] = global_cursor
        obj = objects.get(object_id)
        if not isinstance(obj, dict):
            raise HDXFProcessError(f"Selected frame object is missing: {object_id}")
        source_id = str(obj.get("source_file") or "")
        selected_source_counts[source_id] += count
        _update_frame_storage(
            obj,
            frame_count=count,
            frame_shape=reader.frame_shape,
            global_low=global_cursor + 1,
            global_high=global_cursor + count,
        )
        frame_datasets.append(
            {
                "object": object_id,
                "source_path": str(obj.get("source_path") or ""),
                "global_start_frame": global_cursor,
                "frame_count": count,
            }
        )
        global_cursor += count

    frames = detector.get("frames")
    if not isinstance(frames, dict):
        raise HDXFProcessError("detector_archive.frames is missing")
    frames.pop("block_index", None)
    frames.pop("static_models", None)
    frames["blocks"] = []
    frames["keyframe_index"] = []
    frames["datasets"] = frame_datasets
    frames["frame_count"] = global_cursor
    frames["shape"] = [global_cursor, *reader.frame_shape]
    frames["frame_shape"] = list(reader.frame_shape)

    scalar_payloads, scalar_old_paths = _update_counter_payloads(
        manifest,
        main_source_id=main_source_id,
        selected_source_counts=dict(selected_source_counts),
        total_selected=global_cursor,
    )

    detector["encoding_statistics"] = {
        "frame_blocks": 0,
        "logical_frame_bytes": 0,
        "encoded_frame_bytes": 0,
    }
    manifest["processing"] = operation_doc
    return (
        manifest,
        members_by_object,
        global_start_by_object,
        count_by_object,
        scalar_payloads,
        scalar_old_paths,
    )


def _clone_zipinfo(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    cloned = zipfile.ZipInfo(info.filename, date_time=info.date_time)
    cloned.compress_type = info.compress_type
    cloned.comment = info.comment
    cloned.extra = info.extra
    cloned.internal_attr = info.internal_attr
    cloned.external_attr = info.external_attr
    cloned.create_system = info.create_system
    cloned.flag_bits = info.flag_bits
    return cloned


def _copy_zip_members(
    source: zipfile.ZipFile,
    destination: zipfile.ZipFile,
    *,
    skip_paths: set[str],
    written_names: set[str],
) -> None:
    for info in source.infolist():
        if info.filename == MANIFEST_PATH or info.filename in skip_paths:
            continue
        cloned = _clone_zipinfo(info)
        with source.open(info, "r") as src, destination.open(
            cloned, "w", force_zip64=True
        ) as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
        written_names.add(info.filename)


def _write_stored_member(
    archive: zipfile.ZipFile,
    path: str,
    data: bytes,
    written_names: set[str],
) -> None:
    if path in written_names:
        return
    archive.writestr(path, data, compress_type=zipfile.ZIP_STORED)
    written_names.add(path)


def _parse_remove_range(text: str | None) -> tuple[float | None, float | None]:
    if text is None:
        return None, None
    try:
        left, right = [item.strip() for item in text.split(",", 1)]
        low, high = float(left), float(right)
    except Exception as exc:
        raise HDXFProcessError(
            "Invalid --remove-threshold. Use two numbers such as -5,6"
        ) from exc
    if low > high:
        raise HDXFProcessError(
            f"--remove-threshold invalid: low ({low}) > high ({high})"
        )
    return low, high


def _parse_protect_roi(
    text: str | None,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    try:
        values = [int(item.strip()) for item in text.split(",")]
    except Exception as exc:
        raise HDXFProcessError(
            "Invalid --protect. Use x0,y0,x1,y1 with integer coordinates"
        ) from exc
    if len(values) != 4:
        raise HDXFProcessError(
            "Invalid --protect. Use x0,y0,x1,y1 with integer coordinates"
        )
    x0, y0, x1, y1 = values
    x0 = max(0, min(x0, width))
    x1 = max(0, min(x1, width))
    y0 = max(0, min(y0, height))
    y1 = max(0, min(y1, height))
    if x1 <= x0 or y1 <= y0:
        raise HDXFProcessError("--protect ROI is empty after clamping")
    return x0, y0, x1, y1


def _selected_indices(frame_count: int, args: argparse.Namespace) -> list[int]:
    if args.select_frame:
        try:
            indices = [
                int(item.strip()) - 1
                for item in args.select_frame.split(",")
                if item.strip()
            ]
        except Exception as exc:
            raise HDXFProcessError(
                f"Cannot parse --select-frame {args.select_frame!r}"
            ) from exc
        if not indices:
            raise HDXFProcessError("--select-frame did not contain any frame numbers")
        if any(index < 0 or index >= frame_count for index in indices):
            raise HDXFProcessError(
                f"--select-frame is outside the valid range 1..{frame_count}"
            )
        return sorted(set(indices))

    if args.frame_from is not None or args.frame_to is not None:
        start = (args.frame_from if args.frame_from is not None else 1) - 1
        stop_inclusive = (args.frame_to if args.frame_to is not None else frame_count) - 1
        if start < 0 or stop_inclusive < start or stop_inclusive >= frame_count:
            raise HDXFProcessError(
                f"Invalid frame range; valid frames are 1..{frame_count}"
            )
        return list(range(start, stop_inclusive + 1))

    return list(range(frame_count))


def _selection_label(args: argparse.Namespace) -> str:
    if args.select_frame:
        items = [item.strip() for item in args.select_frame.split(",") if item.strip()]
        return "SEL_" + "-".join(items)
    if args.frame_from is not None or args.frame_to is not None:
        start = args.frame_from if args.frame_from is not None else 1
        stop = args.frame_to if args.frame_to is not None else "end"
        return f"RANGE_{start}-{stop}"
    return "ALL"


def _operation_label(
    args: argparse.Namespace,
    *,
    remove_low: float | None,
    remove_high: float | None,
    protect_roi: tuple[int, int, int, int] | None,
) -> str:
    if args.threshold is not None:
        mode = f"THR{args.threshold:g}"
    elif args.background is not None:
        mode = f"BGf{args.background_frame}"
    elif remove_low is not None and remove_high is not None:
        mode = f"RM_{remove_low:g}_to_{remove_high:g}"
    else:
        mode = "COPY"
    parts = [mode, _selection_label(args)]
    if args.upper_threshold is not None:
        parts.append(f"UP{args.upper_threshold:g}")
    if protect_roi is not None:
        x0, y0, x1, y1 = protect_roi
        parts.append(f"PROT_{x0}-{y0}-{x1}-{y1}_KILL_GE_PROTMAX")
    return "_".join(parts)


def _operation_document(
    args: argparse.Namespace,
    *,
    input_path: Path,
    selected: list[int],
    remove_low: float | None,
    remove_high: float | None,
    protect_roi: tuple[int, int, int, int] | None,
) -> dict[str, Any]:
    mode = (
        "threshold" if args.threshold is not None
        else "background" if args.background is not None
        else "remove" if remove_low is not None
        else "copy"
    )
    return {
        "tool": "subtract_hdxf.py",
        "version": SCRIPT_VERSION,
        "operation": mode,
        "input_archive": input_path.name,
        "selected_frame_count": len(selected),
        "selection": (
            {"type": "explicit", "value": args.select_frame}
            if args.select_frame
            else {
                "type": "range",
                "from_1based": args.frame_from if args.frame_from is not None else 1,
                "to_1based": args.frame_to,
            }
            if args.frame_from is not None or args.frame_to is not None
            else {"type": "all"}
        ),
        "selected_frames_1based": (
            [index + 1 for index in selected] if len(selected) <= 1000 else None
        ),
        "threshold": args.threshold,
        "background_archive": args.background.name if args.background else None,
        "background_frame_1based": args.background_frame if args.background else None,
        "remove_threshold": (
            [remove_low, remove_high] if remove_low is not None else None
        ),
        "upper_threshold": args.upper_threshold,
        "fill": args.fill,
        "protect_roi": list(protect_roi) if protect_roi is not None else None,
        "frame_transform": args.frame_transform,
        "block_frames": args.block_frames,
        "zstd_level": args.zstd_level,
    }


def _group_selection_by_object(
    reader: HDXFArchiveReader,
    selected: Sequence[int],
) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for global_index in selected:
        record = reader.dataset_for_global_index(global_index)
        grouped[record.object_id].append(global_index)
    return dict(grouped)


def _load_background(
    path: Path,
    frame_1based: int,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    with HDXFArchiveReader(path) as archive:
        if frame_1based < 1 or frame_1based > archive.frame_count:
            raise HDXFProcessError(
                f"--background-frame is outside 1..{archive.frame_count}"
            )
        frame = archive.get_frame(frame_1based - 1)
    if tuple(frame.shape) != tuple(expected_shape):
        raise HDXFProcessError(
            f"Background shape {frame.shape} does not match input {expected_shape}"
        )
    return np.ascontiguousarray(frame)


def _progress_printer(total: int, step_percent: float):
    state = {"done": 0, "last": -step_percent, "started": time.perf_counter()}
    step = max(0.1, float(step_percent))

    def update(count: int) -> None:
        state["done"] += int(count)
        done = state["done"]
        percent = 100.0 * done / total if total else 100.0
        if percent + 1e-9 < state["last"] + step and done < total:
            return
        elapsed = time.perf_counter() - state["started"]
        eta = elapsed / done * (total - done) if done else 0.0
        print(
            f"[PROC] {done}/{total} ({percent:.1f}%) "
            f"elapsed={elapsed:.2f}s ETA={max(0.0, eta):.2f}s",
            flush=True,
        )
        state["last"] = percent

    return update


def _verify_output(
    output_path: Path,
    input_path: Path,
    selected: Sequence[int],
    transform: TransformConfig,
    *,
    progress_step: float = 5.0,
) -> None:
    """Full pixel-exact verification using sequential HDXFB block streaming.

    The old verifier called get_frame() once for every source and output frame.
    With chained Delta blocks that could repeatedly walk from a keyframe/anchor
    and made verification substantially slower than the processing pass.

    This implementation keeps the same strong guarantee -- it re-applies the
    requested transform to the original selected pixels and compares every
    output pixel -- but each source/output HDXFB block is decoded at most once.
    """
    total = len(selected)
    print(
        "[VERIFY] full pixel verification (streaming blocks; each block decoded once)",
        flush=True,
    )
    started = time.perf_counter()
    step = max(0.1, float(progress_step))
    last_percent = -step

    with HDXFArchiveReader(input_path, block_cache_size=1) as source, HDXFArchiveReader(
        output_path, block_cache_size=1
    ) as output:
        if output.frame_count != total:
            raise HDXFProcessError(
                f"Output frame count {output.frame_count} != expected {total}"
            )
        if tuple(output.frame_shape) != tuple(source.frame_shape):
            raise HDXFProcessError(
                f"Output frame shape {output.frame_shape} != source {source.frame_shape}"
            )

        source_iter = iter(source.iter_selected_blocks(selected))
        output_iter = iter(output.iter_frame_blocks())

        source_indices: list[int] = []
        expected_block: np.ndarray | None = None
        source_pos = 0
        actual_block: np.ndarray | None = None
        output_pos = 0
        verified = 0

        def load_expected() -> bool:
            nonlocal source_indices, expected_block, source_pos
            try:
                indices, _source_object, source_block = next(source_iter)
            except StopIteration:
                source_indices = []
                expected_block = None
                source_pos = 0
                return False
            source_indices = indices
            expected_block = transform_block(source_block, transform)
            source_pos = 0
            return True

        def load_actual() -> bool:
            nonlocal actual_block, output_pos
            try:
                _global_start, _source_object, block = next(output_iter)
            except StopIteration:
                actual_block = None
                output_pos = 0
                return False
            actual_block = block
            output_pos = 0
            return True

        have_expected = load_expected()
        have_actual = load_actual()

        while have_expected and have_actual:
            assert expected_block is not None
            assert actual_block is not None
            expected_remaining = expected_block.shape[0] - source_pos
            actual_remaining = actual_block.shape[0] - output_pos
            count = min(expected_remaining, actual_remaining)
            if count <= 0:
                raise HDXFProcessError("Verification stream made no forward progress")

            expected_chunk = expected_block[source_pos : source_pos + count]
            actual_chunk = actual_block[output_pos : output_pos + count]

            equal = (
                actual_chunk.dtype == expected_chunk.dtype
                and actual_chunk.shape == expected_chunk.shape
                and np.array_equal(actual_chunk, expected_chunk, equal_nan=True)
            )
            if not equal:
                # The common path compares a whole chunk once.  Only on failure
                # do the more expensive per-frame checks needed for a useful
                # error message.
                for offset in range(count):
                    expected_frame = expected_chunk[offset]
                    actual_frame = actual_chunk[offset]
                    if (
                        actual_frame.dtype != expected_frame.dtype
                        or actual_frame.shape != expected_frame.shape
                        or not np.array_equal(
                            actual_frame, expected_frame, equal_nan=True
                        )
                    ):
                        output_index = verified + offset
                        source_index = source_indices[source_pos + offset]
                        detail = ""
                        if (
                            actual_frame.dtype == expected_frame.dtype
                            and actual_frame.shape == expected_frame.shape
                        ):
                            try:
                                mismatch = np.argwhere(
                                    ~np.isclose(
                                        actual_frame,
                                        expected_frame,
                                        rtol=0.0,
                                        atol=0.0,
                                        equal_nan=True,
                                    )
                                )
                                if mismatch.size:
                                    y, x = (int(v) for v in mismatch[0][:2])
                                    detail = (
                                        f" at pixel (x={x}, y={y}): "
                                        f"actual={actual_frame[y, x]!r}, "
                                        f"expected={expected_frame[y, x]!r}"
                                    )
                            except Exception:
                                pass
                        raise HDXFProcessError(
                            f"Verification failed at output frame {output_index + 1} "
                            f"(source frame {source_index + 1}){detail}"
                        )
                raise HDXFProcessError(
                    f"Verification mismatch in output frames "
                    f"{verified + 1}..{verified + count}"
                )

            source_pos += count
            output_pos += count
            verified += count

            percent = 100.0 * verified / total if total else 100.0
            if percent + 1e-9 >= last_percent + step or verified == total:
                elapsed = time.perf_counter() - started
                eta = elapsed / verified * (total - verified) if verified else 0.0
                print(
                    f"[VERIFY] {verified}/{total} ({percent:.1f}%) frames match "
                    f"elapsed={elapsed:.2f}s ETA={max(0.0, eta):.2f}s",
                    flush=True,
                )
                last_percent = percent

            if source_pos >= expected_block.shape[0]:
                have_expected = load_expected()
            if output_pos >= actual_block.shape[0]:
                have_actual = load_actual()

        if have_expected or have_actual:
            raise HDXFProcessError(
                "Verification stream length mismatch between expected and output frames"
            )
        if verified != total:
            raise HDXFProcessError(
                f"Verification compared {verified} frames, expected {total}"
            )

    elapsed = time.perf_counter() - started
    print(
        f"[VERIFY] PASS - every output frame matches the requested transform "
        f"({total} frames, {elapsed:.2f}s)",
        flush=True,
    )


def process_archive(
    input_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    global DEBUG
    DEBUG = bool(args.debug)
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    if not input_path.is_file():
        raise HDXFProcessError(f"Input HDXF does not exist: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mem and psutil is None and not tracemalloc.is_tracing():
        tracemalloc.start()

    remove_low, remove_high = _parse_remove_range(args.remove_threshold)
    mode_count = sum(
        [
            args.threshold is not None,
            args.background is not None,
            remove_low is not None,
        ]
    )
    if mode_count > 1:
        raise HDXFProcessError(
            "Use at most one of --threshold, --background, or --remove-threshold"
        )
    if args.block_frames < 1:
        raise HDXFProcessError("--block-frames must be at least 1")
    if args.keyframe_interval_blocks < 1:
        raise HDXFProcessError("--keyframe-interval-blocks must be at least 1")
    if args.codec != "auto":
        print(
            f"[WARN] --codec {args.codec!r} is an HDF5-era option and is ignored. "
            "HDXF output uses HDXFB + Zstd.",
            flush=True,
        )
    if args.threads != 1:
        print(
            "[WARN] --threads is currently ignored for one HDXF archive because "
            "Delta decoding/encoding is ordered. Batch archives are processed sequentially.",
            flush=True,
        )
    if args.per_file:
        print(
            "[INFO] --per-file has no additional effect: one HDXF archive is one "
            "global frame sequence.",
            flush=True,
        )

    with HDXFArchiveReader(input_path, block_cache_size=2) as reader:
        selected = _selected_indices(reader.frame_count, args)
        selected_by_object = _group_selection_by_object(reader, selected)
        height, width = reader.frame_shape
        protect_roi = _parse_protect_roi(
            args.protect,
            width=width,
            height=height,
        )
        background = None
        if args.background is not None:
            background = _load_background(
                args.background.resolve(),
                args.background_frame,
                reader.frame_shape,
            )

        mode = (
            "threshold" if args.threshold is not None
            else "background" if args.background is not None
            else "remove" if remove_low is not None
            else "copy"
        )
        transform_config = TransformConfig(
            mode=mode,
            threshold=args.threshold,
            background=background,
            remove_low=remove_low,
            remove_high=remove_high,
            upper_threshold=args.upper_threshold,
            fill=args.fill,
            protect_roi=protect_roi,
        )
        label = _operation_label(
            args,
            remove_low=remove_low,
            remove_high=remove_high,
            protect_roi=protect_roi,
        )
        prefix = args.out_prefix.strip() if args.out_prefix else f"{input_path.stem}_{label}"
        output_path = output_dir / f"{prefix}.hdxf"
        if output_path.resolve() == input_path:
            raise HDXFProcessError("Input and output paths must be different")
        if output_path.exists() and not args.overwrite:
            raise HDXFProcessError(
                f"Output already exists: {output_path} (use --overwrite)"
            )

        operation_doc = _operation_document(
            args,
            input_path=input_path,
            selected=selected,
            remove_low=remove_low,
            remove_high=remove_high,
            protect_roi=protect_roi,
        )
        (
            manifest,
            members_by_object,
            global_start_by_object,
            count_by_object,
            scalar_payloads,
            scalar_old_paths,
        ) = prepare_output_manifest(
            reader,
            selected_by_object=selected_by_object,
            operation_doc=operation_doc,
        )
        detector = manifest["detector_archive"]
        frames = detector["frames"]
        blocks_out: list[dict[str, Any]] = frames["blocks"]
        keyframes_out: list[int] = frames["keyframe_index"]

        old_frame_paths = _old_frame_payload_paths(reader)
        skip_paths = old_frame_paths
        # Keep replaced scalar payload members. Payloads are content-addressed
        # and may theoretically be shared by another metadata Dataset; leaving
        # the now-unreferenced small member is safer than removing shared data.
        if scalar_old_paths:
            dprint(f"[DBG] retained {len(scalar_old_paths)} replaced scalar payload member(s)")

        print(f"[START] HDXF frame processing: {input_path}", flush=True)
        print(
            f"[INFO] source frames={reader.frame_count}, selected={len(selected)}, "
            f"shape={reader.frame_shape}, dtype={reader.dtype}",
            flush=True,
        )
        print(f"[CFG] mode={mode}, fill={args.fill:g}", flush=True)
        if background is not None:
            print(
                f"[CFG] background={args.background}, frame={args.background_frame}",
                flush=True,
            )
        if protect_roi is not None:
            print(f"[CFG] protect ROI={protect_roi}", flush=True)
        print(
            f"[CFG] HDXFB transform={args.frame_transform}, block_frames={args.block_frames}, "
            f"zstd_level={args.zstd_level}, delta_stream={args.delta_stream}",
            flush=True,
        )
        print(f"[OUT] {output_path}", flush=True)
        print_mem("start", args.mem)

        fd, temp_name = tempfile.mkstemp(
            prefix=output_path.name + ".",
            suffix=".tmp",
            dir=output_dir,
        )
        os.close(fd)
        temp_path = Path(temp_name)
        progress = _progress_printer(len(selected), args.progress_step)

        written_names: set[str] = set()
        members_written_by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
        object_block_counts: dict[str, int] = defaultdict(int)
        encoded_logical_bytes = 0
        encoded_payload_bytes = 0
        new_frame_paths: set[str] = set()
        inflation_checked = False

        # Per-object chain state.  Frame datasets are contiguous in the global
        # sequence, but explicit dictionaries make the invariant robust.
        chain_state: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "previous_frame": None,
                "block_index": 0,
                "anchor_global": None,
                "learned_filter": None,
                "local_cursor": 0,
            }
        )

        def write_transformed_block(
            archive: zipfile.ZipFile,
            object_id: str,
            source_global_indices: list[int],
            source_frames: list[np.ndarray],
        ) -> None:
            nonlocal encoded_logical_bytes, encoded_payload_bytes, inflation_checked
            if not source_frames:
                return
            raw_block = np.stack(source_frames, axis=0)
            transformed = transform_block(raw_block, transform_config)
            state = chain_state[object_id]
            block_number = int(state["block_index"])
            is_anchor = block_number % int(args.keyframe_interval_blocks) == 0
            previous = None if is_anchor else state["previous_frame"]
            data, header, candidate_sizes = encode_output_block(
                transformed,
                args=args,
                previous_frame=previous,
                learned_delta_filter=state["learned_filter"],
            )
            if args.delta_filter == "learn" and state["learned_filter"] is None:
                learned = header.get("selected_delta_filter")
                if learned in ("none", "shuffle", "bitshuffle"):
                    state["learned_filter"] = str(learned)
                    print(
                        f"[ENCODE] learned Delta filter for {object_id}: {learned}",
                        flush=True,
                    )

            digest = sha256_bytes(data)
            version = int(header.get("version", 1))
            encoding = _frame_encoding_name(version)
            member_path = f"payloads/{digest}.hdxfb"
            _write_stored_member(archive, member_path, data, written_names)
            new_frame_paths.add(member_path)

            local_start = int(state["local_cursor"])
            global_start = int(global_start_by_object[object_id]) + local_start
            frame_count = int(transformed.shape[0])
            requires_previous = bool(header.get("requires_previous_frame", False))
            if is_anchor or not requires_previous or state["anchor_global"] is None:
                state["anchor_global"] = global_start
            anchor_global = int(state["anchor_global"])

            member: dict[str, Any] = {
                "path": member_path,
                "encoding": encoding,
                "start": [local_start, 0, 0],
                "count": [frame_count, int(height), int(width)],
                "chunk_shape": list(transformed.shape),
                "size_bytes": len(data),
                "sha256": digest,
                "start_frame": local_start,
                "global_start_frame": global_start,
                "frame_count": frame_count,
                "transform": str(header.get("transform") or args.frame_transform),
                "codec": "blosc2-zstd",
                "candidate_sizes": candidate_sizes,
                "logical_sha256": str(header.get("logical_sha256") or ""),
                "hdxfb_version": version,
                "chain_mode": header.get("chain_mode"),
                "requires_previous_frame": requires_previous,
                "chain_anchor_global_start_frame": anchor_global,
                "delta_encoding": header.get("delta_encoding"),
                "keyframe_encoding": header.get("keyframe_encoding"),
                "source_frame_indices_1based": [index + 1 for index in source_global_indices],
            }
            for source_key, target_key in (
                ("mode_counts", "tile_mode_counts"),
                ("tile_shape", "tile_shape"),
                ("delta_dtype", "delta_dtype"),
            ):
                if source_key in header:
                    member[target_key] = _normalise_json(header[source_key])
            member = _normalise_json(member)
            members_written_by_object[object_id].append(dict(member))
            blocks_out.append({**member, "source_object": object_id})
            if not requires_previous:
                keyframes_out.append(global_start)

            state["previous_frame"] = np.ascontiguousarray(transformed[-1])
            state["block_index"] = block_number + 1
            state["local_cursor"] = local_start + frame_count
            object_block_counts[object_id] += 1
            encoded_logical_bytes += int(transformed.nbytes)
            encoded_payload_bytes += int(len(data))

            if args.stop_if_inflating and not inflation_checked:
                inflation_checked = True
                if len(data) >= 0.95 * transformed.nbytes:
                    raise HDXFProcessError(
                        "The first HDXFB block is approximately raw size. "
                        "Aborting due to --stop-if-inflating."
                    )
            progress(frame_count)
            dprint(
                f"[DBG] block object={object_id} source={source_global_indices[0] + 1}.."
                f"{source_global_indices[-1] + 1} logical={transformed.nbytes} "
                f"encoded={len(data)} transform={header.get('transform')}"
            )

        try:
            with zipfile.ZipFile(input_path, "r") as source_zip, zipfile.ZipFile(
                temp_path,
                "w",
                compression=zipfile.ZIP_STORED,
                allowZip64=True,
            ) as output_zip:
                _copy_zip_members(
                    source_zip,
                    output_zip,
                    skip_paths=skip_paths,
                    written_names=written_names,
                )
                for path, data in scalar_payloads.items():
                    _write_stored_member(output_zip, path, data, written_names)

                current_object: str | None = None
                buffer_frames: list[np.ndarray] = []
                buffer_indices: list[int] = []
                for global_index, object_id, frame in reader.iter_selected_frames(selected):
                    expected_object = reader.dataset_for_global_index(global_index).object_id
                    if object_id and object_id != expected_object:
                        raise HDXFProcessError(
                            f"Frame index source_object mismatch at frame {global_index + 1}: "
                            f"block={object_id}, dataset={expected_object}"
                        )
                    object_id = expected_object
                    if current_object is None:
                        current_object = object_id
                    if object_id != current_object or len(buffer_frames) >= args.block_frames:
                        write_transformed_block(
                            output_zip,
                            current_object,
                            buffer_indices,
                            buffer_frames,
                        )
                        buffer_frames = []
                        buffer_indices = []
                        current_object = object_id
                    buffer_frames.append(frame)
                    buffer_indices.append(global_index)
                    if len(buffer_frames) >= args.block_frames:
                        write_transformed_block(
                            output_zip,
                            current_object,
                            buffer_indices,
                            buffer_frames,
                        )
                        buffer_frames = []
                        buffer_indices = []
                if buffer_frames and current_object is not None:
                    write_transformed_block(
                        output_zip,
                        current_object,
                        buffer_indices,
                        buffer_frames,
                    )

                for object_id, members in members_written_by_object.items():
                    obj = manifest["objects"].get(object_id)
                    if not isinstance(obj, dict):
                        raise HDXFProcessError(
                            f"Output frame object disappeared from manifest: {object_id}"
                        )
                    payload = obj.get("payload")
                    if not isinstance(payload, dict):
                        payload = {"encoding": "frame-sequence", "state": "present"}
                        obj["payload"] = payload
                    payload["encoding"] = "frame-sequence"
                    payload["state"] = "present"
                    payload["members"] = members

                for object_id, expected_count in count_by_object.items():
                    state = chain_state[object_id]
                    if int(state["local_cursor"]) != expected_count:
                        raise HDXFProcessError(
                            f"Frame count mismatch for {object_id}: wrote "
                            f"{state['local_cursor']}, expected {expected_count}"
                        )

                policy = detector.setdefault("encoding_policy", {})
                if isinstance(policy, dict):
                    policy.update(
                        {
                            "frame_transform": args.frame_transform,
                            "frame_codec": "HDXFB + blosc2-zstd",
                            "zstd_level": int(args.zstd_level),
                            "delta_filter": args.delta_filter,
                            "delta_stream": args.delta_stream,
                            "keyframe_interval_blocks": int(args.keyframe_interval_blocks),
                            "zip_payload_compression": "stored",
                        }
                    )
                detector["encoding_statistics"] = {
                    "frame_blocks": len(blocks_out),
                    "logical_frame_bytes": encoded_logical_bytes,
                    "encoded_frame_bytes": encoded_payload_bytes,
                    "compression_ratio": (
                        encoded_logical_bytes / encoded_payload_bytes
                        if encoded_payload_bytes > 0
                        else None
                    ),
                }
                integrity = manifest.setdefault("integrity", {})
                if isinstance(integrity, dict):
                    integrity["algorithm"] = "sha256"
                    integrity["unique_payload_members"] = len(
                        [name for name in written_names if name.startswith("payloads/")]
                    )
                    integrity["derived_frame_payload_members"] = len(new_frame_paths)

                manifest_bytes = strict_json_dumps(manifest, indent=2)
                output_zip.writestr(
                    MANIFEST_PATH,
                    manifest_bytes,
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=6,
                )

            if output_path.exists():
                output_path.unlink()
            os.replace(temp_path, output_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    print_mem("end", args.mem)
    size_mib = output_path.stat().st_size / (1024 ** 2)
    ratio = (
        encoded_logical_bytes / encoded_payload_bytes
        if encoded_payload_bytes > 0
        else float("inf")
    )
    print(
        f"[SUMMARY] frames={len(selected)}, blocks={len(blocks_out)}, "
        f"logical={encoded_logical_bytes / 1024**2:.2f} MiB, "
        f"frame_payload={encoded_payload_bytes / 1024**2:.2f} MiB, "
        f"frame_ratio={ratio:.2f}x, archive={size_mib:.2f} MiB",
        flush=True,
    )
    print(f"[OUT HDXF] {output_path}", flush=True)

    if args.verify:
        _verify_output(
            output_path,
            input_path,
            selected,
            transform_config,
            progress_step=args.progress_step,
        )
    print("[RESULT] PASS - HDXF processing completed", flush=True)
    return output_path


def scan_folder_for_archives(
    folder: Path,
    *,
    recursive: bool,
    pattern: str,
) -> list[Path]:
    folder = folder.resolve()
    iterator = folder.rglob(pattern) if recursive else folder.glob(pattern)
    archives = [
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() == ".hdxf"
    ]
    archives.sort()
    return archives


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decode selected HDXF detector frames, apply threshold/background/"
            "range/ROI processing, and write a new HDXF archive."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-i", "--input", type=Path, help="Input .hdxf archive")
    source.add_argument(
        "--input-dir",
        type=Path,
        help="Folder containing .hdxf archives for batch processing",
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--recursive", action="store_true", help="Scan input-dir recursively")
    parser.add_argument(
        "--pattern",
        default="*.hdxf",
        help="Batch filename pattern (default: *.hdxf)",
    )
    parser.add_argument(
        "--mirror-dirs",
        action="store_true",
        default=True,
        help="Mirror input-dir subdirectories under output (default: on)",
    )
    parser.add_argument(
        "--no-mirror-dirs",
        dest="mirror_dirs",
        action="store_false",
        help="Write every batch output directly under --output",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Set selected-frame pixels below this value to --fill",
    )
    parser.add_argument(
        "--background",
        type=Path,
        default=None,
        help="Background .hdxf archive with the same frame shape",
    )
    parser.add_argument(
        "--background-frame",
        type=int,
        default=1,
        help="1-based frame number in the background HDXF (default: 1)",
    )
    parser.add_argument(
        "--remove-threshold",
        type=str,
        default=None,
        help="Inclusive range 'low,high'; matching pixels are set to --fill",
    )
    parser.add_argument(
        "--upper-threshold",
        type=float,
        default=None,
        help="Set pixels strictly above this value to --fill",
    )
    parser.add_argument(
        "--fill",
        type=float,
        default=0.0,
        help="Replacement value for threshold/range operations (default: 0)",
    )
    parser.add_argument(
        "--protect",
        type=str,
        default=None,
        help=(
            "Protected ROI x0,y0,x1,y1. The ROI is restored unchanged; outside "
            "it, values >= the original ROI maximum are set to zero."
        ),
    )

    parser.add_argument(
        "--select-frame",
        type=str,
        default=None,
        help="Comma-separated 1-based global frames to keep",
    )
    parser.add_argument(
        "--frame_from",
        "--frame-from",
        dest="frame_from",
        type=int,
        default=None,
        help="1-based inclusive first frame to keep",
    )
    parser.add_argument(
        "--frame_to",
        "--frame-to",
        dest="frame_to",
        type=int,
        default=None,
        help="1-based inclusive last frame to keep",
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="Compatibility option; one HDXF archive already has one global sequence",
    )

    parser.add_argument(
        "--frame-transform",
        choices=("delta", "adaptive", "auto", "raw"),
        default="delta",
        help="HDXFB frame encoding policy (default: delta)",
    )
    parser.add_argument(
        "--block-frames",
        type=int,
        default=8,
        help="Frames per output HDXFB block (default: 8)",
    )
    parser.add_argument(
        "--keyframe-interval-blocks",
        type=int,
        default=4,
        help="Force a Delta anchor every N output blocks (default: 4)",
    )
    parser.add_argument(
        "--delta-stream",
        choices=("zero-rle-bitpack", "auto", "zstd"),
        default="zero-rle-bitpack",
        help="Delta payload stream (default: zero-rle-bitpack)",
    )
    parser.add_argument(
        "--delta-filter",
        choices=("learn", "auto", "none", "shuffle", "bitshuffle"),
        default="learn",
        help="Delta prefilter policy (default: learn)",
    )
    parser.add_argument(
        "--zero-rle-threshold",
        type=float,
        default=0.15,
        help="Minimum zero fraction for zero-RLE selection (default: 0.15)",
    )
    parser.add_argument(
        "--zstd-level",
        type=int,
        default=3,
        choices=tuple(range(10)),
        help="Blosc2 Zstd compression level 0..9 (default: 3)",
    )
    parser.add_argument(
        "--compute-backend",
        choices=("cpu-streaming", "cpu-vectorized", "cuda-cupy"),
        default="cpu-streaming",
        help="Delta calculation backend (default: cpu-streaming)",
    )
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--tile-height", type=int, default=128)
    parser.add_argument("--tile-width", type=int, default=128)
    parser.add_argument("--sparse-threshold", type=float, default=0.02)
    parser.add_argument(
        "--no-jungfrau-split",
        action="store_true",
        help="Disable Jungfrau-specific adaptive tile splitting",
    )

    # Compatibility with the HDF5 script/UI.  HDXF has its own fixed codec.
    parser.add_argument(
        "--codec",
        choices=("auto", "bs-lz4", "bs-zstd", "gzip", "none"),
        default="auto",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stop-if-inflating",
        action="store_true",
        help="Abort when the first encoded frame block is approximately raw size",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Compatibility option; ordered HDXF processing currently uses one worker",
    )
    parser.add_argument("--mem", action="store_true", help="Print memory usage")
    parser.add_argument(
        "--progress-step",
        type=float,
        default=5.0,
        help="Progress print interval in percent (default: 5)",
    )
    parser.add_argument("--debug", action="store_true", help="Verbose encoding logs")
    parser.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help="Override output filename prefix in single-archive mode",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Replace an existing output archive (default: on)",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Refuse to replace an existing output archive",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=("Full pixel-exact verification after writing; uses sequential "
              "block streaming so each HDXFB block is decoded at most once"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.input_dir is not None:
            input_base = args.input_dir.resolve()
            output_base = args.output.resolve()
            output_base.mkdir(parents=True, exist_ok=True)
            archives = scan_folder_for_archives(
                input_base,
                recursive=args.recursive,
                pattern=args.pattern,
            )
            if not archives:
                print(
                    f"[BATCH] No HDXF archive matched {input_base} "
                    f"(recursive={args.recursive}, pattern={args.pattern!r})",
                    flush=True,
                )
                return 0
            print(f"[BATCH] Found {len(archives)} HDXF archive(s)", flush=True)
            failures = 0
            for index, archive in enumerate(archives, start=1):
                relative = archive.parent.relative_to(input_base) if args.mirror_dirs else Path()
                output_dir = output_base / relative
                print(f"\n[BATCH {index}/{len(archives)}] {archive}", flush=True)
                try:
                    process_archive(archive, output_dir, args)
                except Exception as exc:
                    failures += 1
                    print(
                        f"[BATCH] FAIL {archive}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
            if failures:
                print(f"[BATCH] Completed with {failures} failure(s)", flush=True)
                return 1
            print("[BATCH] PASS - all archives completed", flush=True)
            return 0

        assert args.input is not None
        process_archive(args.input, args.output, args)
        return 0
    except KeyboardInterrupt:
        print("\n[RESULT] STOPPED - interrupted", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(
            f"[RESULT] FAIL - {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        if DEBUG:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
