#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Combine selected frames from one or more HDXF detector archives.

The output is a new, self-contained HDXF archive containing one derived frame.
Supported operations are raw sum, per-frame mean, and exposure-normalized rate.
Frame numbers accepted by the command line are 1-based, matching Albula.

This program intentionally preserves the original HDF5 metadata graph from the
first input archive. It updates frame-dependent metadata (nimages,
exposure/count_time, period/frame_time and other per-frame side variables) and
removes /entry/azint by default because an old azimuthal integration does not
describe the newly summed image.
"""

from __future__ import annotations

import argparse
import base64
import bisect
import gc
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from hdxf_codec import (
    HDXFBError,
    compress_array,
    decode_hdxfb,
    dtype_descriptor,
    logical_sha256,
    pack_hdxfb,
)
from hdxf_index import HDXFIError, index_sha256, pack_frame_index, unpack_frame_index

try:
    import psutil
except Exception:
    psutil = None


FORMAT_NAME = "HDF5-derived Detector eXchange Format"
FORMAT_ID = "hdxf"
FORMAT_VERSION = "0.4.5"
SUPPORTED_FORMAT_VERSIONS = {
    "0.3.0",
    "0.4.0",
    "0.4.2",
    "0.4.3",
    "0.4.4",
    "0.4.5",
}
MANIFEST_PATH = "manifest.json"
SUM_SIDEVAR_NAMES = {"exposure_time", "count_time", "period", "frame_time"}
NIMAGE_NAMES = {"nimages", "nimages_collected", "nimages_written"}


class HDXFSumError(RuntimeError):
    """Fatal input, decoding, selection or output error."""


OPERATION_TAGS = {
    "sum": "SUM",
    "mean": "MEAN",
    "exposure-normalized": "RATE",
}

EXPOSURE_PATHS = (
    "/entry/instrument/detector/exposure_time",
    "/entry/instrument/detector/count_time",
    "/entry/detector/exposure_time",
    "/entry/detector/count_time",
    "/entry/instrument/detector/detectorSpecific/exposure_time",
    "/entry/instrument/detector/detectorSpecific/count_time",
)



def _norm_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def _human_bytes(value: int | float) -> str:
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(number) < 1024.0 or unit == "TiB":
            return f"{number:.2f} {unit}"
        number /= 1024.0
    return f"{number:.2f} TiB"


def _human_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _rss_mb() -> float | None:
    if psutil is None:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0**2)


def _print_mem(tag: str, enabled: bool) -> None:
    if not enabled:
        return
    value = _rss_mb()
    if value is None:
        print(f"[MEM] {tag}: install psutil to report RSS", flush=True)
    else:
        print(f"[MEM] {tag}: RSS={value:.1f} MiB", flush=True)


def _strict_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    ).encode("utf-8")


def _decode_tagged_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_tagged_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    tag = value.get("$type")
    if tag == "bytes":
        return base64.b64decode(str(value.get("base64", "")))
    if tag == "float":
        token = str(value.get("value", "nan"))
        return {
            "nan": float("nan"),
            "+inf": float("inf"),
            "-inf": float("-inf"),
        }.get(token, float("nan"))
    if tag == "complex":
        return complex(
            _decode_tagged_value(value.get("real")),
            _decode_tagged_value(value.get("imag")),
        )
    if tag == "ndarray_npy":
        raw = base64.b64decode(str(value.get("base64", "")))
        return np.load(io.BytesIO(raw), allow_pickle=False)
    if tag == "object_array":
        shape = tuple(int(x) for x in value.get("shape", []))
        items = [_decode_tagged_value(item) for item in value.get("items", [])]
        return np.asarray(items, dtype=object).reshape(shape)
    return {str(k): _decode_tagged_value(v) for k, v in value.items()}


def _dtype_from_descriptor(value: Any) -> np.dtype[Any]:
    if isinstance(value, dict) and "numpy" in value:
        return _dtype_from_descriptor(value["numpy"])
    if isinstance(value, str):
        return np.dtype(value)
    if isinstance(value, list):
        fields = []
        for field in value:
            if not isinstance(field, list):
                raise HDXFSumError(f"Invalid dtype field: {field!r}")
            if len(field) == 2:
                fields.append((field[0], _dtype_from_descriptor(field[1])))
            elif len(field) == 3:
                fields.append(
                    (field[0], _dtype_from_descriptor(field[1]), tuple(field[2]))
                )
            else:
                raise HDXFSumError(f"Invalid dtype field length: {field!r}")
        return np.dtype(fields)
    raise HDXFSumError(f"Unsupported dtype descriptor: {value!r}")


def _hdf5_dtype_info(dtype: np.dtype[Any]) -> dict[str, Any]:
    dtype = np.dtype(dtype)
    info: dict[str, Any] = {
        "numpy": dtype_descriptor(dtype),
        "itemsize": int(dtype.itemsize),
        "kind": dtype.kind,
        "byteorder": dtype.byteorder,
    }
    return info


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value), allow_pickle=False)
    return buffer.getvalue()


def _payload_doc_for_npy(value: np.ndarray, payload_bytes: bytes) -> dict[str, Any]:
    digest = hashlib.sha256(payload_bytes).hexdigest()
    array = np.asarray(value)
    if array.shape == ():
        start = None
        count = None
    else:
        start = [0] * array.ndim
        count = list(array.shape)
    return {
        "path": f"payloads/{digest}.npy",
        "encoding": "npy",
        "start": start,
        "count": count,
        "chunk_shape": list(array.shape),
        "size_bytes": len(payload_bytes),
        "sha256": digest,
    }


def _encode_raw_frame_block(block: np.ndarray, *, clevel: int) -> tuple[bytes, dict[str, Any]]:
    contiguous = np.ascontiguousarray(block)
    compressed = compress_array(contiguous, filter_name="bitshuffle", clevel=clevel)
    segment = {
        "name": "frames",
        "codec": "blosc2-zstd",
        "filter": "bitshuffle",
        "dtype": dtype_descriptor(contiguous.dtype),
        "shape": list(contiguous.shape),
        "uncompressed_size": int(contiguous.nbytes),
        "compressed_size": len(compressed),
    }
    header = {
        "format": "hdxfb",
        "version": 1,
        "kind": "frame-block",
        "transform": "none",
        "logical_dtype": dtype_descriptor(contiguous.dtype),
        "logical_shape": list(contiguous.shape),
        "logical_sha256": logical_sha256(contiguous),
        "segments": [segment],
    }
    return pack_hdxfb(header, [compressed]), header


@dataclass
class InputPlan:
    reader: "HDXFReader"
    selected: list[int]
    suffix: str


class HDXFReader:
    """Low-memory reader for frame blocks and preserved HDF5 Dataset values."""

    def __init__(self, path: Path, *, verify_hashes: bool = True) -> None:
        self.path = path.resolve()
        self.verify_hashes = bool(verify_hashes)
        try:
            self.zip = zipfile.ZipFile(self.path, "r")
        except Exception as exc:
            raise HDXFSumError(f"Cannot open HDXF archive {self.path}: {exc}") from exc

        try:
            self.manifest = json.loads(self.zip.read(MANIFEST_PATH))
        except KeyError as exc:
            self.close()
            raise HDXFSumError(f"{self.path}: manifest.json is missing") from exc
        except Exception as exc:
            self.close()
            raise HDXFSumError(f"{self.path}: cannot parse manifest.json: {exc}") from exc

        identity = self.manifest.get("hdxf", {})
        if identity.get("format_id") != FORMAT_ID or identity.get("format") != FORMAT_NAME:
            self.close()
            raise HDXFSumError(f"Not an HDXF detector archive: {self.path}")
        version = str(identity.get("version", ""))
        if version not in SUPPORTED_FORMAT_VERSIONS:
            self.close()
            raise HDXFSumError(f"Unsupported HDXF version {version!r}: {self.path}")

        try:
            frames = self.manifest["detector_archive"]["frames"]
        except (KeyError, TypeError) as exc:
            self.close()
            raise HDXFSumError(f"{self.path}: detector_archive.frames is missing") from exc
        if not isinstance(frames, dict):
            self.close()
            raise HDXFSumError(f"{self.path}: detector_archive.frames is invalid")

        self.frames_doc = frames
        self.frame_count = int(frames.get("frame_count", 0))
        self.frame_shape = tuple(int(x) for x in frames.get("frame_shape", []))
        self.dtype = _dtype_from_descriptor(frames.get("dtype"))
        self.datasets = sorted(
            [item for item in frames.get("datasets", []) if isinstance(item, dict)],
            key=lambda item: int(item.get("global_start_frame", 0)),
        )
        self._dataset_by_object = {
            str(item.get("object", "")): item
            for item in self.datasets
            if item.get("object")
        }

        blocks = frames.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            blocks = self._load_binary_index(frames)
        self.blocks = sorted(
            [item for item in blocks if isinstance(item, dict)],
            key=lambda item: int(item.get("global_start_frame", 0)),
        )
        self.block_starts = [int(item.get("global_start_frame", 0)) for item in self.blocks]
        self._block_start_to_pos = {
            int(item.get("global_start_frame", 0)): pos
            for pos, item in enumerate(self.blocks)
        }

        for block in self.blocks:
            if not block.get("static_model_id"):
                dataset_doc = self._dataset_by_object.get(
                    str(block.get("source_object", ""))
                )
                if dataset_doc and dataset_doc.get("static_model_id"):
                    block["static_model_id"] = dataset_doc["static_model_id"]

        self._static_model_docs = {
            str(item.get("id")): item
            for item in frames.get("static_models", [])
            if isinstance(item, dict) and item.get("id")
        }
        self._static_model_cache: dict[str, dict[str, np.ndarray]] = {}
        self._dataset_value_cache: dict[str, Any] = {}
        self._object_by_id: dict[str, dict[str, Any]] = {}
        self._path_to_objects: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.main_source_file = str(
            self.manifest.get("source", {}).get("main_file", "")
        )
        objects = self.manifest.get("objects", {})
        if isinstance(objects, dict):
            for object_id, doc in objects.items():
                if not isinstance(doc, dict):
                    continue
                self._object_by_id[str(object_id)] = doc
                if doc.get("type") == "dataset":
                    path_value = str(doc.get("source_path") or "")
                    if path_value:
                        self._path_to_objects[path_value].append(doc)

        if self.frame_count <= 0:
            self.close()
            raise HDXFSumError(f"{self.path}: frame count is zero")
        if len(self.frame_shape) != 2:
            self.close()
            raise HDXFSumError(
                f"{self.path}: expected 2D detector frames; got {self.frame_shape}"
            )
        if not self.blocks or not self.datasets:
            self.close()
            raise HDXFSumError(f"{self.path}: frame block/index mapping is incomplete")

    def _load_binary_index(self, frames: dict[str, Any]) -> list[dict[str, Any]]:
        index_doc = frames.get("block_index")
        if not isinstance(index_doc, dict):
            raise HDXFSumError(f"{self.path}: no frame blocks or binary frame index")
        member_path = str(index_doc.get("path", ""))
        try:
            raw = self.zip.read(member_path)
        except KeyError as exc:
            raise HDXFSumError(f"{self.path}: missing frame index {member_path}") from exc
        expected_size = index_doc.get("size_bytes")
        if isinstance(expected_size, int) and len(raw) != expected_size:
            raise HDXFSumError(f"{self.path}: frame index size mismatch")
        expected_sha = index_doc.get("sha256")
        if self.verify_hashes and isinstance(expected_sha, str):
            actual = hashlib.sha256(raw).hexdigest()
            if actual != expected_sha:
                raise HDXFSumError(f"{self.path}: frame index SHA-256 mismatch")
        try:
            return unpack_frame_index(
                raw,
                datasets=list(frames.get("datasets", [])),
                frame_shape=list(self.frame_shape),
            )
        except HDXFIError as exc:
            raise HDXFSumError(f"{self.path}: invalid frame index: {exc}") from exc

    def close(self) -> None:
        self._static_model_cache.clear()
        self._dataset_value_cache.clear()
        self._object_by_id.clear()
        self._path_to_objects.clear()
        if getattr(self, "zip", None) is not None:
            self.zip.close()

    def __enter__(self) -> "HDXFReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def objects_for_path(self, path: str) -> list[dict[str, Any]]:
        return list(self._path_to_objects.get(path, []))

    def object_for_path(
        self, path: str, *, source_file: str | None = None
    ) -> dict[str, Any] | None:
        docs = self._path_to_objects.get(path, [])
        if source_file:
            for doc in docs:
                if str(doc.get("source_file") or "") == source_file:
                    return doc
        for doc in docs:
            if str(doc.get("source_file") or "") == self.main_source_file:
                return doc
        return docs[0] if docs else None

    def source_mapping_for_frame(
        self, global_index: int
    ) -> tuple[dict[str, Any], int, str]:
        for dataset_doc in self.datasets:
            start = int(dataset_doc.get("global_start_frame", 0))
            count = int(dataset_doc.get("frame_count", 0))
            if start <= global_index < start + count:
                object_doc = self._object_by_id.get(str(dataset_doc.get("object", "")), {})
                source_file = str(object_doc.get("source_file") or "")
                return dataset_doc, global_index - start, source_file
        raise HDXFSumError(
            f"{self.path}: no source Dataset mapping for frame {global_index + 1}"
        )

    def _load_static_model(self, model_id: str | None) -> dict[str, np.ndarray] | None:
        if not model_id:
            return None
        key = str(model_id)
        cached = self._static_model_cache.get(key)
        if cached is not None:
            return cached
        doc = self._static_model_docs.get(key)
        if doc is None:
            raise HDXFSumError(f"{self.path}: missing static model {key}")
        try:
            mask_doc = doc["mask"]
            values_doc = doc["static_values"]
            mask_path = str(mask_doc["path"])
            values_path = str(values_doc["path"])
            packed_mask, _ = decode_hdxfb(
                self.zip.read(mask_path),
                member_path=mask_path,
                verify_hash=self.verify_hashes,
            )
            static_values, _ = decode_hdxfb(
                self.zip.read(values_path),
                member_path=values_path,
                verify_hash=self.verify_hashes,
            )
        except (KeyError, HDXFBError) as exc:
            raise HDXFSumError(f"{self.path}: cannot load static model {key}: {exc}") from exc

        pixel_count = int(doc.get("pixel_count", np.prod(self.frame_shape)))
        bits = np.unpackbits(
            np.asarray(packed_mask, dtype=np.uint8).reshape(-1), bitorder="little"
        )
        if bits.size < pixel_count:
            raise HDXFSumError(f"{self.path}: truncated static model mask {key}")
        mask = bits[:pixel_count].astype(bool).reshape(self.frame_shape)
        values = np.ascontiguousarray(static_values).reshape(-1)
        if values.size != int(np.count_nonzero(mask)):
            raise HDXFSumError(f"{self.path}: static model value count mismatch {key}")
        result = {"mask": mask, "static_values": values}
        self._static_model_cache[key] = result
        return result

    def _decode_block(
        self,
        doc: dict[str, Any],
        *,
        previous_frame: np.ndarray | None,
    ) -> np.ndarray:
        member_path = str(doc.get("path", ""))
        if not member_path:
            raise HDXFSumError(f"{self.path}: frame block path is missing")
        try:
            raw = self.zip.read(member_path)
        except KeyError as exc:
            raise HDXFSumError(f"{self.path}: missing frame payload {member_path}") from exc
        expected_size = doc.get("size_bytes")
        if isinstance(expected_size, int) and len(raw) != expected_size:
            raise HDXFSumError(f"{self.path}: frame payload size mismatch {member_path}")
        expected_sha = doc.get("sha256")
        if self.verify_hashes and isinstance(expected_sha, str):
            actual = hashlib.sha256(raw).hexdigest()
            if actual != expected_sha:
                raise HDXFSumError(f"{self.path}: frame payload SHA-256 mismatch {member_path}")
        try:
            block, header = decode_hdxfb(
                raw,
                member_path=member_path,
                verify_hash=self.verify_hashes,
                previous_frame=previous_frame,
                static_model=self._load_static_model(doc.get("static_model_id")),
            )
        except HDXFBError as exc:
            raise HDXFSumError(f"{self.path}: {exc}") from exc
        if header.get("kind") != "frame-block":
            raise HDXFSumError(f"{self.path}: unsupported HDXFB kind in {member_path}")
        return block

    def _block_position_for_frame(self, frame_index: int) -> int:
        pos = bisect.bisect_right(self.block_starts, frame_index) - 1
        if pos < 0:
            raise HDXFSumError(f"{self.path}: no block covers frame {frame_index + 1}")
        doc = self.blocks[pos]
        start = int(doc.get("global_start_frame", 0))
        count = int(doc.get("frame_count", 0))
        if not start <= frame_index < start + count:
            raise HDXFSumError(f"{self.path}: no block covers frame {frame_index + 1}")
        return pos

    def iter_selected_blocks(
        self, selected_indices: Sequence[int]
    ) -> Iterator[tuple[dict[str, Any], np.ndarray, list[int]]]:
        """Decode only blocks required by selected frames, preserving Delta chains."""
        positions: dict[int, list[int]] = defaultdict(list)
        for index in selected_indices:
            pos = self._block_position_for_frame(int(index))
            block_start = int(self.blocks[pos].get("global_start_frame", 0))
            positions[pos].append(int(index) - block_start)

        last_pos: int | None = None
        previous_frame: np.ndarray | None = None
        for target_pos in sorted(positions):
            target = self.blocks[target_pos]
            target_requires_previous = bool(target.get("requires_previous_frame", False))

            if last_pos is not None and last_pos < target_pos:
                anchor_start = int(
                    target.get(
                        "chain_anchor_global_start_frame",
                        target.get("global_start_frame", 0),
                    )
                )
                anchor_pos = self._block_start_to_pos.get(anchor_start, target_pos)
                if anchor_pos <= last_pos:
                    decode_start = last_pos + 1
                else:
                    decode_start = anchor_pos if target_requires_previous else target_pos
                    previous_frame = None
            else:
                if target_requires_previous:
                    anchor_start = int(target.get("chain_anchor_global_start_frame", -1))
                    decode_start = self._block_start_to_pos.get(anchor_start, -1)
                    if decode_start < 0 or decode_start > target_pos:
                        raise HDXFSumError(
                            f"{self.path}: invalid Delta chain anchor at frame "
                            f"{int(target.get('global_start_frame', 0)) + 1}"
                        )
                else:
                    decode_start = target_pos
                previous_frame = None

            target_block: np.ndarray | None = None
            for pos in range(decode_start, target_pos + 1):
                doc = self.blocks[pos]
                block = self._decode_block(doc, previous_frame=previous_frame)
                previous_frame = np.ascontiguousarray(block[-1])
                if pos == target_pos:
                    target_block = block
                else:
                    del block
            if target_block is None:
                raise HDXFSumError(f"{self.path}: internal block decode failure")
            yield target, target_block, sorted(set(positions[target_pos]))
            last_pos = target_pos

    def _decode_generic_member(self, member: dict[str, Any]) -> Any:
        member_path = str(member.get("path") or "")
        encoding = str(member.get("encoding") or "")
        try:
            raw = self.zip.read(member_path)
        except KeyError as exc:
            raise HDXFSumError(f"{self.path}: missing metadata payload {member_path}") from exc
        expected = member.get("sha256")
        if self.verify_hashes and isinstance(expected, str):
            actual = hashlib.sha256(raw).hexdigest()
            if actual != expected:
                raise HDXFSumError(
                    f"{self.path}: metadata payload SHA-256 mismatch {member_path}"
                )
        if encoding == "npy":
            return np.load(io.BytesIO(raw), allow_pickle=False)
        if encoding == "json":
            doc = json.loads(raw.decode("utf-8"))
            return _decode_tagged_value(doc.get("data"))
        if encoding == "array-block-v1":
            try:
                value, _ = decode_hdxfb(
                    raw,
                    member_path=member_path,
                    verify_hash=self.verify_hashes,
                )
            except HDXFBError as exc:
                raise HDXFSumError(f"{self.path}: {exc}") from exc
            return value
        raise HDXFSumError(
            f"{self.path}: unsupported metadata payload encoding {encoding!r}"
        )

    def read_object_dataset(self, obj: dict[str, Any]) -> Any | None:
        cache_key = str(obj.get("id") or id(obj))
        if cache_key in self._dataset_value_cache:
            return self._dataset_value_cache[cache_key]
        payload = obj.get("payload")
        if not isinstance(payload, dict) or payload.get("state") not in (None, "present"):
            return None
        encoding = str(payload.get("encoding") or "")
        if encoding in (
            "frame-sequence",
            "frame-sequence-indexed",
            "calibration-raw-chunks-v1",
        ):
            return None
        members = payload.get("members")
        if not isinstance(members, list) or not members:
            return None
        parts = [
            self._decode_generic_member(member)
            for member in members
            if isinstance(member, dict)
        ]
        if not parts:
            return None
        if len(parts) == 1:
            value = parts[0]
        else:
            try:
                value = np.concatenate([np.asarray(item) for item in parts], axis=0)
            except Exception:
                value = parts
        try:
            if np.asarray(value).nbytes <= 32 * 1024 * 1024:
                self._dataset_value_cache[cache_key] = value
        except Exception:
            pass
        return value

    def read_dataset(
        self, path: str, *, source_file: str | None = None
    ) -> Any | None:
        obj = self.object_for_path(path, source_file=source_file)
        return None if obj is None else self.read_object_dataset(obj)

    def metadata_value_for_frame(self, path: str, global_index: int) -> Any | None:
        dataset_doc, local_index, frame_source_file = self.source_mapping_for_frame(global_index)
        candidates = self.objects_for_path(path)
        if not candidates:
            return None
        ordered: list[dict[str, Any]] = []
        for source_file in (frame_source_file, self.main_source_file):
            if not source_file:
                continue
            ordered.extend(
                doc
                for doc in candidates
                if str(doc.get("source_file") or "") == source_file and doc not in ordered
            )
        ordered.extend(doc for doc in candidates if doc not in ordered)

        dataset_frame_count = int(dataset_doc.get("frame_count", 0))
        for obj in ordered:
            value = self.read_object_dataset(obj)
            if value is None:
                continue
            array = np.asarray(value)
            if array.shape == ():
                return array.reshape(())
            if array.size == 1:
                return array.reshape(-1)[0]
            if array.ndim >= 1:
                source_file = str(obj.get("source_file") or "")
                if source_file == frame_source_file and array.shape[0] == dataset_frame_count:
                    return array[local_index]
                if array.shape[0] == self.frame_count:
                    return array[global_index]
                if array.shape[0] == 1:
                    return array[0]
        return None


def _discover_inputs(
    cmdline_inputs: Sequence[Path] | None,
    input_dir: Path | None,
    recursive: bool,
    pattern: str,
) -> list[Path]:
    paths: list[Path] = []
    if cmdline_inputs:
        paths.extend(path.resolve() for path in cmdline_inputs)
    if input_dir is not None:
        root = input_dir.resolve()
        iterator = root.rglob(pattern) if recursive else root.glob(pattern)
        paths.extend(path.resolve() for path in iterator if path.is_file())
    return sorted(set(paths))


def _parse_selection(
    *,
    total_frames: int,
    select_text: str | None,
    frame_from: int | None,
    frame_to: int | None,
) -> tuple[list[int], str]:
    if select_text:
        try:
            one_based = [int(item.strip()) for item in select_text.split(",") if item.strip()]
        except ValueError as exc:
            raise HDXFSumError(f"Cannot parse selected frames: {select_text!r}") from exc
        invalid = [value for value in one_based if value < 1 or value > total_frames]
        if invalid:
            raise HDXFSumError(
                f"Selected frame numbers outside 1..{total_frames}: {invalid[:10]}"
            )
        seen: set[int] = set()
        unique = [value for value in one_based if not (value in seen or seen.add(value))]
        return [value - 1 for value in unique], "SEL_" + "-".join(map(str, unique))

    start = 1 if frame_from is None else int(frame_from)
    end = total_frames if frame_to is None else int(frame_to)
    if start < 1 or end < start or end > total_frames:
        raise HDXFSumError(f"Invalid frame range {start}..{end}; valid range is 1..{total_frames}")
    suffix = "ALL" if frame_from is None and frame_to is None else f"{start}-{end}"
    return list(range(start - 1, end)), suffix


def _build_input_plans(readers: list[HDXFReader], args: argparse.Namespace) -> tuple[list[InputPlan], str]:
    starts = [0]
    for reader in readers:
        starts.append(starts[-1] + reader.frame_count)
    total_all = starts[-1]

    have_per_input_override = any(
        value is not None
        for value in (
            args.select_frame1,
            args.frame_from1,
            args.frame_to1,
            args.select_frame2,
            args.frame_from2,
            args.frame_to2,
        )
    )
    use_global = (
        len(readers) > 1
        and any(value is not None for value in (args.select_frame, args.frame_from, args.frame_to))
        and not have_per_input_override
    )

    if use_global:
        global_selected, global_suffix = _parse_selection(
            total_frames=total_all,
            select_text=args.select_frame,
            frame_from=args.frame_from,
            frame_to=args.frame_to,
        )
        per_input: list[list[int]] = [[] for _ in readers]
        for global_index in global_selected:
            position = bisect.bisect_right(starts, global_index) - 1
            per_input[position].append(global_index - starts[position])
        return [
            InputPlan(reader=reader, selected=sorted(set(selected)), suffix=global_suffix)
            for reader, selected in zip(readers, per_input)
        ], global_suffix

    plans: list[InputPlan] = []
    suffix_parts: list[str] = []
    for index, reader in enumerate(readers):
        if index == 0 and any(
            value is not None
            for value in (args.select_frame1, args.frame_from1, args.frame_to1)
        ):
            select_text, frame_from, frame_to = (
                args.select_frame1,
                args.frame_from1,
                args.frame_to1,
            )
        elif index == 1 and any(
            value is not None
            for value in (args.select_frame2, args.frame_from2, args.frame_to2)
        ):
            select_text, frame_from, frame_to = (
                args.select_frame2,
                args.frame_from2,
                args.frame_to2,
            )
        else:
            select_text, frame_from, frame_to = (
                args.select_frame,
                args.frame_from,
                args.frame_to,
            )
        selected, suffix = _parse_selection(
            total_frames=reader.frame_count,
            select_text=select_text,
            frame_from=frame_from,
            frame_to=frame_to,
        )
        plans.append(InputPlan(reader=reader, selected=selected, suffix=suffix))
        suffix_parts.append(f"M{index + 1}_{suffix}" if len(readers) > 1 else suffix)
    return plans, "__".join(suffix_parts)


def _parse_output_dtype(value: str) -> np.dtype[Any] | str:
    text = value.strip().lower()
    if text in ("auto", "source"):
        return text
    mapping = {
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "float32": np.float32,
        "float64": np.float64,
    }
    if text not in mapping:
        raise HDXFSumError(
            "--out-dtype must be auto,source,int16,int32,int64,uint16,uint32,"
            "uint64,float32 or float64"
        )
    return np.dtype(mapping[text])


def _choose_safe_integer_dtype(input_dtype: np.dtype[Any], frames: int) -> np.dtype[Any]:
    info = np.iinfo(input_dtype)
    signed = np.issubdtype(input_dtype, np.signedinteger)
    minimum = int(info.min) * frames if signed else 0
    maximum = int(info.max) * frames
    candidates = (
        (np.int16, np.int32, np.int64)
        if signed
        else (np.uint16, np.uint32, np.uint64)
    )
    for candidate in candidates:
        candidate_info = np.iinfo(candidate)
        if minimum >= int(candidate_info.min) and maximum <= int(candidate_info.max):
            return np.dtype(candidate)
    return np.dtype(np.float64)


def _choose_acc_dtype(input_dtype: np.dtype[Any], frames: int) -> np.dtype[Any]:
    """Use a wide accumulator so a long acquisition cannot overflow mid-sum."""
    if np.issubdtype(input_dtype, np.signedinteger):
        return np.dtype(np.int64)
    if np.issubdtype(input_dtype, np.unsignedinteger):
        return np.dtype(np.uint64)
    return np.dtype(np.float64)


def _choose_output_dtype(
    requested: str,
    input_dtype: np.dtype[Any],
    frames: int,
    operation: str,
) -> tuple[np.dtype[Any], str]:
    parsed = _parse_output_dtype(requested)
    if parsed == "source":
        return np.dtype(input_dtype), "source"
    if parsed == "auto":
        if operation == "sum":
            if np.issubdtype(input_dtype, np.signedinteger):
                result = np.dtype(np.int64)
            elif np.issubdtype(input_dtype, np.unsignedinteger):
                result = np.dtype(np.uint64)
            else:
                result = np.dtype(np.float64)
        else:
            # Mean/rate generally contain fractions. A one-frame float64 output is
            # small, and preserves the accumulated value better than float32.
            result = np.dtype(np.float64)
        return result, f"auto for {operation} from {np.dtype(input_dtype).name} × {frames}"
    assert isinstance(parsed, np.dtype)
    return parsed, requested


def _safe_cast_output(array: np.ndarray, dtype: np.dtype[Any]) -> tuple[np.ndarray, int]:
    source = np.asarray(array)
    clipped = 0
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        below = source < info.min
        above = source > info.max
        clipped = int(np.count_nonzero(below) + np.count_nonzero(above))
        return np.clip(source, info.min, info.max).astype(dtype, copy=False), clipped
    return source.astype(dtype, copy=False), clipped


def _metadata_scalar_total(
    plans: Sequence[InputPlan], paths: Sequence[str]
) -> tuple[str | None, float | None]:
    """Return the first usable selected-frame metadata total from candidate paths."""
    for path in paths:
        value = _sum_metadata_path(plans, path)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float64)
        if array.size != 1:
            continue
        scalar = float(array.reshape(-1)[0])
        if math.isfinite(scalar):
            return path, scalar
    return None, None


def _apply_operation(
    accumulator: np.ndarray,
    *,
    operation: str,
    frame_count: int,
    exposure_total: float | None,
) -> tuple[np.ndarray, str]:
    source = np.asarray(accumulator)
    if operation == "sum":
        return source, "raw summed counts"
    if operation == "mean":
        return source.astype(np.float64, copy=False) / float(frame_count), (
            f"per-frame mean over {frame_count} selected frames"
        )
    if operation == "exposure-normalized":
        if exposure_total is None or not math.isfinite(exposure_total) or exposure_total <= 0:
            raise HDXFSumError(
                "Exposure-normalized output requires a positive exposure_time/count_time "
                "stored in the HDXF metadata"
            )
        return source.astype(np.float64, copy=False) / float(exposure_total), (
            f"exposure-normalized rate using {exposure_total:.12g} s (counts/s)"
        )
    raise HDXFSumError(f"Unsupported operation: {operation}")


def _storage_shape(doc: dict[str, Any]) -> tuple[int, ...] | None:
    storage = doc.get("hdf5_storage")
    if not isinstance(storage, dict):
        return None
    shape = storage.get("shape")
    if not isinstance(shape, list):
        return None
    try:
        return tuple(int(value) for value in shape)
    except Exception:
        return None


def _original_dataset_dtype(doc: dict[str, Any]) -> np.dtype[Any] | None:
    storage = doc.get("hdf5_storage")
    if not isinstance(storage, dict):
        return None
    try:
        return _dtype_from_descriptor(storage.get("dtype"))
    except Exception:
        return None


def _rewrite_dataset_payload(
    obj: dict[str, Any],
    value: np.ndarray,
    payloads: dict[str, bytes],
) -> None:
    array = np.asarray(value)
    raw = _npy_bytes(array)
    member = _payload_doc_for_npy(array, raw)
    payloads[member["path"]] = raw
    old_payload = obj.get("payload")
    target_chunk_bytes = (
        old_payload.get("target_chunk_bytes", 16 * 1024 * 1024)
        if isinstance(old_payload, dict)
        else 16 * 1024 * 1024
    )
    obj["payload"] = {
        "encoding": "chunked",
        "target_chunk_bytes": int(target_chunk_bytes),
        "members": [member],
        "state": "present",
    }
    storage = obj.setdefault("hdf5_storage", {})
    storage["shape"] = list(array.shape)
    storage["maxshape"] = list(array.shape)
    storage["dtype"] = _hdf5_dtype_info(array.dtype)
    if array.shape == ():
        storage["chunks"] = None
    else:
        old_chunks = storage.get("chunks")
        if isinstance(old_chunks, list) and len(old_chunks) == array.ndim:
            chunks = [max(1, min(int(chunk), int(size))) for chunk, size in zip(old_chunks, array.shape)]
            storage["chunks"] = chunks
        else:
            storage["chunks"] = None
    storage["compression"] = None
    storage["compression_options"] = None
    storage["filters"] = []
    storage["shuffle"] = False
    storage["fletcher32"] = False
    storage["scaleoffset"] = None
    storage["external_storage"] = []


def _sum_metadata_path(plans: Sequence[InputPlan], path: str) -> np.ndarray | None:
    total: np.ndarray | None = None
    floating = False
    for plan in plans:
        for global_index in plan.selected:
            value = plan.reader.metadata_value_for_frame(path, global_index)
            if value is None:
                continue
            array = np.asarray(value)
            if array.dtype.kind in "fc":
                contribution = array.astype(np.float64, copy=False)
                floating = True
            elif array.dtype.kind in "iub":
                contribution = array.astype(np.int64, copy=False)
            else:
                continue
            if total is None:
                total = np.asarray(contribution).copy()
            else:
                total = np.asarray(total) + contribution
    if total is None:
        return None
    return np.asarray(total, dtype=np.float64) if floating else np.asarray(total)


def _contains_path(value: Any, prefix: str) -> bool:
    if isinstance(value, str):
        return value == prefix or value.startswith(prefix.rstrip("/") + "/")
    if isinstance(value, list):
        return any(_contains_path(item, prefix) for item in value)
    if isinstance(value, dict):
        return any(_contains_path(item, prefix) for item in value.values())
    return False


def _remove_object_subtree(manifest: dict[str, Any], prefix: str) -> set[str]:
    objects = manifest.get("objects")
    if not isinstance(objects, dict):
        return set()
    remove_ids = {
        str(object_id)
        for object_id, doc in objects.items()
        if isinstance(doc, dict)
        and (
            str(doc.get("source_path") or "") == prefix
            or str(doc.get("source_path") or "").startswith(prefix.rstrip("/") + "/")
        )
    }
    if not remove_ids:
        return set()
    for object_id in remove_ids:
        objects.pop(object_id, None)
    for doc in objects.values():
        if isinstance(doc, dict) and isinstance(doc.get("children"), list):
            doc["children"] = [
                child
                for child in doc["children"]
                if not (isinstance(child, dict) and str(child.get("object")) in remove_ids)
            ]
    links = manifest.get("links")
    if isinstance(links, list):
        manifest["links"] = [link for link in links if not _contains_path(link, prefix)]
    return remove_ids


def _prepare_output_manifest(
    plans: Sequence[InputPlan],
    *,
    output_dtype: np.dtype[Any],
    output_frame_payload: bytes,
    output_frame_header: dict[str, Any],
    keep_azint: bool,
    operation: str,
    selected_frame_count: int,
) -> tuple[dict[str, Any], dict[str, bytes], set[str]]:
    first = plans[0].reader
    manifest = json.loads(json.dumps(first.manifest))
    manifest.setdefault("hdxf", {})["version"] = FORMAT_VERSION
    objects = manifest.get("objects")
    if not isinstance(objects, dict):
        raise HDXFSumError("First input manifest has no HDF5 object graph")

    old_frames = manifest["detector_archive"]["frames"]
    old_datasets = [item for item in old_frames.get("datasets", []) if isinstance(item, dict)]
    if not old_datasets:
        raise HDXFSumError("First input has no frame Dataset mapping")
    chosen_dataset = old_datasets[0]
    if plans[0].selected:
        first_selected_index = plans[0].selected[0]
        for candidate in old_datasets:
            start = int(candidate.get("global_start_frame", 0))
            count = int(candidate.get("frame_count", 0))
            if start <= first_selected_index < start + count:
                chosen_dataset = candidate
                break
    chosen_object_id = str(chosen_dataset.get("object", ""))
    chosen_object = objects.get(chosen_object_id)
    if not isinstance(chosen_object, dict):
        raise HDXFSumError("First frame Dataset object is missing from HDF5 object graph")

    old_frame_object_ids = {
        str(item.get("object")) for item in old_datasets if item.get("object")
    }
    removed_frame_ids = old_frame_object_ids - {chosen_object_id}
    for object_id in removed_frame_ids:
        objects.pop(object_id, None)
    for doc in objects.values():
        if isinstance(doc, dict) and isinstance(doc.get("children"), list):
            doc["children"] = [
                child
                for child in doc["children"]
                if not (
                    isinstance(child, dict)
                    and str(child.get("object")) in removed_frame_ids
                )
            ]
    links = manifest.get("links")
    if isinstance(links, list):
        manifest["links"] = [
            link
            for link in links
            if not (
                isinstance(link, dict)
                and str(link.get("target_object")) in removed_frame_ids
            )
        ]

    height, width = first.frame_shape
    frame_digest = hashlib.sha256(output_frame_payload).hexdigest()
    frame_member_path = f"payloads/{frame_digest}.hdxfb"
    frame_block = {
        "path": frame_member_path,
        "encoding": f"frame-block-v{int(output_frame_header.get('version', 1))}",
        "size_bytes": len(output_frame_payload),
        "sha256": frame_digest,
        "start": [0, 0, 0],
        "count": [1, height, width],
        "chunk_shape": [1, height, width],
        "start_frame": 0,
        "global_start_frame": 0,
        "frame_count": 1,
        "hdxfb_version": int(output_frame_header.get("version", 1)),
        "requires_previous_frame": False,
        "chain_mode": "anchor",
        "chain_anchor_global_start_frame": 0,
        "source_object": chosen_object_id,
    }
    index_raw = pack_frame_index(
        [frame_block], dataset_ordinals={chosen_object_id: 0}, compress=True
    )
    index_digest = index_sha256(index_raw)
    index_path = f"payloads/{index_digest}.hdxfi"

    old_frames.clear()
    old_frames.update(
        {
            "type": "frame_sequence",
            "frame_count": 1,
            "frame_shape": [height, width],
            "dtype": dtype_descriptor(output_dtype),
            "block_frames": 1,
            "tile_shape": [height, width],
            "datasets": [
                {
                    "object": chosen_object_id,
                    "source_path": str(chosen_dataset.get("source_path") or chosen_object.get("source_path") or "/entry/data/data"),
                    "global_start_frame": 0,
                    "frame_count": 1,
                    "static_model_id": None,
                    "block_index_start": 0,
                    "block_index_count": 1,
                }
            ],
            "keyframe_interval_blocks": 1,
            "static_models": [],
            "shape": [1, height, width],
            "block_index": {
                "path": index_path,
                "encoding": "hdxfi-v1",
                "record_count": 1,
                "size_bytes": len(index_raw),
                "sha256": index_digest,
            },
        }
    )

    storage = chosen_object.setdefault("hdf5_storage", {})
    storage["shape"] = [1, height, width]
    storage["maxshape"] = [1, height, width]
    storage["dtype"] = _hdf5_dtype_info(output_dtype)
    old_chunks = storage.get("chunks")
    if isinstance(old_chunks, list) and len(old_chunks) == 3:
        storage["chunks"] = [1, min(height, int(old_chunks[1])), min(width, int(old_chunks[2]))]
    else:
        storage["chunks"] = [1, height, width]
    storage["external_storage"] = []
    chosen_object["payload"] = {
        "encoding": "frame-sequence-indexed",
        "target_chunk_bytes": 16 * 1024 * 1024,
        "state": "present",
        "block_frames": 1,
        "global_start_frame": 0,
        "block_index_start": 0,
        "block_index_count": 1,
    }
    attrs = chosen_object.setdefault("attributes", {})
    if isinstance(attrs, dict):
        for name in ("image_nr_low", "image_nr_high", "nimages"):
            if name in attrs:
                attrs[name] = 1

    if not keep_azint:
        _remove_object_subtree(manifest, "/entry/azint")

    new_payloads: dict[str, bytes] = {
        frame_member_path: output_frame_payload,
        index_path: index_raw,
    }

    first_selected = plans[0].selected
    representative_global = first_selected[0] if first_selected else 0
    chosen_start = int(chosen_dataset.get("global_start_frame", 0))
    chosen_count = int(chosen_dataset.get("frame_count", 0))
    representative_local = min(
        max(0, representative_global - chosen_start), max(0, chosen_count - 1)
    )
    chosen_source_file = str(chosen_object.get("source_file") or "")
    for obj in list(objects.values()):
        if not isinstance(obj, dict) or obj.get("type") != "dataset":
            continue
        object_id = str(obj.get("id") or "")
        if object_id == chosen_object_id:
            continue
        source_path = str(obj.get("source_path") or "")
        if not source_path:
            continue
        name = _norm_name(source_path.rsplit("/", 1)[-1])
        shape = _storage_shape(obj)
        original_dtype = _original_dataset_dtype(obj)

        if name in SUM_SIDEVAR_NAMES:
            total = _sum_metadata_path(plans, source_path)
            if total is None:
                continue
            adjusted = np.asarray(total)
            if operation == "mean":
                adjusted = adjusted.astype(np.float64, copy=False) / float(selected_frame_count)
            elif operation == "exposure-normalized":
                # A rate image is expressed per one second. Keep time fields
                # semantically aligned with the generated image.
                adjusted = np.ones_like(adjusted, dtype=np.float64)
            if shape == ():
                output_value = np.asarray(adjusted).reshape(())
            elif shape:
                output_value = np.asarray(adjusted)[np.newaxis, ...]
            else:
                output_value = np.asarray(adjusted)
            if operation == "sum" and original_dtype is not None:
                try:
                    output_value = output_value.astype(original_dtype, copy=False)
                except Exception:
                    output_value = output_value.astype(np.float64, copy=False)
            else:
                output_value = output_value.astype(np.float64, copy=False)
            _rewrite_dataset_payload(obj, output_value, new_payloads)
            continue

        if name in NIMAGE_NAMES and (
            source_path.startswith("/entry/instrument/detector/")
            or source_path.startswith("/entry/detector/")
        ):
            dtype = original_dtype or np.dtype(np.int64)
            _rewrite_dataset_payload(obj, np.asarray(1, dtype=dtype), new_payloads)
            continue

        if shape and len(shape) >= 1:
            source_file = str(obj.get("source_file") or "")
            expected_frames: int | None = None
            representative_index: int | None = None
            if source_file == first.main_source_file and shape[0] == first.frame_count:
                expected_frames = first.frame_count
                representative_index = representative_global
            elif source_file == chosen_source_file and shape[0] == chosen_count:
                expected_frames = chosen_count
                representative_index = representative_local
            if expected_frames is None or representative_index is None:
                continue
            value = first.read_object_dataset(obj)
            if value is None:
                continue
            array = np.asarray(value)
            if array.ndim < 1 or array.shape[0] != expected_frames:
                continue
            output_value = array[representative_index : representative_index + 1]
            _rewrite_dataset_payload(obj, output_value, new_payloads)

    old_excluded: set[str] = {MANIFEST_PATH}
    old_excluded.update(str(block.get("path")) for block in first.blocks)
    old_index = first.frames_doc.get("block_index")
    if isinstance(old_index, dict) and old_index.get("path"):
        old_excluded.add(str(old_index["path"]))
    for model in first.frames_doc.get("static_models", []):
        if not isinstance(model, dict):
            continue
        for key in ("mask", "static_values"):
            member = model.get(key)
            if isinstance(member, dict) and member.get("path"):
                old_excluded.add(str(member["path"]))

    return manifest, new_payloads, old_excluded


def _referenced_payload_paths(value: Any) -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "path" and isinstance(item, str) and item.startswith("payloads/"):
                paths.add(item)
            else:
                paths.update(_referenced_payload_paths(item))
    elif isinstance(value, list):
        for item in value:
            paths.update(_referenced_payload_paths(item))
    return paths


def _write_output_archive(
    output_path: Path,
    *,
    first_reader: HDXFReader,
    manifest: dict[str, Any],
    new_payloads: dict[str, bytes],
    excluded_members: set[str],
    overwrite: bool,
) -> None:
    output_path = output_path.resolve()
    if output_path.exists() and not overwrite:
        raise HDXFSumError(f"Output already exists: {output_path} (use --overwrite)")
    if output_path == first_reader.path:
        raise HDXFSumError("Output path must differ from every input path")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix=output_path.name + ".", suffix=".tmp", dir=output_path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        referenced = _referenced_payload_paths(manifest)
        with zipfile.ZipFile(
            temp_path,
            "w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as dst:
            for info in first_reader.zip.infolist():
                if (
                    info.filename not in referenced
                    or info.filename in excluded_members
                    or info.filename in new_payloads
                ):
                    continue
                with first_reader.zip.open(info, "r") as src, dst.open(info, "w") as out:
                    shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
            for member_path, raw in sorted(new_payloads.items()):
                dst.writestr(member_path, raw)
            dst.writestr(MANIFEST_PATH, _strict_json_bytes(manifest, pretty=False))
        os.replace(temp_path, output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine selected frames from one or more HDXF archives and write one "
            "single-frame HDXF archive. Operations: sum, mean, or exposure-normalized. "
            "Frame numbers are 1-based."
        )
    )
    parser.add_argument("-i", "--input", type=Path, nargs="+", help="Input .hdxf file(s)")
    parser.add_argument("-d", "--input-dir", type=Path, default=None, help="Directory to scan for HDXF files")
    parser.add_argument("--recursive", action="store_true", help="Scan input directory recursively")
    parser.add_argument("--pattern", default="*.hdxf", help="Input-directory glob pattern (default: *.hdxf)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output .hdxf path, or an output directory for automatic naming",
    )
    parser.add_argument("--out-prefix", default=None, help="Filename prefix when --output is a directory")

    parser.add_argument("--frame-from", "--frame_from", dest="frame_from", type=int, default=None)
    parser.add_argument("--frame-to", "--frame_to", dest="frame_to", type=int, default=None)
    parser.add_argument("--select-frame", dest="select_frame", default=None, help="Comma-separated 1-based frame numbers")
    parser.add_argument("--frame-from1", "--frame_from1", dest="frame_from1", type=int, default=None)
    parser.add_argument("--frame-to1", "--frame_to1", dest="frame_to1", type=int, default=None)
    parser.add_argument("--select-frame1", dest="select_frame1", default=None)
    parser.add_argument("--frame-from2", "--frame_from2", dest="frame_from2", type=int, default=None)
    parser.add_argument("--frame-to2", "--frame_to2", dest="frame_to2", type=int, default=None)
    parser.add_argument("--select-frame2", dest="select_frame2", default=None)

    parser.add_argument(
        "--operation",
        choices=("sum", "mean", "exposure-normalized"),
        default="sum",
        help=(
            "Output operation. sum keeps total counts; mean divides by selected frame "
            "count; exposure-normalized divides by summed exposure/count_time and "
            "outputs counts/s. Default: sum"
        ),
    )
    parser.add_argument(
        "--out-dtype",
        default="auto",
        help=(
            "Output dtype: auto,source,int16,int32,int64,uint16,uint32,uint64,"
            "float32,float64. Default: auto"
        ),
    )
    parser.add_argument("--zstd-level", type=int, default=3, choices=range(0, 10), metavar="0..9")
    parser.add_argument("--progress-step", type=float, default=5.0, help="Progress print interval in percent")
    parser.add_argument("--tmp-dir", type=Path, default=None, help="Directory for the accumulator memmap")
    parser.add_argument("--mem", action="store_true", help="Print process RSS when psutil is installed")
    parser.add_argument("--skip-hash-check", action="store_true", help="Skip input payload SHA-256 verification")
    parser.add_argument("--keep-azint", action="store_true", help="Keep /entry/azint from the first input")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_paths = _discover_inputs(args.input, args.input_dir, args.recursive, args.pattern)
    if not input_paths:
        raise HDXFSumError("No HDXF input files found. Use -i/--input or --input-dir.")

    readers: list[HDXFReader] = []
    accumulator_path: Path | None = None
    accumulator: np.memmap | None = None
    try:
        print(f"[DISCOVER] archives = {len(input_paths)}", flush=True)
        for path in input_paths:
            if not path.is_file():
                raise HDXFSumError(f"Input is not a file: {path}")
            reader = HDXFReader(path, verify_hashes=not args.skip_hash_check)
            readers.append(reader)
            print(
                f"[INPUT] {path.name}: frames={reader.frame_count}, "
                f"shape={reader.frame_shape}, dtype={reader.dtype.name}",
                flush=True,
            )

        first = readers[0]
        for reader in readers[1:]:
            if reader.frame_shape != first.frame_shape:
                raise HDXFSumError(
                    f"Frame shape mismatch: {reader.path.name} has {reader.frame_shape}, "
                    f"expected {first.frame_shape}"
                )
            if reader.dtype != first.dtype:
                raise HDXFSumError(
                    f"Frame dtype mismatch: {reader.path.name} has {reader.dtype}, "
                    f"expected {first.dtype}"
                )

        plans, selection_suffix = _build_input_plans(readers, args)
        total_selected = sum(len(plan.selected) for plan in plans)
        if total_selected <= 0:
            raise HDXFSumError("Selected frame set is empty")
        for index, plan in enumerate(plans, start=1):
            if plan.selected:
                print(
                    f"[SELECT M{index}] {len(plan.selected)} frames "
                    f"(first={plan.selected[0] + 1}, last={plan.selected[-1] + 1})",
                    flush=True,
                )
            else:
                print(f"[SELECT M{index}] 0 frames", flush=True)

        acc_dtype = _choose_acc_dtype(first.dtype, total_selected)
        output_dtype, dtype_note = _choose_output_dtype(
            args.out_dtype, first.dtype, total_selected, args.operation
        )
        print(
            f"[DTYPE] accumulator={acc_dtype.name}, output={output_dtype.name} "
            f"({dtype_note})",
            flush=True,
        )
        if args.operation != "sum" and np.issubdtype(output_dtype, np.integer):
            print(
                f"[WARN] {args.operation} can contain fractional values, but output "
                f"dtype is {output_dtype.name}; use --out-dtype auto or float64 to "
                "preserve fractions",
                flush=True,
            )

        output_arg = args.output.resolve()
        if output_arg.suffix.lower() == ".hdxf":
            output_path = output_arg
        else:
            output_arg.mkdir(parents=True, exist_ok=True)
            base = args.out_prefix.strip() if args.out_prefix and args.out_prefix.strip() else readers[0].path.stem
            multi = f"_X{len(readers)}" if len(readers) > 1 else ""
            operation_tag = OPERATION_TAGS[args.operation]
            output_path = output_arg / f"{base}_{operation_tag}_{selection_suffix}{multi}.hdxf"
        if any(output_path.resolve() == reader.path for reader in readers):
            raise HDXFSumError("Output path must differ from every input path")

        tmp_dir = (args.tmp_dir or output_path.parent).resolve()
        tmp_dir.mkdir(parents=True, exist_ok=True)
        accumulator_path = tmp_dir / f".{output_path.stem}.sum.acc.tmp"
        height, width = first.frame_shape
        accumulator = np.memmap(
            accumulator_path,
            dtype=acc_dtype,
            mode="w+",
            shape=(height, width),
        )
        accumulator[:] = 0
        _print_mem("after accumulator allocation", args.mem)

        processed = 0
        progress_step = max(0.1, float(args.progress_step))
        next_progress = 0.0
        started = time.perf_counter()
        last_flush_frames = 0
        last_flush_time = started

        def report_progress(force: bool = False) -> None:
            nonlocal next_progress
            percent = processed / total_selected * 100.0
            if not force and percent + 1e-9 < next_progress:
                return
            elapsed = time.perf_counter() - started
            eta = (
                elapsed / processed * (total_selected - processed)
                if processed > 0
                else 0.0
            )
            print(
                f"[SUM] {processed}/{total_selected} ({percent:.1f}%) "
                f"elapsed={_human_duration(elapsed)} ETA={_human_duration(eta)}",
                flush=True,
            )
            next_progress = percent + progress_step

        report_progress(force=True)
        for plan in plans:
            if not plan.selected:
                continue
            for block_doc, block, local_indices in plan.reader.iter_selected_blocks(plan.selected):
                selected_block = block[np.asarray(local_indices, dtype=np.int64)]
                block_sum = selected_block.sum(axis=0, dtype=acc_dtype)
                np.add(accumulator, block_sum, out=accumulator, casting="unsafe")
                processed += len(local_indices)
                del selected_block, block_sum, block
                now = time.perf_counter()
                if processed - last_flush_frames >= 500 or now - last_flush_time >= 30.0:
                    accumulator.flush()
                    last_flush_frames = processed
                    last_flush_time = now
                report_progress()
        accumulator.flush()
        report_progress(force=True)
        print(f"[TIME] summation completed in {time.perf_counter() - started:.2f}s", flush=True)

        exposure_path, exposure_total = _metadata_scalar_total(plans, EXPOSURE_PATHS)
        if exposure_total is not None:
            print(
                f"[EXPOSURE] selected total={exposure_total:.12g} s from {exposure_path}",
                flush=True,
            )
        operation_frame, operation_note = _apply_operation(
            np.asarray(accumulator),
            operation=args.operation,
            frame_count=total_selected,
            exposure_total=exposure_total,
        )
        print(f"[OPERATION] {args.operation}: {operation_note}", flush=True)
        output_frame, clipped_pixels = _safe_cast_output(
            operation_frame, output_dtype
        )
        if clipped_pixels:
            print(
                f"[WARN] {clipped_pixels} output pixels were clipped to {output_dtype.name}; "
                "use --out-dtype auto or a wider/floating dtype to avoid clipping",
                flush=True,
            )
        output_block = np.ascontiguousarray(output_frame[np.newaxis, ...])
        print(
            f"[ENCODE] one frame {output_block.shape}, {output_block.dtype.name}, "
            f"logical={_human_bytes(output_block.nbytes)}",
            flush=True,
        )
        frame_payload, frame_header = _encode_raw_frame_block(
            output_block, clevel=args.zstd_level
        )
        print(f"[ENCODE] HDXFB payload={_human_bytes(len(frame_payload))}", flush=True)

        manifest, new_payloads, excluded = _prepare_output_manifest(
            plans,
            output_dtype=output_dtype,
            output_frame_payload=frame_payload,
            output_frame_header=frame_header,
            keep_azint=args.keep_azint,
            operation=args.operation,
            selected_frame_count=total_selected,
        )
        _write_output_archive(
            output_path,
            first_reader=first,
            manifest=manifest,
            new_payloads=new_payloads,
            excluded_members=excluded,
            overwrite=args.overwrite,
        )
        print(f"[OUTPUT] {output_path}", flush=True)
        print(f"[OUTPUT] size={_human_bytes(output_path.stat().st_size)}", flush=True)

        verified_frame: np.ndarray | None = None
        with HDXFReader(output_path, verify_hashes=True) as check_reader:
            for _doc, check_block, local_indices in check_reader.iter_selected_blocks([0]):
                verified_frame = np.ascontiguousarray(check_block[local_indices[0]])
                break
        if verified_frame is None:
            raise HDXFSumError("Output self-check could not read frame 1")
        expected_frame = np.ascontiguousarray(output_frame)
        if verified_frame.dtype != expected_frame.dtype or verified_frame.shape != expected_frame.shape:
            raise HDXFSumError(
                "Output self-check shape/dtype mismatch: "
                f"{verified_frame.shape}/{verified_frame.dtype} != "
                f"{expected_frame.shape}/{expected_frame.dtype}"
            )
        if not np.array_equal(
            verified_frame.view(np.uint8).reshape(-1),
            expected_frame.view(np.uint8).reshape(-1),
        ):
            raise HDXFSumError("Output self-check found a pixel mismatch")
        print("[VERIFY] output frame is bit-exact", flush=True)
        print("[RESULT] PASS — derived HDXF archive created", flush=True)
        return 0
    finally:
        if accumulator is not None:
            try:
                accumulator.flush()
            except Exception:
                pass
            del accumulator
        gc.collect()
        if accumulator_path is not None:
            try:
                accumulator_path.unlink(missing_ok=True)
            except Exception:
                pass
        for reader in readers:
            reader.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[RESULT] STOPPED by user", file=sys.stderr)
        raise SystemExit(130)
    except HDXFSumError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise