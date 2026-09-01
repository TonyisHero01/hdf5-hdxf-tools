#!/usr/bin/env python3
"""Restore an HDXF archive to HDF5 files.

HDXF v0.4.6 exact restoration uses the ``legacy_hdf5`` reconstruction
descriptor written by ``hdf5_to_hdxf_exact_restore.py``.

Exact mode does NOT rebuild or "repair" the master file.  Instead it verifies
the SHA-256 of the original master in ``legacy-template`` and copies that
master byte-for-byte to the output directory.  External HDF5 files are then
reconstructed from their captured file-local object graphs, Dataset storage
descriptors, attributes and links, and archived Dataset values are written
back block-by-block.

This removes the previous Albula-compatibility transformations (extra
/entry/result objects, azint squeeze/resampling, depends_on rewrites, added
conversion attributes, etc.).  The goal is structural/logical restoration of
the source HDF5 layout, not compatibility rewriting.

Older HDXF archives can still be restored through explicit ``manifest`` mode.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
import warnings
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Iterator, Sequence

import h5py
import numpy as np

from hdxf_codec import HDXFBError, decode_hdxfb
from hdxf_index import HDXFIError, unpack_frame_index


def _configure_utf8_stdio() -> None:
    """Make progress output safe on Windows and when stdout is a pipe.

    Python may otherwise select a legacy Windows code page (for example
    cp1252) for a subprocess pipe.  A single non-ASCII status symbol would
    then abort restoration before any HDF5 data is written.
    """
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8:replace")
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_utf8_stdio()

try:  # Registers detector filters such as Bitshuffle/LZ4 when installed.
    import hdf5plugin  # type: ignore  # noqa: F401
    HDF5PLUGIN_AVAILABLE = True
except Exception:
    HDF5PLUGIN_AVAILABLE = False

FORMAT_NAME = "HDF5-derived Detector eXchange Format"
FORMAT_ID = "hdxf"
SUPPORTED_FORMAT_VERSIONS = {
    "0.3.0", "0.4.0", "0.4.2", "0.4.3", "0.4.4", "0.4.5",
    "0.4.6", "0.4.7", "0.4.8",
}
MANIFEST_PATH = "manifest.json"
LEGACY_RECONSTRUCTION_VERSION = "1.1"
SUPPORTED_LEGACY_RECONSTRUCTION_VERSIONS = {"1.0", "1.1"}


def _default_legacy_template_dir() -> Path:
    """Resolve the conventional legacy-template directory robustly.

    Existing project trees have used both ``legacy_template`` and
    ``legacy-template``.  Prefer an existing directory instead of forcing one
    spelling.  If neither exists yet, default to ``legacy_template`` because
    that is the current project layout.
    """
    base = Path(__file__).resolve().parent
    candidates = (
        base / "legacy_template",
        base / "legacy-template",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


DEFAULT_LEGACY_TEMPLATE_DIR = _default_legacy_template_dir()


class RestoreError(RuntimeError):
    """Fatal archive, decoding or HDF5 reconstruction error."""


@dataclass(frozen=True)
class RefSpec:
    target: str | None
    region: bool = False


@dataclass
class RestoreStats:
    groups: int = 0
    datasets: int = 0
    links: int = 0
    frame_blocks: int = 0
    frames: int = 0
    generic_members: int = 0
    derived_datasets: int = 0
    calibration_chunks: int = 0
    storage_fallbacks: int = 0
    creation_property_fallbacks: int = 0
    warnings: int = 0


def _progress(message: str) -> None:
    print(message, flush=True)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_rel_filename(value: str, source_id: str) -> Path:
    """Map a manifest filename to a safe output-relative path."""
    text = str(value or "").replace("\\", "/")
    pure = PurePosixPath(text)
    parts = [p for p in pure.parts if p not in ("", ".", "..", "/")]
    if not parts:
        parts = [f"{source_id}.h5"]
    # Source manifests currently store basenames. Keep nested relative paths if
    # future manifests preserve them, while preventing path traversal.
    path = Path(*parts)
    if path.suffix.lower() not in (".h5", ".hdf5", ".nxs"):
        path = path.with_suffix(".h5")
    return path



def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _safe_plist_set(
    plist: Any,
    method_name: str,
    value: Any,
    *,
    object_path: str,
    stats: RestoreStats,
) -> None:
    """Best-effort restore of non-scientific HDF5 creation properties.

    Shape/dtype/chunks/filter pipeline are treated as critical elsewhere.
    Less-portable creation flags (track-times / creation-order / alloc/fill
    timing) are restored when the local h5py/HDF5 exposes the corresponding
    setter.  A missing setter is reported but does not make an otherwise valid
    HDF5 file unreadable.
    """
    if value is None:
        return
    setter = getattr(plist, method_name, None)
    if setter is None:
        stats.creation_property_fallbacks += 1
        stats.warnings += 1
        _progress(
            f"[WARN] {object_path}: local HDF5 has no {method_name}; "
            "continuing with the library default"
        )
        return
    try:
        setter(value)
    except Exception as exc:
        stats.creation_property_fallbacks += 1
        stats.warnings += 1
        _progress(
            f"[WARN] {object_path}: cannot restore {method_name}={value!r}; "
            f"continuing with the library default ({type(exc).__name__}: {exc})"
        )


def _create_group_from_descriptor(
    h5file: h5py.File,
    path: str,
    creation: Any,
    *,
    stats: RestoreStats,
) -> h5py.Group:
    """Create one Group while replaying captured creation-order properties."""
    if path == "/":
        return h5file["/"]

    parent_path, name = path.rsplit("/", 1)
    parent = h5file[parent_path or "/"]
    if name in parent:
        obj = parent[name]
        if not isinstance(obj, h5py.Group):
            raise RestoreError(
                f"Cannot create Group {h5file.filename}:{path}; "
                "a non-Group object already exists"
            )
        return obj

    gcpl = h5py.h5p.create(h5py.h5p.GROUP_CREATE)
    if isinstance(creation, dict):
        _safe_plist_set(
            gcpl,
            "set_link_creation_order",
            int(creation["link_creation_order"])
            if creation.get("link_creation_order") is not None
            else None,
            object_path=path,
            stats=stats,
        )
        _safe_plist_set(
            gcpl,
            "set_attr_creation_order",
            int(creation["attribute_creation_order"])
            if creation.get("attribute_creation_order") is not None
            else None,
            object_path=path,
            stats=stats,
        )
        if creation.get("object_track_times") is not None:
            _safe_plist_set(
                gcpl,
                "set_obj_track_times",
                bool(creation.get("object_track_times")),
                object_path=path,
                stats=stats,
            )

    try:
        gid = h5py.h5g.create(
            parent.id,
            os.fsencode(name),
            gcpl=gcpl,
        )
        return h5py.Group(gid)
    except Exception as exc:
        raise RestoreError(
            f"Cannot create Group {h5file.filename}:{path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _file_create_kwargs(file_doc: dict[str, Any]) -> dict[str, Any]:
    """Translate portable captured file properties into h5py.File kwargs."""
    kwargs: dict[str, Any] = {}

    userblock = file_doc.get("userblock_size")
    if isinstance(userblock, int) and userblock > 0:
        kwargs["userblock_size"] = int(userblock)

    libver = file_doc.get("libver")
    if isinstance(libver, list) and len(libver) == 2:
        kwargs["libver"] = (str(libver[0]), str(libver[1]))
    elif isinstance(libver, tuple) and len(libver) == 2:
        kwargs["libver"] = (str(libver[0]), str(libver[1]))

    # The original driver is intentionally not forced.  A driver is a host I/O
    # choice rather than an HDF5 object-graph property, and some drivers are not
    # portable across operating systems.
    return kwargs


def _decode_tagged(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_tagged(item) for item in value]
    if not isinstance(value, dict):
        return value
    tag = value.get("$type")
    if tag == "bytes":
        return base64.b64decode(str(value.get("base64", "")))
    if tag == "float":
        token = str(value.get("value", "nan"))
        return {"nan": float("nan"), "+inf": float("inf"), "-inf": float("-inf")}.get(token, float("nan"))
    if tag == "complex":
        return complex(_decode_tagged(value.get("real")), _decode_tagged(value.get("imag")))
    if tag == "ndarray_npy":
        raw = base64.b64decode(str(value.get("base64", "")))
        return np.load(io.BytesIO(raw), allow_pickle=False)
    if tag == "object_array":
        shape = tuple(int(x) for x in value.get("shape", []))
        items = [_decode_tagged(item) for item in value.get("items", [])]
        return np.asarray(items, dtype=object).reshape(shape)
    if tag == "hdf5_reference":
        return RefSpec(target=value.get("target"), region=False)
    if tag == "hdf5_region_reference":
        return RefSpec(target=value.get("target"), region=True)
    if tag == "datetime64":
        return np.datetime64(str(value.get("value")), str(value.get("dtype", "datetime64")))
    if tag == "timedelta64":
        return np.timedelta64(str(value.get("value")))
    if tag in ("attribute_error", "python_repr"):
        return None
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
        # Structured dtype descriptors arrive as JSON lists instead of tuples.
        fields: list[Any] = []
        for item in value:
            if isinstance(item, tuple):
                item = list(item)
            if not isinstance(item, list):
                # Simple subarray descriptors can be represented directly.
                return np.dtype(value)
            if len(item) == 2:
                fields.append((item[0], _dtype_from_descriptor(item[1])))
            elif len(item) == 3:
                fields.append((item[0], _dtype_from_descriptor(item[1]), tuple(item[2])))
            else:
                raise RestoreError(f"Invalid dtype descriptor field: {item!r}")
        return np.dtype(fields)
    raise RestoreError(f"Unsupported dtype descriptor: {value!r}")


def _dtype_from_info(info: Any) -> np.dtype[Any]:
    if not isinstance(info, dict):
        return _dtype_from_descriptor(info)
    base = _dtype_from_descriptor(info.get("numpy"))
    special = info.get("hdf5_special")
    if not isinstance(special, dict):
        return base
    cls = str(special.get("class", ""))
    if cls == "vlen_string":
        return h5py.string_dtype(encoding=str(special.get("encoding") or "utf-8"))
    if cls == "vlen_bytes":
        return h5py.special_dtype(vlen=bytes)
    if cls == "vlen":
        return h5py.vlen_dtype(_dtype_from_descriptor(special.get("base")))
    if cls == "enum":
        members = {str(k): int(v) for k, v in dict(special.get("members") or {}).items()}
        return h5py.enum_dtype(members, basetype=base)
    if cls == "reference":
        return h5py.ref_dtype
    if cls == "region_reference":
        return h5py.regionref_dtype
    return base


def _contains_refspec(value: Any) -> bool:
    if isinstance(value, RefSpec):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return False
        return any(_contains_refspec(item) for item in value.flat)
    if isinstance(value, (list, tuple)):
        return any(_contains_refspec(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_refspec(item) for item in value.values())
    return False


def _resolve_refs(value: Any, h5file: h5py.File) -> Any:
    if isinstance(value, RefSpec):
        if not value.target or value.target not in h5file:
            return h5py.RegionReference() if value.region else h5py.Reference()
        target = h5file[value.target]
        if value.region:
            if isinstance(target, h5py.Dataset):
                try:
                    return target.regionref[...]
                except Exception:
                    return h5py.RegionReference()
            return h5py.RegionReference()
        return target.ref
    if isinstance(value, np.ndarray) and value.dtype == object:
        out = np.empty(value.shape, dtype=object)
        for index in np.ndindex(value.shape):
            out[index] = _resolve_refs(value[index], h5file)
        return out
    if isinstance(value, list):
        return [_resolve_refs(item, h5file) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_refs(item, h5file) for item in value)
    if isinstance(value, dict):
        return {k: _resolve_refs(v, h5file) for k, v in value.items()}
    return value


def _filter_pipeline(dataset: h5py.Dataset) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    try:
        dcpl = dataset.id.get_create_plist()
        for index in range(dcpl.get_nfilters()):
            fid, flags, values, name = dcpl.get_filter(index)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            result.append({
                "id": int(fid), "flags": int(flags),
                "values": [int(v) for v in values], "name": str(name),
            })
    except Exception:
        pass
    return result


def _pipeline_equal(actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> bool:
    if len(actual) != len(expected):
        return False
    for a, e in zip(actual, expected):
        if int(a.get("id", -1)) != int(e.get("id", -2)):
            return False
        if [int(x) for x in a.get("values", [])] != [int(x) for x in e.get("values", [])]:
            return False
    return True


def _bitshuffle_values_compatible(actual_values: list[int], expected_values: list[int]) -> bool:
    """Compare post-creation Bitshuffle parameters semantically.

    The HDF5 Bitshuffle plugin uses a ``set_local`` callback.  The values
    reported by ``get_filter`` commonly have the form
    ``[version_major, version_minor, element_size, block_size, compressor]``,
    while dataset creation expects only ``[block_size, compressor]``.  A
    different installed plugin release can also replace the first two version
    numbers even though the stored raw chunks remain readable.
    """
    av = [int(x) for x in actual_values]
    ev = [int(x) for x in expected_values]
    if av == ev:
        return True
    if len(av) >= 5 and len(ev) >= 5:
        # Element size, block size and compression backend determine the
        # dataset-local decoding parameters.  Plugin version fields may be
        # normalized by the locally installed filter.
        return av[2:] == ev[2:]
    if len(av) >= 2 and len(ev) >= 2:
        return av[-2:] == ev[-2:]
    return False


def _pipeline_compatible(actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> bool:
    """Return True when HDF5 normalized a filter pipeline harmlessly."""
    if len(actual) != len(expected):
        return False
    for a, e in zip(actual, expected):
        fid_a = int(a.get("id", -1))
        fid_e = int(e.get("id", -2))
        if fid_a != fid_e:
            return False
        av = [int(x) for x in a.get("values", [])]
        ev = [int(x) for x in e.get("values", [])]
        if av == ev:
            continue
        if fid_a == 32008 and _bitshuffle_values_compatible(av, ev):
            continue
        return False
    return True


def _filter_creation_values(item: dict[str, Any]) -> tuple[int, ...]:
    """Convert archived post-creation values into filter input values.

    In particular, Bitshuffle filter 32008 expands its two user parameters via
    an HDF5 ``set_local`` callback.  Feeding the already-expanded five-value
    pipeline back into ``set_filter`` causes the plugin to expand it a second
    time, which produces an apparent pipeline mismatch on real Jungfrau files.
    """
    fid = int(item.get("id", -1))
    values = tuple(int(v) for v in item.get("values", []))
    if fid == 32008 and len(values) >= 5:
        return values[-2:]
    return values


def _filter_available(filter_id: int) -> bool:
    try:
        return bool(h5py.h5z.filter_avail(int(filter_id)))
    except Exception:
        return filter_id in (1, 2, 3, 4, 5, 6)


def _create_dataset_exact(
    h5file: h5py.File,
    path: str,
    storage: dict[str, Any],
    *,
    require_exact_filters: bool,
    stats: RestoreStats,
    creation: dict[str, Any] | None = None,
) -> h5py.Dataset:
    """Create one Dataset, preserving storage when possible.

    Custom filters unavailable to the current HDF5 process are replaced by
    gzip+shuffle for logically decoded datasets. Raw Calibration chunks set
    ``require_exact_filters`` and therefore always recreate the exact pipeline.
    """
    dtype = _dtype_from_info(storage.get("dtype"))
    shape_doc = storage.get("shape")
    maxshape_doc = storage.get("maxshape")
    shape = None if shape_doc is None else tuple(int(x) for x in shape_doc)
    maxshape = None
    if isinstance(maxshape_doc, list):
        maxshape = tuple(h5py.h5s.UNLIMITED if x is None else int(x) for x in maxshape_doc)
    parent_path, name = path.rsplit("/", 1) if path != "/" else ("/", "")
    parent = h5file.require_group(parent_path or "/")
    if name in parent:
        del parent[name]

    if shape is None:
        sid = h5py.h5s.create(h5py.h5s.NULL)
    elif shape == ():
        sid = h5py.h5s.create(h5py.h5s.SCALAR)
    else:
        sid = h5py.h5s.create_simple(shape, maxshape)

    expected_filters = [x for x in storage.get("filters", []) if isinstance(x, dict)]
    unavailable = [int(x.get("id", -1)) for x in expected_filters if not _filter_available(int(x.get("id", -1)))]
    use_exact = require_exact_filters or not unavailable
    if require_exact_filters and unavailable:
        # Creation and direct-chunk writing are still legal with unavailable
        # optional filters. Reading the restored data later requires the plugin.
        use_exact = True

    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)

    if isinstance(creation, dict):
        # Replay stable low-level creation properties captured by the v0.4.6
        # converter. Layout/chunk/filter semantics remain critical and are
        # checked separately below.
        layout = creation.get("layout")
        if layout is not None:
            try:
                dcpl.set_layout(int(layout))
            except Exception as exc:
                stats.creation_property_fallbacks += 1
                stats.warnings += 1
                _progress(
                    f"[WARN] {path}: cannot restore HDF5 layout code {layout!r}; "
                    f"continuing with layout inferred from chunks/storage "
                    f"({type(exc).__name__}: {exc})"
                )
        if creation.get("allocation_time") is not None:
            _safe_plist_set(
                dcpl,
                "set_alloc_time",
                int(creation["allocation_time"]),
                object_path=path,
                stats=stats,
            )
        if creation.get("fill_time") is not None:
            _safe_plist_set(
                dcpl,
                "set_fill_time",
                int(creation["fill_time"]),
                object_path=path,
                stats=stats,
            )
        if creation.get("attribute_creation_order") is not None:
            _safe_plist_set(
                dcpl,
                "set_attr_creation_order",
                int(creation["attribute_creation_order"]),
                object_path=path,
                stats=stats,
            )
        if creation.get("object_track_times") is not None:
            _safe_plist_set(
                dcpl,
                "set_obj_track_times",
                bool(creation["object_track_times"]),
                object_path=path,
                stats=stats,
            )

    chunks_doc = storage.get("chunks")
    chunks = tuple(int(x) for x in chunks_doc) if isinstance(chunks_doc, list) else None
    if shape not in (None, ()) and chunks is not None:
        dcpl.set_chunk(chunks)

    fillvalue = _decode_tagged(storage.get("fillvalue"))
    if fillvalue is not None and shape is not None:
        try:
            dcpl.set_fill_value(np.asarray(fillvalue, dtype=dtype))
        except Exception:
            pass

    external_storage = storage.get("external_storage")
    if isinstance(external_storage, list):
        for item in external_storage:
            if not isinstance(item, dict):
                continue
            try:
                dcpl.set_external(
                    os.fsencode(str(item.get("filename") or "external.raw")),
                    int(item.get("offset", 0)),
                    int(item.get("size", h5py.h5f.UNLIMITED)),
                )
            except Exception:
                pass

    if use_exact and chunks is not None:
        for item in expected_filters:
            try:
                dcpl.set_filter(
                    int(item.get("id")), int(item.get("flags", 0)),
                    _filter_creation_values(item),
                )
            except Exception as exc:
                if require_exact_filters:
                    raise RestoreError(f"Cannot recreate exact filter pipeline for {path}: {exc}") from exc
                use_exact = False
                break

    if not use_exact and chunks is not None:
        stats.storage_fallbacks += 1
        stats.warnings += 1
        missing_text = ", ".join(str(x) for x in unavailable) or "unknown"
        _progress(
            f"[WARN] {path}: original HDF5 filter(s) {missing_text} are unavailable; "
            "using gzip+shuffle while preserving logical values"
        )
        # Portable fallback for logical data. Do not apply to direct chunks.
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_chunk(chunks)
        try:
            dcpl.set_shuffle()
            dcpl.set_deflate(4)
        except Exception:
            pass

    try:
        tid = h5py.h5t.py_create(dtype, logical=True)
        did = h5py.h5d.create(parent.id, os.fsencode(name), tid, sid, dcpl=dcpl)
        dataset = h5py.Dataset(did)
    except Exception as exc:
        raise RestoreError(f"Cannot create Dataset {h5file.filename}:{path}: {exc}") from exc

    if require_exact_filters:
        actual_filters = _filter_pipeline(dataset)
        if not _pipeline_equal(actual_filters, expected_filters):
            if _pipeline_compatible(actual_filters, expected_filters):
                stats.warnings += 1
                _progress(
                    f"[WARN] {path}: HDF5 normalized the custom filter parameters; "
                    "the decoding-relevant Bitshuffle settings are compatible"
                )
            else:
                raise RestoreError(
                    f"Incompatible filter pipeline for {h5file.filename}:{path}; "
                    f"expected={expected_filters}, actual={actual_filters}. "
                    "Install a compatible hdf5plugin build and retry."
                )
    return dataset


class HDXFArchive:
    def __init__(self, path: Path, *, verify_hashes: bool = True) -> None:
        self.path = path.resolve()
        self.verify_hashes = bool(verify_hashes)
        try:
            self.zip = zipfile.ZipFile(self.path, "r")
        except Exception as exc:
            raise RestoreError(f"Cannot open HDXF archive {self.path}: {exc}") from exc
        try:
            self.manifest = json.loads(self.zip.read(MANIFEST_PATH))
        except KeyError as exc:
            self.close()
            raise RestoreError("manifest.json is missing") from exc
        except Exception as exc:
            self.close()
            raise RestoreError(f"Cannot parse manifest.json: {exc}") from exc
        identity = self.manifest.get("hdxf", {})
        if identity.get("format") != FORMAT_NAME or identity.get("format_id") != FORMAT_ID:
            self.close()
            raise RestoreError(f"Not an HDXF archive: {self.path}")
        version = str(identity.get("version", ""))
        if version not in SUPPORTED_FORMAT_VERSIONS:
            self.close()
            raise RestoreError(f"Unsupported HDXF version {version!r}")
        self.version = version
        objects = self.manifest.get("objects")
        if not isinstance(objects, dict):
            self.close()
            raise RestoreError("Manifest has no HDF5 object graph")
        self.objects: dict[str, dict[str, Any]] = {
            str(k): v for k, v in objects.items() if isinstance(v, dict)
        }
        self.links = [x for x in self.manifest.get("links", []) if isinstance(x, dict)]
        source = self.manifest.get("source", {})
        self.main_source_id = str(source.get("main_file") or "")
        self.source_files = [x for x in source.get("files", []) if isinstance(x, dict)]
        self.source_by_id = {str(x.get("id")): x for x in self.source_files if x.get("id")}

        legacy = self.manifest.get("legacy_hdf5")
        self.legacy_hdf5: dict[str, Any] | None = (
            legacy if isinstance(legacy, dict) else None
        )

        self.frames_doc: dict[str, Any] | None = None
        self.blocks: list[dict[str, Any]] = []
        detector_archive = self.manifest.get("detector_archive")
        if isinstance(detector_archive, dict) and isinstance(detector_archive.get("frames"), dict):
            self.frames_doc = detector_archive["frames"]
            blocks = self.frames_doc.get("blocks")
            if not isinstance(blocks, list) or not blocks:
                blocks = self._load_binary_index(self.frames_doc)
            self.blocks = sorted(
                [x for x in blocks if isinstance(x, dict)],
                key=lambda x: int(x.get("global_start_frame", 0)),
            )
        self._static_cache: dict[str, dict[str, np.ndarray]] = {}

    def close(self) -> None:
        self._static_cache.clear()
        if getattr(self, "zip", None) is not None:
            self.zip.close()

    def __enter__(self) -> "HDXFArchive":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def read_member(self, doc: dict[str, Any]) -> bytes:
        path = str(doc.get("path") or "")
        try:
            raw = self.zip.read(path)
        except KeyError as exc:
            raise RestoreError(f"Missing archive member {path}") from exc
        expected_size = doc.get("size_bytes")
        if isinstance(expected_size, int) and len(raw) != expected_size:
            raise RestoreError(f"Archive member size mismatch: {path}")
        expected_sha = doc.get("sha256")
        if self.verify_hashes and isinstance(expected_sha, str) and _sha256(raw) != expected_sha:
            raise RestoreError(f"Archive member SHA-256 mismatch: {path}")
        return raw

    def _load_binary_index(self, frames: dict[str, Any]) -> list[dict[str, Any]]:
        index_doc = frames.get("block_index")
        if not isinstance(index_doc, dict):
            raise RestoreError("Frame blocks and block_index are both missing")
        raw = self.read_member(index_doc)
        try:
            return unpack_frame_index(
                raw,
                datasets=list(frames.get("datasets", [])),
                frame_shape=list(frames.get("frame_shape", [])),
            )
        except HDXFIError as exc:
            raise RestoreError(f"Invalid HDXFI frame index: {exc}") from exc

    def _load_static_model(self, model_id: str | None) -> dict[str, np.ndarray] | None:
        if not model_id or self.frames_doc is None:
            return None
        key = str(model_id)
        cached = self._static_cache.get(key)
        if cached is not None:
            return cached
        models = {
            str(x.get("id")): x
            for x in self.frames_doc.get("static_models", [])
            if isinstance(x, dict) and x.get("id")
        }
        doc = models.get(key)
        if doc is None:
            raise RestoreError(f"Static model {key} is missing")
        mask_doc = doc.get("mask")
        values_doc = doc.get("static_values")
        if not isinstance(mask_doc, dict) or not isinstance(values_doc, dict):
            raise RestoreError(f"Static model {key} is incomplete")
        try:
            packed, _ = decode_hdxfb(
                self.read_member(mask_doc),
                member_path=str(mask_doc.get("path")),
                verify_hash=self.verify_hashes,
            )
            values, _ = decode_hdxfb(
                self.read_member(values_doc),
                member_path=str(values_doc.get("path")),
                verify_hash=self.verify_hashes,
            )
        except HDXFBError as exc:
            raise RestoreError(f"Cannot decode static model {key}: {exc}") from exc
        frame_shape = tuple(int(x) for x in self.frames_doc.get("frame_shape", []))
        pixel_count = int(doc.get("pixel_count", np.prod(frame_shape)))
        bits = np.unpackbits(np.asarray(packed, dtype=np.uint8).reshape(-1), bitorder="little")
        if bits.size < pixel_count:
            raise RestoreError(f"Static model {key} mask is truncated")
        mask = bits[:pixel_count].astype(bool).reshape(frame_shape)
        values = np.ascontiguousarray(values).reshape(-1)
        if values.size != int(np.count_nonzero(mask)):
            raise RestoreError(f"Static model {key} value count mismatch")
        result = {"mask": mask, "static_values": values}
        self._static_cache[key] = result
        return result

    def decode_frame_block(self, doc: dict[str, Any], previous_frame: np.ndarray | None) -> np.ndarray:
        raw = self.read_member(doc)
        try:
            block, header = decode_hdxfb(
                raw,
                member_path=str(doc.get("path")),
                verify_hash=self.verify_hashes,
                previous_frame=previous_frame,
                static_model=self._load_static_model(doc.get("static_model_id")),
            )
        except HDXFBError as exc:
            raise RestoreError(f"Cannot decode frame block {doc.get('path')}: {exc}") from exc
        if header.get("kind") != "frame-block":
            raise RestoreError(f"Unexpected HDXFB kind in {doc.get('path')}")
        return np.ascontiguousarray(block)

    def decode_generic_member(self, member: dict[str, Any]) -> Any:
        raw = self.read_member(member)
        encoding = str(member.get("encoding") or "")
        if encoding == "npy":
            return np.load(io.BytesIO(raw), allow_pickle=False)
        if encoding == "json":
            doc = json.loads(raw.decode("utf-8"))
            return _decode_tagged(doc.get("data"))
        if encoding == "array-block-v1":
            try:
                value, _ = decode_hdxfb(
                    raw,
                    member_path=str(member.get("path")),
                    verify_hash=self.verify_hashes,
                )
            except HDXFBError as exc:
                raise RestoreError(f"Cannot decode array block {member.get('path')}: {exc}") from exc
            return value
        raise RestoreError(f"Unsupported Dataset payload encoding {encoding!r}")



class ExactLegacyRestorer:
    """Restore v0.4.6/v0.4.7/v0.4.8 HDXF using the byte-identical original master template."""

    def __init__(
        self,
        archive: HDXFArchive,
        output_dir: Path,
        *,
        template_dir: Path,
        template_file: Path | None,
        overwrite: bool,
        verify_output: bool,
        progress_every: int,
    ) -> None:
        self.archive = archive
        self.output_dir = output_dir.resolve()
        self.template_dir = template_dir.resolve()
        self.template_file = (
            template_file.resolve()
            if template_file is not None
            else None
        )
        self.overwrite = bool(overwrite)
        self.verify_output = bool(verify_output)
        self.progress_every = max(1, int(progress_every))
        self.stats = RestoreStats()
        self.warnings: list[str] = []
        self.deferred_attrs: list[tuple[str, str, str, Any, Any]] = []
        self.deferred_dataset_members: list[
            tuple[str, str, dict[str, Any], Any]
        ] = []
        self.deferred_derived_datasets: list[
            tuple[str, str, dict[str, Any]]
        ] = []

        legacy = self.archive.legacy_hdf5
        if not isinstance(legacy, dict):
            raise RestoreError(
                "This HDXF has no legacy_hdf5 reconstruction descriptor. "
                "Use --restore-mode manifest only for an older archive."
            )
        self.legacy = legacy

        version = str(legacy.get("version") or "")
        if version not in SUPPORTED_LEGACY_RECONSTRUCTION_VERSIONS:
            raise RestoreError(
                f"Unsupported legacy_hdf5 descriptor version {version!r}; "
                f"supported={sorted(SUPPORTED_LEGACY_RECONSTRUCTION_VERSIONS)}"
            )

        coverage = legacy.get("coverage")
        if not isinstance(coverage, dict) or coverage.get(
            "all_external_datasets_archived"
        ) is not True:
            missing = (
                coverage.get("unarchived_datasets")
                if isinstance(coverage, dict)
                else None
            )
            raise RestoreError(
                "HDXF does not guarantee complete external-Dataset coverage; "
                f"exact restoration is unsafe. Missing={missing!r}"
            )

        master_doc = legacy.get("master_template")
        if not isinstance(master_doc, dict):
            raise RestoreError(
                "legacy_hdf5.master_template is missing"
            )
        self.master_doc = master_doc
        self.template_master = self._resolve_template_master()

        external_files = legacy.get("external_files")
        if not isinstance(external_files, list):
            raise RestoreError(
                "legacy_hdf5.external_files is missing"
            )
        self.external_files = [
            item for item in external_files if isinstance(item, dict)
        ]

        self.external_by_source: dict[str, dict[str, Any]] = {}
        for item in self.external_files:
            file_doc = item.get("source_file")
            if not isinstance(file_doc, dict):
                raise RestoreError(
                    "An external-file reconstruction descriptor has no source_file"
                )
            source_id = str(file_doc.get("source_id") or "")
            if not source_id:
                raise RestoreError(
                    "An external-file reconstruction descriptor has no source_id"
                )
            if source_id in self.external_by_source:
                raise RestoreError(
                    f"Duplicate external source_id in reconstruction descriptor: "
                    f"{source_id}"
                )
            self.external_by_source[source_id] = item

        self.source_paths: dict[str, Path] = {}
        self.archive_object_locations: dict[
            str, tuple[str, str]
        ] = {}
        self._build_source_paths()
        self._build_archive_object_locations()

    def warn(self, message: str) -> None:
        self.stats.warnings += 1
        self.warnings.append(message)
        _progress(f"[WARN] {message}")

    def _resolve_template_master(self) -> Path:
        filename = str(self.master_doc.get("filename") or "")
        expected_sha = str(self.master_doc.get("sha256") or "")
        if not filename or not expected_sha:
            raise RestoreError(
                "master_template filename/SHA-256 is incomplete"
            )

        candidate = (
            self.template_file
            if self.template_file is not None
            else self.template_dir / filename
        )
        candidate = candidate.resolve()

        if not candidate.is_file():
            raise RestoreError(
                "Original master template not found: "
                f"{candidate}"
            )

        actual_sha = _sha256_file(candidate)
        if actual_sha != expected_sha:
            raise RestoreError(
                "legacy-template master SHA-256 mismatch.\n"
                f"template: {candidate}\n"
                f"expected: {expected_sha}\n"
                f"actual:   {actual_sha}"
            )

        expected_size = self.master_doc.get("size_bytes")
        if (
            isinstance(expected_size, int)
            and int(candidate.stat().st_size) != expected_size
        ):
            raise RestoreError(
                "legacy-template master size mismatch: "
                f"expected={expected_size}, "
                f"actual={candidate.stat().st_size}"
            )
        return candidate

    def _master_link_output_paths(self) -> dict[str, Path]:
        """Map target source IDs to the relative filenames used by the master."""
        result: dict[str, Path] = {}
        links = self.legacy.get("external_links")
        if not isinstance(links, list):
            return result

        for item in links:
            if not isinstance(item, dict):
                continue
            if str(item.get("parent_source_file") or "") != (
                self.archive.main_source_id
            ):
                continue

            source_id = str(item.get("target_source_file") or "")
            filename = str(item.get("target_file") or "")
            if not source_id or not filename:
                continue

            if item.get("target_file_was_absolute"):
                # The copied master still contains the original absolute path.
                # Rewriting it would destroy byte-identical master restoration.
                raise RestoreError(
                    "Exact standalone restoration cannot relocate a master "
                    "ExternalLink that was absolute in the source HDF5. "
                    f"Link={item.get('link_path')}, file={filename!r}"
                )

            rel = _safe_rel_filename(filename, source_id)
            existing = result.get(source_id)
            if existing is not None and os.path.normcase(str(existing)) != (
                os.path.normcase(str(rel))
            ):
                raise RestoreError(
                    f"Source {source_id} is referenced by multiple different "
                    f"master filenames: {existing} vs {rel}"
                )
            result[source_id] = rel
        return result

    def _build_source_paths(self) -> None:
        master_name = str(
            self.master_doc.get("filename")
            or self.archive.source_by_id.get(
                self.archive.main_source_id, {}
            ).get("filename")
            or "restored_master.h5"
        )
        self.source_paths[self.archive.main_source_id] = (
            self.output_dir
            / _safe_rel_filename(
                master_name,
                self.archive.main_source_id,
            )
        )

        master_link_paths = self._master_link_output_paths()
        used: dict[str, str] = {
            os.path.normcase(
                str(self.source_paths[self.archive.main_source_id].relative_to(
                    self.output_dir
                ))
            ): self.archive.main_source_id
        }

        for source_id, descriptor in self.external_by_source.items():
            file_doc = descriptor.get("source_file") or {}
            rel = master_link_paths.get(source_id)
            if rel is None:
                rel = _safe_rel_filename(
                    str(file_doc.get("filename") or ""),
                    source_id,
                )

            key = os.path.normcase(str(rel))
            other = used.get(key)
            if other is not None and other != source_id:
                raise RestoreError(
                    "Exact restoration cannot disambiguate duplicate physical "
                    f"HDF5 filenames: {rel} ({other}, {source_id})"
                )
            used[key] = source_id
            self.source_paths[source_id] = self.output_dir / rel

    def _build_archive_object_locations(self) -> None:
        for source_id, descriptor in self.external_by_source.items():
            objects = descriptor.get("objects")
            if not isinstance(objects, dict):
                raise RestoreError(
                    f"External descriptor {source_id} has no object table"
                )
            for record in objects.values():
                if not isinstance(record, dict):
                    continue
                archive_object = str(record.get("archive_object") or "")
                if not archive_object:
                    continue
                path = str(record.get("canonical_path") or "")
                if not path:
                    raise RestoreError(
                        f"External Dataset mapping for {archive_object} "
                        "has no canonical_path"
                    )
                previous = self.archive_object_locations.get(
                    archive_object
                )
                current = (source_id, path)
                if previous is not None and previous != current:
                    raise RestoreError(
                        f"Archive object {archive_object} maps to multiple "
                        f"physical locations: {previous} vs {current}"
                    )
                self.archive_object_locations[
                    archive_object
                ] = current

        # Every frame Dataset must map back to one exact physical Dataset.
        frames = self.archive.frames_doc
        if isinstance(frames, dict):
            for item in frames.get("datasets", []):
                if not isinstance(item, dict):
                    continue
                object_id = str(item.get("object") or "")
                if (
                    object_id
                    and object_id not in self.archive_object_locations
                ):
                    obj = self.archive.objects.get(object_id)
                    if (
                        isinstance(obj, dict)
                        and str(obj.get("source_file") or "")
                        != self.archive.main_source_id
                    ):
                        raise RestoreError(
                            "Frame Dataset is missing from the exact external "
                            f"layout descriptor: {object_id}"
                        )

    @property
    def master_output(self) -> Path:
        return self.source_paths[self.archive.main_source_id]

    def _prepare_destination(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Never allow --overwrite to delete the template itself.
        if self.master_output.resolve() == self.template_master.resolve():
            raise RestoreError(
                "Output master path is the same file as legacy-template. "
                "Choose a different output directory."
            )

        existing = [
            path for path in self.source_paths.values() if path.exists()
        ]
        if existing and not self.overwrite:
            listed = "\n".join(str(path) for path in existing[:12])
            raise RestoreError(
                "Output files already exist; use --overwrite:\n"
                f"{listed}"
            )

        for path in self.source_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                path.unlink()

    def _copy_master_template(self) -> None:
        _progress(
            f"[EXACT] Copying byte-identical master template: "
            f"{self.template_master.name}"
        )
        shutil.copy2(self.template_master, self.master_output)

        expected_sha = str(self.master_doc.get("sha256") or "")
        copied_sha = _sha256_file(self.master_output)
        if copied_sha != expected_sha:
            raise RestoreError(
                "Copied master template changed unexpectedly: "
                f"expected={expected_sha}, actual={copied_sha}"
            )

    @staticmethod
    def _parent_path(path: str) -> str:
        parent = str(PurePosixPath(path).parent)
        return "/" if parent == "." else parent

    @staticmethod
    def _attribute_record(
        encoded: Any,
    ) -> tuple[Any, dict[str, Any] | None]:
        if (
            isinstance(encoded, dict)
            and encoded.get("$type") == "hdf5_attribute_v2"
        ):
            return _decode_tagged(encoded.get("value")), encoded
        return _decode_tagged(encoded), None

    @staticmethod
    def _attribute_shape_from_record(
        record: dict[str, Any] | None,
        value: Any,
    ) -> tuple[int, ...] | None:
        if not isinstance(record, dict):
            if isinstance(value, np.ndarray):
                return tuple(int(x) for x in value.shape)
            return ()

        dataspace = record.get("dataspace")
        if isinstance(dataspace, dict):
            cls = str(dataspace.get("class") or "")
            if cls == "null":
                return None
            if cls == "scalar":
                return ()
            shape = dataspace.get("shape")
            if isinstance(shape, list):
                return tuple(int(x) for x in shape)

        shape = record.get("shape")
        if shape is None:
            return ()
        if isinstance(shape, list):
            return tuple(int(x) for x in shape)
        return ()

    @staticmethod
    def _attribute_dtype_from_record(
        record: dict[str, Any] | None,
        value: Any,
    ) -> np.dtype[Any] | None:
        if isinstance(record, dict) and "dtype" in record:
            return _dtype_from_info(record.get("dtype"))
        if isinstance(value, np.ndarray):
            return np.dtype(value.dtype)
        if isinstance(value, np.generic):
            return np.dtype(value.dtype)
        return None

    @staticmethod
    def _fixed_string_type_from_record(
        record: dict[str, Any] | None,
        dtype: np.dtype[Any] | None,
    ) -> Any | None:
        if not isinstance(record, dict):
            return None
        type_doc = record.get("hdf5_type")
        if not isinstance(type_doc, dict):
            return None
        if str(type_doc.get("kind") or "") != "string":
            return None
        if bool(type_doc.get("is_variable_str")):
            return None

        try:
            type_id = h5py.h5t.C_S1.copy()
            size = int(
                type_doc.get(
                    "size",
                    0 if dtype is None else dtype.itemsize,
                )
            )
            if size <= 0:
                return None
            type_id.set_size(size)

            cset = type_doc.get("cset")
            if cset is not None:
                type_id.set_cset(int(cset))

            strpad = type_doc.get("strpad")
            if strpad is not None:
                type_id.set_strpad(int(strpad))
            return type_id
        except Exception:
            return None

    @staticmethod
    def _make_attr_array(
        value: Any,
        dtype: np.dtype[Any],
        shape: tuple[int, ...],
    ) -> np.ndarray:
        if shape == ():
            return np.asarray(value, dtype=dtype).reshape(())
        array = np.asarray(value, dtype=dtype)
        if tuple(array.shape) != tuple(shape):
            array = array.reshape(shape)
        return np.ascontiguousarray(array)

    def _create_attribute_exact_value(
        self,
        target: Any,
        name: str,
        value: Any,
        record: dict[str, Any] | None,
    ) -> None:
        """Create one Attribute without asking h5py to infer its dtype."""
        dtype = self._attribute_dtype_from_record(record, value)
        shape = self._attribute_shape_from_record(record, value)

        if name in target.attrs:
            del target.attrs[name]

        if shape is None:
            if dtype is None:
                raise RestoreError(
                    f"Null Attribute {target.name}@{name} has no dtype"
                )
            type_id = h5py.h5t.py_create(dtype, logical=True)
            space_id = h5py.h5s.create(h5py.h5s.NULL)
            h5py.h5a.create(
                target.id,
                name.encode("utf-8"),
                type_id,
                space_id,
            )
            return

        if dtype is None:
            # Backward-compatible descriptor 1.0 path.
            if isinstance(value, (bytes, bytearray, memoryview)):
                value = np.bytes_(bytes(value))
            target.attrs[name] = value
            return

        fixed_string_type = self._fixed_string_type_from_record(
            record, dtype
        )
        if fixed_string_type is not None:
            if shape == ():
                space_id = h5py.h5s.create(h5py.h5s.SCALAR)
            else:
                space_id = h5py.h5s.create_simple(shape)

            attr_id = h5py.h5a.create(
                target.id,
                name.encode("utf-8"),
                fixed_string_type,
                space_id,
            )
            array = self._make_attr_array(
                value,
                np.dtype(f"S{int(fixed_string_type.get_size())}"),
                shape,
            )
            attr_id.write(array)
            return

        if shape == ():
            scalar = np.asarray(
                value, dtype=dtype
            ).reshape(())[()]
            target.attrs.create(
                name,
                scalar,
                shape=(),
                dtype=dtype,
            )
        else:
            array = self._make_attr_array(
                value, dtype, shape
            )
            target.attrs.create(
                name,
                array,
                shape=shape,
                dtype=dtype,
            )

    @staticmethod
    def _attribute_schema_matches(
        target: Any,
        name: str,
        record: dict[str, Any] | None,
    ) -> tuple[bool, str]:
        if not isinstance(record, dict):
            return True, ""

        try:
            attr_id = target.attrs.get_id(name)
        except Exception as exc:
            return False, f"cannot get Attribute ID: {exc}"

        expected_dtype = _dtype_from_info(record.get("dtype"))
        actual_dtype = np.dtype(attr_id.dtype)
        if actual_dtype != expected_dtype:
            return (
                False,
                f"dtype expected={expected_dtype!r}, "
                f"actual={actual_dtype!r}",
            )

        expected_shape = ExactLegacyRestorer._attribute_shape_from_record(
            record, target.attrs[name]
        )
        actual_shape = (
            None
            if getattr(attr_id, "shape", None) is None
            else tuple(int(x) for x in attr_id.shape)
        )
        if expected_shape != actual_shape:
            return (
                False,
                f"shape expected={expected_shape!r}, "
                f"actual={actual_shape!r}",
            )

        type_doc = record.get("hdf5_type")
        if (
            isinstance(type_doc, dict)
            and str(type_doc.get("kind") or "") == "string"
            and not bool(type_doc.get("is_variable_str"))
        ):
            try:
                type_id = attr_id.get_type()
                checks = {
                    "size": int(type_id.get_size()),
                    "cset": int(type_id.get_cset()),
                    "strpad": int(type_id.get_strpad()),
                }
                for field, actual in checks.items():
                    expected = type_doc.get(field)
                    if (
                        expected is not None
                        and int(expected) != int(actual)
                    ):
                        return (
                            False,
                            f"{field} expected={expected}, "
                            f"actual={actual}",
                        )
            except Exception as exc:
                return (
                    False,
                    f"fixed-string type inspection failed: {exc}",
                )

        return True, ""


    def _apply_attributes_exact(
        self,
        source_id: str,
        object_path: str,
        target: Any,
        attrs: Any,
    ) -> None:
        if not isinstance(attrs, dict):
            return

        for name, encoded in attrs.items():
            value, attr_record = self._attribute_record(encoded)
            if (
                value is None
                and isinstance(encoded, dict)
                and encoded.get("$type")
                in ("attribute_error", "python_repr")
            ):
                raise RestoreError(
                    "Exact restoration cannot recreate source attribute "
                    f"{object_path}@{name}; archive contains "
                    f"{encoded.get('$type')}"
                )

            if _contains_refspec(value):
                self.deferred_attrs.append(
                    (
                        source_id,
                        object_path,
                        str(name),
                        encoded,
                        value,
                    )
                )
                continue

            try:
                self._create_attribute_exact_value(
                    target,
                    str(name),
                    value,
                    attr_record,
                )
            except Exception as exc:
                if isinstance(exc, RestoreError):
                    raise
                raise RestoreError(
                    f"Cannot exactly restore attribute "
                    f"{object_path}@{name}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

    def _create_external_file_structure(
        self,
        source_id: str,
        descriptor: dict[str, Any],
    ) -> None:
        output = self.source_paths[source_id]
        file_doc = descriptor.get("source_file")
        objects = descriptor.get("objects")
        links = descriptor.get("links")
        root_id = str(descriptor.get("root_object") or "")

        if not isinstance(file_doc, dict):
            raise RestoreError(
                f"External descriptor {source_id} has no source_file"
            )
        if not isinstance(objects, dict) or root_id not in objects:
            raise RestoreError(
                f"External descriptor {source_id} has an invalid object graph"
            )
        if not isinstance(links, list):
            raise RestoreError(
                f"External descriptor {source_id} has no link table"
            )

        _progress(
            f"[EXACT] Rebuilding external HDF5 structure: "
            f"{output.name}"
        )

        try:
            h5file = h5py.File(
                output,
                "w",
                **_file_create_kwargs(file_doc),
            )
        except Exception as exc:
            raise RestoreError(
                f"Cannot create external HDF5 file {output}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        with h5file:
            root_record = objects[root_id]
            if (
                not isinstance(root_record, dict)
                or root_record.get("type") != "group"
            ):
                raise RestoreError(
                    f"External descriptor {source_id} root is not a Group"
                )

            self._apply_attributes_exact(
                source_id,
                "/",
                h5file["/"],
                root_record.get("attributes"),
            )

            children_by_parent: dict[
                str, list[dict[str, Any]]
            ] = defaultdict(list)
            for link in links:
                if not isinstance(link, dict):
                    continue
                path = str(link.get("path") or "")
                if not path or path == "/":
                    continue
                children_by_parent[
                    self._parent_path(path)
                ].append(link)

            created: dict[str, str] = {root_id: "/"}

            def create_object(
                object_id: str,
                path: str,
            ) -> None:
                record = objects.get(object_id)
                if not isinstance(record, dict):
                    raise RestoreError(
                        f"Missing local object {object_id} in {output.name}"
                    )

                canonical = str(
                    record.get("canonical_path") or ""
                )
                if canonical and canonical != path:
                    raise RestoreError(
                        "Local hard-link descriptor is inconsistent: "
                        f"object={object_id}, canonical={canonical}, "
                        f"first_create_path={path}"
                    )

                object_type = str(record.get("type") or "")
                if object_type == "group":
                    group = _create_group_from_descriptor(
                        h5file,
                        path,
                        record.get("creation"),
                        stats=self.stats,
                    )
                    created[object_id] = path
                    self._apply_attributes_exact(
                        source_id,
                        path,
                        group,
                        record.get("attributes"),
                    )
                    create_children(path)
                    self.stats.groups += 1
                    return

                if object_type == "dataset":
                    storage = record.get("hdf5_storage")
                    if not isinstance(storage, dict):
                        raise RestoreError(
                            f"Dataset storage metadata missing: "
                            f"{output.name}:{path}"
                        )
                    dataset = _create_dataset_exact(
                        h5file,
                        path,
                        storage,
                        require_exact_filters=True,
                        stats=self.stats,
                        creation=(
                            record.get("creation")
                            if isinstance(
                                record.get("creation"), dict
                            )
                            else None
                        ),
                    )
                    created[object_id] = path
                    self._apply_attributes_exact(
                        source_id,
                        path,
                        dataset,
                        record.get("attributes"),
                    )
                    self.stats.datasets += 1
                    return

                raise RestoreError(
                    f"Unsupported local object type {object_type!r}: "
                    f"{output.name}:{path}"
                )

            def create_children(group_path: str) -> None:
                for link in children_by_parent.get(
                    group_path, []
                ):
                    link_type = str(link.get("type") or "")
                    path = str(link.get("path") or "")
                    parent_path, name = path.rsplit("/", 1)
                    parent = h5file[parent_path or "/"]

                    if name in parent:
                        # The canonical hard-link path is created directly by
                        # create_object.  Any other pre-existing name here means
                        # the descriptor would overwrite a real object/link.
                        existing_link = parent.get(
                            name, getlink=True
                        )
                        if link_type == "hard":
                            target_id = str(
                                link.get("target_object") or ""
                            )
                            if created.get(target_id) == path:
                                continue
                        raise RestoreError(
                            f"Duplicate link path while rebuilding "
                            f"{output.name}:{path}; existing={existing_link}"
                        )

                    if link_type == "hard":
                        target_id = str(
                            link.get("target_object") or ""
                        )
                        if not target_id:
                            raise RestoreError(
                                f"Hard link has no target: "
                                f"{output.name}:{path}"
                            )

                        target_path = created.get(target_id)
                        if target_path is None:
                            create_object(target_id, path)
                        else:
                            parent[name] = h5file[target_path]
                            self.stats.links += 1
                        continue

                    if link_type == "soft":
                        parent[name] = h5py.SoftLink(
                            str(link.get("target_path") or "/")
                        )
                        self.stats.links += 1
                        continue

                    if link_type == "external":
                        if link.get("target_file_was_absolute"):
                            self.warn(
                                f"{output.name}:{path} originally used an "
                                "absolute ExternalLink filename; the converter "
                                "preserved only its basename for portability"
                            )
                        parent[name] = h5py.ExternalLink(
                            str(
                                link.get("target_file")
                                or "missing_external.h5"
                            ),
                            str(link.get("target_path") or "/"),
                        )
                        self.stats.links += 1
                        continue

                    if link_type == "unreadable":
                        raise RestoreError(
                            "Exact restoration cannot reproduce unreadable "
                            f"source link {output.name}:{path}: "
                            f"{link.get('error')}"
                        )

                    raise RestoreError(
                        f"Unsupported local link type {link_type!r}: "
                        f"{output.name}:{path}"
                    )

            create_children("/")

            if len(created) != len(objects):
                missing = [
                    (
                        object_id,
                        record.get("canonical_path")
                        if isinstance(record, dict)
                        else None,
                    )
                    for object_id, record in objects.items()
                    if object_id not in created
                ]
                raise RestoreError(
                    f"External object graph is disconnected in "
                    f"{output.name}; uncreated={missing[:12]!r}"
                )

    def _prepare_files(self) -> None:
        self._prepare_destination()
        self._copy_master_template()

        for source_id, descriptor in self.external_by_source.items():
            self._create_external_file_structure(
                source_id,
                descriptor,
            )

    @staticmethod
    def _assign_member(
        dataset: h5py.Dataset,
        member: dict[str, Any],
        value: Any,
    ) -> None:
        start = member.get("start")
        count = member.get("count")
        if start is None:
            dataset[()] = value
            return
        starts = [int(x) for x in start]
        counts = [int(x) for x in count]
        selection = tuple(
            slice(start0, start0 + count0)
            for start0, count0 in zip(starts, counts)
        )
        array = np.asarray(value)
        dataset[selection] = array.reshape(tuple(counts))

    def _write_raw_calibration(
        self,
        source_id: str,
        path: str,
        payload: dict[str, Any],
    ) -> None:
        members = payload.get("members")
        if not isinstance(members, list):
            return

        output = self.source_paths[source_id]
        with h5py.File(output, "r+") as h5file:
            dataset = h5file[path]
            for member in members:
                if not isinstance(member, dict):
                    continue
                raw = self.archive.read_member(member)
                offset = tuple(
                    int(x)
                    for x in member.get(
                        "chunk_offset",
                        member.get("start", []),
                    )
                )
                filter_mask = int(
                    member.get("filter_mask", 0)
                )
                try:
                    dataset.id.write_direct_chunk(
                        offset,
                        raw,
                        filter_mask=filter_mask,
                    )
                except Exception as exc:
                    raise RestoreError(
                        f"Cannot write raw Calibration chunk "
                        f"{path}@{offset}: {exc}"
                    ) from exc

                if self.verify_output:
                    actual_mask, actual_raw = (
                        dataset.id.read_direct_chunk(offset)
                    )
                    if (
                        int(actual_mask) != filter_mask
                        or bytes(actual_raw) != raw
                    ):
                        raise RestoreError(
                            "Raw Calibration chunk verification failed: "
                            f"{path}@{offset}"
                        )
                self.stats.calibration_chunks += 1

    @staticmethod
    def _derived_predicate(
        source: np.ndarray,
        predicate: str,
    ) -> np.ndarray:
        if predicate == "nonzero":
            return source != 0
        if predicate == "positive":
            return source > 0
        if predicate == "negative":
            return source < 0
        if predicate == "equal-zero":
            return source == 0
        if predicate == "equal-one":
            return source == 1
        if predicate == "equal-minus-one":
            return source == -1
        raise RestoreError(
            f"Unsupported derived predicate: {predicate!r}"
        )

    @staticmethod
    def _logical_array_sha256_exact(array: np.ndarray) -> str:
        contiguous = np.ascontiguousarray(array)
        return hashlib.sha256(
            memoryview(contiguous).cast("B")
        ).hexdigest()

    def _compute_derived_dataset(
        self,
        h5file: h5py.File,
        target: h5py.Dataset,
        recipe: dict[str, Any],
    ) -> np.ndarray:
        if str(recipe.get("format") or "") != "hdxf-derived-recipe":
            raise RestoreError(
                "Invalid derived Dataset recipe format"
            )
        if int(recipe.get("version", 0)) != 1:
            raise RestoreError(
                f"Unsupported derived Dataset recipe version: "
                f"{recipe.get('version')!r}"
            )

        op = str(recipe.get("op") or "")
        target_shape = (
            ()
            if target.shape == ()
            else tuple(int(x) for x in target.shape)
        )
        target_dtype = target.dtype

        if op == "constant-fill-v1":
            encoded = str(recipe.get("scalar_raw_base64") or "")
            try:
                scalar_raw = base64.b64decode(
                    encoded.encode("ascii"),
                    validate=True,
                )
            except Exception as exc:
                raise RestoreError(
                    f"Invalid constant derived scalar: {exc}"
                ) from exc

            itemsize = max(1, int(target_dtype.itemsize))
            if len(scalar_raw) != itemsize:
                raise RestoreError(
                    "Derived constant scalar byte-size mismatch"
                )
            element_count = (
                1
                if target.shape == ()
                else int(np.prod(target.shape, dtype=np.int64))
            )
            raw = scalar_raw * element_count
            value = np.frombuffer(
                raw,
                dtype=target_dtype,
                count=element_count,
            ).copy()
            value = value.reshape(target_shape)
        else:
            source_path = str(recipe.get("source_path") or "")
            if not source_path:
                raise RestoreError(
                    f"Derived recipe {op!r} has no source_path"
                )
            source_obj = h5file.get(source_path)
            if not isinstance(source_obj, h5py.Dataset):
                raise RestoreError(
                    f"Derived recipe dependency is missing: "
                    f"{h5file.filename}:{source_path}"
                )
            source = np.asarray(
                source_obj[...]
                if source_obj.shape != ()
                else source_obj[()]
            )

            prefix_ndim = int(
                recipe.get("reduction_prefix_ndim", 0)
            )
            if prefix_ndim < 0 or prefix_ndim >= source.ndim:
                raise RestoreError(
                    f"Invalid derived reduction prefix_ndim="
                    f"{prefix_ndim} for {source_path}"
                )
            axes = tuple(
                range(prefix_ndim, source.ndim)
            )

            if op == "reduce-count-v1":
                predicate = str(
                    recipe.get("predicate") or ""
                )
                mask = self._derived_predicate(
                    source, predicate
                )
                calculated = np.count_nonzero(
                    mask, axis=axes
                )
            elif op == "reduce-any-v1":
                predicate = str(
                    recipe.get("predicate") or "nonzero"
                )
                mask = self._derived_predicate(
                    source, predicate
                )
                calculated = np.any(mask, axis=axes)
            elif op == "reduce-all-v1":
                predicate = str(
                    recipe.get("predicate") or "nonzero"
                )
                mask = self._derived_predicate(
                    source, predicate
                )
                calculated = np.all(mask, axis=axes)
            elif op == "reduce-sum-int-v1":
                calculated = np.sum(
                    source,
                    axis=axes,
                    dtype=np.int64,
                )
            else:
                raise RestoreError(
                    f"Unsupported derived Dataset operation: {op!r}"
                )

            try:
                value = np.asarray(
                    calculated,
                    dtype=target_dtype,
                ).reshape(target_shape)
            except Exception as exc:
                raise RestoreError(
                    f"Cannot reshape/cast derived result for "
                    f"{target.name}: {exc}"
                ) from exc
            value = np.ascontiguousarray(value)

        expected_hash = str(
            recipe.get("logical_sha256") or ""
        )
        if expected_hash:
            actual_hash = self._logical_array_sha256_exact(
                value
            )
            if actual_hash != expected_hash:
                raise RestoreError(
                    "Derived Dataset recipe verification failed before "
                    f"write: {target.name}; expected={expected_hash}, "
                    f"actual={actual_hash}"
                )
        return value

    def _write_derived_datasets(self) -> None:
        if not self.deferred_derived_datasets:
            return

        _progress(
            "[STAGE] Computing exact derived auxiliary Datasets"
        )

        pending: dict[
            tuple[str, str], dict[str, Any]
        ] = {
            (source_id, path): recipe
            for source_id, path, recipe
            in self.deferred_derived_datasets
        }

        while pending:
            progress = False
            for key in list(pending):
                source_id, path = key
                recipe = pending[key]
                dependency = str(
                    recipe.get("source_path") or ""
                )

                # Wait if the same-file dependency is itself still pending.
                if (
                    dependency
                    and (source_id, dependency) in pending
                ):
                    continue

                output = self.source_paths[source_id]
                with h5py.File(output, "r+") as h5file:
                    target = h5file[path]
                    if not isinstance(target, h5py.Dataset):
                        raise RestoreError(
                            f"Derived target is not a Dataset: "
                            f"{output.name}:{path}"
                        )
                    value = self._compute_derived_dataset(
                        h5file,
                        target,
                        recipe,
                    )
                    if target.shape == ():
                        # Keep the NumPy scalar representation so signed-zero
                        # and NaN payload bits are not routed through a Python
                        # float/int conversion.
                        target[()] = value.reshape(())[()]
                    else:
                        target[...] = value

                    # Re-read only when --verify was requested; the recipe hash
                    # itself is always checked before writing.
                    if self.verify_output:
                        restored = np.asarray(
                            target[...]
                            if target.shape != ()
                            else target[()]
                        )
                        actual_hash = (
                            self._logical_array_sha256_exact(
                                restored
                            )
                        )
                        expected_hash = str(
                            recipe.get(
                                "logical_sha256"
                            ) or ""
                        )
                        if (
                            expected_hash
                            and actual_hash != expected_hash
                        ):
                            raise RestoreError(
                                "Derived Dataset verification failed "
                                f"after write: {output.name}:{path}"
                            )

                del pending[key]
                self.stats.derived_datasets += 1
                progress = True

            if not progress:
                blocked = ", ".join(
                    f"{self.source_paths[source_id].name}:{path}"
                    for source_id, path in list(pending)[:8]
                )
                raise RestoreError(
                    "Derived Dataset dependency cycle or unresolved "
                    f"dependency: {blocked}"
                )


    def _write_generic_datasets(self) -> None:
        _progress(
            "[STAGE] Restoring exact external Dataset values"
        )

        for object_id, location in (
            self.archive_object_locations.items()
        ):
            obj = self.archive.objects.get(object_id)
            if not isinstance(obj, dict):
                raise RestoreError(
                    f"Archive object missing: {object_id}"
                )

            payload = obj.get("payload")
            if not isinstance(payload, dict):
                continue

            encoding = str(payload.get("encoding") or "")
            state = str(payload.get("state") or "present")
            if encoding in (
                "frame-sequence",
                "frame-sequence-indexed",
            ):
                continue

            source_id, path = location

            if encoding == "derived-recipe-v1":
                recipe = payload.get("recipe")
                if not isinstance(recipe, dict):
                    raise RestoreError(
                        f"Derived Dataset has no valid recipe: "
                        f"{self.source_paths[source_id].name}:{path}"
                    )
                self.deferred_derived_datasets.append(
                    (source_id, path, recipe)
                )
                continue
            if encoding == "calibration-raw-chunks-v1":
                self._write_raw_calibration(
                    source_id,
                    path,
                    payload,
                )
                continue

            if (
                encoding == "calibration-reference"
                or state == "external_reference"
            ):
                raise RestoreError(
                    "Exact restoration cannot leave an external-file "
                    f"Dataset as a Calibration reference: "
                    f"{self.source_paths[source_id].name}:{path}"
                )

            if state in ("null_dataspace", "empty"):
                continue

            members = payload.get("members")
            if not isinstance(members, list):
                continue

            output = self.source_paths[source_id]
            with h5py.File(output, "r+") as h5file:
                dataset = h5file[path]
                for member in members:
                    if not isinstance(member, dict):
                        continue
                    value = self.archive.decode_generic_member(
                        member
                    )
                    if _contains_refspec(value):
                        self.deferred_dataset_members.append(
                            (source_id, path, member, value)
                        )
                    else:
                        self._assign_member(
                            dataset,
                            member,
                            value,
                        )
                    self.stats.generic_members += 1

    def _write_frame_datasets(self) -> None:
        if (
            self.archive.frames_doc is None
            or not self.archive.blocks
        ):
            return

        _progress(
            "[STAGE] Decoding detector frame blocks into exact external Datasets"
        )
        total_blocks = len(self.archive.blocks)
        total_frames = int(
            self.archive.frames_doc.get("frame_count", 0)
        )

        previous_frame: np.ndarray | None = None
        previous_source_object: str | None = None
        done_frames = 0

        for index, block_doc in enumerate(
            self.archive.blocks, start=1
        ):
            source_object = str(
                block_doc.get("source_object") or ""
            )
            if not source_object:
                raise RestoreError(
                    f"Frame block {index} has no source_object"
                )

            location = self.archive_object_locations.get(
                source_object
            )
            if location is None:
                # A frame physically stored in the copied master does not need
                # to be rewritten; the template already contains the original
                # bytes/values.  Detector archives in this project normally use
                # external data files, so this is only a defensive fallback.
                object_doc = self.archive.objects.get(source_object)
                if (
                    isinstance(object_doc, dict)
                    and str(object_doc.get("source_file") or "")
                    == self.archive.main_source_id
                ):
                    continue
                raise RestoreError(
                    "Frame source object has no exact physical Dataset "
                    f"mapping: {source_object}"
                )

            requires_previous = bool(
                block_doc.get(
                    "requires_previous_frame", False
                )
            )
            if (
                source_object != previous_source_object
                or not requires_previous
            ):
                previous_frame = None

            block = self.archive.decode_frame_block(
                block_doc,
                previous_frame,
            )
            previous_frame = np.ascontiguousarray(block[-1])
            previous_source_object = source_object

            source_id, path = location
            output = self.source_paths[source_id]
            local_start = int(
                block_doc.get(
                    "start_frame",
                    block_doc.get("start", [0])[0],
                )
            )
            frame_count = int(
                block_doc.get("frame_count", block.shape[0])
            )

            with h5py.File(output, "r+") as h5file:
                dataset = h5file[path]
                dataset[
                    local_start:local_start + frame_count
                ] = block

                if self.verify_output:
                    restored = np.asarray(
                        dataset[
                            local_start:
                            local_start + frame_count
                        ]
                    )
                    if (
                        restored.dtype != block.dtype
                        or restored.shape != block.shape
                        or not np.array_equal(
                            restored,
                            block,
                            equal_nan=True,
                        )
                    ):
                        raise RestoreError(
                            "Frame verification failed for "
                            f"{output.name}:{path}"
                            f"[{local_start}:"
                            f"{local_start + frame_count}]"
                        )

            done_frames += frame_count
            self.stats.frame_blocks += 1
            self.stats.frames += frame_count

            if (
                index == 1
                or index == total_blocks
                or index % self.progress_every == 0
            ):
                pct = (
                    100.0 * done_frames / total_frames
                    if total_frames
                    else 100.0
                )
                _progress(
                    f"[RESTORE] frame block {index}/{total_blocks} "
                    f"({done_frames}/{total_frames} frames, "
                    f"{pct:.1f}%)"
                )

    def _resolve_deferred_references(self) -> None:
        if (
            not self.deferred_attrs
            and not self.deferred_dataset_members
        ):
            return

        _progress(
            "[STAGE] Resolving exact external HDF5 references"
        )

        for source_id, object_path, name, encoded, value in (
            self.deferred_attrs
        ):
            output = self.source_paths[source_id]
            with h5py.File(output, "r+") as h5file:
                try:
                    resolved = _resolve_refs(value, h5file)
                    _decoded, attr_record = (
                        self._attribute_record(encoded)
                    )
                    self._create_attribute_exact_value(
                        h5file[object_path],
                        name,
                        resolved,
                        attr_record,
                    )
                except Exception as exc:
                    raise RestoreError(
                        "Cannot restore reference attribute "
                        f"{output.name}:{object_path}@{name}: {exc}"
                    ) from exc

        for source_id, path, member, value in (
            self.deferred_dataset_members
        ):
            output = self.source_paths[source_id]
            with h5py.File(output, "r+") as h5file:
                try:
                    self._assign_member(
                        h5file[path],
                        member,
                        _resolve_refs(value, h5file),
                    )
                except Exception as exc:
                    raise RestoreError(
                        "Cannot restore reference Dataset member "
                        f"{output.name}:{path}: {exc}"
                    ) from exc

    @staticmethod
    def _decoded_attr_equal(actual: Any, expected: Any) -> bool:
        if isinstance(expected, np.ndarray):
            try:
                return np.array_equal(
                    np.asarray(actual),
                    expected,
                    equal_nan=True,
                )
            except TypeError:
                return np.array_equal(
                    np.asarray(actual),
                    expected,
                )
        if isinstance(expected, (bytes, bytearray)):
            try:
                return bytes(actual) == bytes(expected)
            except Exception:
                return False
        if isinstance(expected, float) and math.isnan(expected):
            try:
                return math.isnan(float(actual))
            except Exception:
                return False
        try:
            return bool(actual == expected)
        except Exception:
            return False

    def _verify_external_structure(self) -> None:
        _progress(
            "[STAGE] Verifying reconstructed HDF5 structure"
        )

        # The master itself must remain byte-identical to the validated
        # template.  We never open it r+ in exact mode.
        expected_master_sha = str(
            self.master_doc.get("sha256") or ""
        )
        actual_master_sha = _sha256_file(
            self.master_output
        )
        if actual_master_sha != expected_master_sha:
            raise RestoreError(
                "Restored master is not byte-identical to the original "
                f"template: expected={expected_master_sha}, "
                f"actual={actual_master_sha}"
            )

        for source_id, descriptor in (
            self.external_by_source.items()
        ):
            output = self.source_paths[source_id]
            objects = descriptor.get("objects") or {}
            links = descriptor.get("links") or []

            with h5py.File(output, "r") as h5file:
                for record in objects.values():
                    if not isinstance(record, dict):
                        continue
                    path = str(
                        record.get("canonical_path") or ""
                    )
                    object_type = str(
                        record.get("type") or ""
                    )
                    if path not in h5file:
                        raise RestoreError(
                            f"Missing restored object: "
                            f"{output.name}:{path}"
                        )
                    obj = h5file[path]
                    if (
                        object_type == "group"
                        and not isinstance(obj, h5py.Group)
                    ):
                        raise RestoreError(
                            f"Expected Group: {output.name}:{path}"
                        )
                    if object_type == "dataset":
                        if not isinstance(obj, h5py.Dataset):
                            raise RestoreError(
                                f"Expected Dataset: "
                                f"{output.name}:{path}"
                            )
                        storage = record.get(
                            "hdf5_storage"
                        )
                        if isinstance(storage, dict):
                            expected_shape = storage.get(
                                "shape"
                            )
                            actual_shape = (
                                None
                                if obj.shape is None
                                else list(obj.shape)
                            )
                            if actual_shape != expected_shape:
                                raise RestoreError(
                                    f"Shape mismatch "
                                    f"{output.name}:{path}: "
                                    f"expected={expected_shape}, "
                                    f"actual={actual_shape}"
                                )
                            expected_dtype = _dtype_from_info(
                                storage.get("dtype")
                            )
                            if obj.dtype != expected_dtype:
                                raise RestoreError(
                                    f"dtype mismatch "
                                    f"{output.name}:{path}: "
                                    f"expected={expected_dtype}, "
                                    f"actual={obj.dtype}"
                                )
                            expected_chunks = storage.get(
                                "chunks"
                            )
                            actual_chunks = (
                                None
                                if obj.chunks is None
                                else list(obj.chunks)
                            )
                            if actual_chunks != expected_chunks:
                                raise RestoreError(
                                    f"chunk mismatch "
                                    f"{output.name}:{path}: "
                                    f"expected={expected_chunks}, "
                                    f"actual={actual_chunks}"
                                )
                            expected_filters = [
                                item
                                for item in storage.get(
                                    "filters", []
                                )
                                if isinstance(item, dict)
                            ]
                            actual_filters = _filter_pipeline(
                                obj
                            )
                            if not _pipeline_equal(
                                actual_filters,
                                expected_filters,
                            ) and not _pipeline_compatible(
                                actual_filters,
                                expected_filters,
                            ):
                                raise RestoreError(
                                    f"filter pipeline mismatch "
                                    f"{output.name}:{path}: "
                                    f"expected={expected_filters}, "
                                    f"actual={actual_filters}"
                                )

                    attrs = record.get("attributes")
                    if isinstance(attrs, dict):
                        for name, encoded in attrs.items():
                            expected, attr_record = (
                                self._attribute_record(encoded)
                            )
                            if name not in obj.attrs:
                                raise RestoreError(
                                    f"Missing attribute "
                                    f"{output.name}:{path}@{name}"
                                )

                            schema_ok, schema_reason = (
                                self._attribute_schema_matches(
                                    obj,
                                    str(name),
                                    attr_record,
                                )
                            )
                            if not schema_ok:
                                raise RestoreError(
                                    f"Attribute schema mismatch "
                                    f"{output.name}:{path}@{name}: "
                                    f"{schema_reason}"
                                )

                            if _contains_refspec(expected):
                                continue

                            actual = obj.attrs[name]
                            if not self._decoded_attr_equal(
                                actual, expected
                            ):
                                raise RestoreError(
                                    f"Attribute mismatch "
                                    f"{output.name}:{path}@{name}: "
                                    f"expected={expected!r}, "
                                    f"actual={actual!r}"
                                )

                for link in links:
                    if not isinstance(link, dict):
                        continue
                    path = str(link.get("path") or "")
                    link_type = str(link.get("type") or "")
                    if not path:
                        continue
                    parent_path, name = path.rsplit("/", 1)
                    parent = h5file[parent_path or "/"]
                    actual = parent.get(
                        name,
                        getlink=True,
                    )

                    if link_type == "hard":
                        if not isinstance(
                            actual, h5py.HardLink
                        ):
                            raise RestoreError(
                                f"HardLink mismatch: "
                                f"{output.name}:{path}"
                            )
                    elif link_type == "soft":
                        if (
                            not isinstance(
                                actual, h5py.SoftLink
                            )
                            or str(actual.path)
                            != str(
                                link.get("target_path")
                                or "/"
                            )
                        ):
                            raise RestoreError(
                                f"SoftLink mismatch: "
                                f"{output.name}:{path}"
                            )
                    elif link_type == "external":
                        if not isinstance(
                            actual, h5py.ExternalLink
                        ):
                            raise RestoreError(
                                f"ExternalLink mismatch: "
                                f"{output.name}:{path}"
                            )
                        if (
                            str(actual.path)
                            != str(
                                link.get("target_path")
                                or "/"
                            )
                        ):
                            raise RestoreError(
                                f"ExternalLink target-path mismatch: "
                                f"{output.name}:{path}"
                            )

        # Verify that every master ExternalLink in the exact descriptor resolves
        # through the untouched copied master into the newly rebuilt data files.
        exact_links = self.legacy.get(
            "external_links"
        )
        if isinstance(exact_links, list):
            with h5py.File(
                self.master_output, "r"
            ) as master:
                for item in exact_links:
                    if not isinstance(item, dict):
                        continue
                    if str(
                        item.get("parent_source_file") or ""
                    ) != self.archive.main_source_id:
                        continue
                    link_path = str(
                        item.get("link_path") or ""
                    )
                    if not link_path:
                        continue
                    try:
                        resolved = master[link_path]
                        if isinstance(
                            resolved, h5py.Dataset
                        ):
                            _ = resolved.shape
                        elif isinstance(
                            resolved, h5py.Group
                        ):
                            _ = list(resolved.keys())
                    except Exception as exc:
                        raise RestoreError(
                            "Copied master ExternalLink does not resolve "
                            f"after rebuilding external files: "
                            f"{link_path}: {exc}"
                        ) from exc

    def run(self) -> Path:
        started = time.perf_counter()
        _progress(
            f"[START] HDXF -> exact HDF5: "
            f"{self.archive.path}"
        )
        _progress(
            f"[INFO] exact master template: "
            f"{self.template_master}"
        )
        _progress(
            f"[INFO] external files to rebuild: "
            f"{len(self.external_by_source)}"
        )
        if self.archive.frames_doc is not None:
            _progress(
                f"[INFO] detector frames: "
                f"{int(self.archive.frames_doc.get('frame_count', 0))}, "
                f"blocks: {len(self.archive.blocks)}"
            )

        self._prepare_files()
        self._write_generic_datasets()
        self._write_frame_datasets()
        self._write_derived_datasets()
        self._resolve_deferred_references()
        self._verify_external_structure()

        elapsed = time.perf_counter() - started
        _progress(
            f"[OUT master] {self.master_output}"
        )
        for source_id, path in self.source_paths.items():
            if source_id != self.archive.main_source_id:
                _progress(
                    f"[OUT external] {path}"
                )

        _progress(
            "[SUMMARY] "
            f"groups={self.stats.groups}, "
            f"datasets={self.stats.datasets}, "
            f"links={self.stats.links}, "
            f"frame_blocks={self.stats.frame_blocks}, "
            f"frames={self.stats.frames}, "
            f"generic_members={self.stats.generic_members}, "
            f"derived_datasets={self.stats.derived_datasets}, "
            f"calibration_chunks={self.stats.calibration_chunks}, "
            f"creation_property_fallbacks="
            f"{self.stats.creation_property_fallbacks}, "
            f"warnings={self.stats.warnings}, "
            f"elapsed={elapsed:.2f}s"
        )
        _progress(
            "[RESULT] PASS - exact HDF5 structure restoration completed"
        )
        return self.master_output


class Restorer:
    def __init__(
        self,
        archive: HDXFArchive,
        output_dir: Path,
        *,
        overwrite: bool,
        verify_output: bool,
        progress_every: int,
        allow_missing_calibration: bool,
        layout: str = "auto",
    ) -> None:
        self.archive = archive
        self.output_dir = output_dir.resolve()
        self.overwrite = bool(overwrite)
        self.verify_output = bool(verify_output)
        self.progress_every = max(1, int(progress_every))
        self.allow_missing_calibration = bool(allow_missing_calibration)
        self.layout_request = str(layout or "auto")
        self.stats = RestoreStats()
        self.warnings: list[str] = []
        self.source_paths: dict[str, Path] = {}
        self.actual_paths: dict[str, str] = {}
        self.external_prefixes: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.deferred_attrs: list[tuple[str, str, str, Any]] = []
        self.deferred_dataset_members: list[tuple[str, str, dict[str, Any], Any]] = []
        self.frame_object_ids = self._frame_object_ids()
        self.primary_frame_object = self._primary_frame_object_id()
        self.primary_frame_source = self._primary_frame_source_id()
        self.compact_single_frame = self._use_compact_single_frame_layout()
        self._build_source_paths()
        self._build_external_prefixes()
        self._build_actual_paths()

    def warn(self, message: str) -> None:
        self.stats.warnings += 1
        self.warnings.append(message)
        _progress(f"[WARN] {message}")

    def _frame_object_ids(self) -> set[str]:
        frames = self.archive.frames_doc
        if not isinstance(frames, dict):
            return set()
        return {
            str(item.get("object"))
            for item in frames.get("datasets", [])
            if isinstance(item, dict) and item.get("object")
        }

    def _primary_frame_object_id(self) -> str | None:
        if self.archive.blocks:
            object_id = str(self.archive.blocks[0].get("source_object") or "")
            if object_id:
                return object_id
        frames = self.archive.frames_doc
        if isinstance(frames, dict):
            for item in frames.get("datasets", []):
                if isinstance(item, dict) and item.get("object"):
                    return str(item["object"])
        return None

    def _primary_frame_source_id(self) -> str | None:
        if not self.primary_frame_object:
            return None
        obj = self.archive.objects.get(self.primary_frame_object)
        if not isinstance(obj, dict):
            return None
        source_id = str(obj.get("source_file") or "")
        return source_id or None

    def _use_compact_single_frame_layout(self) -> bool:
        if self.layout_request not in {"auto", "original", "single-frame"}:
            raise RestoreError(f"Unknown restore layout: {self.layout_request!r}")
        frames = self.archive.frames_doc
        frame_count = int(frames.get("frame_count", 0)) if isinstance(frames, dict) else 0
        can_compact = (
            frame_count == 1
            and bool(self.primary_frame_source)
            and self.primary_frame_source != self.archive.main_source_id
        )
        if self.layout_request == "single-frame":
            if not can_compact:
                raise RestoreError("--layout single-frame requires one externally stored detector frame")
            return True
        if self.layout_request == "original":
            return False
        return can_compact

    def _canonical_single_frame_base(self) -> str:
        stem = self.archive.path.stem
        if stem.lower().endswith("_master"):
            stem = stem[:-7]
        return stem or "restored"

    def _canonical_single_frame_master_name(self) -> str:
        return f"{self._canonical_single_frame_base()}_master.h5"

    def _canonical_single_frame_data_name(self) -> str:
        return f"{self._canonical_single_frame_base()}_data_000001.h5"

    def _source_in_scope(self, source_id: str) -> bool:
        if not self.compact_single_frame:
            return source_id in self.source_paths
        return source_id in {self.archive.main_source_id, self.primary_frame_source}

    def _build_source_paths(self) -> None:
        used: dict[str, str] = {}
        allowed: set[str] | None = None
        if self.compact_single_frame:
            allowed = {self.archive.main_source_id}
            if self.primary_frame_source:
                allowed.add(self.primary_frame_source)
        for entry in self.archive.source_files:
            source_id = str(entry.get("id") or "")
            if not source_id or (allowed is not None and source_id not in allowed):
                continue
            if self.compact_single_frame and source_id == self.primary_frame_source:
                rel = Path(self._canonical_single_frame_data_name())
            elif self.compact_single_frame and source_id == self.archive.main_source_id:
                rel = Path(self._canonical_single_frame_master_name())
            else:
                rel = _safe_rel_filename(str(entry.get("filename") or ""), source_id)
            key = os.path.normcase(str(rel))
            if key in used and used[key] != source_id:
                rel = rel.with_name(f"{rel.stem}_{source_id}{rel.suffix}")
                self.warn(f"Duplicate source filename was disambiguated as {rel}")
            used[os.path.normcase(str(rel))] = source_id
            self.source_paths[source_id] = self.output_dir / rel
        if self.archive.main_source_id not in self.source_paths:
            raise RestoreError("Manifest main source file is missing from source.files")
        if self.compact_single_frame and self.primary_frame_source not in self.source_paths:
            raise RestoreError("Single-frame detector source file is missing from source.files")

    def _build_external_prefixes(self) -> None:
        for link in self.archive.links:
            if str(link.get("type")) != "external":
                continue
            target_source = str(link.get("target_source_file") or "")
            alias = str(link.get("path") or "")
            target = str(link.get("target_path") or "")
            if target_source and alias and target:
                self.external_prefixes[target_source].append((alias.rstrip("/"), target.rstrip("/")))
        for source_id in self.external_prefixes:
            self.external_prefixes[source_id].sort(key=lambda item: len(item[0]), reverse=True)

    def _map_path(self, source_id: str, source_path: str) -> str:
        path = str(source_path or "/")
        for alias, target in self.external_prefixes.get(source_id, []):
            if path == alias:
                return target or "/"
            if path.startswith(alias + "/"):
                suffix = path[len(alias):]
                return (target + suffix) or "/"
        return path

    def _build_actual_paths(self) -> None:
        for object_id, obj in self.archive.objects.items():
            source_id = str(obj.get("source_file") or "")
            if self.compact_single_frame and object_id == self.primary_frame_object:
                self.actual_paths[object_id] = "/entry/data/data"
            else:
                self.actual_paths[object_id] = self._map_path(
                    source_id, str(obj.get("source_path") or "/")
                )

    @property
    def master_output(self) -> Path:
        return self.source_paths[self.archive.main_source_id]

    def _prepare_destination(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing = [path for path in self.source_paths.values() if path.exists()]
        if existing and not self.overwrite:
            listed = "\n".join(str(x) for x in existing[:8])
            raise RestoreError(f"Output files already exist; use --overwrite:\n{listed}")
        for path in self.source_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                path.unlink()

    def _create_files_and_objects(self) -> None:
        _progress("[STAGE] Creating HDF5 files and object graph")
        self._prepare_destination()
        for source_id, path in self.source_paths.items():
            with h5py.File(path, "w"):
                pass

        # Groups first, shallow to deep, separately per physical file.
        groups = [
            (oid, obj) for oid, obj in self.archive.objects.items()
            if str(obj.get("type")) == "group"
        ]
        groups.sort(key=lambda pair: (str(pair[1].get("source_file")), self.actual_paths[pair[0]].count("/")))
        for object_id, obj in groups:
            source_id = str(obj.get("source_file") or "")
            output = self.source_paths.get(source_id)
            if output is None:
                if not self.compact_single_frame:
                    self.warn(f"Skipping Group with unknown source file: {obj.get('source_path')}")
                continue
            path = self.actual_paths[object_id]
            with h5py.File(output, "r+") as h5file:
                group = h5file["/"] if path == "/" else h5file.require_group(path)
                self._apply_attributes(source_id, path, group, obj.get("attributes"))
            self.stats.groups += 1

        datasets = [
            (oid, obj) for oid, obj in self.archive.objects.items()
            if str(obj.get("type")) == "dataset"
        ]
        datasets.sort(key=lambda pair: (str(pair[1].get("source_file")), self.actual_paths[pair[0]].count("/")))
        for object_id, obj in datasets:
            source_id = str(obj.get("source_file") or "")
            output = self.source_paths.get(source_id)
            if output is None:
                if not self.compact_single_frame:
                    self.warn(f"Skipping Dataset with unknown source file: {obj.get('source_path')}")
                continue
            path = self.actual_paths[object_id]
            storage = obj.get("hdf5_storage")
            payload = obj.get("payload")
            if not isinstance(storage, dict):
                raise RestoreError(f"Dataset storage metadata missing: {path}")
            raw_calibration = isinstance(payload, dict) and str(payload.get("encoding")) == "calibration-raw-chunks-v1"
            with h5py.File(output, "r+") as h5file:
                dataset = _create_dataset_exact(
                    h5file, path, storage,
                    require_exact_filters=raw_calibration,
                    stats=self.stats,
                )
                self._apply_attributes(source_id, path, dataset, obj.get("attributes"))
            self.stats.datasets += 1

        if self.compact_single_frame:
            self._ensure_single_frame_data_skeleton()

    def _ensure_single_frame_data_skeleton(self) -> None:
        """Create the minimal external data-file hierarchy expected by Albula.

        Derived one-frame HDXF archives retain the original master metadata but
        often no longer contain Group objects for the selected external data
        file.  Recreate those parents and copy the corresponding master Group
        attributes, matching the traditional master + data_000001 layout.
        """
        if not self.compact_single_frame or not self.primary_frame_source:
            return
        data_output = self.source_paths[self.primary_frame_source]
        with h5py.File(self.master_output, "r") as master, h5py.File(data_output, "r+") as data:
            for group_path in ("/", "/entry", "/entry/data"):
                target = data["/"] if group_path == "/" else data.require_group(group_path)
                if group_path not in master:
                    continue
                source = master[group_path]
                for name, value in source.attrs.items():
                    try:
                        target.attrs[name] = value
                    except Exception:
                        pass
            # NXentry is essential for a conventional NeXus external data file.
            entry = data.require_group("/entry")
            if "NX_class" not in entry.attrs:
                entry.attrs["NX_class"] = np.bytes_(b"NXentry")
            data.require_group("/entry/data")

    def _apply_attributes(self, source_id: str, object_path: str, target: Any, attrs: Any) -> None:
        if not isinstance(attrs, dict):
            return
        for name, encoded in attrs.items():
            value = _decode_tagged(encoded)
            if value is None and isinstance(encoded, dict) and encoded.get("$type") in ("attribute_error", "python_repr"):
                self.warn(f"Attribute {object_path}@{name} could not be preserved by the source archive")
                continue
            if _contains_refspec(value):
                self.deferred_attrs.append((source_id, object_path, str(name), value))
                continue
            try:
                # h5py treats Python ``bytes`` attributes as variable-length
                # strings.  The source encoder tagged bytes explicitly, so use
                # np.bytes_ to restore a fixed-width byte-string attribute.
                if isinstance(value, (bytes, bytearray, memoryview)):
                    value = np.bytes_(bytes(value))
                target.attrs[str(name)] = value
            except Exception as exc:
                self.warn(f"Cannot restore attribute {object_path}@{name}: {type(exc).__name__}: {exc}")

    def _write_generic_datasets(self) -> None:
        _progress("[STAGE] Restoring preserved HDF5 datasets")
        objects = [
            (oid, obj) for oid, obj in self.archive.objects.items()
            if str(obj.get("type")) == "dataset"
        ]
        for object_id, obj in objects:
            payload = obj.get("payload")
            if not isinstance(payload, dict):
                continue
            encoding = str(payload.get("encoding") or "")
            state = str(payload.get("state") or "present")
            if encoding in ("frame-sequence", "frame-sequence-indexed"):
                continue
            if encoding == "calibration-raw-chunks-v1":
                self._write_raw_calibration(object_id, obj, payload)
                continue
            if encoding == "calibration-reference" or state == "external_reference":
                message = f"Referenced Calibration is not embedded for {obj.get('source_path')}"
                if self.allow_missing_calibration:
                    self.warn(message + "; leaving the Dataset at its fill value")
                    continue
                raise RestoreError(message + "; provide an embedded/hybrid HDXF or use --allow-missing-calibration")
            if state in ("null_dataspace", "empty"):
                continue
            members = payload.get("members")
            if not isinstance(members, list):
                continue
            source_id = str(obj.get("source_file") or "")
            output = self.source_paths.get(source_id)
            if output is None:
                continue
            path = self.actual_paths[object_id]
            with h5py.File(output, "r+") as h5file:
                dataset = h5file[path]
                for member in members:
                    if not isinstance(member, dict):
                        continue
                    value = self.archive.decode_generic_member(member)
                    if _contains_refspec(value):
                        self.deferred_dataset_members.append((source_id, path, member, value))
                    else:
                        self._assign_member(dataset, member, value)
                    self.stats.generic_members += 1

    @staticmethod
    def _assign_member(dataset: h5py.Dataset, member: dict[str, Any], value: Any) -> None:
        start = member.get("start")
        count = member.get("count")
        if start is None:
            dataset[()] = value
            return
        starts = [int(x) for x in start]
        counts = [int(x) for x in count]
        selection = tuple(slice(s, s + c) for s, c in zip(starts, counts))
        array = np.asarray(value)
        dataset[selection] = array.reshape(tuple(counts))

    def _write_raw_calibration(self, object_id: str, obj: dict[str, Any], payload: dict[str, Any]) -> None:
        source_id = str(obj.get("source_file") or "")
        output = self.source_paths.get(source_id)
        if output is None:
            if self.compact_single_frame:
                return
            raise RestoreError(f"Calibration source file is unavailable: {source_id}")
        path = self.actual_paths[object_id]
        members = payload.get("members")
        if not isinstance(members, list):
            return
        with h5py.File(output, "r+") as h5file:
            dataset = h5file[path]
            for member in members:
                if not isinstance(member, dict):
                    continue
                raw = self.archive.read_member(member)
                offset = tuple(int(x) for x in member.get("chunk_offset", member.get("start", [])))
                filter_mask = int(member.get("filter_mask", 0))
                try:
                    dataset.id.write_direct_chunk(offset, raw, filter_mask=filter_mask)
                except Exception as exc:
                    raise RestoreError(f"Cannot write raw Calibration chunk {path}@{offset}: {exc}") from exc
                if self.verify_output:
                    actual_mask, actual_raw = dataset.id.read_direct_chunk(offset)
                    if int(actual_mask) != filter_mask or bytes(actual_raw) != raw:
                        raise RestoreError(f"Raw Calibration chunk verification failed: {path}@{offset}")
                self.stats.calibration_chunks += 1

    def _write_frame_datasets(self) -> None:
        if self.archive.frames_doc is None or not self.archive.blocks:
            return
        _progress("[STAGE] Decoding detector frame blocks")
        total_blocks = len(self.archive.blocks)
        total_frames = int(self.archive.frames_doc.get("frame_count", 0))
        datasets = {
            str(item.get("object")): item
            for item in self.archive.frames_doc.get("datasets", [])
            if isinstance(item, dict) and item.get("object")
        }
        previous_frame: np.ndarray | None = None
        previous_source_object: str | None = None
        done_frames = 0
        for index, block_doc in enumerate(self.archive.blocks, start=1):
            source_object = str(block_doc.get("source_object") or "")
            if not source_object:
                raise RestoreError(f"Frame block {index} has no source_object")
            dataset_doc = datasets.get(source_object)
            object_doc = self.archive.objects.get(source_object)
            if dataset_doc is None or object_doc is None:
                raise RestoreError(f"Frame source object is missing: {source_object}")
            requires_previous = bool(block_doc.get("requires_previous_frame", False))
            if source_object != previous_source_object or not requires_previous:
                previous_frame = None
            block = self.archive.decode_frame_block(block_doc, previous_frame)
            previous_frame = np.ascontiguousarray(block[-1])
            previous_source_object = source_object

            source_id = str(object_doc.get("source_file") or "")
            output = self.source_paths.get(source_id)
            if output is None:
                raise RestoreError(f"Frame source file is outside the selected restore layout: {source_id}")
            path = self.actual_paths[source_object]
            local_start = int(block_doc.get("start_frame", block_doc.get("start", [0])[0]))
            frame_count = int(block_doc.get("frame_count", block.shape[0]))
            with h5py.File(output, "r+") as h5file:
                dataset = h5file[path]
                dataset[local_start:local_start + frame_count] = block
                if self.verify_output:
                    restored = np.asarray(dataset[local_start:local_start + frame_count])
                    if restored.dtype != block.dtype or restored.shape != block.shape or not np.array_equal(restored, block, equal_nan=True):
                        raise RestoreError(
                            f"Frame verification failed for {path}[{local_start}:{local_start + frame_count}]"
                        )
            done_frames += frame_count
            self.stats.frame_blocks += 1
            self.stats.frames += frame_count
            if index == 1 or index == total_blocks or index % self.progress_every == 0:
                pct = 100.0 * done_frames / total_frames if total_frames else 100.0
                _progress(
                    f"[RESTORE] frame block {index}/{total_blocks} "
                    f"({done_frames}/{total_frames} frames, {pct:.1f}%)"
                )

    def _link_source(self, link_path: str) -> tuple[str, str]:
        parent_alias = str(PurePosixPath(link_path).parent)
        if parent_alias == ".":
            parent_alias = "/"
        candidates: list[tuple[int, str, str]] = []
        for object_id, obj in self.archive.objects.items():
            if str(obj.get("type")) != "group":
                continue
            source_path = str(obj.get("source_path") or "/")
            if source_path == parent_alias:
                source_id = str(obj.get("source_file") or "")
                actual_parent = self.actual_paths[object_id]
                actual = actual_parent.rstrip("/") + "/" + PurePosixPath(link_path).name
                candidates.append((len(source_path), source_id, actual))
        if candidates:
            _, source_id, actual = sorted(candidates, reverse=True)[0]
            return source_id, actual
        # External subtree paths may need prefix mapping even if the parent group
        # was not a separately stored object.
        for source_id in self.source_paths:
            actual = self._map_path(source_id, link_path)
            if actual != link_path:
                return source_id, actual
        return self.archive.main_source_id, link_path

    def _create_links(self) -> None:
        _progress("[STAGE] Restoring HDF5 links")
        # External and soft links first. Hard links reference already-created objects.
        for link in self.archive.links:
            link_type = str(link.get("type") or "")
            link_path = str(link.get("path") or "")
            if not link_path or link_type == "unreadable":
                continue
            if self.compact_single_frame and link_type == "external" and (
                link_path == "/entry/data" or link_path.startswith("/entry/data/")
            ):
                continue
            source_id, actual_link_path = self._link_source(link_path)
            output = self.source_paths.get(source_id)
            if output is None:
                self.warn(f"Cannot place {link_type} link {link_path}: source file missing")
                continue
            parent_path, name = actual_link_path.rsplit("/", 1)
            with h5py.File(output, "r+") as h5file:
                parent = h5file.require_group(parent_path or "/")
                if name in parent:
                    del parent[name]
                if link_type == "soft":
                    parent[name] = h5py.SoftLink(str(link.get("target_path") or "/"))
                elif link_type == "external":
                    target_source = str(link.get("target_source_file") or "")
                    target_output = self.source_paths.get(target_source)
                    if target_output is not None:
                        target_filename = os.path.relpath(target_output, output.parent).replace("\\", "/")
                    else:
                        target_filename = str(link.get("target_file") or "missing_external.h5")
                    parent[name] = h5py.ExternalLink(
                        target_filename,
                        str(link.get("target_path") or "/"),
                    )
                elif link_type == "hard":
                    target_object = str(link.get("target_object") or "")
                    target_doc = self.archive.objects.get(target_object)
                    if target_doc is None:
                        self.warn(f"Hard-link target missing for {link_path}")
                        continue
                    target_source = str(target_doc.get("source_file") or "")
                    if target_source != source_id:
                        self.warn(f"Cross-file hard link cannot be restored: {link_path}")
                        continue
                    target_path = self.actual_paths[target_object]
                    parent[name] = h5file[target_path]
                else:
                    continue
            self.stats.links += 1

        if self.compact_single_frame:
            self._create_canonical_single_frame_link()

    def _create_canonical_single_frame_link(self) -> None:
        if not self.primary_frame_source:
            raise RestoreError("Single-frame data source is unavailable")
        data_output = self.source_paths[self.primary_frame_source]
        with h5py.File(self.master_output, "r+") as master:
            data_group = master.require_group("/entry/data")
            for name in list(data_group.keys()):
                del data_group[name]
            relative_name = os.path.relpath(data_output, self.master_output.parent).replace("\\", "/")
            data_group["data_000001"] = h5py.ExternalLink(relative_name, "/entry/data/data")
        self.stats.links += 1
        self._normalise_single_frame_metadata()

    def _normalise_single_frame_metadata(self) -> None:
        """Apply the one-image metadata convention used by Albula-style masters."""
        if not self.primary_frame_source:
            return
        scalar_paths = []
        for base in ("/entry/instrument/detector", "/entry/detector"):
            for name in (
                "nimages", "nimages_collected", "nimages_written",
                "detectorSpecific/nimages",
                "detectorSpecific/nimages_collected",
                "detectorSpecific/nimages_written",
            ):
                scalar_paths.append(f"{base}/{name}")
        with h5py.File(self.master_output, "r+") as master:
            for path in scalar_paths:
                if path not in master:
                    continue
                obj = master[path]
                if isinstance(obj, h5py.Dataset) and obj.shape == ():
                    try:
                        obj[()] = np.asarray(1, dtype=obj.dtype)
                    except Exception:
                        pass
        data_output = self.source_paths[self.primary_frame_source]
        with h5py.File(data_output, "r+") as data:
            if "/entry/data/data" not in data:
                raise RestoreError("Restored single-frame data file has no /entry/data/data Dataset")
            dataset = data["/entry/data/data"]
            if dataset.ndim != 3 or dataset.shape[0] != 1:
                raise RestoreError(f"Single-frame Dataset has invalid shape {dataset.shape}")
            for name in ("image_nr_low", "image_nr_high", "nimages"):
                dataset.attrs[name] = np.int32(1)

    def _resolve_deferred_references(self) -> None:
        if not self.deferred_attrs and not self.deferred_dataset_members:
            return
        _progress("[STAGE] Resolving HDF5 references")
        for source_id, object_path, name, value in self.deferred_attrs:
            output = self.source_paths.get(source_id)
            if output is None:
                continue
            with h5py.File(output, "r+") as h5file:
                try:
                    h5file[object_path].attrs[name] = _resolve_refs(value, h5file)
                except Exception as exc:
                    self.warn(f"Cannot restore reference attribute {object_path}@{name}: {exc}")
        for source_id, path, member, value in self.deferred_dataset_members:
            output = self.source_paths.get(source_id)
            if output is None:
                continue
            with h5py.File(output, "r+") as h5file:
                try:
                    self._assign_member(h5file[path], member, _resolve_refs(value, h5file))
                except Exception as exc:
                    self.warn(f"Cannot restore reference Dataset member {path}: {exc}")

    def _verify_links_and_objects(self) -> None:
        if not self.verify_output:
            return
        _progress("[STAGE] Verifying restored HDF5 structure")
        for source_id, output in self.source_paths.items():
            with h5py.File(output, "r") as h5file:
                # File readability and root object are the first structural check.
                _ = list(h5file.attrs.keys())
        if self.compact_single_frame:
            with h5py.File(self.master_output, "r") as master:
                data_group = master["/entry/data"]
                if list(data_group.keys()) != ["data_000001"]:
                    raise RestoreError(
                        f"Single-frame master must contain only /entry/data/data_000001; found {list(data_group.keys())}"
                    )
                link = data_group.get("data_000001", getlink=True)
                if not isinstance(link, h5py.ExternalLink):
                    raise RestoreError("Albula-compatible /entry/data/data_000001 ExternalLink is missing")
                if str(link.path) != "/entry/data/data":
                    raise RestoreError(f"ExternalLink target must be /entry/data/data, got {link.path!r}")
                try:
                    dataset = master["/entry/data/data_000001"]
                except Exception as exc:
                    raise RestoreError(f"Single-frame ExternalLink verification failed: {exc}") from exc
                if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 3 or dataset.shape[0] != 1:
                    raise RestoreError(f"Linked detector Dataset has invalid shape {getattr(dataset, 'shape', None)}")
                for name in ("image_nr_low", "image_nr_high", "nimages"):
                    if int(dataset.attrs.get(name, 1)) != 1:
                        raise RestoreError(f"Detector Dataset attribute {name} is not 1")
            return
        for link in self.archive.links:
            if str(link.get("type")) != "external":
                continue
            source_id, actual_link_path = self._link_source(str(link.get("path") or ""))
            output = self.source_paths[source_id]
            with h5py.File(output, "r") as h5file:
                try:
                    resolved = h5file[actual_link_path]
                    _ = resolved.shape if isinstance(resolved, h5py.Dataset) else list(resolved.keys())
                except Exception as exc:
                    raise RestoreError(f"External link verification failed {actual_link_path}: {exc}") from exc

    def run(self) -> Path:
        started = time.perf_counter()
        _progress(f"[START] HDXF -> HDF5: {self.archive.path}")
        _progress(f"[INFO] archive source files: {len(self.archive.source_files)}")
        _progress(
            f"[INFO] restore layout: {'single-frame Albula master + data_000001' if self.compact_single_frame else 'original'}"
        )
        _progress(f"[INFO] output files: {len(self.source_paths)}")
        if self.archive.frames_doc is not None:
            _progress(
                f"[INFO] detector frames: {int(self.archive.frames_doc.get('frame_count', 0))}, "
                f"blocks: {len(self.archive.blocks)}"
            )
        self._create_files_and_objects()
        self._write_generic_datasets()
        self._write_frame_datasets()
        self._create_links()
        self._resolve_deferred_references()
        self._verify_links_and_objects()
        elapsed = time.perf_counter() - started
        _progress(f"[OUT master] {self.master_output}")
        for source_id, path in self.source_paths.items():
            if source_id != self.archive.main_source_id:
                _progress(f"[OUT external] {path}")
        _progress(
            f"[SUMMARY] groups={self.stats.groups}, datasets={self.stats.datasets}, "
            f"links={self.stats.links}, frame_blocks={self.stats.frame_blocks}, "
            f"frames={self.stats.frames}, generic_members={self.stats.generic_members}, "
            f"calibration_chunks={self.stats.calibration_chunks}, "
            f"storage_fallbacks={self.stats.storage_fallbacks}, warnings={self.stats.warnings}, "
            f"elapsed={elapsed:.2f}s"
        )
        _progress("[RESULT] PASS - HDF5 restoration completed")
        return self.master_output

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Restore HDXF to HDF5. By default v0.4.6/v0.4.7/v0.4.8 archives are restored "
            "exactly by copying the SHA-validated original master template and "
            "rebuilding each external HDF5 file from legacy_hdf5 descriptors."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input .hdxf archive",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help=(
            "Output directory for the restored master and external HDF5 files"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing restored files",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Read back restored frame/chunk values in addition to structural "
            "verification (slower)"
        ),
    )
    parser.add_argument(
        "--no-payload-hash",
        action="store_true",
        help="Skip HDXF payload SHA-256 verification while reading",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print frame progress every N blocks (default: 25)",
    )
    parser.add_argument(
        "--restore-mode",
        choices=("exact", "auto", "manifest"),
        default="exact",
        help=(
            "exact requires legacy_hdf5 + the original master template; "
            "manifest uses the older from-manifest reconstruction; "
            "auto uses exact when legacy_hdf5 exists, otherwise manifest. "
            "Default: exact"
        ),
    )
    parser.add_argument(
        "--legacy-template-dir",
        type=Path,
        default=DEFAULT_LEGACY_TEMPLATE_DIR,
        help=(
            "Directory containing the original master HDF5 template. "
            "Default: auto-detect ./legacy_template or ./legacy-template next to this script"
        ),
    )
    parser.add_argument(
        "--legacy-template",
        type=Path,
        default=None,
        help=(
            "Optional direct path to the original master template. "
            "Overrides --legacy-template-dir."
        ),
    )

    # Older-manifest fallback options. These are intentionally not used by
    # exact v0.4.6/v0.4.7/v0.4.8 restoration.
    parser.add_argument(
        "--allow-missing-calibration",
        action="store_true",
        help=(
            "Manifest-mode only: allow referenced-only Calibration datasets "
            "to remain at fill values"
        ),
    )
    parser.add_argument(
        "--layout",
        choices=("original", "single-frame"),
        default="original",
        help=(
            "Manifest-mode only. Exact mode always restores the original "
            "master/external-file layout. Default: original"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input.resolve()

    if not input_path.is_file():
        raise SystemExit(
            f"Input HDXF does not exist: {input_path}"
        )
    if args.progress_every <= 0:
        raise SystemExit(
            "--progress-every must be greater than zero"
        )

    try:
        with HDXFArchive(
            input_path,
            verify_hashes=not args.no_payload_hash,
        ) as archive:
            requested_mode = str(args.restore_mode)

            if requested_mode == "auto":
                mode = (
                    "exact"
                    if archive.legacy_hdf5 is not None
                    else "manifest"
                )
            else:
                mode = requested_mode

            if mode == "exact":
                restorer = ExactLegacyRestorer(
                    archive,
                    args.output,
                    template_dir=args.legacy_template_dir,
                    template_file=args.legacy_template,
                    overwrite=args.overwrite,
                    verify_output=args.verify,
                    progress_every=args.progress_every,
                )
                restorer.run()
            else:
                _progress(
                    "[WARN] Using older manifest reconstruction mode. "
                    "This mode does not guarantee byte-identical master "
                    "restoration and is retained only for pre-v0.4.6 archives."
                )
                restorer = Restorer(
                    archive,
                    args.output,
                    overwrite=args.overwrite,
                    verify_output=args.verify,
                    progress_every=args.progress_every,
                    allow_missing_calibration=args.allow_missing_calibration,
                    layout=args.layout,
                )
                restorer.run()

        return 0

    except KeyboardInterrupt:
        print(
            "\n[RESULT] STOPPED - restoration interrupted",
            file=sys.stderr,
            flush=True,
        )
        return 130
    except Exception as exc:
        print(
            f"[RESULT] FAIL - {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
