#!/usr/bin/env python3
"""Restore an HDXF detector archive to HDF5.

Single-file restore entry point for HDXF 0.4.8+ embedded-master archives.

Key properties:
- No external master template is required for new archives.
- The original master HDF5 is extracted byte-for-byte from the HDXF archive.
- External HDF5 files are rebuilt from ``legacy_hdf5.external_files``.
- Detector frame payloads, including chained Delta blocks and binary frame
  indexes, are decoded with the same project codecs used by the Viewer.
- No frontend/backend wrapper or second restore script is used.

Project runtime dependencies:
    h5py, numpy, hdxf_codec.py, hdxf_index.py
Optional but strongly recommended for preserving detector HDF5 filters:
    hdf5plugin
"""
from __future__ import annotations

import argparse
from hdxf_release import version_banner
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import base64
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import h5py
import numpy as np
from hdxf_hardware import (snapshot as hardware_snapshot, reserve_mib,
                           gpu_pool_limit_mib, gpu_temperature)

# Register detector filters (Bitshuffle/LZ4, etc.) when available.  Logical
# restoration of normal Delta frame payloads can still work without it, but
# exact raw-HDF5-chunk restoration requires the original filter to exist.
try:
    import hdf5plugin  # type: ignore  # noqa: F401
    HDF5PLUGIN_AVAILABLE = True
except Exception:
    HDF5PLUGIN_AVAILABLE = False

try:
    from hdxf_codec import HDXFBError, decode_hdxfb
except Exception as exc:  # imported lazily by the actual payload paths below
    HDXFBError = RuntimeError  # type: ignore[assignment]
    decode_hdxfb = None  # type: ignore[assignment]
    _HDXF_CODEC_IMPORT_ERROR = exc
else:
    _HDXF_CODEC_IMPORT_ERROR = None

try:
    from hdxf_index import HDXFIError, unpack_frame_index
except Exception as exc:
    HDXFIError = RuntimeError  # type: ignore[assignment]
    unpack_frame_index = None  # type: ignore[assignment]
    _HDXF_INDEX_IMPORT_ERROR = exc
else:
    _HDXF_INDEX_IMPORT_ERROR = None


MANIFEST_PATH = "manifest.json"
DEFAULT_EMBEDDED_MASTER = "reconstruction/master.h5"
RESTORE_BUILD = "2026-10-14-parallel-external-restore-v5"


class RestoreError(RuntimeError):
    """Raised when an HDXF archive cannot be restored safely."""


def _apply_cuda_pool_limit(cuda_device: int, manual_limit_mib: int) -> int:
    """Protect shared VRAM; every spawned process must configure its own CuPy pool."""
    import cupy as cp
    with cp.cuda.Device(cuda_device):
        free, total = cp.cuda.runtime.memGetInfo()
        cap = gpu_pool_limit_mib(free_bytes=free, total_bytes=total,
                                 manual_limit_mib=manual_limit_mib)
        if cap == 0:
            raise RuntimeError("insufficient free CUDA memory for protected HDXF restore")
        cp.get_default_memory_pool().set_limit(size=cap * 1024 * 1024)
        return cap


def _resolve_restore_backend(requested: str, cuda_device: int,
                             gpu_memory_limit_mib: int = 0) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import cupy as cp  # Optional; only CUDA-capable NVIDIA GPUs are supported.
        devices = int(cp.cuda.runtime.getDeviceCount())
        if not 0 <= cuda_device < devices:
            raise RuntimeError(f"CUDA device {cuda_device} not present (count={devices})")
        with cp.cuda.Device(cuda_device):
            probe = cp.arange(4, dtype=cp.int64).cumsum()
            if cp.asnumpy(probe).tolist() != [0, 1, 3, 6]:
                raise RuntimeError("CUDA prefix-sum correctness probe failed")
        cap = _apply_cuda_pool_limit(cuda_device, gpu_memory_limit_mib)
        _progress(f"[GPU GUARD] CuPy pool capped at {cap} MiB (free VRAM reserved)")
        return "cuda-auto" if requested == "auto" else "cuda"
    except Exception as exc:
        if requested == "cuda":
            raise RestoreError(f"CUDA backend unavailable: {type(exc).__name__}: {exc}") from exc
        _progress(f"CUDA restore unavailable; CPU fallback: {type(exc).__name__}: {exc}")
        return "cpu"


def _progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(block_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_output_name(name: str, what: str) -> str:
    """Require an ordinary basename; never trust archive paths as output paths."""
    value = str(name or "").strip()
    if not value:
        raise RestoreError(f"missing {what} filename")
    posix = PurePosixPath(value.replace("\\", "/"))
    if posix.is_absolute() or len(posix.parts) != 1 or posix.name in ("", ".", ".."):
        raise RestoreError(f"unsafe {what} filename in manifest: {name!r}")
    return posix.name


def _read_member_checked(
    zf: zipfile.ZipFile,
    member: str,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> bytes:
    try:
        raw = zf.read(member)
    except KeyError as exc:
        raise RestoreError(f"missing HDXF member: {member}") from exc
    if expected_size is not None and len(raw) != int(expected_size):
        raise RestoreError(
            f"member size mismatch for {member}: expected {expected_size}, got {len(raw)}"
        )
    if expected_sha256:
        actual = _sha256_bytes(raw)
        if actual != expected_sha256:
            raise RestoreError(
                f"member SHA256 mismatch for {member}: expected {expected_sha256}, got {actual}"
            )
    return raw


def _load_manifest(zf: zipfile.ZipFile) -> dict[str, Any]:
    raw = _read_member_checked(zf, MANIFEST_PATH)
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RestoreError(f"manifest.json is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RestoreError("manifest.json root is not an object")
    return manifest


@dataclass(frozen=True)
class _ReferenceToken:
    kind: str
    target: str | None


def _decode_tagged(value: Any) -> Any:
    """Decode strict JSON values emitted by hdf5_to_hdxf.py."""
    if isinstance(value, list):
        return [_decode_tagged(item) for item in value]
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
        return complex(_decode_tagged(value.get("real")), _decode_tagged(value.get("imag")))
    if tag == "ndarray_npy":
        raw = base64.b64decode(str(value.get("base64", "")))
        return np.load(io.BytesIO(raw), allow_pickle=False)
    if tag == "object_array":
        shape = tuple(int(x) for x in value.get("shape", []))
        items = [_decode_tagged(item) for item in value.get("items", [])]
        return np.asarray(items, dtype=object).reshape(shape)
    if tag == "hdf5_reference":
        target = value.get("target")
        return _ReferenceToken("object", None if target is None else str(target))
    if tag == "hdf5_region_reference":
        target = value.get("target")
        return _ReferenceToken("region", None if target is None else str(target))
    if tag in ("datetime64", "timedelta64"):
        dtype = np.dtype(str(value.get("dtype")))
        return np.asarray(value.get("value"), dtype=dtype)[()]
    if tag == "python_repr":
        # This tag was only a last-resort preservation representation.  Keep the
        # textual form rather than evaluating arbitrary Python code.
        return str(value.get("value", ""))
    return {str(k): _decode_tagged(v) for k, v in value.items()}


def _tupleize_dtype_descr(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tupleize_dtype_descr(item) for item in value)
    return value


def _decode_numpy_dtype(value: Any) -> np.dtype[Any]:
    value = _decode_tagged(value)
    if isinstance(value, np.dtype):
        return value
    if isinstance(value, str):
        return np.dtype(value)
    if isinstance(value, list):
        # dtype_to_descr() emits a list of field descriptors for structured dtypes.
        return np.dtype([_tupleize_dtype_descr(item) for item in value])
    if isinstance(value, tuple):
        return np.dtype(value)
    raise RestoreError(f"unsupported NumPy dtype descriptor: {value!r}")


def _decode_hdf5_dtype(info: Any) -> np.dtype[Any]:
    if isinstance(info, dict) and "numpy" in info:
        base = _decode_numpy_dtype(info.get("numpy"))
        special = info.get("hdf5_special")
        if isinstance(special, dict):
            cls = special.get("class")
            if cls == "vlen_string":
                return h5py.string_dtype(encoding=str(special.get("encoding") or "utf-8"))
            if cls == "vlen_bytes":
                return h5py.special_dtype(vlen=bytes)
            if cls == "vlen":
                try:
                    base_descr = special.get("base")
                    vbase = _decode_numpy_dtype(base_descr)
                    return h5py.vlen_dtype(vbase)
                except Exception:
                    return base
            if cls == "enum":
                members = special.get("members") or {}
                try:
                    return h5py.enum_dtype(
                        {str(k): int(v) for k, v in dict(members).items()}, basetype=base
                    )
                except Exception:
                    return base
            if cls == "reference":
                return h5py.ref_dtype
            if cls == "region_reference":
                return h5py.regionref_dtype
        return base
    return _decode_numpy_dtype(info)


def _resolve_references(value: Any, h5file: h5py.File) -> Any:
    if isinstance(value, _ReferenceToken):
        if not value.target:
            return h5py.Reference() if value.kind == "object" else h5py.RegionReference()
        try:
            target = h5file[value.target]
        except Exception:
            return h5py.Reference() if value.kind == "object" else h5py.RegionReference()
        if value.kind == "object":
            return target.ref
        if isinstance(target, h5py.Dataset):
            try:
                return target.regionref[...]
            except Exception:
                pass
        return h5py.RegionReference()
    if isinstance(value, np.ndarray) and value.dtype == object:
        out = np.empty(value.shape, dtype=object)
        for index in np.ndindex(value.shape):
            out[index] = _resolve_references(value[index], h5file)
        return out
    if isinstance(value, list):
        return [_resolve_references(item, h5file) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_references(item, h5file) for item in value)
    return value


def _create_exact_fixed_string_attribute(
    obj: h5py.Group | h5py.Dataset,
    name: str,
    record: dict[str, Any],
    value: Any,
) -> bool:
    """Restore fixed-length HDF5 string type semantics when recorded.

    h5py's high-level ``attrs.create(..., dtype='S...')`` normally creates
    NULLPAD fixed strings.  The source Jungfrau master/external files often use
    NULLTERM instead.  HDXF records the low-level HDF5 string ``cset`` and
    ``strpad`` values, so recreate those attributes through the low-level API.
    """
    type_info = record.get("hdf5_type")
    if not isinstance(type_info, dict):
        return False
    if type_info.get("kind") != "string" or bool(type_info.get("is_variable_str")):
        return False

    dtype = _decode_hdf5_dtype(record.get("dtype"))
    if np.dtype(dtype).kind != "S":
        return False

    dataspace = record.get("dataspace") if isinstance(record.get("dataspace"), dict) else {}
    shape = record.get("shape")
    type_id = None
    space_id = None
    attr_id = None
    try:
        type_id = h5py.h5t.py_create(np.dtype(dtype), logical=True)
        if "size" in type_info:
            type_id.set_size(int(type_info["size"]))
        if "cset" in type_info:
            type_id.set_cset(int(type_info["cset"]))
        if "strpad" in type_info:
            type_id.set_strpad(int(type_info["strpad"]))

        ds_class = str(dataspace.get("class") or "")
        if ds_class == "null":
            space_id = h5py.h5s.create(h5py.h5s.NULL)
        elif shape is None or shape == [] or ds_class == "scalar":
            space_id = h5py.h5s.create(h5py.h5s.SCALAR)
        else:
            dims = tuple(int(x) for x in shape)
            space_id = h5py.h5s.create_simple(dims)

        attr_id = h5py.h5a.create(obj.id, str(name).encode("utf-8"), type_id, space_id)
        if ds_class != "null":
            if shape is not None and shape != []:
                array = np.asarray(value, dtype=dtype).reshape(tuple(int(x) for x in shape))
            else:
                array = np.asarray(value, dtype=dtype).reshape(())
            attr_id.write(array)
        return True
    except Exception:
        # If H5Acreate succeeded but writing failed, remove the partial
        # attribute before the caller falls back to the high-level API.
        try:
            if attr_id is not None:
                attr_id.close()
                attr_id = None
        except Exception:
            pass
        try:
            if str(name) in obj.attrs:
                del obj.attrs[str(name)]
        except Exception:
            pass
        return False
    finally:
        for resource in (attr_id, space_id, type_id):
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    pass


def _set_attributes(obj: h5py.Group | h5py.Dataset, encoded: Any, h5file: h5py.File) -> None:
    if not isinstance(encoded, dict):
        return
    for name, record in encoded.items():
        if not isinstance(record, dict) or record.get("$type") != "hdf5_attribute_v2":
            # Compatibility with older manifests that stored plain encoded values.
            try:
                obj.attrs[str(name)] = _resolve_references(_decode_tagged(record), h5file)
            except Exception as exc:
                raise RestoreError(f"cannot restore attribute {obj.name}@{name}: {exc}") from exc
            continue

        dtype = _decode_hdf5_dtype(record.get("dtype"))
        dataspace = record.get("dataspace") if isinstance(record.get("dataspace"), dict) else {}
        value = _resolve_references(_decode_tagged(record.get("value")), h5file)

        # Preserve fixed-length HDF5 string cset/strpad exactly when possible.
        # This removes the NULLTERM -> NULLPAD difference found by the ultra
        # comparison of reconstructed HDF5 files.
        if _create_exact_fixed_string_attribute(obj, str(name), record, value):
            continue

        if dataspace.get("class") == "null":
            try:
                obj.attrs.create(str(name), h5py.Empty(dtype))
                continue
            except Exception:
                pass

        shape = record.get("shape")
        try:
            if shape is not None and shape != []:
                array = np.asarray(value, dtype=dtype).reshape(tuple(int(x) for x in shape))
                obj.attrs.create(str(name), array, dtype=dtype)
            else:
                # Keep scalar strings/bytes scalar instead of turning them into
                # one-element arrays.
                obj.attrs.create(str(name), value, dtype=dtype)
        except Exception:
            try:
                obj.attrs[str(name)] = value
            except Exception as exc:
                raise RestoreError(f"cannot restore attribute {obj.name}@{name}: {exc}") from exc


def _storage_shape(storage: dict[str, Any]) -> tuple[int, ...] | None:
    raw = storage.get("shape")
    if raw is None:
        return None
    return tuple(int(x) for x in raw)


def _storage_maxshape(storage: dict[str, Any]) -> tuple[int | None, ...] | None:
    raw = storage.get("maxshape")
    if raw is None:
        return None
    return tuple(None if x is None else int(x) for x in raw)


def _filter_creation_values(
    filter_id: int,
    stored_values: Iterable[int],
    dtype: np.dtype[Any],
) -> tuple[int, ...]:
    """Convert stored HDF5 filter cd_values back to user creation options.

    Some HDF5 filters use ``set_local`` during H5Dcreate and rewrite their
    cd_values.  Bitshuffle (32008) reserves the first three stored values for
    its version and the Dataset element size, then appends the user options
    (block-size, compressor, ...).  Feeding the complete stored vector back
    into H5Pset_filter makes set_local prepend a *second* reserved triplet and
    produces the wrong encoder configuration.

    For Bitshuffle, restore only the user portion.  Other filters are currently
    passed through unchanged.
    """
    values = tuple(int(v) for v in stored_values)
    if int(filter_id) == 32008 and len(values) >= 3:
        # Registered Bitshuffle format: cd[0:3] are reserved and generated by
        # bshuf_h5_set_local; cd[3:] are the caller supplied creation options.
        return values[3:]
    return values


def _create_dataset_with_filter_pipeline(
    h5file: h5py.File,
    path: str,
    storage: dict[str, Any],
) -> h5py.Dataset:
    """Create a chunked Dataset using the recorded low-level HDF5 filter pipeline.

    This path is required for ``write_direct_chunk`` restoration. A raw filtered
    chunk is only meaningful when the destination Dataset declares the same
    filter pipeline that produced the source chunk bytes.
    """
    shape = _storage_shape(storage)
    if shape is None or len(shape) == 0:
        raise RestoreError(f"raw filtered-chunk Dataset must be non-scalar: {path}")

    chunks_raw = storage.get("chunks")
    if not isinstance(chunks_raw, list) or not chunks_raw:
        raise RestoreError(f"raw filtered-chunk Dataset has no chunk shape: {path}")
    chunks = tuple(int(x) for x in chunks_raw)
    dtype = _decode_hdf5_dtype(storage.get("dtype"))

    parent_path, name = path.rsplit("/", 1) if "/" in path[1:] else ("/", path.strip("/"))
    parent = h5file.require_group(parent_path)

    maxshape = _storage_maxshape(storage)
    if maxshape is None:
        maxshape = shape
    maxdims = tuple(
        h5py.h5s.UNLIMITED if value is None else int(value)
        for value in maxshape
    )
    space = h5py.h5s.create_simple(tuple(int(x) for x in shape), maxdims)
    type_id = h5py.h5t.py_create(np.dtype(dtype), logical=True)
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_chunk(chunks)

    filters = storage.get("filters")
    if not isinstance(filters, list) or not filters:
        raise RestoreError(f"raw filtered-chunk Dataset has no filter pipeline: {path}")

    for record in filters:
        if not isinstance(record, dict):
            continue
        filter_id = int(record.get("id"))
        flags = int(record.get("flags", 0))
        values = tuple(int(v) for v in (record.get("values") or []))
        create_values = _filter_creation_values(filter_id, values, np.dtype(dtype))

        if filter_id == 32008:
            _progress(
                f"HDF5 filter 32008 for {path}: "
                f"stored_cd={list(values)} -> create_opts={list(create_values)}"
            )

        # A restored detector Dataset must be encoded with the same filter
        # pipeline as the source.  Never silently fall back to an uncompressed
        # 100+ GiB logical Dataset when a plugin is missing.
        try:
            available = bool(h5py.h5z.filter_avail(filter_id))
        except Exception:
            available = False
        if not available:
            filter_name = str(record.get("name") or f"filter {filter_id}")
            hint = (
                " Install hdf5plugin in this venv."
                if filter_id == 32008
                else " Install/enable the original HDF5 filter plugin."
            )
            raise RestoreError(
                f"required HDF5 filter is unavailable for {path}: "
                f"id={filter_id}, name={filter_name!r}.{hint}"
            )
        try:
            config = int(h5py.h5z.get_filter_info(filter_id))
            if not (config & int(h5py.h5z.FILTER_CONFIG_ENCODE_ENABLED)):
                raise RestoreError(
                    f"HDF5 filter {filter_id} is present but encoding is disabled for {path}"
                )
        except RestoreError:
            raise
        except Exception:
            # Some older HDF5 builds do not expose full filter capability info.
            # set_filter/create/write below remains authoritative.
            pass

        try:
            dcpl.set_filter(filter_id, flags, create_values)
        except Exception as exc:
            raise RestoreError(
                f"cannot register HDF5 filter {filter_id} for {path}: {exc}. "
                "Install/enable the same HDF5 filter plugin used by the source "
                "(Jungfrau Bitshuffle data normally requires hdf5plugin)."
            ) from exc

    fillvalue = _decode_tagged(storage.get("fillvalue"))
    if fillvalue is not None and not isinstance(fillvalue, _ReferenceToken):
        try:
            scalar = np.asarray(fillvalue, dtype=dtype).reshape(())
            dcpl.set_fill_value(scalar)
        except Exception:
            pass

    try:
        dset_id = h5py.h5d.create(
            parent.id,
            name.encode("utf-8"),
            type_id,
            space,
            dcpl=dcpl,
        )
        dataset = h5py.Dataset(dset_id)

        # Verify the *effective* filter values after H5Dcreate.  This proves
        # that set_local ran and catches the exact failure mode that previously
        # produced ~3.1 GiB uncompressed detector files.
        try:
            effective_plist = dataset.id.get_create_plist()
            for filter_index in range(effective_plist.get_nfilters()):
                eff_id, _eff_flags, eff_values, _eff_name = effective_plist.get_filter(filter_index)
                if int(eff_id) != 32008:
                    continue
                effective_values = tuple(int(v) for v in eff_values)
                _progress(
                    f"HDF5 filter 32008 effective for {path}: "
                    f"effective_cd={list(effective_values)}"
                )
                if len(effective_values) < 3:
                    raise RestoreError(
                        f"Bitshuffle set_local did not produce reserved cd_values for {path}: "
                        f"{list(effective_values)}"
                    )
                if int(effective_values[2]) != int(np.dtype(dtype).itemsize):
                    raise RestoreError(
                        f"Bitshuffle element-size mismatch for {path}: "
                        f"effective cd[2]={effective_values[2]}, dtype.itemsize={np.dtype(dtype).itemsize}"
                    )
                expected_user = _filter_creation_values(32008, effective_values, np.dtype(dtype))
                if tuple(expected_user) != tuple(create_values):
                    raise RestoreError(
                        f"Bitshuffle effective options mismatch for {path}: "
                        f"requested={list(create_values)}, effective_user={list(expected_user)}"
                    )
        except RestoreError:
            raise
        except Exception as exc:
            raise RestoreError(
                f"cannot verify effective HDF5 filter pipeline for {path}: {exc}"
            ) from exc

        return dataset
    except Exception as exc:
        raise RestoreError(
            f"cannot create filtered Dataset required for raw-chunk restoration: "
            f"{path}: {exc}"
        ) from exc


def _create_dataset_from_storage(
    h5file: h5py.File,
    path: str,
    storage: dict[str, Any],
    *,
    raw_chunk_payload: bool = False,
) -> h5py.Dataset:
    shape = _storage_shape(storage)
    if shape is None:
        # A null dataspace Dataset is represented using h5py.Empty.
        dtype = _decode_hdf5_dtype(storage.get("dtype"))
        parent, name = path.rsplit("/", 1) if "/" in path[1:] else ("/", path.strip("/"))
        group = h5file.require_group(parent)
        return group.create_dataset(name, data=h5py.Empty(dtype))

    dtype = _decode_hdf5_dtype(storage.get("dtype"))
    parent_path, name = path.rsplit("/", 1) if "/" in path[1:] else ("/", path.strip("/"))
    parent = h5file.require_group(parent_path)

    kwargs: dict[str, Any] = {}
    maxshape = _storage_maxshape(storage)
    chunks = storage.get("chunks")
    if chunks is not None and len(shape) > 0:
        kwargs["chunks"] = tuple(int(x) for x in chunks)
        if maxshape is not None:
            kwargs["maxshape"] = maxshape
    # Important: do NOT pass maxshape for a source contiguous Dataset.  h5py
    # interprets any maxshape argument as an extensible Dataset request and
    # silently changes the layout to chunked even when maxshape == shape.

    compression = storage.get("compression")
    compression_options = _decode_tagged(storage.get("compression_options"))
    if compression not in (None, "none", "None") and len(shape) > 0:
        kwargs["compression"] = compression
        if compression_options is not None:
            if isinstance(compression_options, list):
                compression_options = tuple(compression_options)
            kwargs["compression_opts"] = compression_options
    if bool(storage.get("shuffle")) and len(shape) > 0:
        kwargs["shuffle"] = True
    if bool(storage.get("fletcher32")) and len(shape) > 0:
        kwargs["fletcher32"] = True
    if storage.get("scaleoffset") is not None and len(shape) > 0:
        kwargs["scaleoffset"] = storage.get("scaleoffset")

    fillvalue = _decode_tagged(storage.get("fillvalue"))
    if fillvalue is not None and not isinstance(fillvalue, _ReferenceToken):
        try:
            kwargs["fillvalue"] = fillvalue
        except Exception:
            pass

    # Any recorded low-level filter pipeline is authoritative.  This includes
    # Jungfrau Bitshuffle/LZ4 (filter 32008), which h5py reports as
    # compression="unknown".  Creating it through high-level create_dataset()
    # would fail and the old fallback then produced an enormous uncompressed
    # file.  Recreate the exact pipeline first, then normal Dataset writes will
    # be compressed by HDF5 chunk-by-chunk in real time.
    recorded_filters = storage.get("filters")
    if isinstance(recorded_filters, list) and recorded_filters and len(shape) > 0:
        return _create_dataset_with_filter_pipeline(h5file, path, storage)

    try:
        return parent.create_dataset(name, shape=shape, dtype=dtype, **kwargs)
    except Exception as first_exc:
        if raw_chunk_payload:
            raise RestoreError(
                f"cannot recreate filtered Dataset required for raw-chunk restoration: {path}: {first_exc}. "
                "Install/enable the original HDF5 filter plugin (for Jungfrau data, hdf5plugin is normally required)."
            ) from first_exc

        # Physical HDF5 filter/layout identity is secondary to lossless logical
        # restoration.  If an optional source filter is unavailable, rebuild the
        # Dataset uncompressed while preserving shape/dtype/data/attributes.
        fallback: dict[str, Any] = {}
        if chunks is not None and len(shape) > 0:
            fallback["chunks"] = tuple(int(x) for x in chunks)
            if maxshape is not None:
                fallback["maxshape"] = maxshape
        if fillvalue is not None and not isinstance(fillvalue, _ReferenceToken):
            fallback["fillvalue"] = fillvalue
        try:
            return parent.create_dataset(name, shape=shape, dtype=dtype, **fallback)
        except Exception as second_exc:
            raise RestoreError(
                f"cannot create Dataset {path}: original={first_exc}; fallback={second_exc}"
            ) from second_exc


def _decode_generic_member(zf: zipfile.ZipFile, member: dict[str, Any]) -> Any:
    path = str(member.get("path") or "")
    encoding = str(member.get("encoding") or "")
    raw = _read_member_checked(
        zf,
        path,
        expected_size=member.get("size_bytes") if isinstance(member.get("size_bytes"), int) else None,
        expected_sha256=member.get("sha256") if isinstance(member.get("sha256"), str) else None,
    )
    if encoding == "npy":
        return np.load(io.BytesIO(raw), allow_pickle=False)
    if encoding == "json":
        try:
            doc = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise RestoreError(f"invalid JSON payload {path}: {exc}") from exc
        return _decode_tagged(doc.get("data"))
    if encoding == "array-block-v1":
        if decode_hdxfb is None:
            raise RestoreError(f"hdxf_codec.py is required to decode {path}: {_HDXF_CODEC_IMPORT_ERROR}")
        try:
            array, _header = decode_hdxfb(raw, member_path=path, verify_hash=True)
        except Exception as exc:
            raise RestoreError(f"cannot decode HDXFB array member {path}: {exc}") from exc
        return array
    raise RestoreError(f"unsupported Dataset payload encoding: {encoding!r} ({path})")


def _slice_from_member(member: dict[str, Any]) -> tuple[slice, ...] | None:
    start = member.get("start")
    count = member.get("count")
    if start is None or count is None:
        return None
    return tuple(slice(int(s), int(s) + int(c)) for s, c in zip(start, count))


def _write_regular_payload(
    zf: zipfile.ZipFile,
    dataset: h5py.Dataset,
    payload: dict[str, Any],
) -> None:
    state = str(payload.get("state") or "present")
    if state in ("null_dataspace", "empty"):
        return
    members = payload.get("members")
    if not isinstance(members, list):
        raise RestoreError(f"Dataset {dataset.name} has no payload members")
    for member in members:
        if not isinstance(member, dict):
            continue
        value = _decode_generic_member(zf, member)
        selection = _slice_from_member(member)
        if dataset.shape == ():
            dataset[()] = value
        elif selection is None:
            dataset[...] = value
        else:
            dataset[selection] = value


def _unpack_bits_lsb(raw: np.ndarray, bit_width: int, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.uint64)
    if bit_width == 0:
        return np.zeros((count,), dtype=np.uint64)
    bits = np.unpackbits(np.asarray(raw, dtype=np.uint8).reshape(-1), bitorder="little")
    required = int(count) * int(bit_width)
    if bits.size < required:
        raise RestoreError("truncated zero-RLE bit-packed stream")
    bits = bits[:required].reshape(count, bit_width).astype(np.uint64)
    shifts = (np.uint64(1) << np.arange(bit_width, dtype=np.uint64))[None, :]
    return np.sum(bits * shifts, axis=1, dtype=np.uint64)


def _zigzag_decode_u64(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint64)
    positive = (values & np.uint64(1)) == 0
    out = np.empty(values.shape, dtype=np.int64)
    out[positive] = (values[positive] >> np.uint64(1)).astype(np.int64)
    neg = values[~positive] >> np.uint64(1)
    out[~positive] = -neg.astype(np.int64) - 1
    return out


class _FrameDecoder:
    def __init__(self, zf: zipfile.ZipFile, manifest: dict[str, Any], *,
                 delta_backend: str = "cpu", cuda_device: int = 0) -> None:
        self.delta_backend = delta_backend
        self.cuda_device = int(cuda_device)
        self._next_gpu_temp_check = 0.0
        self.zf = zf
        detector = manifest.get("detector_archive")
        if not isinstance(detector, dict):
            self.frames: dict[str, Any] | None = None
            self.blocks: list[dict[str, Any]] = []
            self.dataset_docs: dict[str, dict[str, Any]] = {}
            self.static_models: dict[str, dict[str, Any]] = {}
            self.static_cache: dict[str, dict[str, np.ndarray]] = {}
            return
        frames = detector.get("frames")
        if not isinstance(frames, dict):
            self.frames = None
            self.blocks = []
            self.dataset_docs = {}
            self.static_models = {}
            self.static_cache = {}
            return
        self.frames = frames
        self.blocks = self._load_blocks(frames)
        self.dataset_docs = {
            str(item.get("object")): item
            for item in frames.get("datasets", [])
            if isinstance(item, dict) and item.get("object")
        }
        self.static_models = {
            str(item.get("id")): item
            for item in frames.get("static_models", [])
            if isinstance(item, dict) and item.get("id")
        }
        self.static_cache: dict[str, dict[str, np.ndarray]] = {}

    def _load_blocks(self, frames: dict[str, Any]) -> list[dict[str, Any]]:
        blocks = frames.get("blocks")
        if isinstance(blocks, list) and blocks:
            return [dict(item) for item in blocks if isinstance(item, dict)]
        index_doc = frames.get("block_index")
        if not isinstance(index_doc, dict):
            return []
        if unpack_frame_index is None:
            raise RestoreError(
                f"hdxf_index.py is required for the binary frame index: {_HDXF_INDEX_IMPORT_ERROR}"
            )
        path = str(index_doc.get("path") or "")
        raw = _read_member_checked(
            self.zf,
            path,
            expected_size=index_doc.get("size_bytes") if isinstance(index_doc.get("size_bytes"), int) else None,
            expected_sha256=index_doc.get("sha256") if isinstance(index_doc.get("sha256"), str) else None,
        )
        try:
            result = unpack_frame_index(
                raw,
                datasets=list(frames.get("datasets", [])),
                frame_shape=list(frames.get("frame_shape") or []),
            )
        except Exception as exc:
            raise RestoreError(f"invalid binary frame index {path}: {exc}") from exc
        return [dict(item) for item in result]

    def _load_static_model(self, model_id: str | None) -> dict[str, np.ndarray] | None:
        if not model_id:
            return None
        model_id = str(model_id)
        if model_id in self.static_cache:
            return self.static_cache[model_id]
        doc = self.static_models.get(model_id)
        if doc is None:
            raise RestoreError(f"missing shared static model: {model_id}")
        if decode_hdxfb is None:
            raise RestoreError(f"hdxf_codec.py is required: {_HDXF_CODEC_IMPORT_ERROR}")
        try:
            mask_doc = doc["mask"]
            values_doc = doc["static_values"]
            mask_path = str(mask_doc["path"])
            values_path = str(values_doc["path"])
            mask_raw = _read_member_checked(
                self.zf,
                mask_path,
                expected_size=mask_doc.get("size_bytes") if isinstance(mask_doc.get("size_bytes"), int) else None,
                expected_sha256=mask_doc.get("sha256") if isinstance(mask_doc.get("sha256"), str) else None,
            )
            values_raw = _read_member_checked(
                self.zf,
                values_path,
                expected_size=values_doc.get("size_bytes") if isinstance(values_doc.get("size_bytes"), int) else None,
                expected_sha256=values_doc.get("sha256") if isinstance(values_doc.get("sha256"), str) else None,
            )
            packed_mask, _ = decode_hdxfb(mask_raw, member_path=mask_path, verify_hash=True)
            values, _ = decode_hdxfb(values_raw, member_path=values_path, verify_hash=True)
        except Exception as exc:
            raise RestoreError(f"cannot decode static model {model_id}: {exc}") from exc
        frame_shape = tuple(int(x) for x in doc.get("frame_shape") or ())
        pixel_count = int(doc.get("pixel_count") or np.prod(frame_shape, dtype=np.int64))
        bits = np.unpackbits(np.asarray(packed_mask, dtype=np.uint8).reshape(-1), bitorder="little")
        if bits.size < pixel_count:
            raise RestoreError(f"truncated static model mask: {model_id}")
        mask = bits[:pixel_count].astype(bool).reshape(frame_shape)
        static_values = np.ascontiguousarray(values).reshape(-1)
        if static_values.size != int(np.count_nonzero(mask)):
            raise RestoreError(f"static model value count mismatch: {model_id}")
        result = {"mask": mask, "static_values": static_values}
        self.static_cache[model_id] = result
        return result

    def write_frame_dataset(
        self,
        archive_object_id: str,
        dataset: h5py.Dataset,
        payload: dict[str, Any],
        *,
        progress_every: int,
        physical_guard_bytes: int | None = None,
        source_physical_bytes: int | None = None,
        source_file_name: str = "",
    ) -> None:
        if decode_hdxfb is None:
            raise RestoreError(f"hdxf_codec.py is required to decode detector frames: {_HDXF_CODEC_IMPORT_ERROR}")
        dataset_doc = self.dataset_docs.get(str(archive_object_id))
        if dataset_doc is None:
            raise RestoreError(f"frame Dataset {dataset.name} is missing detector_archive.frames.datasets entry")
        start_index = int(dataset_doc.get("block_index_start", 0))
        block_count = int(dataset_doc.get("block_index_count", 0))
        if block_count <= 0:
            # JSON-index compatibility: filter by source_object when explicit
            # block index ranges were not recorded.
            selected = [b for b in self.blocks if str(b.get("source_object")) == str(archive_object_id)]
        else:
            selected = self.blocks[start_index:start_index + block_count]
        if not selected and dataset.shape not in (None, ()) and int(dataset.shape[0]) > 0:
            raise RestoreError(f"no frame blocks found for {dataset.name}")

        cursor = 0
        previous_frame: np.ndarray | None = None
        for index, block_doc in enumerate(selected, start=1):
            member_path = str(block_doc.get("path") or "")
            raw = _read_member_checked(
                self.zf,
                member_path,
                expected_size=block_doc.get("size_bytes") if isinstance(block_doc.get("size_bytes"), int) else None,
                expected_sha256=block_doc.get("sha256") if isinstance(block_doc.get("sha256"), str) else None,
            )
            requires_previous = bool(block_doc.get("requires_previous_frame", False))
            if not requires_previous:
                previous_frame = None
            if (self.delta_backend in ("cuda", "cuda-auto")
                    and time.monotonic() >= self._next_gpu_temp_check):
                self._next_gpu_temp_check = time.monotonic() + 20.0
                gpu_c = gpu_temperature(self.cuda_device)
                if gpu_c is not None and gpu_c >= 85:
                    _progress(f"[GPU GUARD] GPU {gpu_c}°C; switching Delta restoration "
                              "to CPU for the remaining blocks")
                    self.delta_backend = "cpu"
            try:
                block, header = decode_hdxfb(
                    raw,
                    member_path=member_path,
                    verify_hash=True,
                    previous_frame=previous_frame,
                    static_model=self._load_static_model(block_doc.get("static_model_id")),
                    delta_backend=self.delta_backend,
                    cuda_device=self.cuda_device,
                )
            except Exception as exc:
                raise RestoreError(f"cannot decode frame block {member_path}: {exc}") from exc
            if header.get("kind") != "frame-block":
                raise RestoreError(f"HDXFB member is not a frame block: {member_path}")
            n = int(block.shape[0])
            if cursor + n > int(dataset.shape[0]):
                raise RestoreError(f"frame block sequence exceeds Dataset shape: {dataset.name}")
            # Logical frame values are written directly into the final filtered
            # HDF5 Dataset.  HDF5 applies the recorded chunk/filter pipeline as
            # each chunk is committed; no 186 GiB uncompressed staging Dataset
            # is ever created.
            dataset[cursor:cursor + n] = block
            cursor += n
            previous_frame = np.ascontiguousarray(block[-1])
            if index == 1 or index == len(selected) or index % max(1, progress_every) == 0:
                # Flush periodically so physical on-disk growth reflects the
                # compressed size during a long restore and cached chunks do not
                # accumulate unnecessarily.
                try:
                    dataset.file.flush()
                except Exception as exc:
                    raise RestoreError(
                        f"cannot flush compressed HDF5 output for {dataset.name}: {exc}"
                    ) from exc
                physical_text = ""
                try:
                    physical_bytes = Path(str(dataset.file.filename)).stat().st_size
                    physical_text = f", physical={physical_bytes / (1024 * 1024):.2f} MiB"
                    if (
                        physical_guard_bytes is not None
                        and int(physical_bytes) > int(physical_guard_bytes)
                    ):
                        source_text = (
                            f"{int(source_physical_bytes) / (1024 * 1024):.2f} MiB"
                            if source_physical_bytes is not None
                            else "unknown"
                        )
                        raise RestoreError(
                            f"compressed HDF5 size sanity check failed for {dataset.name}: "
                            f"after frame block {index}/{len(selected)} the temporary file is "
                            f"{physical_bytes / (1024 * 1024):.2f} MiB, while the original "
                            f"physical HDF5 file was {source_text}. Refusing to continue because "
                            f"the source filter pipeline is not compressing correctly."
                        )
                except RestoreError:
                    raise
                except Exception:
                    pass
                _progress(
                    f"restoring frames {source_file_name + '::' if source_file_name else ''}{dataset.name}: block {index}/{len(selected)}"
                    f"{physical_text}"
                )
        if dataset.shape != () and cursor != int(dataset.shape[0]):
            raise RestoreError(
                f"frame count mismatch for {dataset.name}: restored {cursor}, expected {dataset.shape[0]}"
            )


def _write_raw_chunks(zf: zipfile.ZipFile, dataset: h5py.Dataset, payload: dict[str, Any]) -> None:
    members = payload.get("members")
    if not isinstance(members, list):
        raise RestoreError(f"raw-chunk Dataset {dataset.name} has no members")
    if not hasattr(dataset.id, "write_direct_chunk"):
        raise RestoreError("this h5py/HDF5 build does not support write_direct_chunk")
    for member in members:
        if not isinstance(member, dict):
            continue
        path = str(member.get("path") or "")
        raw = _read_member_checked(
            zf,
            path,
            expected_size=member.get("size_bytes") if isinstance(member.get("size_bytes"), int) else None,
            expected_sha256=member.get("sha256") if isinstance(member.get("sha256"), str) else None,
        )
        offset = tuple(int(x) for x in (member.get("chunk_offset") or member.get("start") or []))
        filter_mask = int(member.get("filter_mask", 0))
        try:
            dataset.id.write_direct_chunk(offset, raw, filter_mask=filter_mask)
        except TypeError:
            dataset.id.write_direct_chunk(offset, raw, filter_mask)
        except Exception as exc:
            raise RestoreError(f"cannot restore raw HDF5 chunk {dataset.name}@{offset}: {exc}") from exc


def _decode_constant_scalar(recipe: dict[str, Any]) -> np.generic:
    dtype = _decode_numpy_dtype(recipe.get("dtype"))
    raw = base64.b64decode(str(recipe.get("scalar_raw_base64") or ""))
    if len(raw) != dtype.itemsize:
        raise RestoreError("derived constant scalar byte count does not match dtype")
    return np.frombuffer(raw, dtype=dtype, count=1)[0]


def _evaluate_derived_recipe(h5file: h5py.File, recipe: dict[str, Any]) -> np.ndarray:
    op = str(recipe.get("op") or "")
    if op == "constant-fill-v1":
        shape = tuple(int(x) for x in recipe.get("shape", []))
        dtype = _decode_numpy_dtype(recipe.get("dtype"))
        scalar = _decode_constant_scalar(recipe)
        return np.full(shape, scalar, dtype=dtype)

    source_path = str(recipe.get("source_path") or "")
    if source_path not in h5file:
        raise KeyError(source_path)
    source = np.asarray(h5file[source_path][...])
    prefix_ndim = int(recipe.get("reduction_prefix_ndim", 0))
    axes = tuple(range(prefix_ndim, source.ndim))
    if not axes:
        raise RestoreError(f"derived reduction has no axes: {source_path}")

    if op == "reduce-count-v1":
        predicate = str(recipe.get("predicate") or "")
        if predicate == "nonzero":
            mask = source != 0
        elif predicate == "positive":
            mask = source > 0
        elif predicate == "negative":
            mask = source < 0
        elif predicate == "equal-zero":
            mask = source == 0
        elif predicate == "equal-one":
            mask = source == 1
        elif predicate == "equal-minus-one":
            mask = source == -1
        else:
            raise RestoreError(f"unsupported derived predicate: {predicate}")
        calculated = np.count_nonzero(mask, axis=axes)
    elif op == "reduce-any-v1":
        calculated = np.any(source != 0, axis=axes)
    elif op == "reduce-all-v1":
        calculated = np.all(source != 0, axis=axes)
    elif op == "reduce-sum-int-v1":
        calculated = np.sum(source, axis=axes, dtype=np.int64)
    else:
        raise RestoreError(f"unsupported derived recipe: {op}")

    out_shape = tuple(int(x) for x in recipe.get("output_shape", []))
    out_dtype = _decode_numpy_dtype(recipe.get("output_dtype"))
    return np.asarray(calculated, dtype=out_dtype).reshape(out_shape)


def _verify_logical_sha(array: np.ndarray, expected: str | None, what: str) -> None:
    if not expected:
        return
    contiguous = np.ascontiguousarray(array)
    actual = hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()
    if actual != expected:
        raise RestoreError(f"logical SHA256 mismatch for {what}: expected {expected}, got {actual}")


def _canonical_objects(descriptor: dict[str, Any]) -> dict[str, dict[str, Any]]:
    objects = descriptor.get("objects")
    if not isinstance(objects, dict):
        raise RestoreError("external-file descriptor has no objects map")
    return {str(k): v for k, v in objects.items() if isinstance(v, dict)}



def _cleanup_stale_temp_files(output_dir: Path, filename: str) -> None:
    """Remove stale temp files left by interrupted restores of one target.

    The actively-written temp file is never removed by this helper because callers
    invoke it before creating the new temp and after the temp has been atomically
    renamed to the final target.  On Windows, a temp file still owned by another
    live restore process cannot be unlinked; in that case it is left alone.
    """
    pattern = filename + ".*.tmp"
    for candidate in output_dir.glob(pattern):
        try:
            size = candidate.stat().st_size
        except OSError:
            size = None
        try:
            candidate.unlink()
        except FileNotFoundError:
            continue
        except PermissionError:
            _progress(f"stale temp still in use; keeping: {candidate.name}")
            continue
        except OSError as exc:
            _progress(
                f"warning: could not remove stale temp {candidate.name}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        if size is None:
            _progress(f"removed stale temp: {candidate.name}")
        else:
            _progress(
                f"removed stale temp: {candidate.name} "
                f"({size / (1024 * 1024):.2f} MiB)"
            )

def _restore_external_file(
    zf: zipfile.ZipFile,
    manifest: dict[str, Any],
    descriptor: dict[str, Any],
    output_dir: Path,
    *,
    overwrite: bool,
    verify: bool,
    progress_every: int,
    frame_decoder: _FrameDecoder,
) -> Path:
    source_file = descriptor.get("source_file")
    if not isinstance(source_file, dict):
        raise RestoreError("external-file descriptor has no source_file")
    filename = _safe_output_name(str(source_file.get("filename") or ""), "external HDF5")
    source_physical_bytes = (
        int(source_file.get("size_bytes"))
        if isinstance(source_file.get("size_bytes"), int) and int(source_file.get("size_bytes")) > 0
        else None
    )
    # A reconstructed file can differ slightly because HDF5 metadata allocation
    # is not byte-identical.  But it must never explode from ~28 MiB to ~3 GiB.
    # Four times the source size (with a 64 MiB floor margin) is deliberately
    # generous and still catches a broken detector compression pipeline after
    # the very first decoded block on this dataset.
    physical_guard_bytes = (
        max(source_physical_bytes * 4, source_physical_bytes + 64 * 1024 * 1024)
        if source_physical_bytes is not None
        else None
    )
    target = output_dir / filename
    if target.exists() and not overwrite:
        raise RestoreError(f"output exists: {target} (use --overwrite)")

    objects = _canonical_objects(descriptor)
    archive_objects = manifest.get("objects")
    if not isinstance(archive_objects, dict):
        raise RestoreError("manifest objects map is missing")

    _cleanup_stale_temp_files(output_dir, filename)
    fd, temp_name = tempfile.mkstemp(prefix=filename + ".", suffix=".tmp", dir=output_dir)
    os.close(fd)
    temp_path = Path(temp_name)
    temp_path.unlink(missing_ok=True)
    try:
        with h5py.File(temp_path, "w") as h5file:
            # 1) Create every canonical Group first.
            group_records = sorted(
                (rec for rec in objects.values() if rec.get("type") == "group"),
                key=lambda rec: (str(rec.get("canonical_path") or "/").count("/"), str(rec.get("canonical_path") or "/")),
            )
            for record in group_records:
                path = str(record.get("canonical_path") or "/")
                if path != "/":
                    h5file.require_group(path)

            # 2) Create all non-derived Datasets, then fill their payloads.
            derived: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
            dataset_records = sorted(
                ((local_id, rec) for local_id, rec in objects.items() if rec.get("type") == "dataset"),
                key=lambda item: str(item[1].get("canonical_path") or ""),
            )
            for local_id, record in dataset_records:
                path = str(record.get("canonical_path") or "")
                if not path:
                    raise RestoreError(f"Dataset {local_id} has no canonical_path")
                archive_object_id = str(record.get("archive_object") or "")
                object_doc = archive_objects.get(archive_object_id)
                if not isinstance(object_doc, dict):
                    raise RestoreError(f"Dataset {filename}::{path} has no archive object {archive_object_id!r}")
                payload = object_doc.get("payload")
                if not isinstance(payload, dict):
                    raise RestoreError(f"Dataset {filename}::{path} has no payload descriptor")
                encoding = str(payload.get("encoding") or "")
                if encoding == "derived-recipe-v1":
                    derived.append((local_id, record, object_doc))
                    continue
                storage = record.get("hdf5_storage")
                if not isinstance(storage, dict):
                    raise RestoreError(f"Dataset {filename}::{path} has no hdf5_storage")
                raw_chunks = encoding == "calibration-raw-chunks-v1"
                dataset = _create_dataset_from_storage(
                    h5file, path, storage, raw_chunk_payload=raw_chunks
                )
                if encoding in ("frame-sequence", "frame-sequence-indexed"):
                    frame_decoder.write_frame_dataset(
                        archive_object_id,
                        dataset,
                        payload,
                        progress_every=progress_every,
                        physical_guard_bytes=physical_guard_bytes,
                        source_physical_bytes=source_physical_bytes,
                        source_file_name=filename,
                    )
                elif encoding == "calibration-raw-chunks-v1":
                    _write_raw_chunks(zf, dataset, payload)
                elif encoding == "calibration-reference":
                    raise RestoreError(
                        f"Dataset {filename}::{path} uses an external Calibration reference. "
                        "This archive is not self-contained; use embedded/hybrid Calibration when creating HDXF."
                    )
                else:
                    _write_regular_payload(zf, dataset, payload)

            # 3) Derived Datasets.  Dependencies may themselves be derived, so
            # iterate until all resolvable recipes have been produced.
            pending = list(derived)
            while pending:
                next_pending: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
                made_progress = False
                for local_id, record, object_doc in pending:
                    path = str(record.get("canonical_path") or "")
                    payload = object_doc.get("payload") or {}
                    recipe = payload.get("recipe")
                    if not isinstance(recipe, dict):
                        raise RestoreError(f"derived Dataset {filename}::{path} has no recipe")
                    try:
                        value = _evaluate_derived_recipe(h5file, recipe)
                    except KeyError:
                        next_pending.append((local_id, record, object_doc))
                        continue
                    storage = record.get("hdf5_storage")
                    if not isinstance(storage, dict):
                        raise RestoreError(f"derived Dataset {filename}::{path} has no hdf5_storage")
                    dataset = _create_dataset_from_storage(h5file, path, storage)
                    if dataset.shape == ():
                        dataset[()] = value
                    else:
                        dataset[...] = value
                    _verify_logical_sha(value, recipe.get("logical_sha256"), f"{filename}::{path}")
                    made_progress = True
                if next_pending and not made_progress:
                    unresolved = ", ".join(str(item[1].get("canonical_path")) for item in next_pending[:8])
                    raise RestoreError(f"cannot resolve derived Dataset dependencies in {filename}: {unresolved}")
                pending = next_pending

            # 4) Recreate alias/soft/external links after canonical objects exist.
            canonical_by_id = {
                local_id: str(record.get("canonical_path") or "")
                for local_id, record in objects.items()
            }
            links = descriptor.get("links")
            if isinstance(links, list):
                for link in links:
                    if not isinstance(link, dict):
                        continue
                    path = str(link.get("path") or "")
                    if not path or path == "/":
                        continue
                    kind = str(link.get("type") or "")
                    parent_path, name = path.rsplit("/", 1) if "/" in path[1:] else ("/", path.strip("/"))
                    parent = h5file.require_group(parent_path)
                    if kind == "hard":
                        target_id = str(link.get("target_object") or "")
                        canonical = canonical_by_id.get(target_id, "")
                        if not canonical or canonical == path or path in h5file:
                            continue
                        parent[name] = h5file[canonical]
                    elif kind == "soft":
                        if path in h5file:
                            continue
                        parent[name] = h5py.SoftLink(str(link.get("target_path") or ""))
                    elif kind == "external":
                        if path in h5file:
                            continue
                        target_file = str(link.get("target_file") or "")
                        target_path = str(link.get("target_path") or "/")
                        parent[name] = h5py.ExternalLink(target_file, target_path)

            # 5) Attributes last so object references can resolve.
            for local_id, record in objects.items():
                canonical = str(record.get("canonical_path") or "")
                if canonical == "/":
                    obj = h5file["/"]
                elif canonical in h5file:
                    obj = h5file[canonical]
                else:
                    continue
                _set_attributes(obj, record.get("attributes"), h5file)

            h5file.flush()

        if target.exists():
            target.unlink()
        os.replace(temp_path, target)
        _cleanup_stale_temp_files(output_dir, filename)

        if verify:
            _verify_external_file(target, descriptor)
        _progress(f"restored external HDF5: {filename}")
        return target
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _verify_dataset_storage(dataset: h5py.Dataset, storage: dict[str, Any], label: str) -> None:
    expected_chunks = storage.get("chunks")
    if expected_chunks is None:
        if dataset.chunks is not None:
            raise RestoreError(
                f"verification failed: {label} should be contiguous/non-chunked, got chunks={dataset.chunks}"
            )
    else:
        expected_chunk_tuple = tuple(int(x) for x in expected_chunks)
        if tuple(dataset.chunks or ()) != expected_chunk_tuple:
            raise RestoreError(
                f"verification failed: chunk geometry mismatch {label}: "
                f"expected {expected_chunk_tuple}, got {dataset.chunks}"
            )

    expected_filters = storage.get("filters")
    if isinstance(expected_filters, list) and expected_filters:
        plist = dataset.id.get_create_plist()
        actual: list[tuple[int, tuple[int, ...]]] = []
        for index in range(plist.get_nfilters()):
            fid, _flags, values, _name = plist.get_filter(index)
            actual.append((int(fid), tuple(int(v) for v in values)))
        wanted = [
            (int(item.get("id")), tuple(int(v) for v in (item.get("values") or [])))
            for item in expected_filters
            if isinstance(item, dict) and item.get("id") is not None
        ]
        if actual != wanted:
            raise RestoreError(
                f"verification failed: filter pipeline mismatch {label}: "
                f"expected {wanted}, got {actual}"
            )


def _verify_external_file(path: Path, descriptor: dict[str, Any]) -> None:
    objects = _canonical_objects(descriptor)
    try:
        with h5py.File(path, "r") as h5file:
            for record in objects.values():
                canonical = str(record.get("canonical_path") or "")
                if canonical == "/":
                    continue
                if canonical not in h5file:
                    raise RestoreError(f"verification failed: missing {path.name}::{canonical}")
                obj = h5file[canonical]
                if record.get("type") == "group" and not isinstance(obj, h5py.Group):
                    raise RestoreError(f"verification failed: expected Group {path.name}::{canonical}")
                if record.get("type") == "dataset":
                    if not isinstance(obj, h5py.Dataset):
                        raise RestoreError(f"verification failed: expected Dataset {path.name}::{canonical}")
                    storage = record.get("hdf5_storage")
                    if isinstance(storage, dict):
                        expected_shape = _storage_shape(storage)
                        if obj.shape != expected_shape:
                            raise RestoreError(
                                f"verification failed: shape mismatch {path.name}::{canonical}: "
                                f"expected {expected_shape}, got {obj.shape}"
                            )
                        expected_dtype = _decode_hdf5_dtype(storage.get("dtype"))
                        if np.dtype(obj.dtype) != np.dtype(expected_dtype):
                            raise RestoreError(
                                f"verification failed: dtype mismatch {path.name}::{canonical}: "
                                f"expected {expected_dtype}, got {obj.dtype}"
                            )
                        _verify_dataset_storage(obj, storage, f"{path.name}::{canonical}")
    except OSError as exc:
        raise RestoreError(f"verification failed opening {path}: {exc}") from exc


def _main_source_id(manifest: dict[str, Any]) -> str:
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise RestoreError("manifest source descriptor is missing")
    main = source.get("main_file")
    if not isinstance(main, str) or not main:
        raise RestoreError("manifest source.main_file is missing")
    return main


def _master_objects(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    archive_objects = manifest.get("objects")
    if not isinstance(archive_objects, dict):
        raise RestoreError("manifest objects map is missing")
    main_source = _main_source_id(manifest)
    return {
        str(object_id): record
        for object_id, record in archive_objects.items()
        if isinstance(record, dict) and str(record.get("source_file") or "") == main_source
    }


def _master_local_path(record: dict[str, Any]) -> str:
    path = str(record.get("source_local_path") or record.get("source_path") or "")
    if not path:
        raise RestoreError("master object has no local HDF5 path")
    if not path.startswith("/"):
        path = "/" + path
    return path


def _restore_master_from_manifest(
    zf: zipfile.ZipFile,
    manifest: dict[str, Any],
    output_dir: Path,
    filename: str,
    *,
    overwrite: bool,
    verify: bool,
    progress_every: int,
    frame_decoder: _FrameDecoder,
) -> Path:
    """Rebuild an old ``source=input-master`` archive without an external template."""
    target = output_dir / filename
    if target.exists() and not overwrite:
        raise RestoreError(f"output exists: {target} (use --overwrite)")

    objects = _master_objects(manifest)
    if not objects:
        raise RestoreError(
            "old HDXF has no embedded master and no master-local object records to reconstruct"
        )

    _cleanup_stale_temp_files(output_dir, filename)
    fd, temp_name = tempfile.mkstemp(prefix=filename + ".", suffix=".tmp", dir=output_dir)
    os.close(fd)
    temp_path = Path(temp_name)
    temp_path.unlink(missing_ok=True)

    try:
        with h5py.File(temp_path, "w") as h5file:
            groups = sorted(
                (
                    (object_id, record)
                    for object_id, record in objects.items()
                    if record.get("type") == "group"
                ),
                key=lambda item: (
                    _master_local_path(item[1]).count("/"),
                    _master_local_path(item[1]),
                ),
            )
            for _object_id, record in groups:
                local_path = _master_local_path(record)
                if local_path != "/":
                    h5file.require_group(local_path)

            derived: list[tuple[str, dict[str, Any]]] = []
            datasets = sorted(
                (
                    (object_id, record)
                    for object_id, record in objects.items()
                    if record.get("type") == "dataset"
                ),
                key=lambda item: _master_local_path(item[1]),
            )
            for index, (object_id, record) in enumerate(datasets, start=1):
                local_path = _master_local_path(record)
                payload = record.get("payload")
                storage = record.get("hdf5_storage")
                if not isinstance(payload, dict):
                    raise RestoreError(f"master Dataset {local_path} has no payload descriptor")
                if not isinstance(storage, dict):
                    raise RestoreError(f"master Dataset {local_path} has no hdf5_storage descriptor")

                encoding = str(payload.get("encoding") or "")
                if encoding == "derived-recipe-v1":
                    derived.append((object_id, record))
                    continue

                raw_chunks = encoding == "calibration-raw-chunks-v1"
                dataset = _create_dataset_from_storage(
                    h5file,
                    local_path,
                    storage,
                    raw_chunk_payload=raw_chunks,
                )
                if encoding in ("frame-sequence", "frame-sequence-indexed"):
                    frame_decoder.write_frame_dataset(
                        object_id,
                        dataset,
                        payload,
                        progress_every=progress_every,
                    )
                elif encoding == "calibration-raw-chunks-v1":
                    _write_raw_chunks(zf, dataset, payload)
                elif encoding == "calibration-reference":
                    raise RestoreError(
                        f"master Dataset {local_path} uses an external Calibration reference. "
                        "This archive is not self-contained."
                    )
                elif encoding == "embedded-master-reference":
                    raise RestoreError(
                        f"master Dataset {local_path} is only an embedded-master reference, "
                        "but the embedded master member is missing"
                    )
                else:
                    _write_regular_payload(zf, dataset, payload)

                if (
                    index == 1
                    or index == len(datasets)
                    or index % max(1, progress_every * 25) == 0
                ):
                    _progress(
                        f"reconstructing old master datasets: {index}/{len(datasets)}"
                    )

            pending = list(derived)
            while pending:
                next_pending: list[tuple[str, dict[str, Any]]] = []
                made_progress = False
                for object_id, record in pending:
                    local_path = _master_local_path(record)
                    payload = record.get("payload") or {}
                    recipe = payload.get("recipe")
                    if not isinstance(recipe, dict):
                        raise RestoreError(f"derived master Dataset {local_path} has no recipe")
                    try:
                        value = _evaluate_derived_recipe(h5file, recipe)
                    except KeyError:
                        next_pending.append((object_id, record))
                        continue
                    storage = record.get("hdf5_storage")
                    if not isinstance(storage, dict):
                        raise RestoreError(f"derived master Dataset {local_path} has no storage")
                    dataset = _create_dataset_from_storage(h5file, local_path, storage)
                    if dataset.shape == ():
                        dataset[()] = value
                    else:
                        dataset[...] = value
                    _verify_logical_sha(
                        value,
                        recipe.get("logical_sha256"),
                        f"{filename}::{local_path}",
                    )
                    made_progress = True
                if next_pending and not made_progress:
                    unresolved = ", ".join(_master_local_path(item[1]) for item in next_pending[:8])
                    raise RestoreError(
                        f"cannot resolve derived master Dataset dependencies: {unresolved}"
                    )
                pending = next_pending

            object_local_paths = {
                object_id: _master_local_path(record)
                for object_id, record in objects.items()
            }
            links = manifest.get("links")
            if isinstance(links, list):
                for link in links:
                    if not isinstance(link, dict):
                        continue
                    link_path = str(link.get("path") or "")
                    if not link_path or link_path == "/":
                        continue
                    kind = str(link.get("type") or "")
                    parent_path, name = (
                        link_path.rsplit("/", 1)
                        if "/" in link_path[1:]
                        else ("/", link_path.strip("/"))
                    )
                    if parent_path not in h5file:
                        continue
                    parent = h5file[parent_path]

                    if kind == "hard":
                        target_id = str(link.get("target_object") or "")
                        canonical = object_local_paths.get(target_id)
                        if not canonical or target_id not in objects:
                            continue
                        if link_path in h5file or canonical == link_path:
                            continue
                        parent[name] = h5file[canonical]
                    elif kind == "soft":
                        if link_path in h5file:
                            continue
                        parent[name] = h5py.SoftLink(str(link.get("target_path") or ""))
                    elif kind == "external":
                        if link_path in h5file:
                            continue
                        target_file = _safe_output_name(
                            str(link.get("target_file") or ""),
                            "ExternalLink target",
                        )
                        target_path = str(link.get("target_path") or "/")
                        parent[name] = h5py.ExternalLink(target_file, target_path)

            for object_id, record in objects.items():
                local_path = object_local_paths[object_id]
                if local_path == "/":
                    obj = h5file["/"]
                elif local_path in h5file:
                    obj = h5file[local_path]
                else:
                    continue
                _set_attributes(obj, record.get("attributes"), h5file)

            h5file.flush()

        if target.exists():
            target.unlink()
        os.replace(temp_path, target)
        _cleanup_stale_temp_files(output_dir, filename)

        if verify:
            _verify_reconstructed_master(target, manifest)
        _progress(
            f"reconstructed master from manifest: {filename} "
            "(logical/structural restore; original master file SHA256 is not reproducible)"
        )
        return target
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _verify_reconstructed_master(path: Path, manifest: dict[str, Any]) -> None:
    objects = _master_objects(manifest)
    with h5py.File(path, "r") as h5file:
        for record in objects.values():
            local_path = _master_local_path(record)
            if local_path == "/":
                obj = h5file["/"]
            elif local_path in h5file:
                obj = h5file[local_path]
            else:
                raise RestoreError(
                    f"master verification failed: missing object {path.name}::{local_path}"
                )
            if record.get("type") == "group" and not isinstance(obj, h5py.Group):
                raise RestoreError(
                    f"master verification failed: expected Group {path.name}::{local_path}"
                )
            if record.get("type") == "dataset":
                if not isinstance(obj, h5py.Dataset):
                    raise RestoreError(
                        f"master verification failed: expected Dataset {path.name}::{local_path}"
                    )
                storage = record.get("hdf5_storage")
                if isinstance(storage, dict):
                    expected_shape = _storage_shape(storage)
                    if obj.shape != expected_shape:
                        raise RestoreError(
                            f"master verification failed: shape mismatch {path.name}::{local_path}: "
                            f"expected {expected_shape}, got {obj.shape}"
                        )
                    expected_dtype = _decode_hdf5_dtype(storage.get("dtype"))
                    if np.dtype(obj.dtype) != np.dtype(expected_dtype):
                        raise RestoreError(
                            f"master verification failed: dtype mismatch {path.name}::{local_path}: "
                            f"expected {expected_dtype}, got {obj.dtype}"
                        )


def _load_master_bytes(
    zf: zipfile.ZipFile,
    manifest: dict[str, Any],
    *,
    legacy_template_dir: Path | None,
) -> tuple[str, bytes | None, str | None, str]:
    """Return ``(filename, raw_bytes, source_sha256, mode)``."""
    legacy = manifest.get("legacy_hdf5")
    if not isinstance(legacy, dict):
        raise RestoreError("manifest has no legacy_hdf5 reconstruction descriptor")
    master = legacy.get("master_template")
    if not isinstance(master, dict):
        raise RestoreError("manifest has no legacy_hdf5.master_template descriptor")
    filename = _safe_output_name(str(master.get("filename") or "master.h5"), "master HDF5")
    expected_size = master.get("size_bytes") if isinstance(master.get("size_bytes"), int) else None
    expected_sha = master.get("sha256") if isinstance(master.get("sha256"), str) else None

    member = str(master.get("member") or DEFAULT_EMBEDDED_MASTER)
    if member in zf.namelist():
        raw = _read_member_checked(
            zf, member, expected_size=expected_size, expected_sha256=expected_sha
        )
        return filename, raw, expected_sha, "embedded"

    if legacy_template_dir is not None:
        source = legacy_template_dir.resolve() / filename
        if not source.is_file():
            raise RestoreError(f"legacy master template not found: {source}")
        raw = source.read_bytes()
        if expected_size is not None and len(raw) != expected_size:
            raise RestoreError(f"legacy master template size mismatch: {source}")
        if expected_sha and _sha256_bytes(raw) != expected_sha:
            raise RestoreError(f"legacy master template SHA256 mismatch: {source}")
        return filename, raw, expected_sha, "template"

    source_mode = str(master.get("source") or "")
    if source_mode in ("input-master", "") and _master_objects(manifest):
        return filename, None, expected_sha, "reconstruct"

    raise RestoreError(
        "HDXF has neither an embedded master nor enough master-local records "
        "to reconstruct the master HDF5"
    )


def _write_master(
    output_dir: Path,
    filename: str,
    raw: bytes,
    *,
    overwrite: bool,
    expected_sha: str | None,
) -> Path:
    target = output_dir / filename
    if target.exists() and not overwrite:
        raise RestoreError(f"output exists: {target} (use --overwrite)")
    _cleanup_stale_temp_files(output_dir, filename)
    fd, temp_name = tempfile.mkstemp(prefix=filename + ".", suffix=".tmp", dir=output_dir)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temp = Path(temp_name)
        if expected_sha and _sha256_file(temp) != expected_sha:
            raise RestoreError("master SHA256 changed while writing output")
        if target.exists():
            target.unlink()
        os.replace(temp, target)
        _cleanup_stale_temp_files(output_dir, filename)
        return target
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        Path(temp_name).unlink(missing_ok=True)
        raise


# Each spawned worker owns its own ZIP reader, decoder and destination HDF5
# handle.  h5py/HDF5 objects are never sent across threads or processes.
_RESTORE_WORKER_STATE: dict[str, Any] = {}


def _init_restore_worker(archive_path: str, delta_backend: str,
                         cuda_device: int, gpu_memory_limit_mib: int) -> None:
    if delta_backend in ("cuda", "cuda-auto"):
        try:
            cap = _apply_cuda_pool_limit(cuda_device, gpu_memory_limit_mib)
            _progress(f"[GPU GUARD] restore child CUDA pool={cap} MiB")
        except Exception as exc:
            if delta_backend == "cuda":
                raise
            _progress(f"[GPU GUARD] child GPU allocation unavailable; CPU fallback: {exc}")
            delta_backend = "cpu"
    zf = zipfile.ZipFile(archive_path, "r")
    manifest = _load_manifest(zf)
    _RESTORE_WORKER_STATE["zip"] = zf
    _RESTORE_WORKER_STATE["manifest"] = manifest
    _RESTORE_WORKER_STATE["decoder"] = _FrameDecoder(
        zf, manifest, delta_backend=delta_backend, cuda_device=cuda_device
    )


def _restore_external_task(
    index: int, output_dir: str, overwrite: bool, verify: bool, progress_every: int,
) -> str:
    state = _RESTORE_WORKER_STATE
    manifest = state["manifest"]
    descriptor = manifest["legacy_hdf5"]["external_files"][index]
    path = _restore_external_file(
        state["zip"], manifest, descriptor, Path(output_dir),
        overwrite=overwrite, verify=verify, progress_every=progress_every,
        frame_decoder=state["decoder"],
    )
    return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restore HDXF to the original HDF5 master + external files. Optimized for HDXF 0.4.8 embedded-master archives; older source=input-master archives remain supported."
    )
    parser.add_argument("--version", action="version", version=version_banner(
        "hdxf_to_hdf5", build=RESTORE_BUILD))
    parser.add_argument("input", type=Path, help="Input .hdxf archive")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output HDF5 files")
    parser.add_argument("--verify", action="store_true", help="Verify ZIP integrity, master SHA256, and restored HDF5 shape/dtype/chunk/filter structure")
    parser.add_argument("--progress-every", type=int, default=1, help="Frame-block progress interval (default: 1)")
    parser.add_argument("--compute-backend", choices=("auto", "cpu", "cuda"),
                        default="auto", help="Delta reconstruction GPU/CPU backend (default: auto)")
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--gpu-memory-limit-mib", type=int, default=0,
                        help="CuPy memory pool limit (0=safe automatic VRAM budget)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Independent HDF5 restore processes (0=auto, up to 3; 1=serial)")
    # Old-archive fallback only; intentionally hidden from normal usage.
    parser.add_argument("--legacy-template-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--restore-mode", choices=("exact",), default="exact", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    archive_path = args.input.resolve()
    output_dir = args.output.resolve()
    if not archive_path.is_file():
        parser.error(f"input file does not exist: {archive_path}")
    if args.workers < 0 or args.cuda_device < 0 or args.gpu_memory_limit_mib < 0:
        parser.error("--workers, --cuda-device and --gpu-memory-limit-mib must be >= 0")
    output_dir.mkdir(parents=True, exist_ok=True)
    _progress(f"restore build: {RESTORE_BUILD}")

    try:
        sys_hardware = hardware_snapshot()
        _progress(f"[AUTO-TUNE] CPU {sys_hardware.logical_cpus} logical; RAM "
                  f"{sys_hardware.available_mib}/{sys_hardware.total_mib} MiB free "
                  f"({sys_hardware.used_percent:.1f}% used)")
        delta_backend = _resolve_restore_backend(
            args.compute_backend, args.cuda_device, args.gpu_memory_limit_mib)
        _progress(f"Delta restore compute backend: {delta_backend}; CUDA device: {args.cuda_device}")
        with zipfile.ZipFile(archive_path, "r") as zf:
            if args.verify:
                bad_member = zf.testzip()
                if bad_member is not None:
                    raise RestoreError(f"HDXF ZIP integrity check failed at member: {bad_member}")
                _progress("HDXF ZIP integrity: PASS")
            manifest = _load_manifest(zf)
            legacy = manifest.get("legacy_hdf5")
            if not isinstance(legacy, dict):
                raise RestoreError("archive has no exact HDF5 reconstruction descriptor")

            hdxf_doc = manifest.get("hdxf") if isinstance(manifest.get("hdxf"), dict) else {}
            hdxf_version = str(hdxf_doc.get("version") or "unknown")
            master_doc = legacy.get("master_template") if isinstance(legacy.get("master_template"), dict) else {}
            master_member = str(master_doc.get("member") or DEFAULT_EMBEDDED_MASTER)
            if master_member in zf.namelist():
                manifest_method = zf.getinfo(MANIFEST_PATH).compress_type
                master_method = zf.getinfo(master_member).compress_type
                _progress(
                    f"HDXF {hdxf_version}: embedded master present; "
                    f"ZIP methods manifest={manifest_method}, master={master_method}"
                )
            coverage = legacy.get("coverage")
            if isinstance(coverage, dict) and coverage.get("all_external_datasets_archived") is False:
                missing = coverage.get("unarchived_datasets") or []
                raise RestoreError(
                    f"archive declares incomplete external-HDF5 coverage: {len(missing)} Dataset(s) missing"
                )

            master_name, master_bytes, master_sha, master_mode = _load_master_bytes(
                zf,
                manifest,
                legacy_template_dir=args.legacy_template_dir,
            )
            if master_mode == "embedded":
                _progress(f"embedded master verified: {master_name}")
            elif master_mode == "template":
                _progress(f"legacy master template verified: {master_name}")
            else:
                _progress(
                    f"old HDXF detected: reconstructing master from manifest/object payloads: "
                    f"{master_name}"
                )

            # Check overwrite conditions for the full output set before writing.
            external_files = legacy.get("external_files")
            if not isinstance(external_files, list):
                external_files = []
            planned = [master_name]
            for descriptor in external_files:
                if isinstance(descriptor, dict) and isinstance(descriptor.get("source_file"), dict):
                    planned.append(
                        _safe_output_name(
                            str(descriptor["source_file"].get("filename") or ""),
                            "external HDF5",
                        )
                    )
            if not args.overwrite:
                collisions = [name for name in planned if (output_dir / name).exists()]
                if collisions:
                    raise RestoreError(
                        "output file(s) already exist (use --overwrite): " + ", ".join(collisions[:8])
                    )

            frame_decoder = _FrameDecoder(
                zf, manifest, delta_backend=delta_backend, cuda_device=args.cuda_device
            )
            if len(planned) != len(set(planned)):
                raise RestoreError("archive contains duplicate target HDF5 filenames")
            restored_external: list[Path] = []
            total_external = len(external_files)
            # Separate HDF5 processes have substantial decompression and chunk
            # buffers.  A 6 GB CUDA GPU should not get 3 competing GPU pools.
            default_workers = min(3, max(1, sys_hardware.logical_cpus // 4))
            if delta_backend in ("cuda", "cuda-auto"):
                default_workers = 1  # GPU delta still accelerates this process
            if sys_hardware.total_mib:
                spare = max(0, sys_hardware.available_mib - reserve_mib(sys_hardware.total_mib))
                ram_workers = max(1, spare // 1200)
            else:
                ram_workers = 1
            requested_workers = args.workers or default_workers
            worker_count = min(total_external, requested_workers, ram_workers)
            if total_external:
                worker_count = max(1, worker_count)
            _progress(f"[AUTO-TUNE] HDF5 restore workers: {worker_count} "
                      f"(requested={requested_workers}, RAM-safe={ram_workers}; "
                      "each process owns an independent HDF5 file)")
            if worker_count <= 1:
                for index, descriptor in enumerate(external_files, start=1):
                    if not isinstance(descriptor, dict):
                        continue
                    source = descriptor.get("source_file") or {}
                    _progress(
                        f"restoring external HDF5 {index}/{total_external}: "
                        f"{source.get('filename', '?')}"
                    )
                    restored_external.append(
                        _restore_external_file(
                            zf, manifest, descriptor, output_dir,
                            overwrite=args.overwrite, verify=args.verify,
                            progress_every=max(1, int(args.progress_every)),
                            frame_decoder=frame_decoder,
                        )
                    )
            else:
                # Spawn avoids forking an existing HDF5 library state; this is
                # especially important on Windows and with native filter plugins.
                context = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(
                    max_workers=worker_count, mp_context=context,
                    initializer=_init_restore_worker,
                    initargs=(str(archive_path), delta_backend, args.cuda_device,
                              args.gpu_memory_limit_mib),
                ) as executor:
                    futures = {
                        executor.submit(
                            _restore_external_task, index, str(output_dir),
                            args.overwrite, args.verify,
                            max(1, int(args.progress_every)),
                        ): index
                        for index, descriptor in enumerate(external_files)
                        if isinstance(descriptor, dict)
                    }
                    completed = 0
                    for future in as_completed(futures):
                        index = futures[future]
                        restored_external.append(Path(future.result()))
                        completed += 1
                        source = external_files[index].get("source_file") or {}
                        # Emit progress based on completed files, not queued ones.
                        _progress(f"restoring external HDF5 {completed}/{total_external}: "
                                  f"{source.get('filename', '?')}")

            # Publish/reconstruct master last so its ExternalLinks never point at
            # only partially reconstructed external files.
            if master_mode in ("embedded", "template"):
                assert master_bytes is not None
                master_path = _write_master(
                    output_dir,
                    master_name,
                    master_bytes,
                    overwrite=args.overwrite,
                    expected_sha=master_sha,
                )
                if args.verify and master_sha:
                    actual = _sha256_file(master_path)
                    if actual != master_sha:
                        raise RestoreError(
                            f"master verification failed: expected {master_sha}, got {actual}"
                        )
            else:
                master_path = _restore_master_from_manifest(
                    zf,
                    manifest,
                    output_dir,
                    master_name,
                    overwrite=args.overwrite,
                    verify=args.verify,
                    progress_every=max(1, int(args.progress_every)),
                    frame_decoder=frame_decoder,
                )

        print("", flush=True)
        print("[DONE] HDXF -> HDF5 restore completed", flush=True)
        print(f"[MASTER] {master_path}", flush=True)
        print(f"[EXTERNAL_FILES] {len(restored_external)}", flush=True)
        if master_sha and master_mode in ("embedded", "template"):
            print(f"[MASTER_SHA256] {master_sha}", flush=True)
        elif master_sha:
            print(f"[SOURCE_MASTER_SHA256] {master_sha}", flush=True)
            print(
                "[NOTE] old archive master was reconstructed logically; "
                "its output file SHA256 is not expected to equal the source master SHA256",
                flush=True,
            )
        return 0
    except zipfile.BadZipFile as exc:
        print(f"[ERROR] not a readable HDXF/ZIP archive: {archive_path}: {exc}", file=sys.stderr, flush=True)
        return 2
    except RestoreError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print("[ERROR] interrupted", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
