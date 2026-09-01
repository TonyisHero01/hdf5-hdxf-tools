#!/usr/bin/env python3
"""Convert HDF5 detector data to HDXF.

Tool v0.4.6 adds exact legacy-HDF5 reconstruction descriptors and archives all datasets in followed external HDF5 files.

For detector archives, the converter validates that ``legacy-template``
contains a byte-identical copy of the input master HDF5 file.  The HDXF
manifest records the template identity and the complete local HDF5 structure
of every followed external data file (Groups, Datasets, attributes, links and
Dataset creation/storage properties).  This allows HDXF -> HDF5 restoration
to copy the original master template instead of rebuilding it heuristically,
then reconstruct the externally linked data files with their original HDF5
layout.

Scientific metadata and detector pixels continue to be stored losslessly.
Conversion-host details are not added to the source metadata.
"""

from __future__ import annotations

import argparse
import base64
import csv
import faulthandler
import gc
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
import struct
import time
import uuid
import warnings
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Iterator, Sequence

import h5py
import numpy as np

from hdxf_codec import encode_adaptive_frame_block
from hdxf_index import pack_frame_index

# Importing hdf5plugin registers common detector-data filters such as
# Bitshuffle/LZ4 with HDF5.  The import is optional so uncompressed files still
# work, but Jungfrau/Eiger data using filter 32008 normally requires it.
try:
    import hdf5plugin  # type: ignore  # noqa: F401
    HDF5PLUGIN_AVAILABLE = True
except ImportError:
    HDF5PLUGIN_AVAILABLE = False


# Blosc2 provides the detector-profile block codec (Zstd plus byte/bit shuffle).
try:
    import blosc2  # type: ignore
    BLOSC2_AVAILABLE = True
except ImportError:
    blosc2 = None  # type: ignore
    BLOSC2_AVAILABLE = False

# CuPy is imported lazily only when --compute-backend cuda/auto is requested.
# Keeping it optional preserves the normal CPU-only installation.
_CUPY: Any | None = None
_CUPY_IMPORT_ERROR: BaseException | None = None

FORMAT_NAME = "HDF5-derived Detector eXchange Format"
FORMAT_ID = "hdxf"
FORMAT_VERSION = "0.4.8"
TOOL_VERSION = "0.4.8"
LEGACY_RECONSTRUCTION_VERSION = "1.1"


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
HDXFB_MAGIC = b"HDXFB\x00\x01"
CALIBRATION_ROOT = "/entry/instrument/detector/calibration"
MANIFEST_PATH = "manifest.json"
CALIBRATION_INDEX_NAME = "calibration_index.json"
CALIBRATION_REPORT_NAME = "calibration_report.csv"
CALIBRATION_INDEX_VERSION = "2.0"

# Print Python tracebacks for fatal native-extension failures when possible.
try:
    faulthandler.enable(all_threads=True)
except Exception:
    pass


def _load_cupy() -> Any:
    """Import CuPy once and return the module."""
    global _CUPY, _CUPY_IMPORT_ERROR
    if _CUPY is not None:
        return _CUPY
    if _CUPY_IMPORT_ERROR is not None:
        raise ConversionError(f"CuPy import previously failed: {_CUPY_IMPORT_ERROR}")
    try:
        import cupy as cp  # type: ignore
    except BaseException as exc:
        _CUPY_IMPORT_ERROR = exc
        raise ConversionError(
            "CUDA backend requires CuPy. Install the wheel matching the CUDA "
            "major version, for example cupy-cuda12x or cupy-cuda13x. "
            f"Original import error: {type(exc).__name__}: {exc}"
        ) from exc
    _CUPY = cp
    return cp


def configure_compute_backend(
    requested: str,
    *,
    cuda_device: int,
    gpu_memory_limit_mib: int,
) -> dict[str, Any]:
    """Resolve CPU/CUDA backend and perform a small CUDA correctness test."""
    if requested == "cpu":
        return {"requested": requested, "resolved": "cpu-streaming"}
    if requested == "cpu-vectorized":
        return {"requested": requested, "resolved": "cpu-vectorized"}
    try:
        cp = _load_cupy()
        count = int(cp.cuda.runtime.getDeviceCount())
        if count <= 0:
            raise ConversionError("CuPy found no CUDA devices")
        if cuda_device < 0 or cuda_device >= count:
            raise ConversionError(
                f"CUDA device {cuda_device} is outside the available range 0..{count - 1}"
            )
        with cp.cuda.Device(cuda_device):
            props = cp.cuda.runtime.getDeviceProperties(cuda_device)
            name = props.get("name", b"unknown")
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            if gpu_memory_limit_mib > 0:
                cp.get_default_memory_pool().set_limit(
                    size=int(gpu_memory_limit_mib) * 1024 * 1024
                )
            probe = cp.asarray(np.array([[-32768, 0, 32767], [-32767, 2, 32760]], dtype=np.int16))
            delta = cp.subtract(probe[1:], probe[:-1], dtype=cp.int32)
            result = cp.asnumpy(delta)
            cp.cuda.get_current_stream().synchronize()
            if not np.array_equal(result, np.array([[1, 2, -7]], dtype=np.int32)):
                raise ConversionError("CUDA Delta correctness probe failed")
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        return {
            "requested": requested,
            "resolved": "cuda-cupy",
            "cuda_device": int(cuda_device),
            "device_name": str(name),
            "cupy": str(getattr(cp, "__version__", "unknown")),
            "gpu_total_bytes": int(total_bytes),
            "gpu_free_bytes_at_start": int(free_bytes),
            "gpu_memory_limit_mib": int(gpu_memory_limit_mib),
        }
    except Exception as exc:
        if requested == "auto":
            print(
                f"[WARN] CUDA backend unavailable; using CPU vectorized Delta: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return {
                "requested": requested,
                "resolved": "cpu-streaming",
                "cuda_error": f"{type(exc).__name__}: {exc}",
            }
        raise


def _progress(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


class ConversionError(RuntimeError):
    """Raised when conversion cannot safely continue."""


@contextmanager
def h5r(path: str | os.PathLike[str], mode: str = "r") -> Iterator[h5py.File]:
    """Open and close HDF5 exactly like the proven legacy converter."""
    h5 = h5py.File(path, mode)
    try:
        yield h5
    finally:
        h5.close()


def safe_get(h5: h5py.File | h5py.Group, path: str) -> h5py.Group | h5py.Dataset | None:
    """Return an HDF5 object or None when the path does not exist."""
    try:
        return h5[path]
    except KeyError:
        return None


def _normalise_source_filename(filename: str | bytes | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(filename)))


def _manifest_external_filename(filename: str | bytes) -> tuple[str, bool]:
    """Return a non-absolute ExternalLink filename for the manifest."""
    value = os.fsdecode(filename)
    is_absolute = PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()
    if is_absolute:
        # Preserve the fact that the source link was absolute without exposing
        # the originating machine's directory structure.
        return PureWindowsPath(value).name or PurePosixPath(value).name, True
    return value, False


@dataclass(frozen=True)
class SlicePlan:
    selection: tuple[slice, ...] | None
    start: list[int] | None
    count: list[int] | None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def strict_json_dumps(value: Any, *, indent: int | None = None) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
        separators=(",", ":") if indent is None else None,
    ).encode("utf-8")


def _encode_float(value: float) -> Any:
    if math.isnan(value):
        return {"$type": "float", "value": "nan"}
    if math.isinf(value):
        return {"$type": "float", "value": "+inf" if value > 0 else "-inf"}
    return value


def _npy_bytes(array: np.ndarray) -> bytes:
    if array.dtype.hasobject:
        raise ConversionError("Object arrays cannot be written as safe NPY payloads")
    buffer = io.BytesIO()
    # NPY deliberately does not serialize ``dtype.metadata``.  HDXF stores the
    # original HDF5/NumPy dtype metadata in manifest.json, while the NPY member
    # carries the lossless element values and structural dtype.  Suppress only
    # NumPy's expected warning for this documented split representation.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"metadata on a dtype is not saved to an npy/npz.*",
            category=UserWarning,
        )
        np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def encode_value(value: Any, h5file: h5py.File | None = None) -> Any:
    """Encode metadata/attributes as strict JSON without pickle.

    Tagged objects preserve bytes, complex numbers, ndarrays, references and
    non-finite floats. HDF5 variable-length arrays are encoded recursively.
    """
    if value is None or isinstance(value, (bool, str, int)):
        return value

    if isinstance(value, float):
        return _encode_float(value)

    if isinstance(value, complex):
        return {
            "$type": "complex",
            "real": _encode_float(float(value.real)),
            "imag": _encode_float(float(value.imag)),
        }

    if isinstance(value, (bytes, bytearray, memoryview, np.bytes_)):
        raw = bytes(value)
        return {"$type": "bytes", "base64": base64.b64encode(raw).decode("ascii")}

    if isinstance(value, np.str_):
        return str(value)

    if isinstance(value, np.generic):
        if isinstance(value, (np.datetime64, np.timedelta64)):
            return {
                "$type": type(value).__name__,
                "dtype": str(value.dtype),
                "value": str(value),
            }
        return encode_value(value.item(), h5file)

    if isinstance(value, h5py.RegionReference):
        target = None
        if h5file is not None and value:
            try:
                target = h5file[value].name
            except Exception:
                target = None
        return {"$type": "hdf5_region_reference", "target": target}

    if isinstance(value, h5py.Reference):
        target = None
        if h5file is not None and value:
            try:
                target = h5file[value].name
            except Exception:
                target = None
        return {"$type": "hdf5_reference", "target": target}

    if isinstance(value, np.ndarray):
        if not value.dtype.hasobject:
            payload = _npy_bytes(value)
            return {
                "$type": "ndarray_npy",
                "dtype": dtype_descriptor(value.dtype),
                "shape": list(value.shape),
                "base64": base64.b64encode(payload).decode("ascii"),
            }
        flat = [encode_value(item, h5file) for item in value.flat]
        return {
            "$type": "object_array",
            "shape": list(value.shape),
            "items": flat,
        }

    if isinstance(value, (list, tuple)):
        return [encode_value(item, h5file) for item in value]

    if isinstance(value, dict):
        return {str(k): encode_value(v, h5file) for k, v in value.items()}

    return {"$type": "python_repr", "class": type(value).__name__, "value": repr(value)}


def dtype_descriptor(dtype: np.dtype[Any]) -> Any:
    """Return a JSON-compatible, round-trippable NumPy dtype description."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        descr = np.lib.format.dtype_to_descr(np.dtype(dtype))
    return encode_value(descr)


def hdf5_dtype_info(dtype: np.dtype[Any]) -> dict[str, Any]:
    info: dict[str, Any] = {
        "numpy": dtype_descriptor(dtype),
        "itemsize": int(dtype.itemsize),
        "kind": dtype.kind,
        "byteorder": dtype.byteorder,
    }

    # h5py uses dtype.metadata for HDF5-only semantics such as fixed-string
    # encoding, enums, variable-length values and references.  NPY does not
    # retain this dictionary, so preserve it explicitly in the HDXF manifest.
    if dtype.metadata:
        info["metadata"] = encode_value(dict(dtype.metadata))

    vlen = h5py.check_dtype(vlen=dtype)
    enum = h5py.check_dtype(enum=dtype)
    ref = h5py.check_dtype(ref=dtype)

    if vlen is str:
        info["hdf5_special"] = {"class": "vlen_string", "encoding": "utf-8"}
    elif vlen is bytes:
        info["hdf5_special"] = {"class": "vlen_bytes"}
    elif vlen is not None:
        try:
            vlen_descr = dtype_descriptor(np.dtype(vlen))
        except Exception:
            vlen_descr = repr(vlen)
        info["hdf5_special"] = {"class": "vlen", "base": vlen_descr}
    elif enum is not None:
        info["hdf5_special"] = {
            "class": "enum",
            "members": {str(k): int(v) for k, v in enum.items()},
        }
    elif ref is h5py.Reference:
        info["hdf5_special"] = {"class": "reference"}
    elif ref is h5py.RegionReference:
        info["hdf5_special"] = {"class": "region_reference"}

    return info


def _attribute_dataspace_info(attr_id: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        space = attr_id.get_space()
        extent_type = int(space.get_simple_extent_type())
        result["extent_type"] = extent_type
        if extent_type == int(h5py.h5s.SCALAR):
            result["class"] = "scalar"
            result["shape"] = []
        elif extent_type == int(h5py.h5s.NULL):
            result["class"] = "null"
            result["shape"] = None
        else:
            result["class"] = "simple"
            try:
                result["shape"] = [
                    int(x)
                    for x in space.get_simple_extent_dims()
                ]
            except Exception:
                try:
                    result["shape"] = [
                        int(x) for x in attr_id.shape
                    ]
                except Exception:
                    result["shape"] = []
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _attribute_hdf5_type_info(attr_id: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        type_id = attr_id.get_type()
        type_class = int(type_id.get_class())
        result["class"] = type_class
        try:
            result["size"] = int(type_id.get_size())
        except Exception:
            pass

        if type_class == int(h5py.h5t.STRING):
            result["kind"] = "string"
            try:
                result["is_variable_str"] = bool(
                    type_id.is_variable_str()
                )
            except Exception:
                result["is_variable_str"] = False
            try:
                result["cset"] = int(type_id.get_cset())
            except Exception:
                pass
            try:
                result["strpad"] = int(type_id.get_strpad())
            except Exception:
                pass
        elif type_class == int(h5py.h5t.INTEGER):
            result["kind"] = "integer"
            for key, method in (
                ("order", "get_order"),
                ("precision", "get_precision"),
                ("offset", "get_offset"),
                ("sign", "get_sign"),
            ):
                fn = getattr(type_id, method, None)
                if fn is not None:
                    try:
                        result[key] = int(fn())
                    except Exception:
                        pass
        elif type_class == int(h5py.h5t.FLOAT):
            result["kind"] = "float"
            for key, method in (
                ("order", "get_order"),
                ("precision", "get_precision"),
                ("offset", "get_offset"),
            ):
                fn = getattr(type_id, method, None)
                if fn is not None:
                    try:
                        result[key] = int(fn())
                    except Exception:
                        pass
        elif type_class == int(h5py.h5t.REFERENCE):
            result["kind"] = "reference"
        elif type_class == int(h5py.h5t.ENUM):
            result["kind"] = "enum"
        elif type_class == int(h5py.h5t.COMPOUND):
            result["kind"] = "compound"
        elif type_class == int(h5py.h5t.VLEN):
            result["kind"] = "vlen"
        elif type_class == int(h5py.h5t.ARRAY):
            result["kind"] = "array"
        elif type_class == int(h5py.h5t.OPAQUE):
            result["kind"] = "opaque"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def encode_attributes(
    attrs: h5py.AttributeManager,
    h5file: h5py.File,
) -> dict[str, Any]:
    """Encode Attribute value plus exact HDF5 dtype/dataspace semantics."""
    encoded: dict[str, Any] = {}
    for name in attrs.keys():
        key = str(name)
        try:
            attr_id = attrs.get_id(name)
            dtype = np.dtype(attr_id.dtype)
            encoded[key] = {
                "$type": "hdf5_attribute_v2",
                "value": encode_value(attrs[name], h5file),
                "dtype": hdf5_dtype_info(dtype),
                "shape": (
                    None
                    if getattr(attr_id, "shape", None) is None
                    else [int(x) for x in attr_id.shape]
                ),
                "dataspace": _attribute_dataspace_info(attr_id),
                "hdf5_type": _attribute_hdf5_type_info(attr_id),
            }
        except Exception as exc:
            encoded[key] = {
                "$type": "attribute_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
    return encoded


def dataset_storage_metadata(dataset: h5py.Dataset, h5file: h5py.File) -> dict[str, Any]:
    external_storage: list[dict[str, Any]] = []
    try:
        external_storage = [
            {"filename": str(filename), "offset": int(offset), "size": int(size)}
            for filename, offset, size in (dataset.external or [])
        ]
    except Exception:
        pass

    return {
        "shape": None if dataset.shape is None else list(dataset.shape),
        "maxshape": None if dataset.maxshape is None else [None if x is None else int(x) for x in dataset.maxshape],
        "dtype": hdf5_dtype_info(dataset.dtype),
        "chunks": None if dataset.chunks is None else list(dataset.chunks),
        "compression": dataset.compression,
        "compression_options": encode_value(dataset.compression_opts, h5file),
        "filters": dataset_filter_pipeline(dataset),
        "shuffle": bool(dataset.shuffle),
        "fletcher32": bool(dataset.fletcher32),
        "scaleoffset": dataset.scaleoffset,
        "fillvalue": encode_value(dataset.fillvalue, h5file),
        "external_storage": external_storage,
    }


def safe_object_address(obj: h5py.Group | h5py.Dataset) -> int | None:
    try:
        return int(h5py.h5o.get_info(obj.id).addr)
    except Exception:
        return None


def object_identity(obj: h5py.Group | h5py.Dataset, source_file_path: Path) -> tuple[str, int] | None:
    """HDF5 object addresses are only unique inside one physical file."""
    address = safe_object_address(obj)
    if address is None:
        return None
    return (_normalise_source_filename(source_file_path), address)


def dataset_filter_pipeline(dataset: h5py.Dataset) -> list[dict[str, Any]]:
    """Return the low-level HDF5 filter pipeline without reading payload data."""
    filters: list[dict[str, Any]] = []
    try:
        plist = dataset.id.get_create_plist()
        for index in range(plist.get_nfilters()):
            filter_id, flags, values, name = plist.get_filter(index)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            filters.append({
                "id": int(filter_id),
                "flags": int(flags),
                "values": [int(v) for v in values],
                "name": str(name),
            })
    except Exception:
        pass
    return filters



def _safe_plist_call(plist: Any, method_name: str) -> Any:
    """Call one HDF5 property-list getter without making conversion fragile."""
    method = getattr(plist, method_name, None)
    if method is None:
        return None
    try:
        return method()
    except Exception:
        return None


def _normalise_plist_value(value: Any) -> Any:
    """Convert low-level HDF5 property-list values into strict JSON values."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_normalise_plist_value(item) for item in value]
    if isinstance(value, list):
        return [_normalise_plist_value(item) for item in value]
    if isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def group_creation_metadata(group: h5py.Group) -> dict[str, Any]:
    """Capture stable Group creation properties useful for exact reconstruction."""
    result: dict[str, Any] = {}
    try:
        plist = group.id.get_create_plist()
    except Exception:
        return result

    for key, method_name in (
        ("link_creation_order", "get_link_creation_order"),
        ("attribute_creation_order", "get_attr_creation_order"),
        ("object_track_times", "get_obj_track_times"),
    ):
        value = _safe_plist_call(plist, method_name)
        if value is not None:
            result[key] = _normalise_plist_value(value)
    return result


def dataset_creation_metadata(dataset: h5py.Dataset) -> dict[str, Any]:
    """Capture low-level Dataset creation properties not exposed by high-level h5py."""
    result: dict[str, Any] = {}
    try:
        plist = dataset.id.get_create_plist()
    except Exception:
        return result

    for key, method_name in (
        ("layout", "get_layout"),
        ("allocation_time", "get_alloc_time"),
        ("fill_time", "get_fill_time"),
        ("attribute_creation_order", "get_attr_creation_order"),
        ("object_track_times", "get_obj_track_times"),
    ):
        value = _safe_plist_call(plist, method_name)
        if value is not None:
            result[key] = _normalise_plist_value(value)
    return result


def hdf5_file_properties(h5file: h5py.File, path: Path) -> dict[str, Any]:
    """Capture file-level properties without absolute machine-specific paths."""
    result: dict[str, Any] = {
        "filename": path.name,
    }
    try:
        result["size_bytes"] = int(path.stat().st_size)
    except Exception:
        pass
    for key, getter in (
        ("driver", lambda: h5file.driver),
        ("libver", lambda: h5file.libver),
        ("userblock_size", lambda: h5file.userblock_size),
    ):
        try:
            value = getter()
        except Exception:
            continue
        result[key] = _normalise_plist_value(value)
    return result


def capture_local_hdf5_layout(
    h5file: h5py.File,
    source_file_path: Path,
) -> tuple[dict[str, Any], dict[int, str]]:
    """Capture one physical HDF5 file's *local* object graph.

    External/soft links are recorded but never followed.  Hard-linked objects
    are represented once and links point to a stable local object ID.  Dataset
    pixel values are not read here; this is a structure-only descriptor.

    Returns ``(descriptor, address_to_local_object_id)``.  The address mapping
    is runtime-only and is used to connect archived HDXF payload objects with
    the original file-local Dataset objects.
    """
    objects: dict[str, dict[str, Any]] = {}
    links: list[dict[str, Any]] = []
    address_to_local_id: dict[int, str] = {}
    next_object_number = 0

    def new_local_id() -> str:
        nonlocal next_object_number
        next_object_number += 1
        return f"local-object-{next_object_number:06d}"

    def register_object(
        obj: h5py.Group | h5py.Dataset,
        canonical_path: str,
    ) -> str:
        address = safe_object_address(obj)
        if address is not None and address in address_to_local_id:
            return address_to_local_id[address]

        local_id = new_local_id()
        if address is not None:
            address_to_local_id[address] = local_id

        if isinstance(obj, h5py.Group):
            record: dict[str, Any] = {
                "id": local_id,
                "type": "group",
                "canonical_path": canonical_path,
                "attributes": encode_attributes(obj.attrs, h5file),
                "creation": group_creation_metadata(obj),
            }
        elif isinstance(obj, h5py.Dataset):
            record = {
                "id": local_id,
                "type": "dataset",
                "canonical_path": canonical_path,
                "attributes": encode_attributes(obj.attrs, h5file),
                "hdf5_storage": dataset_storage_metadata(obj, h5file),
                "creation": dataset_creation_metadata(obj),
                # Filled later when the same physical Dataset is archived.
                "archive_object": None,
            }
        else:
            raise ConversionError(
                f"Unsupported local HDF5 object in {source_file_path.name}: "
                f"{canonical_path}: {type(obj).__name__}"
            )

        objects[local_id] = record

        if isinstance(obj, h5py.Group):
            try:
                child_names = list(obj.keys())
            except Exception as exc:
                raise ConversionError(
                    f"Cannot enumerate HDF5 group "
                    f"{source_file_path.name}::{canonical_path}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            for name in child_names:
                child_path = (
                    "/" + str(name)
                    if canonical_path == "/"
                    else canonical_path.rstrip("/") + "/" + str(name)
                )
                link = obj.get(name, getlink=True)

                if isinstance(link, h5py.SoftLink):
                    links.append({
                        "path": child_path,
                        "type": "soft",
                        "target_path": str(link.path),
                    })
                    continue

                if isinstance(link, h5py.ExternalLink):
                    target_file, was_absolute = _manifest_external_filename(
                        link.filename
                    )
                    link_doc: dict[str, Any] = {
                        "path": child_path,
                        "type": "external",
                        "target_file": target_file,
                        "target_path": str(link.path),
                    }
                    if was_absolute:
                        link_doc["target_file_was_absolute"] = True
                    links.append(link_doc)
                    continue

                try:
                    child = obj[name]
                except Exception as exc:
                    links.append({
                        "path": child_path,
                        "type": "unreadable",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    continue

                child_address = safe_object_address(child)
                if (
                    child_address is not None
                    and child_address in address_to_local_id
                ):
                    target_id = address_to_local_id[child_address]
                else:
                    target_id = register_object(child, child_path)

                links.append({
                    "path": child_path,
                    "type": "hard",
                    "target_object": target_id,
                })

        return local_id

    root = h5file["/"]
    root_id = register_object(root, "/")
    descriptor = {
        "source_file": hdf5_file_properties(h5file, source_file_path),
        "root_object": root_id,
        "objects": objects,
        "links": links,
    }
    return descriptor, address_to_local_id


def iter_axis0_slices(dataset: h5py.Dataset, target_bytes: int) -> Iterator[SlicePlan]:
    shape = dataset.shape
    if shape is None:
        return
    if shape == ():
        yield SlicePlan(selection=None, start=None, count=None)
        return
    if any(dim == 0 for dim in shape):
        return

    trailing = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
    itemsize = max(1, int(dataset.dtype.itemsize))
    bytes_per_row = max(1, trailing * itemsize)
    target_rows = max(1, target_bytes // bytes_per_row)

    # Object/vlen rows can be much larger than dtype.itemsize suggests, so use
    # one leading-axis row per payload member for predictable memory use.
    if dataset.dtype.hasobject:
        rows = 1
    elif dataset.chunks is not None and dataset.chunks[0] > 0:
        rows = min(int(dataset.chunks[0]), target_rows)
    else:
        rows = target_rows

    rows = min(rows, int(shape[0]))
    for start0 in range(0, int(shape[0]), rows):
        stop0 = min(int(shape[0]), start0 + rows)
        selection = (slice(start0, stop0),) + tuple(slice(None) for _ in shape[1:])
        start = [start0] + [0] * (len(shape) - 1)
        count = [stop0 - start0] + [int(dim) for dim in shape[1:]]
        yield SlicePlan(selection=selection, start=start, count=count)


def read_dataset_slice(dataset: h5py.Dataset, plan: SlicePlan) -> np.ndarray:
    """Read a dataset chunk while preserving the declared storage dtype.

    h5py may return a scalar fixed-length byte string whose inferred NumPy
    dtype is shortened to the visible content length (for example S5 -> S4).
    Casting explicitly to ``dataset.dtype`` restores the declared itemsize and
    padding bytes before the NPY payload is produced.
    """
    if dataset.shape == ():
        return np.asarray(dataset[()], dtype=dataset.dtype)
    assert plan.selection is not None
    return np.asarray(dataset[plan.selection], dtype=dataset.dtype)


def encode_dataset_chunk(array: np.ndarray, h5file: h5py.File) -> tuple[str, bytes]:
    """Return (encoding, payload). Object data uses tagged strict JSON."""
    if not array.dtype.hasobject:
        return "npy", _npy_bytes(array)

    doc = {
        "shape": list(array.shape),
        "data": encode_value(array, h5file),
    }
    return "json", strict_json_dumps(doc)


def _logical_array_buffer(array: np.ndarray) -> tuple[np.ndarray, memoryview]:
    """Return a contiguous array and a zero-copy byte view for hashing/compression."""
    if array.dtype.hasobject:
        raise ConversionError("Object arrays cannot use HDXFB binary encoding")
    contiguous = np.ascontiguousarray(array)
    return contiguous, memoryview(contiguous).cast("B")


def _logical_array_sha256(array: np.ndarray) -> str:
    _, view = _logical_array_buffer(array)
    return hashlib.sha256(view).hexdigest()


def _blosc_filter(name: str):
    if not BLOSC2_AVAILABLE:
        raise ConversionError(
            "detector-frame-archive requires blosc2; install requirements.txt"
        )
    mapping = {
        "none": blosc2.Filter.NOFILTER,
        "shuffle": blosc2.Filter.SHUFFLE,
        "bitshuffle": blosc2.Filter.BITSHUFFLE,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ConversionError(f"Unsupported Blosc2 filter: {name}") from exc


def _compress_array_blosc2(array: np.ndarray, *, filter_name: str, clevel: int) -> bytes:
    contiguous, raw_view = _logical_array_buffer(array)
    kwargs = dict(
        codec=blosc2.Codec.ZSTD,
        clevel=clevel,
        filters=[blosc2.Filter.NOFILTER] * 5 + [_blosc_filter(filter_name)],
        typesize=max(1, int(contiguous.dtype.itemsize)),
    )
    try:
        # Modern python-blosc2 accepts the buffer protocol.  Passing a memoryview
        # avoids an additional full-size ``array.tobytes()`` allocation.
        return bytes(blosc2.compress2(raw_view, **kwargs))
    except (TypeError, ValueError):
        # Compatibility fallback for older bindings.
        return bytes(blosc2.compress2(raw_view.tobytes(), **kwargs))


def _pack_hdxfb(header: dict[str, Any], segments: list[bytes]) -> bytes:
    cursor = 0
    segment_docs = header.setdefault("segments", [])
    if len(segment_docs) != len(segments):
        raise ConversionError("HDXFB segment metadata/count mismatch")
    for doc, segment in zip(segment_docs, segments):
        doc["offset"] = cursor
        doc["compressed_size"] = len(segment)
        cursor += len(segment)
    header_bytes = strict_json_dumps(header)
    return HDXFB_MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + b"".join(segments)


def _segment_doc(
    name: str, array: np.ndarray, *, filter_name: str, compressed: bytes
) -> dict[str, Any]:
    return {
        "name": name,
        "codec": "blosc2-zstd",
        "filter": filter_name,
        "dtype": dtype_descriptor(array.dtype),
        "shape": list(array.shape),
        "uncompressed_size": int(array.nbytes),
        "compressed_size": len(compressed),
    }


def encode_array_block_hdxfb(
    array: np.ndarray, *, filter_name: str, clevel: int, kind: str = "array-block"
) -> tuple[bytes, dict[str, Any]]:
    contiguous = np.ascontiguousarray(array)
    compressed = _compress_array_blosc2(contiguous, filter_name=filter_name, clevel=clevel)
    header = {
        "format": "hdxfb",
        "version": 1,
        "kind": kind,
        "transform": "none",
        "logical_dtype": dtype_descriptor(contiguous.dtype),
        "logical_shape": list(contiguous.shape),
        "logical_sha256": _logical_array_sha256(contiguous),
        "segments": [
            _segment_doc("array", contiguous, filter_name=filter_name, compressed=compressed)
        ],
    }
    return _pack_hdxfb(header, [compressed]), header


def _smallest_signed_dtype(minimum: int, maximum: int) -> np.dtype[Any]:
    for dtype in (np.dtype("int8"), np.dtype("int16"), np.dtype("int32"), np.dtype("int64")):
        info = np.iinfo(dtype)
        if minimum >= int(info.min) and maximum <= int(info.max):
            return dtype
    raise ConversionError(f"Delta range cannot be represented losslessly: {minimum}..{maximum}")


def encode_frame_raw_hdxfb(
    block: np.ndarray, *, clevel: int
) -> tuple[bytes, dict[str, Any]]:
    contiguous = np.ascontiguousarray(block)
    compressed = _compress_array_blosc2(contiguous, filter_name="bitshuffle", clevel=clevel)
    header = {
        "format": "hdxfb",
        "version": 1,
        "kind": "frame-block",
        "transform": "none",
        "logical_dtype": dtype_descriptor(contiguous.dtype),
        "logical_shape": list(contiguous.shape),
        "logical_sha256": _logical_array_sha256(contiguous),
        "segments": [
            _segment_doc("frames", contiguous, filter_name="bitshuffle", compressed=compressed)
        ],
    }
    return _pack_hdxfb(header, [compressed]), header


def _compute_temporal_deltas_cpu_streaming(
    contiguous: np.ndarray,
) -> tuple[np.ndarray, int, int, dict[str, float]]:
    """Low-memory CPU Delta path used by the proven HDXF 0.4 encoder."""
    started = time.perf_counter()
    scratch = np.empty(contiguous.shape[1:], dtype=np.int64)
    minimum: int | None = None
    maximum: int | None = None
    for index in range(1, contiguous.shape[0]):
        np.subtract(
            contiguous[index], contiguous[index - 1], out=scratch, dtype=np.int64
        )
        current_min = int(scratch.min())
        current_max = int(scratch.max())
        minimum = current_min if minimum is None else min(minimum, current_min)
        maximum = current_max if maximum is None else max(maximum, current_max)
    assert minimum is not None and maximum is not None
    target_dtype = _smallest_signed_dtype(minimum, maximum)
    deltas = np.empty(
        (contiguous.shape[0] - 1,) + contiguous.shape[1:], dtype=target_dtype
    )
    for index in range(1, contiguous.shape[0]):
        np.subtract(
            contiguous[index], contiguous[index - 1], out=scratch, dtype=np.int64
        )
        deltas[index - 1] = scratch
    elapsed = time.perf_counter() - started
    return deltas, minimum, maximum, {
        "delta_compute_seconds": elapsed,
        "gpu_upload_seconds": 0.0,
        "gpu_download_seconds": 0.0,
    }


def _compute_temporal_deltas_cpu(
    contiguous: np.ndarray,
) -> tuple[np.ndarray, int, int, dict[str, float]]:
    """Vectorized exact Delta computation on CPU, with one wide allocation."""
    started = time.perf_counter()
    compute_dtype = np.dtype("int32") if contiguous.dtype.itemsize <= 2 else np.dtype("int64")
    wide = np.subtract(
        contiguous[1:], contiguous[:-1], dtype=compute_dtype
    )
    minimum = int(wide.min())
    maximum = int(wide.max())
    target_dtype = _smallest_signed_dtype(minimum, maximum)
    if wide.dtype == target_dtype:
        deltas = np.ascontiguousarray(wide)
    else:
        deltas = np.ascontiguousarray(wide.astype(target_dtype, copy=False))
    elapsed = time.perf_counter() - started
    return deltas, minimum, maximum, {
        "delta_compute_seconds": elapsed,
        "gpu_upload_seconds": 0.0,
        "gpu_download_seconds": 0.0,
    }


def _compute_temporal_deltas_cuda(
    contiguous: np.ndarray,
    *,
    cuda_device: int,
) -> tuple[np.ndarray, int, int, dict[str, float]]:
    """Compute exact temporal Delta and range on a CUDA device using CuPy."""
    cp = _load_cupy()
    compute_dtype = cp.int32 if contiguous.dtype.itemsize <= 2 else cp.int64
    with cp.cuda.Device(cuda_device):
        stream = cp.cuda.get_current_stream()
        upload_started = time.perf_counter()
        gpu_block = cp.asarray(contiguous)
        stream.synchronize()
        upload_seconds = time.perf_counter() - upload_started

        compute_started = time.perf_counter()
        try:
            gpu_wide = cp.subtract(
                gpu_block[1:], gpu_block[:-1], dtype=compute_dtype
            )
        except TypeError:
            # Compatibility fallback for older CuPy ufunc signatures.
            gpu_wide = (
                gpu_block[1:].astype(compute_dtype)
                - gpu_block[:-1].astype(compute_dtype)
            )
        minimum = int(gpu_wide.min().item())
        maximum = int(gpu_wide.max().item())
        target_dtype = _smallest_signed_dtype(minimum, maximum)
        gpu_compact = gpu_wide.astype(cp.dtype(target_dtype), copy=False)
        stream.synchronize()
        compute_seconds = time.perf_counter() - compute_started

        download_started = time.perf_counter()
        deltas = np.ascontiguousarray(cp.asnumpy(gpu_compact))
        stream.synchronize()
        download_seconds = time.perf_counter() - download_started

        del gpu_compact, gpu_wide, gpu_block
    return deltas, minimum, maximum, {
        "delta_compute_seconds": compute_seconds,
        "gpu_upload_seconds": upload_seconds,
        "gpu_download_seconds": download_seconds,
    }


def encode_frame_delta_hdxfb(
    block: np.ndarray,
    *,
    clevel: int,
    compute_backend: str = "cpu-streaming",
    cuda_device: int = 0,
    delta_filter: str = "auto",
) -> tuple[bytes, dict[str, Any], dict[str, float]] | None:
    contiguous = np.ascontiguousarray(block)
    if contiguous.shape[0] < 2 or contiguous.dtype.kind not in "iu" or contiguous.dtype.itemsize > 4:
        return None

    if compute_backend == "cuda-cupy":
        deltas, minimum, maximum, timings = _compute_temporal_deltas_cuda(
            contiguous, cuda_device=cuda_device
        )
    elif compute_backend == "cpu-vectorized":
        deltas, minimum, maximum, timings = _compute_temporal_deltas_cpu(contiguous)
    elif compute_backend == "cpu-streaming":
        deltas, minimum, maximum, timings = _compute_temporal_deltas_cpu_streaming(contiguous)
    else:
        raise ConversionError(f"Unsupported Delta compute backend: {compute_backend}")

    keyframe = np.ascontiguousarray(contiguous[0])
    compression_started = time.perf_counter()
    key_compressed = _compress_array_blosc2(
        keyframe, filter_name="bitshuffle", clevel=clevel
    )

    if delta_filter == "auto":
        filter_names = ("none", "shuffle", "bitshuffle")
    elif delta_filter in ("none", "shuffle", "bitshuffle"):
        filter_names = (delta_filter,)
    else:
        raise ConversionError(f"Unsupported Delta filter policy: {delta_filter}")

    selected_filter: str | None = None
    delta_compressed: bytes | None = None
    filter_sizes: dict[str, int] = {}
    for filter_name in filter_names:
        candidate = _compress_array_blosc2(
            deltas, filter_name=filter_name, clevel=clevel
        )
        filter_sizes[filter_name] = len(candidate)
        if delta_compressed is None or len(candidate) < len(delta_compressed):
            selected_filter = filter_name
            delta_compressed = candidate
        else:
            del candidate
    assert selected_filter is not None and delta_compressed is not None
    timings["compression_seconds"] = time.perf_counter() - compression_started

    header = {
        "format": "hdxfb",
        "version": 1,
        "kind": "frame-block",
        "transform": "delta-prev",
        "logical_dtype": dtype_descriptor(contiguous.dtype),
        "logical_shape": list(contiguous.shape),
        "logical_sha256": _logical_array_sha256(contiguous),
        "delta_dtype": dtype_descriptor(deltas.dtype),
        "segments": [
            _segment_doc(
                "keyframe", keyframe, filter_name="bitshuffle", compressed=key_compressed
            ),
            _segment_doc(
                "deltas", deltas, filter_name=selected_filter, compressed=delta_compressed
            ),
        ],
    }
    packed = _pack_hdxfb(header, [key_compressed, delta_compressed])
    # Learning information is returned to the converter only and is not packed.
    header["selected_delta_filter"] = selected_filter
    del deltas
    return packed, header, timings


def _smallest_unsigned_dtype(maximum: int) -> np.dtype[Any]:
    for dtype in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("uint32"), np.dtype("uint64")):
        if maximum <= int(np.iinfo(dtype).max):
            return dtype
    raise ConversionError(f"unsigned range cannot be represented: 0..{maximum}")


def _zigzag_encode_int64(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.int64).reshape(-1)
    encoded = np.empty(source.shape, dtype=np.uint64)
    positive = source >= 0
    encoded[positive] = source[positive].astype(np.uint64) * np.uint64(2)
    negative = source[~positive]
    encoded[~positive] = ((-negative - 1).astype(np.uint64) * np.uint64(2)) + np.uint64(1)
    return encoded


def _pack_unsigned_chunked(values: np.ndarray, bit_width: int, *, chunk_values: int = 262144) -> np.ndarray:
    source = np.asarray(values, dtype=np.uint64).reshape(-1)
    bit_width = int(bit_width)
    if bit_width < 0 or bit_width > 64:
        raise ConversionError(f"invalid bit width: {bit_width}")
    if source.size == 0 or bit_width == 0:
        return np.empty((0,), dtype=np.uint8)
    # Keep chunk boundaries byte-aligned so individually packed chunks can be concatenated.
    chunk_values = max(8, int(chunk_values) // 8 * 8)
    shifts = np.arange(bit_width, dtype=np.uint64)
    parts: list[np.ndarray] = []
    for start0 in range(0, source.size, chunk_values):
        chunk = source[start0:start0 + chunk_values]
        bits = ((chunk[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.uint8)
        parts.append(np.packbits(bits.reshape(-1), bitorder="little"))
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def _compute_delta_sequence(
    contiguous: np.ndarray,
    *,
    previous_frame: np.ndarray | None,
    compute_backend: str,
    cuda_device: int,
) -> tuple[np.ndarray, int, int, dict[str, float]]:
    if previous_frame is None:
        sequence = contiguous
        owns_sequence = False
    else:
        if previous_frame.shape != contiguous.shape[1:] or previous_frame.dtype != contiguous.dtype:
            raise ConversionError("previous frame is incompatible with chained Delta block")
        sequence = np.empty((contiguous.shape[0] + 1,) + contiguous.shape[1:], dtype=contiguous.dtype)
        sequence[0] = previous_frame
        sequence[1:] = contiguous
        owns_sequence = True
    try:
        if compute_backend == "cuda-cupy":
            return _compute_temporal_deltas_cuda(sequence, cuda_device=cuda_device)
        if compute_backend == "cpu-vectorized":
            return _compute_temporal_deltas_cpu(sequence)
        if compute_backend == "cpu-streaming":
            return _compute_temporal_deltas_cpu_streaming(sequence)
        raise ConversionError(f"Unsupported Delta compute backend: {compute_backend}")
    finally:
        if owns_sequence:
            del sequence


def _build_zero_rle_candidate(
    deltas: np.ndarray,
    *,
    clevel: int,
    minimum_zero_fraction: float,
) -> tuple[list[dict[str, Any]], list[bytes], dict[str, Any]] | None:
    flat = deltas.reshape(-1)
    total = int(flat.size)
    nonzero_count = int(np.count_nonzero(flat))
    zero_fraction = 1.0 - (nonzero_count / total if total else 0.0)
    if total == 0 or zero_fraction < float(minimum_zero_fraction):
        return None
    positions = np.flatnonzero(flat)
    runs64 = np.empty((nonzero_count + 1,), dtype=np.uint64)
    if nonzero_count:
        runs64[0] = np.uint64(positions[0])
        if nonzero_count > 1:
            runs64[1:nonzero_count] = np.diff(positions).astype(np.uint64) - np.uint64(1)
        runs64[-1] = np.uint64(total - 1 - int(positions[-1]))
        values64 = np.asarray(flat[positions], dtype=np.int64)
        zigzag = _zigzag_encode_int64(values64)
        maximum = int(zigzag.max(initial=0))
        bit_width = maximum.bit_length()
        packed = _pack_unsigned_chunked(zigzag, bit_width)
    else:
        runs64[0] = np.uint64(total)
        bit_width = 0
        packed = np.empty((0,), dtype=np.uint8)
    run_dtype = _smallest_unsigned_dtype(int(runs64.max(initial=0)))
    runs = np.ascontiguousarray(runs64.astype(run_dtype, copy=False))
    runs_compressed = _compress_array_blosc2(runs, filter_name="shuffle", clevel=clevel)
    packed_compressed = _compress_array_blosc2(packed, filter_name="none", clevel=clevel)
    docs = [
        _segment_doc("zero_runs", runs, filter_name="shuffle", compressed=runs_compressed),
        _segment_doc("nonzero_packed", packed, filter_name="none", compressed=packed_compressed),
    ]
    docs[0]["semantic"] = "zero-run-lengths-before-nonzero-values"
    docs[1]["semantic"] = "zigzag-lsb-bitpacked-nonzero-deltas"
    meta = {
        "delta_encoding": "zero-rle-bitpack",
        "nonzero_count": nonzero_count,
        "zero_fraction": zero_fraction,
        "nonzero_bit_width": bit_width,
        "zero_run_dtype": dtype_descriptor(run_dtype),
    }
    return docs, [runs_compressed, packed_compressed], meta


def encode_frame_delta_chain_hdxfb(
    block: np.ndarray,
    *,
    clevel: int,
    compute_backend: str,
    cuda_device: int,
    delta_filter: str,
    previous_frame: np.ndarray | None,
    static_mask: np.ndarray | None,
    static_reference: np.ndarray | None,
    shared_static_model_id: str | None,
    delta_stream: str,
    zero_rle_threshold: float,
) -> tuple[bytes, dict[str, Any], dict[str, float]] | None:
    contiguous = np.ascontiguousarray(block)
    if contiguous.dtype.kind not in "iu" or contiguous.dtype.itemsize > 4:
        return None
    chain_mode = "continuation" if previous_frame is not None else "anchor"
    if chain_mode == "anchor" and contiguous.shape[0] < 2:
        return None

    deltas, minimum, maximum, timings = _compute_delta_sequence(
        contiguous,
        previous_frame=previous_frame,
        compute_backend=compute_backend,
        cuda_device=cuda_device,
    )
    compression_started = time.perf_counter()
    segment_docs: list[dict[str, Any]] = []
    segment_data: list[bytes] = []
    keyframe_encoding: str | None = None
    keyframe_payload_bytes = 0

    if chain_mode == "anchor":
        keyframe = np.ascontiguousarray(contiguous[0])
        full_key = _compress_array_blosc2(keyframe, filter_name="bitshuffle", clevel=clevel)
        selected_docs = [
            _segment_doc("keyframe", keyframe, filter_name="bitshuffle", compressed=full_key)
        ]
        selected_data = [full_key]
        keyframe_encoding = "full"
        if static_mask is not None and static_reference is not None:
            mask = np.asarray(static_mask, dtype=bool).reshape(-1)
            if mask.size == keyframe.size and 0 < int(np.count_nonzero(mask)) < mask.size:
                mask_bytes = np.packbits(mask.astype(np.uint8), bitorder="little")
                static_values = np.ascontiguousarray(static_reference.reshape(-1)[mask])
                dynamic_keyframe = np.ascontiguousarray(keyframe.reshape(-1)[~mask])
                static_parts = []
                static_docs = []
                if shared_static_model_id:
                    key_items = (
                        ("dynamic_keyframe", dynamic_keyframe, "bitshuffle", "anchor-dynamic-keyframe-values"),
                    )
                    candidate_encoding = "shared-static-split"
                else:
                    key_items = (
                        ("static_mask", mask_bytes, "none", "dataset-global-static-mask"),
                        ("static_values", static_values, "bitshuffle", "dataset-global-static-values"),
                        ("dynamic_keyframe", dynamic_keyframe, "bitshuffle", "anchor-dynamic-keyframe-values"),
                    )
                    candidate_encoding = "static-split"
                for name, array, filter_name, semantic in key_items:
                    compressed = _compress_array_blosc2(array, filter_name=filter_name, clevel=clevel)
                    doc = _segment_doc(name, array, filter_name=filter_name, compressed=compressed)
                    doc["semantic"] = semantic
                    static_docs.append(doc)
                    static_parts.append(compressed)
                if sum(map(len, static_parts)) < len(full_key):
                    selected_docs, selected_data = static_docs, static_parts
                    keyframe_encoding = candidate_encoding
                else:
                    del static_docs, static_parts
        segment_docs.extend(selected_docs)
        segment_data.extend(selected_data)
        keyframe_payload_bytes = sum(map(len, selected_data))

    logical_delta_shape = list(deltas.shape)
    encoded_deltas = deltas
    delta_mask_docs: list[dict[str, Any]] = []
    delta_mask_data: list[bytes] = []
    delta_pixel_encoding = "full"
    dynamic_pixel_count = int(np.prod(contiguous.shape[1:]))
    if static_mask is not None:
        mask = np.asarray(static_mask, dtype=bool).reshape(-1)
        pixel_count = int(np.prod(contiguous.shape[1:]))
        static_count = int(np.count_nonzero(mask)) if mask.size == pixel_count else 0
        if mask.size == pixel_count and 0 < static_count < pixel_count:
            dynamic_pixel_count = pixel_count - static_count
            encoded_deltas = np.ascontiguousarray(
                deltas.reshape(deltas.shape[0], pixel_count)[:, ~mask]
            )
            if shared_static_model_id:
                delta_pixel_encoding = "shared-dynamic-mask"
            else:
                mask_bytes = np.packbits(mask.astype(np.uint8), bitorder="little")
                mask_compressed = _compress_array_blosc2(
                    mask_bytes, filter_name="none", clevel=clevel
                )
                mask_doc = _segment_doc(
                    "delta_static_mask", mask_bytes, filter_name="none",
                    compressed=mask_compressed
                )
                mask_doc["semantic"] = "global-static-pixel-mask-for-delta-elision"
                delta_mask_docs.append(mask_doc)
                delta_mask_data.append(mask_compressed)
                delta_pixel_encoding = "dynamic-mask"

    if delta_filter == "auto":
        filter_names = ("none", "shuffle", "bitshuffle")
    elif delta_filter in ("none", "shuffle", "bitshuffle"):
        filter_names = (delta_filter,)
    else:
        raise ConversionError(f"Unsupported Delta filter policy: {delta_filter}")

    representations: list[tuple[str, np.ndarray, list[dict[str, Any]], list[bytes], int]] = [
        ("full", deltas, [], [], int(np.prod(contiguous.shape[1:])))
    ]
    if delta_pixel_encoding == "dynamic-mask":
        representations.append(
            (delta_pixel_encoding, encoded_deltas, delta_mask_docs, delta_mask_data, dynamic_pixel_count)
        )

    stream_candidates: list[dict[str, Any]] = []
    fallback_normal: dict[str, Any] | None = None
    for pixel_mode, source_deltas, mask_docs, mask_data, pixel_count_for_mode in representations:
        selected_for_mode: str | None = None
        normal_for_mode: bytes | None = None
        sizes_for_mode: dict[str, int] = {}
        for filter_name in filter_names:
            compressed = _compress_array_blosc2(
                source_deltas, filter_name=filter_name, clevel=clevel
            )
            sizes_for_mode[filter_name] = len(compressed)
            if normal_for_mode is None or len(compressed) < len(normal_for_mode):
                selected_for_mode, normal_for_mode = filter_name, compressed
            else:
                del compressed
        assert selected_for_mode is not None and normal_for_mode is not None
        normal_doc = _segment_doc(
            "deltas", source_deltas, filter_name=selected_for_mode,
            compressed=normal_for_mode
        )
        normal_doc["semantic"] = "temporal-delta-array"
        normal_entry = {
            "pixel_mode": pixel_mode,
            "encoded": source_deltas,
            "mask_docs": mask_docs,
            "mask_data": mask_data,
            "stream_docs": [normal_doc],
            "stream_data": [normal_for_mode],
            "meta": {
                "delta_encoding": "blosc2-array",
                "zero_fraction": float(
                    1.0 - np.count_nonzero(source_deltas) / source_deltas.size
                ) if source_deltas.size else 1.0,
            },
            "selected_filter": selected_for_mode,
            "filter_sizes": sizes_for_mode,
            "dynamic_pixel_count": pixel_count_for_mode,
            "size": sum(map(len, mask_data)) + len(normal_for_mode),
        }
        if fallback_normal is None or normal_entry["size"] < fallback_normal["size"]:
            fallback_normal = normal_entry
        if delta_stream in ("auto", "zstd"):
            stream_candidates.append(normal_entry)
        if delta_stream in ("auto", "zero-rle-bitpack"):
            rle = _build_zero_rle_candidate(
                source_deltas, clevel=clevel,
                minimum_zero_fraction=zero_rle_threshold
            )
            if rle is not None:
                rle_docs, rle_data, rle_meta = rle
                stream_candidates.append({
                    "pixel_mode": pixel_mode,
                    "encoded": source_deltas,
                    "mask_docs": mask_docs,
                    "mask_data": mask_data,
                    "stream_docs": rle_docs,
                    "stream_data": rle_data,
                    "meta": rle_meta,
                    "selected_filter": selected_for_mode,
                    "filter_sizes": sizes_for_mode,
                    "dynamic_pixel_count": pixel_count_for_mode,
                    "size": sum(map(len, mask_data)) + sum(map(len, rle_data)),
                })
    if not stream_candidates:
        assert fallback_normal is not None
        stream_candidates.append(fallback_normal)
    chosen = min(stream_candidates, key=lambda item: int(item["size"]))
    encoded_deltas = chosen["encoded"]
    delta_pixel_encoding = str(chosen["pixel_mode"])
    dynamic_pixel_count = int(chosen["dynamic_pixel_count"])
    selected_filter = str(chosen["selected_filter"])
    filter_sizes = dict(chosen["filter_sizes"])
    delta_meta = dict(chosen["meta"])
    segment_docs.extend(chosen["mask_docs"])
    segment_data.extend(chosen["mask_data"])
    segment_docs.extend(chosen["stream_docs"])
    segment_data.extend(chosen["stream_data"])
    delta_payload_bytes = int(chosen["size"])
    timings["compression_seconds"] = time.perf_counter() - compression_started

    header = {
        "format": "hdxfb",
        "version": 4 if shared_static_model_id else 3,
        "kind": "frame-block",
        "transform": "delta-chain-v1",
        "chain_mode": chain_mode,
        "requires_previous_frame": chain_mode == "continuation",
        "logical_dtype": dtype_descriptor(contiguous.dtype),
        "logical_shape": list(contiguous.shape),
        "logical_sha256": _logical_array_sha256(contiguous),
        "delta_dtype": dtype_descriptor(deltas.dtype),
        "delta_shape": logical_delta_shape,
        "delta_element_count": int(deltas.size),
        "encoded_delta_shape": list(encoded_deltas.shape),
        "encoded_delta_element_count": int(encoded_deltas.size),
        "delta_pixel_encoding": delta_pixel_encoding,
        "dynamic_pixel_count": dynamic_pixel_count,
        "keyframe_encoding": keyframe_encoding,
        **delta_meta,
        "segments": segment_docs,
    }
    packed = _pack_hdxfb(header, segment_data)
    # These values are returned to the converter for its console summary only;
    # they are deliberately added after packing so they are not stored in HDXF.
    header["keyframe_payload_bytes"] = keyframe_payload_bytes
    header["delta_payload_bytes"] = delta_payload_bytes
    header["selected_delta_filter"] = selected_filter
    del deltas
    return packed, header, timings

def encode_frame_block_hdxfb(
    block: np.ndarray, *, transform: str, clevel: int,
    tile_height: int, tile_width: int, sparse_threshold: float,
    enable_jungfrau_split: bool,
    delta_compute_backend: str = "cpu-streaming",
    cuda_device: int = 0,
    delta_filter: str = "auto",
    previous_frame: np.ndarray | None = None,
    static_mask: np.ndarray | None = None,
    static_reference: np.ndarray | None = None,
    shared_static_model_id: str | None = None,
    delta_stream: str = "zstd",
    zero_rle_threshold: float = 0.90,
) -> tuple[bytes, dict[str, Any], dict[str, int], dict[str, float]]:
    """Encode one frame block using the requested lossless policy.

    ``adaptive`` uses HDXFB v2 and chooses the smallest actual encoded
    representation independently for every tile.

    ``auto`` is state-aware at the frame-block level.  For every block it
    compares three fully encoded candidates:

      * Raw
      * the same chained Delta encoder used by ``--frame-transform delta``
      * Adaptive

    The Delta candidate receives ``previous_frame`` when the caller allows a
    continuation block, so Auto can exploit cross-block temporal correlation.
    Candidate selection still uses the actual final HDXFB byte length.
    """
    if transform == "raw":
        data, header = encode_frame_raw_hdxfb(block, clevel=clevel)
        return data, header, {"raw": len(data)}, {}

    if transform == "delta":
        delta = encode_frame_delta_chain_hdxfb(
            block,
            clevel=clevel,
            compute_backend=delta_compute_backend,
            cuda_device=cuda_device,
            delta_filter=delta_filter,
            previous_frame=previous_frame,
            static_mask=static_mask,
            static_reference=static_reference,
            shared_static_model_id=shared_static_model_id,
            delta_stream=delta_stream,
            zero_rle_threshold=zero_rle_threshold,
        )
        if delta is None:
            data, header = encode_frame_raw_hdxfb(block, clevel=clevel)
            return data, header, {"raw": len(data)}, {}
        data, header, timings = delta
        return data, header, {"delta-chain-v1": len(data)}, timings

    if transform == "adaptive":
        data, header, sizes = encode_adaptive_frame_block(
            block,
            clevel=clevel,
            tile_height=tile_height,
            tile_width=tile_width,
            sparse_threshold=sparse_threshold,
            enable_jungfrau_split=enable_jungfrau_split,
        )
        return data, header, sizes, {}

    if transform != "auto":
        raise ConversionError(f"Unsupported frame transform: {transform}")

    # ------------------------------------------------------------------
    # AUTO: compare Raw vs FULL chained Delta vs Adaptive for this block.
    #
    # Important: this deliberately uses encode_frame_delta_chain_hdxfb()
    # rather than the old block-local delta-prev encoder.  Therefore the
    # Delta candidate has the same chain/stream/static capabilities as the
    # dedicated Delta mode.
    # ------------------------------------------------------------------

    best_data, best_header = encode_frame_raw_hdxfb(block, clevel=clevel)
    best_name = "raw"
    sizes: dict[str, int] = {"raw": len(best_data)}

    delta_timings: dict[str, float] = {}
    auto_delta_selected_filter: str | None = None

    delta = encode_frame_delta_chain_hdxfb(
        block,
        clevel=clevel,
        compute_backend=delta_compute_backend,
        cuda_device=cuda_device,
        delta_filter=delta_filter,
        previous_frame=previous_frame,
        static_mask=static_mask,
        static_reference=static_reference,
        shared_static_model_id=shared_static_model_id,
        delta_stream=delta_stream,
        zero_rle_threshold=zero_rle_threshold,
    )
    if delta is not None:
        delta_data, delta_header, delta_timings = delta
        sizes["delta-chain-v1"] = len(delta_data)

        learned_filter = delta_header.get("selected_delta_filter")
        if learned_filter in ("none", "shuffle", "bitshuffle"):
            auto_delta_selected_filter = str(learned_filter)

        if len(delta_data) < len(best_data):
            best_data, best_header = delta_data, delta_header
            best_name = "delta-chain-v1"
        else:
            del delta_data, delta_header

    adaptive_data, adaptive_header, adaptive_sizes = encode_adaptive_frame_block(
        block,
        clevel=clevel,
        tile_height=tile_height,
        tile_width=tile_width,
        sparse_threshold=sparse_threshold,
        enable_jungfrau_split=enable_jungfrau_split,
    )
    sizes["adaptive"] = len(adaptive_data)

    # Preserve Adaptive's internal candidate-size diagnostics without allowing
    # keys such as "none" to overwrite the top-level Raw candidate size.
    for name, value in adaptive_sizes.items():
        sizes[f"adaptive:{name}"] = int(value)

    if len(adaptive_data) < len(best_data):
        best_data, best_header = adaptive_data, adaptive_header
        best_name = "adaptive"
    else:
        del adaptive_data, adaptive_header

    # These fields are intentionally added AFTER the chosen HDXFB payload has
    # already been packed.  They are converter-only diagnostics and therefore
    # do not change archive payload bytes.
    best_header["auto_selected_candidate"] = best_name
    if auto_delta_selected_filter is not None:
        best_header["auto_delta_selected_filter"] = auto_delta_selected_filter

    return best_data, best_header, sizes, delta_timings

def _iter_local_dataset_paths(group: h5py.Group, base: str) -> Iterator[tuple[str, h5py.Dataset]]:
    for name in group.keys():
        link = group.get(name, getlink=True)
        if isinstance(link, (h5py.SoftLink, h5py.ExternalLink)):
            continue
        obj = group[name]
        path = base.rstrip("/") + "/" + name if base != "/" else "/" + name
        if isinstance(obj, h5py.Dataset):
            yield path, obj
        elif isinstance(obj, h5py.Group):
            yield from _iter_local_dataset_paths(obj, path)



def _supports_raw_chunk_access(dataset: h5py.Dataset) -> bool:
    """Whether HDF5 can expose stored chunk bytes without running filters."""
    return (
        dataset.shape is not None
        and dataset.chunks is not None
        and bool(dataset_filter_pipeline(dataset))
        and hasattr(dataset.id, "get_num_chunks")
        and hasattr(dataset.id, "get_chunk_info")
        and hasattr(dataset.id, "read_direct_chunk")
    )


def _iter_raw_hdf5_chunks(
    dataset: h5py.Dataset,
) -> Iterator[tuple[tuple[int, ...], int, bytes]]:
    """Yield allocated HDF5 chunks as (logical offset, filter mask, raw bytes).

    This uses H5Dread_chunk through h5py and therefore does not invoke the
    Bitshuffle/LZ4 decoder. It is used as a crash-safe preservation path for
    Calibration datasets when a native filter crashes during logical reads.
    """
    try:
        count = int(dataset.id.get_num_chunks())
        infos = [dataset.id.get_chunk_info(index) for index in range(count)]
    except Exception as exc:
        raise ConversionError(
            f"Cannot enumerate raw chunks for {dataset.name}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    infos.sort(key=lambda info: tuple(int(x) for x in info.chunk_offset))
    for info in infos:
        offset = tuple(int(x) for x in info.chunk_offset)
        try:
            filter_mask, raw = dataset.id.read_direct_chunk(offset)
        except Exception as exc:
            raise ConversionError(
                f"Cannot read stored HDF5 chunk {dataset.name} at {offset}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        yield offset, int(filter_mask), bytes(raw)


def _calibration_storage_identity(dataset: h5py.Dataset, h5file: h5py.File) -> dict[str, Any]:
    """Metadata required to identify/reconstruct raw filtered HDF5 chunks."""
    return {
        "shape": None if dataset.shape is None else list(dataset.shape),
        "dtype": hdf5_dtype_info(dataset.dtype),
        "chunks": None if dataset.chunks is None else list(dataset.chunks),
        "filters": dataset_filter_pipeline(dataset),
        "fillvalue": encode_value(dataset.fillvalue, h5file),
        "allocation": "allocated-chunks-only; missing chunks use HDF5 fill value",
    }


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    """Atomically replace a JSON file, avoiding partially written indexes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.write_bytes(strict_json_dumps(document, indent=2))
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write a UTF-8 BOM CSV that opens cleanly in Excel on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    fieldnames = [
        "source_file",
        "status",
        "calibration_alias",
        "calibration_id",
        "detector_serial",
        "dataset_count",
        "logical_bytes",
        "bundle_file",
        "error",
    ]
    try:
        with temp_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _new_calibration_index() -> dict[str, Any]:
    now = utc_now_iso()
    return {
        "format": "hdxf-calibration-library-index",
        "version": CALIBRATION_INDEX_VERSION,
        "created_utc": now,
        "updated_utc": now,
        "identity_policy": {
            "algorithm": "sha256",
            "comparison": "strict-exact",
            "scope": [
                "detector serial",
                "calibration group attributes",
                "dataset path",
                "dataset shape",
                "dataset dtype",
                "dataset attributes",
                "logical dataset values or conservative raw filtered HDF5 chunks",
            ],
            "authoritative_key": "calibration_id",
            "alias_note": (
                "Aliases are stable human-readable labels. "
                "The SHA-256 calibration_id remains authoritative."
            ),
        },
        "library": {},
        "calibrations": [],
        "files": [],
    }


def _load_calibration_index(library_dir: Path) -> dict[str, Any]:
    path = library_dir / CALIBRATION_INDEX_NAME
    if not path.exists():
        return _new_calibration_index()
    try:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ConversionError(
            f"Calibration index is unreadable: {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if document.get("format") != "hdxf-calibration-library-index":
        raise ConversionError(f"Unsupported calibration index format: {path}")
    document.setdefault("calibrations", [])
    document.setdefault("files", [])
    document.setdefault("library", {})
    return document


def _next_calibration_alias(
    index_document: dict[str, Any], prefix: str
) -> str:
    """Return the next stable alias, e.g. calibration-000001."""
    escaped = re.escape(prefix)
    pattern = re.compile(rf"^{escaped}-(\d+)$")
    maximum = 0
    used: set[str] = set()
    for item in index_document.get("calibrations", []):
        alias = str(item.get("alias") or "")
        if alias:
            used.add(alias)
        match = pattern.match(alias)
        if match:
            maximum = max(maximum, int(match.group(1)))
    value = maximum + 1
    while True:
        alias = f"{prefix}-{value:06d}"
        if alias not in used:
            return alias
        value += 1


def _source_index_key(path: Path) -> str:
    """Private stable key without placing the absolute path in the index."""
    normalised = os.path.normcase(str(path.resolve()))
    return hashlib.sha256(normalised.encode("utf-8", errors="surrogatepass")).hexdigest()


class HDXFWriter:
    def __init__(
        self,
        output_path: Path,
        *,
        profile: str,
        compression: str,
        compression_level: int,
        target_chunk_bytes: int,
        overwrite: bool,
        source_checksum: bool,
        follow_external: bool,
        external_root: Path | None,
        allow_missing_external: bool,
        block_frames: int,
        frame_transform: str,
        tile_height: int,
        tile_width: int,
        sparse_threshold: float,
        enable_jungfrau_split: bool,
        zstd_level: int,
        calibration_mode: str,
        calibration_library: Path | None,
        calibration_alias_prefix: str,
        calibration_read_mode: str,
        frame_datasets: set[str],
        progress_every: int,
        compute_info: dict[str, Any],
        cuda_device: int,
        delta_filter_policy: str,
        gc_every: int,
        keyframe_interval_blocks: int,
        delta_stream: str,
        zero_rle_threshold: float,
        global_static_scan: str,
        manifest_layout: str,
        frame_index_layout: str,
        legacy_template_dir: Path | None,
        require_legacy_template: bool,
        auxiliary_policy: str,
        auxiliary_derive_max_mib: float,
    ) -> None:
        self.output_path = output_path
        self.profile = profile
        self.target_chunk_bytes = target_chunk_bytes
        self.overwrite = overwrite
        self.source_checksum = source_checksum
        self.compression_name = compression
        self.compression_level = compression_level
        self.follow_external = follow_external
        self.external_root = external_root.resolve() if external_root is not None else None
        self.allow_missing_external = allow_missing_external
        self.block_frames = block_frames
        self.frame_transform = frame_transform
        self.tile_height = int(tile_height)
        self.tile_width = int(tile_width)
        self.sparse_threshold = float(sparse_threshold)
        self.enable_jungfrau_split = bool(enable_jungfrau_split)
        self.zstd_level = zstd_level
        self.calibration_mode = calibration_mode
        self.calibration_library = (
            calibration_library.resolve() if calibration_library is not None else None
        )
        self.calibration_alias_prefix = calibration_alias_prefix.strip() or "calibration"
        self.calibration_read_mode = calibration_read_mode
        self.frame_datasets = frame_datasets
        self.progress_every = max(1, int(progress_every))
        self.compute_info = dict(compute_info)
        self.delta_compute_backend = str(compute_info.get("resolved", "cpu-streaming"))
        self.cuda_device = int(cuda_device)
        self.delta_filter_policy = str(delta_filter_policy)
        self.gc_every = max(0, int(gc_every))
        self.keyframe_interval_blocks = max(1, int(keyframe_interval_blocks))
        self.delta_stream = str(delta_stream)
        self.zero_rle_threshold = float(zero_rle_threshold)
        self.global_static_scan = str(global_static_scan)
        self.manifest_layout = str(manifest_layout)
        self.frame_index_layout = str(frame_index_layout)
        self.legacy_template_dir = (
            legacy_template_dir.resolve()
            if legacy_template_dir is not None
            else None
        )
        self.require_legacy_template = bool(require_legacy_template)
        self.auxiliary_policy = str(auxiliary_policy)
        self.auxiliary_derive_max_bytes = max(
            0, int(float(auxiliary_derive_max_mib) * 1024 * 1024)
        )

        # Exact HDF5 restoration control data.  This is not scientific metadata;
        # it is a reconstruction descriptor consumed by HDXF -> HDF5.
        self.legacy_hdf5: dict[str, Any] | None = None
        self._legacy_external_layouts: dict[str, dict[str, Any]] = {}
        self._legacy_external_runtime_address_maps: dict[
            str, dict[int, str]
        ] = {}
        self._legacy_external_source_by_path: dict[str, str] = {}
        self._legacy_external_links: list[dict[str, Any]] = []

        self._delta_filter_cache: dict[str, str] = {}
        self._experiment_static_mask: np.ndarray | None = None
        self._experiment_static_reference: np.ndarray | None = None

        if profile == "detector-frame-archive":
            if not BLOSC2_AVAILABLE:
                raise ConversionError(
                    "detector-frame-archive requires blosc2; install requirements.txt"
                )
            # Detector payloads are already compressed; ZIP must not recompress.
            self.zip_compression = zipfile.ZIP_STORED
            self.zip_compresslevel = None
        elif compression == "store":
            self.zip_compression = zipfile.ZIP_STORED
            self.zip_compresslevel = None
        elif compression == "deflate":
            self.zip_compression = zipfile.ZIP_DEFLATED
            self.zip_compresslevel = compression_level
        else:
            raise ValueError(f"Unsupported compression: {compression}")

        self.objects: dict[str, dict[str, Any]] = {}
        self.links: list[dict[str, Any]] = []
        self._address_to_id: dict[tuple[str, int], str] = {}
        self._source_path_to_id: dict[str, str] = {}
        self.source_files: list[dict[str, Any]] = []
        self._written_payloads: dict[str, dict[str, Any]] = {}
        self.payload_reference_count = 0

        self.calibration_paths: set[str] = set()
        self.calibration_manifest: dict[str, Any] | None = None
        self._calibration_index_document: dict[str, Any] | None = None
        self._calibration_alias: str | None = None
        self._calibration_bundle_file: str | None = None
        self._calibration_group_is_new = False
        self.frame_archive: dict[str, Any] = {
            "type": "frame_sequence",
            "frame_count": 0,
            "frame_shape": None,
            "dtype": None,
            "block_frames": block_frames,
            "tile_shape": [self.tile_height, self.tile_width],
            "datasets": [],
            "blocks": [],
            "keyframe_index": [],
            "keyframe_interval_blocks": self.keyframe_interval_blocks,
            "static_models": [],
        }
        self.encoding_stats = {
            "frame_blocks": 0, "raw_blocks": 0, "delta_blocks": 0,
            "adaptive_blocks": 0, "logical_frame_bytes": 0,
            "encoded_frame_bytes": 0, "tile_modes": {},
            "timings": {
                "read_seconds": 0.0,
                "delta_compute_seconds": 0.0,
                "gpu_upload_seconds": 0.0,
                "gpu_download_seconds": 0.0,
                "compression_seconds": 0.0,
                "archive_write_seconds": 0.0,
                "static_scan_seconds": 0.0,
            },
            "compute_backend": self.compute_info,
            "delta_filter_policy": self.delta_filter_policy,
            "delta_stream_policy": self.delta_stream,
            "keyframe_interval_blocks": self.keyframe_interval_blocks,
            "anchor_blocks": 0,
            "continuation_blocks": 0,
            "keyframe_payload_bytes": 0,
            "delta_payload_bytes": 0,
            "delta_zstd_blocks": 0,
            "delta_zero_rle_blocks": 0,
            "static_split_keyframes": 0,
            "static_scan_datasets": 0,
            "static_scan_pixels": 0,
            "static_pixels": 0,
            "shared_static_models": 0,
            "shared_static_model_payload_bytes": 0,
            "embedded_calibration_payload_bytes": 0,
            "unique_payload_bytes": 0,
            "payload_bytes_by_encoding": {},
            "manifest_bytes": 0,
            "frame_index_bytes": 0,
            "archive_file_bytes": 0,
            "container_overhead_bytes": 0,
            "auxiliary_derived_datasets": 0,
            "auxiliary_derived_logical_bytes": 0,
            "auxiliary_auto_hdxfb_members": 0,
            "auxiliary_auto_npy_members": 0,
            "auxiliary_auto_logical_bytes": 0,
            "auxiliary_auto_encoded_bytes": 0,
        }
        self._static_model_by_key: dict[str, dict[str, Any]] = {}

    def convert(self, input_path: Path) -> dict[str, Any]:
        input_path = input_path.resolve()
        output_path = self.output_path.resolve()
        if not input_path.is_file():
            raise ConversionError(f"Input is not a file: {input_path}")
        if output_path.exists() and not self.overwrite:
            raise ConversionError(f"Output already exists: {output_path} (use --overwrite)")
        if input_path == output_path:
            raise ConversionError("Input and output paths must be different")

        _progress(f"starting conversion: {input_path.name}")
        _progress(f"output: {output_path}")

        # Exact reconstruction starts by validating the byte-identical master
        # template.  This happens before expensive frame encoding so a wrong
        # template fails immediately.
        self._prepare_legacy_reconstruction(input_path)

        if self.profile == "detector-frame-archive":
            _progress("scanning detector metadata and Calibration identity")
            self._prepare_detector_profile(input_path)
            if (
                self.global_static_scan == "experiment"
                and self.frame_transform in ("delta", "auto")
            ):
                self._prepare_experiment_static_model(input_path)
            if self.calibration_library and self.calibration_mode in ("referenced", "hybrid"):
                _progress("resolving Calibration library group")
                self._prepare_calibration_library_registration(input_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=output_path.name + ".", suffix=".tmp", dir=output_path.parent
        )
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            master_source_id = self._register_source_file(input_path, role="master")
            with h5r(input_path, "r") as h5file, zipfile.ZipFile(
                temp_path,
                mode="w",
                compression=self.zip_compression,
                compresslevel=self.zip_compresslevel,
                allowZip64=True,
            ) as archive:
                _progress("walking HDF5 object tree and encoding datasets")
                root_id = self._walk_hdf5(h5file, archive, input_path)

                # At this point every followed ExternalLink has been traversed,
                # so we can prove that every Dataset in each external file is
                # covered by an archived object before publishing the HDXF.
                self._finalize_legacy_reconstruction()

                _progress("building manifest")
                if self.profile == "detector-frame-archive" and self.frame_index_layout == "binary":
                    index_doc = self._write_binary_frame_index(archive)
                    self.frame_archive["block_index"] = index_doc
                    self.frame_archive.pop("blocks", None)
                    self.frame_archive.pop("keyframe_index", None)
                manifest: dict[str, Any] = {
                    "hdxf": {
                        "format": FORMAT_NAME,
                        "format_id": FORMAT_ID,
                        "version": FORMAT_VERSION,
                        "profile": self.profile,
                    },
                    "source": {
                        "format": "HDF5",
                        "main_file": master_source_id,
                        "files": self.source_files,
                    },
                    "root_object": root_id,
                    "objects": self.objects,
                    "links": self.links,
                }
                if self.legacy_hdf5 is not None:
                    manifest["legacy_hdf5"] = self.legacy_hdf5
                if self.profile == "detector-frame-archive":
                    if self.frame_archive["frame_count"] == 0:
                        raise ConversionError(
                            "No detector frame dataset was found. Use --frame-dataset PATH "
                            "or select --profile generic-hdf5-preservation."
                        )
                    manifest["detector_archive"] = {
                        "frames": self.frame_archive,
                        "calibration": self.calibration_manifest,
                    }
                manifest_indent = 2 if self.manifest_layout == "pretty" else None
                manifest_bytes = strict_json_dumps(manifest, indent=manifest_indent)
                self.encoding_stats["manifest_bytes"] = len(manifest_bytes)
                archive.writestr(MANIFEST_PATH, manifest_bytes)

            # Build/verify the optional calibration bundle before publishing
            # the experiment archive, so conversion remains atomic.
            if (
                self.profile == "detector-frame-archive"
                and self.calibration_library
                and self.calibration_mode in ("referenced", "hybrid")
            ):
                _progress("writing or reusing Calibration bundle")
                self._write_calibration_bundle(input_path)
                _progress("updating Calibration library index")
                self._update_calibration_library_index(input_path)
            if output_path.exists():
                output_path.unlink()
            os.replace(temp_path, output_path)
            archive_bytes = int(output_path.stat().st_size)
            self.encoding_stats["archive_file_bytes"] = archive_bytes
            known_member_bytes = int(self.encoding_stats.get("unique_payload_bytes", 0)) + int(self.encoding_stats.get("manifest_bytes", 0))
            self.encoding_stats["container_overhead_bytes"] = max(0, archive_bytes - known_member_bytes)
            _progress(f"published archive: {output_path}")
            _progress(
                "space breakdown: "
                f"archive={archive_bytes / (1024*1024):.2f} MiB, "
                f"frames={int(self.encoding_stats.get('encoded_frame_bytes', 0)) / (1024*1024):.2f} MiB, "
                f"shared_static={int(self.encoding_stats.get('shared_static_model_payload_bytes', 0)) / (1024*1024):.2f} MiB, "
                f"embedded_calibration={int(self.encoding_stats.get('embedded_calibration_payload_bytes', 0)) / (1024*1024):.2f} MiB, "
                f"frame_index={int(self.encoding_stats.get('frame_index_bytes', 0)) / (1024*1024):.2f} MiB, "
                f"aux_derived_logical={int(self.encoding_stats.get('auxiliary_derived_logical_bytes', 0)) / (1024*1024):.2f} MiB, "
                f"aux_encoded={int(self.encoding_stats.get('auxiliary_auto_encoded_bytes', 0)) / (1024*1024):.2f} MiB, "
                f"manifest={int(self.encoding_stats.get('manifest_bytes', 0)) / (1024*1024):.2f} MiB, "
                f"container_overhead={int(self.encoding_stats.get('container_overhead_bytes', 0)) / (1024*1024):.2f} MiB"
            )
            return manifest
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def _prepare_legacy_reconstruction(self, input_path: Path) -> None:
        """Validate the original master template and initialise the descriptor."""
        if self.legacy_template_dir is None:
            if self.require_legacy_template:
                raise ConversionError(
                    "Exact legacy HDF5 restoration requires a legacy-template "
                    "directory, but no directory was configured."
                )
            return

        template_path = self.legacy_template_dir / input_path.name
        if not template_path.is_file():
            if self.require_legacy_template:
                raise ConversionError(
                    "Exact legacy HDF5 restoration requires the original master "
                    f"template: {template_path}"
                )
            _progress(
                f"legacy template not found; exact-master restore descriptor "
                f"disabled: {template_path.name}"
            )
            return

        _progress(f"validating legacy master template: {template_path.name}")
        source_sha = sha256_file(input_path)
        template_sha = sha256_file(template_path)
        if source_sha != template_sha:
            raise ConversionError(
                "legacy-template master is not byte-identical to the input "
                f"master HDF5: {template_path.name}\n"
                f"input sha256   = {source_sha}\n"
                f"template sha256= {template_sha}"
            )

        self.legacy_hdf5 = {
            "version": LEGACY_RECONSTRUCTION_VERSION,
            "restore_strategy": (
                "copy-byte-identical-master-template-and-rebuild-external-files"
            ),
            "master_template": {
                "filename": template_path.name,
                "sha256": template_sha,
                "size_bytes": int(template_path.stat().st_size),
                "match": "byte-identical-to-input-master",
            },
            "external_files": [],
            "external_links": self._legacy_external_links,
            "coverage": {
                "all_external_datasets_archived": None,
                "unarchived_datasets": [],
            },
        }

    def _capture_legacy_external_file(
        self,
        h5file: h5py.File,
        source_file_path: Path,
        source_id: str,
    ) -> None:
        """Capture one external file's complete local structure once."""
        if self.legacy_hdf5 is None:
            return
        if source_id in self._legacy_external_layouts:
            return

        descriptor, address_map = capture_local_hdf5_layout(
            h5file,
            source_file_path,
        )
        descriptor["source_file"]["source_id"] = source_id
        if self.source_checksum:
            descriptor["source_file"]["sha256"] = sha256_file(
                source_file_path
            )

        self._legacy_external_layouts[source_id] = descriptor
        self._legacy_external_runtime_address_maps[source_id] = address_map
        self._legacy_external_source_by_path[
            _normalise_source_filename(source_file_path)
        ] = source_id

    def _map_legacy_archive_object(
        self,
        obj: h5py.Group | h5py.Dataset,
        source_file_path: Path,
        archive_object_id: str,
    ) -> None:
        """Attach an HDXF archive object ID to the matching file-local object."""
        if self.legacy_hdf5 is None:
            return
        source_id = self._legacy_external_source_by_path.get(
            _normalise_source_filename(source_file_path)
        )
        if source_id is None:
            return
        address = safe_object_address(obj)
        if address is None:
            return
        local_id = self._legacy_external_runtime_address_maps.get(
            source_id, {}
        ).get(address)
        if local_id is None:
            return
        record = self._legacy_external_layouts[source_id]["objects"].get(
            local_id
        )
        if isinstance(record, dict):
            record["archive_object"] = archive_object_id

    @staticmethod
    def _dataset_logical_bytes(dataset: h5py.Dataset) -> int:
        if dataset.shape is None:
            return 0
        if dataset.shape == ():
            return max(1, int(dataset.dtype.itemsize))
        try:
            return (
                int(np.prod(dataset.shape, dtype=np.int64))
                * max(1, int(dataset.dtype.itemsize))
            )
        except Exception:
            return 0

    @staticmethod
    def _array_bytes_equal(a: np.ndarray, b: np.ndarray) -> bool:
        """Bit-exact numeric-array equality, including NaN payload/sign bits."""
        aa = np.ascontiguousarray(a)
        bb = np.ascontiguousarray(b)
        if aa.shape != bb.shape or aa.dtype != bb.dtype:
            return False
        return memoryview(aa).cast("B") == memoryview(bb).cast("B")

    def _read_auxiliary_array_cached(
        self,
        h5file: h5py.File,
        path: str,
        cache: dict[str, np.ndarray | None],
    ) -> np.ndarray | None:
        if path in cache:
            return cache[path]
        obj = safe_get(h5file, path)
        if not isinstance(obj, h5py.Dataset):
            cache[path] = None
            return None
        if (
            obj.shape is None
            or obj.dtype.hasobject
            or obj.dtype.kind not in "biufc"
        ):
            cache[path] = None
            return None
        logical_bytes = self._dataset_logical_bytes(obj)
        if (
            self.auxiliary_derive_max_bytes <= 0
            or logical_bytes > self.auxiliary_derive_max_bytes
        ):
            cache[path] = None
            return None
        try:
            if obj.shape == ():
                value = np.asarray(obj[()], dtype=obj.dtype)
            else:
                value = np.asarray(obj[...], dtype=obj.dtype)
            value = np.ascontiguousarray(value)
        except Exception:
            cache[path] = None
            return None
        cache[path] = value
        return value

    @staticmethod
    def _constant_scalar_bytes(array: np.ndarray) -> bytes | None:
        """Return one element's raw bytes if every element is bit-identical."""
        contiguous = np.ascontiguousarray(array)
        if contiguous.size == 0:
            return None
        itemsize = max(1, int(contiguous.dtype.itemsize))
        raw = memoryview(contiguous).cast("B")
        first = bytes(raw[:itemsize])
        for offset in range(itemsize, len(raw), itemsize):
            if bytes(raw[offset:offset + itemsize]) != first:
                return None
        return first

    def _infer_auxiliary_derived_recipe(
        self,
        h5file: h5py.File,
        target_path: str,
        candidate_paths: list[str],
        cache: dict[str, np.ndarray | None],
    ) -> dict[str, Any] | None:
        """Infer only recipes that are proved bit-exact on the source values.

        This is intentionally conservative.  It does *not* guess scientific
        algorithms such as azimuthal integration or crystallographic indexing.
        A Dataset is omitted only if this converter can actually regenerate
        every source byte using one of the documented recipes below.
        """
        target_obj = safe_get(h5file, target_path)
        if not isinstance(target_obj, h5py.Dataset):
            return None
        target = self._read_auxiliary_array_cached(
            h5file, target_path, cache
        )
        if target is None:
            return None

        target_shape = list(target.shape)
        target_dtype = dtype_descriptor(target.dtype)
        target_hash = _logical_array_sha256(target)
        logical_bytes = int(target.nbytes)

        # 1) Constant fill.  Store exactly one element's raw representation,
        # so signed zero and NaN payload bits are preserved.
        scalar_bytes = self._constant_scalar_bytes(target)
        if scalar_bytes is not None:
            return {
                "format": "hdxf-derived-recipe",
                "version": 1,
                "op": "constant-fill-v1",
                "shape": target_shape,
                "dtype": target_dtype,
                "scalar_raw_base64": base64.b64encode(
                    scalar_bytes
                ).decode("ascii"),
                "logical_sha256": target_hash,
                "logical_bytes": logical_bytes,
            }

        # Reduction recipes are useful for counts/flags such as nPeaks and
        # peak-category counts.  The dependency must be larger than the target,
        # which also guarantees an acyclic derived-data graph.
        target_size = int(target.size)
        if target.dtype.kind not in "biu":
            return None

        for source_path in candidate_paths:
            if source_path == target_path:
                continue
            source_obj = safe_get(h5file, source_path)
            if not isinstance(source_obj, h5py.Dataset):
                continue
            if (
                source_obj.shape is None
                or source_obj.dtype.hasobject
                or source_obj.dtype.kind not in "biufc"
            ):
                continue
            if self._dataset_logical_bytes(source_obj) <= logical_bytes:
                continue

            source = self._read_auxiliary_array_cached(
                h5file, source_path, cache
            )
            if source is None or source.ndim == 0:
                continue

            # Reduce trailing axes and allow a final reshape when the target
            # uses singleton dimensions such as (N, 1).
            for prefix_ndim in range(0, source.ndim):
                prefix_shape = source.shape[:prefix_ndim]
                prefix_size = (
                    int(np.prod(prefix_shape, dtype=np.int64))
                    if prefix_shape
                    else 1
                )
                if prefix_size != target_size:
                    continue
                axes = tuple(range(prefix_ndim, source.ndim))
                if not axes:
                    continue

                candidates: list[tuple[str, np.ndarray, dict[str, Any]]] = []

                # Count predicates.  All calculations are deterministic integer
                # reductions and are verified byte-for-byte before acceptance.
                predicate_values = (
                    ("nonzero", source != 0, {}),
                    ("positive", source > 0, {}),
                    ("negative", source < 0, {}),
                    ("equal-zero", source == 0, {"value": 0}),
                    ("equal-one", source == 1, {"value": 1}),
                    ("equal-minus-one", source == -1, {"value": -1}),
                )
                for predicate, mask, extra in predicate_values:
                    reduced = np.count_nonzero(mask, axis=axes)
                    candidates.append(
                        (
                            "reduce-count-v1",
                            np.asarray(reduced),
                            {
                                "predicate": predicate,
                                **extra,
                            },
                        )
                    )

                # Any/all flags can reproduce boolean or integer flag Datasets.
                candidates.append(
                    (
                        "reduce-any-v1",
                        np.asarray(np.any(source != 0, axis=axes)),
                        {"predicate": "nonzero"},
                    )
                )
                candidates.append(
                    (
                        "reduce-all-v1",
                        np.asarray(np.all(source != 0, axis=axes)),
                        {"predicate": "nonzero"},
                    )
                )

                # Integer/bool sums are stable and sometimes back count/status
                # arrays.  Float reductions are deliberately excluded because
                # summation order can vary across numerical-library versions.
                if source.dtype.kind in "biu":
                    reduced_sum = np.sum(
                        source,
                        axis=axes,
                        dtype=np.int64,
                    )
                    candidates.append(
                        (
                            "reduce-sum-int-v1",
                            np.asarray(reduced_sum),
                            {"accumulator_dtype": "int64"},
                        )
                    )

                for op, calculated, extra in candidates:
                    if calculated.size != target_size:
                        continue
                    try:
                        calculated = np.asarray(
                            calculated,
                            dtype=target.dtype,
                        ).reshape(target.shape)
                    except Exception:
                        continue
                    calculated = np.ascontiguousarray(calculated)
                    if not self._array_bytes_equal(target, calculated):
                        continue
                    return {
                        "format": "hdxf-derived-recipe",
                        "version": 1,
                        "op": op,
                        "source_path": source_path,
                        "reduction_prefix_ndim": int(prefix_ndim),
                        "output_shape": target_shape,
                        "output_dtype": target_dtype,
                        "logical_sha256": target_hash,
                        "logical_bytes": logical_bytes,
                        **extra,
                    }

        return None

    def _write_auxiliary_numeric_auto(
        self,
        dataset: h5py.Dataset,
        payload_entry: dict[str, Any],
        archive: zipfile.ZipFile,
        source_file_path: Path,
        path: str,
    ) -> None:
        """Choose the smallest actual lossless encoding for each aux chunk."""
        payload_entry.update({
            "encoding": "chunked-auto-v1",
            "state": "present",
            "members": [],
        })
        for index, plan in enumerate(
            iter_axis0_slices(dataset, self.target_chunk_bytes)
        ):
            array = self._read_chunk(
                dataset, plan, source_file_path, index
            )
            logical_bytes = int(array.nbytes)
            candidates: list[
                tuple[int, str, bytes, dict[str, Any]]
            ] = []

            npy_data = _npy_bytes(array)
            candidates.append(
                (
                    len(npy_data),
                    "npy",
                    npy_data,
                    {},
                )
            )

            for filter_name in (
                "none",
                "shuffle",
                "bitshuffle",
            ):
                try:
                    packed, header = encode_array_block_hdxfb(
                        array,
                        filter_name=filter_name,
                        clevel=self.zstd_level,
                        kind="auxiliary-array-block",
                    )
                except Exception:
                    continue
                candidates.append(
                    (
                        len(packed),
                        "array-block-v1",
                        packed,
                        {
                            "transform": "none",
                            "codec": "blosc2-zstd",
                            "filter": filter_name,
                            "logical_sha256": header[
                                "logical_sha256"
                            ],
                        },
                    )
                )

            _size, encoding, data, metadata = min(
                candidates, key=lambda item: item[0]
            )
            member_path, digest = self._store_payload(
                archive, encoding=encoding, data=data
            )
            payload_entry["members"].append({
                "path": member_path,
                "encoding": encoding,
                "start": plan.start,
                "count": plan.count,
                "chunk_shape": list(array.shape),
                "size_bytes": len(data),
                "sha256": digest,
                **metadata,
            })

            self.encoding_stats[
                "auxiliary_auto_logical_bytes"
            ] += logical_bytes
            self.encoding_stats[
                "auxiliary_auto_encoded_bytes"
            ] += len(data)
            if encoding == "array-block-v1":
                self.encoding_stats[
                    "auxiliary_auto_hdxfb_members"
                ] += 1
            else:
                self.encoding_stats[
                    "auxiliary_auto_npy_members"
                ] += 1


    def _archive_remaining_legacy_external_datasets(
        self,
        h5file: h5py.File,
        archive: zipfile.ZipFile,
        source_file_path: Path,
        source_id: str,
    ) -> None:
        """Archive every still-unmapped Dataset in one external HDF5 file.

        A detector master commonly links only to ``/entry/data/data`` while the
        physical data file also contains MX, azimuthal-integration, indexing,
        peak-list, diagnostics, or other per-file datasets.

        The exact-reconstruction descriptor captures the *whole* physical file
        graph.  Therefore all of those Dataset values must also be represented
        in HDXF, even though they are not reachable through the master's
        ExternalLink target.

        These auxiliary datasets are deliberately encoded as non-frame payloads
        so they cannot accidentally enlarge/change the unified detector frame
        sequence.
        """
        if self.legacy_hdf5 is None:
            return

        descriptor = self._legacy_external_layouts.get(source_id)
        if not isinstance(descriptor, dict):
            return
        objects = descriptor.get("objects")
        if not isinstance(objects, dict):
            return

        pending: list[tuple[str, dict[str, Any]]] = []
        for record in objects.values():
            if not isinstance(record, dict):
                continue
            if record.get("type") != "dataset":
                continue
            if record.get("archive_object"):
                continue
            canonical_path = str(record.get("canonical_path") or "")
            if not canonical_path:
                raise ConversionError(
                    f"External Dataset in {source_file_path.name} has no canonical path"
                )
            pending.append((canonical_path, record))

        if not pending:
            return

        pending.sort(key=lambda item: item[0])
        all_dataset_paths = sorted(
            str(record.get("canonical_path") or "")
            for record in objects.values()
            if isinstance(record, dict)
            and record.get("type") == "dataset"
            and record.get("canonical_path")
        )
        auxiliary_array_cache: dict[
            str, np.ndarray | None
        ] = {}

        _progress(
            f"archiving {len(pending)} additional Dataset(s) required for exact "
            f"reconstruction: {source_file_path.name}; "
            f"auxiliary_policy={self.auxiliary_policy}"
        )

        for index, (canonical_path, record) in enumerate(pending, start=1):
            obj = safe_get(h5file, canonical_path)
            if not isinstance(obj, h5py.Dataset):
                raise ConversionError(
                    "Exact HDF5 reconstruction descriptor references a Dataset "
                    f"that cannot be reopened: "
                    f"{source_file_path.name}::{canonical_path}"
                )

            # Reuse an already archived physical object if a hard link or an
            # earlier route has registered it since the pending list was built.
            identity = object_identity(obj, source_file_path)
            if identity is not None and identity in self._address_to_id:
                object_id = self._address_to_id[identity]
                record["archive_object"] = object_id
                self._map_legacy_archive_object(
                    obj,
                    source_file_path,
                    object_id,
                )
                continue

            object_id = uuid.uuid4().hex
            if identity is not None:
                self._address_to_id[identity] = object_id

            entry: dict[str, Any] = {
                "id": object_id,
                "type": "dataset",
                # For reconstruction-only auxiliary objects, source_path is the
                # real physical-file path. They are not inserted into the
                # master's logical link graph.
                "source_path": canonical_path,
                "source_local_path": canonical_path,
                "source_file": source_id,
                "attributes": encode_attributes(obj.attrs, h5file),
                "hdf5_storage": dataset_storage_metadata(obj, h5file),
                "payload": {
                    "encoding": "chunked",
                    "target_chunk_bytes": int(self.target_chunk_bytes),
                    "members": [],
                },
                "reconstruction_only": True,
            }
            self.objects[object_id] = entry
            self._map_legacy_archive_object(
                obj,
                source_file_path,
                object_id,
            )

            derived_recipe = None
            if self.auxiliary_policy == "auto":
                derived_recipe = (
                    self._infer_auxiliary_derived_recipe(
                        h5file,
                        canonical_path,
                        all_dataset_paths,
                        auxiliary_array_cache,
                    )
                )

            if derived_recipe is not None:
                entry["payload"] = {
                    "encoding": "derived-recipe-v1",
                    "state": "derived",
                    "recipe": derived_recipe,
                    "members": [],
                }
                self.encoding_stats[
                    "auxiliary_derived_datasets"
                ] += 1
                self.encoding_stats[
                    "auxiliary_derived_logical_bytes"
                ] += int(
                    derived_recipe.get(
                        "logical_bytes",
                        self._dataset_logical_bytes(obj),
                    )
                )
            else:
                self._write_dataset(
                    obj,
                    entry["payload"],
                    h5file,
                    archive,
                    source_file_path,
                    canonical_path,
                    object_id,
                    allow_frame=False,
                    auxiliary_auto=(
                        self.auxiliary_policy == "auto"
                    ),
                )

            if (
                index == 1
                or index == len(pending)
                or index % 100 == 0
            ):
                _progress(
                    f"exact external datasets {source_file_path.name}: "
                    f"{index}/{len(pending)}"
                )

        auxiliary_array_cache.clear()


    def _finalize_legacy_reconstruction(self) -> None:
        """Verify that every external Dataset has archived values/payloads."""
        if self.legacy_hdf5 is None:
            return

        unarchived: list[dict[str, Any]] = []
        external_files: list[dict[str, Any]] = []

        def source_sort_key(item: dict[str, Any]) -> str:
            return str(
                item.get("source_file", {}).get("filename", "")
            ).lower()

        for source_id, descriptor in self._legacy_external_layouts.items():
            for record in descriptor.get("objects", {}).values():
                if (
                    isinstance(record, dict)
                    and record.get("type") == "dataset"
                    and not record.get("archive_object")
                ):
                    unarchived.append({
                        "source_id": source_id,
                        "filename": descriptor["source_file"].get(
                            "filename"
                        ),
                        "path": record.get("canonical_path"),
                    })
            external_files.append(descriptor)

        external_files.sort(key=source_sort_key)
        self.legacy_hdf5["external_files"] = external_files
        coverage = self.legacy_hdf5["coverage"]
        coverage["unarchived_datasets"] = unarchived
        coverage["all_external_datasets_archived"] = not unarchived

        if unarchived:
            preview = ", ".join(
                f"{item['filename']}::{item['path']}"
                for item in unarchived[:8]
            )
            more = (
                f" (+{len(unarchived) - 8} more)"
                if len(unarchived) > 8
                else ""
            )
            raise ConversionError(
                "Exact HDF5 reconstruction coverage failed even after the "
                "auxiliary external-Dataset archival pass: one or more Datasets "
                "inside an external HDF5 file are still not represented by an "
                "archived HDXF object. Refusing to publish an archive that "
                f"cannot be fully restored. Missing: {preview}{more}"
            )


    def _prepare_detector_profile(self, input_path: Path) -> None:
        """Compute a strict, chunk-size-independent calibration identity."""
        digest = hashlib.sha256()
        datasets: list[dict[str, Any]] = []
        detector_serial = None
        total_logical_bytes = 0

        with h5r(input_path, "r") as h5file:
            for candidate in (
                "/entry/instrument/detector/serial_number",
                "/entry/instrument/detector/detector_number",
                "/entry/instrument/detector/description",
            ):
                obj = safe_get(h5file, candidate)
                if isinstance(obj, h5py.Dataset) and obj.shape == ():
                    value = obj[()]
                    if isinstance(value, bytes):
                        detector_serial = value.decode(
                            "utf-8", errors="replace"
                        ).rstrip("\x00")
                    else:
                        detector_serial = str(value)
                    break

            digest.update(strict_json_dumps({"detector_serial": detector_serial}))
            group = safe_get(h5file, CALIBRATION_ROOT)
            if isinstance(group, h5py.Group):
                group_attributes = encode_attributes(group.attrs, h5file)
                digest.update(
                    strict_json_dumps(
                        {"calibration_group_attributes": group_attributes}
                    )
                )

                for path, dataset in _iter_local_dataset_paths(
                    group, CALIBRATION_ROOT
                ):
                    self.calibration_paths.add(path)
                    identity_doc = {
                        "path": path,
                        "shape": (
                            None
                            if dataset.shape is None
                            else list(dataset.shape)
                        ),
                        "dtype": dtype_descriptor(dataset.dtype),
                        "attributes": encode_attributes(dataset.attrs, h5file),
                    }
                    digest.update(strict_json_dumps(identity_doc))

                    use_raw_chunks = (
                        self.calibration_read_mode == "raw-chunks"
                        or (
                            self.calibration_read_mode == "auto"
                            and _supports_raw_chunk_access(dataset)
                        )
                    )
                    if self.calibration_read_mode == "raw-chunks" and not _supports_raw_chunk_access(dataset):
                        use_raw_chunks = False

                    dataset_digest = hashlib.sha256()
                    dataset_logical_bytes = 0
                    dataset_record: dict[str, Any] = {**identity_doc}
                    _progress(
                        f"Calibration dataset: {path}; shape={dataset.shape}; "
                        f"dtype={dataset.dtype}; read_mode="
                        f"{'raw-chunks' if use_raw_chunks else 'logical'}"
                    )
                    if dataset.shape is None:
                        dataset_digest.update(b"null-dataspace")
                        dataset_record["identity_mode"] = "logical-values-v1"
                    elif (
                        dataset.shape != ()
                        and any(int(x) == 0 for x in dataset.shape)
                    ):
                        dataset_digest.update(b"empty-dataset")
                        dataset_record["identity_mode"] = "logical-values-v1"
                    elif use_raw_chunks:
                        storage_doc = _calibration_storage_identity(dataset, h5file)
                        mode_doc = {
                            "identity_mode": "hdf5-filtered-chunks-v1",
                            "storage": storage_doc,
                        }
                        encoded_mode = strict_json_dumps(mode_doc)
                        digest.update(encoded_mode)
                        dataset_digest.update(encoded_mode)
                        stored_bytes = 0
                        allocated_chunks = 0
                        for offset, filter_mask, raw_chunk in _iter_raw_hdf5_chunks(dataset):
                            chunk_doc = {
                                "chunk_offset": list(offset),
                                "filter_mask": filter_mask,
                                "size_bytes": len(raw_chunk),
                            }
                            encoded_chunk_doc = strict_json_dumps(chunk_doc)
                            digest.update(encoded_chunk_doc)
                            digest.update(raw_chunk)
                            dataset_digest.update(encoded_chunk_doc)
                            dataset_digest.update(raw_chunk)
                            stored_bytes += len(raw_chunk)
                            allocated_chunks += 1
                        dataset_logical_bytes = int(np.prod(dataset.shape, dtype=np.int64)) * max(1, int(dataset.dtype.itemsize))
                        dataset_record.update({
                            "identity_mode": "hdf5-filtered-chunks-v1",
                            "storage": storage_doc,
                            "allocated_chunks": allocated_chunks,
                            "stored_chunk_bytes": stored_bytes,
                        })
                    else:
                        dataset_record["identity_mode"] = "logical-values-v1"
                        for plan in iter_axis0_slices(
                            dataset, self.target_chunk_bytes
                        ):
                            array = read_dataset_slice(dataset, plan)
                            if array.dtype.hasobject:
                                encoded = strict_json_dumps(
                                    encode_value(array, h5file)
                                )
                                digest.update(encoded)
                                dataset_digest.update(encoded)
                                dataset_logical_bytes += len(encoded)
                            else:
                                contiguous, raw_view = _logical_array_buffer(array)
                                digest.update(raw_view)
                                dataset_digest.update(raw_view)
                                dataset_logical_bytes += int(contiguous.nbytes)

                    total_logical_bytes += dataset_logical_bytes
                    dataset_record.update({
                        "identity_sha256": dataset_digest.hexdigest(),
                        "logical_bytes": dataset_logical_bytes,
                    })
                    if dataset_record["identity_mode"] == "logical-values-v1":
                        dataset_record["logical_sha256"] = dataset_digest.hexdigest()
                    datasets.append(dataset_record)

        calibration_id = "sha256:" + digest.hexdigest() if datasets else None
        mode = self.calibration_mode if datasets else "absent"
        self.calibration_manifest = {
            "mode": mode,
            "calibration_id": calibration_id,
            "calibration_alias": None,
            "detector_serial": detector_serial,
            "required": bool(datasets),
            "dataset_count": len(datasets),
            "logical_bytes": total_logical_bytes,
            "datasets": datasets,
        }

    def _prepare_calibration_library_registration(
        self, input_path: Path
    ) -> None:
        """Resolve/reuse a stable alias before writing the experiment manifest."""
        if (
            self.calibration_library is None
            or not self.calibration_manifest
            or not self.calibration_manifest.get("calibration_id")
        ):
            return

        library_dir = self.calibration_library
        library_dir.mkdir(parents=True, exist_ok=True)
        index_document = _load_calibration_index(library_dir)
        calibration_id = str(self.calibration_manifest["calibration_id"])

        existing_group = next(
            (
                item
                for item in index_document.get("calibrations", [])
                if item.get("calibration_id") == calibration_id
            ),
            None,
        )

        if existing_group is not None:
            alias = str(existing_group["alias"])
            bundle_file = str(
                existing_group.get("bundle_file")
                or (calibration_id.split(":", 1)[1] + ".hdxf")
            )
            self._calibration_group_is_new = False
        else:
            alias = _next_calibration_alias(
                index_document, self.calibration_alias_prefix
            )
            legacy_bundle = (
                library_dir
                / (calibration_id.split(":", 1)[1] + ".hdxf")
            )
            bundle_file = (
                legacy_bundle.name
                if legacy_bundle.exists()
                else f"{alias}.hdxf"
            )
            self._calibration_group_is_new = True

        self._calibration_index_document = index_document
        self._calibration_alias = alias
        self._calibration_bundle_file = bundle_file
        self.calibration_manifest["calibration_alias"] = alias
        self.calibration_manifest["library_file"] = bundle_file
        self.calibration_manifest["library_index"] = CALIBRATION_INDEX_NAME

    def _update_calibration_library_index(self, input_path: Path) -> None:
        """Persist calibration grouping after the bundle has been verified."""
        if (
            self.calibration_library is None
            or self._calibration_index_document is None
            or not self.calibration_manifest
            or not self.calibration_manifest.get("calibration_id")
            or not self._calibration_alias
            or not self._calibration_bundle_file
        ):
            return

        library_dir = self.calibration_library
        # Reload to preserve entries that may have been added since preparation.
        index_document = _load_calibration_index(library_dir)
        calibration_id = str(self.calibration_manifest["calibration_id"])
        alias = self._calibration_alias
        bundle_file = self._calibration_bundle_file

        group = next(
            (
                item
                for item in index_document.get("calibrations", [])
                if item.get("calibration_id") == calibration_id
            ),
            None,
        )
        if group is None:
            # Another sequential conversion may have consumed our proposed alias.
            used_aliases = {
                str(item.get("alias"))
                for item in index_document.get("calibrations", [])
            }
            if alias in used_aliases:
                alias = _next_calibration_alias(
                    index_document, self.calibration_alias_prefix
                )
                old_bundle = library_dir / bundle_file
                new_bundle_file = f"{alias}.hdxf"
                new_bundle = library_dir / new_bundle_file
                if old_bundle.exists() and old_bundle != new_bundle:
                    if new_bundle.exists():
                        raise ConversionError(
                            f"Calibration alias bundle already exists: {new_bundle}"
                        )
                    os.replace(old_bundle, new_bundle)
                bundle_file = new_bundle_file
                self._calibration_alias = alias
                self._calibration_bundle_file = bundle_file
                self.calibration_manifest["calibration_alias"] = alias
                self.calibration_manifest["library_file"] = bundle_file

            group = {
                "alias": alias,
                "calibration_id": calibration_id,
                "bundle_file": bundle_file,
                "detector_serial": self.calibration_manifest.get(
                    "detector_serial"
                ),
                "dataset_count": int(
                    self.calibration_manifest.get("dataset_count", 0)
                ),
                "logical_bytes": int(
                    self.calibration_manifest.get("logical_bytes", 0)
                ),
                "source_count": 0,
                "representative_source": input_path.name,
                "datasets": self.calibration_manifest.get("datasets", []),
            }
            index_document.setdefault("calibrations", []).append(group)
        else:
            # Existing authoritative group wins.
            alias = str(group.get("alias") or alias)
            bundle_file = str(group.get("bundle_file") or bundle_file)
            self._calibration_alias = alias
            self._calibration_bundle_file = bundle_file
            self.calibration_manifest["calibration_alias"] = alias
            self.calibration_manifest["library_file"] = bundle_file

        source_key = _source_index_key(input_path)
        file_entry = {
            "source_key": source_key,
            "source_file": input_path.name,
            "status": "grouped",
            "calibration_alias": alias,
            "calibration_id": calibration_id,
            "bundle_file": bundle_file,
            "detector_serial": self.calibration_manifest.get(
                "detector_serial"
            ),
            "dataset_count": int(
                self.calibration_manifest.get("dataset_count", 0)
            ),
            "logical_bytes": int(
                self.calibration_manifest.get("logical_bytes", 0)
            ),
        }

        files = index_document.setdefault("files", [])
        replaced = False
        for index, item in enumerate(files):
            if item.get("source_key") == source_key:
                files[index] = file_entry
                replaced = True
                break
        if not replaced:
            files.append(file_entry)

        source_counts: dict[str, int] = {}
        for item in files:
            current_id = item.get("calibration_id")
            if current_id:
                source_counts[str(current_id)] = (
                    source_counts.get(str(current_id), 0) + 1
                )
        for item in index_document.get("calibrations", []):
            item["source_count"] = source_counts.get(
                str(item.get("calibration_id")), 0
            )

        index_document["updated_utc"] = utc_now_iso()
        index_document["version"] = CALIBRATION_INDEX_VERSION
        index_document["library"] = {
            "directory": ".",
            "unique_calibrations": len(
                index_document.get("calibrations", [])
            ),
            "source_files_indexed": len(files),
        }

        _atomic_write_json(
            library_dir / CALIBRATION_INDEX_NAME, index_document
        )
        csv_rows = []
        for item in sorted(
            files,
            key=lambda row: (
                str(row.get("calibration_alias") or ""),
                str(row.get("source_file") or ""),
            ),
        ):
            csv_rows.append(
                {
                    "source_file": item.get("source_file", ""),
                    "status": item.get("status", ""),
                    "calibration_alias": item.get(
                        "calibration_alias", ""
                    ),
                    "calibration_id": item.get("calibration_id", ""),
                    "detector_serial": item.get(
                        "detector_serial", ""
                    )
                    or "",
                    "dataset_count": item.get("dataset_count", 0),
                    "logical_bytes": item.get("logical_bytes", 0),
                    "bundle_file": item.get("bundle_file", ""),
                    "error": item.get("error", ""),
                }
            )
        _atomic_write_csv(
            library_dir / CALIBRATION_REPORT_NAME, csv_rows
        )

    def _write_calibration_bundle(self, input_path: Path) -> None:
        if not self.calibration_manifest or not self.calibration_paths:
            return
        assert self.calibration_library is not None
        self.calibration_library.mkdir(parents=True, exist_ok=True)
        calibration_id = str(self.calibration_manifest["calibration_id"])
        bundle_file = str(
            self.calibration_manifest.get("library_file")
            or (calibration_id.split(":", 1)[1] + ".hdxf")
        )
        output = self.calibration_library / bundle_file
        if output.exists():
            try:
                with zipfile.ZipFile(output, "r") as existing:
                    existing_manifest = json.loads(existing.read(MANIFEST_PATH))
                existing_id = existing_manifest.get("calibration", {}).get("calibration_id")
            except Exception as exc:
                raise ConversionError(
                    f"Existing calibration bundle is unreadable: {output}: {exc}"
                ) from exc
            if existing_id != calibration_id:
                raise ConversionError(
                    f"Existing calibration bundle ID mismatch: {output}"
                )
            return
        fd, temp_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
        os.close(fd)
        temp = Path(temp_name)
        members: list[dict[str, Any]] = []
        written: set[str] = set()
        try:
            with h5r(input_path, "r") as h5file, zipfile.ZipFile(
                temp, "w", compression=zipfile.ZIP_STORED, allowZip64=True
            ) as archive:
                dataset_records = {
                    str(item.get("path")): item
                    for item in self.calibration_manifest.get("datasets", [])
                }
                for path in sorted(self.calibration_paths):
                    dataset = safe_get(h5file, path)
                    if not isinstance(dataset, h5py.Dataset) or dataset.shape is None:
                        continue
                    record = dataset_records.get(path, {})
                    if record.get("identity_mode") == "hdf5-filtered-chunks-v1":
                        _progress(f"preserving Calibration raw HDF5 chunks: {path}")
                        for offset, filter_mask, raw_chunk in _iter_raw_hdf5_chunks(dataset):
                            sha = sha256_bytes(raw_chunk)
                            member_path = f"payloads/{sha}.h5chunk"
                            if member_path not in written:
                                archive.writestr(member_path, raw_chunk)
                                written.add(member_path)
                            members.append({
                                "dataset_path": path,
                                "path": member_path,
                                "encoding": "hdf5-filtered-chunk-v1",
                                "sha256": sha,
                                "size_bytes": len(raw_chunk),
                                "chunk_offset": list(offset),
                                "filter_mask": filter_mask,
                            })
                        continue
                    for plan in iter_axis0_slices(dataset, self.target_chunk_bytes):
                        array = read_dataset_slice(dataset, plan)
                        data, header = encode_array_block_hdxfb(
                            array, filter_name="shuffle", clevel=self.zstd_level, kind="calibration-block"
                        )
                        sha = sha256_bytes(data)
                        member_path = f"payloads/{sha}.hdxfb"
                        if member_path not in written:
                            archive.writestr(member_path, data)
                            written.add(member_path)
                        members.append({
                            "dataset_path": path,
                            "path": member_path,
                            "encoding": "array-block-v1",
                            "sha256": sha,
                            "size_bytes": len(data),
                            "start": plan.start,
                            "count": plan.count,
                            "chunk_shape": list(array.shape),
                            "logical_sha256": header["logical_sha256"],
                        })
                archive.writestr(
                    MANIFEST_PATH,
                    strict_json_dumps({
                        "hdxf": {
                            "format": FORMAT_NAME,
                            "format_id": FORMAT_ID,
                            "version": FORMAT_VERSION,
                            "profile": "calibration-bundle",
                        },
                        "calibration": self.calibration_manifest,
                        "members": members,
                    }, indent=2),
                )
            os.replace(temp, output)
        except Exception:
            temp.unlink(missing_ok=True)
            raise

    def _walk_hdf5(self, h5file: h5py.File, archive: zipfile.ZipFile, source_file_path: Path) -> str:
        return self._register_object(h5file["/"], "/", h5file, archive, source_file_path)

    def _resolve_external_file(self, parent_file_path: Path, link_filename: str) -> Path:
        raw = os.path.expandvars(os.path.expanduser(os.fsdecode(link_filename)))
        linked = Path(raw)
        if linked.is_absolute():
            return linked.resolve()
        base = self.external_root if self.external_root is not None else parent_file_path.parent
        return (base / linked).resolve()

    def _register_source_file(self, path: Path, *, role: str) -> str:
        path = path.resolve()
        key = _normalise_source_filename(path)
        existing = self._source_path_to_id.get(key)
        if existing is not None:
            for entry in self.source_files:
                if entry.get("id") == existing:
                    current_role = entry.get("role")
                    if current_role != role:
                        roles = set(entry.get("roles") or [])
                        if current_role:
                            roles.add(str(current_role))
                        roles.add(str(role))
                        entry["roles"] = sorted(roles)
                    break
            return existing
        source_id = f"source-{len(self.source_files) + 1:06d}"
        entry: dict[str, Any] = {
            "id": source_id,
            "filename": path.name,
            "role": role,
        }
        if self.source_checksum:
            entry["sha256"] = sha256_file(path)
            entry["size_bytes"] = int(path.stat().st_size)
        self._source_path_to_id[key] = source_id
        self.source_files.append(entry)
        return source_id

    def _source_id(self, path: Path) -> str:
        key = _normalise_source_filename(path)
        source_id = self._source_path_to_id.get(key)
        if source_id is None:
            source_id = self._register_source_file(path, role="external-data")
        return source_id

    def _store_payload(self, archive: zipfile.ZipFile, *, encoding: str, data: bytes) -> tuple[str, str]:
        digest = sha256_bytes(data)
        extension = {
            "npy": "npy", "json": "json",
            "frame-block-v1": "hdxfb", "frame-block-v2": "hdxfb",
            "frame-block-v3": "hdxfb", "frame-block-v4": "hdxfb",
            "frame-index-v1": "hdxfi",
            "array-block-v1": "hdxfb",
            "hdf5-filtered-chunk-v1": "h5chunk",
        }.get(encoding)
        if extension is None:
            raise ConversionError(f"Unsupported payload encoding: {encoding}")
        member_path = f"payloads/{digest}.{extension}"
        existing = self._written_payloads.get(member_path)
        if existing is None:
            archive.writestr(member_path, data)
            self._written_payloads[member_path] = {
                "sha256": digest, "size_bytes": len(data), "encoding": encoding
            }
            self.encoding_stats["unique_payload_bytes"] += len(data)
            by_encoding = self.encoding_stats["payload_bytes_by_encoding"]
            by_encoding[encoding] = int(by_encoding.get(encoding, 0)) + len(data)
        elif existing["size_bytes"] != len(data) or existing["sha256"] != digest:
            raise ConversionError(f"Content-address collision: {member_path}")
        self.payload_reference_count += 1
        return member_path, digest

    def _register_external_link(
        self, *, link: h5py.ExternalLink, child_path: str, parent_file_path: Path, archive: zipfile.ZipFile
    ) -> str | None:
        external_path = self._resolve_external_file(parent_file_path, link.filename)
        _progress(f"opening external HDF5: {external_path.name}::{link.path}")
        manifest_target_file, target_was_absolute = _manifest_external_filename(link.filename)
        link_entry: dict[str, Any] = {
            "path": child_path, "type": "external", "target_file": manifest_target_file, "target_path": link.path
        }
        if target_was_absolute:
            link_entry["target_file_was_absolute"] = True
        self.links.append(link_entry)
        if not self.follow_external:
            return None
        if not external_path.is_file():
            link_entry["resolution"] = "missing"
            link_entry["error"] = "External HDF5 data file not found"
            if self.allow_missing_external:
                return None
            raise ConversionError(f"External HDF5 data file not found: {external_path}")
        try:
            with h5r(external_path, "r") as external_h5:
                target = safe_get(external_h5, link.path)
                if target is None:
                    link_entry["resolution"] = "missing_target"
                    if self.allow_missing_external:
                        return None
                    raise ConversionError(f"External target does not exist: {external_path}::{link.path}")
                external_source_id = self._register_source_file(
                    external_path, role="external-data"
                )
                link_entry["target_source_file"] = external_source_id

                # Capture the complete file-local HDF5 graph before archiving
                # the target object.  This preserves Groups/attrs/links that
                # are not visible through the master's ExternalLink path.
                self._capture_legacy_external_file(
                    external_h5,
                    external_path,
                    external_source_id,
                )
                if self.legacy_hdf5 is not None:
                    legacy_link_doc = {
                        "link_path": child_path,
                        "parent_source_file": self._source_id(
                            parent_file_path
                        ),
                        "target_source_file": external_source_id,
                        "target_file": manifest_target_file,
                        "target_path": str(link.path),
                    }
                    if target_was_absolute:
                        legacy_link_doc[
                            "target_file_was_absolute"
                        ] = True
                    self._legacy_external_links.append(
                        legacy_link_doc
                    )

                object_id = self._register_object(
                    target,
                    child_path,
                    external_h5,
                    archive,
                    external_path,
                )

                # The master usually exposes only the detector Dataset, but the
                # physical data file can contain thousands of additional MX /
                # azint / peak / diagnostic Datasets.  Exact reconstruction
                # requires those values too.
                self._archive_remaining_legacy_external_datasets(
                    external_h5,
                    archive,
                    external_path,
                    external_source_id,
                )

                link_entry["resolution"] = "embedded"
                link_entry["target_object"] = object_id
                return object_id
        except ConversionError:
            raise
        except Exception as exc:
            link_entry["resolution"] = "error"
            link_entry["error"] = type(exc).__name__
            if self.allow_missing_external:
                return None
            raise ConversionError(
                f"Cannot open external HDF5 target {external_path}::{link.path}: {type(exc).__name__}: {exc}"
            ) from exc

    def _register_object(
        self, obj: h5py.Group | h5py.Dataset, path: str, h5file: h5py.File, archive: zipfile.ZipFile, source_file_path: Path
    ) -> str:
        identity = object_identity(obj, source_file_path)
        if identity is not None and identity in self._address_to_id:
            object_id = self._address_to_id[identity]
            self.links.append(
                {"path": path, "type": "hard", "target_object": object_id}
            )
            self._map_legacy_archive_object(
                obj, source_file_path, object_id
            )
            return object_id
        object_id = uuid.uuid4().hex
        if identity is not None:
            self._address_to_id[identity] = object_id
        source_file_value = self._source_id(source_file_path)
        if isinstance(obj, h5py.Group):
            entry: dict[str, Any] = {
                "id": object_id,
                "type": "group",
                "source_path": path,
                "source_local_path": str(obj.name),
                "source_file": source_file_value,
                "attributes": encode_attributes(obj.attrs, h5file),
                "children": [],
            }
            self.objects[object_id] = entry
            self._map_legacy_archive_object(
                obj, source_file_path, object_id
            )
            for name in obj.keys():
                child_path = path.rstrip("/") + "/" + name if path != "/" else "/" + name
                link = obj.get(name, getlink=True)
                if isinstance(link, h5py.SoftLink):
                    self.links.append({"path": child_path, "type": "soft", "target_path": link.path})
                    entry["children"].append({"name": name, "link_path": child_path})
                    continue
                if isinstance(link, h5py.ExternalLink):
                    child_id = self._register_external_link(
                        link=link, child_path=child_path, parent_file_path=source_file_path, archive=archive
                    )
                    entry["children"].append(
                        {"name": name, "link_path": child_path} if child_id is None else
                        {"name": name, "object": child_id, "via_external_link": child_path}
                    )
                    continue
                try:
                    child_obj = obj[name]
                except Exception as exc:
                    self.links.append({"path": child_path, "type": "unreadable", "error": f"{type(exc).__name__}: {exc}"})
                    entry["children"].append({"name": name, "link_path": child_path})
                    continue
                child_id = self._register_object(child_obj, child_path, h5file, archive, source_file_path)
                entry["children"].append({"name": name, "object": child_id})
            return object_id
        if isinstance(obj, h5py.Dataset):
            entry = {
                "id": object_id,
                "type": "dataset",
                "source_path": path,
                "source_local_path": str(obj.name),
                "source_file": source_file_value,
                "attributes": encode_attributes(obj.attrs, h5file),
                "hdf5_storage": dataset_storage_metadata(obj, h5file),
                "payload": {"encoding": "chunked", "target_chunk_bytes": int(self.target_chunk_bytes), "members": []},
            }
            self.objects[object_id] = entry
            self._map_legacy_archive_object(
                obj, source_file_path, object_id
            )
            self._write_dataset(
                obj,
                entry["payload"],
                h5file,
                archive,
                source_file_path,
                path,
                object_id,
                allow_frame=True,
            )
            return object_id
        raise ConversionError(f"Unsupported HDF5 object at {path}: {type(obj).__name__}")

    def _prepare_experiment_static_model(self, input_path: Path) -> None:
        _progress("pre-scanning experiment-global static pixels")
        started = time.perf_counter()
        reference: np.ndarray | None = None
        static_mask: np.ndarray | None = None
        dataset_count = 0
        scan_block_frames = max(self.block_frames, 64)
        with h5r(input_path, "r") as h5file:
            candidates: list[tuple[str, h5py.Dataset]] = []
            if self.frame_datasets:
                for path in sorted(self.frame_datasets):
                    obj = safe_get(h5file, path)
                    if isinstance(obj, h5py.Dataset):
                        candidates.append((path, obj))
            else:
                group = safe_get(h5file, "/entry/data")
                if isinstance(group, h5py.Group):
                    for name in sorted(group.keys()):
                        path = f"/entry/data/{name}"
                        try:
                            obj = group[name]
                        except Exception as exc:
                            raise ConversionError(
                                f"Cannot open frame Dataset during global static scan: {path}: {exc}"
                            ) from exc
                        if isinstance(obj, h5py.Dataset) and self._is_frame_dataset(path, obj):
                            candidates.append((path, obj))
            if not candidates:
                raise ConversionError("experiment static scan found no frame Datasets")
            for path, dataset in candidates:
                if dataset.shape is None or int(dataset.shape[0]) <= 0:
                    continue
                if reference is None:
                    reference = np.ascontiguousarray(dataset[0])
                    static_mask = np.ones(reference.shape, dtype=bool)
                elif tuple(dataset.shape[1:]) != reference.shape or dataset.dtype != reference.dtype:
                    raise ConversionError(
                        f"Frame Dataset {path} is incompatible with experiment static model"
                    )
                assert static_mask is not None and reference is not None
                dataset_count += 1
                for start0 in range(0, int(dataset.shape[0]), scan_block_frames):
                    stop0 = min(int(dataset.shape[0]), start0 + scan_block_frames)
                    scan_block = np.asarray(dataset[start0:stop0])
                    static_mask &= np.all(scan_block == reference[None, ...], axis=0)
                    del scan_block
                    if not np.any(static_mask):
                        break
                if not np.any(static_mask):
                    break
        if reference is None or static_mask is None:
            return
        elapsed = time.perf_counter() - started
        static_count = int(np.count_nonzero(static_mask))
        pixel_count = int(static_mask.size)
        self.encoding_stats["timings"]["static_scan_seconds"] += elapsed
        self.encoding_stats["static_scan_datasets"] += dataset_count
        self.encoding_stats["static_scan_pixels"] += pixel_count
        self.encoding_stats["static_pixels"] += static_count
        _progress(
            f"experiment static scan: datasets={dataset_count}, "
            f"{static_count}/{pixel_count} pixels ({100.0 * static_count / pixel_count:.2f}%), "
            f"{elapsed:.1f}s"
        )
        if 0 < static_count < pixel_count:
            self._experiment_static_mask = static_mask
            self._experiment_static_reference = reference

    def _is_frame_dataset(self, path: str, dataset: h5py.Dataset) -> bool:
        if self.profile != "detector-frame-archive":
            return False
        if self.frame_datasets:
            return path in self.frame_datasets
        return (
            path.startswith("/entry/data/")
            and dataset.shape is not None
            and len(dataset.shape) >= 3
            and dataset.dtype.kind in "iuf"
            and not dataset.dtype.hasobject
        )

    def _read_chunk(self, dataset: h5py.Dataset, plan: SlicePlan, source_file_path: Path, index: int) -> np.ndarray:
        try:
            return read_dataset_slice(dataset, plan)
        except OSError as exc:
            filters = dataset_filter_pipeline(dataset)
            filter_ids = [item["id"] for item in filters]
            hint = ""
            if 32008 in filter_ids and not HDF5PLUGIN_AVAILABLE:
                hint = " Bitshuffle/LZ4 filter 32008 detected; install hdf5plugin."
            raise ConversionError(
                f"Cannot read dataset {source_file_path}::{dataset.name} at chunk {index}; "
                f"filters={filters}. {type(exc).__name__}: {exc}.{hint}"
            ) from exc

    def _scan_dataset_static_model(
        self, dataset: h5py.Dataset, source_file_path: Path, path: str
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.global_static_scan == "experiment":
            return self._experiment_static_mask, self._experiment_static_reference
        if (
            self.global_static_scan != "dataset"
            or self.frame_transform not in ("delta", "auto")
        ):
            return None, None
        assert dataset.shape is not None and int(dataset.shape[0]) > 0
        _progress(f"pre-scanning dataset-global static pixels: {path}")
        started = time.perf_counter()
        first_plan = SlicePlan(
            selection=(slice(0, 1),) + tuple(slice(None) for _ in dataset.shape[1:]),
            start=[0] + [0] * (len(dataset.shape) - 1),
            count=[1] + [int(x) for x in dataset.shape[1:]],
        )
        first_block = self._read_chunk(dataset, first_plan, source_file_path, -1)
        reference = np.ascontiguousarray(first_block[0])
        static_mask = np.ones(reference.shape, dtype=bool)
        scan_block_frames = max(self.block_frames, 64)
        for start0 in range(0, int(dataset.shape[0]), scan_block_frames):
            stop0 = min(int(dataset.shape[0]), start0 + scan_block_frames)
            plan = SlicePlan(
                selection=(slice(start0, stop0),) + tuple(slice(None) for _ in dataset.shape[1:]),
                start=[start0] + [0] * (len(dataset.shape) - 1),
                count=[stop0 - start0] + [int(x) for x in dataset.shape[1:]],
            )
            scan_block = self._read_chunk(dataset, plan, source_file_path, start0 // scan_block_frames)
            static_mask &= np.all(scan_block == reference[None, ...], axis=0)
            del scan_block
            if not np.any(static_mask):
                break
        elapsed = time.perf_counter() - started
        static_count = int(np.count_nonzero(static_mask))
        pixel_count = int(static_mask.size)
        self.encoding_stats["timings"]["static_scan_seconds"] += elapsed
        self.encoding_stats["static_scan_datasets"] += 1
        self.encoding_stats["static_scan_pixels"] += pixel_count
        self.encoding_stats["static_pixels"] += static_count
        _progress(
            f"static scan {path}: {static_count}/{pixel_count} pixels "
            f"({100.0 * static_count / pixel_count:.2f}%), {elapsed:.1f}s"
        )
        if static_count == 0 or static_count == pixel_count:
            # All-static data is already handled efficiently by Delta; avoid a split with no dynamic values.
            return None, None
        return static_mask, reference

    def _register_shared_static_model(
        self,
        archive: zipfile.ZipFile,
        *,
        source_object: str,
        source_path: str,
        static_mask: np.ndarray | None,
        static_reference: np.ndarray | None,
    ) -> str | None:
        """Store one reusable static mask/value dictionary and return its ID."""
        if static_mask is None or static_reference is None:
            return None
        mask = np.asarray(static_mask, dtype=bool)
        reference = np.ascontiguousarray(static_reference)
        if mask.shape != reference.shape:
            raise ConversionError("shared static mask/reference shape mismatch")
        static_count = int(np.count_nonzero(mask))
        if static_count <= 0 or static_count >= mask.size:
            return None
        mask_bytes = np.packbits(mask.reshape(-1).astype(np.uint8), bitorder="little")
        static_values = np.ascontiguousarray(reference.reshape(-1)[mask.reshape(-1)])
        identity = hashlib.sha256()
        identity.update(memoryview(mask_bytes).cast("B"))
        identity.update(memoryview(static_values).cast("B"))
        identity.update(str(reference.dtype).encode("ascii"))
        identity.update(repr(reference.shape).encode("ascii"))
        key = identity.hexdigest()
        existing = self._static_model_by_key.get(key)
        if existing is not None:
            return str(existing["id"])
        mask_payload, mask_header = encode_array_block_hdxfb(
            mask_bytes, filter_name="none", clevel=self.zstd_level, kind="static-model-mask"
        )
        values_payload, values_header = encode_array_block_hdxfb(
            static_values, filter_name="bitshuffle", clevel=self.zstd_level, kind="static-model-values"
        )
        mask_path, mask_sha = self._store_payload(
            archive, encoding="array-block-v1", data=mask_payload
        )
        values_path, values_sha = self._store_payload(
            archive, encoding="array-block-v1", data=values_payload
        )
        model_id = f"static-model-{len(self.frame_archive['static_models']) + 1:06d}"
        model = {
            "id": model_id,
            "identity_sha256": key,
            "source_object": source_object,
            "source_path": source_path,
            "frame_shape": list(reference.shape),
            "logical_dtype": dtype_descriptor(reference.dtype),
            "pixel_count": int(mask.size),
            "static_pixel_count": static_count,
            "dynamic_pixel_count": int(mask.size - static_count),
            "mask": {
                "path": mask_path, "encoding": "array-block-v1",
                "size_bytes": len(mask_payload), "sha256": mask_sha,
                "logical_sha256": mask_header["logical_sha256"],
                "chunk_shape": list(mask_bytes.shape),
            },
            "static_values": {
                "path": values_path, "encoding": "array-block-v1",
                "size_bytes": len(values_payload), "sha256": values_sha,
                "logical_sha256": values_header["logical_sha256"],
                "chunk_shape": list(static_values.shape),
            },
        }
        self.frame_archive["static_models"].append(model)
        self._static_model_by_key[key] = model
        self.encoding_stats["shared_static_models"] += 1
        self.encoding_stats["shared_static_model_payload_bytes"] += len(mask_payload) + len(values_payload)
        _progress(
            f"shared static model {model_id}: static={static_count}/{mask.size} "
            f"({100.0 * static_count / mask.size:.2f}%), payload={(len(mask_payload)+len(values_payload))/1024:.1f} KiB"
        )
        return model_id

    def _calibration_record(self, path: str) -> dict[str, Any]:
        if not self.calibration_manifest:
            return {}
        for item in self.calibration_manifest.get("datasets", []):
            if str(item.get("path")) == path:
                return item
        return {}

    def _write_embedded_calibration_raw_chunks(
        self,
        dataset: h5py.Dataset,
        payload_entry: dict[str, Any],
        archive: zipfile.ZipFile,
        *,
        path: str,
    ) -> None:
        """Embed the exact filtered HDF5 chunks without invoking native decoding."""
        if dataset.shape is None or dataset.chunks is None:
            raise ConversionError(f"Calibration raw-chunk embedding requires chunked storage: {path}")
        payload_entry.update({
            "encoding": "calibration-raw-chunks-v1",
            "state": "present",
            "members": [],
            "filter_pipeline": dataset_filter_pipeline(dataset),
        })
        for offset, filter_mask, raw_chunk in _iter_raw_hdf5_chunks(dataset):
            member_path, digest = self._store_payload(
                archive, encoding="hdf5-filtered-chunk-v1", data=raw_chunk
            )
            count = [
                min(int(dataset.chunks[axis]), int(dataset.shape[axis]) - int(offset[axis]))
                for axis in range(len(dataset.shape))
            ]
            member = {
                "path": member_path,
                "encoding": "hdf5-filtered-chunk-v1",
                "start": [int(x) for x in offset],
                "count": count,
                "chunk_shape": count,
                "size_bytes": len(raw_chunk),
                "sha256": digest,
                "chunk_offset": [int(x) for x in offset],
                "filter_mask": int(filter_mask),
            }
            payload_entry["members"].append(member)
            self.encoding_stats["embedded_calibration_payload_bytes"] += len(raw_chunk)

    def _write_frame_dataset(
        self, dataset: h5py.Dataset, payload_entry: dict[str, Any], archive: zipfile.ZipFile,
        source_file_path: Path, path: str, object_id: str
    ) -> None:
        assert dataset.shape is not None and len(dataset.shape) >= 3
        frame_shape = list(dataset.shape[1:])
        dtype = dtype_descriptor(dataset.dtype)
        if self.frame_archive["frame_shape"] is None:
            self.frame_archive["frame_shape"] = frame_shape
            self.frame_archive["dtype"] = dtype
        elif self.frame_archive["frame_shape"] != frame_shape or self.frame_archive["dtype"] != dtype:
            raise ConversionError(
                f"Frame dataset {path} is incompatible with the unified sequence: "
                f"shape={frame_shape}, dtype={dtype}"
            )
        global_start = int(self.frame_archive["frame_count"])
        frame_count = int(dataset.shape[0])
        payload_entry.update({
            "encoding": "frame-sequence", "state": "present", "block_frames": self.block_frames,
            "global_start_frame": global_start, "members": []
        })
        total_blocks = (frame_count + self.block_frames - 1) // self.block_frames
        frame_index_start = len(self.frame_archive["blocks"])
        static_mask, static_reference = self._scan_dataset_static_model(
            dataset, source_file_path, path
        )
        # Dataset-scoped static masks stay inline (HDXFB v3).  The 0.4.3
        # shared-model experiment increased real archive size for this data.
        # Keep shared HDXFB v4 only for explicitly requested experiment scope.
        shared_static_model_id = None
        if self.global_static_scan == "experiment":
            shared_static_model_id = self._register_shared_static_model(
                archive,
                source_object=object_id,
                source_path=path,
                static_mask=static_mask,
                static_reference=static_reference,
            )
        previous_frame: np.ndarray | None = None
        current_anchor_global_start = global_start
        _progress(
            f"frame dataset {path}: frames={frame_count}, shape={tuple(dataset.shape[1:])}, "
            f"dtype={dataset.dtype}, blocks={total_blocks}, block_frames={self.block_frames}"
        )
        for block_index, start0 in enumerate(range(0, frame_count, self.block_frames)):
            stop0 = min(frame_count, start0 + self.block_frames)
            plan = SlicePlan(
                selection=(slice(start0, stop0),) + tuple(slice(None) for _ in dataset.shape[1:]),
                start=[start0] + [0] * (len(dataset.shape) - 1),
                count=[stop0 - start0] + [int(x) for x in dataset.shape[1:]],
            )
            should_report = (
                block_index == 0
                or block_index + 1 == total_blocks
                or (block_index + 1) % self.progress_every == 0
            )
            if should_report:
                _progress(
                    f"reading frame block {block_index + 1}/{total_blocks}: "
                    f"local frames {start0}..{stop0 - 1}"
                )
            read_started = time.perf_counter()
            block = self._read_chunk(dataset, plan, source_file_path, block_index)
            self.encoding_stats["timings"]["read_seconds"] += time.perf_counter() - read_started
            if should_report:
                _progress(
                    f"encoding frame block {block_index + 1}/{total_blocks}: "
                    f"memory={block.nbytes / (1024 * 1024):.1f} MiB, transform={self.frame_transform}"
                )
            if self.delta_filter_policy == "learn":
                effective_delta_filter = self._delta_filter_cache.get(path, "auto")
            else:
                effective_delta_filter = self.delta_filter_policy

            # Delta and Auto share the same chain schedule.  At the configured
            # interval we force the Delta candidate to be an anchor; otherwise
            # it may continue from the previous logical frame even when the
            # previous selected Auto block happened to be Raw or Adaptive.
            #
            # That is safe because Raw/Adaptive blocks are independently
            # decodable and therefore themselves act as restart/keyframe
            # points in the frame index.
            force_delta_anchor = (
                self.frame_transform in ("delta", "auto")
                and (
                    block_index % self.keyframe_interval_blocks == 0
                    or previous_frame is None
                )
            )

            data, header, candidate_sizes, encode_timings = encode_frame_block_hdxfb(
                block,
                transform=self.frame_transform,
                clevel=self.zstd_level,
                tile_height=self.tile_height,
                tile_width=self.tile_width,
                sparse_threshold=self.sparse_threshold,
                enable_jungfrau_split=self.enable_jungfrau_split,
                delta_compute_backend=self.delta_compute_backend,
                cuda_device=self.cuda_device,
                delta_filter=effective_delta_filter,
                previous_frame=None if force_delta_anchor else previous_frame,
                static_mask=static_mask,
                static_reference=static_reference if force_delta_anchor else None,
                shared_static_model_id=shared_static_model_id,
                delta_stream=self.delta_stream,
                zero_rle_threshold=self.zero_rle_threshold,
            )

            for timing_name, timing_value in encode_timings.items():
                if timing_name in self.encoding_stats["timings"]:
                    self.encoding_stats["timings"][timing_name] += float(timing_value)

            if self.delta_filter_policy == "learn" and path not in self._delta_filter_cache:
                learned = (
                    header.get("selected_delta_filter")
                    or header.get("auto_delta_selected_filter")
                )
                if learned in ("none", "shuffle", "bitshuffle"):
                    self._delta_filter_cache[path] = str(learned)
                    _progress(f"learned Delta filter for {path}: {learned}")

            if self.frame_transform == "auto" and should_report:
                raw_size = candidate_sizes.get("raw")
                delta_size = candidate_sizes.get("delta-chain-v1")
                adaptive_size = candidate_sizes.get("adaptive")

                def _candidate_mib(value: int | None) -> str:
                    return (
                        f"{value / (1024 * 1024):.2f} MiB"
                        if value is not None
                        else "n/a"
                    )

                _progress(
                    "auto candidates: "
                    f"raw={_candidate_mib(raw_size)}, "
                    f"delta-chain={_candidate_mib(delta_size)}, "
                    f"adaptive={_candidate_mib(adaptive_size)}, "
                    f"selected={header.get('auto_selected_candidate', header.get('transform'))}"
                )
            hdxfb_version = int(header.get("version", 1))
            block_encoding = (
                "frame-block-v4" if hdxfb_version == 4
                else "frame-block-v3" if hdxfb_version == 3
                else "frame-block-v2" if hdxfb_version == 2
                else "frame-block-v1"
            )
            write_started = time.perf_counter()
            member_path, digest = self._store_payload(archive, encoding=block_encoding, data=data)
            self.encoding_stats["timings"]["archive_write_seconds"] += time.perf_counter() - write_started
            transform = str(header["transform"])
            requires_previous_frame = bool(
                header.get("requires_previous_frame", False)
            )

            # Any independently decodable selected block (Raw, Adaptive, or a
            # Delta anchor) becomes the newest restart point for following
            # continuation blocks.
            if not requires_previous_frame:
                current_anchor_global_start = global_start + start0

            self.encoding_stats["frame_blocks"] += 1
            self.encoding_stats["logical_frame_bytes"] += int(block.nbytes)
            self.encoding_stats["encoded_frame_bytes"] += int(len(data))
            if transform in ("delta-prev", "delta-chain-v1"):
                self.encoding_stats["delta_blocks"] += 1
                chain_mode = str(header.get("chain_mode", "anchor"))
                self.encoding_stats["anchor_blocks" if chain_mode == "anchor" else "continuation_blocks"] += 1
                self.encoding_stats["keyframe_payload_bytes"] += int(header.get("keyframe_payload_bytes", 0))
                self.encoding_stats["delta_payload_bytes"] += int(header.get("delta_payload_bytes", 0))
                if header.get("delta_encoding") == "zero-rle-bitpack":
                    self.encoding_stats["delta_zero_rle_blocks"] += 1
                else:
                    self.encoding_stats["delta_zstd_blocks"] += 1
                if header.get("keyframe_encoding") in ("static-split", "shared-static-split"):
                    self.encoding_stats["static_split_keyframes"] += 1
            elif transform == "tile-adaptive-v1":
                self.encoding_stats["adaptive_blocks"] += 1
                tile_stats = self.encoding_stats["tile_modes"]
                for mode, count in header.get("mode_counts", {}).items():
                    tile_stats[mode] = int(tile_stats.get(mode, 0)) + int(count)
            else:
                self.encoding_stats["raw_blocks"] += 1
            member = {
                "path": member_path, "encoding": block_encoding, "start": plan.start, "count": plan.count,
                "chunk_shape": list(block.shape), "size_bytes": len(data), "sha256": digest,
                "start_frame": start0, "global_start_frame": global_start + start0,
                "frame_count": stop0 - start0, "transform": transform,
                "codec": "blosc2-zstd", "candidate_sizes": candidate_sizes,
                "logical_sha256": header["logical_sha256"],
                "hdxfb_version": hdxfb_version,
                "chain_mode": header.get("chain_mode"),
                "requires_previous_frame": requires_previous_frame,
                "chain_anchor_global_start_frame": current_anchor_global_start,
                "delta_encoding": header.get("delta_encoding"),
                "keyframe_encoding": header.get("keyframe_encoding"),
                "static_model_id": shared_static_model_id,
            }
            if "mode_counts" in header:
                member["tile_mode_counts"] = header["mode_counts"]
            if "tile_shape" in header:
                member["tile_shape"] = header["tile_shape"]
            if "delta_dtype" in header:
                member["delta_dtype"] = header["delta_dtype"]
            if self.frame_index_layout == "json":
                payload_entry["members"].append(member)
            self.frame_archive["blocks"].append({**member, "source_object": object_id})
            if not requires_previous_frame:
                self.frame_archive["keyframe_index"].append(global_start + start0)
            previous_frame = np.ascontiguousarray(block[-1])
            del data, header, block
            if self.gc_every and (block_index + 1) % self.gc_every == 0:
                gc.collect()
        frame_index_count = len(self.frame_archive["blocks"]) - frame_index_start
        if self.frame_index_layout == "binary":
            payload_entry["encoding"] = "frame-sequence-indexed"
            payload_entry["block_index_start"] = frame_index_start
            payload_entry["block_index_count"] = frame_index_count
            payload_entry.pop("members", None)
        self.frame_archive["datasets"].append({
            "object": object_id, "source_path": path, "global_start_frame": global_start,
            "frame_count": frame_count, "static_model_id": shared_static_model_id,
            "block_index_start": frame_index_start,
            "block_index_count": frame_index_count,
        })
        self.frame_archive["frame_count"] = global_start + frame_count
        self.frame_archive["shape"] = [int(self.frame_archive["frame_count"])] + frame_shape

    def _write_binary_frame_index(self, archive: zipfile.ZipFile) -> dict[str, Any]:
        blocks = list(self.frame_archive.get("blocks", []))
        datasets = list(self.frame_archive.get("datasets", []))
        dataset_ordinals = {
            str(item.get("object")): index for index, item in enumerate(datasets)
        }
        raw = pack_frame_index(
            blocks, dataset_ordinals=dataset_ordinals, compress=True
        )
        member_path, digest = self._store_payload(
            archive, encoding="frame-index-v1", data=raw
        )
        self.encoding_stats["frame_index_bytes"] = len(raw)
        return {
            "path": member_path,
            "encoding": "hdxfi-v1",
            "record_count": len(blocks),
            "size_bytes": len(raw),
            "sha256": digest,
        }

    def _write_special_array(
        self, dataset: h5py.Dataset, payload_entry: dict[str, Any], h5file: h5py.File, archive: zipfile.ZipFile,
        source_file_path: Path, path: str, filter_name: str, kind: str
    ) -> None:
        payload_entry.update({"encoding": "array-blocks", "state": "present", "members": []})
        for index, plan in enumerate(iter_axis0_slices(dataset, self.target_chunk_bytes)):
            array = self._read_chunk(dataset, plan, source_file_path, index)
            data, header = encode_array_block_hdxfb(
                array, filter_name=filter_name, clevel=self.zstd_level, kind=kind
            )
            member_path, digest = self._store_payload(archive, encoding="array-block-v1", data=data)
            payload_entry["members"].append({
                "path": member_path, "encoding": "array-block-v1", "start": plan.start, "count": plan.count,
                "chunk_shape": list(array.shape), "size_bytes": len(data), "sha256": digest,
                "transform": "none", "codec": "blosc2-zstd", "filter": filter_name,
                "logical_sha256": header["logical_sha256"],
            })

    def _write_dataset(
        self,
        dataset: h5py.Dataset,
        payload_entry: dict[str, Any],
        h5file: h5py.File,
        archive: zipfile.ZipFile,
        source_file_path: Path,
        path: str,
        object_id: str,
        *,
        allow_frame: bool = True,
        auxiliary_auto: bool = False,
    ) -> None:
        if dataset.shape is None:
            payload_entry["state"] = "null_dataspace"
            return
        if dataset.shape != () and any(dim == 0 for dim in dataset.shape):
            payload_entry["state"] = "empty"
            return
        if allow_frame and self._is_frame_dataset(path, dataset):
            self._write_frame_dataset(
                dataset,
                payload_entry,
                archive,
                source_file_path,
                path,
                object_id,
            )
            return
        if self.profile == "detector-frame-archive" and path in self.calibration_paths:
            if self.calibration_mode == "referenced":
                payload_entry.update({
                    "encoding": "calibration-reference", "state": "external_reference",
                    "calibration_id": self.calibration_manifest["calibration_id"] if self.calibration_manifest else None,
                    "members": [],
                })
                return
            record = self._calibration_record(path)
            if record.get("identity_mode") == "hdf5-filtered-chunks-v1":
                _progress(f"embedding Calibration raw HDF5 chunks: {path}")
                self._write_embedded_calibration_raw_chunks(
                    dataset, payload_entry, archive, path=path
                )
            else:
                self._write_special_array(
                    dataset, payload_entry, h5file, archive, source_file_path, path,
                    "shuffle", "calibration-block"
                )
                self.encoding_stats["embedded_calibration_payload_bytes"] += sum(
                    int(item.get("size_bytes", 0)) for item in payload_entry.get("members", [])
                )
            return
        if (
            auxiliary_auto
            and self.profile == "detector-frame-archive"
            and not dataset.dtype.hasobject
            and dataset.dtype.kind in "biufc"
        ):
            self._write_auxiliary_numeric_auto(
                dataset,
                payload_entry,
                archive,
                source_file_path,
                path,
            )
            return

        if (
            self.profile == "detector-frame-archive"
            and not dataset.dtype.hasobject
            and dataset.dtype.kind in "biufc"
            and dataset.shape != ()
            and ("pixel_mask" in path.lower() or int(np.prod(dataset.shape)) * max(1, dataset.dtype.itemsize) >= 1024 * 1024)
        ):
            filter_name = "none" if "mask" in path.lower() else "shuffle"
            self._write_special_array(
                dataset, payload_entry, h5file, archive, source_file_path, path, filter_name, "array-block"
            )
            return
        for index, plan in enumerate(iter_axis0_slices(dataset, self.target_chunk_bytes)):
            array = self._read_chunk(dataset, plan, source_file_path, index)
            encoding, data = encode_dataset_chunk(array, h5file)
            member_path, digest = self._store_payload(archive, encoding=encoding, data=data)
            payload_entry["members"].append({
                "path": member_path, "encoding": encoding, "start": plan.start, "count": plan.count,
                "chunk_shape": list(array.shape), "size_bytes": len(data), "sha256": digest,
            })
        payload_entry["state"] = "present"

# -----------------------------------------------------------------------------
# Human-readable HDF5 -> HDXF analysis report
# -----------------------------------------------------------------------------

def _analysis_human_bytes(value: int) -> str:
    value_f = float(max(0, int(value)))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value_f < 1024.0 or unit == "TiB":
            return f"{value_f:.2f} {unit}" if unit != "B" else f"{int(value_f)} B"
        value_f /= 1024.0
    return f"{value_f:.2f} TiB"


def _analysis_metadata_category(path: str, logical_bytes: int = 0) -> str | None:
    """Classify likely scientific metadata without treating detector frames as metadata."""
    p = str(path).lower()
    if p.startswith("/entry/data/"):
        return None
    if p.startswith(("/entry/mx/", "/entry/azint/", "/entry/result/")):
        return None
    if "calibration" in p or any(token in p for token in ("pedestal", "flatfield", "flat_field")):
        return "calibration"
    if "mask" in p or "bad_pixel" in p or "badpixel" in p:
        return "pixel_mask"
    if any(token in p for token in (
        "/transformations/", "/module/", "beam_center", "detector_distance",
        "/distance", "pixel_direction", "module_offset", "translation", "rotation",
        "geometry", "orientation",
    )):
        return "geometry"
    if "/beam/" in p or any(token in p for token in ("wavelength", "photon_energy", "beam_energy")):
        return "beam_information"
    if any(token in p for token in (
        "count_time", "frame_time", "exposure", "trigger", "ntrigger", "nimages",
        "acquisition", "frame_count", "number_of_images", "images_per_file",
    )):
        return "acquisition_parameters"
    if "/instrument/detector" in p and any(token in p for token in (
        "serial", "detector_number", "description", "manufacturer", "model", "sensor",
        "pixel_size", "x_pixel_size", "y_pixel_size", "threshold", "firmware", "software_version",
    )):
        return "detector_information"
    if any(token in p for token in (
        "/entry/sample", "/entry/title", "/entry/start_time", "/entry/end_time",
        "/entry/experiment", "/entry/program_name", "/entry/definition",
    )):
        return "experiment_information"
    # Small non-frame, non-result datasets are still useful as "other metadata" candidates.
    if int(logical_bytes) <= 1024 * 1024:
        return "other_metadata"
    return None


def _analysis_dataset_logical_bytes(dataset: h5py.Dataset) -> int:
    if dataset.shape is None:
        return 0
    if dataset.shape == ():
        return max(1, int(dataset.dtype.itemsize))
    try:
        return int(np.prod(dataset.shape, dtype=np.int64)) * max(1, int(dataset.dtype.itemsize))
    except Exception:
        return 0


def _analysis_dataset_content_digest(
    dataset: h5py.Dataset,
    h5file: h5py.File,
    *,
    target_bytes: int = 16 * 1024 * 1024,
) -> tuple[str | None, str, str | None]:
    """Return a stable content digest without risking native HDF5 filter crashes.

    IMPORTANT: filtered/chunked detector datasets are NEVER read through
    ``dataset[...]`` here.  Some Windows HDF5 filter/plugin combinations can
    terminate the interpreter with an access violation before Python can raise
    an Exception.  The conversion path already avoids that for Calibration by
    using ``H5Dread_chunk`` / ``read_direct_chunk``.  The analysis pass must do
    the same *before* attempting a logical read, not as an exception fallback.

    For filtered datasets the digest covers the exact HDF5 storage/filter
    descriptor plus every allocated raw filtered chunk.  This is a strict
    physical-content identity: equal digests mean the stored filtered dataset
    representation is identical.  For unfiltered datasets we hash logical
    values as before.
    """
    digest = hashlib.sha256()

    if dataset.shape is None:
        digest.update(b"null-dataspace")
        return digest.hexdigest(), "logical-values", None
    if dataset.shape != () and any(int(x) == 0 for x in dataset.shape):
        digest.update(b"empty-dataset")
        return digest.hexdigest(), "logical-values", None

    filters = dataset_filter_pipeline(dataset)

    # SAFETY FIRST: never invoke the native filter pipeline during analysis.
    # A Windows access violation is a process-level crash and cannot be caught
    # by the try/except below.  Read exact stored chunks directly instead.
    if filters:
        if not _supports_raw_chunk_access(dataset):
            return (
                None,
                "filtered-dataset-skipped",
                "Filtered dataset was not logically read during analysis because raw chunk access is unavailable; skipped to avoid a native HDF5/plugin crash.",
            )
        try:
            digest.update(strict_json_dumps({
                "identity_mode": "hdf5-filtered-chunks-analysis-v1",
                "storage": _calibration_storage_identity(dataset, h5file),
            }))
            allocated_chunks = 0
            for offset, filter_mask, raw in _iter_raw_hdf5_chunks(dataset):
                chunk_doc = {
                    "offset": list(offset),
                    "filter_mask": int(filter_mask),
                    "size": len(raw),
                }
                digest.update(strict_json_dumps(chunk_doc))
                digest.update(raw)
                allocated_chunks += 1
            digest.update(strict_json_dumps({
                "allocated_chunks": allocated_chunks,
            }))
            return digest.hexdigest(), "raw-filtered-chunks-safe", None
        except Exception as raw_exc:
            return (
                None,
                "unreadable-filtered-dataset",
                f"raw chunk analysis failed: {type(raw_exc).__name__}: {raw_exc}",
            )

    # Unfiltered datasets are safe to read normally.  Keep this non-fatal for
    # ordinary Python/HDF5 errors.
    try:
        for plan in iter_axis0_slices(dataset, target_bytes):
            array = read_dataset_slice(dataset, plan)
            if array.dtype.hasobject:
                digest.update(strict_json_dumps(encode_value(array, h5file)))
            else:
                contiguous = np.ascontiguousarray(array)
                digest.update(memoryview(contiguous).cast("B"))
        return digest.hexdigest(), "logical-values", None
    except Exception as logical_exc:
        return None, "unreadable", f"{type(logical_exc).__name__}: {logical_exc}"


def _analysis_collect_physical_files(
    writer: "HDXFWriter",
    master_path: Path,
) -> list[tuple[str, Path]]:
    by_id: dict[str, Path] = {}
    for normalised_path, source_id in writer._source_path_to_id.items():
        try:
            by_id[str(source_id)] = Path(normalised_path).resolve()
        except Exception:
            continue
    ordered: list[tuple[str, Path]] = []
    for item in writer.source_files:
        source_id = str(item.get("id") or "")
        path = by_id.get(source_id)
        if path is None:
            filename = str(item.get("filename") or "")
            candidate = (master_path.parent / filename).resolve()
            if candidate.exists():
                path = candidate
        if path is not None and path.exists():
            ordered.append((source_id, path))
    if not ordered:
        ordered.append(("source-000001", master_path.resolve()))
    return ordered


def _analysis_collect_metadata(
    writer: "HDXFWriter",
    master_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    dataset_rows: list[dict[str, Any]] = []
    attribute_rows: list[dict[str, Any]] = []
    warnings_out: list[str] = []
    for source_id, source_path in _analysis_collect_physical_files(writer, master_path):
        try:
            with h5r(source_path, "r") as h5file:
                descriptor, _ = capture_local_hdf5_layout(h5file, source_path)
                records = descriptor.get("objects", {})
                for record in records.values():
                    if not isinstance(record, dict):
                        continue
                    path = str(record.get("canonical_path") or "")
                    obj = safe_get(h5file, path) if path else None
                    attrs = record.get("attributes") or {}
                    if isinstance(attrs, dict):
                        category_for_attrs = _analysis_metadata_category(path, 0) or "other_metadata"
                        for attr_name, attr_doc in attrs.items():
                            attr_blob = strict_json_dumps(attr_doc)
                            attribute_rows.append({
                                "source_id": source_id,
                                "filename": source_path.name,
                                "object_path": path,
                                "attribute_name": str(attr_name),
                                "category": category_for_attrs,
                                "identity": hashlib.sha256(attr_blob).hexdigest(),
                                "encoded_bytes": len(attr_blob),
                                "value": attr_doc,
                            })
                    if not isinstance(obj, h5py.Dataset):
                        continue
                    logical_bytes = _analysis_dataset_logical_bytes(obj)
                    category = _analysis_metadata_category(path, logical_bytes)
                    if category is None:
                        continue
                    content_sha, identity_mode, read_error = _analysis_dataset_content_digest(obj, h5file)
                    attr_blob = strict_json_dumps(attrs)
                    schema_blob = strict_json_dumps({
                        "shape": None if obj.shape is None else list(obj.shape),
                        "dtype": hdf5_dtype_info(obj.dtype),
                    })
                    dataset_rows.append({
                        "source_id": source_id,
                        "filename": source_path.name,
                        "path": path,
                        "category": category,
                        "shape": None if obj.shape is None else list(obj.shape),
                        "dtype": str(obj.dtype),
                        "logical_bytes": logical_bytes,
                        "attributes_sha256": hashlib.sha256(attr_blob).hexdigest(),
                        "schema_sha256": hashlib.sha256(schema_blob).hexdigest(),
                        "content_sha256": content_sha,
                        "identity_mode": identity_mode,
                        "read_error": read_error,
                        "attributes": attrs,
                    })
                    if read_error:
                        warnings_out.append(f"{source_path.name}::{path}: {read_error}")
        except Exception as exc:
            warnings_out.append(
                f"Cannot analyse physical file {source_path}: {type(exc).__name__}: {exc}"
            )
    return dataset_rows, attribute_rows, warnings_out


def _analysis_duplicate_dataset_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        content_sha = row.get("content_sha256")
        if not content_sha:
            continue
        key = (
            row.get("path"),
            row.get("schema_sha256"),
            row.get("attributes_sha256"),
            row.get("identity_mode"),
            content_sha,
        )
        groups.setdefault(key, []).append(row)
    result = []
    for key, items in groups.items():
        if len(items) <= 1:
            continue
        per_copy = max(int(item.get("logical_bytes", 0)) for item in items)
        result.append({
            "path": key[0],
            "category": str(items[0].get("category") or "other_metadata"),
            "copies": len(items),
            "redundant_copies": len(items) - 1,
            "logical_bytes_per_copy": per_copy,
            "estimated_redundant_logical_bytes": per_copy * (len(items) - 1),
            "shape": items[0].get("shape"),
            "dtype": items[0].get("dtype"),
            "identity_mode": items[0].get("identity_mode"),
            "content_sha256": items[0].get("content_sha256"),
            "attributes_sha256": items[0].get("attributes_sha256"),
            "files": [str(item.get("filename")) for item in items],
        })
    result.sort(key=lambda item: (
        -int(item["estimated_redundant_logical_bytes"]),
        -int(item["copies"]),
        str(item["path"]),
    ))
    return result


def _analysis_duplicate_attribute_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("object_path")),
            str(row.get("attribute_name")),
            str(row.get("identity")),
        )
        groups.setdefault(key, []).append(row)
    result = []
    for key, items in groups.items():
        if len(items) <= 1:
            continue
        per_copy = max(int(item.get("encoded_bytes", 0)) for item in items)
        result.append({
            "object_path": key[0],
            "attribute_name": key[1],
            "category": str(items[0].get("category") or "other_metadata"),
            "copies": len(items),
            "redundant_copies": len(items) - 1,
            "estimated_redundant_encoded_bytes": per_copy * (len(items) - 1),
            "identity_sha256": key[2],
            "value": items[0].get("value"),
            "files": [str(item.get("filename")) for item in items],
        })
    result.sort(key=lambda item: (-int(item["copies"]), item["object_path"], item["attribute_name"]))
    return result


def _analysis_cross_path_matches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if not row.get("content_sha256"):
            continue
        key = (
            row.get("schema_sha256"),
            row.get("attributes_sha256"),
            row.get("identity_mode"),
            row.get("content_sha256"),
        )
        groups.setdefault(key, []).append(row)
    result = []
    for items in groups.values():
        paths = sorted({str(item.get("path")) for item in items})
        if len(paths) <= 1:
            continue
        result.append({
            "copies": len(items),
            "paths": paths,
            "files_and_paths": [f"{item.get('filename')}::{item.get('path')}" for item in items],
            "content_sha256": items[0].get("content_sha256"),
            "dtype": items[0].get("dtype"),
            "shape": items[0].get("shape"),
        })
    result.sort(key=lambda item: (-int(item["copies"]), item["paths"]))
    return result


def _analysis_write_original_structure(
    stream: Any,
    writer: "HDXFWriter",
    master_path: Path,
) -> tuple[int, int, int, int, int]:
    file_count = group_count = dataset_count = attribute_count = link_count = 0
    for source_id, source_path in _analysis_collect_physical_files(writer, master_path):
        file_count += 1
        stream.write("\n" + "-" * 100 + "\n")
        stream.write(f"PHYSICAL HDF5 FILE: {source_path.name}  [{source_id}]\n")
        stream.write(f"Path: {source_path}\n")
        try:
            stream.write(f"Size: {_analysis_human_bytes(source_path.stat().st_size)} ({source_path.stat().st_size} bytes)\n")
        except Exception:
            pass
        try:
            with h5r(source_path, "r") as h5file:
                descriptor, _ = capture_local_hdf5_layout(h5file, source_path)
                objects = list((descriptor.get("objects") or {}).values())
                objects.sort(key=lambda rec: str(rec.get("canonical_path") or ""))
                for rec in objects:
                    if not isinstance(rec, dict):
                        continue
                    obj_type = str(rec.get("type") or "unknown")
                    path = str(rec.get("canonical_path") or "")
                    attrs = rec.get("attributes") or {}
                    if obj_type == "group":
                        group_count += 1
                        stream.write(f"[GROUP]   {path}\n")
                        creation = rec.get("creation") or {}
                        if creation:
                            stream.write(f"          creation={json.dumps(creation, ensure_ascii=False)}\n")
                    elif obj_type == "dataset":
                        dataset_count += 1
                        storage = rec.get("hdf5_storage") or {}
                        stream.write(f"[DATASET] {path}\n")
                        stream.write(f"          shape={storage.get('shape')}\n")
                        stream.write(f"          dtype={json.dumps(storage.get('dtype'), ensure_ascii=False)}\n")
                        stream.write(f"          chunks={storage.get('chunks')}\n")
                        stream.write(f"          compression={storage.get('compression')} options={storage.get('compression_options')}\n")
                        stream.write(f"          filters={json.dumps(storage.get('filters'), ensure_ascii=False)}\n")
                        stream.write(f"          shuffle={storage.get('shuffle')} fletcher32={storage.get('fletcher32')} scaleoffset={storage.get('scaleoffset')}\n")
                        stream.write(f"          fillvalue={json.dumps(storage.get('fillvalue'), ensure_ascii=False)}\n")
                        stream.write(f"          external_storage={json.dumps(storage.get('external_storage'), ensure_ascii=False)}\n")
                        creation = rec.get("creation") or {}
                        if creation:
                            stream.write(f"          creation={json.dumps(creation, ensure_ascii=False)}\n")
                    if isinstance(attrs, dict):
                        attribute_count += len(attrs)
                        for attr_name, attr_value in attrs.items():
                            stream.write(
                                f"          @{attr_name} = "
                                f"{json.dumps(attr_value, ensure_ascii=False, separators=(',', ':'))}\n"
                            )
                links = descriptor.get("links") or []
                link_count += len(links)
                if links:
                    stream.write("\n  Links:\n")
                    for link in sorted(links, key=lambda item: str(item.get("path") or "")):
                        stream.write(f"    {json.dumps(link, ensure_ascii=False, separators=(',', ':'))}\n")
        except Exception as exc:
            stream.write(f"[ERROR] Could not inspect this file: {type(exc).__name__}: {exc}\n")
    return file_count, group_count, dataset_count, attribute_count, link_count


def _analysis_write_hdxf_structure(
    stream: Any,
    output_path: Path,
    manifest: dict[str, Any],
    writer: "HDXFWriter",
) -> None:
    stream.write("\nHDXF LOGICAL OBJECT GRAPH\n")
    stream.write("-" * 100 + "\n")
    source_names = {
        str(item.get("id")): str(item.get("filename"))
        for item in manifest.get("source", {}).get("files", [])
        if isinstance(item, dict)
    }
    objects = list((manifest.get("objects") or {}).items())
    objects.sort(key=lambda pair: (
        source_names.get(str(pair[1].get("source_file")), ""),
        str(pair[1].get("source_local_path") or pair[1].get("source_path") or ""),
    ))
    for object_id, obj in objects:
        obj_type = str(obj.get("type") or "unknown")
        source_id = str(obj.get("source_file") or "")
        source_name = source_names.get(source_id, source_id)
        source_path = str(obj.get("source_local_path") or obj.get("source_path") or "")
        stream.write(f"[{obj_type.upper()}] object={object_id} source={source_name}::{source_path}\n")
        if obj_type == "dataset":
            payload = obj.get("payload") or {}
            stream.write(
                f"          payload_encoding={payload.get('encoding')} state={payload.get('state')} "
                f"reconstruction_only={obj.get('reconstruction_only', False)}\n"
            )
            members = payload.get("members")
            if isinstance(members, list):
                stream.write(f"          payload_members={len(members)}\n")
                for member in members:
                    stream.write(
                        f"            {json.dumps(member, ensure_ascii=False, separators=(',', ':'))}\n"
                    )
            if payload.get("recipe") is not None:
                stream.write(
                    f"          recipe={json.dumps(payload.get('recipe'), ensure_ascii=False, separators=(',', ':'))}\n"
                )
        attrs = obj.get("attributes") or {}
        if isinstance(attrs, dict) and attrs:
            stream.write(f"          attributes={len(attrs)}\n")
        children = obj.get("children")
        if isinstance(children, list):
            stream.write(f"          children={len(children)}\n")
            for child in children:
                stream.write(f"            {json.dumps(child, ensure_ascii=False, separators=(',', ':'))}\n")

    stream.write("\nHDXF LINKS\n")
    stream.write("-" * 100 + "\n")
    for link in manifest.get("links", []) or []:
        stream.write(json.dumps(link, ensure_ascii=False, separators=(",", ":")) + "\n")

    detector_archive = manifest.get("detector_archive")
    if isinstance(detector_archive, dict):
        stream.write("\nHDXF DETECTOR ARCHIVE SUMMARY\n")
        stream.write("-" * 100 + "\n")
        stream.write(json.dumps(detector_archive, ensure_ascii=False, indent=2) + "\n")

    stream.write("\nHDXF ZIP MEMBER STRUCTURE\n")
    stream.write("-" * 100 + "\n")
    try:
        with zipfile.ZipFile(output_path, "r") as archive:
            for info in archive.infolist():
                method = "STORED" if info.compress_type == zipfile.ZIP_STORED else str(info.compress_type)
                stream.write(
                    f"{info.filename}\n"
                    f"    size={info.file_size} ({_analysis_human_bytes(info.file_size)})\n"
                    f"    compressed_size={info.compress_size} ({_analysis_human_bytes(info.compress_size)})\n"
                    f"    method={method} crc32={info.CRC:08x}\n"
                )
    except Exception as exc:
        stream.write(f"[ERROR] Could not list HDXF ZIP members: {type(exc).__name__}: {exc}\n")




def _analysis_metadata_field_family(category: str, path: str) -> str:
    """Return a concise, path-based field family for the human-readable report.

    This intentionally does not inspect or interpret Dataset values.  The labels
    are based only on the already assigned metadata category and HDF5 path, so
    the report can explain roughly what kinds of fields are present without
    exposing potentially huge arrays or pretending to infer scientific meaning
    from their numeric contents.
    """
    category = str(category or "other_metadata")
    p = str(path).lower()
    if category != "other_metadata":
        return category
    if p.startswith("/entry/detector/"):
        return "per_frame_detector_auxiliary"
    if p.startswith("/entry/image/"):
        return "per_frame_image_diagnostics"
    if p.startswith("/entry/roi/"):
        return "per_frame_roi_statistics"
    if p.startswith("/entry/xfel/"):
        return "per_frame_xfel_event_fields"
    if p.startswith("/entry/instrument/detector/detectorspecific/"):
        return "detector_specific_technical_fields"
    if p.startswith("/entry/instrument/detector/"):
        return "detector_flags_and_technical_fields"
    if p.startswith("/entry/source/"):
        return "source_information"
    if p.startswith("/entry/instrument/"):
        return "instrument_information"
    return "other_metadata"


def _analysis_shape_label(shape: Any) -> str:
    if shape is None:
        return "null-dataspace"
    if shape == []:
        return "scalar"
    return str(shape)


def _analysis_write_metadata_field_summary(
    stream: Any,
    dataset_rows: list[dict[str, Any]],
    attribute_rows: list[dict[str, Any]],
    dataset_dupes: list[dict[str, Any]],
) -> None:
    """Write a value-free, human-readable inventory of metadata field names.

    The detailed report already contains hashes and complete HDF5 structure.
    This section is deliberately concise: each unique Dataset path is shown once
    with its field name, category/family, shape, dtype and how many physical HDF5
    files contain it.  No Dataset values are read or printed here.
    """
    stream.write("\n6. HUMAN-READABLE METADATA FIELD SUMMARY (NO DATA VALUES)\n")
    stream.write("=" * 100 + "\n")
    stream.write(
        "Purpose: show roughly what metadata/auxiliary information exists using field names only.\n"
        "No scalar values or array contents are printed in this section.\n\n"
    )

    # Collapse repeated physical-file occurrences down to one row per HDF5 path.
    by_path: dict[str, list[dict[str, Any]]] = {}
    for row in dataset_rows:
        by_path.setdefault(str(row.get("path") or ""), []).append(row)

    # Strict-duplicate information is useful context, but we keep it compact.
    dupes_by_path: dict[str, list[dict[str, Any]]] = {}
    for item in dataset_dupes:
        dupes_by_path.setdefault(str(item.get("path") or ""), []).append(item)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for path, items in by_path.items():
        if not path:
            continue
        first = items[0]
        category = str(first.get("category") or "other_metadata")
        family = _analysis_metadata_field_family(category, path)
        filenames = sorted({str(item.get("filename") or "") for item in items})
        shapes = sorted({repr(item.get("shape")) for item in items})
        dtypes = sorted({str(item.get("dtype") or "") for item in items})
        duplicate_groups = dupes_by_path.get(path, [])
        largest_identical_group = max(
            (int(item.get("copies", 0)) for item in duplicate_groups),
            default=1,
        )
        field_name = path.rstrip("/").split("/")[-1] or "/"
        grouped.setdefault(family, []).append({
            "field_name": field_name,
            "path": path,
            "category": category,
            "files": filenames,
            "shapes": shapes,
            "dtypes": dtypes,
            "largest_identical_group": largest_identical_group,
        })

    stream.write(f"Unique Dataset field paths: {len(by_path)}\n")
    unique_attr_keys = {
        (str(row.get("object_path") or ""), str(row.get("attribute_name") or ""))
        for row in attribute_rows
    }
    stream.write(f"Unique Attribute field paths: {len(unique_attr_keys)}\n")

    family_order = (
        "detector_information",
        "beam_information",
        "geometry",
        "acquisition_parameters",
        "calibration",
        "pixel_mask",
        "experiment_information",
        "per_frame_detector_auxiliary",
        "per_frame_image_diagnostics",
        "per_frame_roi_statistics",
        "per_frame_xfel_event_fields",
        "detector_specific_technical_fields",
        "detector_flags_and_technical_fields",
        "source_information",
        "instrument_information",
        "other_metadata",
    )
    remaining = sorted(set(grouped) - set(family_order))
    for family in list(family_order) + remaining:
        items = grouped.get(family)
        if not items:
            continue
        stream.write(f"\n[{family}]  {len(items)} unique Dataset field(s)\n")
        stream.write("-" * 100 + "\n")
        for item in sorted(items, key=lambda value: value["path"]):
            shapes = ", ".join(item["shapes"])
            dtypes = ", ".join(item["dtypes"])
            file_count = len(item["files"])
            duplicate_note = ""
            if int(item["largest_identical_group"]) > 1:
                duplicate_note = (
                    f", largest_strict_identical_group={item['largest_identical_group']}"
                )
            stream.write(
                f"  - {item['field_name']}\n"
                f"      path={item['path']}\n"
                f"      shape={shapes}; dtype={dtypes}; present_in_files={file_count}"
                f"{duplicate_note}\n"
            )

    # Attribute names are also metadata fields.  Collapse them by object path + name
    # and deliberately omit their values.
    attributes_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in attribute_rows:
        key = (
            str(row.get("object_path") or ""),
            str(row.get("attribute_name") or ""),
        )
        attributes_by_key.setdefault(key, []).append(row)
    if attributes_by_key:
        stream.write("\n[HDF5 attribute field names]\n")
        stream.write("-" * 100 + "\n")
        for (object_path, attr_name), items in sorted(attributes_by_key.items()):
            files = {str(item.get("filename") or "") for item in items}
            stream.write(
                f"  - {object_path}@{attr_name}  present_in_files={len(files)}\n"
            )

def write_hdf5_hdxf_analysis_log(
    writer: "HDXFWriter",
    master_path: Path,
    output_path: Path,
    manifest: dict[str, Any],
    *,
    log_path: Path,
) -> Path:
    """Generate a human-readable TXT report after a successful conversion."""
    master_path = master_path.resolve()
    output_path = output_path.resolve()
    log_path = log_path.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_rows, attribute_rows, analysis_warnings = _analysis_collect_metadata(writer, master_path)
    dataset_dupes = _analysis_duplicate_dataset_groups(dataset_rows)
    attribute_dupes = _analysis_duplicate_attribute_groups(attribute_rows)
    cross_path = _analysis_cross_path_matches(dataset_rows)

    category_summary: dict[str, dict[str, int]] = {}
    for item in dataset_dupes:
        category = str(item["category"])
        summary = category_summary.setdefault(category, {
            "duplicate_groups": 0,
            "copies": 0,
            "redundant_copies": 0,
            "redundant_bytes": 0,
        })
        summary["duplicate_groups"] += 1
        summary["copies"] += int(item["copies"])
        summary["redundant_copies"] += int(item["redundant_copies"])
        summary["redundant_bytes"] += int(item["estimated_redundant_logical_bytes"])

    source_paths = _analysis_collect_physical_files(writer, master_path)
    original_size = sum(path.stat().st_size for _, path in source_paths if path.exists())
    hdxf_size = output_path.stat().st_size if output_path.exists() else 0

    with log_path.open("w", encoding="utf-8-sig", newline="\n") as stream:
        stream.write("HDF5 -> HDXF CONVERSION ANALYSIS REPORT\n")
        stream.write("=" * 100 + "\n")
        stream.write(f"Generated UTC : {utc_now_iso()}\n")
        stream.write(f"Input master  : {master_path}\n")
        stream.write(f"Output HDXF   : {output_path}\n")
        stream.write(f"HDXF version  : {manifest.get('hdxf', {}).get('version')}\n")
        stream.write(f"Profile       : {manifest.get('hdxf', {}).get('profile')}\n")
        stream.write("\n")

        stream.write("1. HDF5 / HDXF OVERALL COMPARISON\n")
        stream.write("=" * 100 + "\n")
        stream.write(f"Original physical HDF5 files : {len(source_paths)}\n")
        stream.write(f"Original physical size       : {original_size} ({_analysis_human_bytes(original_size)})\n")
        stream.write(f"HDXF archive size            : {hdxf_size} ({_analysis_human_bytes(hdxf_size)})\n")
        if hdxf_size > 0:
            stream.write(f"Physical size ratio HDF5/HDXF: {original_size / hdxf_size:.4f}x\n")
        stream.write(f"HDXF logical objects         : {len(manifest.get('objects') or {})}\n")
        stream.write(f"HDXF links                   : {len(manifest.get('links') or [])}\n")
        stream.write(f"Payload references           : {writer.payload_reference_count}\n")
        stream.write(f"Unique payloads              : {len(writer._written_payloads)}\n")
        stream.write(f"Deduplicated references      : {writer.payload_reference_count - len(writer._written_payloads)}\n")
        stream.write(f"Frame logical bytes          : {writer.encoding_stats.get('logical_frame_bytes', 0)} ({_analysis_human_bytes(int(writer.encoding_stats.get('logical_frame_bytes', 0)))})\n")
        stream.write(f"Frame encoded bytes          : {writer.encoding_stats.get('encoded_frame_bytes', 0)} ({_analysis_human_bytes(int(writer.encoding_stats.get('encoded_frame_bytes', 0)))})\n")

        stream.write("\n2. DUPLICATE METADATA SUMMARY BY TYPE\n")
        stream.write("=" * 100 + "\n")
        stream.write(
            "Type                         duplicate_groups   copies   redundant_copies   estimated_redundant_logical_bytes\n"
        )
        stream.write("-" * 100 + "\n")
        for category, summary in sorted(
            category_summary.items(),
            key=lambda pair: (-pair[1]["redundant_bytes"], pair[0]),
        ):
            stream.write(
                f"{category:<28} {summary['duplicate_groups']:>16} {summary['copies']:>8} "
                f"{summary['redundant_copies']:>18} "
                f"{summary['redundant_bytes']:>14} ({_analysis_human_bytes(summary['redundant_bytes'])})\n"
            )
        if not category_summary:
            stream.write("No strict duplicate metadata groups were found.\n")
        total_groups = len(dataset_dupes)
        total_copies = sum(int(item["copies"]) for item in dataset_dupes)
        total_redundant = sum(int(item["redundant_copies"]) for item in dataset_dupes)
        total_redundant_bytes = sum(int(item["estimated_redundant_logical_bytes"]) for item in dataset_dupes)
        stream.write("-" * 100 + "\n")
        stream.write(
            f"TOTAL strict duplicate groups={total_groups}, copies={total_copies}, "
            f"redundant_copies={total_redundant}, estimated_redundant_logical_bytes="
            f"{total_redundant_bytes} ({_analysis_human_bytes(total_redundant_bytes)})\n"
        )
        stream.write(
            "Definition: same HDF5 path + same shape/dtype + same attributes + same content digest.\n"
        )

        stream.write("\n3. STRICT DUPLICATE METADATA DATASETS - DETAILS\n")
        stream.write("=" * 100 + "\n")
        for index, item in enumerate(dataset_dupes, 1):
            stream.write(f"\n[DUPLICATE {index:04d}]\n")
            stream.write(f"type/category       : {item['category']}\n")
            stream.write(f"path                : {item['path']}\n")
            stream.write(f"copies              : {item['copies']}\n")
            stream.write(f"redundant_copies    : {item['redundant_copies']}\n")
            stream.write(f"logical_bytes/copy  : {item['logical_bytes_per_copy']} ({_analysis_human_bytes(item['logical_bytes_per_copy'])})\n")
            stream.write(f"redundant_bytes_est : {item['estimated_redundant_logical_bytes']} ({_analysis_human_bytes(item['estimated_redundant_logical_bytes'])})\n")
            stream.write(f"shape               : {item['shape']}\n")
            stream.write(f"dtype               : {item['dtype']}\n")
            stream.write(f"identity_mode       : {item['identity_mode']}\n")
            stream.write(f"content_sha256      : {item['content_sha256']}\n")
            stream.write(f"attributes_sha256   : {item['attributes_sha256']}\n")
            stream.write("files:\n")
            for filename in item["files"]:
                stream.write(f"  - {filename}\n")
        if not dataset_dupes:
            stream.write("No strict duplicate metadata datasets were found.\n")

        stream.write("\n4. DUPLICATE HDF5 ATTRIBUTES\n")
        stream.write("=" * 100 + "\n")
        stream.write(f"Strict duplicate attribute groups: {len(attribute_dupes)}\n")
        for index, item in enumerate(attribute_dupes, 1):
            stream.write(f"\n[ATTRIBUTE DUPLICATE {index:04d}]\n")
            stream.write(f"type/category    : {item['category']}\n")
            stream.write(f"object_path      : {item['object_path']}\n")
            stream.write(f"attribute_name   : {item['attribute_name']}\n")
            stream.write(f"copies           : {item['copies']}\n")
            stream.write(f"redundant_copies : {item['redundant_copies']}\n")
            stream.write(f"identity_sha256  : {item['identity_sha256']}\n")
            stream.write(f"value/schema     : {json.dumps(item['value'], ensure_ascii=False, separators=(',', ':'))}\n")
            stream.write("files:\n")
            for filename in item["files"]:
                stream.write(f"  - {filename}\n")

        stream.write("\n5. CONTENT-ONLY CROSS-PATH METADATA MATCHES (NOT INCLUDED IN STRICT TOTALS)\n")
        stream.write("=" * 100 + "\n")
        stream.write(
            "These have identical shape/dtype, attributes and content, but occur under different HDF5 paths.\n"
            "They are reported for manual review and are NOT counted as strict duplicate metadata.\n"
        )
        for index, item in enumerate(cross_path, 1):
            stream.write(f"\n[CROSS-PATH {index:04d}] copies={item['copies']} shape={item['shape']} dtype={item['dtype']}\n")
            stream.write(f"content_sha256={item['content_sha256']}\n")
            stream.write("paths:\n")
            for path in item["paths"]:
                stream.write(f"  - {path}\n")
            stream.write("occurrences:\n")
            for value in item["files_and_paths"]:
                stream.write(f"  - {value}\n")
        if not cross_path:
            stream.write("No cross-path identical metadata candidates found.\n")

        _analysis_write_metadata_field_summary(
            stream, dataset_rows, attribute_rows, dataset_dupes
        )

        stream.write("\n7. METADATA INVENTORY USED FOR DUPLICATE ANALYSIS\n")
        stream.write("=" * 100 + "\n")
        for row in sorted(dataset_rows, key=lambda item: (str(item['category']), str(item['path']), str(item['filename']))):
            stream.write(
                f"{row['category']:<24} {row['filename']}::{row['path']} "
                f"shape={row['shape']} dtype={row['dtype']} logical={row['logical_bytes']} "
                f"digest={row['content_sha256']} mode={row['identity_mode']}"
            )
            if row.get("read_error"):
                stream.write(f" ERROR={row['read_error']}")
            stream.write("\n")

        stream.write("\n8. FULL ORIGINAL HDF5 PHYSICAL STRUCTURE\n")
        stream.write("=" * 100 + "\n")
        counts = _analysis_write_original_structure(stream, writer, master_path)
        stream.write("\nORIGINAL HDF5 STRUCTURE TOTALS\n")
        stream.write("-" * 100 + "\n")
        stream.write(
            f"physical_files={counts[0]}, groups={counts[1]}, datasets={counts[2]}, "
            f"attributes={counts[3]}, links={counts[4]}\n"
        )

        stream.write("\n9. HDF5 OBJECT -> HDXF OBJECT / PAYLOAD MAPPING\n")
        stream.write("=" * 100 + "\n")
        source_names = {
            str(item.get("id")): str(item.get("filename"))
            for item in manifest.get("source", {}).get("files", [])
            if isinstance(item, dict)
        }
        mapping_rows = []
        for object_id, obj in (manifest.get("objects") or {}).items():
            if not isinstance(obj, dict):
                continue
            source_id = str(obj.get("source_file") or "")
            source_name = source_names.get(source_id, source_id)
            source_local_path = str(obj.get("source_local_path") or obj.get("source_path") or "")
            payload = obj.get("payload") or {}
            mapping_rows.append((source_name, source_local_path, object_id, obj, payload))
        mapping_rows.sort(key=lambda item: (item[0], item[1]))
        for source_name, source_local_path, object_id, obj, payload in mapping_rows:
            stream.write(
                f"{source_name}::{source_local_path}\n"
                f"  -> HDXF object={object_id}, type={obj.get('type')}, "
                f"payload_encoding={payload.get('encoding')}, state={payload.get('state')}, "
                f"reconstruction_only={obj.get('reconstruction_only', False)}\n"
            )

        stream.write("\n10. FULL HDXF STRUCTURE\n")
        stream.write("=" * 100 + "\n")
        _analysis_write_hdxf_structure(stream, output_path, manifest, writer)

        stream.write("\n11. ANALYSIS WARNINGS\n")
        stream.write("=" * 100 + "\n")
        if analysis_warnings:
            for warning in analysis_warnings:
                stream.write(f"- {warning}\n")
        else:
            stream.write("None.\n")

    return log_path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert HDF5 detector masters to HDXF v0.4.8 with exact Attribute-schema restoration and conservative automatic auxiliary reconstruction",
    )
    parser.add_argument("input", type=Path, help="Input master file or directory containing masters")
    parser.add_argument("-o", "--output", type=Path, help="Output .hdxf file, or output directory for batch input")
    parser.add_argument(
        "--profile",
        choices=("detector-frame-archive", "generic-hdf5-preservation"),
        default="detector-frame-archive",
        help="HDXF profile (default: detector-frame-archive)",
    )
    parser.add_argument("--chunk-mib", type=float, default=16.0, help="Generic/calibration chunk target MiB")
    parser.add_argument("--block-frames", type=int, default=64, help="Frames per detector block (default: 64)")
    parser.add_argument(
        "--frame-transform", choices=("adaptive", "auto", "raw", "delta"), default="adaptive",
        help=(
            "Frame transform. adaptive selects per-tile lossless modes; auto "
            "compares Raw, full chained Delta, and Adaptive for every frame "
            "block (default: adaptive)"
        ),
    )
    parser.add_argument("--zstd-level", type=int, default=7, choices=range(0, 10), metavar="0-9")
    parser.add_argument("--tile-height", type=int, default=256, help="Adaptive tile height (default: 256)")
    parser.add_argument("--tile-width", type=int, default=256, help="Adaptive tile width (default: 256)")
    parser.add_argument(
        "--sparse-threshold", type=float, default=0.20,
        help="Maximum nonzero temporal-delta density for sparse mode (default: 0.20)",
    )
    parser.add_argument(
        "--disable-jungfrau-split", action="store_true",
        help="Disable the exact 2-bit Gain + 14-bit ADC candidate for 16-bit tiles",
    )
    parser.add_argument(
        "--calibration-mode", choices=("embedded", "referenced", "hybrid"), default="embedded",
        help=("Calibration storage: embedded keeps it inside the experiment HDXF; "
              "referenced uses a reusable external bundle; hybrid does both (default: embedded)"),
    )
    parser.add_argument(
        "--calibration-library", type=Path,
        help="Optional directory for a reusable <calibration-id>.hdxf bundle",
    )
    parser.add_argument(
        "--calibration-read-mode",
        choices=("auto", "logical", "raw-chunks"),
        default="auto",
        help=(
            "How to fingerprint/store Calibration. auto uses raw stored HDF5 "
            "chunks for filtered datasets to avoid native filter crashes; "
            "logical forces decompression (default: auto)"
        ),
    )
    parser.add_argument(
        "--calibration-alias-prefix",
        default="calibration",
        help="Stable human-readable Calibration group prefix (default: calibration)",
    )
    parser.add_argument(
        "--pattern",
        default="*_master.h5",
        help="Master filename pattern when input is a directory (default: *_master.h5)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan an input directory",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="In batch mode, continue converting remaining files after an error",
    )
    parser.add_argument(
        "--frame-dataset", action="append", default=[],
        help="Explicit frame dataset path; repeat for multiple source datasets",
    )
    parser.add_argument(
        "--compression", choices=("store", "deflate"), default="deflate",
        help="ZIP compression for generic profile only; detector profile always uses Stored",
    )
    parser.add_argument("--compression-level", type=int, default=6, choices=range(0, 10), metavar="0-9")
    parser.add_argument("--no-follow-external", action="store_true")
    parser.add_argument("--external-root", type=Path)
    parser.add_argument("--allow-missing-external", action="store_true")
    parser.add_argument("--source-checksum", action="store_true")
    parser.add_argument(
        "--auxiliary-policy",
        choices=("auto", "store"),
        default="auto",
        help=(
            "Exact external auxiliary Dataset policy. auto first omits only "
            "Datasets that can be proven bit-exactly reconstructable from a "
            "documented recipe, then chooses the smallest actual NPY/HDXFB "
            "encoding for the rest. store preserves the previous payload "
            "behaviour (default: auto)."
        ),
    )
    parser.add_argument(
        "--auxiliary-derive-max-mib",
        type=float,
        default=32.0,
        help=(
            "Maximum logical size of one numeric auxiliary Dataset/dependency "
            "considered by automatic derived-recipe inference (default: 32 MiB)."
        ),
    )
    parser.add_argument(
        "--legacy-template-dir",
        type=Path,
        default=DEFAULT_LEGACY_TEMPLATE_DIR,
        help=(
            "Directory containing the original master HDF5 template. "
            "By default the script auto-detects ./legacy_template or ./legacy-template next to this script. "
            "For detector archives the matching template must be byte-identical "
            "to the input master."
        ),
    )
    parser.add_argument(
        "--allow-missing-legacy-template",
        action="store_true",
        help=(
            "Allow conversion without a matching legacy master template. "
            "This disables guaranteed exact-master restoration and is intended "
            "only for non-exact/legacy workflows."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--blosc-threads", type=int, default=1,
        help="Blosc2 worker threads (default: 1 for Windows stability)",
    )
    parser.add_argument(
        "--compute-backend", choices=("cpu", "cpu-vectorized", "cuda", "auto"), default="cpu",
        help=("Temporal-Delta compute backend. cuda uses CuPy; auto falls back "
              "to the low-memory CPU path (default: cpu)"),
    )
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument(
        "--gpu-memory-limit-mib", type=int, default=0,
        help="Optional CuPy device-memory-pool limit in MiB; 0 means unlimited",
    )
    parser.add_argument(
        "--delta-filter", choices=("learn", "auto", "none", "shuffle", "bitshuffle"),
        default="learn",
        help=("Delta Blosc filter policy. learn tests all filters on the first "
              "block of each source Dataset and reuses the winner (default: learn)"),
    )
    parser.add_argument(
        "--gc-every", type=int, default=0,
        help="Force a full Python garbage collection every N frame blocks; 0 disables it",
    )
    parser.add_argument(
        "--keyframe-interval-blocks", type=int, default=4,
        help="Anchor/keyframe interval in Delta blocks (default: 4; 1 disables chaining)",
    )
    parser.add_argument(
        "--delta-stream", choices=("auto", "zstd", "zero-rle-bitpack"), default="auto",
        help="Delta payload encoding policy (default: auto)",
    )
    parser.add_argument(
        "--zero-rle-threshold", type=float, default=0.90,
        help="Minimum zero fraction before testing zero-RLE bit-packing (default: 0.90)",
    )
    parser.add_argument(
        "--global-static-scan", choices=("off", "dataset", "experiment"), default="off",
        help="Exact two-pass static-pixel scan: per Dataset or across the full experiment (default: off)",
    )
    parser.add_argument(
        "--manifest-layout", choices=("compact", "pretty"), default="compact",
        help="manifest.json layout; compact saves archive bytes (default: compact)",
    )
    parser.add_argument(
        "--frame-index-layout", choices=("binary", "json"), default="binary",
        help=("Frame-block index layout. binary removes repeated block JSON from "
              "manifest.json (default: binary)"),
    )
    parser.add_argument(
        "--no-analysis-log",
        action="store_true",
        help="Do not generate the post-conversion HDF5/HDXF analysis TXT report",
    )
    parser.add_argument(
        "--analysis-log-dir",
        type=Path,
        help="Optional directory for <output>_analysis.txt reports (default: next to the HDXF)",
    )
    parser.add_argument(
        "--progress-every", type=int, default=25,
        help="Print progress every N frame blocks (default: 25)",
    )
    return parser


def _discover_master_inputs(
    input_path: Path, *, pattern: str, recursive: bool
) -> list[Path]:
    input_path = input_path.resolve()
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise ConversionError(f"Input does not exist: {input_path}")
    iterator = (
        input_path.rglob(pattern)
        if recursive
        else input_path.glob(pattern)
    )
    masters = sorted(
        (path.resolve() for path in iterator if path.is_file()),
        key=lambda value: os.path.normcase(str(value)),
    )
    if not masters:
        raise ConversionError(
            f"No master files matched {pattern!r} in {input_path}"
        )
    return masters


def _output_for_master(
    master: Path,
    *,
    input_was_directory: bool,
    output_argument: Path | None,
) -> Path:
    if not input_was_directory:
        return output_argument or master.with_suffix(".hdxf")
    output_dir = (
        output_argument.resolve()
        if output_argument is not None
        else (master.parent / "hdxf-output").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / master.with_suffix(".hdxf").name


def _print_conversion_summary(
    output: Path,
    manifest: dict[str, Any],
    *,
    encoding_stats: dict[str, Any],
    payload_references: int,
    unique_payloads: int,
) -> None:
    dataset_count = sum(
        1 for obj in manifest["objects"].values() if obj["type"] == "dataset"
    )
    group_count = sum(
        1 for obj in manifest["objects"].values() if obj["type"] == "group"
    )
    print(f"created: {output.resolve()}")
    print(f"profile: {manifest['hdxf']['profile']}")
    print(
        f"groups: {group_count}, datasets: {dataset_count}, "
        f"links: {len(manifest['links'])}"
    )
    print(
        f"source files: {len(manifest['source']['files'])}, "
        f"payload references: {payload_references}, "
        f"unique payloads: {unique_payloads}, "
        f"deduplicated references: {payload_references - unique_payloads}"
    )
    legacy = manifest.get("legacy_hdf5")
    if isinstance(legacy, dict):
        template = legacy.get("master_template") or {}
        coverage = legacy.get("coverage") or {}
        print(
            "legacy HDF5 restore: "
            f"template={template.get('filename')}, "
            f"external_files={len(legacy.get('external_files') or [])}, "
            f"all_external_datasets_archived="
            f"{coverage.get('all_external_datasets_archived')}"
        )
        print(
            "auxiliary auto: "
            f"derived={int(encoding_stats.get('auxiliary_derived_datasets', 0))}, "
            f"derived_logical="
            f"{int(encoding_stats.get('auxiliary_derived_logical_bytes', 0)) / (1024 * 1024):.2f} MiB, "
            f"hdxfb_members={int(encoding_stats.get('auxiliary_auto_hdxfb_members', 0))}, "
            f"npy_members={int(encoding_stats.get('auxiliary_auto_npy_members', 0))}, "
            f"encoded="
            f"{int(encoding_stats.get('auxiliary_auto_encoded_bytes', 0)) / (1024 * 1024):.2f} MiB"
        )
    if manifest["hdxf"]["profile"] == "detector-frame-archive":
        frames = manifest["detector_archive"]["frames"]
        stats = encoding_stats
        calibration = manifest["detector_archive"]["calibration"] or {}
        print(
            f"frames: {frames['frame_count']}, "
            f"blocks: {stats['frame_blocks']} "
            f"(raw={stats['raw_blocks']}, delta={stats['delta_blocks']}, "
            f"adaptive={stats.get('adaptive_blocks', 0)})"
        )
        logical_bytes = int(stats.get("logical_frame_bytes", 0))
        encoded_bytes = int(stats.get("encoded_frame_bytes", 0))
        if logical_bytes and encoded_bytes:
            print(
                f"frame payload: {encoded_bytes / (1024 * 1024):.2f} MiB, "
                f"logical: {logical_bytes / (1024 * 1024):.2f} MiB, "
                f"ratio: {logical_bytes / encoded_bytes:.2f}x"
            )
        timings = stats.get("timings") or {}
        if timings:
            total_timed = sum(float(v) for v in timings.values())
            print(
                "timings: "
                + ", ".join(f"{key}={float(value):.1f}s" for key, value in timings.items())
                + f", accounted={total_timed:.1f}s"
            )
        print(
            f"compute backend: {stats.get('compute_backend', {}).get('resolved')}, "
            f"delta filter policy: {stats.get('delta_filter_policy')}"
        )
        if int(stats.get("delta_blocks", 0)):
            print(
                "delta chain: "
                f"anchors={stats.get('anchor_blocks', 0)}, "
                f"continuations={stats.get('continuation_blocks', 0)}, "
                f"interval={stats.get('keyframe_interval_blocks', 1)} blocks"
            )
            print(
                "delta payload breakdown: "
                f"keyframes={int(stats.get('keyframe_payload_bytes', 0)) / (1024 * 1024):.2f} MiB, "
                f"deltas={int(stats.get('delta_payload_bytes', 0)) / (1024 * 1024):.2f} MiB, "
                f"zstd_blocks={stats.get('delta_zstd_blocks', 0)}, "
                f"zero_rle_blocks={stats.get('delta_zero_rle_blocks', 0)}"
            )
        print(
            "calibration: "
            f"{calibration.get('mode')} "
            f"{calibration.get('calibration_alias')} "
            f"{calibration.get('calibration_id')}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.chunk_mib <= 0:
        print(
            "error: --chunk-mib must be greater than zero",
            file=sys.stderr,
        )
        return 2
    if args.block_frames <= 0:
        print(
            "error: --block-frames must be greater than zero",
            file=sys.stderr,
        )
        return 2
    if args.tile_height <= 0 or args.tile_width <= 0:
        print("error: --tile-height and --tile-width must be positive", file=sys.stderr)
        return 2
    if not (0.0 <= args.sparse_threshold <= 1.0):
        print("error: --sparse-threshold must be between 0 and 1", file=sys.stderr)
        return 2
    if args.blosc_threads <= 0:
        print("error: --blosc-threads must be greater than zero", file=sys.stderr)
        return 2
    if args.progress_every <= 0:
        print("error: --progress-every must be greater than zero", file=sys.stderr)
        return 2
    if args.cuda_device < 0:
        print("error: --cuda-device must be non-negative", file=sys.stderr)
        return 2
    if args.gpu_memory_limit_mib < 0:
        print("error: --gpu-memory-limit-mib must be non-negative", file=sys.stderr)
        return 2
    if args.gc_every < 0:
        print("error: --gc-every must be non-negative", file=sys.stderr)
        return 2
    if args.keyframe_interval_blocks <= 0:
        print("error: --keyframe-interval-blocks must be positive", file=sys.stderr)
        return 2
    if args.auxiliary_derive_max_mib < 0:
        print(
            "error: --auxiliary-derive-max-mib must be non-negative",
            file=sys.stderr,
        )
        return 2
    if args.calibration_mode == "referenced" and args.calibration_library is None:
        print("error: --calibration-mode referenced requires --calibration-library", file=sys.stderr)
        return 2
    if not (0.0 <= args.zero_rle_threshold <= 1.0):
        print("error: --zero-rle-threshold must be between 0 and 1", file=sys.stderr)
        return 2
    if args.global_static_scan == "experiment" and args.frame_index_layout == "binary":
        print(
            "error: experiment-shared static models require --frame-index-layout json; "
            "the 0.4.4 binary index intentionally targets the smaller non-shared path",
            file=sys.stderr,
        )
        return 2
    try:
        compute_info = configure_compute_backend(
            args.compute_backend,
            cuda_device=args.cuda_device,
            gpu_memory_limit_mib=args.gpu_memory_limit_mib,
        )
        _progress(
            f"Delta compute backend: {compute_info.get('resolved')}"
            + (f" ({compute_info.get('device_name')})" if compute_info.get('device_name') else "")
        )
    except Exception as exc:
        print(f"error: cannot initialise compute backend: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if BLOSC2_AVAILABLE and hasattr(blosc2, "set_nthreads"):
        try:
            blosc2.set_nthreads(int(args.blosc_threads))
            _progress(f"Blosc2 threads: {args.blosc_threads}")
        except Exception as exc:
            print(f"[WARN] cannot set Blosc2 threads: {exc}", file=sys.stderr, flush=True)

    try:
        masters = _discover_master_inputs(
            args.input,
            pattern=args.pattern,
            recursive=args.recursive,
        )
    except Exception as exc:
        print(
            f"input discovery failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    input_was_directory = args.input.resolve().is_dir()
    if input_was_directory and args.output is not None:
        # In batch mode -o is always interpreted as a directory.
        if args.output.exists() and not args.output.is_dir():
            print(
                "error: batch --output must be a directory",
                file=sys.stderr,
            )
            return 2
        args.output.mkdir(parents=True, exist_ok=True)

    if not HDF5PLUGIN_AVAILABLE:
        print(
            "[WARN] hdf5plugin is not installed; detector HDF5 "
            "filters may be unreadable.",
            file=sys.stderr,
        )

    failures = 0
    for file_index, master in enumerate(masters, 1):
        output = _output_for_master(
            master,
            input_was_directory=input_was_directory,
            output_argument=args.output,
        )
        if len(masters) > 1:
            print(
                f"\n[CONVERT {file_index}/{len(masters)}] {master}"
            )

        try:
            writer = HDXFWriter(
                output,
                profile=args.profile,
                compression=args.compression,
                compression_level=args.compression_level,
                target_chunk_bytes=max(
                    1, int(args.chunk_mib * 1024 * 1024)
                ),
                overwrite=args.overwrite,
                source_checksum=args.source_checksum,
                follow_external=not args.no_follow_external,
                external_root=args.external_root,
                allow_missing_external=args.allow_missing_external,
                block_frames=args.block_frames,
                frame_transform=args.frame_transform,
                tile_height=args.tile_height,
                tile_width=args.tile_width,
                sparse_threshold=args.sparse_threshold,
                enable_jungfrau_split=not args.disable_jungfrau_split,
                zstd_level=args.zstd_level,
                calibration_mode=args.calibration_mode,
                calibration_library=args.calibration_library,
                calibration_alias_prefix=(
                    args.calibration_alias_prefix
                ),
                calibration_read_mode=args.calibration_read_mode,
                frame_datasets=set(args.frame_dataset),
                progress_every=args.progress_every,
                compute_info=compute_info,
                cuda_device=args.cuda_device,
                delta_filter_policy=args.delta_filter,
                gc_every=args.gc_every,
                keyframe_interval_blocks=args.keyframe_interval_blocks,
                delta_stream=args.delta_stream,
                zero_rle_threshold=args.zero_rle_threshold,
                global_static_scan=args.global_static_scan,
                manifest_layout=args.manifest_layout,
                frame_index_layout=args.frame_index_layout,
                legacy_template_dir=args.legacy_template_dir,
                require_legacy_template=(
                    args.profile == "detector-frame-archive"
                    and not args.allow_missing_legacy_template
                ),
                auxiliary_policy=args.auxiliary_policy,
                auxiliary_derive_max_mib=(
                    args.auxiliary_derive_max_mib
                ),
            )
            manifest = writer.convert(master)
            _print_conversion_summary(
                output, manifest,
                encoding_stats=writer.encoding_stats,
                payload_references=writer.payload_reference_count,
                unique_payloads=len(writer._written_payloads),
            )
            if not args.no_analysis_log:
                try:
                    analysis_dir = (
                        args.analysis_log_dir.resolve()
                        if args.analysis_log_dir is not None
                        else output.resolve().parent
                    )
                    analysis_dir.mkdir(parents=True, exist_ok=True)
                    analysis_path = analysis_dir / f"{output.stem}_analysis.txt"
                    # Analysis must use raw HDF5 chunk access for filtered datasets.
                    # Do not change it back to dataset[...] reads: on Windows some
                    # detector filter/plugin combinations can cause a fatal access
                    # violation that bypasses Python exception handling.
                    written_analysis = write_hdf5_hdxf_analysis_log(
                        writer,
                        master,
                        output,
                        manifest,
                        log_path=analysis_path,
                    )
                    print(f"analysis report: {written_analysis}")
                except Exception as analysis_exc:
                    print(
                        f"[WARN] analysis report failed for {master.name}: "
                        f"{type(analysis_exc).__name__}: {analysis_exc}",
                        file=sys.stderr,
                        flush=True,
                    )
        except Exception as exc:
            failures += 1
            print(
                f"conversion failed for {master}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            if not args.continue_on_error:
                return 1

    if input_was_directory:
        print(
            f"\nbatch completed: total={len(masters)}, "
            f"success={len(masters) - failures}, "
            f"failed={failures}"
        )
        if args.calibration_library:
            library = args.calibration_library.resolve()
            print(
                f"calibration index: "
                f"{library / CALIBRATION_INDEX_NAME}"
            )
            print(
                f"calibration report: "
                f"{library / CALIBRATION_REPORT_NAME}"
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())