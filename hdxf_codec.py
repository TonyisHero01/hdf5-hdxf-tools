#!/usr/bin/env python3
"""Shared HDXFB codec for HDXF detector-frame archives.

HDXFB v1 supports whole-block Raw and temporal Delta.
HDXFB v2 adds lossless tile-adaptive encoding.
HDXFB v3 adds chained temporal Delta blocks, static-split anchor
keyframes, and zero-RLE + ZigZag bit-packed Delta streams.
HDXFB v4 adds externally stored shared static models for backward compatibility.

The codec always verifies the logical SHA-256 when decoding unless disabled.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import warnings
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

try:
    import blosc2  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("hdxf_codec requires blosc2") from exc

HDXFB_MAGIC = b"HDXFB\x00\x01"
HDXFB_FORMAT = "hdxfb"


class HDXFBError(RuntimeError):
    """Raised for malformed, unsupported, or checksum-invalid HDXFB data."""


def strict_json_dumps(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def dtype_descriptor(dtype: np.dtype[Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        descr = np.lib.format.dtype_to_descr(np.dtype(dtype))
    return json.loads(json.dumps(descr, ensure_ascii=False))


def descriptor_to_dtype(value: Any) -> np.dtype[Any]:
    if isinstance(value, str):
        return np.dtype(value)
    if isinstance(value, list):
        fields = []
        for field in value:
            if not isinstance(field, list):
                raise TypeError(f"invalid dtype field: {field!r}")
            if len(field) == 2:
                name, field_dtype = field
                fields.append((name, descriptor_to_dtype(field_dtype)))
            elif len(field) == 3:
                name, field_dtype, shape = field
                fields.append((name, descriptor_to_dtype(field_dtype), tuple(shape)))
            else:
                raise TypeError(f"invalid dtype field length: {field!r}")
        return np.dtype(fields)
    raise TypeError(f"unsupported dtype descriptor: {value!r}")


def logical_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _blosc_filter(name: str):
    mapping = {
        "none": blosc2.Filter.NOFILTER,
        "shuffle": blosc2.Filter.SHUFFLE,
        "bitshuffle": blosc2.Filter.BITSHUFFLE,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise HDXFBError(f"unsupported Blosc2 filter: {name}") from exc


def compress_array(array: np.ndarray, *, filter_name: str, clevel: int) -> bytes:
    contiguous = np.ascontiguousarray(array)
    view = memoryview(contiguous).cast("B")
    kwargs = dict(
        codec=blosc2.Codec.ZSTD,
        clevel=int(clevel),
        filters=[blosc2.Filter.NOFILTER] * 5 + [_blosc_filter(filter_name)],
        typesize=max(1, int(contiguous.dtype.itemsize)),
    )
    try:
        return bytes(blosc2.compress2(view, **kwargs))
    except (TypeError, ValueError):
        return bytes(blosc2.compress2(view.tobytes(), **kwargs))


def _compress_best(array: np.ndarray, *, filters: Iterable[str], clevel: int) -> tuple[str, bytes]:
    best_name: str | None = None
    best_data: bytes | None = None
    for name in filters:
        candidate = compress_array(array, filter_name=name, clevel=clevel)
        if best_data is None or len(candidate) < len(best_data):
            best_name = name
            best_data = candidate
        else:
            del candidate
    assert best_name is not None and best_data is not None
    return best_name, best_data


def _segment_doc(name: str, array: np.ndarray, *, filter_name: str, compressed: bytes, semantic: str | None = None) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "name": name,
        "codec": "blosc2-zstd",
        "filter": filter_name,
        "dtype": dtype_descriptor(array.dtype),
        "shape": list(array.shape),
        "uncompressed_size": int(array.nbytes),
        "compressed_size": len(compressed),
    }
    if semantic:
        doc["semantic"] = semantic
    return doc


def pack_hdxfb(header: dict[str, Any], segments: list[bytes]) -> bytes:
    docs = header.setdefault("segments", [])
    if len(docs) != len(segments):
        raise HDXFBError("HDXFB segment metadata/count mismatch")
    cursor = 0
    for doc, segment in zip(docs, segments):
        doc["offset"] = cursor
        doc["compressed_size"] = len(segment)
        cursor += len(segment)
    header_bytes = strict_json_dumps(header)
    return HDXFB_MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + b"".join(segments)


def parse_hdxfb(raw: bytes) -> tuple[dict[str, Any], bytes]:
    if not raw.startswith(HDXFB_MAGIC):
        raise HDXFBError("invalid HDXFB magic")
    if len(raw) < len(HDXFB_MAGIC) + 4:
        raise HDXFBError("truncated HDXFB header")
    size = struct.unpack_from("<I", raw, len(HDXFB_MAGIC))[0]
    start = len(HDXFB_MAGIC) + 4
    end = start + size
    if end > len(raw):
        raise HDXFBError("HDXFB header exceeds payload")
    try:
        header = json.loads(raw[start:end])
    except Exception as exc:
        raise HDXFBError(f"invalid HDXFB JSON: {exc}") from exc
    if header.get("format") != HDXFB_FORMAT:
        raise HDXFBError(f"invalid HDXFB format: {header.get('format')!r}")
    if header.get("version") not in (1, 2, 3, 4):
        raise HDXFBError(f"unsupported HDXFB version: {header.get('version')!r}")
    return header, raw[end:]


def _read_segments(header: dict[str, Any], body: bytes) -> dict[str, np.ndarray]:
    segments: dict[str, np.ndarray] = {}
    occupied: list[tuple[int, int]] = []
    for doc in header.get("segments", []):
        name = doc.get("name")
        offset = doc.get("offset")
        compressed_size = doc.get("compressed_size")
        uncompressed_size = doc.get("uncompressed_size")
        if not isinstance(name, str):
            raise HDXFBError("HDXFB segment has no string name")
        if not all(isinstance(x, int) and x >= 0 for x in (offset, compressed_size, uncompressed_size)):
            raise HDXFBError(f"invalid segment sizes for {name}")
        end = offset + compressed_size
        if end > len(body):
            raise HDXFBError(f"segment {name} exceeds HDXFB body")
        for old_start, old_end in occupied:
            if max(offset, old_start) < min(end, old_end):
                raise HDXFBError(f"segment {name} overlaps another segment")
        occupied.append((offset, end))
        try:
            decompressed = bytes(blosc2.decompress2(body[offset:end]))
        except Exception as exc:
            raise HDXFBError(f"cannot decompress segment {name}: {exc}") from exc
        if len(decompressed) != uncompressed_size:
            raise HDXFBError(f"segment {name} uncompressed-size mismatch")
        dtype = descriptor_to_dtype(doc.get("dtype"))
        shape = tuple(int(x) for x in doc.get("shape", []))
        expected = int(math.prod(shape)) if shape else 1
        array = np.frombuffer(decompressed, dtype=dtype)
        if array.size != expected:
            raise HDXFBError(f"segment {name} element-count mismatch")
        segments[name] = array.reshape(shape)
    return segments


def _smallest_signed_dtype(minimum: int, maximum: int) -> np.dtype[Any]:
    for value in ("int8", "int16", "int32", "int64"):
        dtype = np.dtype(value)
        info = np.iinfo(dtype)
        if minimum >= int(info.min) and maximum <= int(info.max):
            return dtype
    raise HDXFBError(f"range cannot be represented: {minimum}..{maximum}")


def _smallest_unsigned_dtype(maximum: int) -> np.dtype[Any]:
    for value in ("uint8", "uint16", "uint32", "uint64"):
        dtype = np.dtype(value)
        if maximum <= int(np.iinfo(dtype).max):
            return dtype
    raise HDXFBError(f"unsigned range cannot be represented: 0..{maximum}")


def _zigzag_encode(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.int64).reshape(-1)
    encoded = np.empty(source.shape, dtype=np.uint64)
    positive = source >= 0
    encoded[positive] = source[positive].astype(np.uint64) * np.uint64(2)
    negative_values = source[~positive]
    encoded[~positive] = ((-negative_values - 1).astype(np.uint64) * np.uint64(2)) + np.uint64(1)
    return encoded


def _zigzag_decode(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.uint64)
    decoded = (source >> np.uint64(1)).astype(np.int64)
    negative = (source & np.uint64(1)) != 0
    decoded[negative] = -decoded[negative] - 1
    return decoded


def pack_unsigned(values: np.ndarray, bit_width: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.uint64).reshape(-1)
    bit_width = int(bit_width)
    if bit_width < 0 or bit_width > 64:
        raise HDXFBError(f"invalid bit width: {bit_width}")
    if bit_width == 0 or source.size == 0:
        return np.empty((0,), dtype=np.uint8)
    shifts = np.arange(bit_width, dtype=np.uint64)
    bits = ((source[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.uint8)
    return np.packbits(bits.reshape(-1), bitorder="little")


def unpack_unsigned(packed: np.ndarray, *, count: int, bit_width: int) -> np.ndarray:
    count = int(count)
    bit_width = int(bit_width)
    if count < 0 or bit_width < 0 or bit_width > 64:
        raise HDXFBError("invalid bit-packed stream dimensions")
    if count == 0:
        return np.empty((0,), dtype=np.uint64)
    if bit_width == 0:
        return np.zeros((count,), dtype=np.uint64)
    required_bits = count * bit_width
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8), bitorder="little")
    if bits.size < required_bits:
        raise HDXFBError("truncated bit-packed stream")
    matrix = bits[:required_bits].reshape(count, bit_width).astype(np.uint64)
    shifts = np.arange(bit_width, dtype=np.uint64)
    return np.sum(matrix << shifts[None, :], axis=1, dtype=np.uint64)


@dataclass
class _Candidate:
    mode: str
    tile_doc: dict[str, Any]
    segment_docs: list[dict[str, Any]]
    segment_data: list[bytes]

    @property
    def estimated_size(self) -> int:
        return (
            sum(len(item) for item in self.segment_data)
            + len(strict_json_dumps(self.tile_doc))
            + sum(len(strict_json_dumps(doc)) for doc in self.segment_docs)
        )


def _make_segment(name: str, array: np.ndarray, *, filters: Iterable[str], clevel: int, semantic: str | None = None) -> tuple[dict[str, Any], bytes]:
    filter_name, compressed = _compress_best(array, filters=filters, clevel=clevel)
    return _segment_doc(name, array, filter_name=filter_name, compressed=compressed, semantic=semantic), compressed


def _candidate_raw(tile: np.ndarray, prefix: str, clevel: int) -> _Candidate:
    doc, data = _make_segment(f"{prefix}_raw", tile, filters=("bitshuffle", "shuffle", "none"), clevel=clevel, semantic="raw-tile-stack")
    return _Candidate("raw", {"mode": "raw", "segments": {"raw": doc["name"]}}, [doc], [data])


def _candidate_repeat(tile: np.ndarray, prefix: str, clevel: int) -> _Candidate | None:
    if tile.shape[0] < 2 or not np.array_equal(tile[1:], np.broadcast_to(tile[0], tile[1:].shape)):
        return None
    key = np.ascontiguousarray(tile[0])
    doc, data = _make_segment(f"{prefix}_key", key, filters=("bitshuffle", "shuffle", "none"), clevel=clevel, semantic="repeated-tile-keyframe")
    return _Candidate("repeat", {"mode": "repeat", "segments": {"keyframe": doc["name"]}}, [doc], [data])


def _candidate_static_mask(tile: np.ndarray, prefix: str, clevel: int) -> _Candidate | None:
    if tile.shape[0] < 2:
        return None
    flat = tile.reshape(tile.shape[0], -1)
    mask = np.all(flat == flat[0:1], axis=0)
    static_count = int(np.count_nonzero(mask))
    dynamic_count = int(mask.size - static_count)
    if static_count == 0 or dynamic_count == 0:
        return None
    mask_bytes = np.packbits(mask.astype(np.uint8), bitorder="little")
    static_values = np.ascontiguousarray(flat[0, mask])
    dynamic_values = np.ascontiguousarray(flat[:, ~mask])
    docs: list[dict[str, Any]] = []
    data: list[bytes] = []
    for name, array, filters, semantic in (
        (f"{prefix}_smask", mask_bytes, ("none",), "static-pixel-mask-packbits"),
        (f"{prefix}_svalues", static_values, ("bitshuffle", "shuffle", "none"), "static-pixel-values"),
        (f"{prefix}_dynamic", dynamic_values, ("bitshuffle", "shuffle", "none"), "dynamic-pixel-values"),
    ):
        doc, payload = _make_segment(name, array, filters=filters, clevel=clevel, semantic=semantic)
        docs.append(doc); data.append(payload)
    return _Candidate(
        "static-mask",
        {
            "mode": "static-mask",
            "pixel_count": int(mask.size),
            "static_count": static_count,
            "dynamic_count": dynamic_count,
            "segments": {
                "mask": docs[0]["name"],
                "static_values": docs[1]["name"],
                "dynamic_values": docs[2]["name"],
            },
        },
        docs,
        data,
    )


def _temporal_deltas(tile: np.ndarray) -> np.ndarray:
    return tile[1:].astype(np.int64) - tile[:-1].astype(np.int64)


def _candidate_sparse_delta(tile: np.ndarray, prefix: str, clevel: int, sparse_threshold: float) -> _Candidate | None:
    if tile.shape[0] < 2 or tile.dtype.kind not in "iu" or tile.dtype.itemsize > 4:
        return None
    deltas = _temporal_deltas(tile)
    total = int(deltas.size)
    positions = np.flatnonzero(deltas.reshape(-1))
    nonzero = int(positions.size)
    if nonzero == 0:
        return _candidate_repeat(tile, prefix, clevel)
    density = nonzero / total
    if density > float(sparse_threshold):
        return None
    values64 = deltas.reshape(-1)[positions]
    value_dtype = _smallest_signed_dtype(int(values64.min()), int(values64.max()))
    values = np.ascontiguousarray(values64.astype(value_dtype))
    gaps64 = np.empty(positions.shape, dtype=np.uint64)
    gaps64[0] = np.uint64(positions[0])
    if nonzero > 1:
        gaps64[1:] = np.diff(positions).astype(np.uint64)
    gap_dtype = _smallest_unsigned_dtype(int(gaps64.max(initial=0)))
    gaps = np.ascontiguousarray(gaps64.astype(gap_dtype))
    key = np.ascontiguousarray(tile[0])
    docs: list[dict[str, Any]] = []
    payloads: list[bytes] = []
    for name, array, filters, semantic in (
        (f"{prefix}_key", key, ("bitshuffle", "shuffle", "none"), "sparse-delta-keyframe"),
        (f"{prefix}_gaps", gaps, ("shuffle", "none", "bitshuffle"), "sparse-delta-position-gaps"),
        (f"{prefix}_values", values, ("bitshuffle", "shuffle", "none"), "sparse-delta-values"),
    ):
        doc, payload = _make_segment(name, array, filters=filters, clevel=clevel, semantic=semantic)
        docs.append(doc); payloads.append(payload)
    return _Candidate(
        "sparse-delta",
        {
            "mode": "sparse-delta",
            "delta_shape": list(deltas.shape),
            "delta_element_count": total,
            "nonzero_count": nonzero,
            "density": density,
            "position_encoding": "delta-gaps",
            "delta_dtype": dtype_descriptor(value_dtype),
            "position_dtype": dtype_descriptor(gap_dtype),
            "segments": {
                "keyframe": docs[0]["name"],
                "position_gaps": docs[1]["name"],
                "values": docs[2]["name"],
            },
        },
        docs,
        payloads,
    )


def _candidate_delta_bitpack(tile: np.ndarray, prefix: str, clevel: int) -> _Candidate | None:
    if tile.shape[0] < 2 or tile.dtype.kind not in "iu" or tile.dtype.itemsize > 4:
        return None
    deltas = _temporal_deltas(tile)
    zigzag = _zigzag_encode(deltas)
    maximum = int(zigzag.max(initial=0))
    bit_width = maximum.bit_length()
    if bit_width == 0:
        return _candidate_repeat(tile, prefix, clevel)
    packed = pack_unsigned(zigzag, bit_width)
    key = np.ascontiguousarray(tile[0])
    key_doc, key_data = _make_segment(f"{prefix}_key", key, filters=("bitshuffle", "shuffle", "none"), clevel=clevel, semantic="delta-bitpack-keyframe")
    packed_doc, packed_data = _make_segment(f"{prefix}_packed", packed, filters=("none",), clevel=clevel, semantic="zigzag-lsb-bitpack")
    return _Candidate(
        "delta-bitpack",
        {
            "mode": "delta-bitpack",
            "delta_shape": list(deltas.shape),
            "delta_element_count": int(deltas.size),
            "bit_width": bit_width,
            "packing": "zigzag-lsb-first",
            "segments": {"keyframe": key_doc["name"], "packed": packed_doc["name"]},
        },
        [key_doc, packed_doc],
        [key_data, packed_data],
    )


def _candidate_jungfrau_split(tile: np.ndarray, prefix: str, clevel: int) -> _Candidate | None:
    if tile.dtype.kind not in "iu" or tile.dtype.itemsize != 2 or not np.little_endian:
        return None
    if tile.dtype.byteorder not in ("=", "<", "|"):
        return None
    raw_u16 = np.ascontiguousarray(tile).view(np.uint16)
    gain = (raw_u16 >> np.uint16(14)).astype(np.uint64)
    adc = np.ascontiguousarray(raw_u16 & np.uint16(0x3FFF))
    gain_packed = pack_unsigned(gain, 2)
    gain_doc, gain_data = _make_segment(f"{prefix}_gain", gain_packed, filters=("none",), clevel=clevel, semantic="jungfrau-gain-2bit")
    adc_doc, adc_data = _make_segment(f"{prefix}_adc", adc, filters=("bitshuffle", "shuffle", "none"), clevel=clevel, semantic="jungfrau-adc-14bit-in-u16")
    return _Candidate(
        "jungfrau-split",
        {
            "mode": "jungfrau-split",
            "element_count": int(tile.size),
            "gain_bit_width": 2,
            "adc_mask": 16383,
            "segments": {"gain": gain_doc["name"], "adc": adc_doc["name"]},
        },
        [gain_doc, adc_doc],
        [gain_data, adc_data],
    )


def encode_adaptive_frame_block(
    block: np.ndarray,
    *,
    clevel: int,
    tile_height: int = 256,
    tile_width: int = 256,
    sparse_threshold: float = 0.20,
    enable_jungfrau_split: bool = True,
) -> tuple[bytes, dict[str, Any], dict[str, int]]:
    contiguous = np.ascontiguousarray(block)
    if contiguous.ndim != 3:
        raise HDXFBError(f"tile-adaptive encoding requires F×H×W, got {contiguous.shape}")
    if contiguous.dtype.hasobject:
        raise HDXFBError("object arrays cannot use HDXFB")
    if tile_height <= 0 or tile_width <= 0:
        raise HDXFBError("tile dimensions must be positive")
    if not (0.0 <= sparse_threshold <= 1.0):
        raise HDXFBError("sparse threshold must be between 0 and 1")

    frames, height, width = contiguous.shape
    header: dict[str, Any] = {
        "format": HDXFB_FORMAT,
        "version": 2,
        "kind": "frame-block",
        "transform": "tile-adaptive-v1",
        "logical_dtype": dtype_descriptor(contiguous.dtype),
        "logical_shape": list(contiguous.shape),
        "logical_sha256": logical_sha256(contiguous),
        "tile_shape": [int(tile_height), int(tile_width)],
        "sparse_threshold": float(sparse_threshold),
        "tiles": [],
        "segments": [],
    }
    all_segment_data: list[bytes] = []
    mode_counts: dict[str, int] = {}
    mode_payload_bytes: dict[str, int] = {}
    candidate_totals: dict[str, int] = {}
    tile_index = 0

    for y0 in range(0, height, tile_height):
        y1 = min(height, y0 + tile_height)
        for x0 in range(0, width, tile_width):
            x1 = min(width, x0 + tile_width)
            tile = np.ascontiguousarray(contiguous[:, y0:y1, x0:x1])
            prefix = f"t{tile_index:04d}"
            base_doc = {
                "index": tile_index,
                "origin": [int(y0), int(x0)],
                "shape": [int(y1 - y0), int(x1 - x0)],
            }

            first = tile.reshape(-1)[0]
            constant_is_json_safe = not (
                tile.dtype.kind == "f" and not bool(np.isfinite(first))
            )
            if constant_is_json_safe and np.all(tile == first):
                value = first.item() if isinstance(first, np.generic) else first
                chosen = _Candidate("constant", {"mode": "constant", "value": value}, [], [])
                candidates = [chosen]
            else:
                candidates: list[_Candidate] = [_candidate_raw(tile, prefix, clevel)]
                repeat = _candidate_repeat(tile, prefix, clevel)
                if repeat is not None:
                    candidates.append(repeat)
                static_mask = _candidate_static_mask(tile, prefix, clevel)
                if static_mask is not None:
                    candidates.append(static_mask)
                sparse = _candidate_sparse_delta(tile, prefix, clevel, sparse_threshold)
                if sparse is not None and sparse.mode != "repeat":
                    candidates.append(sparse)
                bitpacked = _candidate_delta_bitpack(tile, prefix, clevel)
                if bitpacked is not None and bitpacked.mode != "repeat":
                    candidates.append(bitpacked)
                if enable_jungfrau_split:
                    jungfrau = _candidate_jungfrau_split(tile, prefix, clevel)
                    if jungfrau is not None:
                        candidates.append(jungfrau)
                chosen = min(candidates, key=lambda item: item.estimated_size)

            for candidate in candidates:
                candidate_totals[candidate.mode] = candidate_totals.get(candidate.mode, 0) + candidate.estimated_size
            tile_doc = {**base_doc, **chosen.tile_doc, "estimated_size": chosen.estimated_size}
            header["tiles"].append(tile_doc)
            header["segments"].extend(chosen.segment_docs)
            all_segment_data.extend(chosen.segment_data)
            mode_counts[chosen.mode] = mode_counts.get(chosen.mode, 0) + 1
            mode_payload_bytes[chosen.mode] = mode_payload_bytes.get(chosen.mode, 0) + sum(len(x) for x in chosen.segment_data)
            tile_index += 1
            del tile, candidates, chosen

    header["mode_counts"] = mode_counts
    header["mode_payload_bytes"] = mode_payload_bytes
    header["tile_count"] = tile_index
    packed = pack_hdxfb(header, all_segment_data)
    candidate_totals["tile-adaptive-v1"] = len(packed)
    return packed, header, candidate_totals


def _decode_v1(header: dict[str, Any], segments: dict[str, np.ndarray]) -> np.ndarray:
    logical_dtype = descriptor_to_dtype(header.get("logical_dtype"))
    logical_shape = tuple(int(x) for x in header.get("logical_shape", []))
    transform = header.get("transform")
    if transform == "none":
        source = segments.get("frames")
        if source is None:
            source = segments.get("array")
        if source is None:
            raise HDXFBError("transform=none has no frames/array segment")
        return np.asarray(source, dtype=logical_dtype).reshape(logical_shape)
    if transform == "delta-prev":
        key = segments.get("keyframe")
        deltas = segments.get("deltas")
        if key is None or deltas is None:
            raise HDXFBError("delta-prev is missing keyframe/deltas")
        output = np.empty(logical_shape, dtype=logical_dtype)
        output[0] = key
        previous = output[0].astype(np.int64)
        info = np.iinfo(logical_dtype)
        for index in range(1, logical_shape[0]):
            current = previous + deltas[index - 1].astype(np.int64)
            if current.min(initial=0) < int(info.min) or current.max(initial=0) > int(info.max):
                raise HDXFBError("delta reconstruction exceeds logical dtype")
            output[index] = current.astype(logical_dtype)
            previous = current
        return output
    raise HDXFBError(f"unsupported HDXFB v1 transform: {transform!r}")


def _reconstruct_temporal(key: np.ndarray, deltas: np.ndarray, logical_dtype: np.dtype[Any]) -> np.ndarray:
    shape = (deltas.shape[0] + 1,) + tuple(key.shape)
    output = np.empty(shape, dtype=logical_dtype)
    output[0] = key
    previous = output[0].astype(np.int64)
    info = np.iinfo(logical_dtype)
    for index in range(1, shape[0]):
        current = previous + deltas[index - 1].astype(np.int64)
        if current.min(initial=0) < int(info.min) or current.max(initial=0) > int(info.max):
            raise HDXFBError("tile delta reconstruction exceeds logical dtype")
        output[index] = current.astype(logical_dtype)
        previous = current
    return output


def _decode_tile(tile_doc: dict[str, Any], segments: dict[str, np.ndarray], *, frame_count: int, logical_dtype: np.dtype[Any]) -> np.ndarray:
    height, width = (int(x) for x in tile_doc["shape"])
    mode = str(tile_doc.get("mode"))
    refs = tile_doc.get("segments", {})
    shape = (frame_count, height, width)
    if mode == "constant":
        return np.full(shape, tile_doc.get("value"), dtype=logical_dtype)
    if mode == "repeat":
        key = np.asarray(segments[refs["keyframe"]], dtype=logical_dtype)
        return np.broadcast_to(key, shape).copy()
    if mode == "raw":
        return np.asarray(segments[refs["raw"]], dtype=logical_dtype).reshape(shape).copy()
    if mode == "static-mask":
        pixel_count = int(tile_doc["pixel_count"])
        mask_bits = np.unpackbits(np.asarray(segments[refs["mask"]], dtype=np.uint8), bitorder="little")
        if mask_bits.size < pixel_count:
            raise HDXFBError("truncated static mask")
        mask = mask_bits[:pixel_count].astype(bool)
        static_values = np.asarray(segments[refs["static_values"]], dtype=logical_dtype).reshape(-1)
        dynamic_values = np.asarray(segments[refs["dynamic_values"]], dtype=logical_dtype)
        if static_values.size != int(np.count_nonzero(mask)):
            raise HDXFBError("static-value count mismatch")
        if dynamic_values.shape != (frame_count, pixel_count - static_values.size):
            raise HDXFBError("dynamic-value shape mismatch")
        flat = np.empty((frame_count, pixel_count), dtype=logical_dtype)
        flat[:, mask] = static_values[None, :]
        flat[:, ~mask] = dynamic_values
        return flat.reshape(shape)
    if mode == "sparse-delta":
        key = np.asarray(segments[refs["keyframe"]], dtype=logical_dtype)
        delta_shape = tuple(int(x) for x in tile_doc["delta_shape"])
        total = int(tile_doc["delta_element_count"])
        gaps = np.asarray(segments[refs["position_gaps"]], dtype=np.uint64).reshape(-1)
        values = np.asarray(segments[refs["values"]], dtype=np.int64).reshape(-1)
        if gaps.size != values.size or gaps.size != int(tile_doc["nonzero_count"]):
            raise HDXFBError("sparse delta count mismatch")
        positions = np.cumsum(gaps, dtype=np.uint64)
        if positions.size and int(positions[-1]) >= total:
            raise HDXFBError("sparse delta position outside stream")
        dense = np.zeros((total,), dtype=np.int64)
        dense[positions.astype(np.intp)] = values
        return _reconstruct_temporal(key, dense.reshape(delta_shape), logical_dtype)
    if mode == "delta-bitpack":
        key = np.asarray(segments[refs["keyframe"]], dtype=logical_dtype)
        delta_shape = tuple(int(x) for x in tile_doc["delta_shape"])
        count = int(tile_doc["delta_element_count"])
        packed = np.asarray(segments[refs["packed"]], dtype=np.uint8)
        zigzag = unpack_unsigned(packed, count=count, bit_width=int(tile_doc["bit_width"]))
        deltas = _zigzag_decode(zigzag).reshape(delta_shape)
        return _reconstruct_temporal(key, deltas, logical_dtype)
    if mode == "jungfrau-split":
        count = int(tile_doc["element_count"])
        gain_packed = np.asarray(segments[refs["gain"]], dtype=np.uint8)
        gain = unpack_unsigned(gain_packed, count=count, bit_width=int(tile_doc["gain_bit_width"])).astype(np.uint16)
        adc = np.asarray(segments[refs["adc"]], dtype=np.uint16).reshape(-1)
        if adc.size != count:
            raise HDXFBError("Jungfrau ADC element-count mismatch")
        raw = ((gain << np.uint16(14)) | (adc & np.uint16(0x3FFF))).astype(np.uint16)
        raw = raw.reshape(shape)
        if logical_dtype.kind == "i":
            return raw.view(np.int16).astype(logical_dtype, copy=False)
        return raw.astype(logical_dtype, copy=False)
    raise HDXFBError(f"unsupported tile mode: {mode!r}")


def _decode_v2(header: dict[str, Any], segments: dict[str, np.ndarray]) -> np.ndarray:
    if header.get("transform") != "tile-adaptive-v1":
        raise HDXFBError(f"unsupported HDXFB v2 transform: {header.get('transform')!r}")
    logical_dtype = descriptor_to_dtype(header.get("logical_dtype"))
    logical_shape = tuple(int(x) for x in header.get("logical_shape", []))
    if len(logical_shape) != 3:
        raise HDXFBError("tile-adaptive logical shape must be F×H×W")
    output = np.empty(logical_shape, dtype=logical_dtype)
    coverage = np.zeros(logical_shape[1:], dtype=np.uint8)
    for tile_doc in header.get("tiles", []):
        y0, x0 = (int(x) for x in tile_doc["origin"])
        height, width = (int(x) for x in tile_doc["shape"])
        y1, x1 = y0 + height, x0 + width
        if y0 < 0 or x0 < 0 or y1 > logical_shape[1] or x1 > logical_shape[2]:
            raise HDXFBError("tile lies outside logical frame")
        if np.any(coverage[y0:y1, x0:x1]):
            raise HDXFBError("overlapping adaptive tiles")
        tile = _decode_tile(tile_doc, segments, frame_count=logical_shape[0], logical_dtype=logical_dtype)
        if tile.shape != (logical_shape[0], height, width):
            raise HDXFBError("decoded tile shape mismatch")
        output[:, y0:y1, x0:x1] = tile
        coverage[y0:y1, x0:x1] = 1
    if not np.all(coverage):
        raise HDXFBError("adaptive tile coverage is incomplete")
    return output


def _decode_v3_delta_stream(
    header: dict[str, Any],
    segments: dict[str, np.ndarray],
    *,
    work_dtype: np.dtype[Any],
    static_model: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    delta_shape = tuple(int(x) for x in header.get("delta_shape", []))
    total = int(header.get("delta_element_count", math.prod(delta_shape) if delta_shape else 0))
    if total != int(math.prod(delta_shape)):
        raise HDXFBError("HDXFB v3 Delta element-count mismatch")
    encoded_shape = tuple(int(x) for x in header.get("encoded_delta_shape", delta_shape))
    encoded_total = int(header.get("encoded_delta_element_count", math.prod(encoded_shape) if encoded_shape else 0))
    if encoded_total != int(math.prod(encoded_shape)):
        raise HDXFBError("HDXFB v3 encoded Delta element-count mismatch")
    encoding = str(header.get("delta_encoding", "blosc2-array"))
    encoded: np.ndarray
    if encoding == "blosc2-array":
        source = segments.get("deltas")
        if source is None:
            raise HDXFBError("HDXFB v3 has no deltas segment")
        encoded = np.asarray(source).reshape(encoded_shape)
    elif encoding == "zero-rle-bitpack":
        runs = segments.get("zero_runs")
        packed = segments.get("nonzero_packed")
        if runs is None or packed is None:
            raise HDXFBError("zero-rle-bitpack is missing zero_runs/nonzero_packed")
        runs64 = np.asarray(runs, dtype=np.uint64).reshape(-1)
        nonzero_count = int(header.get("nonzero_count", -1))
        if nonzero_count < 0 or runs64.size != nonzero_count + 1:
            raise HDXFBError("zero-RLE run count mismatch")
        if int(runs64.sum(dtype=np.uint64)) + nonzero_count != encoded_total:
            raise HDXFBError("zero-RLE stream does not cover the Delta array")
        zigzag = unpack_unsigned(
            np.asarray(packed, dtype=np.uint8),
            count=nonzero_count,
            bit_width=int(header.get("nonzero_bit_width", 0)),
        )
        values = _zigzag_decode(zigzag).astype(work_dtype, copy=False)
        dense = np.zeros((encoded_total,), dtype=work_dtype)
        if nonzero_count:
            positions = np.cumsum(runs64[:-1], dtype=np.uint64) + np.arange(nonzero_count, dtype=np.uint64)
            if int(positions[-1]) >= encoded_total:
                raise HDXFBError("zero-RLE nonzero position outside Delta stream")
            dense[positions.astype(np.intp)] = values
        encoded = dense.reshape(encoded_shape)
    else:
        raise HDXFBError(f"unsupported HDXFB v3 Delta encoding: {encoding!r}")

    pixel_encoding = str(header.get("delta_pixel_encoding", "full"))
    if pixel_encoding == "full":
        return encoded.reshape(delta_shape)
    pixel_count = int(math.prod(delta_shape[1:]))
    if pixel_encoding == "dynamic-mask":
        mask_segment = segments.get("delta_static_mask")
        if mask_segment is None:
            raise HDXFBError("dynamic-mask Delta is missing delta_static_mask")
        bits = np.unpackbits(np.asarray(mask_segment, dtype=np.uint8), bitorder="little")
        if bits.size < pixel_count:
            raise HDXFBError("truncated Delta static mask")
        static_mask = bits[:pixel_count].astype(bool)
    elif pixel_encoding == "shared-dynamic-mask":
        if static_model is None or "mask" not in static_model:
            raise HDXFBError("shared-dynamic-mask requires a shared static model")
        static_mask = np.asarray(static_model["mask"], dtype=bool).reshape(-1)
        if static_mask.size != pixel_count:
            raise HDXFBError("shared static model mask shape mismatch")
    else:
        raise HDXFBError(f"unsupported Delta pixel encoding: {pixel_encoding!r}")
    dynamic_count = pixel_count - int(np.count_nonzero(static_mask))
    if encoded.shape != (delta_shape[0], dynamic_count):
        raise HDXFBError("dynamic Delta shape does not match static mask")
    full = np.zeros(delta_shape, dtype=encoded.dtype)
    full.reshape(delta_shape[0], pixel_count)[:, ~static_mask] = encoded
    return full


def _decode_v3(
    header: dict[str, Any],
    segments: dict[str, np.ndarray],
    *,
    previous_frame: np.ndarray | None,
    static_model: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    if header.get("transform") != "delta-chain-v1":
        raise HDXFBError(f"unsupported HDXFB v3 transform: {header.get('transform')!r}")
    logical_dtype = descriptor_to_dtype(header.get("logical_dtype"))
    logical_shape = tuple(int(x) for x in header.get("logical_shape", []))
    if len(logical_shape) < 2 or logical_shape[0] <= 0:
        raise HDXFBError("HDXFB v3 logical shape must contain one or more frames")
    if logical_dtype.kind not in "iu":
        raise HDXFBError("HDXFB v3 Delta requires an integer logical dtype")
    chain_mode = str(header.get("chain_mode"))

    keyframe: np.ndarray | None = None
    if chain_mode == "anchor":
        key_encoding = str(header.get("keyframe_encoding", "full"))
        if key_encoding == "full":
            key = segments.get("keyframe")
            if key is None:
                raise HDXFBError("anchor block is missing keyframe")
            keyframe = np.asarray(key, dtype=logical_dtype).reshape(logical_shape[1:]).copy()
        elif key_encoding in ("static-split", "shared-static-split"):
            dynamic_values = segments.get("dynamic_keyframe")
            pixel_count = int(math.prod(logical_shape[1:]))
            if dynamic_values is None:
                raise HDXFBError(f"{key_encoding} keyframe is missing dynamic_keyframe")
            if key_encoding == "static-split":
                mask_segment = segments.get("static_mask")
                static_values = segments.get("static_values")
                if mask_segment is None or static_values is None:
                    raise HDXFBError("static-split keyframe is missing a segment")
                bits = np.unpackbits(np.asarray(mask_segment, dtype=np.uint8), bitorder="little")
                if bits.size < pixel_count:
                    raise HDXFBError("truncated static keyframe mask")
                mask = bits[:pixel_count].astype(bool)
                static_flat = np.asarray(static_values, dtype=logical_dtype).reshape(-1)
            else:
                if static_model is None or "mask" not in static_model or "static_values" not in static_model:
                    raise HDXFBError("shared-static-split requires a shared static model")
                mask = np.asarray(static_model["mask"], dtype=bool).reshape(-1)
                if mask.size != pixel_count:
                    raise HDXFBError("shared static model mask shape mismatch")
                static_flat = np.asarray(static_model["static_values"], dtype=logical_dtype).reshape(-1)
            flat = np.empty((pixel_count,), dtype=logical_dtype)
            dynamic_flat = np.asarray(dynamic_values, dtype=logical_dtype).reshape(-1)
            if static_flat.size != int(np.count_nonzero(mask)):
                raise HDXFBError("static keyframe value-count mismatch")
            if dynamic_flat.size != pixel_count - static_flat.size:
                raise HDXFBError("dynamic keyframe value-count mismatch")
            flat[mask] = static_flat
            flat[~mask] = dynamic_flat
            keyframe = flat.reshape(logical_shape[1:])
        else:
            raise HDXFBError(f"unsupported keyframe encoding: {key_encoding!r}")
    elif chain_mode == "continuation":
        if previous_frame is None:
            raise HDXFBError("continuation block requires the previous decoded frame")
        if tuple(previous_frame.shape) != logical_shape[1:]:
            raise HDXFBError("previous frame shape does not match continuation block")
        keyframe = np.asarray(previous_frame, dtype=logical_dtype)
    else:
        raise HDXFBError(f"invalid HDXFB v3 chain mode: {chain_mode!r}")

    # int32 safely covers every difference between 8/16-bit integer pixels.
    # Wider logical types retain int64 reconstruction semantics.
    work_dtype = np.dtype(np.int32 if logical_dtype.itemsize <= 2 else np.int64)
    deltas = _decode_v3_delta_stream(
        header, segments, work_dtype=work_dtype, static_model=static_model
    )
    expected_delta_frames = logical_shape[0] - 1 if chain_mode == "anchor" else logical_shape[0]
    if deltas.shape != (expected_delta_frames,) + logical_shape[1:]:
        raise HDXFBError("HDXFB v3 Delta shape mismatch")

    output = np.empty(logical_shape, dtype=logical_dtype)
    info = np.iinfo(logical_dtype)
    previous = keyframe.astype(work_dtype)
    delta_offset = 0
    if chain_mode == "anchor":
        output[0] = keyframe
        delta_offset = 1
    for index in range(delta_offset, logical_shape[0]):
        delta_index = index - delta_offset
        current = previous + deltas[delta_index].astype(work_dtype, copy=False)
        if current.min(initial=0) < int(info.min) or current.max(initial=0) > int(info.max):
            raise HDXFBError("HDXFB v3 Delta reconstruction exceeds logical dtype")
        output[index] = current.astype(logical_dtype)
        previous = current
    return output


def decode_hdxfb(
    raw: bytes,
    *,
    member_path: str = "<memory>",
    verify_hash: bool = True,
    previous_frame: np.ndarray | None = None,
    static_model: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    header, body = parse_hdxfb(raw)
    segments = _read_segments(header, body)
    version = int(header["version"])
    if version == 1:
        output = _decode_v1(header, segments)
    elif version == 2:
        output = _decode_v2(header, segments)
    elif version in (3, 4):
        output = _decode_v3(
            header, segments, previous_frame=previous_frame,
            static_model=static_model,
        )
    else:
        raise HDXFBError(f"unsupported HDXFB version: {version}")
    expected_shape = tuple(int(x) for x in header.get("logical_shape", []))
    expected_dtype = descriptor_to_dtype(header.get("logical_dtype"))
    if output.shape != expected_shape:
        raise HDXFBError(f"logical shape mismatch in {member_path}")
    if dtype_descriptor(output.dtype) != dtype_descriptor(expected_dtype):
        raise HDXFBError(f"logical dtype mismatch in {member_path}")
    if verify_hash:
        expected_hash = header.get("logical_sha256")
        actual_hash = logical_sha256(output)
        if expected_hash and actual_hash != expected_hash:
            raise HDXFBError(
                f"logical SHA-256 mismatch in {member_path}; expected={expected_hash}, actual={actual_hash}"
            )
    return output, header