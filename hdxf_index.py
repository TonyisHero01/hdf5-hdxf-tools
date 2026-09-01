#!/usr/bin/env python3
"""Compact binary frame-block index used by HDXF 0.4.4.

The index removes thousands of repeated JSON fields from manifest.json while
keeping every frame block content-addressed and independently checksummed.
"""
from __future__ import annotations

import hashlib
import struct
import zlib
from typing import Any, Iterable

MAGIC = b"HDXFI\x00\x01"
HEADER = struct.Struct("<7sIHB")  # magic, count, record_size, compression
RECORD = struct.Struct("<32sIQIIIQI")
COMPRESSION_NONE = 0
COMPRESSION_ZLIB = 1
FLAG_REQUIRES_PREVIOUS = 1 << 0
VERSION_SHIFT = 8
VERSION_MASK = 0xFF << VERSION_SHIFT


class HDXFIError(ValueError):
    pass


def _digest_bytes(value: str) -> bytes:
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise HDXFIError(f"invalid SHA-256: {value!r}") from exc
    if len(raw) != 32:
        raise HDXFIError(f"invalid SHA-256 length: {value!r}")
    return raw


def pack_frame_index(
    blocks: Iterable[dict[str, Any]],
    *,
    dataset_ordinals: dict[str, int],
    compress: bool = True,
) -> bytes:
    records = bytearray()
    count = 0
    for block in blocks:
        source_object = str(block.get("source_object", ""))
        if source_object not in dataset_ordinals:
            raise HDXFIError(f"frame block has unknown source object: {source_object!r}")
        version = int(block.get("hdxfb_version", 3))
        if not (0 <= version <= 255):
            raise HDXFIError(f"HDXFB version out of range: {version}")
        flags = version << VERSION_SHIFT
        if bool(block.get("requires_previous_frame", False)):
            flags |= FLAG_REQUIRES_PREVIOUS
        records.extend(
            RECORD.pack(
                _digest_bytes(str(block["sha256"])),
                int(block["size_bytes"]),
                int(block["global_start_frame"]),
                int(block.get("start_frame", 0)),
                int(block["frame_count"]),
                int(flags),
                int(block.get("chain_anchor_global_start_frame", block["global_start_frame"])),
                int(dataset_ordinals[source_object]),
            )
        )
        count += 1
    body = bytes(records)
    compression = COMPRESSION_NONE
    if compress and body:
        compressed = zlib.compress(body, level=9)
        if len(compressed) < len(body):
            body = compressed
            compression = COMPRESSION_ZLIB
    return HEADER.pack(MAGIC, count, RECORD.size, compression) + body


def unpack_frame_index(
    raw: bytes,
    *,
    datasets: list[dict[str, Any]],
    frame_shape: list[int] | tuple[int, ...],
) -> list[dict[str, Any]]:
    if len(raw) < HEADER.size:
        raise HDXFIError("truncated HDXFI header")
    magic, count, record_size, compression = HEADER.unpack_from(raw, 0)
    if magic != MAGIC:
        raise HDXFIError("invalid HDXFI magic")
    if record_size != RECORD.size:
        raise HDXFIError(
            f"unsupported HDXFI record size: {record_size}; expected {RECORD.size}"
        )
    body = raw[HEADER.size:]
    if compression == COMPRESSION_ZLIB:
        try:
            body = zlib.decompress(body)
        except zlib.error as exc:
            raise HDXFIError(f"invalid HDXFI zlib stream: {exc}") from exc
    elif compression != COMPRESSION_NONE:
        raise HDXFIError(f"unsupported HDXFI compression: {compression}")
    expected = int(count) * RECORD.size
    if len(body) != expected:
        raise HDXFIError(f"HDXFI body size mismatch: {len(body)} != {expected}")

    shape_tail = [int(x) for x in frame_shape]
    blocks: list[dict[str, Any]] = []
    offset = 0
    for index in range(int(count)):
        digest_raw, size_bytes, global_start, local_start, frame_count, flags, anchor, dataset_index = RECORD.unpack_from(body, offset)
        offset += RECORD.size
        if dataset_index >= len(datasets):
            raise HDXFIError(
                f"record {index} references dataset ordinal {dataset_index}, count={len(datasets)}"
            )
        dataset_doc = datasets[dataset_index]
        source_object = str(dataset_doc.get("object", ""))
        digest = digest_raw.hex()
        version = (int(flags) & VERSION_MASK) >> VERSION_SHIFT
        requires_previous = bool(int(flags) & FLAG_REQUIRES_PREVIOUS)
        block = {
            "path": f"payloads/{digest}.hdxfb",
            "encoding": f"frame-block-v{version}",
            "size_bytes": int(size_bytes),
            "sha256": digest,
            "start": [int(local_start)] + [0] * len(shape_tail),
            "count": [int(frame_count)] + shape_tail,
            "chunk_shape": [int(frame_count)] + shape_tail,
            "start_frame": int(local_start),
            "global_start_frame": int(global_start),
            "frame_count": int(frame_count),
            "hdxfb_version": int(version),
            "requires_previous_frame": requires_previous,
            "chain_mode": "continuation" if requires_previous else "anchor",
            "chain_anchor_global_start_frame": int(anchor),
            "source_object": source_object,
        }
        blocks.append(block)
    return blocks


def index_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()