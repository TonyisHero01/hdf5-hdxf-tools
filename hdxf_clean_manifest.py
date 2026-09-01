#!/usr/bin/env python3
"""Remove non-source provenance fields from an existing HDXF manifest.

Payload members are copied byte-for-byte. Only manifest.json is replaced.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import zipfile

MANIFEST_PATH = "manifest.json"
FORMAT_VERSION = "0.4.5"


def _clean_manifest(manifest: dict) -> dict:
    hdxf = manifest.get("hdxf", {}) if isinstance(manifest.get("hdxf"), dict) else {}
    cleaned: dict = {
        "hdxf": {
            "format": hdxf.get("format", "HDF5-derived Detector eXchange Format"),
            "format_id": hdxf.get("format_id", "hdxf"),
            "version": FORMAT_VERSION,
            "profile": hdxf.get("profile"),
        }
    }

    source = manifest.get("source", {}) if isinstance(manifest.get("source"), dict) else {}
    source_files = []
    for item in source.get("files", []):
        if isinstance(item, dict):
            source_files.append({
                "id": item.get("id"),
                "filename": item.get("filename"),
            })
    cleaned["source"] = {
        "format": "HDF5",
        "main_file": source.get("main_file"),
        "files": source_files,
    }

    for key in ("root_object", "objects", "links"):
        if key in manifest:
            cleaned[key] = manifest[key]

    detector = manifest.get("detector_archive")
    if isinstance(detector, dict):
        cleaned["detector_archive"] = {
            "frames": detector.get("frames", {}),
            "calibration": detector.get("calibration", {}),
        }

    # Calibration bundle profile uses a different top-level layout.
    if "calibration" in manifest and "detector_archive" not in manifest:
        cleaned["calibration"] = manifest.get("calibration")
    if "members" in manifest:
        cleaned["members"] = manifest.get("members")

    return cleaned


def _copy_info(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    clone = zipfile.ZipInfo(info.filename, date_time=info.date_time)
    clone.compress_type = info.compress_type
    clone.comment = info.comment
    clone.extra = info.extra
    clone.create_system = info.create_system
    clone.create_version = info.create_version
    clone.extract_version = info.extract_version
    clone.flag_bits = info.flag_bits
    clone.volume = info.volume
    clone.internal_attr = info.internal_attr
    clone.external_attr = info.external_attr
    return clone


def clean_archive(input_path: Path, output_path: Path, *, overwrite: bool) -> tuple[int, int]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if input_path == output_path:
        raise ValueError("input and output must be different paths")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=output_path.name + ".", suffix=".tmp", dir=output_path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with zipfile.ZipFile(input_path, "r") as source_zip:
            if MANIFEST_PATH not in source_zip.namelist():
                raise ValueError("archive has no manifest.json")
            manifest = json.loads(source_zip.read(MANIFEST_PATH))
            cleaned = _clean_manifest(manifest)
            manifest_bytes = json.dumps(
                cleaned, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")

            with zipfile.ZipFile(tmp_path, "w", allowZip64=True) as target_zip:
                for info in source_zip.infolist():
                    if info.filename == MANIFEST_PATH:
                        continue
                    target_zip.writestr(_copy_info(info), source_zip.read(info.filename))
                manifest_info = zipfile.ZipInfo(MANIFEST_PATH, date_time=(1980, 1, 1, 0, 0, 0))
                manifest_info.compress_type = zipfile.ZIP_STORED
                target_zip.writestr(manifest_info, manifest_bytes)

        if output_path.exists():
            output_path.unlink()
        os.replace(tmp_path, output_path)
        return input_path.stat().st_size, output_path.stat().st_size
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean non-HDF5 provenance fields from an existing HDXF manifest")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output or args.input.with_name(args.input.stem + "-source-metadata-only.hdxf")
    before, after = clean_archive(args.input, output, overwrite=args.overwrite)
    print(f"created: {output.resolve()}")
    print(f"before: {before} bytes")
    print(f"after:  {after} bytes")
    print(f"saved:  {before - after} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
