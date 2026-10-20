"""Human-readable HDXF *software suite* version information.

The suite version is independent from archive manifest versions, HDXFB block
versions, and older utility build identifiers. Do not reuse SUITE_VERSION
when writing or accepting archive formats.
"""
from __future__ import annotations

SUITE_VERSION = "0.4.8"
CURRENT_ARCHIVE_FORMAT_VERSION = "0.4.8"


def version_banner(tool: str, *, tool_version: str | None = None,
                   output_archive_format: str | None = None,
                   build: str | None = None) -> str:
    parts = [f"{tool} | HDXF suite {SUITE_VERSION}"]
    if tool_version:
        parts.append(f"tool {tool_version}")
    if output_archive_format:
        parts.append(f"output archive format {output_archive_format}")
    if build:
        parts.append(f"build {build}")
    return " | ".join(parts)
