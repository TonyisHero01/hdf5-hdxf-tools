# Version identifiers and compatibility

Different version strings in this repository serve different purposes. They must not be rewritten to match one another unless the corresponding code and format support have actually been upgraded and tested.

| Identifier | Current source value | Meaning |
| --- | --- | --- |
| **Suite release label** | `0.4.8` | Common display label of the source snapshot, exported by `hdxf_release.py` for `--version` reporting |
| **Main converter tool version** | `0.4.8` | `hdf5_to_hdxf.py` implementation's declared version |
| **Main converter output archive format** | `0.4.8` | `manifest.json` HDXF format version written by the main encoder |
| **Viewer build label** | `0.4.8.0` | Historical Viewer implementation label, not the HDXF archive format |
| **HDXFB payload versions** | `1`, `2`, `3`, `4` | Independent binary frame-block encoding revisions decoded by the shared codec |
| **Sum tool output format** | `0.4.5` | Actual format version selected by the shipped `hdxf_sum.py` writer; do **not** silently change it to 0.4.8 |
| **Pixel operation tool build** | `1.0.0-hdxf-pixelop` | Script-specific implementation identifier |
| **Subtract tool build** | `1.1.0-hdxf-stream-verify` | Script-specific implementation identifier |
| **Restore tool build** | `2026-10-14-parallel-external-restore-v5` | Diagnostic build label, not an archive version |

## CLI version reporting

The user-invoked commands `hdxf_viewer.py`, `hdf5_to_hdxf.py`, `hdxf_to_hdf5.py`, `hdxf_sum.py`, `hdxf_subtract.py`, `hdxf_pixelop.py`, and `hdf5_processing_tree.py` accept `--version` and show the suite label together with any applicable tool/build/output-format information.

The shared label does **not** control archive decoding or encoding. Individual `SUPPORTED_FORMAT_VERSIONS`, `FORMAT_VERSION`, and HDXFB checks remain authoritative.

## Known compatibility limitation

The version-acceptance lists in the uploaded `hdxf_sum.py`, `hdxf_subtract.py`, and `hdxf_pixelop.py` end at HDXF archive format `0.4.5`. Archives from the main 0.4.8 converter are therefore **not guaranteed to be accepted by these auxiliary tools**. The first-stage documentation/CLI release intentionally does not bypass this check. Full compatibility requires testing decoded frames, metadata preservation, and round-trip behavior before expanding the list.

## Future release policy

- The suite release label may advance with software releases.
- Archive format versions advance **only** when the on-disk archive schema or compatibility semantics require it.
- Frame-block codec versions advance **only** for new binary block encoding features.
- Tool build identifiers may be used for diagnostics, without implying a new archive revision.
- When adding new archive acceptance versions to auxiliary tools, include round-trip tests with representative archives and document the new support explicitly.
