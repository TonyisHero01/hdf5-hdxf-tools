# Changelog

Documented changes to the **software repository**, not automatic changes to the HDXF on-disk archive format. Historical release entries are omitted because this source snapshot does not contain enough verified release notes to reconstruct them reliably.

## [Unreleased] — follow-up CLI cleanup (2026-10-20)

### Changed

- Removed the deprecated misspelled division argument entirely; `hdxf_pixelop.py` accepts only `--divide`.
- Standardized the internal division operation name and output processing metadata to `divide`.
- Updated the Viewer's command generation, README, maintenance notes, and regression tests to reflect the single supported option.

### Compatibility

- Existing command lines using the deprecated spelling must switch to `--divide`.
- Division arithmetic, HDXFB encoding/decoding, and on-disk HDXF format versions are unchanged. Only division operation metadata naming is standardized.

## [Unreleased] — 2026-10-20 (first-stage maintenance)

### Added

- Central human-readable suite version metadata in `hdxf_release.py`.
- `--version` reporting for the desktop Viewer and six runnable processing/conversion tools.
- `VERSIONS.md` explaining suite, tool, HDXF archive, HDXFB codec, and build version labels.
- `INSTALL_WINDOWS.md` adapted from a contributor's Windows Python 3.13 installation guide.

### Changed

- Updated the README to describe only the tools present in the supplied repository; removed the non-existent standalone Structure Explorer launch instructions.
- Clarified optional GPU setup, virtual environment activation, and Viewer startup using `python hdxf_viewer.py`.
- Re-encoded `requirements.txt` from UTF-16 to UTF-8 without changing dependency pins.
- Updated the Viewer's pixel-operation command builder to use the correctly spelled `--divide` flag.

### Fixed

- `hdxf_pixelop.py` initially added a correctly spelled `--divide` argument alongside an older compatibility alias (removed in the follow-up cleanup above).

### Known limitations

- `hdxf_sum.py`, `hdxf_subtract.py`, and `hdxf_pixelop.py` still restrict their accepted archive versions to `0.4.5` and older. Their support for the main 0.4.8 archive output has **not** been asserted or unlocked.
- A separate HDF5/HDXF Structure Explorer script is not shipped in this repository snapshot.
- Dedicated detector-tilt geometry correction, PNG/JPEG frame export, detector-independent automatic gap masks, and direct-beam intensity normalization have not been added in this maintenance stage.

### Compatibility and data integrity

- The main converter's archive format version, binary block encoding, decoding algorithms, and HDF5 restoration algorithms were **not changed** by this documentation/CLI update.
- Added the correctly spelled argument while retaining existing arithmetic semantics; see the follow-up cleanup for removal of the legacy alias.
