# Stage 1 maintenance summary

Baseline: user-provided `hdf5-hdxf-tools.zip` (2026-10-20).

This update intentionally contains **no changes** to HDF5/HDXF encoding, payload decoding, lossless reconstruction, scientific integration algorithms, or detector geometry. It is a code/documentation maintenance update, not the upcoming feature release.

## Apply the update

**Full source package:** Extract the `hdf5-hdxf-tools/` folder to a new location, or compare/copy it over your existing working tree after backing up local changes.

**Changed-files package:** Copy the files under `hdf5-hdxf-tools/` into the corresponding Git working-tree location. `hdxf_release.py` must be copied together with the modified runnable scripts; it is imported by them. The patch-only ZIP is *not* a standalone distribution.

Neither package contains `.git`, `venv`/`.venv`, or user datasets. The existing `.git` directory in your local clone should be left untouched.

## Tests

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
python hdxf_viewer.py --version
python hdxf_pixelop.py --help
```

Run all commands inside the environment used for HDXF. The verification run accompanying this package used Linux with a limited local environment, not Windows 11 or a CUDA card; it did not test a GUI session or a real HDF5/HDXF dataset end-to-end.

## Changed files

- `hdxf_viewer.py`: pixel-op Divide invokes `--divide`; GUI now supports `--version` and has an unambiguous Viewer-build version alias.
- `hdxf_pixelop.py`: uses only `--divide`; the legacy misspelling and its internal operation identifier were removed without changing division arithmetic.
- `hdf5_to_hdxf.py`, `hdxf_to_hdf5.py`, `hdxf_sum.py`, `hdxf_subtract.py`, `hdf5_processing_tree.py`: display a common suite version via `--version`.
- `hdxf_release.py`: new shared version-label helper.
- `README.md`: revised to match actual source distribution.
- `requirements.txt`: UTF-8 conversion only (versions not changed).
- `INSTALL_WINDOWS.md`, `VERSIONS.md`, `CHANGELOG.md`: new documentation.
- `tests/test_stage1_cli.py`: maintenance regression checks.

## Deferred work

- Numeric fidelity and compatibility of `hdxf_sum.py`, `hdxf_subtract.py`, `hdxf_pixelop.py` with HDXF archive format 0.4.8.
- Viewer PNG/JPEG image export.
- Module gap masks beyond the existing Notebook-specific exclusions.
- New normalization based on direct-beam/monitor intensity.
- Detector tilt/distance correction with pyFAI geometry and associated validation.

These are separately testable follow-up changes and should not be combined blindly with a formatting/CLI release.
