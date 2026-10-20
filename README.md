# HDF5 and HDXF Universal Tools

Desktop viewing, lossless conversion, and frame processing for scientific HDF5/NeXus detector data and **HDXF** archives.

**Current software-suite version:** `0.4.8` (source snapshot; see [VERSIONS.md](VERSIONS.md)). This is **not** a claim that every auxiliary tool writes or accepts archive format `0.4.8`.

## Included in this repository

| Tool | File | What it does |
| --- | --- | --- |
| HDXF Viewer | `hdxf_viewer.py` | Desktop GUI for HDF5/HDXF frame navigation, pixel inspection, ROI tools, beam center, resolution rings, pyFAI 1D/2D integration, and the separate Notebook-reference radial calculation |
| HDF5 → HDXF | `hdf5_to_hdxf.py` | Archive detector data and metadata in HDXF |
| HDXF → HDF5 | `hdxf_to_hdf5.py` | Reconstruct the HDF5 master and associated data files |
| HDXF operations | `hdxf_sum.py`, `hdxf_subtract.py`, `hdxf_pixelop.py` | Sum/mean/exposure-normalized output, filtering/subtraction, and scalar arithmetic |
| HDF5 operations | `hdf5_processing_tree.py` | HDF5 frame processing while retaining the relevant HDF5 structure |
| Core libraries | `hdxf_codec.py`, `hdxf_index.py`, `hdxf_hardware.py`, `hdxf_notebook_reference.py`, `hdxf_release.py` | Codec, indexing, hardware guidance, Notebook reference, and release metadata |

**Not included:** There is **no separate `hdf5_hdxf_structure_explorer.py`** in the current repository. The Viewer has its own metadata/data browsing functions; an independent Structure Explorer is not part of this distribution.

**Scope of supported features:** The Viewer has a PONI/pyFAI geometry interface, but a dedicated detector-tilt correction workflow is not yet released. The Notebook-reference mode provides configurable gap-row exclusions; do not treat them as a validated automatic module-gap mask for every detector. Image exports to JPEG/PNG, per-frame direct-beam normalization, and a generalized geometry-correction workflow are planned, not features promised by this snapshot.

## Requirements

- **64-bit CPython 3.13** is the target for the current pinned Python packages; other Python versions are not covered by these pins.
- Windows 11 or a compatible Linux desktop with Tk/Tkinter.
- Enough available RAM and disk space for the chosen detector archive.
- An NVIDIA GPU and compatible driver **only** for optional CUDA operations.

For Ubuntu, install the platform-provided Tk and venv packages (with matching Python versions):

```bash
sudo apt update
sudo apt install python3-venv python3-tk
```

## Install (CPU / standard)

For step-by-step Windows troubleshooting, see **[INSTALL_WINDOWS.md](INSTALL_WINDOWS.md)**.

### Windows PowerShell

Open PowerShell **inside the folder containing `requirements.txt`**, then run:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Confirm that `python --version` reports Python 3.13 before creating the environment. If PowerShell blocks activation, see the Windows guide. Avoid installing into the global Python environment: activate `.venv` **before** installing.

### Ubuntu / Linux

Use an installed, compatible Python 3.13 executable (adjust its name to your system):

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

## Optional CUDA 12 installation

This is **optional**; the CPU paths are usable without it. CUDA acceleration applies to supported processing paths, not all Viewer or pyFAI operations.

After installing and activating the base environment:

```bash
python -m pip install -r requirements_gpu_cuda12.txt
nvidia-smi
python -c "import cupy as cp; print('CUDA devices:', cp.cuda.runtime.getDeviceCount())"
```

`requirements_gpu_cuda12.txt` already includes `requirements.txt`; it may be used directly for a fresh GPU-enabled environment. Your NVIDIA driver must support the selected CUDA runtime. CuPy installation alone does not ensure an available CUDA device.

## Start the Viewer GUI

Run from the project directory in the active virtual environment:

```bash
python hdxf_viewer.py
```

A graphical desktop session is required. You can also supply a file path as an optional argument: `python hdxf_viewer.py --help` documents invocation details.

## Command-line tools

These tools are shipped in the same directory as the core modules. Query their supported options before processing data:

```bash
python hdf5_to_hdxf.py --help
python hdxf_to_hdf5.py --help
python hdxf_sum.py --help
python hdxf_subtract.py --help
python hdxf_pixelop.py --help
python hdf5_processing_tree.py --help
```

For a uniform software version query, each runnable script also supports `--version`. **The HDXF archive format version, its binary frame-block (`HDXFB`) version, and each tool's build/version are distinct**; see [VERSIONS.md](VERSIONS.md).

Pixel division uses the `--divide` argument exclusively. Existing commands must use this spelling.

**Compatibility caution:** The included `hdxf_sum.py`, `hdxf_subtract.py`, and `hdxf_pixelop.py` have older archive-version acceptance policies than the current 0.4.8 converter. Do not assume every old processing tool supports HDXF 0.4.8. Consult [VERSIONS.md](VERSIONS.md) before processing newly created archives.

## Detector data notes

- HDF5 master files may reference separate external data files; preserve their relative links or use the converter's documented external-root support.
- `hdf5plugin` registers detector compression filters such as Bitshuffle/LZ4. File metadata may be readable even when pixel decoding fails because a filter is missing.
- HDXF encoding aims to preserve detector data losslessly; use `--verify` where supported and retain original source files until restoration has been checked.
- JPEG and display-oriented PNG are not substitutes for original scientific arrays when numerical precision matters.
- Use the Notebook-reference radial mode only when its geometry, beam center, masks, and averaging assumptions are appropriate for your experiment; it is not identical to pyFAI integration.

## Release history

Changes to the repository and compatibility notes are recorded in [CHANGELOG.md](CHANGELOG.md).

## License

No license file is included in this uploaded repository snapshot. A project license must be chosen and added before inviting others to reuse or redistribute the source under explicit terms.
