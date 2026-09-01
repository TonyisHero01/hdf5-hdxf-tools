# HDF5 and HDXF Universal Tools

A collection of utilities for working with **HDF5** and **HDXF** detector data.

The project currently includes:

- **HDXF Viewer** — open and inspect HDXF data.
- **HDF5 / HDXF Structure Explorer** — inspect and compare HDF5 and HDXF internal structures.
- Standard CPU support.
- Optional **CUDA 12** dependencies for GPU-accelerated workflows.

---

## Requirements

- Python 3.10+ recommended
- `pip`
- Windows or Linux
- NVIDIA GPU and a CUDA 12-compatible driver only if using the optional GPU requirements

---

## Installation

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the virtual environment

#### Windows PowerShell

```powershell
venv\Scripts\Activate.ps1
```

#### Linux

```bash
source venv/bin/activate
```

After activation, the terminal should show the virtual environment name, for example:

```text
(venv)
```

### 3. Install the standard dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Optional: install CUDA 12 GPU dependencies

Only install this file if GPU acceleration is required:

```bash
pip install -r requirements_gpu_cuda12.txt
```

If no compatible NVIDIA GPU / CUDA environment is available, skip this step.

---

## Run

### HDXF Viewer

```bash
python hdxf_viewer
```

### HDF5 / HDXF Structure Explorer

```bash
python hdf5_hdxf_structure_explorer.py
```

---

## HDF5 / HDXF Structure Explorer

The structure explorer is intended for inspecting a complete detector dataset consisting of a Master HDF5 file and its related external Data HDF5 files, and for comparing that structure with an HDXF archive.

### HDF5 view

Open the folder containing one complete HDF5 dataset.

The explorer can display:

- Master HDF5 structure
- External Data HDF5 files
- Merged logical HDF5 structure
- Groups
- Datasets
- Attributes
- Soft links
- External links
- Dataset shape and dtype
- Chunk layout
- Compression / HDF5 filter information
- Logical and physical storage size
- Dataset occurrences across physical HDF5 files

Typical detector paths may include:

```text
/entry/instrument/detector
/entry/instrument/beam
/entry/instrument/detector/calibration
/entry/detector
/entry/image
/entry/roi
/entry/xfel
```

### HDXF view

Open an `.hdxf` archive to inspect:

- HDXF overview
- Logical HDF5 object graph
- Archived objects grouped by original source file
- Legacy HDF5 reconstruction information
- Detector archive information
- Payload/archive members
- `manifest.json`

The HDF5 and HDXF trees can be displayed side by side for direct comparison.

### Cross-highlighting

Selecting a mapped HDF5 object can highlight its corresponding HDXF object, and vice versa.

For physical HDF5 files, matching uses both:

```text
source file + HDF5 path
```

This prevents objects with the same local path in different Data HDF5 files from being treated as the same physical object.

---

## GPU Support

The standard environment is installed from:

```text
requirements.txt
```

CUDA 12 support is installed separately from:

```text
requirements_gpu_cuda12.txt
```

This keeps the base environment usable on systems without an NVIDIA GPU.

GPU acceleration may be used by HDXF processing paths when the required CUDA/CuPy environment is available.

---

## Recommended Project Layout

```text
project/
├── hdxf_viewer
├── hdf5_hdxf_structure_explorer.py
├── requirements.txt
├── requirements_gpu_cuda12.txt
├── README.md
└── venv/
```

If the project contains additional converter or codec modules, keep them in the package/repository structure expected by the viewer.

---

## Quick Start

### Windows

```powershell
python -m venv venv
venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

# Optional CUDA 12 support
pip install -r requirements_gpu_cuda12.txt

python hdxf_viewer
```

To open the structure explorer:

```powershell
python hdf5_hdxf_structure_explorer.py
```

### Linux

```bash
python -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

# Optional CUDA 12 support
pip install -r requirements_gpu_cuda12.txt

python hdxf_viewer
```

To open the structure explorer:

```bash
python hdf5_hdxf_structure_explorer.py
```

---

## Troubleshooting

### PowerShell blocks virtual-environment activation

If Windows PowerShell refuses to run `Activate.ps1`, you may need to adjust the execution policy for the current user.

### GPU packages fail to install

Check that:

- An NVIDIA GPU is available.
- The installed NVIDIA driver is compatible with CUDA 12.
- You are using the intended Python virtual environment.
- The base requirements were installed successfully first.

You can always use the CPU-only environment by installing only:

```bash
pip install -r requirements.txt
```

### Confirm the active Python environment

```bash
python --version
python -m pip --version
```

Both commands should point to the virtual environment after activation.

---

## Notes

- Keep the Master HDF5 file and all related external Data HDF5 files together when using the structure explorer.
- The structure explorer is designed to inspect HDF5 metadata and structure without automatically reading large detector datasets.
- Filtered detector datasets should only be decoded when required, especially when native HDF5 filters such as Bitshuffle/LZ4 are involved.
- HDXF is intended to preserve the information required for exact reconstruction while providing a more compact detector-oriented archive representation.

---

## License

Add the project license here if required.
