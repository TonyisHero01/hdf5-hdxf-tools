# HDF5 and HDXF Universal Tools

A collection of utilities for working with **HDF5** and **HDXF** scientific detector data.

The project includes:

- **HDXF Viewer** — a desktop GUI for opening, visualizing, and analyzing HDF5 and HDXF detector frames.
- **HDF5 / HDXF Converters** — lossless conversion between HDF5 and HDXF formats.
- **HDF5 / HDXF Structure Explorer** — inspection and comparison of HDF5 and HDXF internal structures.
- **Scientific Analysis** — ROI tools, pixel operations, frame summation, subtraction, and pyFAI 1D/2D integration.
- **Hardware Auto-Tuning** — adaptive CPU and memory management.
- **Optional NVIDIA CUDA 12 Acceleration** — GPU-accelerated processing for supported operations.

---

## Requirements

- Python 3.13 recommended for the current pinned dependencies
- pip
- Windows 11 or a compatible Linux distribution
- Tkinter for the desktop GUI
- NVIDIA GPU with a compatible driver for optional CUDA acceleration

On Windows, Tkinter is normally included with the standard Python installer.

On Ubuntu, install the system packages:

```bash
sudo apt update
sudo apt install python3-venv python3-tk
```

---

## Installation

### 1. Create a virtual environment

**Windows PowerShell**

```powershell
py -3.13 -m venv venv
```

**Linux**

```bash
python3 -m venv venv
```

### 2. Activate the virtual environment

**Windows PowerShell**

```powershell
.\venv\Scripts\Activate.ps1
```

**Linux**

```bash
source venv/bin/activate
```

After activation, the terminal should display the environment name:

```text
(venv)
```

### 3. Install dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Verify the installation:

```bash
python -m pip check
```

**Important:** Always activate the virtual environment before installing dependencies.

Alternatively, use the virtual environment's Python executable directly to ensure packages are installed in the correct environment.

---

## GPU Support (Optional)

HDXF supports optional NVIDIA CUDA acceleration through CuPy.

GPU acceleration is available for supported processing operations, including CUDA-based frame transformations during conversion and restoration.

### Requirements

- A compatible NVIDIA GPU
- An NVIDIA driver compatible with CUDA 12
- The CUDA 12 Python dependencies
- Sufficient available GPU memory

### Installation

With the virtual environment activated, run:

```bash
python -m pip install -r requirements_gpu_cuda12.txt
```

This requirements file includes the standard project dependencies and the CUDA 12 CuPy package.

### Verify GPU availability

```bash
nvidia-smi
```

Verify CUDA access from Python:

```bash
python -c "import cupy as cp; print('CUDA devices:', cp.cuda.runtime.getDeviceCount())"
```

GPU support is optional. HDXF can run with its CPU backend without a compatible NVIDIA GPU.

---

## Run the Applications

### HDXF Viewer (GUI)

Launch the desktop Viewer:

```bash
python hdxf_viewer.py
```

The graphical interface opens in a desktop window.

The Viewer supports:

- Opening HDF5 and HDXF detector files
- Frame navigation and visualization
- Zooming, panning, and pixel inspection
- Multiple synchronized views
- Beam-center selection and resolution rings
- ROI creation and editing
- Histogram and contrast controls
- pyFAI 1D and 2D integration
- Notebook-reference radial analysis
- Detector frame processing
- HDF5 ↔ HDXF conversion tools

A graphical desktop environment is required to display the GUI.

### HDF5 / HDXF Structure Explorer

If the separate Structure Explorer script is included in your installation, launch it with:

```bash
python hdf5_hdxf_structure_explorer.py
```

---

## HDF5 / HDXF Structure Explorer

The Structure Explorer is designed to inspect and compare the organization of scientific detector datasets.

A typical HDF5 detector dataset consists of a Master HDF5 file and one or more external Data HDF5 files.

### HDF5 View

The explorer can display:

- Master HDF5 structure
- External Data HDF5 files
- Merged logical HDF5 structure
- Groups and datasets
- Dataset attributes
- Soft links and external links
- Dataset shapes and data types
- Chunk layout
- Compression and HDF5 filters
- Logical and physical storage sizes
- Dataset occurrences across physical HDF5 files

Typical detector paths include:

```text
/entry/instrument/detector
/entry/instrument/beam
/entry/instrument/detector/calibration
/entry/detector
/entry/image
/entry/roi
/entry/xfel
```

### HDXF View

The explorer can inspect:

- HDXF overview
- Logical HDF5 object graph
- Archived objects grouped by original source file
- Legacy HDF5 reconstruction information
- Detector frame archives
- Payload and archive members
- `manifest.json`

HDF5 and HDXF structures can be displayed side by side.

### Cross-Highlighting

Selecting a mapped object in one view can highlight the corresponding object in the other view.

For physical HDF5 files, matching uses:

```text
source file + HDF5 path
```

This prevents objects with identical internal paths in different Data HDF5 files from being incorrectly treated as the same physical object.

---

## Recommended Project Layout

```text
hdf5-hdxf-tools/
│
├── hdxf_viewer.py
├── hdxf_codec.py
├── hdxf_index.py
├── hdxf_hardware.py
├── hdxf_notebook_reference.py
│
├── hdf5_to_hdxf.py
├── hdxf_to_hdf5.py
├── hdf5_processing_tree.py
│
├── hdxf_sum.py
├── hdxf_subtract.py
├── hdxf_pixelop.py
│
├── hdf5_hdxf_structure_explorer.py
│
├── requirements.txt
├── requirements_gpu_cuda12.txt
├── README.md
│
└── venv/
```

Some tools are optional and may be distributed separately.

Keep shared Python modules in the locations expected by the application imports.

---

## Quick Start

### Windows PowerShell

```powershell
# Create virtual environment
py -3.13 -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Launch HDXF Viewer GUI
python hdxf_viewer.py
```

### Ubuntu / Linux

```bash
# Install system dependencies
sudo apt update
sudo apt install python3-venv python3-tk

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Launch HDXF Viewer GUI
python hdxf_viewer.py
```

For GPU-enabled installation, refer to the **GPU Support** section above.

---

## Troubleshooting

### PowerShell blocks virtual-environment activation

If PowerShell refuses to run `Activate.ps1`, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

This changes the execution policy only for the current PowerShell session.

### ModuleNotFoundError

If a required module such as NumPy cannot be found, verify the active Python environment:

```bash
python -c "import sys; print(sys.executable)"
python -m pip --version
```

Both should point to your virtual environment.

Reinstall project dependencies if necessary:

```bash
python -m pip install -r requirements.txt
```

### HDF5 filter errors

Some detector datasets require external HDF5 compression filters, including Bitshuffle/LZ4.

Make sure `hdf5plugin` is installed:

```bash
python -m pip show hdf5plugin
```

### Verify installed dependencies

```bash
python -m pip check
```

### Close the virtual environment

```bash
deactivate
```

---

## Notes

- Keep Master HDF5 files and their associated external Data HDF5 files accessible when working with complete detector datasets.
- HDXF is designed for lossless scientific detector-data archiving.
- The Viewer and converters use shared HDXF codec and index modules.
- Filtered detector datasets should only be decoded when necessary.
- GPU acceleration availability depends on hardware, drivers, and the selected processing operation.
- Exact scientific data preservation and reconstruction are important design goals of the project.

---

## License

Add the project license here.