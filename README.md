# HDF5 and HDXF Universal Tools

Desktop viewing, lossless conversion, and frame processing for scientific HDF5/NeXus detector data and **HDXF** archives.

**Current software-suite version:** `0.4.8` (source snapshot; see [VERSIONS.md](VERSIONS.md)). This is **not** a claim that every auxiliary tool writes or accepts archive format `0.4.8`.

## I0 / direct-beam Sum Normalization

Use **HDF5 / HDXF Sum / Mean** and choose `i0-normalized` or `beam-normalized`.
The first requires a real absolute HDF5 monitor Dataset path; the second
requires a rectangular direct-beam ROI and known saturation threshold.
All selected source frames are summed and divided by the **sum of corresponding
monitor readings**. Auto output dtype is float64; only committed Manual/Gap
masks from the matching Viewer source are forwarded to beam-ROI processing.
See [SUM_NORMALIZATION.md](SUM_NORMALIZATION.md) for the formula, usage,
scientific caveats and HDXF 0.4.8 Sum-backend compatibility limitation.

## Automatic Module Gap Mask (Viewer → TOOLS)

Use **NeXus module metadata** for explicit `NXdetector_module` `data_origin`/`data_size` rectangles, or enter **Manual rows / columns** with zero-based, inclusive detector coordinates. Click **Preview Gap Mask** (amber) then **Apply Preview to Integration** (teal). **Clear Gap Mask** does not remove your pink right-click manual Mask. Applied gaps exclude pixels from pyFAI 1D/2D and Notebook Reference while leaving original detector values unchanged. Zero-valued pixels are never automatically identified as gaps. See [MODULE_GAP_MASK.md](MODULE_GAP_MASK.md) for limitations, examples and scientific cautions.

## Included in this repository

| Tool | File | What it does |
| --- | --- | --- |
| HDXF Viewer | `hdxf_viewer.py` | Desktop GUI for HDF5/HDXF frame navigation, pixel inspection, ROI tools, beam center, resolution rings, visual bright-spot highlighting, PNG/JPEG/NPY detector-frame export, pyFAI 1D/2D integration, and Notebook-reference radial calculation |
| HDF5 → HDXF | `hdf5_to_hdxf.py` | Archive detector data and metadata in HDXF |
| HDXF → HDF5 | `hdxf_to_hdf5.py` | Reconstruct the HDF5 master and associated data files |
| HDXF operations | `hdxf_sum.py`, `hdxf_subtract.py`, `hdxf_pixelop.py` | Sum/mean/exposure/I0/direct-beam-normalized output, filtering/subtraction, and scalar arithmetic |
| HDF5 operations | `hdf5_processing_tree.py` | HDF5 frame processing while retaining the relevant HDF5 structure |
| Core libraries | `hdxf_codec.py`, `hdxf_index.py`, `hdxf_hardware.py`, `hdxf_notebook_reference.py`, `hdxf_mask_comparison.py`, `hdxf_bright_spots.py`, `hdxf_image_export.py`, `hdxf_normalization.py`, `hdxf_release.py` | Codec, indexing, hardware guidance, Notebook reference, and release metadata |

**Not included:** There is **no separate `hdf5_hdxf_structure_explorer.py`** in the current repository. The Viewer has its own metadata/data browsing functions; an independent Structure Explorer is not part of this distribution.

**Scope of supported features:** The Viewer has a PONI/pyFAI geometry interface, but a dedicated detector-tilt correction workflow is not yet released. The Notebook-reference mode provides configurable gap-row exclusions; do not treat them as a validated automatic module-gap mask for every detector. A generalized geometry-correction workflow and per-frame beam-flux correction beyond the released ratio-of-sums remain planned. Detector-frame PNG/JPEG/NPY export is available under Viewer → TOOLS.

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

## Export detector frames (TOOLS)

Open an HDF5 or HDXF detector view, choose the **TOOLS** tab, then find **Export Detector Frames**:

1. Choose **PNG (display, 8-bit)**, **JPEG (display, 8-bit)**, **NPY (pixel values, lossless)**, or **PNG 16-bit (uint16 only)**.
2. For visual PNG/JPEG, optionally check **Include Viewer overlays** to render the selected ROI (blue fill/green outline), manual Mask (pink), beam-center marker, resolution rings, and enabled bright-spot boxes.
3. Click **Export Current Frame…** to save one image, or enter **START / END / STEP** (1-based inclusive image numbers) and click **Export Frame Range…** to choose a destination folder. A progress window lets you cancel batch export.

**Scientific integrity:** Display PNG/JPEG export uses the currently selected grayscale minimum, maximum and inversion, at the native full-detector resolution. The Viewer viewport pan/zoom is NOT exported; overlay shapes are projected into detector coordinates. Both visual modes reduce raw data to 8-bit; JPEG also introduces lossy compression. To preserve original numeric pixel values and NumPy dtype, choose **NPY**. For a detector frame whose dtype is exactly `uint16`, **PNG 16-bit** also preserves its 16-bit intensity array. Overlays are disabled for NPY and 16-bit PNG. Non-`uint16` inputs are rejected by the 16-bit PNG option rather than silently clipped/cast.

These exports reflect temporary Viewer live pixel edits, if any, but never modify the original HDF5/HDXF archive. Manual Mask changes are visualization/analysis-only and are not baked into scientific pixel-value exports. Batch export reads one frame at a time, preserves the current selection/zoom, and never overwrites an existing file in the selected batch destination. Filename pattern: `<source>_frame_000001.png` (or `.jpg` / `.npy`).

## Export pyFAI / Notebook integration charts

After computing a **pyFAI 1D all-frame average**, **pyFAI 2D Cake**, or **Notebook Reference 1D** integration, click **Save Chart…** in the pyFAI sidebar or at the top of the **Integration Result** window. Supported formats: **PNG**, **JPEG**, **SVG**, and **PDF**. PNG/JPEG are rendered at 180 DPI; SVG/PDF are useful for document and publication workflows. Saving a chart does not require a visible or unobstructed image window.

The **Export Data…** button remains separate: it continues to write **CSV** for 1D and Notebook profiles, and compressed **NPZ** for 2D Cake. These files retain integration values; chart images are for presentation only, and 2D export applies the *display* contrast transform.

For Notebook mask comparison, the saved chart respects the currently selected **Before / After**, **Difference Δ**, or **Notebook 3 curves** view and the selected **Method**. Bins excluded by the center radius or containing no valid pixels remain gaps, never zero-valued samples. On a 2D Cake image, the first data row remains at the top and the azimuthal coordinate labels follow the actual row sequence.

The chart renderer uses the already-required Matplotlib package; no new dependency needs to be installed if `requirements.txt` was installed successfully. All exports operate on the already-computed results and do not modify detector files or trigger a new integration.

## Right-click beam center and manual analysis masks

In the HDXF Viewer, **right-click once on a detector pixel** (outside ROI drawing mode) to open its context menu:

- **Set Beam Center Here** / **Clear Beam Center** — adjust the Viewer-only beam center and resolution rings.
- **Mask Pixel** / **Unmask Pixel** — exclude or restore that pixel from scientific integrations.
- **Mask Inside ROI** / **Unmask Inside ROI** — when an ROI has been drawn, mask its interior (useful for module gaps).
- **Clear All User Masks** — reset the active view's manually marked pixels.

Manual masks appear as **pink/red-tinted pixels**, remain active when changing frames of the same opened dataset, and are **automatically applied** to pyFAI 1D, pyFAI 2D, and Notebook-reference integrations. Masks are not saved to the HDF5/HDXF archive, do not change pixel values, and reset when opening a different source in that View. After editing masks, **rerun the integration** to update an existing plot.

### Verify Notebook manual masks quantitatively

When using **1D Notebook Reference** with one or more manual masks, press **Run Notebook Reference 1D**. The result window offers **Mask comparison** controls:

- **Before / After** overlays the same method's curve without and with manual masks, in blue and pink. **Before** still includes the file-provided detector `pixel_mask` if that checkbox is selected, plus Notebook's inherent filters.
- **Difference Δ** plots the signed difference `after − before` on a zero-centered axis. Choose which of the original three Notebook methods to compare with the **Method** selector.
- **Notebook 3 curves** switches back to the original three-curve chart (the after-mask result).
- Diagnostics show manual Mask pixels, newly excluded pixels beyond the Notebook base/source mask, changed radial bins **outside Exclude Center Radius**, bins which became empty, and maximum/mean absolute change.
- **Export…** writes the original Notebook columns and adds before/after/Δ plus corresponding valid-pixel counts for every method to the same CSV.

The Viewer reuses one averaged input image and fixes the same Beam Center for both calculations. Computing two Notebook reductions does add CPU processing time. If the new Mask covers only the excluded beam-center radius (default 60 px), or pixels that Notebook already ignores, the plotted change can correctly be zero. Comparison is currently implemented for Notebook Reference 1D, **not** standard pyFAI 1D/2D. Empty bins export as NaN in comparison columns (do not interpret them as intensity zero).


**Existing controls:** Double-left-click still edits a pixel value; left drag pans; right drag in ROI-drawing mode creates shapes and right clicks add Polygon vertices. To open the mask menu while drawing an ROI, finish or exit the ROI drawing mode first. There is no separate Ctrl+left-click beam-center shortcut.

## Bright Spot Detection (visual highlight)

Open a detector frame in the Viewer, select **TOOLS** in the right sidebar, and enable **Show bright spots (red boxes)**. The current frame is analyzed in a background worker; connected bright regions are outlined with **thin red rectangles**. Boxes move correctly with pan/zoom and are recalculated when the frame or manual masks change.

- **Sensitivity (σ)**: Robust local-background contrast multiplier, default `6`. Lower values find more candidates (including false positives), higher values keep only stronger features. Valid range: `1`–`50`.
- **Min Spot Pixels**: Smallest connected bright region to display; default `2`. Set `1` to detect single hot pixels.
- At most **250** highest-scoring regions are outlined per frame to avoid hiding the detector image under excessive annotations.
- Manual user-masked pixels, negative detector gap codes, and nonfinite values are excluded. For scientifically validated module-gap handling, continue using the dedicated masks and instrument geometry.

**Red boxes are visualization only.** They do not modify pixel values, apply a mask, save annotations to HDXF/HDF5, or exclude detected regions from pyFAI / Notebook calculations. Use the separate right-click Mask action when you actually need to exclude pixels. Detection uses `scipy.ndimage`, supplied by the standard `requirements.txt`.

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

### Troubleshooting Module Gap Preview

If selecting **NeXus module metadata** and clicking Preview creates no overlay, read the highlighted error just below the detection method. That mode requires complete `NXdetector_module` geometry; black stripes alone do not meet the requirement. Try **Image stripes (heuristic preview)** for a full-row/column candidate based on local dark-band contrast, including sparse detector backgrounds (inspect before Apply), or **Manual rows / columns** with precise zero-based ranges. Unsuccessful Preview clicks show a warning dialog and disable Apply; successful previews report a pixel count. Neither preview option affects integration until applied. See `MODULE_GAP_MASK.md`.

### Detector geometry correction (standard pyFAI)

The PYFAI page provides editable PONI/distance/pixel size and `ROT1/ROT2/ROT3` angles (degrees), Load/Save `.poni`, and **Geometry Preview (2θ / Δ2θ)**. Standard 1D radial integration now defaults to **Corrected θ (rad)** and can instead produce 2θ (degrees) or q (Å⁻¹). The historical radius-derived 1D θ is still available as **Legacy** at zero tilt; native angular binning can produce different intensity bins than radius binning even with no tilt. pyFAI 2D Cake already accepts the editable rotations. Changes invalidate old results.

**Tilted PONI coordinates are not direct-beam coordinates.** Calibrate geometry before interpreting q/2θ. Notebook Reference remains a separate uncorrected algorithm. See `GEOMETRY_CORRECTION.md`.
