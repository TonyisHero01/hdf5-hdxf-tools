# Detector Module Gap Mask — HDXF Viewer

## Purpose

The Viewer can preview and apply a **non-destructive module-gap mask** in `TOOLS → Detector Module Gap Mask`. The applied mask is separate from manually masked pixels and from the source HDF5/HDXF `pixel_mask` metadata. **True means excluded**; source image values and the HDXF archive format are never changed.

## Automatic mode: explicit NeXus module geometry

Select **NeXus module metadata** and click **Preview Gap Mask**. The software looks only for explicit `NXdetector_module`-style groups directly under `/entry/instrument/detector`, such as `module`, `module_0`, `module_1`, etc. Each of at least **two** modules must supply `data_origin` and `data_size` as two integer values.

The NeXus indexing order is **slow-to-fast, i.e. `[row, column]`** for 2-D frames. The software creates module coverage rectangles, then marks uncovered pixels **only inside the bounding rectangle of the modules**. It rejects invalid, overlapping, incomplete, or gap-free rectangles. This conservative method does not try to infer module gaps from brightness, zero counts, `pixel_mask` flags, detector names, or guessed Jungfrau spacing. If module-geometry metadata is missing or not stored at the expected paths, the Viewer reports **NO PREVIEW** immediately below the detection selector and in the main window status bar. Use image heuristic preview or Manual mode instead.

Reference: [NeXus NXdetector_module definition](https://nexus-definitions.readthedocs.io/en/latest/classes/base_classes/NXdetector_module.html).

## Image stripes (heuristic preview): for files without module geometry

Choose **Image stripes (heuristic preview)** if the detector visibly contains dark horizontal or vertical module boundaries but NeXus metadata mode says no geometry is available. Click **Preview Gap Mask**. This examines the **currently displayed frame only** for continuous **internal** bands with substantially lower nonzero-pixel occupancy than the active detector rows/columns on BOTH sides. Isolated nonzero counts inside a black band are tolerated. Only runs at least 3 pixels wide, at least 12 pixels inside the frame with non-empty sensor regions on **both** sides are proposed; suspiciously large dark regions and near-empty frames are rejected.

**Not a guaranteed detector module identification.** Dark strips could also be physical shadows, shutter artifacts, sensor faults, or background effects. Not all real gaps are exactly zero. The automatic image mode never directly applies a mask; inspect the **amber overlay** before clicking Apply. A detection failure does not mean there are no detector gaps. If the detector strips are not detected cleanly, use the manual row/column method. **NO PREVIEW** is shown in a warning dialog as well as in the tool card and status bar, so a failed click is never silent. The heuristic intentionally does not guess arbitrary zero pixels to avoid confusing low photon counts with detector gaps.

## Manual mode: rows and columns

Select **Manual rows / columns**; enter comma-separated, **zero-based inclusive** pixel coordinates:

- Gap rows `500:534, 1100:1130` masks rows 500–534 and 1100–1130 across the full frame width.
- Gap columns `255:258, 514` masks columns 255–258 and 514 across the full frame height.
- Leave either field empty if unused; supply at least one field. `x:y` and `x-y` are equivalent; `x` masks one row/column.
- **Gap Padding**, default 0, dilates the gap up to 32 pixels in both axes. Increasing padding intentionally excludes additional neighboring pixels; review the preview before applying.

Manual ranges are not detector defaults. Measure the exact row/column boundaries for your detector first. **There is no global automatic assumption that rows 500 and 1100 are module gaps.**

## Meaning of colors / action buttons

- **Amber**: uncommitted preview; has **zero effect** on scientific integration.
- **Teal**: applied Module Gap Mask. It is combined with manual Mask and, when selected, the source detector mask for pyFAI 1D/2D and Notebook Reference.
- **Pink**: right-click manual Mask; maintained separately.
- **Apply Preview to Integration**: replaces the applied gap layer with this preview. If pyFAI/Notebook previously produced a result, that old result is invalidated; press Run again.
- **Clear Gap Mask / Preview**: removes this layer while **preserving every manual pixel/ROI mask**.
- Right-click **Clear All User Masks** clears only the manual layer while leaving the applied Gap Mask untouched.

The mask applies to all frames within the same open View. Switching source files resets both the applied Gap Mask and its preview. Additional View panes manage their masks separately; there is no export/persistence for these session-only masks yet.

## Scientific interpretation

The pyFAI checkbox **Source HDF5/HDXF pixel_mask** is independent of this feature. Disabling that checkbox leaves manually applied masks **and applied Gap Mask** active. `pixel_mask` flags usually mean detector-invalid pixels, which are not necessarily physical module gaps. The final mask is a **union**, so individual pixel counts may overlap.

Notebook Reference also has its own historical `gap_positions` row exclusion, which is **not** automatically changed or disabled by this new feature. To avoid applying an inappropriate historical gap pattern to a new detector, check the Notebook Gap Positions setting separately. Masked pixels have no effect on output bins that are already excluded by the Notebook's center radius, negative-pixel or hotspot rules.

## Validation and limits

Tests cover synthetic NeXus/HDF5 module geometry, manual range validation, padding, applied/preview union semantics, pyFAI 1D/2D mask construction, GUI Apply/Clear, and existing Viewer regressions. Live Windows GUI and real beamline detector data require local verification. Image-stripe detection only proposes heuristic candidates from frame occupancy; it does not verify actual module geometry and must be manually confirmed.

## Troubleshooting: Preview Gap Mask seems to do nothing

- Check the **amber feedback directly below Detection Method** and the bottom application status bar: they show the error or candidate count.
- `NeXus module metadata` does **not** use the Gap Rows/Columns input fields and does **not** inspect image brightness. If your HDF5 has no complete `NXdetector_module` geometry, use **Image stripes (heuristic preview)** or **Manual rows / columns**.
- Clear any overlarge `Gap Padding`; start from `0`, verify preview, then expand only if justified.
- **Apply Preview to Integration** can only be used after a successful preview.
