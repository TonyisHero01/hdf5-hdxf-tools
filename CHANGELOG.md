## [Unreleased] — pyFAI geometry-corrected standard 1D (2026-10-21)

- Fixed a scientific mismatch in standard pyFAI 1D: the former `r_mm → θ = ½atan(R/L)` conversion assumes a detector perpendicular to the beam and is not reliable under tilt. New corrected 1D axes bin natively in pyFAI `2th_rad`, `2th_deg` or `q_A^-1`, using dist/PONI/rot1/rot2/rot3; chart and CSV labels follow the selected axis.
- Preserved the historical radius-based 1D as an explicit **Legacy** selection for zero-tilt comparisons. It refuses nonzero rotations; even zero-tilt angular and radius binning need not yield identical output bins.
- Added **Geometry Preview (2θ / Δ2θ)** showing per-pixel pyFAI angular coordinates and deviation from zero rotation; preview does not resample detector image pixels.
- Clarified that editable PONI X/Y differ from actual beam-center pixel positions under tilt. Existing `.poni` Load/Save and existing 2D Cake geometry remain available.
- Changing geometry/integration fields invalidates previous 1D/2D results, including results computed with stale geometry during a worker run. Reload View and switching files reset tilt rather than silently reusing another detector's geometry.
- Added `hdxf_geometry.py`, `GEOMETRY_CORRECTION.md` and geometry/Tk regression tests. No change to HDXF archive, codecs, source detector pixels, or Notebook Reference 1D algorithm. **Real pyFAI and Windows hardware validation remains necessary**.

## [Unreleased] — Sum Normalization (2026-10-21)

- Added ratio-of-sums I0-monitor and direct-beam-ROI modes to HDF5/HDXF Sum scripts and the Viewer processing panel: `sum(C_i)/sum(I0_i)` and `sum(C_i)/sum(beam_ROI_i)`. No change to existing sum, mean or exposure-rate math.
- I0 readings must come from an explicit HDF5 Dataset path (scalar/per-frame, including master/external data lookup); missing, invalid and non-positive readings abort safely.
- Direct-beam normalization requires an explicit rectangular ROI and a known positive saturation threshold; saturated ROIs fail rather than bias the reference. Optional .npy bool exclusion mask. Matching Viewer Manual + applied Gap masks transfer to the processing job, but preview layers and unrelated sources do not.
- Derived HDF5 frame Dataset stores provenance attributes; HDXF manifest records the mode/reference/denominator. Float64 is selected automatically for new modes. Includes numerical unit/integration tests.
- **Compatibility:** Legacy `hdxf_sum.py` still does not accept HDXF 0.4.8 archives (input allowed through 0.4.5); no unverified archive version change. See `SUM_NORMALIZATION.md`.

## Module Gap image-preview reliability fix (2026-10-21)

- Fixed overly restrictive image-stripe thresholds: candidates now require **locally contrasting detector occupancy on both sides**, allowing sparse photon-counting backgrounds and occasional nonzero pixels inside otherwise dark bands. Metadata and manual-row modes remain unchanged.
- Every Preview click has an explicit scan/result status; failures now show a warning dialog in addition to status text, including unexpected exceptions (which are also logged to the console).
- Apply remains unavailable without a valid preview. No automatic Mask application; all generated image-based bands still require visual verification. No source HDF5/HDXF data or codec changes.
- Added noisy sparse detector regression and physical Tk button invocation tests.

## Module Gap Mask: visible preview diagnostics and image-stripe suggestion mode (2026-10-21)

- Improved Preview feedback: missing/invalid NXdetector_module metadata now reports NO PREVIEW beside the detection selector and in the window status bar, rather than only below off-screen buttons. Feedback is retained through sidebar refresh and cleared on mode/source change.
- Added opt-in **Image stripes (heuristic preview)** to detect long internal near-zero/sentinel module-like rows/columns when explicit detector geometry is unavailable. The proposal is non-destructive and preview-only; users must visually validate it and click Apply.
- Added conservative tests for synthetic gaps, sparse/blank frames and GUI no-metadata feedback. No archive/codec changes.

# Changelog

Documented changes to the **software repository**, not automatic changes to the HDXF on-disk archive format. Historical release entries are omitted because this source snapshot does not contain enough verified release notes to reconstruct them reliably.

## [Unreleased] — NeXus detector module gap masking (2026-10-21)

### Added

- `TOOLS → Detector Module Gap Mask`: conservative detection from explicit multi-module `NXdetector_module` data-origin/data-size geometry, or manual zero-based inclusive row/column ranges with optional 0–32 px padding. No pixel-intensity heuristics or detector-specific hardcoded gap coordinates.
- Separate amber preview and teal **applied** gap-mask layer; pink manual User Mask remains independent. Clear Gap Mask does not remove manual masks; reopening a source clears both gap and manual masks.
- Applied gap masks are unioned with manual/optional source `pixel_mask` in pyFAI 1D/2D and Notebook Reference. Notebook before/after manually masked comparison retains applied gaps in both baselines.
- Updated integration mask counts and invalidation when applied gap masks change, with non-destructive full-resolution image-overlay export support.
- New `hdxf_module_gaps.py`, `MODULE_GAP_MASK.md`, synthetic NeXus HDF5 and Tk regression tests. No HDXF archive-format change.

### Cautions

- Metadata inference only works when multiple complete `NXdetector_module` pixel slabs are available; source `pixel_mask` alone does not specify physical module gaps. The Viewer does not infer gaps from all-zero image regions.
- Notebook Reference has separate legacy configurable gap positions; verify those independently.
- Windows and real detector-data behavior remains to be checked in the user's environment.

## [Unreleased] — pyFAI/Notebook integration chart export (2026-10-20)

### Added

- Separate **Save Chart…** action in the pyFAI sidebar and Integration Result window; existing **Export Data…** still exports numerical CSV (1D/Notebook) and NPZ (2D).
- Export 1D all-frame pyFAI curves, 2D Cake intensity maps and Notebook Reference charts to high-resolution PNG/JPEG or SVG/PDF. The exports are rendered from stored integration arrays, not viewport screenshots.
- Notebook before/after, signed Difference Δ, and original three-curve views are exported using the selected comparison method and exactly the same center-exclusion and NaN-gap handling as the displayed result.
- New noninteractive `hdxf_plot_export.py` Matplotlib renderer; atomic output replacement after successful encoding and error handling without modifying original HDF5/HDXF files.

### Notes

- Scientific values should be preserved using the **Export Data…** CSV/NPZ action. Image output is a visualization, never a replacement for integrated numeric arrays.
- 2D Cake uses the same grayscale percentile/log-display transfer as the Viewer, and the visual image retains source row 0 at the top. The Viewer 2D endpoint azimuth labels were corrected to match row order (no matrix or integration values changed).
- Matplotlib is already listed in the main `requirements.txt` and is imported on demand for chart export.

## [Unreleased] — detector image export (2026-10-20)

### Added

- TOOLS → Export Detector Frames with current frame and 1-based inclusive START/END/STEP range export.
- Display PNG and high-quality JPEG at full detector-frame resolution using the same 8-bit grayscale contrast/invert controls as Viewer; optional ROI/Mask/beam center/resolution ring/bright-spot annotations are rendered in source pixel coordinates.
- Native-dtype lossless `.npy` and uint16-only lossless 16-bit PNG without annotations. Non-uint16 source images are rejected for the 16-bit PNG option.
- Cancelable incremental batch processing (one frame per Tk event-loop step), destination collision protection and atomic single-frame file replacement after successful encoding.
- Export of temporary current-frame live edits without writing source archives, plus unit and GUI tests.

### Scope and limitations

- Display PNG is lossless **only for its 8-bit rendering**, not for the original scientific pixel values. JPEG is lossy. Zoom/pan do not crop the full-frame export.
- Exports are snapshot-based: batch uses the contrast/mask/geometry parameters captured when started; dynamic bright-spot recognition uses those settings on each frame. Source detector `pixel_mask` is not itself painted unless manually selected with the user Mask tool.
- No HDXF archive format or codec changes.

## [Unreleased] — Notebook manual-mask before/after comparison (2026-10-20)

### Added

- Notebook 1D Reference runs a **paired reduction** when the active View has manually masked pixels: before (the same source detector mask if enabled, but no manual mask) and after (source mask plus manual mask).
- The averaged source image is prepared **once**. Beam center and geometry are held fixed for both reductions; the Notebook beam fit is not rerun on the second profile.
- The integration result window provides **Before / After**, **Difference Δ = After − Before**, and the original **Notebook 3 curves** views. The comparison method selector supports Spots included, Hot/Bragg spots masked, and Ring MAD σ-clipped.
- On-screen diagnostics report manual pixel count, newly excluded pixels beyond Notebook's built-in/source base mask, number of changed visible radial bins, bins becoming empty, max |Δ| and mean |Δ|.
- Notebook CSV export appends before/after, signed delta, and valid pixel counts for each of the three original profiles; the original columns remain intact.
- Empty radial bins export **NaN**, not a fabricated zero difference, and radial coordinates are checked for comparability.

### Scope

- Comparison is available for **Notebook Reference 1D** with manual masks; traditional pyFAI 1D/2D still use the existing result workflow.
- Comparing two full Notebook reductions takes more CPU time than one reduction, but does not read or average the HDF5/HDXF image stack twice.
- The chart respects `Exclude Center Radius`; a mask affecting only excluded central pixels may legitimately give zero changed visible bins.
- No changes to source pixels, on-disk HDXF format, codec, or ordinary Notebook reduction formulas.

## [Unreleased] — pyFAI mask-result transparency and stale-result guard (2026-10-20)

### Fixed

- Mask changes now invalidate old 1D / 2D / Notebook integration results immediately. The old result window is closed and its export action is disabled until integration is run again.
- Background integrations capture a mask-revision number. If user/source mask choices change while processing, the completed result is rejected rather than silently displaying outdated science.
- Result status includes actual manually excluded pixel count, source `pixel_mask` count, and combined excluded pixel count (or Notebook extra-mask count).
- Renamed the optional Notebook checkbox to explicitly refer to the source HDF5/HDXF `pixel_mask`; manual right-click masks apply separately, with or without that option checked.

### Scope

- This does not alter the pyFAI mask convention, source data, archive format, or compression algorithms.
- Standard 1D integrates every frame and averages the 1D outputs; masking very few pixels may yield an almost invisible difference. Re-running integration is necessary and can take time.
- Manual masks are Viewer-only; automatic beam-center fitting does not automatically exclude them.

## [Unreleased] — local bright-spot highlighting (2026-10-20)

### Added

- Right-side **Tools → Bright Spot Detection** option to outline locally high-intensity pixel clusters with thin (1 px) red rectangles.
- Adjustable detection sensitivity (robust residual noise multiplier; default `6`) and minimum region size (default `2` pixels, set `1` for isolated hot pixels).
- Background-worker detection with a maximum of 250 highlighted regions per frame, refreshed on frame changes, input changes and manual mask edits.
- Independent `hdxf_bright_spots.py` detector module and automated algorithm/Tk overlay tests.

### Behavior

- Red rectangles are **visual annotations only**; they do **not** change the source frame, add any Mask bits, or affect pyFAI / Notebook integration.
- Negative/nonfinite pixels and currently user-masked pixels are excluded from detection. Local bright-feature detection is heuristic, not crystallographic Bragg-peak identification.
- This GUI-only change does not alter the HDXF archive or codec formats.

## [Unreleased] — higher-contrast right-click menu (2026-10-20)

### Changed

- Styled the detector pixel right-click menu with plain white text on a deep-navy background, a readable 11-point font, and a contrasting selection highlight. No mask, beam-center, or ROI gestures changed.

## [Unreleased] — right-click beam center and analysis masks (2026-10-20)

### Added

- Context menu on stationary right-click in the detector canvas: set or clear the Viewer-only beam center, mask/unmask the clicked pixel, mask/unmask the existing ROI, or clear all manual masks.
- Pink/red non-destructive mask overlay with per-view mask state retained across detector frames and cleared on source changes.
- Manual mask integration into pyFAI 1D/2D and Notebook-reference calculations, independently of the optional source detector mask toggle.

### Changed

- Retired Ctrl+left-click beam-center placement and implicit beam-center clearing on right-click; actions are explicit menu choices.
- Preserved right-button ROI drawing, Polygon vertex selection, and right-drag pan in ordinary mode.
- Retained left-double-click pixel-value editing and scientific source data unchanged.

### Scope

- User masks currently exist only in Viewer memory: no mask serialization or automatic detector-specific gap recognition is included.
- This GUI update does not modify HDXF archives, compression, reconstruction, or file-format versions.

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
- At the original first-stage maintenance snapshot, dedicated detector-tilt geometry correction, image export, detector-independent automatic gap masks and direct-beam intensity normalization were not included. PNG/JPEG/NPY export was subsequently added; the other features remain outstanding.

### Compatibility and data integrity

- The main converter's archive format version, binary block encoding, decoding algorithms, and HDF5 restoration algorithms were **not changed** by this documentation/CLI update.
- Added the correctly spelled argument while retaining existing arithmetic semantics; see the follow-up cleanup for removal of the legacy alias.
