# Detector Geometry Correction — pyFAI 1D/2D

## What changed

The Viewer already had editable detector distance, pixel size, PONI, `rot1` / `rot2` / `rot3`, and PONI load/save. It already passed those rotations to **2D Cake** via pyFAI.

**The important correction is for standard 1D integration:** older code integrated in detector radius `r_mm` and subsequently applied `θ = 1/2 atan(R/L)` to the radius-bin centers. That is a **normal-incidence-only** conversion and is unsuitable for tilted detector geometry. The new standard 1D choices request pyFAI's own geometry-corrected `2th_rad`, `2th_deg`, or `q_A^-1` coordinate **before the intensity is binned**, and preserve the resulting axis in plots and CSV exports.

| Standard pyFAI 1D axis choice | pyFAI integration unit | Display / CSV axis |
| --- | --- | --- |
| Corrected θ (rad) — default | `2th_rad` | `θ = 2θ/2`, radians |
| Corrected 2θ (deg) | `2th_deg` | 2θ, degrees |
| Corrected q (Å⁻¹) | `q_A^-1` | q, inverse ångströms |
| Legacy θ = ½atan(R/L) (rad) | `r_mm` | Existing historical normal-incidence conversion |

The Legacy choice refuses nonzero rotations rather than presenting a misleading tilted-detector result. Even when all rotations are zero, **native angular binning and legacy radial binning can produce different binned intensities**: they do not define bins with identical boundaries. To reproduce legacy results, explicitly select Legacy.

## How to use

1. Open an HDF5 or HDXF detector image, then open the **PYFAI** page.
2. Enter **DISTANCE (M)**, **WAVELENGTH (Å)** and detector **PIXEL X/Y (M)**; ideally load a trustworthy calibrated `.poni` file using **Load .poni…**.
3. Check **PONI X/Y (PX)** and **ROT1/ROT2/ROT3 (°)**. They are editable; the Viewer converts angles to radians internally. **PONI means Point of Normal Incidence, not necessarily the direct-beam pixel for tilted detectors.** The right-click beam-center operation initializes PONI correctly only for an untilted detector.
4. Click **Geometry Preview (2θ / Δ2θ)…** to examine the actual per-pixel two-theta coordinate map and the **difference from the same PONI/distance with zero rotation**. The preview visualizes geometry only, not a transformed or resampled detector frame.
5. Select **1D radial integration** and choose a corrected 1D radial axis (recommended) or select **2D cake / azimuthal integration**. Set radial bins; press Run.
6. View / export the numerical result via **Export Data…** (CSV for 1D, NPZ for 2D) or use **Save Chart…**.

Changes to geometry and integration fields invalidate old results and prevent obsolete charts being exported. While long-running all-frame integration is in progress, a result completed against an older configuration is discarded.

**Reload View** clears PONI overrides and returns rotation angles to zero. Switching detector files also drops previous tilt settings; this prevents accidental carry-over. It does not infer a new rotation from data in the absence of a valid PONI.

## Limitations / safety

- No original detector pixels, HDF5/HDXF bytes, masks, or compression codecs are altered. Coordinate re-mapping is performed inside pyFAI during integration, without modifying the stored frame.
- **Notebook Reference 1D** remains an independent normal-incidence radial-mean algorithm and does *not* apply pyFAI `rot1/rot2/rot3` corrections. Its old calculations are unchanged.
- A geometrically valid `.poni` calibration is necessary for scientifically reliable absolute q or 2θ values. Manually entering arbitrary rotations is **not** a calibration.
- The Geometry Preview shows two full pyFAI angle maps. It rejects images over 8 million detector pixels to protect GUI memory and may take a moment to compute on large images.
- This environment did not have pyFAI installed. All unit-logic, fake-integrator wiring, Tk GUI, chart export and prior regression tests passed, but the native pyFAI numerical integration must still be checked with a real calibrated detector in the Windows installation.
- Changing from old radius-based 1D to the new native angular-axis binning is intentionally *not* guaranteed to yield numerically identical profiles. Use Legacy to compare directly against previous results at zero tilt.

## Version semantics

This is a Viewer analysis-layer update. The **HDXF on-disk format, HDXFB frame codecs and lossless conversion behavior are unchanged**.

## pyFAI references

- Geometry definitions: https://pyfai.readthedocs.io/en/stable/geometry_conversion.html
- Per-pixel 2θ and rotation documentation: https://pyfai.readthedocs.io/en/latest/api/geometry/
- Integration usage: https://www.edna-site.org/pub/doc/pyFAI/dev/usage/cookbook/integration_with_python.html
