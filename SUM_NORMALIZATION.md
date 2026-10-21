# Sum normalization — I0 monitor and direct-beam ROI

The HDXF and HDF5 **Sum / Mean** processing panels now include two additional
normalization operations. This feature writes a **new derived one-frame file**;
it does **not** modify input detector frames or perform pyFAI integration.

## Modes and exact formula

| Operation | Output frame |
|---|---|
| `sum` | `Σ frame_i` (existing) |
| `mean` | `Σ frame_i / N` (existing) |
| `exposure-normalized` | `Σ frame_i / Σ exposure_time_i` (existing, counts/s) |
| `i0-normalized` | `Σ frame_i / Σ I0_i` (new, counts per I0 unit) |
| `beam-normalized` | `Σ frame_i / Σ valid_direct_beam_ROI_counts_i` (new, dimensionless pixel/beam-ROI-count ratio) |

Only **selected** frames contribute to numerator and denominator. The
reference is a scalar per frame, not an image-wise division by the beam ROI.
The new modes are **ratio-of-sums**, not the arithmetic mean of per-frame ratios.
As a result, stronger-monitor frames carry more weight. This is intentional and
must be documented when comparing to alternative processing pipelines.

## Viewer workflow

1. Open a source HDF5 master or a supported HDXF archive.
2. Open **HDF5 Sum / Mean** or **HDXF Sum / Mean** from the processing menu.
3. Choose an operation and select frames (default: all). Leave output dtype
   at **auto** so fractional values are saved as float64.
4. For **I0**, enter the *actual* absolute HDF5 monitor Dataset path in
   **I0 DATASET PATH** (e.g. `/entry/instrument/monitor/i0` **only if that
   specific Dataset really exists in your file**).
5. For **Direct Beam**, draw a **rectangular ROI** around a clean, unsaturated
   reference beam, then click **Current ROI**. Enter the detector's actual
   **first saturated count** in **SATURATED AT**. It is required; there is no
   automatic saturation estimate. Optionally select an external bool `.npy`
   beam exclusion Mask (`True` = exclude).
6. When the processing source is the same as the currently open Viewer source,
   **Also use applied Viewer Manual / Gap Masks** (enabled by default) passes
   those committed masks to the **beam reference calculation only** (not to
   the numerator pixel sum). Preview-only Gap masks
   are excluded. Masks are **not** copied to unrelated source files, even if
   they happen to have the same dimensions.
7. Start processing. The log prints the monitor denominator. For HDF5 output,
   the derived frame Dataset has `hdxf_normalization_*` attributes recording
   the method, denominator, frame count, formula and reference. The HDXF result
   records equivalent information in `manifest.json` → `normalization`.

### Important detector cautions

- An apparent bright spot is **not** necessarily a measurable direct beam.
  Beamstops, dynamic-range overflow, unknown gain calibration, incomplete
  detector correction and detector motion can make ROI intensity invalid.
- Beam ROI normalization rejects known saturated frames rather than silently
  dropping saturated pixels. Negative or nonfinite pixels are excluded. At
  least 95% of the ROI must remain valid after exclusion.
- An arbitrary unsaturated pixel threshold **is not** a validated detector
  saturation calibration. Use known detector/calibration information.
- A true upstream I0 monitor is preferable when the direct beam is blocked,
  saturated, attenuated differently, or altered by the sample. Use consistent
  acquisition units and calibrations throughout the selected dataset.
- The ROI is rectangular even if the Viewer has other ROI shapes. The beam
  normalization value is the **sum** of valid ROI intensities, not average.
- Direct beam ROI is evaluated from **input source files**, not unsaved pixel
  edits in Viewer. The Viewer can transfer the committed mask separately.
- Don't force an integer output dtype for a normalized image: precision can be
  irretrievably lost.
- This is not per-frame flux correction, beam fitting, exposure correction, or
  automatic physics/geometry calibration. It is a precisely defined global
  monitor-weighted normalization.

## Examples (paths and numbers are illustrative)

From a supported HDF5 master:

```powershell
python hdf5_processing_tree.py -i source_master.h5 -o output_master.h5 `
  --mode i0-normalized --i0-path /entry/instrument/monitor/i0 --out-dtype auto
```

```powershell
python hdf5_processing_tree.py -i source_master.h5 -o output_master.h5 `
  --mode beam-normalized --beam-roi 480,960,24,24 `
  --beam-saturation 65535 --out-dtype auto
```

Corresponding options are available in `hdxf_sum.py` using `--operation` instead
of `--mode` and writing a derived `.hdxf` archive.

**HDXF compatibility limitation:** The shipped `hdxf_sum.py` still accepts
input HDXF archive versions **through 0.4.5** and writes archive version
0.4.5. In particular, the 0.4.8 main converter's newer input archives are
**not** yet supported by that older Sum backend. This feature does not silently
change the archive schema or bypass version checks. For 0.4.8 projects with an
original HDF5 master available, use `hdf5_processing_tree.py` until the HDXF
Sum writer is upgraded and round-trip validated.

## Validation

Unit tests check a known numeric array with two selected I0 readings,
beam-ROI reference counts, failure on saturated/missing monitor inputs, transfer
of committed Viewer masks only for matching source paths, and derived HDF5
master+ExternalLink/data consistency. Run:

```powershell
python -m unittest discover -s tests -p test_sum_normalization.py -v
```

No new HDXF codec, archive format, original pixel values or pyFAI numerical
integration code is changed.
