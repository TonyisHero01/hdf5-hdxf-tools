# Notebook Reference: manual-mask before / after verification

## What to do

1. Open the detector data and choose **1D Notebook Reference**.
2. Draw an ROI outside the `Exclude Center Radius` around a bright feature and choose **Mask Inside ROI** from the right-click menu. The marked pixels turn pink.
3. Press **Run Notebook Reference 1D**. Wait for the background all-frame average and two reductions.
4. In the result window, use **Mask comparison → Before / After** or **Difference Δ**. Change the **Method** between Spots included, Hot/Bragg spots masked, and Ring MAD σ-clipped.
5. Inspect **Newly excluded from base**, **Bins changed**, **Max |Δ|**, and the CSV produced by **Export…**.

## Meaning of curves

- **Before manual Mask**: Notebook's base filtering (negative, nonfinite, saturation, gap bands), and if enabled the saved HDF5/HDXF detector `pixel_mask`, but without Viewer manual pixels.
- **After manual Mask**: exactly the same options, averaged image, and beam center, plus Viewer manual pixels.
- **Δ**: `After − Before` for bins with valid pixels on both sides. If either bin has no valid pixels, its value and delta are NaN in the comparison CSV.
- Hot/Bragg and MAD profiles have their own automatic outlier exclusions. They may change less than Spots included.
- Plotting honors `Exclude Center Radius`, so a pink ROI near the center can have no impact on the visible curve.

## Scientific scope

This numerical comparison is for **Notebook Reference** (radius-bin mean), not pyFAI's integrator. It does not alter the Notebook reference algorithm, source pixel values, archives, or masks persisted to file. It recomputes the Notebook radial reduction twice after **one** input-image average. A fresh Gaussian beam fit (when enabled) is computed once and its coordinates are reused by both passes to ensure a controlled comparison.

The first reduction runs with manual Mask and the second without it, preserving any optional source detector mask in both passes. The header in the result window reports manually selected pixels and the count newly excluded beyond Notebook's existing base filter. Results are invalidated by manual/source-mask changes; press Run again.

## Verification

- Syntax: `python -m compileall -q -l .`
- Scientific/regression tests: `python -m unittest discover -s tests -p test_notebook_mask_comparison.py -v`
- GUI result/CSV tests: `python -m unittest discover -s tests -p test_notebook_comparison_gui.py -v` (requires a desktop session)

A real Windows pyFAI+Jungfrau end-to-end test is still recommended. Notebook Reference depends on SciPy but does not use pyFAI numerically.
