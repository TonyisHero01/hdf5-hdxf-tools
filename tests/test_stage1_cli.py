"""First-stage maintenance regression checks (run: python -m unittest discover -s tests)."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
TOOLS = (
    "hdxf_viewer.py",
    "hdf5_to_hdxf.py",
    "hdxf_to_hdf5.py",
    "hdxf_sum.py",
    "hdxf_subtract.py",
    "hdxf_pixelop.py",
    "hdf5_processing_tree.py",
)


class CLIMaintenanceTests(unittest.TestCase):
    def test_all_tools_report_software_suite_version_without_opening_gui(self):
        for script in TOOLS:
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, str(PROJECT / script), "--version"],
                    cwd=PROJECT, capture_output=True, text=True, timeout=25,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("HDXF suite 0.4.8", result.stdout)

    def test_divide_only_and_rounding(self):
        from hdxf_pixelop import (
            TransformConfig, _apply_pixel_operation, _selected_operation,
            build_parser,
        )
        import numpy as np
        parser = build_parser()
        args = parser.parse_args(["-i", "fake.hdxf", "-o", "output", "--divide", "2"])
        operation, value = _selected_operation(args)
        self.assertEqual(operation, "divide")
        self.assertEqual(value, 2.0)
        actual = _apply_pixel_operation(
            np.array([[-4, -1, 0, 1, 2, 5]], dtype=np.int16),
            TransformConfig(operation=operation, value=value, roi=None),
        )
        np.testing.assert_array_equal(actual, np.array([[-2, 0, 0, 0, 1, 2]], dtype=np.int16))
        args = parser.parse_args(["-i", "fake.hdxf", "-o", "output", "--divide", "0"])
        with self.assertRaisesRegex(Exception, "non-zero"):
            _selected_operation(args)
        # Ensure the deprecated, incorrectly spelled option is rejected outright.
        old_flag = "--" + "de" + "vide"
        with self.assertRaises(SystemExit) as captured:
            parser.parse_args(["-i", "fake.hdxf", "-o", "output", old_flag, "2"])
        self.assertEqual(captured.exception.code, 2)

    def test_main_and_legacy_archive_versions_are_not_silently_changed(self):
        import hdf5_to_hdxf
        import hdxf_sum
        import hdxf_pixelop
        self.assertEqual(hdf5_to_hdxf.FORMAT_VERSION, "0.4.8")
        self.assertEqual(hdxf_sum.FORMAT_VERSION, "0.4.5")
        self.assertNotIn("0.4.8", hdxf_pixelop.SUPPORTED_FORMAT_VERSIONS)


if __name__ == "__main__":
    unittest.main()
