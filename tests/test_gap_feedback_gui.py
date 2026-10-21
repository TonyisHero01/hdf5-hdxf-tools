"""Gap preview failure is visible in the tool card and status bar."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import h5py
import tkinter as tk
from unittest.mock import patch
from hdxf_viewer import ViewerWorkspaceManager


class GapFeedbackGuiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'no_modules.h5'
        frame = np.ones((80, 100), dtype='uint16') * 10
        frame[38:43, :] = 0
        with h5py.File(self.path, 'w') as f:
            f.create_dataset('/entry/data/data', data=frame[None])
        self.root = tk.Tk()
        self.root.geometry('1200x800')
        self.manager = ViewerWorkspaceManager(self.root)
        self.root.update()
        self.app = self.manager.app
        self.app.open_path(self.path)
        self.root.update()
        self.app._show_sidebar_page('tools')

    def tearDown(self):
        self.app._shutdown_resources()
        self.root.destroy()
        self.tmp.cleanup()

    def test_missing_geometry_explained_and_survives_refresh(self):
        app = self.app
        app.gap_detection_mode_var.set('NeXus module metadata')
        with patch('hdxf_viewer.messagebox.showwarning') as warning:
            app._gap_preview_active()
            self.assertEqual(warning.call_count, 1)
        self.assertIsNone(app.active_view.canvas.gap_preview)
        self.assertIn('NO PREVIEW', app.gap_status_var.get())
        self.assertIn('Image stripes', app.gap_status_var.get())
        self.assertIn('NO PREVIEW', app.status_var.get())
        self.assertEqual(str(app.gap_apply_button['state']), 'disabled')
        app._refresh_tools_sidebar()
        self.assertIn('NO PREVIEW', app.gap_status_var.get())

    def test_image_heuristic_preview_only_and_apply(self):
        app = self.app
        app.gap_detection_mode_var.set('Image stripes (heuristic preview)')
        app._gap_preview_active()
        self.assertIsNotNone(app.active_view.canvas.gap_preview)
        self.assertIsNone(app.active_view.canvas.gap_mask)
        self.assertEqual(int(np.count_nonzero(app.active_view.canvas.gap_preview)), 500)
        self.assertIn('VERIFY', app.gap_status_var.get())
        app._gap_apply_active()
        self.assertEqual(app.active_view.canvas.gap_masked_pixel_count(), 500)
        self.assertIsNone(app.active_view.canvas.gap_preview)

    def test_actual_preview_button_dispatch(self):
        app = self.app
        app.gap_detection_mode_var.set('Image stripes (heuristic preview)')
        app._gap_mode_changed()
        self.assertEqual(str(app.gap_preview_button['state']), 'normal')
        app.gap_preview_button.invoke()
        self.assertIn('PREVIEW READY', app.gap_status_var.get())
        self.assertIsNotNone(app.active_view.canvas.gap_preview)
        self.assertEqual(str(app.gap_apply_button['state']), 'normal')

    def test_preview_click_failure_always_visible_even_if_unexpected(self):
        app = self.app
        app.gap_detection_mode_var.set('Image stripes (heuristic preview)')
        with patch('hdxf_viewer.image_stripe_gap_proposal', side_effect=RuntimeError('diagnostic test')), \
             patch('hdxf_viewer.messagebox.showwarning') as warning:
            app._gap_preview_active()
            self.assertEqual(warning.call_count, 1)
        self.assertIsNone(app.active_view.canvas.gap_preview)
        self.assertIn('Unexpected RuntimeError', app.gap_status_var.get())
        self.assertIn('diagnostic test', app.status_var.get())
        self.assertEqual(str(app.gap_apply_button['state']), 'disabled')

if __name__ == '__main__':
    unittest.main()
