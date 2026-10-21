"""Real Viewer workflow smoke-test for Module Gap masks."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import h5py
import tkinter as tk
from unittest.mock import patch
from hdxf_viewer import ViewerWorkspaceManager

class GapGuiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'test.h5'
        with h5py.File(self.path, 'w') as f:
            f.create_dataset('/entry/data/data', data=np.ones((2,12,9), dtype='int16'))
            d = f.require_group('/entry/instrument/detector')
            for i, row in enumerate((0,7)):
                m = d.create_group(f'module_{i}')
                m.attrs['NX_class'] = 'NXdetector_module'
                m.create_dataset('data_origin', data=[row,0])
                m.create_dataset('data_size', data=[5,9])
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

    def test_preview_apply_clear_keep_manual(self):
        a = self.app
        canvas = a.active_view.canvas
        canvas._edit_user_mask((1,1), masked=True)
        a.gap_detection_mode_var.set('NeXus module metadata')
        old_rev = a._pyfai_input_revision
        a._gap_preview_active()
        self.assertIsNotNone(canvas.gap_preview)
        self.assertIsNone(canvas.gap_mask)
        self.assertEqual(a._pyfai_input_revision, old_rev)
        self.assertEqual(int(np.count_nonzero(canvas.gap_preview)), 18)
        self.assertEqual(str(a.gap_apply_button['state']), 'normal')
        a._gap_apply_active()
        self.assertIsNone(canvas.gap_preview)
        self.assertEqual(canvas.gap_masked_pixel_count(), 18)
        self.assertEqual(canvas.masked_pixel_count(), 1)
        self.assertEqual(a._pyfai_input_revision, old_rev+1)
        self.assertIn('APPLIED', a.gap_status_var.get())
        self.assertEqual(str(a.gap_apply_button['state']), 'disabled')
        a._gap_clear_active()
        self.assertIsNone(canvas.gap_mask)
        self.assertEqual(canvas.masked_pixel_count(), 1)
        self.assertEqual(a._pyfai_input_revision, old_rev+2)

    def test_manual_ranges_invalid_preview_clears(self):
        a = self.app
        a.gap_detection_mode_var.set('Manual rows / columns')
        a.gap_rows_var.set('2:3')
        a.gap_columns_var.set('')
        a.gap_padding_var.set('0')
        a._gap_preview_active()
        self.assertEqual(int(np.count_nonzero(a.active_view.canvas.gap_preview)), 18)
        a.gap_rows_var.set('1000')
        with patch('hdxf_viewer.messagebox.showwarning') as warning:
            a._gap_preview_active()
            self.assertEqual(warning.call_count, 1)
        self.assertIsNone(a.active_view.canvas.gap_preview)
        self.assertIn('outside', a.gap_status_var.get())
        self.assertEqual(str(a.gap_apply_button['state']), 'disabled')

if __name__ == '__main__': unittest.main()
