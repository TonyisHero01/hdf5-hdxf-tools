"""Detector geometry: axis mapping, no-tilt guards, map preview and stale outputs."""
from pathlib import Path
import sys
import math
import tkinter as tk
import unittest
from tkinter import ttk
from unittest.mock import patch, PropertyMock

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdxf_geometry import (CORRECTED_THETA, CORRECTED_TWOTHETA, CORRECTED_Q,
                           LEGACY_THETA, get_1d_axis_spec, format_1d_axis,
                           two_theta_degrees, color_geometry_map)
from hdxf_viewer import HDXFViewerApp
from hdxf_plot_export import build_integration_figure


class GeometryMathTests(unittest.TestCase):
    def test_native_corrected_axes_and_legacy_tilt_rejection(self):
        self.assertEqual(get_1d_axis_spec(CORRECTED_THETA, .15, 0, 0).integration_unit, '2th_rad')
        self.assertEqual(get_1d_axis_spec(CORRECTED_TWOTHETA, 0, .15, 0).integration_unit, '2th_deg')
        self.assertEqual(get_1d_axis_spec(CORRECTED_Q, 0, 0, .15).integration_unit, 'q_A^-1')
        self.assertEqual(get_1d_axis_spec(LEGACY_THETA, 0, 0, 0).integration_unit, 'r_mm')
        with self.assertRaisesRegex(ValueError, 'not valid for tilted'):
            get_1d_axis_spec(LEGACY_THETA, math.radians(5), 0, 0)

    def test_native_angles_are_not_converted_from_radius(self):
        tth = np.asarray([0.2, 0.4, 0.8])
        np.testing.assert_allclose(format_1d_axis(tth, display_axis='theta_corrected_rad', distance_m=.065), tth / 2)
        np.testing.assert_array_equal(format_1d_axis(tth, display_axis='2th_deg', distance_m=.065), tth)
        np.testing.assert_array_equal(format_1d_axis(tth, display_axis='q_A^-1', distance_m=.065), tth)
        old = format_1d_axis(np.asarray([0, 65]), display_axis='theta_custom_rad', distance_m=.065)
        np.testing.assert_allclose(old, [0, math.pi / 8])

    def test_map_and_color_preview(self):
        class FakeIntegrator:
            def twoThetaArray(self, shape):
                y, x = np.indices(shape)
                return (x+y) / 100.0
        img = two_theta_degrees(FakeIntegrator(), (7, 9))
        self.assertEqual(img.shape, (7, 9))
        rgb, lo, hi = color_geometry_map(img)
        self.assertEqual(rgb.shape, (7,9,3))
        self.assertAlmostEqual(lo, 0)
        self.assertGreater(hi, 0)
        delta_rgb, a, b = color_geometry_map(img - np.mean(img), delta=True)
        self.assertEqual(delta_rgb.shape, rgb.shape)
        self.assertAlmostEqual(-a, b)

    def test_1d_chart_axis_is_explicit(self):
        result={'kind':'1d', 'radial':np.arange(5.), 'intensity':np.arange(5.),
                'source':{'filename':'run.h5','frame_count_averaged':2},
                'config':{'one_d_axis':'theta_corrected_rad'}}
        fig=build_integration_figure(result)
        self.assertIn('geometry corrected', fig.axes[0].get_xlabel())


class GeometryUIRegression(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk()
        self.root.withdraw()
        self.app=HDXFViewerApp.__new__(HDXFViewerApp)
        app=self.app
        app.root=self.root
        app._pyfai_result={'kind':'1d', 'intensity':np.array([1.])}
        app._pyfai_result_window=tk.Toplevel(self.root)
        app._pyfai_result_canvas=None
        app._pyfai_result_photo=None
        app._pyfai_worker=None
        app._pyfai_input_revision=0
        app._pyfai_recompute_required=False
        app._pyfai_ai_cache={'old':'cached'}
        app.pyfai_status_var=tk.StringVar(master=self.root)
        for name in ('pyfai_show_result_button','pyfai_export_button','pyfai_export_chart_button'):
            setattr(app,name,ttk.Button(self.root))

    def tearDown(self):
        self.root.destroy()

    def test_geometry_change_prevents_exporting_old_result(self):
        self.app._pyfai_geometry_setting_changed()
        self.assertIsNone(self.app._pyfai_result)
        self.assertEqual(self.app._pyfai_input_revision,1)
        self.assertEqual(self.app._pyfai_ai_cache,{})
        self.assertIn('Geometry',self.app.pyfai_status_var.get())
        self.assertEqual(str(self.app.pyfai_export_chart_button.cget('state')),'disabled')


if __name__=='__main__':
    unittest.main()

class ViewerGeometryWiringTests(unittest.TestCase):
    def test_config_passes_rotations_and_native_unit_to_integrator(self):
        from types import SimpleNamespace
        class V:
            def __init__(self, text): self.text=str(text)
            def get(self): return self.text
        viewer=HDXFViewerApp.__new__(HDXFViewerApp)
        args={
            'pyfai_distance_var':V('.065'), 'pyfai_pixel_x_var':V('.000075'),
            'pyfai_pixel_y_var':V('.000075'), 'pyfai_wavelength_var':V('1.4'),
            'pyfai_beam_x_var':V('100'), 'pyfai_beam_y_var':V('80'),
            'pyfai_rot1_var':V('3'), 'pyfai_rot2_var':V('-2'), 'pyfai_rot3_var':V('1'),
            'pyfai_radial_bins_var':V('150'), 'pyfai_mode_var':V('1D radial integration'),
            'pyfai_1d_axis_var':V(CORRECTED_THETA),
            'pyfai_azimuth_bins_var':V('100'), 'pyfai_azimuth_min_var':V('-180'),
            'pyfai_azimuth_max_var':V('180'), 'pyfai_polarization_var':V(''),
            'pyfai_unit_var':V('q_A^-1'), 'pyfai_solid_angle_var':V(False),
        }
        for key,value in args.items(): setattr(viewer,key,value)
        frame=SimpleNamespace(current_frame=np.zeros((12,16)))
        config=viewer._pyfai_config_from_ui(frame)
        self.assertEqual(config['integration_unit'],'2th_rad')
        self.assertEqual(config['one_d_axis'],'theta_corrected_rad')
        self.assertAlmostEqual(config['rot1'],math.radians(3))
        self.assertAlmostEqual(config['rot2'],math.radians(-2))
        self.assertAlmostEqual(config['rot3'],math.radians(1))
        self.assertAlmostEqual(config['poni2'],100.5*.000075)
        self.assertAlmostEqual(config['poni1'],80.5*.000075)
        class FakeDetector:
            def __init__(self,**kwargs): self.kwargs=kwargs
        class FakeAI:
            def __init__(self,**kwargs): self.kwargs=kwargs
        viewer._pyfai_loaded_detector=None
        viewer._pyfai_ai_cache={}
        with patch('hdxf_viewer.PYFAI_AVAILABLE',True), \
             patch('hdxf_viewer.PyFAIDetector',FakeDetector), \
             patch('hdxf_viewer.PyFAIAzimuthalIntegrator',FakeAI):
            ai=viewer._pyfai_integrator_for_config(config)
        self.assertAlmostEqual(ai.kwargs['rot1'],math.radians(3))
        self.assertAlmostEqual(ai.kwargs['poni2'],config['poni2'])
        self.assertEqual(ai.kwargs['detector'].kwargs['max_shape'],(12,16))

class GeometryPreviewGUIClick(unittest.TestCase):
    def test_preview_opens_and_updates_real_tk_window(self):
        from types import SimpleNamespace
        self_root=tk.Tk()
        self_root.withdraw()
        try:
            app=HDXFViewerApp.__new__(HDXFViewerApp)
            app.root=self_root
            config={'shape':(40,50), 'distance_m':.1, 'poni1':.002,
                    'poni2':.002, 'rot1':.1, 'rot2':.03, 'rot3':.0,
                    'wavelength_m':1e-10}
            app._pyfai_config_from_ui=lambda _: config
            app._pyfai_detector_for_config=lambda _: object()
            class FakeIntegrator:
                def __init__(self, **kwargs): self.rot=float(kwargs.get('rot1',.1))
                def twoThetaArray(self, shape):
                    y,x=np.indices(shape)
                    return .01 + x*.0002 + y*.0001 + self.rot*.05
            app._pyfai_integrator_for_config=lambda _:FakeIntegrator(rot1=.1)
            with patch.object(HDXFViewerApp, 'active_view', new_callable=PropertyMock) as active, \
                 patch('hdxf_viewer.PYFAI_AVAILABLE', True), \
                 patch('hdxf_viewer.PyFAIAzimuthalIntegrator', FakeIntegrator):
                active.return_value=SimpleNamespace(current_frame=np.zeros((40, 50)))
                app._pyfai_geometry_preview()
                self_root.update()
            matches=[w for w in self_root.winfo_children()
                     if isinstance(w,tk.Toplevel) and w.title()=='pyFAI Detector Geometry Preview']
            self.assertEqual(len(matches),1)
            self.assertTrue(matches[0].winfo_exists())
            widgets=[w for w in matches[0].winfo_children()]
            self.assertTrue(any(isinstance(w,tk.Canvas) for w in widgets))
        finally:
            self_root.destroy()
