import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from hdxf_normalization import (
    NormalizationError, beam_roi_intensity, load_beam_mask, monitor_scalar,
    normalized_sum, parse_beam_roi, validate_normalization_options,
)
from hdf5_processing_tree import main as hdf5_main
from hdxf_sum import _apply_operation, build_parser as hdxf_parser


class NormalizationTests(unittest.TestCase):
    def test_beam_roi_guards(self):
        roi = parse_beam_roi("1,1,2,2", (5, 5))
        frame = np.full((5, 5), 2, dtype=np.uint16)
        frame[1:3, 1:3] = 10
        self.assertEqual(beam_roi_intensity(frame, roi, saturation=100), 40.)
        frame[2, 2] = 100
        with self.assertRaisesRegex(NormalizationError, "saturated"):
            beam_roi_intensity(frame, roi, saturation=100)
        with self.assertRaises(NormalizationError):
            validate_normalization_options("beam-normalized", beam_roi="1,1,2,2", frame_shape=(5,5))
        with self.assertRaises(NormalizationError):
            parse_beam_roi("4,4,2,2", (5,5))
        with self.assertRaises(NormalizationError):
            monitor_scalar(0, label="I0")
        with self.assertRaises(NormalizationError):
            normalized_sum(np.ones((2,2)), float('nan'), label="I0")

    def test_mask_and_monitor_math(self):
        frame = np.full((10,10), 10, dtype=np.int16)
        mask = np.zeros((10,10), bool)
        mask[0,0]=True
        value=beam_roi_intensity(frame, (0,0,10,10), saturation=50, exclude_mask=mask)
        self.assertEqual(value,990.)
        self.assertTrue(np.array_equal(normalized_sum(np.array([40,80]), 20., label="I0"),[2.,4.]))
        for op in ('i0-normalized','beam-normalized'):
            a,note=_apply_operation(np.array([[40,80]]), operation=op,frame_count=2,exposure_total=None,monitor_total=20.)
            np.testing.assert_array_equal(a, [[2.,4.]])
            self.assertIn(op,note)
            self.assertIn(op, hdxf_parser()._option_string_actions['--operation'].choices)

    def _make_sample(self, base):
        base=Path(base)
        master=base/'run_master.h5'
        arr=np.zeros((3,8,8),np.uint16)
        arr[:,:,:]=1
        arr[0,3:5,3:5] = 10
        arr[1,3:5,3:5] = 20
        arr[2,3:5,3:5] = 30
        data=base/'run_data_000001.h5'
        with h5py.File(data,'w') as f:
            f.create_dataset('/entry/data/data',data=arr,maxshape=(None,8,8),chunks=(1,8,8))
        with h5py.File(master,'w') as f:
            f.require_group('/entry/data')
            f['/entry/data/data_000001']=h5py.ExternalLink(data.name,'/entry/data/data')
            f.create_dataset('/entry/instrument/monitor/i0', data=[2.,4.,6.])
        return master,arr

    def test_end_to_end_hdf5_i0_and_beam(self):
        with tempfile.TemporaryDirectory() as tmp:
            master, arr=self._make_sample(tmp)
            output=Path(tmp)/'out_i0'/'out_master.h5'
            result=hdf5_main(['-i',str(master),'-o',str(output),'--mode','i0-normalized',
                              '--i0-path','/entry/instrument/monitor/i0', '--out-dtype','auto',
                              '--select-frame','1,3'])
            self.assertEqual(result,0)
            with h5py.File(output) as f:
                actual=f['/entry/data/data_000001'][...]
                expected=(arr[0].astype(float)+arr[2])/8.
                np.testing.assert_allclose(actual[0], expected)
                self.assertEqual(f['/entry/data/data_000001'].attrs['hdxf_normalization_denominator'],8.)
            beam=Path(tmp)/'out_beam'/'out_master.h5'
            hdf5_main(['-i',str(master),'-o',str(beam),'--mode','beam-normalized',
                       '--beam-roi','3,3,2,2','--beam-saturation','100','--out-dtype','auto'])
            with h5py.File(beam) as f:
                expected=arr.sum(axis=0,dtype=float)/240.0
                np.testing.assert_allclose(f['/entry/data/data_000001'][0],expected)
                self.assertEqual(f['/entry/data/data_000001'].attrs['hdxf_normalization_denominator'],240.)

    def test_failed_reference_does_not_write_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            master,_=self._make_sample(tmp)
            out=Path(tmp)/'failed_master.h5'
            with self.assertRaises((NormalizationError,ValueError)):
                hdf5_main(['-i',str(master),'-o',str(out),'--mode','i0-normalized','--i0-path','/missing'])
            self.assertFalse(out.exists())

class ViewerBeamMaskTransferTests(unittest.TestCase):
    def test_committed_masks_passed_only_to_matching_source(self):
        from hdxf_viewer import HDXFViewerApp
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'input.h5'
            source.touch()
            other = Path(tmp)/'other.h5'
            other.touch()
            sample = np.zeros((8,8),np.int32)
            final_mask = np.zeros((8,8),bool)
            final_mask[0,0]=True
            view = SimpleNamespace(
                archive=SimpleNamespace(path=source),
                current_frame=sample,
                canvas=SimpleNamespace(get_analysis_mask=lambda shape: final_mask.copy()),
            )
            fake = HDXFViewerApp.__new__(HDXFViewerApp)
            fake._processing_kind='sum'
            fake.processing_operation_var=SimpleNamespace(get=lambda:'beam-normalized')
            fake.processing_use_view_masks_var=SimpleNamespace(get=lambda:True)
            fake.processing_input_var=SimpleNamespace(get=lambda:str(source))
            fake.views=[view]
            fake.active_view_index=0
            cmd=['python','hdf5_processing_tree.py','--beam-roi','1,1,2,2']
            fake._prepare_processing_viewer_beam_mask(cmd)
            self.assertIn('--beam-mask',cmd)
            temp = Path(cmd[cmd.index('--beam-mask') + 1])
            np.testing.assert_array_equal(np.load(temp), final_mask)
            fake._cleanup_processing_beam_mask()
            self.assertFalse(temp.exists())
            fake.processing_input_var=SimpleNamespace(get=lambda:str(other))
            cmd=['python']
            fake._prepare_processing_viewer_beam_mask(cmd)
            self.assertNotIn('--beam-mask', cmd)

class MultiSourceI0Tests(unittest.TestCase):
    def test_global_master_i0_vector_across_two_data_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            master=root/'sample_master.h5'
            with h5py.File(master,'w') as m:
                m.require_group('/entry/data')
                m.create_dataset('/entry/instrument/monitor/i0', data=[2.,3.,5.,10.])
                for j in range(2):
                    data=root/f'sample_data_{j+1:06d}.h5'
                    with h5py.File(data,'w') as d:
                        d.create_dataset('/entry/data/data',data=np.full((2,8,8),j+1,np.uint16),
                                         chunks=(1,8,8),maxshape=(None,8,8))
                    m[f'/entry/data/data_{j+1:06d}']=h5py.ExternalLink(data.name,'/entry/data/data')
            out=root/'out'/'merged_master.h5'
            hdf5_main(['-i',str(master),'-o',str(out),'--mode','i0-normalized',
                       '--i0-path','/entry/instrument/monitor/i0'])
            with h5py.File(out) as f:
                np.testing.assert_allclose(f['/entry/data/data_000001'][0], 6/20)
                self.assertEqual(f['/entry/data/data_000001'].attrs['hdxf_normalization_denominator'], 20.)

if __name__ == '__main__':
    unittest.main()
