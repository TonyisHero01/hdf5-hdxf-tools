"""Image stripe Gap proposal: conservative and explicitly user-confirmed."""
import unittest
import numpy as np
from hdxf_module_gaps import image_stripe_gap_proposal, GapMaskError


class GapImageHeuristicTests(unittest.TestCase):
    def test_two_internal_horizontal_and_one_vertical(self):
        rng = np.random.default_rng(105)
        frame = rng.poisson(8, (210, 280)).astype(np.int16)
        frame[52:59, :] = 0
        frame[142:149, :] = 0
        frame[:, 120:126] = 0
        proposal = image_stripe_gap_proposal(frame)
        self.assertTrue(np.all(proposal.mask[52:59, :]))
        self.assertTrue(np.all(proposal.mask[142:149, :]))
        self.assertTrue(np.all(proposal.mask[:, 120:126]))
        self.assertFalse(proposal.mask[20, 20])
        self.assertIn('HEURISTIC', proposal.description)

    def test_realistic_sparse_frame_noisy_black_bands(self):
        # Simulate a photon-counting image: most pixels are zero; true module
        # gaps contain a few stray nonzero counts (> former 0.5% threshold).
        rng = np.random.default_rng(509)
        frame = rng.poisson(0.11, (1614, 1030)).astype(np.uint16)
        frame[485:520] = rng.poisson(0.013, (35, 1030)).astype(np.uint16)
        frame[1085:1120] = rng.poisson(0.013, (35, 1030)).astype(np.uint16)
        proposal = image_stripe_gap_proposal(frame)
        self.assertTrue(np.all(proposal.mask[485:520, :]))
        self.assertTrue(np.all(proposal.mask[1085:1120, :]))
        self.assertFalse(bool(proposal.mask[400, 500]))
        self.assertEqual(int(np.count_nonzero(proposal.mask)), 70 * 1030)

    def test_gently_darkening_detector_not_misclassified(self):
        rng = np.random.default_rng(1051)
        # Gradual illumination change does not produce a sharp internal strip.
        frame = np.zeros((240, 260), dtype=np.uint16)
        for y in range(240):
            count = 0.15 - 0.08 * y / 239.0
            frame[y] = rng.poisson(count, 260).astype(np.uint16)
        with self.assertRaisesRegex(GapMaskError, 'No confident'):
            image_stripe_gap_proposal(frame)

    def test_blank_detector_or_extreme_sparse_frame_is_not_a_gap(self):
        with self.assertRaisesRegex(GapMaskError, 'No confident'):
            image_stripe_gap_proposal(np.zeros((180, 180), dtype=np.uint16))
        rng = np.random.default_rng(2)
        sparse = (rng.random((180, 180)) < 0.001).astype(np.uint16)
        with self.assertRaisesRegex(GapMaskError, 'No confident'):
            image_stripe_gap_proposal(sparse)

    def test_dark_exterior_is_not_interpreted_as_module_gap(self):
        frame = np.ones((120, 150), dtype=np.uint16)
        frame[:16, :] = 0
        frame[-20:, :] = 0
        with self.assertRaisesRegex(GapMaskError, 'No confident'):
            image_stripe_gap_proposal(frame)

    def test_padding_and_negative_sentinel(self):
        frame = np.full((140, 180), 10, dtype=np.int16)
        frame[61:67, :] = np.iinfo(np.int16).min
        proposal = image_stripe_gap_proposal(frame, padding=2)
        self.assertTrue(np.all(proposal.mask[59:69, :]))
        self.assertFalse(proposal.mask[58, 5])

    def test_nonuniform_black_patch_is_not_full_module_gap(self):
        frame = np.ones((160, 200), dtype=np.int16)
        frame[70:80, 60:100] = 0
        with self.assertRaisesRegex(GapMaskError, 'No confident'):
            image_stripe_gap_proposal(frame)

if __name__ == '__main__':
    unittest.main()
