"""Numerical checks for equivalent 3D lowering and deployment preprocessing."""
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tensorflow as tf
import torch
from torch import nn

from neural_methods.model.phase_shifted_rppg import BBPNet
from tools.phase_shifted_tflite.tf_bridge import apply, BBPBridge
from tools.phase_shifted_tflite.runtime import upsample_standardize


class BridgeTests(unittest.TestCase):
    def compare3d(self, layer, shape):
        rng = np.random.default_rng(125)
        x = rng.normal(size=shape).astype(np.float32)
        with torch.inference_mode():
            ref = layer(torch.from_numpy(x)).numpy()
        converted = apply(layer, tf.constant(x.transpose(0, 2, 3, 4, 1))).numpy().transpose(0, 4, 1, 2, 3)
        np.testing.assert_allclose(converted, ref, atol=3e-6, rtol=3e-5)

    def test_dilated_temporal_and_spatial_conv3d(self):
        self.compare3d(nn.Conv3d(3, 5, (3, 3, 3), padding=(4, 0, 0), dilation=(4, 2, 2)), (1, 3, 12, 9, 9))

    def test_padded_strided_conv3d(self):
        self.compare3d(nn.Conv3d(3, 5, (3, 5, 5), padding=(4, 2, 2), dilation=(4, 1, 1), stride=(1, 2, 2)), (1, 3, 12, 9, 9))

    def test_spatial_temporal_pool3d_negative_values(self):
        for cls in (nn.MaxPool3d, nn.AvgPool3d):
            with self.subTest(pool=cls.__name__):
                self.compare3d(cls((4, 2, 2), stride=(4, 2, 2)), (1, 3, 13, 9, 9))

    def test_bbp_features_and_attention_match(self):
        torch.manual_seed(125)
        model = BBPNet(60., 240., 30., 140., .5).eval()
        # Keep the randomly initialized bounded output away from saturation.
        with torch.no_grad():
            for branch in (model.Subnetwork1, model.Subnetwork2):
                branch.regressor2[-1].weight.mul_(.00001)
        rng = np.random.default_rng(101)
        a, b = [rng.normal(size=(1, 600)).astype(np.float32) for _ in range(2)]
        with torch.inference_mode():
            sbp, dbp = model(torch.from_numpy(a), torch.from_numpy(b))
        bridge = BBPBridge(model)
        out = bridge.serve(a, b)
        np.testing.assert_allclose(out['sbp'], sbp.numpy(), atol=1e-4, rtol=1e-5)
        np.testing.assert_allclose(out['dbp'], dbp.numpy(), atol=1e-4, rtol=1e-5)

    def test_unbiased_normalization_matches_training(self):
        from scipy.signal import resample
        wave = np.random.default_rng(125).normal(size=(1, 150)).astype(np.float32)
        x = torch.from_numpy(resample(wave, 600, axis=-1))
        reference = ((x-x.mean())/x.std()).numpy()
        np.testing.assert_allclose(upsample_standardize(wave), reference, atol=5e-7, rtol=1e-5)
        with self.assertRaisesRegex(ValueError, 'flat/nonfinite'):
            upsample_standardize(np.zeros((1, 150), np.float32))


if __name__ == '__main__':
    torch.set_num_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    unittest.main()
