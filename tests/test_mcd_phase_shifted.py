"""MCD contracts: synthetic fixtures, no real training or OpenCV requirement.

Trainer imports are isolated as in test_phase_shifted_rppg. These tests do not
claim that the complete main.py environment or real video face cropper runs.
"""
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_phase_shifted_rppg import Trainer, config_fixture, namespace
from dataset.mcd_preprocessing import read_sync, regular_grid, sample_indices, split_subjects, aligned_clips
from neural_methods.loss.PhaseShiftedLoss import CuffBPLoss
from neural_methods.model.phase_shifted_rppg.inference import estimate_hr, predict_signals


class MCDTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = config_fixture(self.tmp.name)
        p = self.config.MODEL.PHASE_SHIFTED_RPPG
        p.LABEL_SOURCE, p.PREPROCESS_VERSION = 'PPG_CUFF', 'mcd_ppg_cuff_v1'
        p.SBP_MIN, p.SBP_MAX, p.DBP_MIN, p.DBP_MAX = 60., 240., 30., 140.
        self.config.INFERENCE.BATCH_SIZE = 1
        for name in ('TRAIN', 'VALID', 'TEST'):
            data = getattr(self.config, name).DATA
            data.DATASET = 'MCD-rPPG'
            data.MCD = namespace(dict(CAMERAS=['FullHDwebcam'], SPLIT_SEED=100,
                                     MAX_SYNC_ERROR=.02, MAX_FRAME_GAP=.1, MAX_TAIL_DROP=32))

    def test_cuff_loss_backpropagates_without_abp(self):
        sbp = torch.tensor([110., 130.], requires_grad=True)
        dbp = torch.tensor([70., 90.], requires_grad=True)
        loss, stats = CuffBPLoss()(sbp, dbp, sbp.detach()+5, dbp.detach()-5)
        loss.backward()
        self.assertTrue((sbp.grad < 0).all())
        self.assertTrue((dbp.grad > 0).all())
        self.assertNotIn('waveform_loss', stats)
        zero, _ = CuffBPLoss()(sbp, dbp, sbp.detach(), dbp.detach())
        self.assertEqual(float(zero.detach()), 0.)  # A correct prediction may legitimately have zero loss.

    def test_real_bbp_receives_cuff_gradients_with_drp_frozen(self):
        trainer = Trainer(self.config, {})
        class DRP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.))
            def forward(self, video):
                wave = torch.sin(torch.arange(150)[None, :]*.3).expand(2, -1)*self.scale
                return wave, wave.roll(2, -1)
        trainer.drpnet = DRP()
        sbp, dbp, _ = trainer._stage2_forward(torch.zeros(2, 1))
        loss, _ = trainer.stage2_loss(sbp, dbp, torch.tensor([105., 113.]), torch.tensor([78., 78.]))
        loss.backward()
        self.assertIsNone(trainer.drpnet.scale.grad)
        self.assertFalse(trainer.drpnet.scale.requires_grad)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer.bbpnet.parameters()))

    def test_state_weights_compensate_duplicate_windows(self):
        a = torch.tensor([110., 110., 130.])
        loss, _ = CuffBPLoss()(a, a, a+torch.tensor([1., 1., 5.]), a,
                              torch.tensor([.75, .75, 1.5]))
        self.assertAlmostEqual(float(loss), (.5+4.5)/2)

    def test_ppg_contract_and_signed_checkpoint(self):
        trainer = Trainer(self.config, {})
        video = torch.zeros(1, 3, 150, 128, 128)
        labels = dict(ppg=torch.ones(1, 150), sbp=torch.tensor([105.]),
                      dbp=torch.tensor([78.]), bp_valid=torch.tensor([True]))
        _, target = trainer._batch((video, labels), 'STAGE2')
        self.assertNotIn('abp', target)
        path = trainer.save_model('Best', 'FULL')
        trainer._load_checkpoint(path, 'FULL')
        payload = torch.load(path, weights_only=True)
        payload['signature']['LABEL_SOURCE'] = 'ABP_MMHG'
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, 'signature'):
            trainer._load_checkpoint(path, 'FULL')
        labels['sbp'][0] = 250
        with self.assertRaisesRegex(ValueError, 'outside'):
            trainer._batch((video, labels), 'STAGE2')

    def test_inconsistent_split_policy_rejected(self):
        self.config.TRAIN.DATA.MCD.SPLIT_SEED = 101
        with self.assertRaisesRegex(ValueError, 'SPLIT_SEED'):
            Trainer.validate_config(self.config)

    def test_subject_split_keeps_states_and_cameras_together(self):
        rows = [dict(patient_id=str(i), step=s, camera=c) for i in range(20)
                for s in ('before', 'after') for c in ('a', 'b')]
        parts = [split_subjects(rows, a, b) for a, b in ((0, .7), (.7, .85), (.85, 1))]
        ids = [{r['patient_id'] for r in p} for p in parts]
        self.assertEqual([len(s) for s in ids], [14, 3, 3])
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertEqual(sum(map(len, parts)), len(rows))
        self.assertEqual(parts[0], split_subjects(list(reversed(rows)), 0, .7)[::-1])

    def test_loader_overlap_preflight(self):
        trainer = Trainer(self.config, {})
        class Loader:
            dataset = namespace(dict(subject_ids={'1020'}))
        with self.assertRaisesRegex(ValueError, 'leakage'):
            trainer._check_loaders(dict(train=Loader(), test=Loader()), 'STAGE2')

    def test_timestamp_mapping_rejects_shift_and_preserves_duration(self):
        root = Path(self.tmp.name)
        origin = datetime(2026, 1, 1)
        meta, raw, sync = root/'meta.txt', root/'raw.PW', root/'sync.txt'
        meta.write_text('\n'.join(f'{i+1} {origin+timedelta(seconds=i*.04)}' for i in range(151)))
        raw.write_text('\n'.join(f'{i} {origin+timedelta(seconds=i*.01)}' for i in range(601)))
        values = np.column_stack((np.arange(151)*4, np.zeros(151)))
        np.savetxt(sync, values)
        row = dict(meta=meta, ppg=raw, ppg_sync=sync)
        t, ppg, _, extra = read_sync(row)
        grid = regular_grid(t, 25)
        self.assertEqual(extra, 0)
        self.assertAlmostEqual(grid[-1], 6.)
        np.testing.assert_array_equal(sample_indices(t, grid), np.arange(151))
        values[20, 0] += 1
        np.savetxt(sync, values)
        with self.assertRaisesRegex(ValueError, 'amplitude'):
            read_sync(row)

    def test_bad_intervals_not_compressed_into_valid_windows(self):
        t = np.arange(451)/25
        ppg = np.sin(2*np.pi*1.2*t)
        valid = np.ones(451, bool); valid[200] = False
        frames = np.zeros((451, 1, 1, 3), np.uint8)
        with patch('dataset.mcd_preprocessing.read_sync', return_value=(t, ppg, np.zeros(451), 0)), \
             patch('dataset.mcd_preprocessing.decode_faces', return_value=(frames, valid, 25, {})):
            clips, audit = aligned_clips(dict(video='synthetic'))
        self.assertEqual([c[3] for c in clips], [0, 2])
        self.assertAlmostEqual(clips[1][2][0], 12.)
        self.assertEqual(audit['rejected'], [dict(chunk=1, reason='face_detection')])

    def test_state_metrics_average_predictions_before_errors(self):
        groups = {}
        Trainer._collect_groups(groups, ['a', 'a', 'b'], torch.tensor([90., 110., 130.]),
                                torch.tensor([70., 90., 85.]),
                                dict(sbp=torch.tensor([100., 100., 120.]), dbp=torch.tensor([80., 80., 80.])))
        np.testing.assert_allclose(Trainer._group_errors(groups), [[0, 0], [10, 5]])

    def test_label_free_outputs_and_hr(self):
        class DRP(torch.nn.Module):
            def forward(self, video):
                wave = torch.sin(torch.arange(150)[None, :]*2*torch.pi*1.2/25)
                return wave, wave.roll(2, -1)
        class BBP(torch.nn.Module):
            def forward(self, face, acral):
                assert face.shape == acral.shape == (1, 600)
                return torch.tensor([120.]), torch.tensor([80.])
        outputs = predict_signals(DRP(), BBP(), torch.zeros(1))
        self.assertEqual(outputs['facial_rppg'].shape, (1, 150))
        self.assertAlmostEqual(float(outputs['hr_facial'][0]), 72., delta=2.)
        self.assertTrue(bool(outputs['bp_order_valid'][0]))
        hr, valid = estimate_hr(np.zeros((1, 150)), 25)
        self.assertFalse(valid[0]); self.assertTrue(np.isnan(hr[0]))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
