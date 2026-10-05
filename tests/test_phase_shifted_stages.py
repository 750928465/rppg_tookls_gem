"""Independent trainer lifecycle, weight handoff and output contracts; no real training."""
import ast
import copy
import csv
import importlib
import json
import sys
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import torch
import yaml

from test_phase_shifted_rppg import ROOT, config_fixture, namespace

# Load real stage trainers without unrelated legacy trainers/environment dependencies.
isolated = types.ModuleType('neural_methods.trainer')
isolated.__path__ = [str(ROOT/'neural_methods/trainer')]
with patch.dict(sys.modules, {'neural_methods.trainer': isolated}):
    package = importlib.import_module('neural_methods.trainer.phase_shifted_rppg')
DRPTrainer, BBPTrainer = package.DRPNetTrainer, package.BBPNetTrainer


def stage_config(root, name):
    def merge(a, b):
        for key, value in b.items():
            if isinstance(value, dict) and isinstance(a.get(key), dict):
                merge(a[key], value)
            else:
                a[key] = copy.deepcopy(value)
    def plain(obj):
        return {k: plain(v) for k, v in vars(obj).items()} if isinstance(obj, types.SimpleNamespace) else obj
    raw = plain(config_fixture(str(root)))
    directory = ROOT/'configs/train_configs/phase_shifted_rppg'
    merge(raw, yaml.safe_load((directory/'MCD_COMMON.yaml').read_text()))
    merge(raw, yaml.safe_load((directory/f'MCD_{name}.yaml').read_text()))
    raw['DEVICE'] = 'cpu'
    p = raw['MODEL']['PHASE_SHIFTED_RPPG']
    p['CHECKPOINT_DIR'] = str(root/name/'weights')
    p['RESULT_DIR'] = str(root/name/'results')
    p['STAGE1_CHECKPOINT'] = '' if name == 'DRPNet' else str(root/'DRPNet/weights/DRPNet_STAGE1_Best.pth')
    return namespace(raw)


class Loader:
    def __init__(self, subject, batch=None):
        self.dataset = namespace(dict(subject_ids={subject}))
        self.batch = batch
    def __len__(self):
        return 1
    def __iter__(self):
        yield self.batch


class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.drp_config = stage_config(self.root, 'DRPNet')
        self.bbp_config = stage_config(self.root, 'BBPNet')
        self.loaders = {k: Loader(v) for k, v in [('train', '1'), ('valid', '2'), ('test', '3')]}

    def save_drp(self):
        trainer = DRPTrainer(self.drp_config, self.loaders)
        path = trainer.save_model('Best', 'STAGE1')
        return trainer, path

    def test_stage1_only_owns_drp_and_saves_explicit_path(self):
        trainer, path = self.save_drp()
        self.assertIsNone(trainer.bbpnet)
        self.assertEqual(path, self.bbp_config.MODEL.PHASE_SHIFTED_RPPG.STAGE1_CHECKPOINT)
        payload = torch.load(path, weights_only=True)
        self.assertIn('drpnet', payload)
        self.assertNotIn('bbpnet', payload)
        self.assertEqual(payload['subject_splits']['test'], ['3'])

    def test_stage1_lifecycle_stops_before_bbp(self):
        trainer = DRPTrainer(self.drp_config, self.loaders)
        calls = []
        def fit(loaders, stage, epochs, lr):
            calls.append((stage, epochs, lr))
            trainer.best_paths[stage] = trainer.save_model('Best', stage)
        with patch.object(trainer, '_check_loaders'), patch.object(trainer.stage1_loss, 'check_dependencies'), \
             patch.object(trainer, '_train_stage', side_effect=fit):
            trainer.train(self.loaders)
        self.assertEqual(calls, [('STAGE1', 20, .001)])
        self.assertEqual(set(trainer.best_paths), {'STAGE1'})

    def test_best_epoch_is_restored_and_history_written_to_configured_directories(self):
        self.drp_config.TRAIN.EPOCHS = 2
        trainer = DRPTrainer(self.drp_config, self.loaders)
        epoch = 0
        def run_epoch(loader, stage, optimizer=None):
            nonlocal epoch
            if optimizer is not None:
                epoch += 1
                with torch.no_grad():
                    next(trainer.drpnet.parameters()).fill_(epoch)
                optimizer.step()  # No gradients: scheduler bookkeeping only.
            return dict(loss=float(epoch), facial_hr_mae=float(epoch), acral_hr_mae=float(epoch))
        with patch.object(trainer, '_check_loaders'), patch.object(trainer.stage1_loss, 'check_dependencies'), \
             patch.object(trainer, '_run_epoch', side_effect=run_epoch):
            trainer.train(self.loaders)
        self.assertTrue((next(trainer.drpnet.parameters()) == 1).all())
        history = json.loads((trainer.model_dir/'DRPNet_STAGE1_history.json').read_text())
        self.assertEqual(len(history), 2)
        ET.parse(trainer.result_dir/'DRPNet_STAGE1_history.svg')

    def test_stage2_loads_exact_drp_freezes_it_and_only_trains_bbp(self):
        first, _ = self.save_drp()
        trainer = BBPTrainer(self.bbp_config, self.loaders)
        for key, value in first.drpnet.state_dict().items():
            torch.testing.assert_close(trainer.drpnet.state_dict()[key], value)
        self.assertFalse(trainer.drpnet.training)
        self.assertTrue(all(not p.requires_grad for p in trainer.drpnet.parameters()))
        before = {k: v.clone() for k, v in trainer.drpnet.state_dict().items()}
        calls = []
        def fit(loaders, stage, epochs, lr):
            calls.append((stage, epochs, lr))
            trainer.best_paths[stage] = trainer.save_model('Best', stage)
        with patch.object(trainer, '_check_loaders'), \
             patch.object(trainer.stage1_loss, 'check_dependencies', side_effect=AssertionError('Must not use Stage 1 loss')), \
             patch.object(trainer, '_train_stage', side_effect=fit):
            trainer.train(self.loaders)
        self.assertEqual(calls, [('STAGE2', 200, .001)])
        for key, value in before.items():
            torch.testing.assert_close(trainer.drpnet.state_dict()[key], value)
        payload = torch.load(trainer.best_paths['FULL'], weights_only=True)
        self.assertTrue({'drpnet', 'bbpnet'} <= payload.keys())

    def test_stage2_rejects_missing_checkpoint_before_data(self):
        with self.assertRaisesRegex(ValueError, 'does not exist'):
            BBPTrainer.validate_config(self.bbp_config)

    def test_stage2_rejects_changed_subject_partition(self):
        self.save_drp()
        loaders = dict(self.loaders, test=Loader('4'))
        with self.assertRaisesRegex(ValueError, 'subject split differs'):
            BBPTrainer(self.bbp_config, loaders)

    def test_model_stage_mismatch_rejected(self):
        self.drp_config.MODEL.PHASE_SHIFTED_RPPG.TRAIN_STAGE = 'FULL'
        with self.assertRaisesRegex(ValueError, 'STAGE1'):
            DRPTrainer.validate_config(self.drp_config)

    def test_stage1_forward_shape_and_test_exports(self):
        trainer, checkpoint = self.save_drp()
        raw = torch.zeros(1, 3, 150, 128, 128)
        # Real DRPNet: verifies the standalone trainer does not need BBPNet.
        outputs = trainer.predict_clip(raw)
        self.assertEqual(tuple(outputs['facial_rppg'].shape), (1, 150))
        self.assertEqual(tuple(outputs['hr_facial'].shape), (1,))
        self.assertNotIn('sbp', outputs)
        self.drp_config.TOOLBOX_MODE = 'only_test'
        self.drp_config.INFERENCE.MODEL_PATH = checkpoint
        t = torch.arange(150, dtype=torch.float64)[None, :]/25
        labels = dict(ppg=torch.sin(t.float()*2*torch.pi*1.2), time=t,
                      max_sync_error=torch.tensor([.001]), group_id=['3_before'])
        batch = (raw, labels, ['3_before_cam'], ['0'])
        metrics = trainer.test({'test': Loader('3', batch)})
        directory = trainer.result_dir
        self.assertTrue((directory/'DRPNet_STAGE1_outputs.pth').is_file())
        self.assertTrue((directory/'DRPNet_STAGE1_metrics.json').is_file())
        with (directory/'DRPNet_STAGE1_windows.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['sbp_mmHg'], '')
        ET.parse(directory/'DRPNet_STAGE1_waveform_000.svg')
        self.assertIn('facial_hr_bpm', metrics)

    def test_stage2_only_test_full_bundle_and_bp_output(self):
        self.save_drp()
        trainer = BBPTrainer(self.bbp_config, self.loaders)
        full = trainer.save_model('Best', 'FULL')
        self.bbp_config.TOOLBOX_MODE = 'only_test'
        self.bbp_config.INFERENCE.MODEL_PATH = full
        self.bbp_config.MODEL.PHASE_SHIFTED_RPPG.STAGE1_CHECKPOINT = ''
        t = torch.arange(150, dtype=torch.float64)[None, :]/25
        labels = dict(ppg=torch.sin(t.float()*2*torch.pi*1.2), time=t,
                      sbp=torch.tensor([110.]), dbp=torch.tensor([75.]), bp_valid=torch.tensor([True]),
                      max_sync_error=torch.tensor([.001]), group_id=['3_before'])
        batch = (torch.zeros(1, 3, 150, 128, 128), labels, ['3_before_cam'], ['0'])
        evaluator = BBPTrainer(self.bbp_config, {'test': Loader('3', batch)})
        metrics = evaluator.test({'test': Loader('3', batch)})
        self.assertEqual(metrics['sbp_state_mmHg']['N_states'], 1)
        payload = torch.load(evaluator.result_dir/'BBPNet_STAGE2_outputs.pth', weights_only=True)
        record = payload['records'][0]
        self.assertEqual(tuple(record['sbp'].shape), (1,))
        self.assertEqual(tuple(record['facial_rppg'].shape), (1, 150))
        self.assertTrue((evaluator.result_dir/'BBPNet_STAGE2_states.csv').is_file())

    def test_main_routes_both_modes_to_the_independent_trainers(self):
        tree = ast.parse((ROOT/'main.py').read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('train_and_test', 'run_model_test')]
        events = []
        class Stub:
            def __init__(self, config, loaders): events.append(config.MODEL.NAME)
            def train(self, loaders): events.append('train')
            def test(self, loaders): events.append('test')
        scope = {'trainer': namespace(dict(phase_shifted_rppg=dict(DRPNetTrainer=Stub, BBPNetTrainer=Stub)))}
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'main.py', 'exec'), scope)
        for name in ('DRPNet', 'BBPNet'):
            config = namespace(dict(MODEL=dict(NAME=name)))
            scope['train_and_test'](config, {})
            scope['run_model_test'](config, {})
        self.assertEqual(events, ['DRPNet', 'train', 'test', 'DRPNet', 'test', 'BBPNet', 'train', 'test', 'BBPNet', 'test'])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
