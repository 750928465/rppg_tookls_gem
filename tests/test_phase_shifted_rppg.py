"""Contract/lifecycle unit tests; no datasets, training, or environment installs.

The trainer is loaded with the real BaseTrainer but without executing the legacy
trainer/__init__.py eager imports. This isolates unit tests from unrelated model
dependencies; it is NOT evidence that the full main.py environment is installed.
"""
import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def isolated_trainer_class():
    name = 'neural_methods.trainer'
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT / 'neural_methods/trainer')]
    with patch.dict(sys.modules, {name: package}):
        spec = importlib.util.spec_from_file_location(
            '_phase_shifted_trainer_under_test', ROOT / 'neural_methods/trainer/PhaseShiftedRPPGTrainer.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.PhaseShiftedRPPGTrainer


Trainer = isolated_trainer_class()


def namespace(value):
    return types.SimpleNamespace(**{k: namespace(v) for k, v in value.items()}) if isinstance(value, dict) else value


def config_fixture(directory):
    # Extract literal model defaults, not a replacement/mock implementation of yacs.
    defaults = {}
    for node in ast.parse((ROOT / 'config.py').read_text()).body:
        if isinstance(node, ast.Assign):
            target = ast.unparse(node.targets[0])
            if target.startswith('_C.MODEL.PHASE_SHIFTED_RPPG.'):
                defaults[target.rsplit('.', 1)[1]] = ast.literal_eval(node.value)
    raw = yaml.safe_load((ROOT / 'configs/train_configs/PHASE_SHIFTED_RPPG_TEMPLATE.yaml').read_text())
    defaults.update(raw['MODEL']['PHASE_SHIFTED_RPPG'])
    raw['MODEL']['PHASE_SHIFTED_RPPG'] = defaults
    raw['MODEL']['MODEL_DIR'] = directory
    raw['TEST']['OUTPUT_SAVE_DIR'] = directory
    return namespace(raw)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = config_fixture(self.tmp.name)

    def verified(self):
        self.config.MODEL.PHASE_SHIFTED_RPPG.LABEL_SOURCE = 'ABP_MMHG'
        return self.config

    def test_unverified_template_stops_before_data(self):
        with self.assertRaisesRegex(ValueError, 'UNVERIFIED'):
            Trainer.validate_config(self.config)

    def test_config_contract_and_layout(self):
        Trainer.validate_config(self.verified())
        self.config.TRAIN.DATA.DATA_FORMAT = 'NDCHW'
        with self.assertRaisesRegex(ValueError, 'NCDHW'):
            Trainer.validate_config(self.config)

    def test_stage2_rejects_plain_waveform_and_does_not_create_labels(self):
        trainer = Trainer(self.verified(), {})
        batch = (torch.zeros(1, 3, 150, 128, 128), torch.ones(1, 150) * 100)
        with self.assertRaisesRegex(ValueError, 'real abp, sbp, dbp'):
            trainer._batch(batch, 'STAGE2')
        _, labels = trainer._batch(batch, 'STAGE1')
        self.assertEqual(set(labels), {'abp'})
        torch.testing.assert_close(labels['abp'], batch[1])

    def test_explicit_rgb_bgr_and_scaling(self):
        trainer = Trainer(self.verified(), {})
        video = torch.zeros(1, 3, 150, 128, 128)
        video[:, 0] = 255
        converted, _ = trainer._batch((video, torch.ones(1, 150) * 100), 'STAGE1')
        self.assertTrue((converted[:, 2] == 1).all())
        self.assertTrue((converted[:, 0] == 0).all())

    def test_checkpoint_full_roundtrip_and_dataparallel_prefix(self):
        trainer = Trainer(self.verified(), {})
        path = trainer.save_model('Best', 'FULL')
        state = {k: v.clone() for k, v in trainer.drpnet.state_dict().items()}
        with torch.no_grad():
            next(trainer.drpnet.parameters()).add_(1)
        trainer._load_checkpoint(path, 'FULL')
        for key, value in trainer.drpnet.state_dict().items():
            torch.testing.assert_close(value, state[key])
        raw_path = Path(self.tmp.name) / 'upstream.pth'
        torch.save({'module.' + k: v for k, v in state.items()}, raw_path)
        trainer._load_checkpoint(raw_path, 'STAGE1')
        trainer.p.PIXEL_SCALE = 1.
        with self.assertRaisesRegex(ValueError, 'signature'):
            trainer._load_checkpoint(path, 'FULL')

    def test_frozen_stage1_and_stage2_gradient(self):
        trainer = Trainer(self.verified(), {})

        class TinyDRP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))

            def forward(self, video):
                signal = torch.sin(torch.arange(150, dtype=torch.float32)[None, :] * 0.3)
                signal = signal.expand(video.shape[0], -1) * self.weight
                return signal, signal.roll(2, -1)

        trainer.drpnet = TinyDRP()
        trainer.bbpnet.train()
        sbp, dbp, _ = trainer._stage2_forward(torch.zeros(2, 1))
        (sbp.sum() + dbp.sum()).backward()
        self.assertFalse(trainer.drpnet.training)
        self.assertFalse(trainer.drpnet.weight.requires_grad)
        self.assertIsNone(trainer.drpnet.weight.grad)
        self.assertTrue(any(p.grad is not None for p in trainer.bbpnet.parameters()))

    def test_full_lifecycle_loads_selected_stage1_before_stage2(self):
        trainer = Trainer(self.verified(), {})
        calls = []

        def stage_without_training(loaders, stage, epochs, lr):
            calls.append(stage)
            if stage == 'STAGE2':
                self.assertFalse(trainer.drpnet.training)
                self.assertTrue(all(not p.requires_grad for p in trainer.drpnet.parameters()))
            trainer.best_paths[stage] = trainer.save_model('Best', stage)

        with patch.object(trainer, '_check_loaders') as check, \
             patch.object(trainer.stage1_loss, 'check_dependencies'), \
             patch.object(trainer, '_train_stage', side_effect=stage_without_training):
            trainer.train({})
        check.assert_called_once_with({}, 'STAGE2')
        self.assertEqual(calls, ['STAGE1', 'STAGE2'])
        payload = torch.load(trainer.best_paths['FULL'], weights_only=True)
        self.assertTrue({'drpnet', 'bbpnet'} <= payload.keys())

    def test_stage2_pretrained_mode_does_not_run_stage1(self):
        self.verified().MODEL.PHASE_SHIFTED_RPPG.TRAIN_STAGE = 'STAGE2'
        with self.assertRaisesRegex(ValueError, 'STAGE1_CHECKPOINT'):
            Trainer.validate_config(self.config)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
