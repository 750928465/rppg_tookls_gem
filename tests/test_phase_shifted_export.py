"""CPU synthetic checks for interrupted Stage-2 testing and portable bundles."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import torch
import yaml

from test_phase_shifted_stages import BBPTrainer, DRPTrainer, Loader, stage_config
from tools.export_phase_shifted_bundle import export_bundle


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.loaders = {k: Loader(s) for k, s in [('train', '1'), ('valid', '2'), ('test', '3')]}
        first = DRPTrainer(stage_config(self.root, 'DRPNet'), self.loaders)
        self.stage1 = first.save_model('Best', 'STAGE1')
        self.config = stage_config(self.root, 'BBPNet')
        second = BBPTrainer(self.config, self.loaders)
        self.stage2 = second.save_model('Best', 'STAGE2')
        self.output = self.root / 'export' / 'selected.pth'

    def test_separate_checkpoint_test_matches_exported_full_bundle(self):
        before = [Path(p).read_bytes() for p in (self.stage1, self.stage2)]
        export_bundle(self.stage1, self.stage2, self.output)
        bundle = torch.load(self.output, weights_only=True)
        self.assertEqual(bundle['export']['stage1_sha256'], hashlib.sha256(before[0]).hexdigest())
        self.assertEqual(bundle['export']['stage2_sha256'], hashlib.sha256(before[1]).hexdigest())
        for data, key in zip(before, ('drpnet', 'bbpnet')):
            import io
            original = torch.load(io.BytesIO(data), weights_only=True)
            for name, tensor in original[key].items():
                torch.testing.assert_close(bundle[key][name], tensor, rtol=0, atol=0)
        t = torch.arange(150, dtype=torch.float64)[None] / 25
        labels = dict(ppg=torch.sin(t.float()*2*torch.pi*1.2), time=t,
                      sbp=torch.tensor([110.]), dbp=torch.tensor([75.]),
                      bp_valid=torch.tensor([True]), max_sync_error=torch.tensor([.001]),
                      group_id=['3_before'])
        batch = (torch.rand(1, 3, 150, 128, 128)*255, labels, ['3_before_cam'], ['0'])
        self.config.TOOLBOX_MODE = 'only_test'
        self.config.INFERENCE.MODEL_PATH = self.stage2
        evaluator = BBPTrainer(self.config, {'test': Loader('3')})
        separate_metrics = evaluator.test({'test': Loader('3', batch)})
        outputs_file = evaluator.result_dir / 'BBPNet_STAGE2_outputs.pth'
        separate = torch.load(outputs_file, weights_only=True)['records'][0]
        full_config = copy.deepcopy(self.config)
        full_config.INFERENCE.MODEL_PATH = str(self.output)
        full_config.MODEL.PHASE_SHIFTED_RPPG.STAGE1_CHECKPOINT = ''
        full = BBPTrainer(full_config, {'test': Loader('3')})
        full_metrics = full.test({'test': Loader('3', batch)})
        together = torch.load(outputs_file, weights_only=True)['records'][0]
        for key in ('facial_rppg', 'acral_rppg', 'sbp', 'dbp'):
            torch.testing.assert_close(together[key], separate[key], rtol=0, atol=0)
        self.assertEqual(full_metrics['sbp_state_mmHg'], separate_metrics['sbp_state_mmHg'])
        self.assertEqual(before, [Path(p).read_bytes() for p in (self.stage1, self.stage2)])

    def test_refuses_mismatched_signature_splits_and_stage1_path(self):
        original = torch.load(self.stage2, weights_only=True)
        for field, value, message in [
            ('signature', {}, 'signatures differ'),
            ('subject_splits', {}, 'subject splits differ'),
            ('stage1_checkpoint', '/unrelated/DRPNet.pth', 'Stage-1 path differs'),
        ]:
            with self.subTest(field=field):
                modified = dict(original, **{field: value})
                torch.save(modified, self.stage2)
                with self.assertRaisesRegex(ValueError, message):
                    export_bundle(self.stage1, self.stage2, self.output)
                self.assertFalse(self.output.exists())

    def test_refuses_incompatible_weights(self):
        payload = torch.load(self.stage2, weights_only=True)
        payload['bbpnet'].pop(next(iter(payload['bbpnet'])))
        torch.save(payload, self.stage2)
        with self.assertRaises(RuntimeError):
            export_bundle(self.stage1, self.stage2, self.output)
        self.assertFalse(self.output.exists())

    def test_refuses_overwriting_output(self):
        self.output.parent.mkdir()
        self.output.write_bytes(b'keep existing artifact')
        with self.assertRaises(FileExistsError):
            export_bundle(self.stage1, self.stage2, self.output)
        self.assertEqual(self.output.read_bytes(), b'keep existing artifact')

    def test_only_test_yaml_keeps_cache_split_device_and_weight_pair(self):
        directory = Path(__file__).resolve().parents[1] / 'configs/train_configs/phase_shifted_rppg'
        def merged(path):
            current = yaml.safe_load(path.read_text())
            result = {}
            def update(dst, src):
                for k, v in src.items():
                    if isinstance(v, dict) and isinstance(dst.get(k), dict):
                        update(dst[k], v)
                    else:
                        dst[k] = copy.deepcopy(v)
            for base in current.get('BASE', []):
                if base:
                    update(result, merged(path.parent / base))
            update(result, current)
            return result
        train = merged(directory / 'MCD_BBPNet.yaml')
        test = merged(directory / 'MCD_BBPNet_TEST.yaml')
        self.assertEqual(test['TOOLBOX_MODE'], 'only_test')
        self.assertEqual(test['MODEL']['NAME'], 'BBPNet')
        self.assertEqual(test['MODEL']['PHASE_SHIFTED_RPPG']['TRAIN_STAGE'], 'STAGE2')
        self.assertEqual(test['TEST']['DATA'], train['TEST']['DATA'])
        self.assertFalse(test['TEST']['DATA']['DO_PREPROCESS'])
        self.assertEqual(test['DEVICE'], 'cuda:1')
        self.assertEqual(test['INFERENCE']['BATCH_SIZE'], 1)
        self.assertEqual(test['INFERENCE']['MODEL_PATH'], train['INFERENCE']['MODEL_PATH'])
        self.assertEqual(test['MODEL']['PHASE_SHIFTED_RPPG']['STAGE1_CHECKPOINT'],
                         train['MODEL']['PHASE_SHIFTED_RPPG']['STAGE1_CHECKPOINT'])
        self.assertNotEqual(test['MODEL']['PHASE_SHIFTED_RPPG']['RESULT_DIR'],
                            train['MODEL']['PHASE_SHIFTED_RPPG']['RESULT_DIR'])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
