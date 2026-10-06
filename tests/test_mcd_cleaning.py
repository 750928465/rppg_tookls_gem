"""Cleaning completion, recovery, source provenance and loader handoff contracts."""
import csv
from datetime import datetime, timedelta
import importlib.util
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from dataset.mcd.cleaning import clean_dataset, digest_json, load_clean_manifest
from dataset.mcd.video import DataQualityError, audit_video, inspect_avi_tail
from dataset.mcd_preprocessing import decode_faces

ROOT = Path(__file__).resolve().parents[1]


def make_source(root, ids=('1', '2')):
    for d in ('video', 'meta', 'ppg', 'ppg_sync'):
        (root / d).mkdir(parents=True, exist_ok=True)
    rows = []
    origin = datetime(2026, 1, 1)
    for subject in ids:
        row = dict(patient_id=subject, step='before', camera='FullHDwebcam', upper_ap='120', lower_ap='80',
                   video=f'video/{subject}.avi', meta=f'meta/{subject}.txt',
                   ppg=f'ppg/{subject}.PW', ppg_sync=f'ppg_sync/{subject}.txt')
        (root / row['video']).write_bytes(b'synthetic_video')
        times = [(origin + timedelta(seconds=i / 25)).isoformat() for i in range(150)]
        values = np.sin(np.arange(150) / 4).astype(np.float32)
        (root / row['meta']).write_text('\n'.join(f'{i+1} {t}' for i, t in enumerate(times)))
        (root / row['ppg']).write_text('\n'.join(f'{v} {t}' for v, t in zip(values, times)))
        np.savetxt(root / row['ppg_sync'], np.column_stack([values, np.zeros(150)]))
        rows.append(row)
    with (root / 'db.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def fake_audit(*args, **kwargs):
    return dict(status='PASS', strict=dict(frames=150, width=128, height=128), recovery=None)


class CleaningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw = Path(self.tmp.name) / 'raw'
        self.raw.mkdir()
        self.out = Path(self.tmp.name) / 'audit'
        make_source(self.raw)

    def clean(self, **kwargs):
        with patch('dataset.mcd.cleaning.subprocess.run', return_value=types.SimpleNamespace(stdout='ffmpeg fixture\n')):
            return clean_dataset(self.raw, self.out, **kwargs)

    def test_complete_resume_and_actual_ppg_mapping(self):
        with patch('dataset.mcd.cleaning.audit_video', side_effect=fake_audit) as decoder:
            summary = self.clean()
            self.assertEqual(summary['accepted_records'], 2)
            self.assertEqual(decoder.call_count, 2)
            self.assertEqual(self.clean()['resumed'], 2)
            self.assertEqual(decoder.call_count, 2)
            self.clean(force=True)
            self.assertEqual(decoder.call_count, 4)
        rows, payload = load_clean_manifest(self.out / 'accepted_manifest.json', self.raw, ['FullHDwebcam'])
        self.assertTrue(payload['complete'])
        self.assertEqual(rows[0]['clean_approval']['frames'], 150)
        self.assertEqual(len((self.out / 'records.jsonl').read_text().splitlines()), 4)

    def test_data_failure_does_not_abort_but_io_failure_does(self):
        def bad(path, *a):
            if Path(path).stem == '1':
                raise DataQualityError('synthetic internal frame damage')
            return fake_audit()
        with patch('dataset.mcd.cleaning.audit_video', side_effect=bad):
            summary = self.clean()
        self.assertEqual(summary['status_counts'], {'REJECTED': 1, 'PASS': 1})
        self.assertIn('internal frame damage', (self.out / 'rejected.csv').read_text())
        with patch('dataset.mcd.cleaning.audit_video', side_effect=OSError('disk failure')):
            with self.assertRaisesRegex(OSError, 'disk failure'):
                self.clean(force=True)
        self.assertFalse(json.loads((self.out / 'accepted_manifest.json').read_text())['complete'])
        self.assertFalse((self.out / '.clean.lock').exists())

    def test_interrupt_resume_retains_completed_record(self):
        with patch('dataset.mcd.cleaning.audit_video', side_effect=[fake_audit(), KeyboardInterrupt()]):
            with self.assertRaises(KeyboardInterrupt):
                self.clean()
        with patch('dataset.mcd.cleaning.audit_video', side_effect=fake_audit) as decoder:
            self.assertEqual(self.clean()['resumed'], 1)
            self.assertEqual(decoder.call_count, 1)

    def test_partial_output_cannot_be_loaded(self):
        with patch('dataset.mcd.cleaning.audit_video', side_effect=fake_audit):
            self.clean(limit=1)
        with self.assertRaisesRegex(ValueError, 'partial'):
            load_clean_manifest(self.out / 'accepted_manifest.json', self.raw, ['FullHDwebcam'])

    def test_source_or_db_change_requires_reaudit(self):
        with patch('dataset.mcd.cleaning.audit_video', side_effect=fake_audit):
            self.clean()
        (self.raw / 'video/1.avi').write_bytes(b'changed_source')
        with self.assertRaisesRegex(ValueError, 'Stale'):
            load_clean_manifest(self.out / 'accepted_manifest.json', self.raw, ['FullHDwebcam'])
        with patch('dataset.mcd.cleaning.audit_video', side_effect=fake_audit) as decoder:
            self.assertEqual(self.clean()['resumed'], 1)
            self.assertEqual(decoder.call_count, 1)
        with (self.raw / 'db.csv').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'db.csv changed'):
            load_clean_manifest(self.out / 'accepted_manifest.json', self.raw, ['FullHDwebcam'])

    def test_invalid_bp_is_kept_for_ppg_and_missing_files_are_recorded(self):
        csv_path = self.raw / 'db.csv'
        csv_path.write_text(csv_path.read_text().replace(',120,80,', ',,80,'))
        (self.raw / 'video/2.avi').unlink()
        with patch('dataset.mcd.cleaning.audit_video', side_effect=fake_audit):
            summary = self.clean()
        self.assertEqual(summary['invalid_bp_accepted'], 1)
        rows, _ = load_clean_manifest(self.out / 'accepted_manifest.json', self.raw, ['FullHDwebcam'])
        self.assertFalse(rows[0]['bp_valid'])
        self.assertEqual(rows[0]['sbp'], 0.)
        self.assertIn('Missing source files', (self.out / 'rejected.csv').read_text())

    def test_manifest_mutation_and_concurrent_writer_rejected(self):
        with patch('dataset.mcd.cleaning.audit_video', side_effect=fake_audit):
            self.clean()
        p = self.out / 'accepted_manifest.json'
        content = json.loads(p.read_text()); content['records'][0]['usable_frames'][1] = 10
        p.write_text(json.dumps(content))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_clean_manifest(p, self.raw, ['FullHDwebcam'])
        (self.out / '.clean.lock').write_text('123')
        with self.assertRaisesRegex(RuntimeError, 'lock'):
            self.clean()

    def test_loader_uses_only_approved_records_and_saves_split(self):
        with patch('dataset.mcd.cleaning.audit_video', side_effect=[DataQualityError('bad video'), fake_audit()]):
            self.clean()
        package = types.ModuleType('dataset.data_loader'); package.__path__ = [str(ROOT / 'dataset/data_loader')]
        base = types.ModuleType('dataset.data_loader.BaseLoader'); base.BaseLoader = object
        spec = importlib.util.spec_from_file_location('mcd_loader_fixture', ROOT / 'dataset/data_loader/MCDRPPGLoader.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'dataset.data_loader': package, 'dataset.data_loader.BaseLoader': base}):
            spec.loader.exec_module(module)
        ns = types.SimpleNamespace
        config = ns(DATA_FORMAT='NCDHW', FS=25, BEGIN=0., END=1., CACHED_PATH=str(self.out / 'cache'), DO_PREPROCESS=True,
                    PREPROCESS=ns(DATA_TYPE=['Raw'], LABEL_TYPE='Raw', USE_PSUEDO_PPG_LABEL=False,
                                  CHUNK_LENGTH=150, RESIZE=ns(H=128,W=128), DO_CHUNK=True, DATA_AUG=['None']),
                    MCD=ns(CAMERAS=['FullHDwebcam'], CLEAN_MANIFEST=str(self.out / 'accepted_manifest.json'),
                           SPLIT_SEED=100, MAX_SYNC_ERROR=.02, MAX_FRAME_GAP=.1, MAX_TAIL_DROP=32))
        def clips(row, **kwargs):
            self.assertEqual(row['recording_id'], '2_before_FullHDwebcam')
            self.assertEqual(row['clean_approval']['frames'], 150)
            return [(np.zeros((150,128,128,3), np.uint8), np.sin(np.arange(150)/4), np.arange(150)/25, 0, .001)], {'rejected': []}
        with patch.object(module, 'aligned_clips', side_effect=clips):
            loader = module.MCDRPPGLoader('train', self.raw, config)
        self.assertEqual(loader[0][0].shape, (3,150,128,128))
        self.assertEqual(loader[0][1]['ppg'].shape, (150,))
        self.assertEqual(json.loads((Path(loader.cached_path)/'train_split.json').read_text())['subjects'], ['2'])


@unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg executable required for real decode fixture')
class VideoTests(unittest.TestCase):
    def test_clean_tail_and_internal_damage_with_real_ffmpeg(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'valid.avi'
            subprocess.run(['ffmpeg','-hide_banner','-v','error','-f','lavfi','-i','testsrc=size=32x32:rate=5',
                            '-frames:v','8','-c:v','mpeg4','-bf','0','-pix_fmt','yuv420p',str(path)], check=True)
            self.assertEqual(audit_video(path)['status'], 'PASS')
            info = inspect_avi_tail(path)
            data = path.read_bytes(); last = info['last_packet']
            tail = Path(d) / 'tail.avi'
            tail.write_bytes(data[:last['offset'] + 8 + 10])
            recovered = audit_video(tail)
            self.assertEqual(recovered['status'], 'RECOVERED_PREFIX')
            self.assertEqual(recovered['strict']['frames'], 7)
            # Damage the first encoded packet but keep the container's later packets.
            position = data.index(b'00dc', data.index(b'movi') + 4)
            length = struct.unpack_from('<I', data, position + 4)[0]
            broken = bytearray(data); broken[position+8:position+8+length] = b'\0'*length
            internal = Path(d) / 'internal.avi'; internal.write_bytes(broken)
            with self.assertRaises(DataQualityError):
                audit_video(internal)


class ApprovedPrefixTests(unittest.TestCase):
    def test_face_decoder_accepts_only_the_exact_approved_prefix(self):
        cv2 = types.ModuleType('cv2')
        cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS = 1, 2, 3
        cv2.COLOR_RGB2GRAY, cv2.INTER_AREA = 4, 5
        cv2.VideoCapture = lambda path: types.SimpleNamespace(get=lambda k: 25 if k == 3 else 2, release=lambda: None)
        cv2.CascadeClassifier = lambda path: types.SimpleNamespace(empty=lambda: False, detectMultiScale=lambda *a, **k: [(0,0,2,2)])
        cv2.cvtColor = lambda data, mode: data
        cv2.resize = lambda data, shape, **k: np.zeros((128,128,3), np.uint8)
        commands = []
        def process(command, **kwargs):
            commands.append(command)
            kwargs['stderr'].write(b'final packet corrupt')
            return types.SimpleNamespace(stdout=io.BytesIO(bytes(2*2*3*2)), wait=lambda: 1, poll=lambda: 1)
        approval = dict(frames=2,width=2,height=2,status='RECOVERED_PREFIX')
        with patch.dict(sys.modules, {'cv2': cv2}), \
                patch('dataset.mcd_preprocessing.shutil.which', return_value='ffmpeg'), \
                patch('dataset.mcd_preprocessing.subprocess.Popen', side_effect=process):
            frames, _, _, audit = decode_faces('fixture.avi', 3, approval=approval)
            self.assertEqual(len(frames), 2)
            self.assertEqual(audit['ffmpeg_returncode'], 1)
            self.assertIn('-frames:v', commands[-1])
            with self.assertRaisesRegex(ValueError, 'Approved prefix unavailable'):
                decode_faces('fixture.avi', 3, approval=dict(approval, frames=3))
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                decode_faces('fixture.avi', 2)


if __name__ == '__main__':
    unittest.main()
