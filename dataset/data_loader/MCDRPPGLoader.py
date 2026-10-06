"""Toolbox MCD adapter with PPG + state-level cuff BP (not continuous ABP)."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm
from dataset.data_loader.BaseLoader import BaseLoader
from dataset.mcd_preprocessing import inventory, split_subjects, aligned_clips, cache_signature, VERSION


class MCDRPPGLoader(BaseLoader):
    # Override cache management: BaseLoader assumes one waveform .npy label and
    # silently reconstructs file lists. Multi-target labels require a manifest.
    def __init__(self, name, data_path, config_data, device=None):
        self.config_data = config_data
        self.dataset_name = name
        self.data_format = config_data.DATA_FORMAT
        p, m = config_data.PREPROCESS, config_data.MCD
        if (self.data_format != 'NCDHW' or list(p.DATA_TYPE) != ['Raw'] or
                p.LABEL_TYPE != 'Raw' or p.USE_PSUEDO_PPG_LABEL or
                p.CHUNK_LENGTH != 150 or p.RESIZE.H != 128 or p.RESIZE.W != 128 or
                config_data.FS != 25 or not p.DO_CHUNK or list(p.DATA_AUG) != ['None']):
            raise ValueError('MCD v1 requires Raw RGB/PPG, NCDHW, 25 Hz, 150 frames, 128x128, no augmentation.')
        print(f'MCD {name}: scanning recordings in {data_path}', flush=True)
        rows, missing = inventory(data_path, tuple(m.CAMERAS))
        self.rows = split_subjects(rows, config_data.BEGIN, config_data.END, m.SPLIT_SEED)
        self.subject_ids = {r['patient_id'] for r in self.rows}
        if not self.rows:
            raise ValueError(f'{name}: no subjects in split. A one-subject demo cannot create independent train/valid/test sets.')
        params = {'fs': config_data.FS, 'length': p.CHUNK_LENGTH,
                  'max_sync_error': m.MAX_SYNC_ERROR, 'max_frame_gap': m.MAX_FRAME_GAP,
                  'max_tail_drop': m.MAX_TAIL_DROP}
        signature = cache_signature(self.rows, params)
        # Signature includes labels, source paths/size/mtime and preprocessing version.
        self.cached_path = str(Path(config_data.CACHED_PATH) / signature[:16])
        directory = Path(self.cached_path)
        manifest = directory / f'{name}_manifest.json'
        print(f'MCD {name}: {len(self.subject_ids)} subjects, {len(self.rows)} videos; '
              f'cache={directory}', flush=True)
        if config_data.DO_PREPROCESS and not manifest.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print(f'MCD {name}: no completed manifest; processing all videos in this split. '
                  'Existing window files do not provide per-video resume.', flush=True)
            entries, audit = [], []
            rejected_windows = 0
            # Count a video only after its accepted windows have been saved.
            with tqdm(total=len(self.rows), desc=f'MCD {name} preprocessing',
                      unit='video', dynamic_ncols=True) as progress:
                for row in self.rows:
                    progress.set_postfix(video=row['recording_id'], saved=len(entries),
                                         rejected=rejected_windows)
                    try:
                        clips, info = aligned_clips(row, **params)
                        audit.append(dict(info, recording_id=row['recording_id']))
                        for video, ppg, times, chunk, sync_error in clips:
                            stem = f"{row['recording_id']}_chunk{chunk}"
                            np.save(directory / f'{stem}_input.npy', video)
                            np.savez_compressed(directory / f'{stem}_label.npz', ppg=ppg,
                                                sbp=np.float32(row['sbp'] if row['bp_valid'] else 0),
                                                dbp=np.float32(row['dbp'] if row['bp_valid'] else 0),
                                                bp_valid=np.bool_(row['bp_valid']), time=times,
                                                max_sync_error=np.float32(sync_error))
                            entries.append({'stem': stem, 'recording_id': row['recording_id'],
                                            'group_id': row['group_id'], 'subject_id': row['patient_id'],
                                            'chunk': chunk, 'bp_valid': row['bp_valid']})
                    except Exception as exc:
                        tqdm.write(f"MCD {name}: failed recording={row['recording_id']}; "
                                   f"video={row['video']}; {type(exc).__name__}: {exc}",
                                   file=progress.fp)
                        raise
                    rejected_windows += len(info['rejected'])
                    progress.set_postfix(video=row['recording_id'], saved=len(entries),
                                         rejected=rejected_windows, refresh=False)
                    progress.update(1)
            payload = {'signature': signature, 'version': VERSION, 'params': params,
                       'entries': entries, 'audit': audit, 'missing_files': missing}
            temporary = manifest.with_suffix('.tmp')
            temporary.write_text(json.dumps(payload, indent=2))
            temporary.replace(manifest)
            print(f'MCD {name}: preprocessing finished; saved={len(entries)}, '
                  f'rejected={rejected_windows}; manifest={manifest}', flush=True)
        elif manifest.exists():
            print(f'MCD {name}: loading existing manifest={manifest}', flush=True)
        if not manifest.exists():
            raise ValueError(f'Cache absent: {manifest}. Run preprocessing on the server first.')
        payload = json.loads(manifest.read_text())
        if payload['signature'] != signature:
            raise ValueError('MCD cache signature mismatch.')
        self.entries = payload['entries']
        if not self.entries:
            raise ValueError(f'No quality-approved clips; inspect {manifest}')
        counts = Counter(e['group_id'] for e in self.entries)
        self.weights = [len(self.entries)/(len(counts)*counts[e['group_id']]) for e in self.entries]
        self.preprocessed_data_len = len(self.entries)
        # Toolbox-style CSV is informational; the validated manifest remains authoritative.
        if config_data.DO_PREPROCESS:
            with (directory / f'{name}_files.csv').open('w', newline='') as f:
                writer = csv.writer(f); writer.writerow(['input_files'])
                writer.writerows([[str(directory / f"{e['stem']}_input.npy")] for e in self.entries])
        print(f'MCD {name}: {len(self.subject_ids)} subjects, {len(self.entries)} windows; manifest={manifest}; missing records={len(missing)}')

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        path = Path(self.cached_path)
        video = np.load(path / f"{entry['stem']}_input.npy", allow_pickle=False)
        with np.load(path / f"{entry['stem']}_label.npz", allow_pickle=False) as data:
            labels = {k: data[k].copy() for k in data.files}
        labels['group_id'] = entry['group_id']
        labels['weight'] = np.float32(self.weights[index])
        return video.transpose(3, 0, 1, 2).astype(np.float32), labels, entry['recording_id'], str(entry['chunk'])
