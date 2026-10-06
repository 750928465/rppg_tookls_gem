"""Resumable MCD cleaning manifest, independent of models and dataset splits."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from tqdm import tqdm

from dataset.mcd_preprocessing import read_sync
from .video import DataQualityError, audit_video

SCHEMA = 'mcd_clean_v1'
SOURCE_FIELDS = ('video', 'meta', 'ppg', 'ppg_sync')


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def scan_records(root, cameras):
    root = Path(root).resolve()
    rows, seen = [], set()
    with (root / 'db.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'patient_id', 'step', 'camera', 'upper_ap', 'lower_ap', *SOURCE_FIELDS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'db.csv lacks fields: {sorted(required - set(reader.fieldnames or []))}')
        for raw in reader:
            if raw['camera'] not in cameras:
                continue
            key = (raw['patient_id'], raw['step'], raw['camera'])
            if key in seen:
                raise ValueError(f'Duplicate MCD recording: {key}')
            seen.add(key)
            row = {k: raw[k] for k in ('patient_id', 'step', 'camera')}
            row.update(recording_id='_'.join(key), group_id='_'.join(key[:2]))
            row['path_error'] = None
            for field in SOURCE_FIELDS:
                p = (root / raw[field]).resolve()
                if not raw[field] or not p.is_relative_to(root):
                    row['path_error'] = f'Invalid dataset path: {field}'
                row[field] = str(p)
            for dst, src in [('sbp', 'upper_ap'), ('dbp', 'lower_ap')]:
                try:
                    value = float(raw[src])
                except (TypeError, ValueError):
                    value = math.nan
                row[dst] = value if math.isfinite(value) else None
            row['bp_valid'] = bool(row['sbp'] is not None and row['dbp'] is not None
                                   and row['sbp'] > row['dbp'] > 0)
            rows.append(row)
    return sorted(rows, key=lambda r: r['recording_id'])


def source_fingerprint(row, memo=None):
    """Full SHA-256 during audit, with a per-run cache for shared PPG files."""
    memo = {} if memo is None else memo
    result = {}
    for field in SOURCE_FIELDS:
        p = Path(row[field])
        try:
            stat = p.stat()
        except FileNotFoundError:
            result[field] = None
            continue
        if not p.is_file():
            result[field] = None
            continue
        key = (str(p), stat.st_size, stat.st_mtime_ns)
        if key not in memo:
            sha = hashlib.sha256()
            with p.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    sha.update(block)
            after = p.stat()
            if (after.st_size, after.st_mtime_ns) != key[1:]:
                raise DataQualityError(f'Source changed during hashing: {p}')
            memo[key] = sha.hexdigest()
        result[field] = {'path': str(p), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                         'sha256': memo[key]}
    return result


def validate_source_stats(sources):
    for expected in sources.values():
        if expected is None:
            continue
        stat = Path(expected['path']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (expected['size'], expected['mtime_ns']):
            raise DataQualityError(f"Source changed since audit started: {expected['path']}")


def audit_record(row, params, log_dir):
    if row['path_error']:
        raise DataQualityError(row['path_error'])
    missing = [k for k in SOURCE_FIELDS if not Path(row[k]).is_file()]
    if missing:
        raise DataQualityError(f'Missing source files: {missing}')
    # Existing MCD checks verify acquisition frame IDs and the raw PPG nearest
    # sample mapping. This never fabricates a pressure waveform from PPG.
    t, ppg, sync_error, meta_extra = read_sync(row)
    video = audit_video(row['video'], params['ffmpeg'], params['timeout'], log_dir)
    decoded, reference = video['strict']['frames'], len(t)
    if decoded > reference + meta_extra or abs(reference - decoded) > params['max_tail_drop']:
        raise DataQualityError(f'Video/reference count mismatch: decoded={decoded}, '
                               f'ppg_sync={reference}, meta={reference + meta_extra}')
    stop = min(decoded, reference)
    if stop < 2:
        raise DataQualityError('No aligned reference prefix.')
    t, ppg, sync_error = t[:stop], ppg[:stop], sync_error[:stop]
    if np.std(ppg) < 1e-6:
        raise DataQualityError('Entire reference PPG is flat.')
    # Retain local quality defects without removing frames or shifting indices.
    sync_bad = np.flatnonzero(sync_error > params['max_sync_error']).tolist()
    gap_bad = np.flatnonzero(np.diff(t) > params['max_frame_gap']).tolist()
    return {'status': video['status'], 'usable_frames': [0, stop],
            'usable_time_seconds': [float(t[0]), float(t[-1])],
            'decode': video, 'reference_samples': reference,
            'unused_meta_suffix': meta_extra, 'unused_reference_suffix': reference - stop,
            'unused_video_suffix': decoded - stop,
            'quality': {'sync_error_frames': sync_bad, 'frame_gap_after_frames': gap_bad,
                        'max_sync_error_seconds': float(sync_error.max()),
                        'max_frame_gap_seconds': float(np.diff(t).max())}}


def clean_dataset(root, output, cameras=('FullHDwebcam',), ffmpeg='ffmpeg', timeout=300,
                  max_tail_drop=32, max_sync_error=.02, max_frame_gap=.10,
                  force=False, recording_ids=None, limit=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    # Separate audit output from raw data and prevent two cleaners writing together.
    if output == root or output.is_relative_to(root):
        raise ValueError('Choose an audit output directory outside the raw dataset.')
    if root.is_relative_to(output):
        raise ValueError('The audit output must not be an ancestor of the raw dataset.')
    if timeout <= 0 or max_tail_drop < 0 or max_sync_error <= 0 or max_frame_gap <= 0:
        raise ValueError('Invalid audit thresholds or timeout.')
    ffmpeg_version = subprocess.run([ffmpeg, '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    params = dict(ffmpeg=ffmpeg, ffmpeg_version=ffmpeg_version, timeout=timeout,
                  max_tail_drop=max_tail_drop, max_sync_error=max_sync_error, max_frame_gap=max_frame_gap)
    db_source = source_fingerprint({k: str(root / 'db.csv') for k in SOURCE_FIELDS})['video']
    rows = scan_records(root, cameras)
    scope_complete = recording_ids is None and limit is None
    if recording_ids is not None:
        requested = set(recording_ids)
        unknown = requested - {r['recording_id'] for r in rows}
        if unknown:
            raise ValueError(f'Unknown recording IDs for selected cameras: {sorted(unknown)}')
        rows = [r for r in rows if r['recording_id'] in requested]
    if limit is not None:
        if limit < 1:
            raise ValueError('--limit must be positive.')
        rows = rows[:limit]
    if not rows:
        raise ValueError('No MCD records for the requested cameras/scope.')
    output.mkdir(parents=True, exist_ok=True)
    state_dir, log_dir = output / 'records', output / 'decode_logs'
    state_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    lock = output / '.clean.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f'Cleaner lock exists: {lock}. Check its PID; remove only after confirming no cleaner is running.') from exc
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(str(os.getpid()))
        # A prior manifest must not advertise completion of this new audit run.
        atomic_json(output / 'accepted_manifest.json', {'schema': SCHEMA, 'complete': False, 'records': []})
        memo, records, resumed = {}, [], 0
        with (output / 'audit.log').open('a', encoding='utf-8') as log, \
                (output / 'records.jsonl').open('a', encoding='utf-8') as journal, \
                tqdm(total=len(rows), desc='MCD cleaning', unit='video', dynamic_ncols=True) as bar:
            def note(message):
                line = f'{datetime.now(timezone.utc).isoformat()} {message}'
                log.write(line + '\n'); log.flush()
                tqdm.write(message)
            note(f'Scanning {len(rows)} records, cameras={list(cameras)}; output={output}; '
                 'this command audits videos and reference signals only.')
            for row in rows:
                started = time.monotonic()
                bar.set_postfix(video=row['recording_id'], resumed=resumed)
                fingerprint = ({k: None for k in SOURCE_FIELDS} if row['path_error']
                               else source_fingerprint(row, memo))
                identity = digest_json([SCHEMA, params, row, fingerprint])
                token = hashlib.sha256(row['recording_id'].encode()).hexdigest()[:24]
                state_path = state_dir / (token + '.json')
                previous = json.loads(state_path.read_text()) if state_path.exists() else None
                if not force and previous and previous.get('identity') == identity:
                    record = previous
                    resumed += 1
                else:
                    detail_dir = log_dir / token
                    detail_dir.mkdir(exist_ok=True)
                    record = dict(row, identity=identity, sources=fingerprint,
                                  audited_at=datetime.now(timezone.utc).isoformat())
                    try:
                        record.update(audit_record(row, params, detail_dir))
                        validate_source_stats(fingerprint)
                    except (DataQualityError, ValueError, IndexError) as exc:
                        record.update(status='REJECTED', reason=f'{type(exc).__name__}: {exc}')
                    # IO failures propagate rather than silently excluding records.
                    record['elapsed_seconds'] = round(time.monotonic() - started, 3)
                    atomic_json(state_path, record)
                    journal.write(json.dumps(record, allow_nan=False) + '\n')
                    journal.flush(); os.fsync(journal.fileno())
                    note(f"{record['status']} {row['recording_id']} "
                         f"frames={record.get('usable_frames')} {record.get('reason', '')}")
                records.append(record)
                counts = Counter(r['status'] for r in records)
                bar.set_postfix(video=row['recording_id'], passed=counts['PASS'],
                                recovered=counts['RECOVERED_PREFIX'], rejected=counts['REJECTED'],
                                resumed=resumed, refresh=False)
                bar.update(1)
            accepted = [r for r in records if r['status'] in ('PASS', 'RECOVERED_PREFIX')]
            validate_source_stats({'db': db_source})
            signature = digest_json([SCHEMA, str(root), list(cameras), params,
                                     [(r['identity'], r['status'], r.get('usable_frames')) for r in records]])
            manifest = {'schema': SCHEMA, 'complete': scope_complete, 'data_root': str(root),
                        'cameras': list(cameras), 'params': params, 'signature': signature,
                        'records_checksum': digest_json(accepted), 'records': accepted,
                        'db_source': db_source}
            with (output / 'rejected.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['recording_id', 'patient_id', 'video', 'reason'])
                writer.writeheader()
                writer.writerows({k: r[k] for k in writer.fieldnames} for r in records if r['status'] == 'REJECTED')
            summary = {'schema': SCHEMA, 'complete': scope_complete, 'signature': signature,
                       'records': len(records), 'accepted_records': len(accepted),
                       'accepted_subjects': len({r['patient_id'] for r in accepted}),
                       'status_counts': dict(Counter(r['status'] for r in records)), 'resumed': resumed,
                       'accepted_duration_seconds': sum(r['usable_time_seconds'][1] - r['usable_time_seconds'][0] for r in accepted),
                       'invalid_bp_accepted': sum(not r['bp_valid'] for r in accepted)}
            atomic_json(output / 'summary.json', summary)
            atomic_json(output / 'accepted_manifest.json', manifest)
            note(f'Finished: {summary}')
        return summary
    finally:
        lock.unlink(missing_ok=True)


def load_clean_manifest(path, root, cameras):
    """Reject partial/stale manifests before face preprocessing or training."""
    payload = json.loads(Path(path).read_text())
    if payload.get('schema') != SCHEMA or payload.get('complete') is not True:
        raise ValueError('MCD cleaning manifest is partial or incompatible. Complete the full cleaning run first.')
    if Path(payload['data_root']).resolve() != Path(root).resolve():
        raise ValueError('MCD cleaning manifest data root differs from DATA_PATH.')
    if not set(cameras).issubset(payload['cameras']):
        raise ValueError('MCD cleaning manifest does not cover the requested cameras.')
    if payload.get('records_checksum') != digest_json(payload['records']):
        raise ValueError('MCD cleaning manifest records checksum mismatch.')
    try:
        validate_source_stats({'db': payload['db_source']})
    except DataQualityError as exc:
        raise ValueError('MCD db.csv changed since cleaning. Re-run cleaning.') from exc
    rows, seen = [], set()
    for record in payload['records']:
        if record['camera'] not in cameras:
            continue
        if record['recording_id'] in seen or record['status'] not in ('PASS', 'RECOVERED_PREFIX'):
            raise ValueError('Duplicate or unapproved MCD cleaning record.')
        seen.add(record['recording_id'])
        for field in SOURCE_FIELDS:
            expected = record['sources'][field]
            p = Path(record[field]).resolve()
            if not p.is_relative_to(Path(root).resolve()) or str(p) != expected['path']:
                raise ValueError('Invalid source path in MCD cleaning manifest.')
            stat = p.stat()
            if (stat.st_size, stat.st_mtime_ns) != (expected['size'], expected['mtime_ns']):
                raise ValueError(f'Stale cleaning manifest: source changed: {p}. Re-run cleaning.')
        start, stop = record['usable_frames']
        if start != 0 or not isinstance(stop, int) or stop < 2:
            raise ValueError('This MCD loader requires a validated prefix starting at frame zero.')
        row = dict(record)
        # Invalid cuff values are masked, not inferred from contact PPG.
        row['sbp'] = record['sbp'] if record['bp_valid'] else 0.
        row['dbp'] = record['dbp'] if record['bp_valid'] else 0.
        row['clean_approval'] = {'frames': stop, 'width': record['decode']['strict']['width'],
                                 'height': record['decode']['strict']['height'],
                                 'manifest_signature': payload['signature'], 'status': record['status']}
        rows.append(row)
    if not rows:
        raise ValueError('No accepted MCD records for the requested cameras.')
    return rows, payload
