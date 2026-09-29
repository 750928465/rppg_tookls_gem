"""MCD timestamp/quality utilities. Importable without OpenCV or a training stack.

PPG amplitudes are never converted to mmHg. ECG is deliberately not used.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from datetime import datetime

import numpy as np

VERSION = 'mcd_ppg_cuff_v1'


def inventory(root, cameras=('FullHDwebcam',)):
    root = Path(root).resolve()
    rows, missing, seen = [], [], set()
    with (root / 'db.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row['camera'] not in cameras:
                continue
            key = (row['patient_id'], row['step'], row['camera'])
            if key in seen:
                raise ValueError(f'Duplicate MCD record: {key}')
            seen.add(key)
            row = dict(row, recording_id='_'.join(key), group_id='_'.join(key[:2]))
            paths = {}
            for field in ('video', 'ppg', 'ppg_sync', 'meta'):
                path = (root / row[field]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError(f'Path outside dataset root: {field}')
                paths[field] = str(path)
            absent = [k for k, p in paths.items() if not Path(p).is_file()]
            if absent:
                missing.append({'recording_id': row['recording_id'], 'missing': absent})
                continue
            row.update(paths)
            row['sbp'], row['dbp'] = float(row['upper_ap']), float(row['lower_ap'])
            row['bp_valid'] = bool(np.isfinite([row['sbp'], row['dbp']]).all()
                                   and row['sbp'] > row['dbp'] > 0)
            rows.append(row)
    return rows, missing


def split_subjects(rows, begin, end, seed=100):
    if not 0 <= begin < end <= 1:
        raise ValueError('Require 0 <= BEGIN < END <= 1.')
    subjects = sorted({r['patient_id'] for r in rows})
    order = np.random.default_rng(seed).permutation(subjects).tolist()
    selected = set(order[int(len(order)*begin):int(len(order)*end)])
    return [r for r in rows if r['patient_id'] in selected]


def read_timestamps(path):
    rows = [s.split(maxsplit=1) for s in Path(path).read_text().splitlines() if s.strip()]
    ids = np.array([int(r[0]) for r in rows])
    if not np.array_equal(ids, np.arange(1, len(ids)+1)):
        raise ValueError('Nonconsecutive meta frame IDs; do not compress missing frames.')
    dates = [datetime.fromisoformat(r[1]) for r in rows]
    t = np.array([(d-dates[0]).total_seconds() for d in dates])
    if len(t) < 2 or not np.all(np.diff(t) > 0):
        raise ValueError('Video timestamps must strictly increase.')
    return t


def read_sync(row):
    t = read_timestamps(row['meta'])
    sync = np.loadtxt(row['ppg_sync'], ndmin=2)
    if sync.shape[1] != 2 or not np.isfinite(sync).all() or (sync[:, 1] < 0).any():
        raise ValueError('ppg_sync must contain finite [PPG, absolute time difference in seconds].')
    if len(sync) > len(t):
        raise ValueError('More PPG rows than meta timestamps.')
    # Verify the prefix mapping from actual raw PPG timestamps, rather than
    # assuming that every ppg_sync file was generated with the same convention.
    if 'ppg' in row:
        lines = [s.split(maxsplit=1) for s in Path(row['ppg']).read_text().splitlines() if s.strip()]
        meta_first = next(s for s in Path(row['meta']).read_text().splitlines() if s.strip())
        origin = datetime.fromisoformat(meta_first.split(maxsplit=1)[1])
        pt = np.array([(datetime.fromisoformat(s[1])-origin).total_seconds() for s in lines])
        pv = np.array([float(s[0]) for s in lines])
        if len(pt) < 2 or not np.isfinite(pv).all() or (np.diff(pt) < 0).any():
            raise ValueError('Invalid or decreasing raw PPG timestamps.')
        for timestamp, (value, error) in zip(t, sync):
            right = np.clip(np.searchsorted(pt, timestamp), 1, len(pt)-1)
            nearest = min(abs(pt[right]-timestamp), abs(pt[right-1]-timestamp))
            if abs(nearest-error) > 2e-6:
                raise ValueError('ppg_sync/meta timestamp mapping does not match raw PPG.')
            lo = np.searchsorted(pt, timestamp-nearest-2e-6)
            hi = np.searchsorted(pt, timestamp+nearest+2e-6, side='right')
            if not np.any((pv[lo:hi] == value) & (np.abs(np.abs(pt[lo:hi]-timestamp)-nearest) <= 2e-6)):
                raise ValueError('ppg_sync amplitude does not match nearest raw PPG sample.')
    return t[:len(sync)], sync[:, 0], sync[:, 1], len(t)-len(sync)


def regular_grid(t, fs):
    if fs <= 0 or len(t) < 2 or not np.all(np.diff(t) > 0):
        raise ValueError('Invalid sample rate or time axis.')
    # Keep the same physical interval; never stretch N source samples to N target samples.
    return t[0] + np.arange(int(np.floor((t[-1]-t[0])*fs))+1)/fs


def sample_indices(t, grid):
    right = np.clip(np.searchsorted(t, grid), 1, len(t)-1)
    left = right-1
    return np.where(grid-t[left] <= t[right]-grid, left, right)


def decode_faces(path, expected=None, max_tail_drop=32):
    """Strict FFmpeg decode, streamed through an OpenCV Haar face cropper.

    A corrupt suffix is tolerated ONLY when an expected ppg_sync length exists
    and the missing suffix is bounded. No frames after an error are accepted.
    Inference without reference files fails on a decode error.
    """
    import cv2
    if not shutil.which('ffmpeg'):
        raise RuntimeError('ffmpeg executable is required for strict video decoding.')
    cap = cv2.VideoCapture(str(path))
    width, height = (int(cap.get(p)) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if width <= 0 or height <= 0:
        raise ValueError(f'Cannot read video dimensions: {path}')
    cascade = Path(__file__).parent / 'haarcascade_frontalface_default.xml'
    detector = cv2.CascadeClassifier(str(cascade))
    if detector.empty():
        raise RuntimeError(f'Cannot load face detector: {cascade}')
    command = ['ffmpeg', '-nostdin', '-hide_banner', '-v', 'error', '-xerror',
               '-threads', '1', '-i', str(path), '-map', '0:v:0', '-an',
               '-vsync', '0', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1']
    faces, valid, box = [], [], None
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            nbytes = width*height*3
            while True:
                raw = process.stdout.read(nbytes)
                if not raw:
                    break
                if len(raw) != nbytes:
                    raise ValueError('Partial decoded frame; cannot preserve frame indexing.')
                frame = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
                # Re-detect every 30 original frames. A failed detection invalidates
                # the interval; never substitute the full image as a detected face.
                if len(faces) % 30 == 0:
                    found = detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
                                                      scaleFactor=1.1, minNeighbors=5)
                    box = max(found, key=lambda b: b[2]*b[3]) if len(found) else None
                if box is None:
                    faces.append(np.zeros((128, 128, 3), np.uint8)); valid.append(False)
                    continue
                x, y, w, h = map(int, box)
                side = max(w, h)*1.3
                x0, y0 = max(0, int(x+w/2-side/2)), max(0, int(y+h/2-side/2))
                x1, y1 = min(width, int(x+w/2+side/2)), min(height, int(y+h/2+side/2))
                crop = frame[y0:y1, x0:x1]
                faces.append(cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA))
                valid.append(True)
            code = process.wait()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill(); process.wait()
        errors.seek(0)
        message = errors.read().decode(errors='replace')
    count = len(faces)
    if count < 2:
        raise ValueError(f'No usable decoded frames: {message}')
    if expected is not None:
        if not 0 <= expected-count <= max_tail_drop:
            raise ValueError(f'Decode/PPG count mismatch: decoded={count}, PPG={expected}. {message}')
        if code and count == expected:
            raise ValueError(f'Decode error with ambiguous frame mapping: {message}')
    elif code:
        raise ValueError(f'Video decode failed: {message}')
    return np.asarray(faces), np.asarray(valid), fps, {
        'decoded_frames': count, 'expected_frames': expected,
        'dropped_suffix': expected-count if expected is not None else 0,
        'ffmpeg_returncode': code, 'decoder_message': message,
    }


def aligned_clips(row, fs=25, length=150, max_sync_error=.02,
                  max_frame_gap=.10, max_tail_drop=32):
    t, ppg, error, meta_extra = read_sync(row)
    frames, face_valid, _, audit = decode_faces(row['video'], len(t), max_tail_drop)
    n = len(frames)
    t, ppg, error = t[:n], ppg[:n], error[:n]
    grid = regular_grid(t, fs)
    idx = sample_indices(t, grid)
    # Interpolation follows irregular acquisition timestamps, no constant-fps assumption.
    target = np.interp(grid, t, ppg)
    rejected, clips = [], []
    for start in range(0, len(grid)-length+1, length):
        stop = start+length
        chosen = idx[start:stop]
        lo, hi = max(0, chosen.min()-1), min(n, chosen.max()+2)
        wave = target[start:stop].astype(np.float32)
        reason = None
        if not face_valid[lo:hi].all(): reason = 'face_detection'
        elif error[lo:hi].max() > max_sync_error: reason = 'sync_error'
        elif np.diff(t[lo:hi]).max() > max_frame_gap: reason = 'frame_gap'
        elif np.std(wave) < 1e-6: reason = 'flat_ppg'
        if reason:
            rejected.append({'chunk': start//length, 'reason': reason}); continue
        clips.append((frames[chosen], wave, grid[start:stop], start//length,
                      float(error[lo:hi].max())))
    audit.update(meta_unused_suffix=meta_extra, rejected=rejected,
                 accepted_windows=len(clips), trailing_grid_samples=len(grid)%length)
    return clips, audit


def cache_signature(rows, params):
    sources = []
    for row in rows:
        sources.append({k: row[k] for k in ('recording_id', 'sbp', 'dbp')})
        for key in ('video', 'meta', 'ppg', 'ppg_sync'):
            p = Path(row[key]); stat = p.stat()
            sources.append([str(p), stat.st_size, stat.st_mtime_ns])
    return hashlib.sha256(json.dumps([VERSION, params, sources], sort_keys=True).encode()).hexdigest()
