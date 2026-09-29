"""Predict rPPG, HR and cuff BP from video + a trained MCD FULL bundle.

No label CSV/PPG/ECG is read. Optional camera timestamps improve time fidelity.
Run from the Toolbox root; see docs/PhaseShiftedMCD.md.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.mcd_preprocessing import decode_faces, read_timestamps, regular_grid, sample_indices, VERSION
from neural_methods.model.phase_shifted_rppg import DRPNet, BBPNet
from neural_methods.model.phase_shifted_rppg.inference import predict_signals


def predict_video(video_path, checkpoint, output_dir, device='cpu', timestamps=None, fps_override=None):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    sig = payload.get('signature', {})
    if (payload.get('stage') != 'FULL' or sig.get('LABEL_SOURCE') != 'PPG_CUFF' or
            sig.get('PREPROCESS_VERSION') != VERSION):
        raise ValueError('Require a trained MCD PPG_CUFF FULL checkpoint with matching preprocessing version.')
    if (sig['FRAME_NUM'] != 150 or sig['FS'] != 25 or
            sig['VIDEO_CHANNEL_ORDER'] != 'RGB' or sig['PIXEL_SCALE'] != 255.):
        raise ValueError('Checkpoint incompatible with MCD v1 video preprocessing.')
    drp = DRPNet().to(device)
    bbp = BBPNet(sig['SBP_MIN'], sig['SBP_MAX'], sig['DBP_MIN'], sig['DBP_MAX'], sig['TEMPERATURE']).to(device)
    drp.load_state_dict(payload['drpnet'], strict=True)
    bbp.load_state_dict(payload['bbpnet'], strict=True)
    source_times = read_timestamps(timestamps) if timestamps else None
    policy = sig['MCD_POLICY']
    frames, valid, fps, audit = decode_faces(video_path, len(source_times) if source_times is not None else None,
                                           policy['MAX_TAIL_DROP'])
    if source_times is not None:
        source_times = source_times[:len(frames)]
        timing = 'camera_timestamps'
    else:
        fps = fps_override if fps_override is not None else fps
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Missing frame rate; provide --fps or --timestamps.')
        source_times = np.arange(len(frames))/fps
        timing = 'assumed_constant_fps'
    grid = regular_grid(source_times, sig['FS'])
    ix = sample_indices(source_times, grid)
    results, rejected = [], []
    for start in range(0, len(grid)-sig['FRAME_NUM']+1, sig['FRAME_NUM']):
        stop = start+sig['FRAME_NUM']
        ids = ix[start:stop]
        lo, hi = max(0, ids.min()-1), min(len(frames), ids.max()+2)
        if not valid[lo:hi].all() or np.diff(source_times[lo:hi]).max() > policy['MAX_FRAME_GAP']:
            rejected.append({'start_s': float(grid[start]), 'reason': 'face_or_frame_gap'})
            continue
        raw = torch.from_numpy(frames[ids].transpose(3, 0, 1, 2).copy()).unsqueeze(0).to(device, torch.float32)
        prepared = (raw/255.)[:, [2, 1, 0]]
        outputs = predict_signals(drp, bbp, prepared, sig['FS'], sig['RESAMPLE_FACTOR'])
        results.append({k: v.numpy()[0] for k, v in outputs.items()} | {'time': grid[start:stop]})
    if not results:
        raise ValueError('No valid full windows; inspect video timing and face detection.')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dest = output / f'{Path(video_path).stem}_predictions.npz'
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite previous predictions: {dest}')
    np.savez_compressed(dest, **{key: np.stack([r[key] for r in results]) for key in results[0]})
    report = {'video': str(video_path), 'checkpoint': str(checkpoint), 'signature': sig,
              'time_basis': timing, 'quality_note': 'hr_valid and bp_order_valid are numerical checks, not accuracy guarantees',
              'hr_method': 'linear detrend, Butterworth SOS 0.5-3 Hz, Hamming FFT peak',
              'units': {'time': 'seconds', 'rppg': 'arbitrary model amplitude', 'hr': 'bpm', 'bp': 'mmHg'},
              'decode': audit, 'rejected': rejected, 'n_windows': len(results),
              'mean_sbp': float(np.mean([r['sbp'] for r in results])),
              'mean_dbp': float(np.mean([r['dbp'] for r in results]))}
    dest.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(dest)
    return dest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--timestamps', help='Optional MCD meta/*.txt; no physiological labels required')
    parser.add_argument('--fps', type=float, help='For a verified constant-rate video without timestamps')
    args = parser.parse_args()
    predict_video(args.video, args.checkpoint, args.output_dir, args.device, args.timestamps, args.fps)
