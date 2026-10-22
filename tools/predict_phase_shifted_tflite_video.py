"""Predict window SBP/DBP from a recorded camera video using a TFLite bundle.

Run within the Toolbox or as predict_video.py inside the exported deployment ZIP.
No PyTorch, PPG, cuff labels, db.csv or training cache is used.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase_shifted_tflite.runtime import TFLitePredictor
if (HERE/'phase_shifted_tflite/preprocessing.py').is_file():
    from phase_shifted_tflite.preprocessing import decode_faces, read_timestamps, regular_grid, sample_indices
else:
    sys.path.insert(0, str(HERE.parent))
    from dataset.mcd_preprocessing import decode_faces, read_timestamps, regular_grid, sample_indices


def predict_video(video, model_dir, output_dir, variant='fp32', timestamps=None, fps=None, threads=2):
    model = TFLitePredictor(model_dir, variant, threads)
    sig = model.manifest['signature']
    policy = sig['MCD_POLICY']
    dest = Path(output_dir)/f'{Path(video).stem}_{variant}_predictions'
    if any(dest.with_suffix(ext).exists() for ext in ('.json', '.csv', '.npz')):
        raise FileExistsError(f'Prediction output already exists: {dest}')
    source_times = read_timestamps(timestamps) if timestamps else None
    faces, valid, source_fps, audit = decode_faces(
        video, len(source_times) if source_times is not None else None, policy['MAX_TAIL_DROP'])
    if source_times is not None:
        source_times = source_times[:len(faces)]
        time_basis = 'camera_timestamps'
    else:
        source_fps = source_fps if fps is None else fps
        if not np.isfinite(source_fps) or source_fps <= 0:
            raise ValueError('Missing source frame rate; supply --fps or --timestamps.')
        source_times = np.arange(len(faces))/source_fps
        time_basis = 'assumed_constant_fps'
    grid = regular_grid(source_times, 25)
    indices = sample_indices(source_times, grid)
    rows, waves, rejected = [], [], []
    for start in range(0, len(grid)-149, 150):
        ix = indices[start:start+150]
        lo, hi = max(0, int(ix.min())-1), min(len(faces), int(ix.max())+2)
        if not valid[lo:hi].all() or np.diff(source_times[lo:hi]).max() > policy['MAX_FRAME_GAP']:
            rejected.append({'start_s': float(grid[start]), 'reason': 'face_or_frame_gap'})
            continue
        result = model.predict_rgb(faces[ix])
        sbp, dbp = float(result['sbp'][0]), float(result['dbp'][0])
        rows.append({'start_s': float(grid[start]), 'end_s': float(grid[start]+6),
                     'sbp_mmHg': sbp, 'dbp_mmHg': dbp, 'bp_order_valid': sbp > dbp})
        waves.append(dict(result, time=grid[start:start+150][None]))
        print(f'{variant} {grid[start]:.2f}-{grid[start]+6:.2f}s: SBP={sbp:.3f}, DBP={dbp:.3f} mmHg', flush=True)
    if not rows:
        raise ValueError('No valid 150-frame face windows.')
    dest.parent.mkdir(parents=True, exist_ok=True)
    report = {'video': str(video), 'model_dir': str(model_dir), 'variant': variant,
              'units': {'bp': 'mmHg', 'time': 'seconds', 'waveform': 'model amplitude'},
              'time_basis': time_basis, 'decode': audit, 'n_windows': len(rows),
              'mean_sbp_mmHg': float(np.mean([r['sbp_mmHg'] for r in rows])),
              'mean_dbp_mmHg': float(np.mean([r['dbp_mmHg'] for r in rows])),
              'quality_note': 'BP order is a numerical check; these are predictions without reference labels.',
              'windows': rows, 'rejected': rejected}
    dest.with_suffix('.json').write_text(json.dumps(report, indent=2))
    with dest.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(dest.with_suffix('.npz'), **{key: np.concatenate([r[key] for r in waves]) for key in waves[0]})
    print(f'Saved {dest}.json; mean SBP={report["mean_sbp_mmHg"]:.3f}, DBP={report["mean_dbp_mmHg"]:.3f} mmHg', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--model-dir', default=str(HERE))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--variant', choices=('fp32', 'fp16'), default='fp32')
    parser.add_argument('--timestamps')
    parser.add_argument('--fps', type=float)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    predict_video(args.video, args.model_dir, args.output_dir, args.variant, args.timestamps, args.fps, args.threads)


if __name__ == '__main__':
    main()
