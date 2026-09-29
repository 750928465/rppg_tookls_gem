"""Server-side real sample preprocessing/forward check. Never trains or installs.

Without a checkpoint the optional forward pass uses RANDOM weights: structure
verification only, not physiological predictions or reproduction results.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.mcd_preprocessing import inventory, aligned_clips


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--recording-id', default='1020_before_FullHDwebcam')
    parser.add_argument('--forward', action='store_true', help='RANDOM weights, one window; no optimizer')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    rows, _ = inventory(args.data_root)
    row = next((r for r in rows if r['recording_id']==args.recording_id), None)
    if row is None:
        raise ValueError(f'Local recording not found: {args.recording_id}')
    clips, audit = aligned_clips(row)
    print(json.dumps(audit, indent=2))
    if not clips:
        raise ValueError('No valid full windows.')
    video, ppg, times, _, _ = clips[0]
    print('video', video.shape, 'ppg', ppg.shape, 'time', times[[0, -1]],
          'cuff_mmHg', (row['sbp'], row['dbp']))
    if args.forward:
        import torch
        from neural_methods.model.phase_shifted_rppg import DRPNet, BBPNet
        from neural_methods.model.phase_shifted_rppg.inference import predict_signals
        torch.set_num_threads(2)
        x = torch.from_numpy(video.transpose(3, 0, 1, 2).copy()).unsqueeze(0).to(args.device, torch.float32)
        out = predict_signals(DRPNet().to(args.device), BBPNet(60,240,30,140).to(args.device),
                              (x/255)[:, [2,1,0]])
        print('RANDOM-WEIGHT SHAPE CHECK ONLY:', {k: list(v.shape) for k,v in out.items()})
