"""Select a best-case PPG fit by mean two-branch z-score RMSE.

Use all finite, non-flat waveform windows. No filtering, lag search, polarity
flip, smoothing, or phase alignment is applied. Ties retain saved test order.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def standardized(values):
    wave = np.asarray(values, dtype=np.float64)
    if wave.ndim != 1 or not np.isfinite(wave).all() or wave.std() <= 1e-8:
        return None
    return (wave-wave.mean()) / wave.std()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Destination SVG')
    args = parser.parse_args()
    data = torch.load(args.results, map_location='cpu', weights_only=True)
    if data.get('stage') != 'STAGE1':
        raise ValueError('Expected STAGE1 prediction results.')
    best, eligible, ties = None, 0, 0
    for record in data['records']:
        for i in range(len(record['facial_rppg'])):
            waves = [standardized(v[i]) for v in (
                record['labels']['ppg'], record['facial_rppg'], record['acral_rppg'])]
            if any(w is None for w in waves):
                continue
            ref, face, acral = waves
            if not (ref.shape == face.shape == acral.shape):
                raise ValueError('Mismatched waveform lengths.')
            errors = [float(np.sqrt(np.mean((w-ref)**2))) for w in (face, acral)]
            score = float(np.mean(errors))
            eligible += 1
            if best is None or score < best[0]:
                best = (score, record, i, waves, errors)
                ties = 1
            elif score == best[0]:
                ties += 1
    if best is None:
        raise ValueError('No finite non-flat waveform windows.')
    score, record, i, waves, errors = best
    correlations = [float(np.mean(waves[0]*w)) for w in waves[1:]]
    time = np.asarray(record['time'][i], dtype=float)
    recording = str(record['recording_id'][i])
    chunk = str(record['chunk_id'][i])
    hrs = [float(record[k][i]) for k in ('reference_hr', 'hr_facial', 'hr_acral')]
    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = [f'Reference PPG (HR = {hrs[0]:.2f} bpm)',
              f'Facial rPPG: r = {correlations[0]:.3f}, z-RMSE = {errors[0]:.3f}',
              f'Acral rPPG: r = {correlations[1]:.3f}, z-RMSE = {errors[1]:.3f}']
    for wave, label, color, style in zip(waves, labels,
            ['#222222', '#0072B2', '#D55E00'], ['-', '--', '-.']):
        ax.plot(time, wave, label=label, color=color, linestyle=style, linewidth=1.6)
    ax.set(xlabel='Time from recording start (s)', ylabel='Standardized amplitude (a.u.)',
           title=f'Best-case PPG fit: {recording}, chunk {chunk}\n'
                 f'Minimum mean two-branch z-RMSE = {score:.3f}')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.18)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.20), frameon=False)
    fig.text(.5, .02, 'Selected best-case test window; per-waveform z-score only. No filtering or phase alignment.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format='svg', bbox_inches='tight')
    fig.savefig(args.output.with_suffix('.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)
    audit = dict(source=str(args.results.resolve()),
                 selection='minimum mean facial/acral RMSE after per-waveform z-score (ddof=0)',
                 tie_break='first in saved test order', eligible_windows=eligible,
                 tied_best_windows=ties, recording_id=recording, chunk_id=chunk,
                 start_s=float(time[0]), end_s=float(time[-1]),
                 mean_z_rmse=score, facial_z_rmse=errors[0], acral_z_rmse=errors[1],
                 facial_pearson_r=correlations[0], acral_pearson_r=correlations[1],
                 reference_hr_bpm=hrs[0], facial_hr_bpm=hrs[1], acral_hr_bpm=hrs[2],
                 filtering=False, phase_alignment=False, polarity_flip=False,
                 note='Selected best case, not aggregate performance; contact PPG is the common reference for both branches.')
    args.output.with_suffix('.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2))
    print(f'SVG saved: {args.output.resolve()}')


if __name__ == '__main__':
    main()
