"""Export the test window with the smallest mean facial/acral HR error.

This is a selected best-case HR example, not a waveform-quality ranking.
No model, original video, or GPU is needed. Ties retain test-file order.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True,
                        help='DRPNet_STAGE1_outputs.pth from the independent test')
    parser.add_argument('--output', type=Path, required=True, help='Destination SVG')
    args = parser.parse_args()
    data = torch.load(args.results, map_location='cpu', weights_only=True)
    if data.get('stage') != 'STAGE1':
        raise ValueError('Expected STAGE1 prediction results, not a model checkpoint.')

    best, ties, eligible = None, 0, 0
    for record in data['records']:
        for i in range(len(record['reference_hr'])):
            if not (bool(record['hr_valid'][i]) and bool(record['reference_hr_valid'][i])):
                continue
            reference = float(record['reference_hr'][i])
            facial = float(record['hr_facial'][i])
            acral = float(record['hr_acral'][i])
            if not np.isfinite([reference, facial, acral]).all():
                continue
            score = (abs(facial-reference) + abs(acral-reference))/2
            eligible += 1
            if best is None or score < best[0]:
                best = (score, record, i, reference, facial, acral)
                ties = 1
            elif score == best[0]:
                ties += 1
    if best is None:
        raise ValueError('No valid HR windows found.')

    score, record, i, reference, facial, acral = best
    time = np.asarray(record['time'][i], dtype=float)
    series = [
        ('Reference PPG', record['labels']['ppg'][i], reference, '#222222', '-'),
        ('Predicted facial rPPG', record['facial_rppg'][i], facial, '#0072B2', '--'),
        ('Predicted acral rPPG', record['acral_rppg'][i], acral, '#D55E00', '-.'),
    ]
    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for label, wave, hr, color, style in series:
        wave = np.asarray(wave, dtype=float)
        if not np.isfinite(wave).all() or wave.std() <= 1e-8:
            raise ValueError('Selected waveform is invalid or flat.')
        ax.plot(time, (wave-wave.mean())/wave.std(), color=color, linestyle=style,
                linewidth=1.4, label=f'{label} (HR = {hr:.2f} bpm)')
    recording_id = str(record['recording_id'][i])
    chunk_id = str(record['chunk_id'][i])
    ax.set(xlabel='Time from recording start (s)', ylabel='Standardized amplitude (a.u.)',
           title=f'Best-case HR example: {recording_id}, chunk {chunk_id}\n'
                 f'Mean absolute HR error of two branches = {score:.2f} bpm')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.18)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.22), ncol=1, frameon=False)
    fig.text(.5, .015, 'Selected by HR error; first tied window. Display normalization only; no phase alignment.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format='svg', bbox_inches='tight')
    plt.close(fig)
    audit = {
        'source': str(args.results.resolve()),
        'selection': 'min((abs(facial_hr-reference_hr)+abs(acral_hr-reference_hr))/2)',
        'tie_break': 'first in saved test order', 'eligible_windows': eligible,
        'tied_best_windows': ties, 'recording_id': recording_id, 'chunk_id': chunk_id,
        'start_s': float(time[0]), 'end_s': float(time[-1]),
        'reference_hr_bpm': reference, 'facial_hr_bpm': facial, 'acral_hr_bpm': acral,
        'facial_abs_error_bpm': abs(facial-reference), 'acral_abs_error_bpm': abs(acral-reference),
        'mean_abs_error_bpm': score,
        'note': 'Best-case HR illustration only; not evidence of best waveform morphology or phase.'
    }
    args.output.with_suffix('.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2))
    print(f'SVG saved: {args.output.resolve()}')


if __name__ == '__main__':
    main()
