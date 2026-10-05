"""Inspectable CSV/SVG outputs for the two stages; prediction tensors stay in PTH."""
import csv

import numpy as np


def plot_history(history, output, prefix):
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    epochs = [h['epoch']+1 for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ('train', 'valid'):
        axes[0].plot(epochs, [h[split]['loss'] for h in history], label=split)
    axes[0].set(xlabel='Epoch', ylabel='Loss', title='Training / validation loss')
    axes[0].legend()
    axes[1].plot(epochs, [h['lr'] for h in history])
    axes[1].set(xlabel='Epoch', ylabel='Learning rate')
    fig.tight_layout()
    fig.savefig(output / f'{prefix}_history.svg')
    plt.close(fig)


def export_results(records, groups, output, prefix, preview_limit):
    """Export every test window; preview only the first N, never select by error."""
    output.mkdir(parents=True, exist_ok=True)
    columns = ['recording_id', 'chunk_id', 'state_id', 'start_s', 'end_s',
               'reference_hr_bpm', 'facial_hr_bpm', 'acral_hr_bpm', 'hr_valid',
               'reference_sbp_mmHg', 'reference_dbp_mmHg', 'sbp_mmHg', 'dbp_mmHg', 'bp_order_valid']
    preview_count = 0
    with (output / f'{prefix}_windows.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for record in records:
            for i in range(len(record['hr_facial'])):
                time = np.asarray(record['time'][i])
                row = dict(recording_id=record['recording_id'][i], chunk_id=record['chunk_id'][i],
                           state_id=record['group_id'][i], start_s=float(time[0]), end_s=float(time[-1]),
                           reference_hr_bpm=float(record['reference_hr'][i]),
                           facial_hr_bpm=float(record['hr_facial'][i]), acral_hr_bpm=float(record['hr_acral'][i]),
                           hr_valid=bool(record['hr_valid'][i] and record['reference_hr_valid'][i]))
                if 'sbp' in record:
                    row.update(reference_sbp_mmHg=float(record['labels']['sbp'][i]),
                               reference_dbp_mmHg=float(record['labels']['dbp'][i]),
                               sbp_mmHg=float(record['sbp'][i]), dbp_mmHg=float(record['dbp'][i]),
                               bp_order_valid=bool(record['bp_order_valid'][i]))
                writer.writerow(row)
                if preview_count < preview_limit:
                    _wave_preview(record, i, time, output / f'{prefix}_waveform_{preview_count:03d}.svg')
                    preview_count += 1
    if groups:
        with (output / f'{prefix}_states.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['state_id', 'n_windows', 'reference_sbp_mmHg', 'reference_dbp_mmHg', 'sbp_mmHg', 'dbp_mmHg'])
            for key, values in groups.items():
                writer.writerow([key, len(values['predictions']), *values['target'], *np.mean(values['predictions'], axis=0)])


def _wave_preview(record, i, time, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.5))
    series = [('Reference PPG', record['labels']['ppg'][i]),
              ('Predicted facial rPPG', record['facial_rppg'][i]),
              ('Predicted acral rPPG', record['acral_rppg'][i])]
    for label, values in series:
        wave = np.asarray(values)
        # Normalize only the visual overlay; saved raw waveforms and metrics are untouched.
        ax.plot(time, (wave-wave.mean())/max(float(wave.std()), 1e-8), label=label, alpha=.8)
    ax.set(xlabel='Time from recording start (s)', ylabel='Standardized amplitude (a.u.)',
           title=f"{record['recording_id'][i]} / chunk {record['chunk_id'][i]} (display normalization only)")
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
