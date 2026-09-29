"""Read-only MCD inventory/timestamp audit; no video decode, training or caching."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.mcd_preprocessing import inventory, read_sync, split_subjects


def audit(root):
    rows, missing = inventory(root)
    summary = {'local_records': len(rows), 'local_subjects': len({r['patient_id'] for r in rows}),
               'missing_records': len(missing), 'missing_preview': missing[:5],
               'invalid_bp_records': [r['recording_id'] for r in rows if not r['bp_valid']],
               'splits': {}, 'signals': []}
    for name, begin, end in [('train', 0, .7), ('valid', .7, .85), ('test', .85, 1)]:
        subset = split_subjects(rows, begin, end)
        summary['splits'][name] = {'subjects': len({r['patient_id'] for r in subset}), 'records': len(subset)}
    for r in rows:
        t, ppg, error, extra = read_sync(r)
        summary['signals'].append({'recording_id': r['recording_id'], 'samples': len(t),
                                   'duration_s': float(t[-1]-t[0]), 'max_sync_error_s': float(error.max()),
                                   'unused_meta_suffix': extra, 'sbp': r['sbp'], 'dbp': r['dbp']})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.data_root), indent=2))
