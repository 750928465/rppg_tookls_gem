"""Run with: python -m dataset.mcd --data-root ... --output ..."""
import argparse
import json
import shutil

from .cleaning import clean_dataset


def main():
    parser = argparse.ArgumentParser(description='Clean MCD videos and reference mappings; no face crop or training.')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cameras', nargs='+', default=['FullHDwebcam'])
    parser.add_argument('--ffmpeg', default=shutil.which('ffmpeg') or 'ffmpeg')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--max-tail-drop', type=int, default=32)
    parser.add_argument('--max-sync-error', type=float, default=.02)
    parser.add_argument('--max-frame-gap', type=float, default=.10)
    parser.add_argument('--force', action='store_true', help='Re-audit completed records too.')
    parser.add_argument('--recording', nargs='+', dest='recording_ids', help='Audit only selected recording IDs; output is marked partial.')
    parser.add_argument('--limit', type=int, help='Small diagnostic run; output is marked partial and cannot be used for training.')
    args = vars(parser.parse_args())
    args['root'] = args.pop('data_root')
    print(json.dumps(clean_dataset(**args), indent=2))


if __name__ == '__main__':
    main()
