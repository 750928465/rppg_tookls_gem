"""Copy only the selected model's deployment files, preserving relative paths.

Build into a new directory; after independent execution checks, seal that directory
with SHA256SUMS and a ZIP. Original Toolbox files and weights are never relocated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = Path('model_outputs/phase_shifted_tflite')
FILES = [
    'LICENSE', 'dataset/__init__.py', 'dataset/mcd_preprocessing.py',
    'dataset/haarcascade_frontalface_default.xml',
    'neural_methods/__init__.py', 'neural_methods/model/__init__.py',
    'neural_methods/loss/__init__.py', 'neural_methods/loss/PhaseShiftedLoss.py',
    *[f'neural_methods/model/phase_shifted_rppg/{name}' for name in
      ('__init__.py', 'DRPNet.py', 'BBPNet.py', 'inference.py', 'LICENSE')],
    'tools/phase_shifted_predict_video.py', 'tools/predict_phase_shifted_tflite_video.py',
    'tools/convert_phase_shifted_tflite.py',
    *[f'tools/phase_shifted_tflite/{name}' for name in ('__init__.py', 'runtime.py', 'tf_bridge.py')],
    'requirements-phase-shifted-tflite.txt', 'tests/test_phase_shifted_tflite.py',
    str(DEPLOY/'source/DRPNet_STAGE1_Best.pth'),
    str(DEPLOY/'source/BBPNet_STAGE2_Best.pth'),
    str(DEPLOY/'export/manifest.json'),
    *[str(DEPLOY/f'export/{variant}/{name}.tflite')
      for variant in ('fp32', 'fp16') for name in ('DRPNet', 'BBPNet')],
    str(DEPLOY/'conversion_environment.lock.txt'),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(destination):
    import torch
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'Use a new project directory: {destination}')
    manifest = json.loads((ROOT/DEPLOY/'export/manifest.json').read_text())
    if not manifest.get('verified'):
        raise ValueError('TFLite exports must pass verification before handoff.')
    inputs = {stage: ROOT/DEPLOY/'source'/name for stage, name in
              [('stage1', 'DRPNet_STAGE1_Best.pth'), ('stage2', 'BBPNet_STAGE2_Best.pth')]}
    for stage, path in inputs.items():
        if sha256(path) != manifest['source'][stage]['sha256']:
            raise ValueError(f'{stage}: source weight changed since conversion.')
    for pair in manifest['variants'].values():
        for item in pair.values():
            if sha256(ROOT/DEPLOY/'export'/item['file']) != item['sha256']:
                raise ValueError('TFLite model differs from conversion manifest.')
    missing = [name for name in FILES if not (ROOT/name).is_file()]
    if missing:
        raise FileNotFoundError(f'Required files absent: {missing}')
    destination.mkdir(parents=True)
    for name in FILES:
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    first = torch.load(inputs['stage1'], map_location='cpu', weights_only=True)
    second = torch.load(inputs['stage2'], map_location='cpu', weights_only=True)
    if first['signature'] != second['signature'] or first['subject_splits'] != second['subject_splits']:
        raise ValueError('Cannot combine checkpoints with different contracts/splits.')
    full = {'method': 'PhaseShiftedRPPG', 'stage': 'FULL', 'signature': first['signature'],
            'subject_splits': first['subject_splits'],
            'stage1_checkpoint': second['stage1_checkpoint'],
            'drpnet': first['drpnet'], 'bbpnet': second['bbpnet'],
            'export': {'selection': 'Explicitly selected relocated Best copies, same as TFLite source.',
                       'source': manifest['source'],
                       'note': 'Source hashes identify copied input files, not historical training-time file contents.'}}
    full_path = destination/DEPLOY/'export/PhaseShifted_MCD_selected_FULL.pth'
    torch.save(full, full_path)
    values = {variant: max(error['max_abs'] for case in manifest['verification']
                          for error in case['variants'][variant]['end_to_end_bp'].values())
              for variant in ('fp32', 'fp16')}
    table = ['| 模型 | FP32 | FP16 权重 |', '|---|---:|---:|']
    for name in ('DRPNet', 'BBPNet'):
        sizes = [manifest['variants'][variant][name]['bytes']/1024**2 for variant in ('fp32', 'fp16')]
        table.append(f'| {name} | {sizes[0]:.3f} MiB | {sizes[1]:.3f} MiB |')
    readme = (ROOT/'docs/PhaseShiftedServerREADME.md').read_text()
    readme = readme.replace('{{MODEL_TABLE}}', '\n'.join(table))
    for variant in ('fp32', 'fp16'):
        readme = readme.replace('{{BP_'+variant.upper()+'}}', f'{values[variant]:.9f}')
    (destination/'README.md').write_text(readme)
    (destination/'requirements-inference.txt').write_text(
        '# Install the appropriate CPU/CUDA PyTorch build separately.\n'
        'numpy>=2.0,<2.6\nscipy>=1.15,<1.18\nopencv-python-headless==4.13.0.92\n')
    (destination/'requirements-inference-tflite.txt').write_text(
        '-r requirements-inference.txt\ntensorflow==2.21.0\n')
    copied = {name: {'sha256': sha256(destination/name), 'source': name} for name in FILES}
    generated = ['README.md', 'requirements-inference.txt', 'requirements-inference-tflite.txt',
                 str(full_path.relative_to(destination))]
    copied.update({name: {'sha256': sha256(destination/name), 'source': 'generated for handoff'} for name in generated})
    info = {'schema': 'phase_shifted_server_handoff_v1', 'date': '2026-10-21',
            'source_project': str(ROOT), 'relative_paths_preserved': True,
            'scope': 'Inference/preprocessing/selected weights/conversion sources; no raw human video or training cache.',
            'files': copied, 'independent_execution_verified': False}
    (destination/'HANDOFF_MANIFEST.json').write_text(json.dumps(info, indent=2))
    print(f'Built {destination}; {len(copied)} files, original relative paths retained.')
    return destination


def seal(project):
    project = Path(project).resolve()
    info = json.loads((project/'HANDOFF_MANIFEST.json').read_text())
    tests = json.loads((project/'HANDOFF_TESTS.json').read_text())
    if not tests.get('passed'):
        raise ValueError('Independent execution checks must pass before sealing.')
    for name, item in info['files'].items():
        if sha256(project/name) != item['sha256']:
            raise ValueError(f'Handoff file changed unexpectedly: {name}')
    info['independent_execution_verified'] = True
    (project/'HANDOFF_MANIFEST.json').write_text(json.dumps(info, indent=2))
    files = sorted(p for p in project.rglob('*') if p.is_file()
                   and '__pycache__' not in p.parts and p.suffix != '.pyc' and p.name != 'SHA256SUMS')
    (project/'SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.relative_to(project)}\n' for p in files))
    files.append(project/'SHA256SUMS')
    archive = project.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as handle:
        for path in files:
            handle.write(path, arcname=str(Path(project.name)/path.relative_to(project)))
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip() is not None:
            raise ValueError('ZIP CRC validation failed.')
    print(f'Sealed {archive}; {len(files)} files, {archive.stat().st_size} bytes.')
    return archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output', help='Create a new standalone project directory.')
    mode.add_argument('--seal', help='Seal an independently tested project as ZIP.')
    args = parser.parse_args()
    if args.output:
        build(args.output)
    else:
        seal(args.seal)
