"""Package verified TFLite models with standalone recorded-video inference."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def package(model_dir):
    root = Path(model_dir).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    if not manifest.get('verified'):
        raise ValueError('Package only numerically verified exports.')
    runtime = root/'phase_shifted_tflite'
    runtime.mkdir(exist_ok=True)
    for name in ('__init__.py', 'runtime.py'):
        shutil.copy2(ROOT/'tools/phase_shifted_tflite'/name, runtime/name)
    shutil.copy2(ROOT/'dataset/mcd_preprocessing.py', runtime/'preprocessing.py')
    shutil.copy2(ROOT/'dataset/haarcascade_frontalface_default.xml', runtime/'haarcascade_frontalface_default.xml')
    shutil.copy2(ROOT/'tools/predict_phase_shifted_tflite_video.py', root/'predict_video.py')
    shutil.copy2(ROOT/'neural_methods/model/phase_shifted_rppg/LICENSE', root/'MODEL_LICENSE')
    (root/'requirements.txt').write_text(
        '# Python 3.12 recommended; CPU inference only, no PyTorch needed.\n'
        'tensorflow==2.21.0\nnumpy>=2.0,<2.6\nscipy>=1.15,<1.18\n'
        'opencv-python-headless==4.13.0.92\n')
    drifts = {v: max(error['max_abs'] for case in manifest['verification']
                    for error in case['variants'][v]['end_to_end_bp'].values())
              for v in manifest['variants']}
    (root/'README.md').write_text(f'''# Phase-Shifted MCD TFLite deployment

Contains the explicitly selected DRPNet and BBPNet checkpoints converted to
FP32 and FP16-weight TFLite. INT8 is skipped at the user's request.
Model source: GyutaeHwang/phase_shifted_rPPG; retained MIT license: MODEL_LICENSE.

## Server setup

FFmpeg must be installed and on PATH. Use a separate environment, for example:

```bash
conda create -n phase_tflite python=3.12 -y
conda activate phase_tflite
python -m pip install -r requirements.txt
conda install -c conda-forge ffmpeg -y
```

## Predict a recorded camera video

```bash
python -u predict_video.py --video /path/to/face_video.mp4 --variant fp32 --output-dir ./results
python -u predict_video.py --video /path/to/face_video.mp4 --variant fp16 --output-dir ./results
```

The model directory defaults to this extracted folder. Set --model-dir explicitly
when invoking predict_video.py from another copy. Use --threads to set CPU threads.
This entrypoint takes recorded video files; it does not open a live camera.
FP16 keeps float32 I/O; CPU arithmetic may remain float32. CUDA is not used by
this TFLite Interpreter entrypoint. No PPG/BP reference files are read.

Output JSON/CSV: each accepted ~6-second window's SBP/DBP in mmHg and mean values.
NPZ: predicted facial/acral rPPG, SBP/DBP and sample times.
Frame rate is inferred from the video; --fps overrides a verified constant rate.
Optional --timestamps accepts camera timestamps in the existing MCD meta format.
No valid face windows or ambiguous decode errors are rejected. In particular,
the old MCD cleaning approvals are not automatically applied to corrupt videos.

## Fixed model interfaces

- DRPNet input `video`: float32 BGR / 255, [1,3,150,128,128].
- DRPNet outputs `facial_rppg`, `acral_rppg`: float32 [1,150].
- Between stages: SciPy Fourier resample 150 -> 600, global mean/std with ddof=1,
  one window per invocation. Do not substitute linear interpolation or population std.
- BBPNet inputs `facial`, `acral`: float32 [1,600]. Outputs `sbp`, `dbp`: [1], mmHg.
- Sampling 25 Hz; Haar face crop 128x128 is the same as the existing MCD code.

Conv3D is rendered as equivalent temporal gather + Conv2D, and Pool3D as separable
Pool2D. The original temporal kernels, dilation, padding, weights and BN statistics
are retained. No retraining. Models use only built-in TFLite operators (no Flex).

## Verification

Four windows: two deterministic synthetic and two valid local MCD video windows.
Maximum end-to-end BP difference vs selected PyTorch pair:

- FP32: {drifts.get('fp32', float('nan')):.9f} mmHg.
- FP16: {drifts.get('fp16', float('nan')):.9f} mmHg.

These measure numerical conversion fidelity, not BP MAE against measured BP.
No clinical accuracy or target-server throughput is established by this check.
See manifest.json for all input hashes, model hashes, operators and per-case results.
Source checkpoints were supplied as local copies; original training-time Stage-1
file hashes were not recorded by the older trainer.
''')
    archive = root.parent/'phase_shifted_tflite_fp32_fp16.zip'
    files = [root/'manifest.json', root/'README.md', root/'MODEL_LICENSE',
             root/'requirements.txt', root/'predict_video.py',
             *runtime.glob('*.py'), runtime/'haarcascade_frontalface_default.xml']
    for variant, pair in manifest['variants'].items():
        files.extend(root/item['file'] for item in pair.values())
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as handle:
        for path in files:
            handle.write(path, arcname=str(Path('phase_shifted_tflite')/path.relative_to(root)))
    print(f'Packaged: {archive}')
    return archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()
    package(args.model_dir)
