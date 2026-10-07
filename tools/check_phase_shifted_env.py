"""Check real main.py imports, Stage-1 autograd and optionally the MCD manifest.

No training, optimizer updates, face preprocessing or package installation.
Run from the Toolbox root so config-relative paths match main.py.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def check_audio_compatibility(torch_version, audio_version):
    """Apply the official stable-ABI exception, not unconditional version equality.

    https://docs.pytorch.org/audio/stable/installation.html
    TorchAudio 2.11.0 supports Torch 2.11 and later. Earlier releases need
    matching Torch releases; actual native/device compatibility is probed below.
    """
    from packaging.version import Version
    torch_release = Version(Version(torch_version).base_version)
    audio_release = Version(Version(audio_version).base_version)
    if audio_release == Version('2.11.0') and torch_release >= Version('2.11.0'):
        return
    if audio_release != torch_release:
        raise RuntimeError(
            'Incompatible release pair: use TorchAudio 2.11.0 with Torch >=2.11, '
            'or matching older Torch/TorchAudio releases. CUDA builds must also be compatible.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_file', default='configs/train_configs/phase_shifted_rppg/MCD_DRPNet.yaml')
    parser.add_argument('--check-data', action='store_true', help='Validate all approved source paths/size/mtime, without decoding.')
    parser.add_argument('--expected-records', type=int, help='Require this many approved recordings; use with --check-data.')
    args = parser.parse_args()
    if args.expected_records is not None and not args.check_data:
        parser.error('--expected-records requires --check-data')
    if sys.version_info < (3, 9):
        raise RuntimeError('This integration uses Python >=3.9 APIs; use the documented Python 3.11 baseline.')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    os.environ.setdefault('MPLBACKEND', 'Agg')
    print(f'Python: {sys.executable}; {sys.version.split()[0]}', flush=True)

    # Exercise the complete real registration/import chain, without mocks.
    import main as toolbox
    import cv2
    import torch
    import torchaudio
    import torchvision
    from neural_methods.loss.PhaseShiftedLoss import PhaseShiftedStage1Loss

    print(f'torch={torch.__version__}; torchvision={torchvision.__version__}; '
          f'torchaudio={torchaudio.__version__}; CUDA runtime={torch.version.cuda}', flush=True)
    check_audio_compatibility(torch.__version__, torchaudio.__version__)
    config = toolbox.get_config(args)
    if config.MODEL.NAME not in ('DRPNet', 'BBPNet'):
        raise ValueError('Use an independent DRPNet or BBPNet config.')
    trainer = getattr(toolbox.trainer.phase_shifted_rppg, config.MODEL.NAME + 'Trainer')
    trainer.validate_config(config)
    print(f'Trainer: {trainer.__module__}; device={config.DEVICE}', flush=True)

    executable = shutil.which('ffmpeg')
    if not executable:
        raise RuntimeError('FFmpeg executable is missing from PATH.')
    version = subprocess.run([executable, '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    print(f'FFmpeg: {executable}; {version}', flush=True)
    cascade = cv2.CascadeClassifier(str(root / 'dataset/haarcascade_frontalface_default.xml'))
    if cascade.empty():
        raise RuntimeError('OpenCV cannot load the MCD Haar cascade XML.')

    device = torch.device(config.DEVICE)
    if device.type == 'cuda':
        index = device.index if device.index is not None else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f'{device} is unavailable; visible GPUs={torch.cuda.device_count()}. Check DEVICE and CUDA_VISIBLE_DEVICES.')
        print(f'GPU: {torch.cuda.get_device_name(device)}; capability={torch.cuda.get_device_capability(device)}', flush=True)
    # This small synthetic gradient probe tests the filter/FFT/linear solve on
    # the chosen device. It does not assert real-data or model-training success.
    p = config.MODEL.PHASE_SHIFTED_RPPG
    t = torch.arange(p.FRAME_NUM, device=device, dtype=torch.float32) / config.TEST.DATA.FS
    reference = (torch.sin(2 * torch.pi * 1.2 * t) + .2 * torch.sin(2 * torch.pi * 2.4 * t)).repeat(2, 1)
    face = torch.sin(2 * torch.pi * 1.3 * t + .15).repeat(2, 1).requires_grad_()
    acral = torch.sin(2 * torch.pi * 1.1 * t + .25).repeat(2, 1).requires_grad_()
    criterion = PhaseShiftedStage1Loss(
        fs=config.TEST.DATA.FS, detrend_lambda=p.DETREND_LAMBDA,
        low_hz=p.LOW_HZ, high_hz=p.HIGH_HZ, filter_order=p.FILTER_ORDER,
        fft_multiplier=p.FFT_MULTIPLIER, hr_weight=p.HR_WEIGHT,
        frequency_weight=p.FREQUENCY_WEIGHT, pv_weight=p.PV_WEIGHT,
        waveform_weight=p.STAGE1_WAVEFORM_WEIGHT)
    loss, _ = criterion(face, acral, reference)
    if not torch.isfinite(loss):
        raise RuntimeError('Synthetic Stage-1 loss is nonfinite.')
    loss.backward()
    for name, value in [('facial', face), ('acral', acral)]:
        if value.grad is None or not torch.isfinite(value.grad).all() or not torch.any(value.grad != 0):
            raise RuntimeError(f'{name} synthetic gradient is absent, zero or nonfinite.')
    print(f'Stage-1 synthetic loss/backward: PASS ({float(loss.detach()):.6f})', flush=True)

    if args.check_data:
        from dataset.mcd.cleaning import load_clean_manifest
        from dataset.mcd_preprocessing import split_subjects
        data = config.TRAIN.DATA if config.TOOLBOX_MODE == 'train_and_test' else config.TEST.DATA
        if not data.MCD.CLEAN_MANIFEST:
            raise ValueError('CLEAN_MANIFEST is empty; this experiment must use the approved list.')
        rows, manifest = load_clean_manifest(data.MCD.CLEAN_MANIFEST, data.DATA_PATH, tuple(data.MCD.CAMERAS))
        if args.expected_records is not None and len(rows) != args.expected_records:
            raise ValueError(f'Approved count {len(rows)} != expected {args.expected_records}.')
        print(f'Manifest: {len(rows)} approved recordings; {len({r["patient_id"] for r in rows})} subjects; '
              f'signature={manifest["signature"]}', flush=True)
        sets = {}
        for name in ('TRAIN', 'VALID', 'TEST') if config.TOOLBOX_MODE == 'train_and_test' else ('TEST',):
            data = getattr(config, name).DATA
            selected = split_subjects(rows, data.BEGIN, data.END, data.MCD.SPLIT_SEED)
            sets[name] = {r['patient_id'] for r in selected}
            if not selected:
                raise ValueError(f'{name} split is empty.')
            print(f'{name}: {len(sets[name])} subjects; {len(selected)} recordings; seed={data.MCD.SPLIT_SEED}', flush=True)
        names = list(sets)
        if any(sets[a] & sets[b] for i, a in enumerate(names) for b in names[i + 1:]):
            raise ValueError('Subject overlap between configured splits.')
    print('Environment checks passed. Real video preprocessing and training have not been run.', flush=True)


if __name__ == '__main__':
    main()
