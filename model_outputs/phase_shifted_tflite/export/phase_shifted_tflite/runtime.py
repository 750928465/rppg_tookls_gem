"""TFLite-only signal/BP inference. No PyTorch import or checkpoint needed."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def upsample_standardize(wave):
    wave = np.asarray(wave, dtype=np.float32)
    if wave.shape != (1, 150) or not np.isfinite(wave).all():
        raise ValueError('Require one finite [1,150] DRPNet waveform.')
    x = resample(wave, 600, axis=-1).astype(np.float32)
    std = x.std(ddof=1)
    if not np.isfinite(std) or std <= 0:
        raise ValueError('Cannot standardize a flat/nonfinite waveform.')
    return ((x-x.mean())/std).astype(np.float32)


def interpreter(path, threads=2):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
    obj = Interpreter(model_path=str(path), num_threads=threads)
    obj.allocate_tensors()
    return obj


class TFLitePredictor:
    def __init__(self, model_dir, variant='fp32', threads=2):
        root = Path(model_dir)
        self.manifest = json.loads((root/'manifest.json').read_text())
        if self.manifest.get('schema') != 'phase_shifted_tflite_v1':
            raise ValueError('Unsupported deployment manifest.')
        if not self.manifest.get('verified'):
            raise ValueError('This model bundle has not passed numerical verification.')
        self.variant = variant
        entry = self.manifest['variants'][variant]
        self.models = {}
        for name in ('DRPNet', 'BBPNet'):
            path = root/entry[name]['file']
            if sha256(path) != entry[name]['sha256']:
                raise ValueError(f'{name}: model hash differs from manifest.')
            self.models[name] = interpreter(path, threads)
        self.drp = self.models['DRPNet'].get_signature_runner()
        self.bbp = self.models['BBPNet'].get_signature_runner()

    def predict_prepared(self, video):
        video = np.asarray(video, dtype=np.float32)
        if video.shape != (1, 3, 150, 128, 128) or not np.isfinite(video).all():
            raise ValueError('Require finite prepared video [1,3,150,128,128].')
        if video.min() < 0 or video.max() > 1:
            raise ValueError('Prepared input must be BGR in [0,1].')
        waves = self.drp(video=np.ascontiguousarray(video))
        bp = self.bbp(facial=upsample_standardize(waves['facial_rppg']),
                      acral=upsample_standardize(waves['acral_rppg']))
        if not all(np.isfinite(v).all() for v in [*waves.values(), *bp.values()]):
            raise FloatingPointError('Nonfinite model prediction.')
        return dict(waves, **bp)

    def predict_rgb(self, frames):
        frames = np.asarray(frames)
        if frames.shape != (150, 128, 128, 3) or not np.isfinite(frames).all():
            raise ValueError('Require cropped RGB frames [150,128,128,3].')
        if frames.min() < 0 or frames.max() > 255:
            raise ValueError('Raw RGB must lie in [0,255].')
        video = frames.transpose(3, 0, 1, 2)[None].astype(np.float32)/255.
        return self.predict_prepared(video[:, [2, 1, 0]])
