"""Label-free waveform/HR/BP prediction. No training or torchaudio dependency."""
import numpy as np
import torch
from scipy.signal import butter, detrend, sosfiltfilt, resample
from neural_methods.loss.PhaseShiftedLoss import batch_standardize


def estimate_hr(waves, fs, low=.5, high=3.):
    """Reporting HR: linear detrend + zero-phase SOS + Hamming FFT peak.

    This is a reporting method, distinct from the differentiable Stage-1 filter.
    quality=False denotes only invalid/flat signals, not clinical validation.
    """
    hr, valid = [], []
    sos = butter(3, [low, high], btype='band', fs=fs, output='sos')
    for wave in np.asarray(waves):
        ok = bool(np.isfinite(wave).all() and np.std(wave) > 1e-6)
        value = np.nan
        if ok:
            filtered = sosfiltfilt(sos, detrend(wave))
            spec = np.abs(np.fft.rfft(filtered*np.hamming(len(wave)), n=len(wave)*40))
            freq = np.fft.rfftfreq(len(wave)*40, 1/fs)
            band = (freq >= low) & (freq <= high)
            ok = bool(spec[band].max() > 1e-8)
            if ok: value = float(freq[band][spec[band].argmax()]*60)
        hr.append(value); valid.append(ok)
    return np.asarray(hr), np.asarray(valid)


@torch.no_grad()
def predict_signals(drpnet, bbpnet, prepared_video, fs=25., factor=4, low=.5, high=3.):
    drpnet.eval()
    face, acral = drpnet(prepared_video)
    face_hr, face_valid = estimate_hr(face.cpu().numpy(), fs, low, high)
    acral_hr, acral_valid = estimate_hr(acral.cpu().numpy(), fs, low, high)
    result = {'facial_rppg': face.cpu(), 'acral_rppg': acral.cpu(),
              'hr_facial': torch.from_numpy(face_hr), 'hr_acral': torch.from_numpy(acral_hr),
              'hr_valid': torch.from_numpy(face_valid & acral_valid)}
    if bbpnet is not None:
        bbpnet.eval()
        def up(w):
            x = resample(w.cpu().numpy(), w.shape[-1]*factor, axis=-1)
            return batch_standardize(torch.as_tensor(x, device=w.device, dtype=w.dtype))
        sbp, dbp = bbpnet(up(face), up(acral))
        result.update(sbp=sbp.cpu(), dbp=dbp.cpu(), bp_order_valid=(sbp > dbp).cpu())
    return result
