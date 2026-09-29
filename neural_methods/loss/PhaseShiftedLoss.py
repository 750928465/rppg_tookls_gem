"""Losses adapted from GyutaeHwang/phase_shifted_rPPG, commit 7d46167.

Copyright (c) 2024 Gyutae Hwang, MIT; license is retained at
neural_methods/model/phase_shifted_rppg/LICENSE.
Sources: PV_MSELoss.py, utils.py, main_stage1.py and main_stage2.py.
No TensorFlow/FILM, dataset code, plotting or CUDA reset is imported here.
"""
import torch
from torch import nn


def batch_standardize(signal):
    """Preserve upstream global batch mean/std (including unbiased std).

    This is deliberately not per-window normalization. Consequently predictions
    depend on batch composition; changing that would change the original method.
    """
    std = signal.std()
    if not torch.isfinite(signal).all() or not torch.isfinite(std) or std <= 0:
        raise ValueError("Phase-Shifted rPPG requires finite, nonconstant signals.")
    return (signal - signal.mean()) / std


def peak_valley_values(signal):
    """Original strict sign-change extrema and above/below-mean selection.

    Extrema indices are nondifferentiable, selected amplitudes retain gradients.
    Invalid flat/monotone windows fail explicitly instead of inventing extrema.
    """
    diff = torch.diff(signal)
    peaks = signal[1:-1][(diff[:-1] > 0) & (diff[1:] < 0)]
    valleys = signal[1:-1][(diff[:-1] < 0) & (diff[1:] > 0)]
    if peaks.numel() == 0 or valleys.numel() == 0:
        raise ValueError("PV loss needs at least one strict peak and valley in each waveform.")
    return torch.stack((peaks[peaks >= peaks.mean()].mean(),
                        valleys[valleys <= valleys.mean()].mean()))


class PVLoss(nn.Module):
    def forward(self, preds, labels):
        if preds.ndim != 2 or preds.shape != labels.shape:
            raise ValueError("PVLoss expects matching [B, T] waveforms.")
        pred_values = torch.stack([peak_valley_values(x) for x in preds])
        label_values = torch.stack([peak_valley_values(x) for x in labels])
        return nn.functional.mse_loss(pred_values, label_values)


class PhaseShiftedStage1Loss(nn.Module):
    """Facial: 1e-4 HR-MSE + 100 spectrum-MSE + PV-MSE.

    Acral: the same plus filtered waveform MSE. Sum both branches for backward;
    upstream performs two backward calls before one optimizer step, not averaging.
    HR uses FFT argmax, so its reported MSE has no gradient path to the waveform.
    """
    def __init__(self, fs=25., detrend_lambda=50., low_hz=0.5, high_hz=3.,
                 filter_order=3, fft_multiplier=40, hr_weight=1e-4,
                 frequency_weight=100., pv_weight=1., waveform_weight=1.):
        super().__init__()
        if not 0 < low_hz < high_hz < fs / 2:
            raise ValueError("Invalid bandpass frequency range.")
        self.fs, self.detrend_lambda = fs, detrend_lambda
        self.low_hz, self.high_hz = low_hz, high_hz
        self.filter_order, self.fft_multiplier = filter_order, fft_multiplier
        self.hr_weight, self.frequency_weight = hr_weight, frequency_weight
        self.pv_weight, self.waveform_weight = pv_weight, waveform_weight
        self.pv = PVLoss()

    @staticmethod
    def check_dependencies():
        # torchaudio.filtfilt is a CORE differentiable Stage-1 operation. SciPy
        # filtfilt has different padding/initial conditions and cannot replace it.
        try:
            from torchaudio.functional import filtfilt
            from scipy.signal import butter
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Stage 1 needs scipy and a torchaudio build compatible with the "
                "installed torch. No automatic installation or downgrade is performed."
            ) from exc
        return filtfilt, butter

    def process(self, waveform):
        filtfilt, butter = self.check_dependencies()
        from scipy import sparse
        import numpy as np

        waveform = batch_standardize(waveform)
        length = waveform.shape[-1]
        # Construct in NumPy float64 then cast, as in upstream utils.py.
        d2 = sparse.diags([1, -2, 1], [0, 1, 2], shape=(length - 2, length)).toarray()
        system = np.eye(length) + self.detrend_lambda ** 2 * (d2.T @ d2)
        system = torch.as_tensor(system, dtype=waveform.dtype, device=waveform.device)
        detrended = waveform - torch.linalg.solve(system, waveform.T).T
        b, a = butter(self.filter_order, [self.low_hz, self.high_hz], fs=self.fs, btype='band')
        a = torch.as_tensor(a, dtype=waveform.dtype, device=waveform.device)
        b = torch.as_tensor(b, dtype=waveform.dtype, device=waveform.device)
        filtered = filtfilt(detrended, a, b, clamp=False)
        window = torch.hamming_window(length, dtype=waveform.dtype, device=waveform.device)
        n_fft = length * self.fft_multiplier
        amplitude = torch.fft.fft(filtered * window, n=n_fft, dim=-1).abs() / length
        frequency = torch.fft.fftfreq(n_fft, 1 / self.fs, device=waveform.device)
        mask = (self.low_hz <= frequency) & (frequency <= self.high_hz)
        amplitude, frequency = amplitude[:, mask], frequency[mask]
        # Do not hardcode upstream's 600 frequency bins: float boundary rounding
        # and changes in fs/length determine the actual number of retained bins.
        hr = frequency[amplitude.argmax(dim=-1)] * 60
        return filtered, amplitude, hr

    def forward(self, facial, acral, reference):
        if facial.ndim != 2 or facial.shape != acral.shape or facial.shape != reference.shape:
            raise ValueError("Stage 1 expects three matching [B, T] waveforms.")
        ref, ref_amp, ref_hr = self.process(reference)
        face, face_amp, face_hr = self.process(facial)
        finger, finger_amp, finger_hr = self.process(acral)
        mse = nn.functional.mse_loss
        face_loss = (self.hr_weight * mse(face_hr, ref_hr)
                     + self.frequency_weight * mse(face_amp, ref_amp)
                     + self.pv_weight * self.pv(face, ref))
        acral_loss = (self.hr_weight * mse(finger_hr, ref_hr)
                      + self.frequency_weight * mse(finger_amp, ref_amp)
                      + self.pv_weight * self.pv(finger, ref)
                      + self.waveform_weight * mse(finger, ref))
        return face_loss + acral_loss, {
            'facial_loss': face_loss.detach(), 'acral_loss': acral_loss.detach(),
            'facial_hr_mae': (face_hr - ref_hr).abs().mean(),
            'acral_hr_mae': (finger_hr - ref_hr).abs().mean(),
        }


class PhaseShiftedStage2Loss(nn.Module):
    """Original Huber(SBP) + Huber(DBP) + MSE(reconstructed pressure, ABP).

    SBP/DBP alone cannot reproduce this objective: raw continuous pressure in
    mmHg is also required. Do not normalize the ABP target or substitute PPG.
    """
    def __init__(self, huber_delta=1., sbp_weight=1., dbp_weight=1., waveform_weight=1.):
        super().__init__()
        self.huber = nn.HuberLoss(delta=huber_delta)
        self.sbp_weight, self.dbp_weight = sbp_weight, dbp_weight
        self.waveform_weight = waveform_weight

    def forward(self, sbp, dbp, acral, abp, target_sbp, target_dbp):
        if acral.ndim != 2 or acral.shape != abp.shape:
            raise ValueError("Stage 2 needs acral and raw ABP with matching [B, L] shapes.")
        if any(x.shape != (acral.shape[0],) for x in (sbp, dbp, target_sbp, target_dbp)):
            raise ValueError("Each SBP/DBP prediction and label must have shape [B].")
        span = acral.amax(dim=-1, keepdim=True) - acral.amin(dim=-1, keepdim=True)
        if (span <= 0).any():
            raise ValueError("Cannot reconstruct pressure from a constant acral waveform.")
        normalized = (acral - acral.amin(dim=-1, keepdim=True)) / span
        reconstructed = normalized * (sbp - dbp)[:, None] + dbp[:, None]
        sbp_loss = self.huber(sbp, target_sbp)
        dbp_loss = self.huber(dbp, target_dbp)
        wave_loss = nn.functional.mse_loss(reconstructed, abp)
        loss = (self.sbp_weight * sbp_loss + self.dbp_weight * dbp_loss
                + self.waveform_weight * wave_loss)
        return loss, {'sbp_loss': sbp_loss.detach(), 'dbp_loss': dbp_loss.detach(),
                      'waveform_loss': wave_loss.detach(),
                      'sbp_mae': (sbp - target_sbp).abs().mean().detach(),
                      'dbp_mae': (dbp - target_dbp).abs().mean().detach()}


class CuffBPLoss(nn.Module):
    """State-level cuff targets only. No ABP argument or reconstructed pressure.

    Dataset weights compensate for unequal window counts per subject/state.
    A constant epsilon cannot replace missing supervision and is not added.
    """
    def __init__(self, huber_delta=1., sbp_weight=1., dbp_weight=1.):
        super().__init__()
        self.delta, self.sw, self.dw = huber_delta, sbp_weight, dbp_weight

    def forward(self, sbp, dbp, target_sbp, target_dbp, weight=None):
        if any(v.shape != sbp.shape for v in (dbp, target_sbp, target_dbp)) or sbp.ndim != 1:
            raise ValueError('Cuff predictions and targets must have matching [B] shapes.')
        if weight is None:
            weight = torch.ones_like(sbp)
        if weight.shape != sbp.shape or not torch.isfinite(weight).all() or (weight <= 0).any():
            raise ValueError('Invalid per-state window weights.')
        sl = nn.functional.huber_loss(sbp, target_sbp, reduction='none', delta=self.delta)
        dl = nn.functional.huber_loss(dbp, target_dbp, reduction='none', delta=self.delta)
        loss = (weight*(self.sw*sl+self.dw*dl)).mean()
        return loss, {'sbp_loss': sl.mean().detach(), 'dbp_loss': dl.mean().detach(),
                      'sbp_mae': (sbp-target_sbp).abs().mean().detach(),
                      'dbp_mae': (dbp-target_dbp).abs().mean().detach()}
