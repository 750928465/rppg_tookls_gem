"""Forward/gradient checks with synthetic tensors, never a training experiment.

Run from the Toolbox root: python tools/phase_shifted_smoke_test.py
Optional --upstream-dir points to the fixed source snapshot for numeric parity.
Optional --stage1-loss checks the original differentiable torchaudio filter too.
"""
import argparse
import ast
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scipy.signal import resample
from neural_methods.model.phase_shifted_rppg import DRPNet, BBPNet
from neural_methods.loss.PhaseShiftedLoss import (
    PVLoss, PhaseShiftedStage1Loss, PhaseShiftedStage2Loss, batch_standardize,
)


def original_module(path):
    # Only remove unused torchinfo import; preserve original model definitions.
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not
                 (isinstance(n, ast.ImportFrom) and n.module == 'torchinfo')]
    namespace = {}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-dir', type=Path)
    parser.add_argument('--stage1-loss', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(100)
    drp = DRPNet().eval()
    bbp = BBPNet().eval()
    shapes = {}
    hooks = []
    for name in ('Feature_extracter', 'DilatedConv', 'Attention_temporal',
                 'Attention_spatial', 'face_rPPG_extraction', 'finger_rPPG_extraction'):
        hooks.append(getattr(drp, name).register_forward_hook(
            lambda module, inputs, output, name=name: shapes.update({name: list(output.shape)})))
    video = torch.rand(1, 3, 150, 128, 128)
    with torch.no_grad():
        facial, acral = drp(video)
        assert facial.shape == acral.shape == (1, 150)
        facial_up, acral_up = [batch_standardize(torch.from_numpy(resample(x.numpy(), 600, axis=1)))
                              for x in (facial, acral)]
        features = bbp.waveform_features(facial_up, acral_up)
        assert features.shape == (1, 6, 600)
        sbp, dbp = bbp(facial_up, acral_up)
        assert sbp.shape == dbp.shape == (1,)
        assert 85 <= sbp.item() <= 155 and 45 <= dbp.item() <= 95
        assert sbp.dtype == torch.float32 and sbp.device.type == 'cpu'
    for hook in hooks:
        hook.remove()
    print('DRPNet shapes:', shapes)
    print('Full forward: [1,3,150,128,128] -> two [1,150] -> two [1,600] -> SBP/DBP [1]. PASS')

    if args.upstream_dir:
        original_drp = original_module(args.upstream_dir / 'Shifted_rPPG_extraction.py')['Shifted_rPPG_extraction']().eval()
        original_bbp = original_module(args.upstream_dir / 'BP_estimation.py')['BP_estimation']().eval()
        original_drp.load_state_dict(drp.state_dict(), strict=True)
        original_bbp.load_state_dict(bbp.state_dict(), strict=True)
        with torch.no_grad():
            for old, new in zip(original_drp(video), (facial, acral)):
                torch.testing.assert_close(old, new, rtol=0, atol=0)
            for old, new in zip(original_bbp(facial_up, acral_up, 'cpu'), (sbp, dbp)):
                torch.testing.assert_close(old, new, rtol=0, atol=0)
        print('Upstream state_dict + DRPNet/BBPNet numerical parity: exact match. PASS')

    # Synthetic values only test the gradient graph; never feed a real loader.
    waves = torch.randn(2, 600)
    bbp.train()
    sbp, dbp = bbp(waves, waves.roll(2, -1))
    loss, _ = PhaseShiftedStage2Loss()(sbp, dbp, waves, waves * 10 + 100,
                                     torch.tensor([120., 125.]), torch.tensor([75., 80.]))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in bbp.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in bbp.parameters())
    print('BBPNet + original Stage-2 objective backward (no optimizer step): PASS')

    t = torch.arange(150) / 25
    ref = torch.stack([torch.sin(2 * torch.pi * f * t) for f in (1., 1.2)])
    pred = (ref * 0.8).requires_grad_()
    pv = PVLoss()(pred, ref)
    pv.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
    print('PVLoss value/gradient: PASS')
    if args.stage1_loss:
        face, acral = (ref + 0.03 * torch.randn_like(ref)).requires_grad_(), (ref * 0.9).requires_grad_()
        loss, _ = PhaseShiftedStage1Loss()(face, acral, ref)
        loss.backward()
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (face, acral))
        print('Stage-1 torchaudio-filtered loss backward: PASS')
    else:
        print('Stage-1 filter/loss backward: NOT RUN (use --stage1-loss with matching torchaudio).')


if __name__ == '__main__':
    main()
