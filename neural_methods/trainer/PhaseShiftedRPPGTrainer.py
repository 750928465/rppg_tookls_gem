"""Backward-compatible automatic two-stage entry point.

For separate commands use phase_shifted_rppg.DRPNetTrainer / BBPNetTrainer.
"""
from neural_methods.trainer.phase_shifted_rppg.base import PhaseShiftedTrainerBase


class PhaseShiftedRPPGTrainer(PhaseShiftedTrainerBase):
    def train(self, data_loader):
        stage = self.p.TRAIN_STAGE
        self._check_loaders(data_loader, 'STAGE2' if stage in ('FULL', 'STAGE2') else 'STAGE1')
        if stage in ('FULL', 'STAGE1'):
            self.stage1_loss.check_dependencies()
            self.drpnet.requires_grad_(True)
            self._train_stage(data_loader, 'STAGE1', self.config.TRAIN.EPOCHS, self.config.TRAIN.LR)
        if stage in ('FULL', 'STAGE2'):
            path = self.best_paths['STAGE1'] if stage == 'FULL' else self.p.STAGE1_CHECKPOINT
            self._load_checkpoint(path, 'STAGE1')
            self._freeze_stage1()
            self._train_stage(data_loader, 'STAGE2', self.p.STAGE2_EPOCHS, self.p.STAGE2_LR)
            self._load_checkpoint(self.best_paths['STAGE2'], 'STAGE2')
            self.best_paths['FULL'] = self.save_model('Best', 'FULL')
