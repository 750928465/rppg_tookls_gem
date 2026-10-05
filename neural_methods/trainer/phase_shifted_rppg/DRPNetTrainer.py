"""Stage 1: train DRPNet, select validation-best weights, evaluate rPPG and HR."""
from .base import PhaseShiftedTrainerBase


class DRPNetTrainer(PhaseShiftedTrainerBase):
    @staticmethod
    def validate_config(config):
        if config.MODEL.PHASE_SHIFTED_RPPG.TRAIN_STAGE != 'STAGE1':
            raise ValueError('MODEL.NAME=DRPNet requires TRAIN_STAGE=STAGE1.')
        PhaseShiftedTrainerBase.validate_config(config)

    def train(self, data_loader):
        self._check_loaders(data_loader, 'STAGE1')
        self.stage1_loss.check_dependencies()
        self.drpnet.requires_grad_(True)
        self._train_stage(data_loader, 'STAGE1', self.config.TRAIN.EPOCHS, self.config.TRAIN.LR)
        self._load_checkpoint(self.best_paths['STAGE1'], 'STAGE1')

    def predict_clip(self, raw_video, with_bp=False):
        if with_bp:
            raise ValueError('DRPNet produces rPPG/HR; use BBPNetTrainer for BP prediction.')
        return super().predict_clip(raw_video, with_bp=False)
