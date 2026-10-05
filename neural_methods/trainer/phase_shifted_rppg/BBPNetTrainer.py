"""Stage 2: load and freeze selected DRPNet, train BBPNet, evaluate BP."""
from pathlib import Path

from .base import PhaseShiftedTrainerBase


class BBPNetTrainer(PhaseShiftedTrainerBase):
    @staticmethod
    def validate_config(config):
        p = config.MODEL.PHASE_SHIFTED_RPPG
        if p.TRAIN_STAGE != 'STAGE2':
            raise ValueError('MODEL.NAME=BBPNet requires TRAIN_STAGE=STAGE2.')
        PhaseShiftedTrainerBase.validate_config(config)
        if config.TOOLBOX_MODE == 'train_and_test' or p.STAGE1_CHECKPOINT:
            if not Path(p.STAGE1_CHECKPOINT).is_file():
                raise ValueError(f'DRPNet checkpoint does not exist: {p.STAGE1_CHECKPOINT}. Run Stage 1 first.')

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        if config.TOOLBOX_MODE == 'train_and_test':
            self._load_checkpoint(self.p.STAGE1_CHECKPOINT, 'STAGE1')
        self._freeze_stage1()

    def train(self, data_loader):
        self._check_loaders(data_loader, 'STAGE2')
        self._load_checkpoint(self.p.STAGE1_CHECKPOINT, 'STAGE1')
        self._freeze_stage1()
        # Each standalone trainer uses the standard Toolbox TRAIN.EPOCHS / TRAIN.LR.
        self._train_stage(data_loader, 'STAGE2', self.config.TRAIN.EPOCHS, self.config.TRAIN.LR)
        self._load_checkpoint(self.best_paths['STAGE2'], 'STAGE2')
        # Also save a self-contained bundle for video-only inference.
        self.best_paths['FULL'] = self.save_model('Best', 'FULL')
