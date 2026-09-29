"""Unified Toolbox trainer for the two-stage Phase-Shifted rPPG method.

Adapted from GyutaeHwang/phase_shifted_rPPG, commit 7d46167.
Copyright (c) 2024 Gyutae Hwang, MIT; retained license:
neural_methods/model/phase_shifted_rppg/LICENSE.

Uses Toolbox loaders/splits and validation-only model selection. No original
KFold, dataset, augmentation, test-set selection or record directory is copied.
"""
from collections.abc import Mapping
from pathlib import Path
import json

import numpy as np
import torch
from scipy.signal import resample

from neural_methods.model.phase_shifted_rppg import DRPNet, BBPNet
from neural_methods.model.phase_shifted_rppg.inference import predict_signals, estimate_hr
from neural_methods.loss.PhaseShiftedLoss import (
    PhaseShiftedStage1Loss, PhaseShiftedStage2Loss, CuffBPLoss, batch_standardize,
)
from neural_methods.trainer.BaseTrainer import BaseTrainer


class PhaseShiftedRPPGTrainer(BaseTrainer):
    @staticmethod
    def validate_config(config):
        """Run before dataset construction so incompatible configs fail early."""
        p = config.MODEL.PHASE_SHIFTED_RPPG
        if config.TOOLBOX_MODE not in ('train_and_test', 'only_test'):
            raise ValueError('PhaseShiftedRPPG supports train_and_test or only_test.')
        if p.TRAIN_STAGE not in ('STAGE1', 'STAGE2', 'FULL'):
            raise ValueError('TRAIN_STAGE must be STAGE1, STAGE2 or FULL.')
        if p.LABEL_SOURCE not in ('ABP_MMHG', 'PPG_CUFF'):
            raise ValueError(
                'PhaseShiftedRPPG data contract is UNVERIFIED. The original method '
                'requires synchronized continuous blood-pressure waveform in mmHg; '
                'Stage 2 additionally needs real SBP/DBP. PPG/BVP is not ABP. '
                'Confirm the data source before setting LABEL_SOURCE: ABP_MMHG. '
                'See docs/PhaseShiftedRPPG.md.'
            )
        if p.LABEL_SOURCE == 'PPG_CUFF':
            if p.PREPROCESS_VERSION != 'mcd_ppg_cuff_v1':
                raise ValueError('MCD requires PREPROCESS_VERSION=mcd_ppg_cuff_v1.')
            if config.INFERENCE.BATCH_SIZE != 1:
                raise ValueError('MCD inference uses BATCH_SIZE=1 with upstream batch standardization.')
            if (p.FRAME_NUM != 150 or p.VIDEO_CHANNEL_ORDER != 'RGB' or p.PIXEL_SCALE != 255.
                    or p.LOW_HZ != .5 or p.HIGH_HZ != 3.):
                raise ValueError('MCD v1 requires 150 frames, Raw RGB/255 and 0.5-3 Hz HR band.')
        if config.NUM_OF_GPU_TRAIN != 1:
            raise ValueError('Initial PhaseShiftedRPPG integration uses one explicit DEVICE; set NUM_OF_GPU_TRAIN=1.')
        if not 128 <= p.FRAME_NUM < 256 or p.RESAMPLE_FACTOR < 1:
            raise ValueError('Require 128 <= FRAME_NUM < 256 and RESAMPLE_FACTOR >= 1.')
        if p.VIDEO_CHANNEL_ORDER not in ('RGB', 'BGR') or p.PIXEL_SCALE <= 0:
            raise ValueError('Specify source VIDEO_CHANNEL_ORDER RGB/BGR and positive PIXEL_SCALE.')
        if config.TOOLBOX_MODE == 'train_and_test':
            if config.TEST.USE_LAST_EPOCH:
                raise ValueError('PhaseShiftedRPPG requires validation-selected best checkpoints: USE_LAST_EPOCH=False.')
            if config.TRAIN.EPOCHS < 1 or p.STAGE2_EPOCHS < 1:
                raise ValueError('Both stage epoch counts must be positive.')
            if config.TRAIN.LR <= 0 or p.STAGE2_LR <= 0 or not 0 < p.LR_DECAY <= 1:
                raise ValueError('Invalid learning rate or LR_DECAY.')
            if p.TRAIN_STAGE == 'STAGE2' and not p.STAGE1_CHECKPOINT:
                raise ValueError('STAGE2 requires STAGE1_CHECKPOINT.')
            splits = ('TRAIN', 'VALID', 'TEST')
        else:
            if not config.INFERENCE.MODEL_PATH:
                raise ValueError('only_test requires INFERENCE.MODEL_PATH.')
            splits = ('TEST',)
        for split in splits:
            data = getattr(config, split).DATA
            pre = data.PREPROCESS
            if p.LABEL_SOURCE == 'PPG_CUFF' and data.DATASET != 'MCD-rPPG':
                raise ValueError('PPG_CUFF mode requires the MCD-rPPG loader.')
            if p.LABEL_SOURCE == 'PPG_CUFF':
                for key in ('CAMERAS', 'SPLIT_SEED', 'MAX_SYNC_ERROR', 'MAX_FRAME_GAP', 'MAX_TAIL_DROP'):
                    if getattr(data.MCD, key) != getattr(config.TEST.DATA.MCD, key):
                        raise ValueError(f'MCD {key} must match across active splits.')
            if data.DATA_FORMAT != 'NCDHW':
                raise ValueError(f'{split}: use NCDHW [B,3,T,H,W]; no implicit permutation is performed.')
            if pre.RESIZE.H != 128 or pre.RESIZE.W != 128 or pre.CHUNK_LENGTH != p.FRAME_NUM:
                raise ValueError(f'{split}: original DRPNet requires 128x128 and CHUNK_LENGTH=FRAME_NUM.')
            if data.FS <= 2 * p.HIGH_HZ or data.FS != config.TEST.DATA.FS:
                raise ValueError('All active splits must have the same verified FS above twice HIGH_HZ.')
            if list(pre.DATA_TYPE) != ['Raw'] or pre.LABEL_TYPE != 'Raw' or pre.USE_PSUEDO_PPG_LABEL:
                raise ValueError(f'{split}: require Raw video and Raw pressure labels; pseudo/difference labels are incompatible.')

    def __init__(self, config, data_loader):
        super().__init__()
        self.validate_config(config)
        self.config, self.p = config, config.MODEL.PHASE_SHIFTED_RPPG
        self.device = torch.device(config.DEVICE)
        self.model_dir = Path(config.MODEL.MODEL_DIR)
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME or 'PhaseShiftedRPPG'
        self.drpnet = DRPNet().to(self.device)
        self.bbpnet = BBPNet(self.p.SBP_MIN, self.p.SBP_MAX, self.p.DBP_MIN,
                           self.p.DBP_MAX, self.p.TEMPERATURE).to(self.device)
        self.stage1_loss = PhaseShiftedStage1Loss(
            fs=config.TEST.DATA.FS, detrend_lambda=self.p.DETREND_LAMBDA,
            low_hz=self.p.LOW_HZ, high_hz=self.p.HIGH_HZ,
            filter_order=self.p.FILTER_ORDER, fft_multiplier=self.p.FFT_MULTIPLIER,
            hr_weight=self.p.HR_WEIGHT, frequency_weight=self.p.FREQUENCY_WEIGHT,
            pv_weight=self.p.PV_WEIGHT, waveform_weight=self.p.STAGE1_WAVEFORM_WEIGHT)
        self.stage2_loss = PhaseShiftedStage2Loss(
            huber_delta=self.p.HUBER_DELTA, sbp_weight=self.p.SBP_WEIGHT,
            dbp_weight=self.p.DBP_WEIGHT, waveform_weight=self.p.STAGE2_WAVEFORM_WEIGHT)
        self.cuff = self.p.LABEL_SOURCE == 'PPG_CUFF'
        self.reference_key = 'ppg' if self.cuff else 'abp'
        if self.cuff:
            self.stage2_loss = CuffBPLoss(self.p.HUBER_DELTA, self.p.SBP_WEIGHT, self.p.DBP_WEIGHT)
        self.best_paths = {}

    def _signature(self):
        result = {key: getattr(self.p, key) for key in (
            'FRAME_NUM', 'RESAMPLE_FACTOR', 'PIXEL_SCALE', 'VIDEO_CHANNEL_ORDER',
            'SBP_MIN', 'SBP_MAX', 'DBP_MIN', 'DBP_MAX', 'TEMPERATURE',
            'LABEL_SOURCE', 'PREPROCESS_VERSION'
        )} | {'FS': self.config.TEST.DATA.FS}
        if self.cuff:
            result['MCD_POLICY'] = {k: getattr(self.config.TEST.DATA.MCD, k) for k in
                                    ('MAX_SYNC_ERROR', 'MAX_FRAME_GAP', 'MAX_TAIL_DROP')}
        return result

    def _video(self, raw):
        video = raw.to(self.device, dtype=torch.float32)
        if video.ndim != 5 or tuple(video.shape[1:]) != (3, self.p.FRAME_NUM, 128, 128):
            raise ValueError('Video must be NCDHW [B,3,FRAME_NUM,128,128].')
        if not torch.isfinite(video).all() or video.min() < 0 or video.max() > self.p.PIXEL_SCALE:
            raise ValueError('Raw video values must lie in [0, PIXEL_SCALE].')
        video = video / self.p.PIXEL_SCALE
        return video[:, [2, 1, 0], ...] if self.p.VIDEO_CHANNEL_ORDER == 'RGB' else video

    def _batch(self, batch, stage):
        """Preserve (video, labels, recording_id, chunk_id) Toolbox structure.

        Stage 1 accepts raw pressure [B,T] or {'abp': [B,T]}.
        Stage 2 requires {'abp': [B,T], 'sbp': [B], 'dbp': [B]}.
        No existing loader is modified and no BP label is inferred from PPG.
        """
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError('Expected a Toolbox (video, labels, ...) batch.')
        if not torch.is_tensor(batch[0]):
            raise ValueError('Require one video tensor; BigSmall multi-input batches need a future adapter.')
        video = self._video(batch[0])
        labels = batch[1]
        if self.cuff:
            required = {'ppg'} | ({'sbp', 'dbp', 'bp_valid'} if stage == 'STAGE2' else set())
            if not isinstance(labels, Mapping) or not required <= labels.keys():
                raise ValueError(f'MCD requires {sorted(required)}; PPG must not be relabeled ABP.')
            values = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
                      for k, v in labels.items() if k in ('ppg', 'sbp', 'dbp', 'bp_valid', 'weight')}
            if values['ppg'].shape != (video.shape[0], self.p.FRAME_NUM):
                raise ValueError('ppg must be [B, FRAME_NUM].')
            if not all(torch.isfinite(v).all() for v in values.values()):
                raise ValueError('Nonfinite MCD labels.')
            if stage == 'STAGE2':
                if any(values[k].shape != (video.shape[0],) for k in ('sbp', 'dbp', 'bp_valid')):
                    raise ValueError('Cuff labels must be [B].')
                if not values['bp_valid'].bool().all() or (values['sbp'] <= values['dbp']).any():
                    raise ValueError('Missing/invalid cuff BP: cannot train or evaluate Stage 2.')
                if ((values['sbp'] <= self.p.SBP_MIN) | (values['sbp'] >= self.p.SBP_MAX) |
                    (values['dbp'] <= self.p.DBP_MIN) | (values['dbp'] >= self.p.DBP_MAX)).any():
                    raise ValueError('Cuff target outside configured output range; never clip labels.')
            return video, values
        if stage == 'STAGE2' and (not isinstance(labels, Mapping) or
                                 not {'abp', 'sbp', 'dbp'} <= labels.keys()):
            raise ValueError('Stage 2 data blocker: need labels dict with real abp, sbp, dbp in mmHg. Existing scalar-waveform loaders do not provide this contract.')
        if isinstance(labels, Mapping):
            if 'abp' not in labels:
                raise ValueError('Missing continuous blood-pressure target: abp.')
            labels = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
                      for k, v in labels.items() if k in ('abp', 'sbp', 'dbp')}
        else:
            labels = {'abp': torch.as_tensor(labels, dtype=torch.float32, device=self.device)}
        if labels['abp'].shape != (video.shape[0], self.p.FRAME_NUM):
            raise ValueError('abp must have shape [B, FRAME_NUM], synchronized to the video.')
        if any(not torch.isfinite(v).all() for v in labels.values()):
            raise ValueError('Pressure labels contain NaN/Inf; apply a documented exclusion policy first.')
        if stage == 'STAGE2':
            if any(labels[k].shape != (video.shape[0],) for k in ('sbp', 'dbp')):
                raise ValueError('sbp/dbp must be window labels with shape [B].')
            if (labels['sbp'] <= labels['dbp']).any():
                raise ValueError('SBP must exceed DBP. Labels are never clipped to output ranges.')
        return video, labels

    def _resample(self, waveform):
        # Preserve SciPy FFT resample, T=150 -> 600 (25 -> 100 Hz for 6 s).
        # Deliberately detached: Stage 2 never updates DRPNet.
        values = resample(waveform.detach().cpu().numpy(),
                          self.p.FRAME_NUM * self.p.RESAMPLE_FACTOR, axis=-1)
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _freeze_stage1(self):
        self.drpnet.eval()
        self.drpnet.requires_grad_(False)

    def _stage2_forward(self, video):
        self._freeze_stage1()
        with torch.no_grad():
            facial, acral = self.drpnet(video)
            facial = batch_standardize(self._resample(facial))
            acral = batch_standardize(self._resample(acral))
        sbp, dbp = self.bbpnet(facial, acral)
        return sbp, dbp, acral

    def _check_loaders(self, data_loader, stage):
        if self.cuff:
            sets = {k: getattr(v.dataset, 'subject_ids', set()) for k, v in data_loader.items() if v is not None}
            for a, b in (('train', 'valid'), ('train', 'test'), ('valid', 'test')):
                if sets.get(a, set()) & sets.get(b, set()):
                    raise ValueError(f'Subject leakage between {a} and {b}.')
        for name in ('train', 'valid', 'test'):
            loader = data_loader.get(name)
            if loader is None or len(loader) == 0:
                raise ValueError(f'Missing or empty {name} loader.')
            # Check the actual data contract before ANY stage starts in FULL mode.
            self._batch(next(iter(loader)), stage)
            if self.cuff and stage == 'STAGE2' and any(not e['bp_valid'] for e in loader.dataset.entries):
                raise ValueError(f'{name}: missing cuff labels; resolve before starting either stage.')
            if self.cuff and stage == 'STAGE2':
                for row in loader.dataset.rows:
                    if not (row['bp_valid'] and self.p.SBP_MIN < row['sbp'] < self.p.SBP_MAX
                            and self.p.DBP_MIN < row['dbp'] < self.p.DBP_MAX):
                        raise ValueError(f'{name}: invalid/out-of-range cuff label in {row["recording_id"]}.')
            if name == 'train' and stage == 'STAGE2':
                size = getattr(loader, 'batch_size', None)
                if size is not None and (size < 2 or (not loader.drop_last and len(loader.dataset) % size == 1)):
                    raise ValueError('Stage 2 BatchNorm requires every training batch >=2. Adjust batch size or explicitly use drop_last in a future loader configuration.')

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

    def _train_stage(self, loaders, stage, epochs, lr):
        model = self.drpnet if stage == 'STAGE1' else self.bbpnet
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.p.LR_DECAY)
        best = float('inf')
        history = []
        for epoch in range(epochs):
            train = self._run_epoch(loaders['train'], stage, optimizer)
            valid = self._run_epoch(loaders['valid'], stage)
            # Upstream chooses mean branch HR MAE / mean BP MAE. Here select on
            # VALID, never TEST; preserve the metric while preventing test leakage.
            score = (valid['facial_hr_mae'] + valid['acral_hr_mae']) / 2 if stage == 'STAGE1' else (valid['sbp_mae'] + valid['dbp_mae']) / 2
            self.save_model(epoch, stage)
            if score < best:
                best = score
                self.best_paths[stage] = self.save_model('Best', stage)
            history.append({'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'],
                            'train': train, 'valid': valid})
            print(f'{stage} epoch={epoch} train={train} valid={valid}')
            scheduler.step()
        (self.model_dir / f'{self.model_file_name}_{stage}_history.json').write_text(
            json.dumps(history, indent=2), encoding='utf-8')

    def _run_epoch(self, loader, stage, optimizer=None):
        training = optimizer is not None
        self.drpnet.train(training and stage == 'STAGE1')
        self.bbpnet.train(training and stage == 'STAGE2')
        total, count, grouped = {}, 0, {}
        with torch.set_grad_enabled(training):
            for batch in loader:
                video, labels = self._batch(batch, stage)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                if stage == 'STAGE1':
                    facial, acral = self.drpnet(video)
                    loss, stats = self.stage1_loss(facial, acral, labels[self.reference_key])
                else:
                    sbp, dbp, acral = self._stage2_forward(video)
                    if self.cuff:
                        loss, stats = self.stage2_loss(sbp, dbp, labels['sbp'], labels['dbp'], labels.get('weight'))
                        if not training:
                            self._collect_groups(grouped, batch[1]['group_id'], sbp, dbp, labels)
                    else:
                        loss, stats = self.stage2_loss(sbp, dbp, acral, self._resample(labels['abp']), labels['sbp'], labels['dbp'])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite {stage} loss; inspect signal quality.')
                if training:
                    loss.backward()
                    optimizer.step()
                stats = dict(stats, loss=loss.detach())
                # Preserve upstream mean over batches, including the final batch.
                for key, value in stats.items():
                    total[key] = total.get(key, 0.) + float(value)
                count += 1
        if count == 0:
            raise ValueError('Empty loader.')
        result = {k: v / count for k, v in total.items()}
        if grouped:
            errors = self._group_errors(grouped)
            result.update(sbp_mae=float(np.abs(errors[:, 0]).mean()),
                          dbp_mae=float(np.abs(errors[:, 1]).mean()), n_states=len(grouped))
        return result

    @staticmethod
    def _collect_groups(groups, ids, sbp, dbp, labels):
        for i, key in enumerate(ids):
            target = [float(labels['sbp'][i]), float(labels['dbp'][i])]
            entry = groups.setdefault(key, {'target': target, 'predictions': []})
            if entry['target'] != target:
                raise ValueError(f'Conflicting cuff labels for state {key}.')
            entry['predictions'].append([float(sbp[i].detach()), float(dbp[i].detach())])

    @staticmethod
    def _group_errors(groups):
        return np.asarray([np.mean(v['predictions'], axis=0)-v['target'] for v in groups.values()])

    def valid(self, data_loader, stage=None):
        stage = stage or ('STAGE1' if self.p.TRAIN_STAGE == 'STAGE1' else 'STAGE2')
        return self._run_epoch(data_loader['valid'], stage)

    def save_model(self, index, stage):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        path = self.model_dir / f'{self.model_file_name}_{stage}_{index}.pth'
        payload = {'method': 'PhaseShiftedRPPG', 'stage': stage, 'signature': self._signature()}
        if stage in ('STAGE1', 'FULL'):
            payload['drpnet'] = self.drpnet.state_dict()
        if stage in ('STAGE2', 'FULL'):
            payload['bbpnet'] = self.bbpnet.state_dict()
        torch.save(payload, path)
        return str(path)

    def _load_checkpoint(self, path, stage):
        if not path or not Path(path).is_file():
            raise ValueError(f'Missing {stage} checkpoint: {path}')
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if self.cuff and 'signature' not in payload:
            raise ValueError('MCD requires a signed PPG_CUFF checkpoint; raw legacy weights have no supervision provenance.')
        if 'signature' in payload:
            saved = dict(payload['signature'])
            # Old signed bundles predate explicit label-source fields; they were
            # ABP-only. Never interpret them as PPG_CUFF.
            saved.setdefault('LABEL_SOURCE', 'ABP_MMHG')
            saved.setdefault('PREPROCESS_VERSION', 'legacy_abp_v1')
            if saved != self._signature():
                raise ValueError('Checkpoint preprocessing/model signature differs from the active config.')
        names = ('drpnet', 'bbpnet') if stage == 'FULL' else (('drpnet',) if stage == 'STAGE1' else ('bbpnet',))
        for name in names:
            if name in payload:
                state = payload[name]
            elif stage != 'FULL' and 'method' not in payload:
                # Author checkpoints are plain state_dict, optionally DataParallel.
                state = payload.get('state_dict', payload)
            else:
                raise ValueError(f'Checkpoint is missing {name}.')
            state = {k.removeprefix('module.'): v for k, v in state.items()}
            getattr(self, name).load_state_dict(state, strict=True)

    def predict_clip(self, raw_video, with_bp=True):
        """Label-free [1,3,T,128,128] Raw RGB input; caller must load trained weights."""
        if raw_video.shape[0] != 1:
            raise ValueError('Use batch size 1 for stable batch-standardized inference.')
        return predict_signals(self.drpnet, self.bbpnet if with_bp else None,
                               self._video(raw_video), self.config.TEST.DATA.FS,
                               self.p.RESAMPLE_FACTOR, self.p.LOW_HZ, self.p.HIGH_HZ)

    def test(self, data_loader):
        loader = data_loader.get('test')
        if loader is None or len(loader) == 0:
            raise ValueError('Missing or empty test loader.')
        stage = 'STAGE1' if self.p.TRAIN_STAGE == 'STAGE1' else 'STAGE2'
        self._batch(next(iter(loader)), stage)
        if self.config.TOOLBOX_MODE == 'only_test':
            path = self.config.INFERENCE.MODEL_PATH
            if stage == 'STAGE1':
                self._load_checkpoint(path, 'STAGE1')
            elif self.p.STAGE1_CHECKPOINT:
                self._load_checkpoint(self.p.STAGE1_CHECKPOINT, 'STAGE1')
                self._load_checkpoint(path, 'STAGE2')
            else:
                self._load_checkpoint(path, 'FULL')
        elif stage == 'STAGE1':
            self._load_checkpoint(self.best_paths['STAGE1'], 'STAGE1')
        else:
            self._load_checkpoint(self.best_paths['FULL'], 'FULL')
        self.drpnet.eval()
        self.bbpnet.eval()
        records, face_errors, acral_errors, grouped = [], [], [], {}
        with torch.no_grad():
            for batch in loader:
                video, labels = self._batch(batch, stage)
                record = {'labels': {k: v.cpu() for k, v in labels.items()}}
                if len(batch) >= 4:
                    record.update(recording_id=batch[2], chunk_id=batch[3])
                if self.cuff:
                    record.update(time=batch[1]['time'], max_sync_error=batch[1]['max_sync_error'],
                                  group_id=batch[1]['group_id'])
                    outputs = predict_signals(self.drpnet, self.bbpnet if stage == 'STAGE2' else None,
                                              video, self.config.TEST.DATA.FS, self.p.RESAMPLE_FACTOR,
                                              self.p.LOW_HZ, self.p.HIGH_HZ)
                    record.update(outputs)
                    ref_hr, ref_valid = estimate_hr(labels['ppg'].cpu().numpy(), self.config.TEST.DATA.FS,
                                                   self.p.LOW_HZ, self.p.HIGH_HZ)
                    valid = ref_valid & outputs['hr_valid'].numpy()
                    face_errors.extend((outputs['hr_facial'].numpy()-ref_hr)[valid].tolist())
                    acral_errors.extend((outputs['hr_acral'].numpy()-ref_hr)[valid].tolist())
                    if stage == 'STAGE2':
                        self._collect_groups(grouped, batch[1]['group_id'], outputs['sbp'], outputs['dbp'], labels)
                    records.append(record)
                    continue
                if stage == 'STAGE1':
                    face, acral = self.drpnet(video)
                    ref_hr = self.stage1_loss.process(labels['abp'])[2]
                    face_hr, acral_hr = self.stage1_loss.process(face)[2], self.stage1_loss.process(acral)[2]
                    face_errors.extend((face_hr - ref_hr).cpu().tolist())
                    acral_errors.extend((acral_hr - ref_hr).cpu().tolist())
                    record.update(facial=face.cpu(), acral=acral.cpu())
                else:
                    sbp, dbp, _ = self._stage2_forward(video)
                    record.update(sbp=sbp.cpu(), dbp=dbp.cpu())
                records.append(record)
        if self.cuff:
            metrics = {k: self._error_metrics(v) for k, v in
                       (('facial_hr_bpm', face_errors), ('acral_hr_bpm', acral_errors))}
            if grouped:
                errors = self._group_errors(grouped)
                for i, key in enumerate(('sbp', 'dbp')):
                    state_metrics = self._error_metrics(errors[:, i])
                    state_metrics['N_states'] = state_metrics.pop('N_windows')
                    metrics[f'{key}_state_mmHg'] = state_metrics
                    metrics[f'{key}_window_mmHg'] = self._error_metrics(torch.cat(
                        [r[key]-r['labels'][key] for r in records]).numpy())
        elif stage == 'STAGE1':
            metrics = {k: self._error_metrics(v) for k, v in
                       (('facial_hr_bpm', face_errors), ('acral_hr_bpm', acral_errors))}
        else:
            metrics = {f'{key}_mmHg': self._error_metrics(torch.cat(
                [r[key] - r['labels'][key] for r in records]).numpy()) for key in ('sbp', 'dbp')}
        output = Path(self.config.TEST.OUTPUT_SAVE_DIR)
        output.mkdir(parents=True, exist_ok=True)
        torch.save({'stage': stage, 'signature': self._signature(), 'records': records},
                   output / f'{self.model_file_name}_{stage}_outputs.pth')
        if self.cuff:
            (output / f'{self.model_file_name}_state_predictions.json').write_text(json.dumps(grouped, indent=2))
        (output / f'{self.model_file_name}_{stage}_metrics.json').write_text(json.dumps(metrics, indent=2))
        print('PhaseShiftedRPPG test metrics:', metrics)
        return metrics

    @staticmethod
    def _error_metrics(errors):
        errors = np.asarray(errors, dtype=float)
        if errors.size == 0:
            return {'MAE': None, 'RMSE': None, 'ME': None, 'SD': None, 'N_windows': 0}
        return {'MAE': float(np.abs(errors).mean()), 'RMSE': float(np.sqrt((errors ** 2).mean())),
                'ME': float(errors.mean()), 'SD': float(errors.std()), 'N_windows': int(errors.size)}
