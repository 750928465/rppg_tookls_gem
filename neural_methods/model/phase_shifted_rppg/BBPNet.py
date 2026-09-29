"""Adapted from GyutaeHwang/phase_shifted_rPPG.
Upstream commit: 7d46167a8a22acb37a03e1410e8fff86482c5f1b.
Copyright (c) 2024 Gyutae Hwang. MIT license: see LICENSE in this package.
Layer names are preserved to support upstream state_dict checkpoints.
"""
import torch
import torch.nn as nn


def weight_init_kaiming_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv1d):
        torch.nn.init.kaiming_uniform_(submodule.weight)
        nn.init.constant_(submodule.bias, 0)
    elif isinstance(submodule, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(submodule.weight, nonlinearity='relu')
        nn.init.constant_(submodule.bias, 0)
    elif isinstance(submodule, nn.BatchNorm1d):
        nn.init.constant_(submodule.weight, 1)
        nn.init.constant_(submodule.bias, 0)


class BBPNet(nn.Module):
    def __init__(self, sbp_min=85., sbp_max=155., dbp_min=45., dbp_max=95., temperature=0.5):
        super(BBPNet, self).__init__()
        self.SBP_MAX = sbp_max
        self.SBP_MIN = sbp_min
        self.DBP_MAX = dbp_max
        self.DBP_MIN = dbp_min
        self.TEMPERATURE = temperature
        if not sbp_min < sbp_max or not dbp_min < dbp_max or temperature <= 0:
            raise ValueError("BP ranges must increase and temperature must be positive.")

        self.PwConv1 = nn.Conv1d(6, 16, 1, stride=1, padding=0)
        self.pool1d = nn.MaxPool1d(2, 2)
        self.act = nn.Hardswish()

        self.Backbone_layer1 = MSF_block(16, 64, 32)
        self.Backbone_layer2 = MSF_block(32, 64, 32)
        self.Backbone_layer3 = MSF_block(32, 64, 32)
        self.Backbone_layer4 = MSF_block(32, 64, 32)
        self.Backbone_layer5 = MSF_block(32, 128, 64)
        self.Backbone_layer6 = MSF_block(64, 128, 64)
        self.Backbone_layer7 = MSF_block(64, 512, 128)

        self.Subnetwork1 = BAM_regressor(128)
        self.Subnetwork2 = BAM_regressor(128)

        torch.nn.init.kaiming_uniform_(self.PwConv1.weight)
        nn.init.constant_(self.PwConv1.bias, 0)

    @staticmethod
    def waveform_features(facial_PPG, acral_PPG):
        """Upstream forward differences with zero append, without multiplying by fs.

        PPG/VPG/APG here are discrete waveform features, not derivatives in SI units.
        Ordering is facial PPG/VPG/APG then acral PPG/VPG/APG: [B, 6, L].
        """
        if facial_PPG.ndim != 2 or facial_PPG.shape != acral_PPG.shape:
            raise ValueError("BBPNet expects two equally shaped [B, L] waveforms.")
        if facial_PPG.device != acral_PPG.device or facial_PPG.dtype != acral_PPG.dtype:
            raise ValueError("Facial and acral waveforms must share device and dtype.")
        features = []
        for ppg in (facial_PPG, acral_PPG):
            vpg = torch.diff(ppg, dim=1, append=ppg.new_zeros(ppg.shape[0], 1))
            apg = torch.diff(ppg, dim=1, n=2, append=ppg.new_zeros(ppg.shape[0], 2))
            features.extend((ppg, vpg, apg))
        return torch.stack(features, dim=1)

    def forward(self, facial_PPG, acral_PPG):
        physiological_signals = self.waveform_features(facial_PPG, acral_PPG)
        if physiological_signals.shape[-1] < 16:
            raise ValueError("BBPNet needs at least 16 samples for its four pooling layers.")
        if self.training and physiological_signals.shape[0] < 2:
            raise ValueError("BBPNet training needs batch size >= 2 for BatchNorm1d.")
        physiological_signals = self.act(self.PwConv1(physiological_signals))  # [B, 16, L]; default L=600

        # Feed to backbone
        physiological_signals = self.Backbone_layer1(physiological_signals)
        physiological_signals = self.pool1d(physiological_signals)
        physiological_signals = self.Backbone_layer2(physiological_signals)
        physiological_signals = self.Backbone_layer3(physiological_signals)
        physiological_signals = self.Backbone_layer4(physiological_signals)
        physiological_signals = self.pool1d(physiological_signals)
        physiological_signals = self.Backbone_layer5(physiological_signals)
        physiological_signals = self.Backbone_layer6(physiological_signals)
        physiological_signals = self.pool1d(physiological_signals)
        physiological_signals = self.Backbone_layer7(physiological_signals)
        physiological_signals = self.pool1d(physiological_signals)  # [B, 128, floor(L/16)]; default 37

        # Feed to subnetworks
        SBP_logit = self.Subnetwork1(physiological_signals)
        DBP_logit = self.Subnetwork2(physiological_signals)

        SBP = self.SBP_MIN + (self.SBP_MAX - self.SBP_MIN) / (1 + torch.exp(-SBP_logit * self.TEMPERATURE))
        DBP = self.DBP_MIN + (self.DBP_MAX - self.DBP_MIN) / (1 + torch.exp(-DBP_logit * self.TEMPERATURE))

        return SBP, DBP


'''
Build Multiscale Fusion block
'''
class MSF_block(nn.Module):
    def __init__(self, depth_in, depth_inter, depth_out):
        super(MSF_block, self).__init__()
        self.PwConv1 = nn.Sequential(nn.Conv1d(depth_in, depth_inter, 1, stride=1, padding=0), nn.BatchNorm1d(depth_inter))
        self.PwConv2 = nn.Sequential(nn.Conv1d(depth_inter, depth_out, 1, stride=1, padding=0), nn.BatchNorm1d(depth_out))
        self.PwConv3 = nn.Sequential(nn.Conv1d(depth_in, depth_out, 1, stride=1, padding=0), nn.BatchNorm1d(depth_out))

        self.DwConv3 = nn.Sequential(
            nn.Conv1d(depth_inter, depth_inter, 3, stride=1, padding=2, dilation=2, groups=depth_inter),
            nn.BatchNorm1d(depth_inter),
            nn.Hardswish()
        )
        self.DwConv5 = nn.Sequential(
            nn.Conv1d(depth_inter, depth_inter, 5, stride=1, padding=2, groups=depth_inter),
            nn.BatchNorm1d(depth_inter),
            nn.Hardswish()
        )

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.W = nn.Sequential(nn.Linear(depth_inter, depth_inter//4), nn.BatchNorm1d(depth_inter//4), nn.ReLU())
        self.A = nn.Sequential(nn.Linear(depth_inter // 4, depth_inter), nn.BatchNorm1d(depth_inter), nn.ReLU())
        self.B = nn.Sequential(nn.Linear(depth_inter // 4, depth_inter), nn.BatchNorm1d(depth_inter), nn.ReLU())

        self.PwConv1.apply(weight_init_kaiming_uniform)
        self.PwConv2.apply(weight_init_kaiming_uniform)
        self.PwConv3.apply(weight_init_kaiming_uniform)
        self.DwConv3.apply(weight_init_kaiming_uniform)
        self.DwConv5.apply(weight_init_kaiming_uniform)
        self.W.apply(weight_init_kaiming_uniform)
        self.A.apply(weight_init_kaiming_uniform)
        self.B.apply(weight_init_kaiming_uniform)

    def forward(self, embedded_feature):
        feature = self.relu(self.PwConv1(embedded_feature))
        scaled_feature3 = self.DwConv3(feature)
        scaled_feature5 = self.DwConv5(feature)

        attention = torch.add(scaled_feature3, scaled_feature5)
        attention = self.GAP(attention).view(embedded_feature.shape[0], -1)
        attention = self.W(attention)
        attention = self.softmax(torch.stack((self.A(attention), self.B(attention)), dim=-1))

        scaled_feature3 = torch.mul(scaled_feature3, torch.unsqueeze(attention[:, :, 0], dim=-1))
        scaled_feature5 = torch.mul(scaled_feature5, torch.unsqueeze(attention[:, :, 1], dim=-1))

        out = torch.add(scaled_feature3, scaled_feature5)
        out = self.PwConv2(out)
        embedded_feature = self.PwConv3(embedded_feature)
        out = torch.add(out, embedded_feature)

        return out


'''
Build BAM & regression layers for subnetworks
'''
class BAM_regressor(nn.Module):
    def __init__(self, depth):
        super(BAM_regressor, self).__init__()
        self.CA = nn.Sequential(
            nn.Linear(depth, depth // 4, bias=True),
            nn.ReLU(),
            nn.Linear(depth // 4, depth, bias=True),
            nn.BatchNorm1d(depth)
        )
        self.TA = nn.Sequential(
            nn.Conv1d(depth, depth // 4, 1, stride=1, padding=0),
            nn.Conv1d(depth // 4, depth // 4, 3, stride=1, padding=4, dilation=4),
            nn.Conv1d(depth // 4, depth // 4, 3, stride=1, padding=4, dilation=4),
            nn.Conv1d(depth // 4, 1, 1, stride=1, padding=0),
            nn.BatchNorm1d(1)
        )
        self.regressor1 = nn.Sequential(
            nn.Conv1d(depth, depth*6, 1, stride=1, padding=0),
            nn.Hardswish(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.regressor2 = nn.Sequential(
            nn.Conv1d(depth * 6, depth, 1, stride=1, padding=0),
            nn.Hardswish(),
            nn.Conv1d(depth, 1, 1, stride=1, padding=0)
        )

        self.GAP = nn.AdaptiveAvgPool1d(1)

        self.CA.apply(weight_init_kaiming_uniform)
        self.TA.apply(weight_init_kaiming_uniform)
        self.regressor1.apply(weight_init_kaiming_uniform)
        self.regressor2.apply(weight_init_kaiming_uniform)

    def forward(self, embedded_feature):
        channel_attention = torch.unsqueeze(self.CA(self.GAP(embedded_feature).view(embedded_feature.shape[0], -1)), dim=-1)
        temporal_attention = self.TA(embedded_feature)
        attention = torch.mul(channel_attention, temporal_attention)

        feature = torch.mul(attention, embedded_feature)
        feature = torch.add(feature, embedded_feature)

        feature = self.regressor1(feature)
        BP = self.regressor2(feature).view(embedded_feature.shape[0])

        return BP
