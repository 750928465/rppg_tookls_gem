"""Adapted from GyutaeHwang/phase_shifted_rPPG.
Upstream commit: 7d46167a8a22acb37a03e1410e8fff86482c5f1b.
Copyright (c) 2024 Gyutae Hwang. MIT license: see LICENSE in this package.
Layer names are preserved to support upstream state_dict checkpoints.
"""
import torch.nn as nn
import torch


class DRPNet(nn.Module):
    def __init__(self):
        super(DRPNet, self).__init__()

        self.Feature_extracter = nn.Sequential(
            nn.Conv3d(3, 16, [3,5,5],stride=1, padding=(4,2,2), dilation=(4,1,1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=(4,1,1), dilation=(4,1,1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=(4,1,1), dilation=(4,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        self.DilatedConv = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=(4,0,0), dilation=(4,4,4)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=(4,0,0), dilation=(4,4,4)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=(4,0,0), dilation=(4,4,4)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, [3, 2, 2], stride=1, padding=(4,0,0), dilation=(4,4,4)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.Attention_temporal = nn.Sequential(
            nn.Conv3d(64, 32, [1, 1, 1], stride=1, padding=0),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool3d((1, 4, 4), stride=(1, 4, 4)),

            nn.Conv3d(32, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool3d((1, 4, 4), stride=(1, 4, 4)),

            nn.Conv3d(32, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(32, 1, [1, 1, 1], stride=1, padding=0),

            nn.Sigmoid()
        )

        self.Attention_spatial = nn.Sequential(
            nn.Conv3d(64, 32, [1, 1, 1], stride=1, padding=0),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool3d((4, 2, 2), stride=(4, 2, 2)),

            nn.Conv3d(32, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool3d((4, 2, 2), stride=(4, 2, 2)),

            nn.Conv3d(32, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool3d((4, 2, 2), stride=(4, 2, 2)),

            nn.Conv3d(32, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.AvgPool3d((2, 1, 1), stride=(2, 1, 1)),

            nn.Conv3d(32, 1, [1, 1, 1], stride=1, padding=0),

            nn.Sigmoid()
        )

        self.finger_rPPG_extraction = nn.Sequential(
            nn.AvgPool3d((1, 4, 4), stride=(1, 4, 4)),

            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(4, 0, 0), dilation=(4,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(4, 0, 0), dilation=(4,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(4, 0, 0), dilation=(4,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, [1, 1, 1], stride=1, padding=0),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 32, [1, 1, 1], stride=1, padding=0),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 1, [1, 1, 1],stride=1, padding=0),
        )

        self.face_rPPG_extraction = nn.Sequential(
            nn.AvgPool3d((1, 4, 4), stride=(1, 4, 4)),

            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(4, 0, 0), dilation=(4,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(4, 0, 0), dilation=(4,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(4, 0, 0), dilation=(4,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, [1, 1, 1], stride=1, padding=0),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 32, [1, 1, 1], stride=1, padding=0),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 1, [1, 1, 1],stride=1, padding=0),
        )

    def forward(self, x1):
        # Toolbox must supply NCDHW, identical to upstream [B, 3, T, H, W].
        # No implicit axis permutation or adaptive pooling: preserve attention.
        if x1.ndim != 5 or x1.shape[1] != 3:
            raise ValueError("DRPNet expects [B, 3, T, 128, 128] (NCDHW).")
        if x1.shape[-2:] != (128, 128) or not 128 <= x1.shape[2] < 256:
            raise ValueError("Original DRPNet requires 128x128 and 128 <= T < 256; default T=150.")

        # T is preserved; spatial sizes: 128 -> 64 -> 32.
        x2 = self.Feature_extracter(x1)
        # Dilated spatial convolutions: 32 -> 24 -> 16 -> 8 -> 4.
        x3 = self.DilatedConv(x2)

        At = self.Attention_temporal(x2)  # [B, 1, T, 1, 1]
        x3_At = torch.mul(x3, At)

        As = self.Attention_spatial(x2)  # [B, 1, 1, 4, 4]
        x3_Ats = torch.mul(x3_At, As)

        face_rPPG = self.face_rPPG_extraction(x3_Ats)
        face_rPPG = face_rPPG.reshape(face_rPPG.shape[0], face_rPPG.shape[2])  # [B, T]

        finger_rPPG = self.finger_rPPG_extraction(x3_Ats)
        finger_rPPG = finger_rPPG.reshape(finger_rPPG.shape[0], finger_rPPG.shape[2])

        return face_rPPG, finger_rPPG
