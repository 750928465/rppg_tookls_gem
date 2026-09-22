"""Temporal Shift Convolutional Attention Network (TS-CAN).

论文：Multi-Task Temporal Shift Attention Networks for On-Device Contactless
Vitals Measurement, NeurIPS 2020, Xin Liu et al.

阅读提示（对应论文 Fig. 2 和 Sec. 3.2）：
1. Motion branch 处理相邻视频帧的归一化差分，突出随血容量变化的微弱颜色变化。
2. Appearance branch 提取面部的空间信息，并生成软注意力掩码。
3. TSM 在 Motion branch 中沿时间维移动部分通道，用 2D 卷积的计算量近似获得
   3D 卷积的邻帧信息交换能力。
4. Appearance branch 的注意力掩码与 Motion branch 特征逐元素相乘，抑制背景、
   运动和光照变化等干扰区域。
5. TSCAN 输出单任务 BVP/rPPG 波形；MTTS_CAN 共享骨干并同时输出 BVP 与呼吸波形。

论文与本文件实现并非完全相同：论文实验使用 TensorFlow、36x36 输入，并描述将窗口内
相邻帧平均后送入 Appearance branch；rPPG-Toolbox 的预处理会按配置生成逐帧的
Standardized 外观输入，常用配置为 72x72。阅读或复现实验时应以实际 YAML、
BaseLoader 和 Trainer 为准，不能把论文参数直接当作本实现参数。
"""

import torch
import torch.nn as nn


class Attention_mask(nn.Module):
    """实现论文公式 (7) 中 soft-attention mask 的空间 L1 归一化部分。

    注意：前面的 1x1 卷积和 sigmoid 不在本类中，而是在 TSCAN.forward 中完成。
    这里把每张注意力图的空间和归一化为 H*W/2，使平均注意力权重为 0.5，避免少数
    极端像素支配 Motion branch。该层本身没有可训练参数。
    """

    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        # 输入 x: [batch*时间长度, 1, H, W]，已经经过 1x1 Conv2d 和 sigmoid。
        # 先沿宽、高方向求和，得到每张注意力图的 L1 范数。
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        # 对应论文公式 (7) 的 H_k*W_k/(2*||mask||_1) 缩放。
        return x / xsum * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        """保留自早期实现的配置接口；PyTorch 前向传播不依赖此方法。"""
        config = super(Attention_mask, self).get_config()
        return config


class TSM(nn.Module):
    """Temporal Shift Module：不增加参数的邻帧信息交换模块。

    论文 Sec. 3.2 将通道分为三组：第一组向较早时间位置移动一帧，第二组向较晚
    时间位置移动一帧，第三组保持不变。后续普通 2D 卷积因而可以同时看到相邻帧信息。

    输入第一维已经把 batch 和时间展平为 batch*时间长度，因此 forward 会先恢复
    时间维。第一维必须能被 n_segment 整除，Trainer 会在送入模型前裁掉不足部分。
    """

    def __init__(self, n_segment=10, fold_div=3):
        super(TSM, self).__init__()
        # n_segment 对应论文中的时间窗口 N；Toolbox 配置名为 FRAME_DEPTH。
        self.n_segment = n_segment
        # fold_div=3 表示把通道近似均分为“前移、后移、不移动”三组。
        self.fold_div = fold_div

    def forward(self, x):
        # 输入形状：[batch*n_segment, C, H, W]。
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        # 恢复显式时间维，方便沿时间轴移动通道。
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        # 边界帧没有相邻帧可取，因此使用零填充。
        out = torch.zeros_like(x)
        # 第一组：输出时刻 t 接收输入时刻 t+1，即沿时间轴左移一帧。
        out[:, :-1, :fold] = x[:, 1:, :fold]
        # 第二组：输出时刻 t 接收输入时刻 t-1，即沿时间轴右移一帧。
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        # 第三组：保留当前帧特征，兼顾空间表征与时间表征。
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        # 再次展平时间维，交给普通 Conv2d。
        return out.view(nt, c, h, w)


class TSCAN(nn.Module):
    """论文 TS-CAN 的单任务版本：为每个输入帧预测一个 BVP/rPPG 样本。"""

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, frame_depth=20, img_size=36):
        """构建 TS-CAN 双分支网络。

        Args:
          in_channels: 每种输入表示的通道数；RGB 为 3。总输入实际为 6 通道，前 3
            通道是帧差分，后 3 通道是外观帧。
          frame_depth: TSM 使用的时间窗口长度，即论文中的 N。
          img_size: 裁剪后人脸图像的高/宽。论文使用 36，Toolbox 常用配置使用 72。
        Returns:
          每帧一个标量，连续起来构成预测的 BVP/rPPG 波形。
        """
        super(TSCAN, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        # Motion branch 共四层卷积；每层卷积前放置一个 TSM，使时间信息贯穿网络。
        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)
        # Motion branch：处理相邻帧的归一化差分，学习脉搏引起的细微时空变化。
        # 第1、3层使用 padding 保持分辨率；第2、4层不 padding，使尺寸各缩小2像素。
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv2 = nn.Conv2d(
            self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv4 = nn.Conv2d(
            self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # Appearance branch：处理外观输入，提取皮肤/面部空间特征供注意力模块使用。
        # 拼写“apperance”来自原仓库，为兼容已有 checkpoint 不重命名这些属性。
        self.apperance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv2 = nn.Conv2d(
            self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.apperance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv4 = nn.Conv2d(
            self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # 两个 1x1 卷积把多通道外观特征压缩成单通道空间注意力图；它们位于两次池化前。
        self.apperance_att_conv1 = nn.Conv2d(
            self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(
            self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()
        # 论文 Fig. 2：第2、4个卷积之后做平均池化；Motion 和 Appearance 分支同步降采样。
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        # 前两个分支块使用较小 dropout，最终全连接层前使用较大 dropout。
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)
        # 两次卷积去边界并经过两次 2x2 池化后，不同输入分辨率对应不同展平长度。
        # 例如 72x72 最终得到 64x16x16=16384 个特征。
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif img_size == 72:
            self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
        elif img_size == 128:
            self.final_dense_1 = nn.Linear(57600, self.nb_dense, bias=True)
        else:
            raise Exception('Unsupported image size')
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)

    def forward(self, inputs, params=None):
        """执行论文 Fig. 2 中从双分支输入到单任务波形预测的完整路径。"""

        # BaseLoader 按 DATA_TYPE 配置拼接通道。TS-CAN 配置使用
        # ['DiffNormalized', 'Standardized']，因此前3通道是运动输入，后3通道是外观输入。
        diff_input = inputs[:, :3, :, :]
        raw_input = inputs[:, 3:, :, :]

        # ----- 第一层级：Motion 特征 + Appearance 注意力 -----
        # 先移动时间通道，再用 2D 卷积提取局部时空特征；tanh 与论文/原实现保持一致。
        diff_input = self.TSM_1(diff_input)
        d1 = torch.tanh(self.motion_conv1(diff_input))
        d1 = self.TSM_2(d1)
        d2 = torch.tanh(self.motion_conv2(d1))

        # Appearance branch 只做空间卷积，不使用 TSM。
        r1 = torch.tanh(self.apperance_conv1(raw_input))
        r2 = torch.tanh(self.apperance_conv2(r1))

        # 1x1卷积 + sigmoid 生成未归一化注意力，再按论文公式 (7) 做空间 L1 归一化。
        g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        # 注意力是两个分支的桥梁：仅保留 Motion 特征中更可能包含生理信号的区域。
        gated1 = d2 * g1

        # 两个分支分别池化和正则化，为下一层级准备相同空间尺寸的特征。
        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)

        # ----- 第二层级：重复时间建模、空间注意力和特征门控 -----
        d4 = self.TSM_3(d4)
        d5 = torch.tanh(self.motion_conv3(d4))
        d5 = self.TSM_4(d5)
        d6 = torch.tanh(self.motion_conv4(d5))

        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))

        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        # ----- 波形回归头 -----
        # 池化后展平空间和通道维；时间已合并在 batch 维，因此每行仍对应一个视频帧。
        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)
        # 两层全连接把每帧特征回归为一个标量；按时间顺序排列即预测 BVP/rPPG 波形。
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        out = self.final_dense_2(d11)

        return out


class MTTS_CAN(nn.Module):
    """论文 MTTS-CAN 多任务版本：共享骨干，同时预测 BVP 和呼吸波形。

    当前 rPPG-Toolbox 的 TscanTrainer 使用上面的单任务 TSCAN；本类用于理解论文的
    多任务思想，但没有被该 Trainer 的默认训练/测试路径调用。
    """

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, frame_depth=20):
        super(MTTS_CAN, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        # 以下双分支骨干与 TSCAN 相同：Motion 分支含 TSM，Appearance 分支生成注意力。
        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)
        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv2 = nn.Conv2d(
            self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv4 = nn.Conv2d(
            self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # Apperance branch convs
        self.apperance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv2 = nn.Conv2d(
            self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.apperance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv4 = nn.Conv2d(
            self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(
            self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(
            self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()
        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4_y = nn.Dropout(self.dropout_rate2)
        self.dropout_4_r = nn.Dropout(self.dropout_rate2)

        # 共享特征 d9 之后分成两个任务头：y 表示脉搏/BVP，r 表示呼吸。
        # 16384 对应本实现固定的 72x72 输入；这与论文实验中的 36x36 设置不同。
        self.final_dense_1_y = nn.Linear(16384, self.nb_dense, bias=True)
        self.final_dense_2_y = nn.Linear(self.nb_dense, 1, bias=True)
        self.final_dense_1_r = nn.Linear(16384, self.nb_dense, bias=True)
        self.final_dense_2_r = nn.Linear(self.nb_dense, 1, bias=True)

    def forward(self, inputs, params=None):
        # 与单任务 TSCAN 一样，先拆分 Motion 和 Appearance 两种三通道输入。
        diff_input = inputs[:, :3, :, :]
        raw_input = inputs[:, 3:, :, :]

        # 第一层级 Motion branch：通过 TSM + 2D Conv 交换邻帧信息并提取变化特征。
        diff_input = self.TSM_1(diff_input)
        d1 = torch.tanh(self.motion_conv1(diff_input))
        d1 = self.TSM_2(d1)
        d2 = torch.tanh(self.motion_conv2(d1))

        # 第一层级 Appearance branch：生成面部空间特征。
        r1 = torch.tanh(self.apperance_conv1(raw_input))
        r2 = torch.tanh(self.apperance_conv2(r1))

        # 第一张注意力图对 Motion 特征进行空间门控。
        g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)

        # 第二层级继续进行时间移位、卷积和空间注意力门控。
        d4 = self.TSM_3(d4)
        d5 = torch.tanh(self.motion_conv3(d4))
        d5 = self.TSM_4(d5)
        d6 = torch.tanh(self.motion_conv4(d5))

        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))

        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        # 得到由两个任务共享的时空特征表示。
        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)

        # BVP/pulse 任务头：每帧输出一个脉搏波形样本。
        d10 = torch.tanh(self.final_dense_1_y(d9))
        d11 = self.dropout_4_y(d10)
        out_y = self.final_dense_2_y(d11)

        # Respiration 任务头：从同一共享特征预测呼吸波形样本。
        d10 = torch.tanh(self.final_dense_1_r(d9))
        d11 = self.dropout_4_r(d10)
        out_r = self.final_dense_2_r(d11)

        return out_y, out_r
