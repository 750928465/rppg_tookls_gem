# Phase-Shifted rPPG 源码接入与 Review

2026-09-29 更新：新增 MCD 的 `PPG_CUFF` 适配、专用 Loader、袖带监督及无标签视频推理。
当前使用方法和验证边界见 [PhaseShiftedMCD.md](PhaseShiftedMCD.md)。以下记录原始 ABP 分支
在 2026-09-24 的接入，不应把其中“尚无适配器”的历史状态用于判断新增 MCD 分支。

日期：2026-09-24。范围：模型、loss、统一 Trainer、配置、main.py 注册和最小检查；未运行真实数据训练。

## 来源与证据范围

- 作者仓库：[GyutaeHwang/phase_shifted_rPPG](https://github.com/GyutaeHwang/phase_shifted_rPPG)。
- 本次固定源码：[7d46167a8a22acb37a03e1410e8fff86482c5f1b](https://github.com/GyutaeHwang/phase_shifted_rPPG/tree/7d46167a8a22acb37a03e1410e8fff86482c5f1b)。
- 论文入口：[arXiv:2401.04560](https://arxiv.org/abs/2401.04560)。本轮是固定版本源码审查；论文全文与实验协议逐项复核尚未完成，下面的默认值与行为均以该源码为依据。
- 许可证：MIT，Copyright (c) 2024 Gyutae Hwang。全文保留于 `neural_methods/model/phase_shifted_rppg/LICENSE`。
- 接入前本地 Toolbox HEAD：`bae390b29a4b7351801f7a28e0b4771fa4c116a2`，与旧交接快照不同。已有 PhysNet YAML 修改、log/ 和 .DS_Store 未改动。

## A. 原项目文件与迁移位置

| 原文件 | 职责 | 本次处理 |
|---|---|---|
| Shifted_rPPG_extraction.py | DRP-Net | model/phase_shifted_rppg/DRPNet.py，保留所有层名 |
| BP_estimation.py | BBP-Net、MSF、BAM | 同 package 的 BBPNet.py；组件仅此网络使用，不额外拆 blocks.py |
| PV_MSELoss.py | 峰谷振幅 loss | loss/PhaseShiftedLoss.py |
| utils.py | 滤波、去趋势、FFT、统计、作图 | 仅迁入 loss 所需数学操作，不整体复制 |
| main_stage1.py / main_stage2.py | 训练、验证、保存、评价 | trainer/PhaseShiftedRPPGTrainer.py |
| Dataloader.py | 三元组、BP 峰谷标签、FILM/HRDA | 只审查；不迁移 |
| Preprocessing.py | 裁脸、128×128、BP 重采样 | 只审查；不迁移 |
| interpolator.py | FILM 封装 | 不迁移、不引入 TensorFlow |
| requirements.txt | 旧实验环境 | 不复制、不合并、不安装 |
| V4V/ | 原仓库另一个实验目录 | 本次不迁移；当前实现对应根目录 MMSE-HR 脚本 |

## B. 实际 Pipeline 与信号含义

```text
Toolbox Raw video [B,3,150,128,128] (NCDHW)
  -> /PIXEL_SCALE；RGB 输入显式转为原源码使用的 BGR
  -> DRPNet
  -> facial waveform [B,150] + acral waveform [B,150]
  -> 冻结 DRPNet，SciPy FFT resample 150 -> 600
  -> 各分支在整个 batch 上标准化（保持原行为）
  -> 每分支 PPG / 一阶差分 VPG / 二阶差分 APG
  -> [B,6,600] -> BBPNet MSF -> 两个 BAM regressor
  -> scaled sigmoid -> SBP [B] / DBP [B]，单位 mmHg
```

Facial 与 acral 均是网络从**同一段面部视频推断**的脉动信号，并非输入中直接拍摄到的两个身体部位；acral 分支也不是实际指端传感器读数。外周与面部不同血管路径可能产生脉动时序差异，但模型输出的相位差不能直接当成已校准的 PTT 或因果血压测量。生理机理与同步误差需要后续专门验证。

VPG/APG 在这里严格指原实现的离散一阶/二阶差分，不乘采样率；保留尾部补零边界。它们描述上升、下降及曲率变化，供 BP 回归学习使用，不等于从导数公式就能直接确定血压。

原代码把参考波形命名为 `ref_ppg`、`t_y`，Stage 2 又命名为 `ABP`；实际 `Preprocessing.py` 读取 `BP_mmHg` 文件，按 1000 Hz -> 25 Hz 重采样。这里称为**连续血压参考波形**，代码变量名不能证明它来自有创动脉导管。参考设备、数据授权与同步仍待数据方核实。

## C. 依赖关系

```text
DRPNet / BBPNet -> torch（无 torchinfo 必需依赖）
PVLoss / Stage2Loss -> torch
Stage1Loss -> torch + numpy + scipy + torchaudio.functional.filtfilt
Trainer -> BaseTrainer + 两个模型 + losses + scipy.signal.resample
main.py -> Toolbox 原有数据/训练/评价依赖 + 此 Trainer
```

`torchaudio` 是原 Stage 1 可微滤波的核心依赖，不属于可直接删除的 augmentation 依赖。保留 `clamp=False`；不替换为 SciPy filtfilt 或 torch interpolate。TensorFlow/FILM、PyTorch3D、numba.cuda reset 与模型接入无关，未引入。

## D. Tensor Shape

DRPNet 默认 T=150，卷积不改变 T：

| 关键操作 | 输出 |
|---|---|
| 输入 | [B,3,150,128,128] |
| Feature Conv 3->16 | [B,16,150,128,128] |
| spatial max-pool 2 | [B,16,150,64,64] |
| Conv 16->32->64 | [B,64,150,64,64] |
| spatial max-pool 2 | [B,64,150,32,32] |
| DilatedConv 1/2/3/4 | [B,64,150,24,24] -> [B,64,150,16,16] -> [B,64,150,8,8] -> [B,64,150,4,4] |
| Temporal attention 从 32×32 特征出发 | [B,32,150,32,32] -> pool4: 8×8 -> pool4: 2×2 -> avg2: 1×1 -> [B,1,150,1,1] |
| Spatial attention 从 32×32 特征出发 | pool(4,2,2): [B,32,37,16,16] -> [B,32,9,8,8] -> [B,32,2,4,4] -> avg(2,1,1) -> [B,1,1,4,4] |
| 广播乘两个 attention | [B,64,150,4,4] |
| 各输出分支 avg-pool 4×4 | [B,64,150,1,1] |
| 各分支 Conv 64->64->64->64->32->1 | [B,1,150,1,1] |
| reshape | 两个 [B,150] |

第一版固定 H=W=128，允许 128<=T<256；T=150 是原默认。其他空间尺寸不通过静默 adaptive pooling 迁就。NCDHW 与原模型一致，Trainer 不做时间/通道轴转换。

BBPNet 默认 resample 后 L=600：

| 关键操作 | 输出 |
|---|---|
| facial / acral PPG | 各 [B,600] |
| 一阶/二阶差分，尾部补 1/2 个零 | 各 [B,600] |
| stack 顺序 face PPG,VPG,APG,acral PPG,VPG,APG | [B,6,600] |
| pointwise Conv | [B,16,600] |
| MSF1 (inter=64) / pool | [B,32,600] -> [B,32,300] |
| MSF2/3/4 (inter=64) / pool | [B,32,300] -> [B,32,150] |
| MSF5/6 (inter=128) / pool | [B,64,150] -> [B,64,75] |
| MSF7 (inter=512) / pool | [B,128,75] -> [B,128,37] |
| 每个 MSF 内 3/5 kernel 深度卷积分支 | [B,inter,L]；GAP 后 [B,inter] -> inter/4 -> 双分支 softmax [B,inter,2] |
| 每个 BAM 通道/时间注意力 | [B,128,1] 与 [B,1,37]，相乘并残差作用于 [B,128,37] |
| regressor1 / GAP / regressor2 | [B,768,37] -> [B,768,1] -> [B,128,1] -> [B,1,1] |
| reshape + scaled sigmoid | SBP/DBP 各 [B] |

原注释的 `[16,200]`、`[128,12]` 与实际 Stage 2 输入不一致，此处以实际 600 点数据流为准。

## Loss 保留情况

Stage 1：先按整个 batch 标准化、lam=50 去趋势、0.5–3 Hz 三阶 Butterworth 双向滤波。FFT 用 periodic Hamming window，n_fft=T×40，取频带振幅与 argmax HR。

- Facial：`0.0001 * HR_MSE + 100 * spectrum_MSE + PV_MSE`。
- Acral：上述三项再加 `filtered_waveform_MSE`。
- 两分支**求和**反传，与原先先后两次 backward 等价；没有额外除以 2。
- argmax HR 不对预测波形提供梯度，保留此事实，不伪装成可微 HR loss。
- PV 保留严格符号变化、峰值高于峰均值、谷值低于谷均值的筛选；无峰/谷明确报错。
- 频带长度动态决定，不保留原硬编码 600 bins；float 频率边界可能改变 bin 数。

Stage 2：`Huber(SBP) + Huber(DBP) + MSE(reconstructed_pressure, continuous_pressure)`，默认权重全 1、Huber delta=1。把 acral 每窗 min-max 波形映射到 `[pred_DBP,pred_SBP]` 计算波形项。目标压力保留 mmHg，不标准化、不裁剪真实标签。

## E/F. Toolbox 对接与数据阻碍

直接复用：`BaseTrainer` 方法风格、Toolbox train/valid/test loader、配置合并、MODEL.MODEL_DIR、TEST.OUTPUT_SAVE_DIR、main.py train_and_test/only_test 两个分支。

需要适配：两个独立网络、分阶段 checkpoint、BP 专用 loss/评价、原源码 BGR 与 Toolbox RGB 的差异。BP 输出不能送进原本按 BVP/HR 设计的 `calculate_metrics`。

统一 batch 保留 `(video, labels, recording_id, chunk_id)` 外层结构：

```python
# Stage 1，标签来源必须先被确认是 mmHg 连续压力
labels = abp_tensor                    # [B,T]
# 或 labels = {'abp': abp_tensor}

# Stage 2 / FULL：将来经核实的数据适配器应提供
labels = {'abp': abp_tensor,            # [B,T]，Raw mmHg
          'sbp': sbp_tensor,            # [B]，真实窗口 SBP
          'dbp': dbp_tensor}            # [B]，真实窗口 DBP
```

| 当前数据路径 | Stage 1 原监督 | Stage 2 原监督 |
|---|---|---|
| PURE / UBFC-rPPG / UBFC-Phys 等单一 PPG/BVP 缓存 | 不能宣称按原压力监督复现；本版不自动替代 | 缺真实 BP 波形及 SBP/DBP |
| BP4DPlusLoader | 源码确实读 BP_mmHg；Raw、25 Hz、128×128、150帧、同步和授权核实后有接入基础 | 当前只返回一条 waveform，没有 abp/sbp/dbp 字典 |
| BP4DPlusBigSmallLoader | 多输入视频与本方法不兼容 | 源码有 bp_wave/sys/dia 字段，但双视频与49列标签接口不能直接接入 |

因此不能笼统说“整个 Toolbox 没有 BP 数据”。真实阻碍是**当前选定数据是否具有所需标签，以及 loader 是否保留单位和给出已核实的数据契约**。

原 Dataloader 从未标准化压力的局部极大/极小值取均值生成 SBP/DBP（SciPy argrelextrema, order=10）；测试窗口还应用 SBP 80–160、DBP 40–100 筛选。本轮不复制标签生成/筛选/增强，也不把这些行为隐含加入 Toolbox。以后是否按相同规则构建标签需单独确定。

本次未更改任何 dataset 文件。模板 `LABEL_SOURCE: UNVERIFIED` 会在 main.py 构建 dataset **之前**明确拒绝；不能只改这一开关就认为数据已适配。FULL 在训练前检查 Stage 2 标签接口，缺少标签时不先花时间训练 Stage 1。

## G. 兼容性与有意差异

| 原行为 | 接入行为 | 原因/边界 |
|---|---|---|
| forward 显式 DEVICE，float32 zeros | zeros 继承 waveform device/dtype | 不改变 float32 数学行为，避免设备错配 |
| torchinfo 仅导入、未用于网络运算 | 删除无用导入 | 模型核心只依赖 Torch |
| view 输出 | DRP 输出 reshape | 支持非连续存储；shape 不变 |
| numba CUDA reset / CUDA 假设 | 不迁移；使用 config.DEVICE | 不重置用户设备上下文 |
| DRP eval + detach numpy | eval + requires_grad=False + no_grad | Stage 2 原本无梯度流回 DRP，显式落实冻结 |
| SciPy resample | 保留 150->600 和 CPU roundtrip | 保留频域重采样；有 CPU/GPU 同步成本 |
| 固定 ./record/... / 自动挑首个文件 | 明确路径、阶段最佳 checkpoint、完整 bundle | 避免加载错误模型 |
| torch.load 默认设备、module. 前缀 | map_location + weights_only=True + strict load + 前缀剥离 | 不关闭结构检查；原始 state_dict 无预处理元数据需人工核对 |
| test split 挑 best | valid split 挑 best | 避免测试集参与模型选择；不等同原论文实验协议 |
| batch-wide 标准化 | 保留 | 推理受 batch 组成影响，示例 INFERENCE.BATCH_SIZE=1 |
| BBP BatchNorm1d | 保留，拒绝训练 singleton batch | 不静默丢样本或改网络；需合理 batch size |
| 多卡 DataParallel | 第一版单个显式 DEVICE | NUM_OF_GPU_TRAIN=1；可读旧 DP 权重，多卡训练未实现 |
| 报告窗口指标 | MAE/RMSE/ME/SD + N_windows，保存记录/片段 ID | 单位明确；窗口数不是独立受试者数，无临床有效性结论 |

两个模型 CPU float32 前向已与固定上游源码共享权重逐值比较，结果完全一致。Stage 1 数值滤波兼容性仍需在具有匹配 torchaudio 的环境验证。

## H. 文件、配置、checkpoint

新增：model package（DRPNet.py、BBPNet.py、__init__.py、LICENSE）、loss/PhaseShiftedLoss.py、trainer/PhaseShiftedRPPGTrainer.py、本说明、配置模板、smoke 脚本及契约测试。

修改现有文件仅：`config.py`、`main.py`、`neural_methods/trainer/__init__.py`。已有模型、dataset、requirements、setup.sh 和 PhysNet YAML 均未修改。

- `MODEL.NAME: PhaseShiftedRPPG`。
- `MODEL.PHASE_SHIFTED_RPPG.TRAIN_STAGE: STAGE1 | STAGE2 | FULL`。
- 复用 `*.DATA.FS=25`、`PREPROCESS.CHUNK_LENGTH=150`、128×128、NCDHW。
- Stage 1：`TRAIN.EPOCHS=50`、`TRAIN.LR=0.001`、batch=8。
- Stage 2：`STAGE2_EPOCHS=200`、`STAGE2_LR=0.001`，两阶段指数 LR decay=0.99。
- BP range、温度0.5、重采样×4、滤波/FFT/loss 权重均在同一 config 节点，默认保持源码。
- `STAGE2` 需要 `STAGE1_CHECKPOINT`；FULL 自动用验证集选出的 Stage 1 best。
- 每阶段按 Toolbox MODEL_DIR 保存 epoch 和 Best；最终 `PhaseShiftedRPPG_FULL_Best.pth` 同时含两个 state_dict 与预处理签名。
- `only_test + STAGE1`：INFERENCE.MODEL_PATH 指向 DRP 权重。
- `only_test + STAGE2/FULL`：INFERENCE.MODEL_PATH 指向 FULL bundle；也可同时指定 STAGE1_CHECKPOINT 和单独 BBP 的 INFERENCE.MODEL_PATH。
- 模板划分仅演示配置结构，不能当作论文 KFold 协议；正式实验前须审查受试者不重叠。

## 验证记录与继续入口

2026-09-24 在现有 `/opt/anaconda3/bin/python`（Torch 2.11.0、NumPy 2.3.5、SciPy 1.16.3）运行 CPU smoke：

- DRPNet [1,3,150,128,128] 前向、两个 [1,150] 输出：通过。
- 完整 video -> DRP -> SciPy resample -> PPG/VPG/APG -> BBP 前向：通过。
- SBP/DBP shape、float32、CPU、边界检查：通过。
- 与固定原代码共享同一 state_dict，DRP/BBP 两网络输出逐值一致：通过。
- Stage 2 原 loss 反向、PV loss 反向：通过；没有 optimizer.step，没有训练实验。
- 8 项数据契约、RGB/BGR 转换、权重往返、DataParallel 前缀、冻结与 FULL 生命周期单元测试：全部通过；生命周期测试不执行训练循环。
- 新增/修改 Python 的语法解析、YAML 解析及 git whitespace 检查：通过；YAML 尚未通过真实 yacs 合并运行检查。
- Stage 1 含 torchaudio 滤波的 loss 反向：**未运行**，当前环境缺 torchaudio。
- 完整 main.py 导入：**未通过环境检查**，当前首先缺 yacs；未安装任何依赖，也未绕过它冒充入口成功。

可在依赖齐全的现有环境执行：

```bash
python tools/phase_shifted_smoke_test.py
python tools/phase_shifted_smoke_test.py --stage1-loss
python tests/test_phase_shifted_rppg.py -v
```

契约测试加载真实 BaseTrainer，但隔离了旧 trainer package 对所有模型的 eager imports；该单元测试不证明完整 main.py 的依赖已满足。完整入口仍需在用户实际训练环境重新检查。

后续顺序：确认现有环境依赖 -> 核验真实数据/授权/单位/同步/受试者划分 -> 单独设计 BP 数据适配 -> 真实样例前向与评价 -> 再考虑训练。源码接入和 synthetic smoke 不等于论文复现结果。
