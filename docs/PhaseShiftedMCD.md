# Phase-Shifted rPPG：MCD 数据接入与远程运行

2026-10-05：新增两个独立 Trainer 与阶段 YAML。先观察 DRPNet 结果再手动启动 BBPNet，
请使用 [分阶段运行说明](PhaseShiftedStages.md)。本文中 FULL 命令保留为兼容的自动两阶段入口。

更新：2026-09-29。已实现代码适配，未训练、未报告真实数据精度。
这是 **PPG + 袖带 BP 监督的改编实验**，不等同于原论文连续 ABP 监督的复现。

## 数据与模型契约

| 来源 | 含义与用途 |
|---|---|
| `video/*.avi` | 面部 RGB 视频；默认只用 FullHDwebcam |
| `meta/*.txt` | 连续帧编号及绝对采集时间；建立视频实际时间轴 |
| `ppg/*.PW` | 原始接触式光学 PPG 幅值及时间戳，用于核对同步文件 |
| `ppg_sync/*.txt` | 第 1 列：匹配到视频时刻的 PPG；第 2 列：匹配时间绝对差，单位秒，不是时间轴 |
| `db.csv` | `upper_ap` / `lower_ap` 为受试者某状态的 SBP / DBP（mmHg）；不是逐帧压力 |
| `ecg/*.json` | ECG；本次训练不使用 |

加载器从 db.csv 读取相对路径，不根据文件名猜标签。缺失文件记入 manifest；重复记录、
越界路径、同步数值不一致直接报错。真实采集设备和授权仍需随数据来源记录，
软件时间戳匹配不能单独证明硬件同步。没有连续 ABP，也不会由 PPG 合成 ABP 标签。

输入为 Raw RGB `[B,3,150,128,128]`，25 Hz，每窗口 6 秒；Trainer 显式除以 255 并转为 BGR。
DRPNet 输出 facial / acral 两个 `[B,150]` 波形，幅值为模型任意单位，不能解释成 mmHg。
其 acral 分支也是从面部视频预测的模型分支，并不表示新增了真实手指视频或标签。
波形重采样为 `[B,600]` 输入 BBPNet，输出 `[B]` SBP、DBP；HR 从预测波形计算，单位 bpm。
**最终没有可验证的连续血压波形输出。**

- Stage 1：保留原波形、频域、HR 等损失结构，真实接触式 PPG 作为参考。
- Stage 2：冻结 DRPNet，仅用真实 SBP/DBP 的 Huber loss 更新 BBPNet；完全跳过 ABP 重建损失。
- 不添加常数 epsilon 代替缺失监督；常数没有梯度，正确预测时损失为零也是正常现象。
- 原 `ABP_MMHG` 分支保留；checkpoint 包含标签来源和预处理签名，禁止与 `PPG_CUFF` 混用。
- 输出区间预设为 SBP 60–240、DBP 30–140 mmHg，越界标签报错，不裁剪。

## 预处理、排除和评价

严格 FFmpeg 解码到首个错误即停止，不跨越损坏帧。仅在有参考帧数时容许最多 32 帧的
末尾缺失，并记录 stderr、返回码及帧数；无时间戳的视频推理遇到解码错误直接失败。
Haar 最大人脸检测每 30 个原始帧更新一次，扩大 1.3 倍后缩放到 128×128；检测失败的
时间区间保留位置并标无效，涉及该区间的整个窗口被排除，不把坏帧删除后拼接。
这些裁脸参数在 v1 中固定；通用 Toolbox CROP_FACE 参数不控制此专用加载器。

按 meta 时间重建均匀 25 Hz 网格，视频取最近帧，PPG 按同步时刻线性插值，
保持实际时长。未额外加入抗混叠滤波；Stage 1 沿用原信号处理。
同步差超过 20 ms、帧间隙超过 100 ms、人脸检测无效、平坦 PPG 的窗口排除。
不足 150 点的末尾不补齐。上述阈值是预处理策略，不代表生理信号质量得到临床验证。

在所选相机的本地完整记录集合上，按唯一受试者 ID 排序后用 seed=100 打乱，
70%/15%/15% 划分。before/after 及所有相机保持同集合；Trainer 再检查集合交叉。
三个 split 的相机、种子和质量阈值必须相同。变更下载子集会改变划分，因此应固定原始数据
清单并保存三个 manifest，不能在实验中途增删数据。

SBP/DBP 按“受试者 + 状态”赋予逆窗口数量权重，减少长录像和多视角的重复标签权重。
验证模型选择及主要 BP 指标先在状态内平均预测，再计算 MAE/RMSE/ME/SD；窗口指标另外输出。
训练 batch 至少 2，drop_last=True 避免 BBP BatchNorm 的单样本末批；验证/推理 batch=1，
减少原实现跨 batch 标准化带来的依赖。每轮最后不足一个 batch 的随机样本会被舍弃。

测试 HR 使用线性去趋势、0.5–3 Hz Butterworth SOS 零相位滤波及 Hamming FFT 峰值。
这是报告方法，与 Stage 1 的原可微滤波方法不同；比较其他方法时应统一评价协议。
`hr_valid` 只检查有限值/非平坦信号，`bp_order_valid` 只检查 SBP > DBP，不证明预测准确。

## 服务器执行顺序

把本次修改后的整个 Toolbox 同步至远程服务器，使用已配好的项目 Python 环境。
需要兼容的 PyTorch/torchaudio、NumPy、SciPy、OpenCV、yacs、PyYAML、Toolbox 其他依赖和
PATH 中的 FFmpeg；本次没有修改环境或依赖版本。以下命令中的 `/remote/...` 均需手动替换。

```bash
cd /remote/rPPG-Toolbox
python tools/mcd_audit.py --data-root /remote/mcd_rppg
python tools/mcd_demo_check.py --data-root /remote/mcd_rppg --forward --device cuda:0
python tools/phase_shifted_smoke_test.py --stage1-loss
python tests/test_phase_shifted_rppg.py -v
python tests/test_mcd_phase_shifted.py -v
```

`mcd_demo_check` 默认读 `1020_before_FullHDwebcam`；可用 `--recording-id` 指定其他记录。
其 `--forward` 使用随机权重，只检查样例预处理与网络形状，不输出可用于研究的预测结果。
`mcd_audit` 不解码视频；正式服务器流程仍需单独通过 demo 检查。

在 `configs/train_configs/MCD_MCD_MCD_PHASE_SHIFTED.yaml` 中修改：

1. TRAIN/VALID/TEST 的 DATA_PATH：同一个原始数据根目录，含 db.csv 和对应子目录。
2. 三个 CACHED_PATH：原始数据以外的缓存目录；LOG.PATH：实验输出目录。
3. DEVICE 及需要的实验参数；使用多视角或调整质量阈值时同步修改三个 DATA.MCD 节点。
4. 确认多受试者数据已获准使用，审查清单、容量及受试者划分后再训练。

```bash
python main.py --config_file configs/train_configs/MCD_MCD_MCD_PHASE_SHIFTED.yaml
```

FULL 自动先训练 Stage 1、载入验证集最优 DRP，再冻结它训练 Stage 2，最后测试最佳组合。
默认 50 / 200 个 epoch 是配置值，尚未经本项目实验验证。只有一个受试者的 demo 会因空集合
而停止，不能通过把同人窗口分散进三个集合绕过限制。

首次 DO_PREPROCESS=True 生成缓存；之后可设 False。缓存签名包含源路径/大小/mtime、
BP 标签、质量参数和版本，改变数据或跨主机移动缓存可能要求重新生成，不能混用旧缓存。
manifest 记录受试者、状态、保留窗口和排除原因；原始文件不会被覆盖。
每个 uint8 视频窗口约 7.4 MB，预处理逐录像进行，但缓存全量前仍应核对服务器容量。

最佳组合权重默认位于：
`LOG.PATH/MCD_PPG_CUFF_v1/PreTrainedModels/MCD_PhaseShiftedRPPG_FULL_Best.pth`。
测试波形、HR、BP、标签、时刻保存在 `saved_test_outputs/*_outputs.pth`；
状态内预测保存在 `*_state_predictions.json`，指标另存 `*_metrics.json`。

## 仅输入视频的推理

训练完成后使用 FULL 权重，无需 db.csv、PPG、ECG 或 BP 标签：

```bash
python tools/phase_shifted_predict_video.py \
  --video /remote/mcd_rppg/video/1020_FullHDwebcam_before.avi \
  --timestamps /remote/mcd_rppg/meta/1020_FullHDwebcam_before.txt \
  --checkpoint /remote/runs/MCD_PPG_CUFF_v1/PreTrainedModels/MCD_PhaseShiftedRPPG_FULL_Best.pth \
  --output-dir /remote/predictions/run1 --device cuda:0
```

外部正常视频可省略 `--timestamps`，使用容器 FPS 或显式 `--fps`，此时假定恒定帧率；
不能声称具有 MCD 相同的时间精度。示例 MCD 视频存在损坏尾部，建议保留时间戳参数。
输出 NPZ 包含各窗口 `time`、两路 rPPG、两路 HR、SBP/DBP 和数值有效标志，JSON 含单位、
时间依据、解码审计和整段平均 BP。脚本拒绝覆盖同名 NPZ。不能用随机权重代替训练权重。

## 本地验证与未完成项

2026-09-29 本地只读取现有 demo：2 条完整记录、1 名受试者；FullHDwebcam 清单中其余
1198 条未下载，不等于原数据集损坏。同步 PPG before/after 分别 5382/5383 点，
最大时间匹配差 13.963/16.954 ms；对应 BP 105/78、113/78 mmHg。
逐点验证同步幅值与原始 PPG 最近时间样本一致。严格 FFmpeg 解码实测在损坏尾部停止，
before/after 分别得到 5381/5382 帧；处理策略是审计后裁去对应末尾，不补造帧或标签。

19 项单元测试通过，覆盖契约、袖带损失梯度、真实 BBP 冻结/梯度、时间匹配、坏区间排除、
受试者隔离、状态指标、checkpoint 和无标签预测接口。完整网络合成前向、原 Stage 2
反向传播及 PVLoss 梯度检查通过；11 个 Python 文件语法检查与 MCD YAML 解析通过。
本地缺少 OpenCV、yacs、torchaudio，因此未执行真实视频裁脸、完整 main.py、Stage 1 可微
滤波反向传播及真实训练/推理。必须在远程环境完成上述检查；尚无训练权重或性能结论。
