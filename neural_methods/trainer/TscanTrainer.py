"""rPPG-Toolbox 中单任务 TS-CAN 的训练与测试流程。

与 NeurIPS 2020 论文实验设置的关键区别：论文的 MTTS-CAN 使用 TensorFlow、Adadelta
和脉搏/呼吸多任务绝对误差；本 Trainer 实例化单任务 PyTorch TSCAN，使用 MSE、
AdamW 和 OneCycleLR。这里复现的是 Toolbox 当前实现，不能把运行结果称为论文原始
训练设置的逐项复现。
"""

import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.TS_CAN import TSCAN
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm


class TscanTrainer(BaseTrainer):
    """连接数据、TS-CAN、优化器、checkpoint 与评价模块。"""

    def __init__(self, config, data_loader):
        """根据 YAML 构建训练或仅测试所需的单任务 TS-CAN。"""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        # FRAME_DEPTH 对应 TSM 的时间窗口 N；每个 TSM 都按该长度恢复时间维。
        self.frame_depth = config.MODEL.TSCAN.FRAME_DEPTH
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        # DataParallel 会沿第一维分配数据。后面把帧数裁成 GPU数*时间窗口 的整数倍，
        # 确保每张 GPU 上的 TSM 输入仍能按 frame_depth 重排。
        self.base_len = self.num_of_gpu * self.frame_depth
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config 
        self.min_valid_loss = None
        self.best_epoch = 0

        if config.TOOLBOX_MODE == "train_and_test":
            # 训练模式使用 TRAIN 预处理尺寸构造模型。
            self.model = TSCAN(frame_depth=self.frame_depth, img_size=config.TRAIN.DATA.PREPROCESS.RESIZE.H).to(self.device)
            # 单卡运行：沿用 YAML 的 DEVICE 编号，NUM_OF_GPU_TRAIN 应设为 1。
            self.model = torch.nn.DataParallel(
                self.model, device_ids=[self.device.index], output_device=self.device.index)

            self.num_train_batches = len(data_loader["train"])
            # Toolbox 差异：这里是单任务逐帧 MSE；论文 Eq. (8) 是 BVP+呼吸的多任务 L1。
            self.criterion = torch.nn.MSELoss()
            # Toolbox 差异：当前实现使用 AdamW；论文实验使用 Adadelta(lr=1.0)。
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            # 仅测试模式使用 TEST 预处理尺寸，随后在 test() 中加载 MODEL_PATH。
            self.model = TSCAN(frame_depth=self.frame_depth, img_size=config.TEST.DATA.PREPROCESS.RESIZE.H).to(self.device)
            # 仅测试时同样使用 YAML 的 DEVICE，保持模型和输入位于同一张 GPU。
            self.model = torch.nn.DataParallel(
                self.model, device_ids=[self.device.index], output_device=self.device.index)
        else:
            raise ValueError("TS-CAN trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """逐 epoch 训练，并按配置决定使用最后一轮还是验证集最优 checkpoint。"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                # Loader 输出 data=[N,D,C,H,W]、label=[N,D]：
                # N 是 clip batch，D 是每个 clip 的帧数，C 通常为6个双分支输入通道。
                data, labels = batch[0].to(
                    self.device), batch[1].to(self.device)
                N, D, C, H, W = data.shape
                # TS-CAN 的 Conv2d 逐帧工作，所以先把 batch 与时间维展平。
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)
                # 丢弃末尾不足一个完整 TSM 窗口的帧，避免 TSM.view 形状错误。
                data = data[:(N * D) // self.base_len * self.base_len]
                labels = labels[:(N * D) // self.base_len * self.base_len]
                self.optimizer.zero_grad()
                # 每帧预测一个 rPPG/BVP 标量，与同一时刻的接触式 PPG 标签计算 MSE。
                pred_ppg = self.model(data)
                loss = self.criterion(pred_ppg, labels)
                # 标准反向传播：清梯度 -> 前向 -> 损失 -> 反向 -> 更新参数与学习率。
                loss.backward()

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                tbar.set_postfix(loss=loss.item())

            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(train_loss))

            # 每个 epoch 都保存，便于随后选最后一轮或验证损失最低的一轮。
            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                # 注意：是否运行验证由 TEST.USE_LAST_EPOCH 控制，这是 Toolbox 的现有接口设计。
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """关闭梯度，在验证集上计算与训练阶段一致的逐帧 MSE。"""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(
                    self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                # 验证阶段保持与训练完全相同的展平和完整时间窗口裁剪规则。
                data_valid = data_valid.view(N * D, C, H, W)
                labels_valid = labels_valid.view(-1, 1)
                data_valid = data_valid[:(N * D) // self.base_len * self.base_len]
                labels_valid = labels_valid[:(N * D) // self.base_len * self.base_len]
                pred_ppg_valid = self.model(data_valid)
                loss = self.criterion(pred_ppg_valid, labels_valid)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """加载 checkpoint，预测测试波形，按视频重组并计算 HR/信号指标。"""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing===")

        # 测试集的 clip 长度可能不同于训练集，重组预测时必须使用 TEST 的长度。
        self.chunk_len = self.config.TEST.DATA.PREPROCESS.CHUNK_LENGTH

        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            # 预训练推理：严格从 YAML 的 INFERENCE.MODEL_PATH 读取权重。
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                # 选择最后一个 epoch；这种做法不使用验证集进行模型选择。
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                # 选择验证 MSE 最低的 epoch，测试集本身不参与模型选择。
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                data_test = data_test[:(N * D) // self.base_len * self.base_len]
                labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
                # 输出仍是逐帧波形值，不是直接输出 BPM。
                pred_ppg_test = self.model(data_test)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    labels_test = labels_test.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()

                for idx in range(batch_size):
                    # Loader 同时返回视频/受试者标识和 clip 序号；以二级字典保存，
                    # evaluation.metrics 会按序号拼回连续波形后再做 FFT 或峰值检测。
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        print('')
        # 将预测波形与 GT PPG 后处理为 HR，并计算 YAML 指定的 MAE、RMSE、Pearson、SNR等。
        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
