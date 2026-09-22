import neural_methods.trainer.BaseTrainer
import neural_methods.trainer.PhysnetTrainer
import neural_methods.trainer.iBVPNetTrainer
import neural_methods.trainer.TscanTrainer
import neural_methods.trainer.DeepPhysTrainer
import neural_methods.trainer.EfficientPhysTrainer
import neural_methods.trainer.BigSmallTrainer
import neural_methods.trainer.PhysFormerTrainer
# 当前只运行 TS-CAN，暂不加载需要 mamba_ssm 的 PhysMamba。
# 如需运行 PhysMamba，请先准备对应依赖并恢复下面的导入。
# import neural_methods.trainer.PhysMambaTrainer
import neural_methods.trainer.RhythmFormerTrainer
import neural_methods.trainer.FactorizePhysTrainer
