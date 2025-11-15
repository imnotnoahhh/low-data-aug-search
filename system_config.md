# 运行环境与关键参数记录

## 系统与驱动
- 操作系统：Ubuntu 24.04 LTS 64 位
- GPU：NVIDIA A10 (24GB)
- CUDA 驱动版本：570.133.20
- CUDA Toolkit 版本：12.8.1（云镜像已预装）
- cuDNN 版本：9.8.0.87（随驱动安装）

## Python 与依赖
- Python 版本：3.14（虚拟环境conda dl）
- PyTorch：`torch` (cu121，对应 CUDA 12.1 轮子)
- TorchVision：`torchvision` (cu121)
- 其他基础库：`numpy`、`pillow`、`tqdm`、`pandas`

> 说明：CUDA 12.8 驱动向下兼容 CUDA 12.1 的 PyTorch 轮子，因此无需额外安装 Toolkit。

## 数据与路径
- CIFAR-100 数据根目录：`data/`
- CIFAR-100-C 路径：`data/CIFAR-100-C/`
- 低样本划分索引：`artifacts/splits/low_data.json`（运行 `prepare_low_data_split` 时生成或加载）

## 训练核心参数（默认）
- Batch size：256（AMP 开启）
- 优化器：SGD (lr=0.2, momentum=0.9, weight_decay=5e-4)
- 学习率调度：CosineAnnealingLR，warmup 5 epoch，`T_max = max_epochs - warmup`
- Mixup：alpha=0.2，默认启用
- 阶段 A ASHA：rungs {10,20,30}，reduction factor 2
- 阶段 B ASHA：rungs {30,60,120}，reduction factor 3
- 阶段 C 贪心：ΔTop-1 ≥ 0.3%、p < 0.05、K_max = 4

