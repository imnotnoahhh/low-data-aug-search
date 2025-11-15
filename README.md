# 运行与上传指南

> 说明：以下命令均假设项目根目录为 `/Users/qinfuyao/dl`，若你在服务器上使用其他路径，请自行替换。

## 1. 环境准备

1. **Python 与虚拟环境**（使用 Conda）
   ```bash
   conda create -n dl python=3.14 -y
   conda activate dl
   ```

2. **依赖安装**（以 CUDA 12.x 为例，若服务器上已有 PyTorch，可跳过重新安装）
   ```bash
   pip3 install torch torchvision
   conda install numpy pillow tqdm pandas
   ```

3. **目录结构**
   ```
   dl/
     ├── plan.md
     ├── README.md
     └── src/
         └── ...
   ```

## 2. 数据准备

1. **CIFAR-100 原始数据**
   - 代码默认将数据放在 `data/` 目录。首次运行时会自动下载，也可以提前手动下载到服务器并解压至相同路径。

2. **CIFAR-100-C**
   - 官方下载链接：https://zenodo.org/records/3555552
   - 下载后，将所有 `.npy` 文件置于 `data/CIFAR-100-C/`，目录结构应为：
     ```
     data/CIFAR-100-C/
       ├── brightness.npy
       ├── contrast.npy
       ├── ...
       └── labels.npy
     ```

## 3. 分阶段运行

以下示例均使用 `python - <<'PY' ... PY` 的方式在命令行直接执行，也可以把代码保存成脚本再运行。

### 3.1 阶段 A：单增强 Sobol + ASHA（NoAug 基准）

```bash
python - <<'PY'
from src.search.stage_a import StageAConfig, StageAScreener
from src.data.dataset import DataModuleConfig

cfg = StageAConfig(
    transform_name="ColorJitter",
    n_samples=32,
    sobol_seed=0,
    data=DataModuleConfig(root="data"),
    output_dir="artifacts/stage_a/colorjitter",
)
StageAScreener(cfg).run()
PY
```

输出包括：
- `*_results.csv`：所有 rung 的指标；
- `*_topk.json`：最终 Top-4 配置；
- `*_examples.png`：各候选的原图/增强图示例。

### 3.2 阶段 B：联合调参

把多个阶段 A 的 `*_topk.json` 传入 sampler：

```bash
python - <<'PY'
from src.search.stage_b import StageBConfig, StageBTuner, StageBPolicySamplerConfig
from src.data.dataset import DataModuleConfig
from src.train.engine import OptimizerConfig, SchedulerConfig, TrainingConfig
from src.train.mixup import MixupConfig

cfg = StageBConfig(
    sampler=StageBPolicySamplerConfig(
        candidate_paths=[
            "artifacts/stage_a/colorjitter/ColorJitter_topk.json",
            "artifacts/stage_a/randomerasing/RandomErasing_topk.json",
            # ...添加其它增强
        ],
        num_policies=150,
    ),
    data=DataModuleConfig(root="data"),
    optimizer=OptimizerConfig(lr=0.2),
    scheduler=SchedulerConfig(max_epochs=120, warmup_epochs=5, eta_min=1e-4),
    training=TrainingConfig(epochs=120, grad_clip=1.0, use_amp=True),
    mixup=MixupConfig(alpha=0.2, enabled=True),
    output_dir="artifacts/stage_b",
)
StageBTuner(cfg).run()
PY
```

结果：`artifacts/stage_b/stage_b_top10.json`，供阶段 C 使用。

### 3.3 阶段 C：策略级采样 + 贪心组合

```bash
python - <<'PY'
from src.search.stage_c import StageCConfig, StageCPolicyEnsembler
from src.data.dataset import LowDataSplitConfig
from src.train.engine import OptimizerConfig, SchedulerConfig, TrainingConfig
from src.train.mixup import MixupConfig

cfg = StageCConfig(
    policy_path="artifacts/stage_b/stage_b_top10.json",
    data_root="data",
    split_config=LowDataSplitConfig(root="data", subset_indices_path="artifacts/splits/low_data.json"),
    optimizer=OptimizerConfig(lr=0.1),
    scheduler=SchedulerConfig(max_epochs=200, warmup_epochs=5, eta_min=1e-4),
    training=TrainingConfig(epochs=200, grad_clip=1.0, use_amp=True),
    mixup=MixupConfig(alpha=0.2, enabled=True),
    seeds=(0,1,2,3,4),
    output_dir="artifacts/stage_c",
)
StageCPolicyEnsembler(cfg).greedy_select()
PY
```

输出：`stage_c_selection.json`，记录在增益 ≥0.3%、p<0.05 条件下选择的策略集合。

### 3.4 CIFAR-100-C 鲁棒性

```bash
python - <<'PY'
import torch
from src.eval.cifar100c import CIFAR100CConfig, CIFAR100CEvaluator
from src.models.resnet import build_resnet18

model = build_resnet18()
model.load_state_dict(torch.load("path/to/final_policy.pt"))

cfg = CIFAR100CConfig(root="data/CIFAR-100-C")
scores = CIFAR100CEvaluator(cfg).evaluate(model)
print(scores["mCA"])
PY
```

### 3.5 结果汇总

将多 seed 结果写入 `ResultsAggregator`，生成 `results_summary.csv`：

```bash
python - <<'PY'
from src.eval.report import EvalEntry, ResultsAggregator

agg = ResultsAggregator()
agg.add(EvalEntry(name="NoAug", seed=0, split="val", top1=48.2, top5=72.5))
# ... 添加其它条目
agg.to_csv("artifacts/results_summary.csv")
PY
```

## 4. 上传到服务器

1. **打包项目**
   ```bash
   cd /Users/qinfuyao
   tar czf dl.tar.gz dl
   ```

2. **上传**（示例：上传到 `user@server:/home/user/projects/`）
   ```bash
   scp dl.tar.gz user@server:/home/user/projects/
   ```

3. **服务器解压**
   ```bash
   ssh user@server
   cd /home/user/projects
   tar xzf dl.tar.gz
   cd dl
   ```

4. **按“环境准备”章节重新创建虚拟环境并执行各阶段脚本**。

## 5. 常见问题

- **GPU 不够用**：阶段 A/B 可通过减少 `n_samples`、`num_policies` 或缩短 `SchedulerConfig.max_epochs` 暂时做 smoke test。
- **长时间运行**：建议把每个阶段脚本封装成 job 提交到集群，并在 `output_dir/logs` 中记录日志。
- **断点续跑**：阶段 A/B 输出中每个 trial 都会保存指标，可灵活重启；阶段 C 由于含多 seed 全训练，建议保留 `StageCPolicyEnsembler.evaluate_policy_set` 的中间结果（可在类中扩展缓存逻辑）。

完成以上步骤后即可在服务器上按计划启动实验，并将结果（策略 JSON、模型权重、CSV/图像）打包备份。

