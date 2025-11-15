# An Empirical Study of Simple Data-Augmentation Combinations in Low-Data Image Classification

---

## 一、数据与设定

### （补充）训练种子与统计说明
- 阶段 A/B 搜索全部使用 **1 seed**，仅用于粗排和候选筛选，不报告正式显著性结论。
- **5 seeds** 仅用于阶段 B/C 选出的少量候选与最终模型的正式评估。
- 由于验证集较小且 seed 较少，阶段 A/B 中的统计检验（如 bootstrap CI）仅作为**排序辅助**，正式的 t-test / BH-FDR 只在 5 seeds 结果上报告。

- **数据集**：CIFAR-100，每类仅使用 100 张样本（20%），其中 90 用于训练、10 用于验证；测试使用官方 test 集。
- **模型**：ResNet-18。
- **评估指标**：Top-1 Accuracy（mean ± std，5 seeds）、Top-5 Accuracy、混淆矩阵、每类准确率条形图与 loss 曲线。

---

## 二、增强空间与合法范围

### （补充）NoAug 基准与增强组合原则
- 阶段 A/B 默认从 **NoAug** 出发（仅 Resize+Normalize），将待搜索的基础增强串联成 pipeline，不再附加 WeakAug。

每个基础增强定义为 (变换类型, 概率 p, 幅度 m)，并设定合法范围。

| 类别 | 操作 | 合法范围（宽起点） |
|------|------|------------------|
| **几何类 (Geometric)** | RandomResizedCrop | scale ∈ (0.5, 1.0]；ratio ∈ [0.75, 1.33]；p ∈ [0.25, 1.0] |
|  | RandomCrop | padding ∈ {0, 2, 4, 8}；p ∈ [0.5, 1.0] |
|  | RandomRotation | degrees ∈ {±5, ±10, ±15}；p ∈ [0.25, 0.75] |
|  | RandomPerspective | distortion_scale ∈ [0.05, 0.6]；p ∈ [0.25, 0.75] |
|  | RandomHorizontalFlip | p ∈ [0.25, 0.9] |
| **颜色/光度类 (Photometric)** | ColorJitter | brightness/contrast/saturation ∈ [0.2, 0.6]；hue ∈ [0.05, 0.15]；p ∈ [0.25, 0.9] |
|  | RandomGrayscale | p ∈ [0.05, 0.5] |
|  | GaussianBlur | kernel_size ∈ {3, 5}；sigma ∈ [0.1, 2.0]；p ∈ [0.25, 0.75] |
|  | GaussianNoise | σ ∈ [0.02, 0.20]；p ∈ [0.25, 0.75] |
| **遮挡/噪声类 (Occlusion/Noise)** | RandomErasing | scale ∈ [0.02, 0.40]；ratio ∈ [0.3, 3.3]；p ∈ [0.25, 0.75] |

**语义保真约束**：  
1. **人类先验**：禁止上下翻转，旋转角度 |θ| ≤ 15°。  
2. **一致性检测（Consistency Check）**  
   - 使用在 **完整 CIFAR-100 训练集（50k）** 上训练的 ResNet-18 Baseline 模型，仅用于一致性评估，不参与增强搜索和最终对比。  
   - Baseline 训练配置：NoAug（仅 Resize+Normalize）、SGD(momentum=0.9, weight_decay=5e-4)、CosineAnnealingLR、200 epoch（或与主实验一致的标准设定），可以复用公开实现。  
   - 一致率定义：在固定的一致性评估集（默认使用全部 50,000 张训练图像，或每类 100 张、共 10,000 张子集）上，原图与增强图 argmax 相同的比例。  
   - 阶段 A 约束：一致率 ≥ 0.85，若某组 (p,m) 未通过约束则直接丢弃，不进入训练比较。  

**附加输出要求**：  
在阶段 A 初步扫描合法范围时，**每种变换的所有 m 值**均需输出一张示例图像（原图 + 增强图并排显示），以供人工筛查与论文附图使用。

---

## 三、阶段化搜索与组合流程

### 阶段 A：筛选（Screening）

**目标**：粗粒度探索每个增强在合法幅度范围 m 和概率 p 下的效果，筛选出显著有效的子范围。  
**Baseline**：NoAug（仅 Resize+Normalize）。

#### 参数空间说明

| 参数 | 含义 | 取值范围 | 说明 |
|------|------|-----------|------|
| **p** | 增强触发概率 | [0.25, 1.0] | 小于 0.25 效果太弱，大于 1 无意义。 |
| **m** | 幅度（变换强度） | 各增强合法范围（见上表） | 例如 Rotation ±15°、ColorJitter 亮度/对比 0.2–0.6。 |
| **采样方式** | Sobol 序列 | 32 组 (p, m) 组合 / 每种增强 | 均匀覆盖，低方差采样。 |

#### 训练与优化配置（A10 最优设定）

- **硬件**：NVIDIA A10 (24GB VRAM)
- **批大小**：256（自动混合精度开启 AMP）
- **优化器**：SGD(momentum=0.9, weight_decay=5e-4)
- **学习率调度器**：CosineAnnealingLR(T_max=30, eta_min=1e-4)
- **初始学习率**：0.2（线性 warmup 前 5 epoch）
- **梯度裁剪**：1.0
- **训练轮数**：30 epoch
- **早停策略**：ASHA（用于粗筛；Top-k 需 full training），最小资源 10 epoch，rungs = {10, 20, 30}，reduction factor η = 2。
- **具体规则**：所有 trial 先训练到 10 epoch，保留 Top 1/2 进入 20 epoch，再从 20 epoch 保留 Top 1/2 进入 30 epoch。

#### 筛选标准

- ΔTop-1 ≥ +1.0%，95% bootstrap CI 不跨 0（基于单 seed、多次评估 / batch 级 bootstrap，用于排序参考）。  
- bootstrap(10k) 或近似 t 检验 p < 0.05，作为排序辅助；正式显著性结论在 5 seeds 结果上再做配对 t-test + BH-FDR(q=0.1)。  
- 一致率 ≥ 0.85。  
- 每变换保留 Top-4 候选点（Δp ≥ 0.15，Δm ≥ 0.1 去重）。

#### 阶段输出

- 每变换有效子范围 + Top-4 (p,m) JSON  
- (p,m) → Top-1 热力图  
- 增强可视化示例图  
- 日志 CSV  
- 模型保存：`A_stage_transform_p{p}_m{m}.pt`

---

### 阶段 B：联合调参（Tuning）

**目标**：在阶段 A 筛选出的子范围内联合优化各变换的 (p,m)。  

- **算法**：ASHA（粗筛 + Top-k full training），最小资源 30 epoch，rungs = {30, 60, 120}，reduction factor η = 3。  
- **搜索预算**：最多 150 组策略（全部先训练到 30 epoch，随后保留约 1/3 进入 60 epoch，再保留约 1/3 进入 120 epoch）。  
- **训练轮数**：最多 120 epoch（仅对通过 ASHA 的少量候选 fully train）。  
- **调度**：CosineAnnealingLR  
- **输出**：Top-10 策略及日志。

---

### 阶段 C：策略组合（Policy Ensemble）

- **策略定义**：每个策略 Sᵢ 表示在 **NoAug 基准** 上串联的一组基础增强（若干变换及其 (p,m)），即所有策略都从无增强起步，仅由搜索到的增强序列构成。  
- **Mixup 位置**：默认在 Sᵢ 之后、网络输入之前应用 Mixup，即 `NoAug + Sᵢ (+Mixup)`；可在消融中关闭以验证贡献。  
- **策略级采样**：训练时每个 batch 等概率从候选集合中采样一条策略 Sᵢ，该 batch 内所有样本使用同一条策略（再接 Mixup）。  

**贪心选择流程**：  
1. 在阶段 B 的 Top-10 中，先选单策略表现最好的 S₁。  
2. 依次尝试加入新的策略 Sᵢ，形成集合 {S₁, ..., S_k}，按上述策略级采样重新训练模型。  
3. 若相对于当前最佳集合的增益 ΔTop-1 < +0.3% 或配对 t-test 的 p ≥ 0.05，则停止加入新的策略。  
4. 最多保留 K_max = 4 条策略。  

**训练配置**：  
- q = 1/K  
- epoch = 200  
- cosine LR + AMP  
- 保存：`final_policy.pt`

---

## 四、对照与消融

1. NoAug（baseline）  
2. NoAug + Mixup  
3. RandAugment(N=2, M=10)  
4. WeakAug（参考对照，用于衡量与常规弱增强的差距）  
5. 仅调 p / 仅调 m / 同时调 (p,m)（在相同预算下比较搜索效率）  
6. 跳过阶段 A 全范围搜索（直接在原始合法范围上做联合搜索）  
7. 去掉语义保真约束：Ours w/o Consistency Constraint（不施加一致率 ≥ 0.85）  
8. 去掉 Mixup：Ours (no Mixup)，验证增益是否主要来自图像增强策略本身  
9. 人工强增强：NoAug + 人工设定的 ColorJitter + RandomErasing（经验型组合），与自动搜索策略对比  

---

## 五、统计与报告规范

- 5 seeds → mean ± std  
- 搜索阶段使用 bootstrap CI 作为排序辅助，最终报告基于 5 seeds 的配对 t-test + BH-FDR(q=0.1)。  
- 输出训练曲线  
- 记录阶段耗时  
- `results_summary.csv`

---

## 六、鲁棒性验证（CIFAR-100-C）

- 在 CIFAR-100 上训练好的最终模型上直接评估，不在 CIFAR-100-C 上做任何额外调参或微调。  
- 评价对象：NoAug、NoAug+Mixup、RandAugment、WeakAug（参考）、单策略最优 S₁、最终策略集（Policy Ensemble）等关键方法，默认在同样的 5 个 seeds 下分别计算 mCA 并汇总 mean ± std。  
- 按 CIFAR-100-C 官方设定，对 15 种 corruption × 5 个 severity 的 top-1 accuracy 取平均：  
  \[
  \text{mCA} = \frac{1}{15 \times 5}\sum_{\text{corruption}, \text{severity}} \text{Acc}
  \]  
- 报告 mCA（mean ± std），并与 RandAugment / WeakAug（参考）等方法对照。

---

## 七、复现要求

- **数据子集与划分**：  
  - 每类固定采样 100 张图像作为低数据子集，其中 90/10 划分为 train/val；  
  - 公开发布每类子集的索引文件（如 `.txt/.npy`），而不仅仅给出随机种子，保证不同实现间的一致性。  
- **随机性控制**：  
  - 固定 Python / NumPy / PyTorch / CUDA 等的随机种子；  
  - 说明是否启用 `cudnn.deterministic=True` 和 `cudnn.benchmark=False`（如为加速未完全 deterministic，在附录中注明）。  
- **环境与版本**：  
  - 固定并在附录中列出 PyTorch、torchvision、CUDA/cuDNN、Python 等关键库版本，以及 GPU 型号（如 NVIDIA A10）。  
- **验证与测试流程**：  
  - val/test 仅做 Resize+Normalize，不叠加任何训练增强。  
- **日志与结果文件**：  
  - 统一以 `results_summary.csv` 汇总关键结果（包含 dataset、policy_id、seed、Top-1、Top-5、mCA 等列）；  
  - 阶段 A/B 的 (p,m) 搜索结果与 Top-k 策略以 JSON/CSV 形式公开；  
  - 阶段 C 最终策略集以 JSON 形式公开（列出每条策略所包含的基础变换及 (p,m)）。

---
