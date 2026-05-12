# STEC 条件扩散模型数据集划分文档

## 目录
1. [数据组织架构](#数据组织架构)
2. [核心设计原则](#核心设计原则)
3. [数据集划分策略](#数据集划分策略)
4. [三种数据模式](#三种数据模式)
5. [归一化处理](#归一化处理)
6. [训练与验证流程](#训练与验证流程)
7. [配置参数](#配置参数)
8. [关键设计决策](#关键设计决策)

---

## 数据组织架构

### 1.1 物理目录结构

```
stec_diffusionv3/
├── data/
│   ├── model_stations/          # 建模数据目录（训练+验证）
│   │   ├── 2024021900.csv      # 历元文件（多星联合）
│   │   ├── 2024021901.csv
│   │   └── ...
│   │
│   └── val_stations/            # 独立测试数据目录
│       ├── 2024021900.csv
│       ├── 2024021901.csv
│       └── ...
│
└── configs/
    └── default.yaml             # 配置文件
```

### 1.2 历元文件格式

每个 CSV 文件代表一个历元（时刻），包含该时刻所有测站和卫星的 IPP 点观测：

**必要字段**：
- `station_name`: 测站名称
- `ipp_latitude`: 穿刺点纬度（度）
- `ipp_longitude`: 穿刺点经度（度）
- `azimuth_deg`: 方位角（度）
- `elevation_deg`: 高度角（度）
- `stec`: 斜向总电子含量（TECU）
- `system_id`: 卫星系统 ID（0=GPS, 1=GLONASS, 2=Galileo, 3=BDS）
- `satellite_id`: 卫星编号（用于结果分组）

**示例数据**：
```csv
station_name,ipp_latitude,ipp_longitude,azimuth_deg,elevation_deg,stec,system_id,satellite_id
BJFS,39.608,116.191,45.2,30.5,18.5,0,12
BJFS,39.612,116.195,120.8,25.3,20.1,1,7
CHAN,43.792,125.443,180.5,40.2,15.8,0,12
...
```

### 1.3 数据来源说明

- **model_stations/**: 包含大部分测站的观测数据，用于模型训练和验证
- **val_stations/**: 包含少量独立测站的观测数据，用于最终测试评估
- **多星联合**: 每个历元文件包含多个卫星系统的 IPP 点（GPS, GLONASS, Galileo, BDS）
- **历元对齐**: 两个目录中的同名文件代表同一时刻的观测

---

## 核心设计原则

### 2.1 历元级样本组织

**关键特性**：
- 每个历元文件 = 一个样本
- 单个样本包含多颗卫星的 IPP 点
- 不同历元的 IPP 点数量可能不同（变长样本）

**优势**：
- 保持时间一致性，避免时间混淆
- 支持多星联合建模，提升泛化能力
- 自然支持变长样本处理

### 2.2 IPP 点级划分

**与传统站点级划分的区别**：

| 划分方式 | 传统方法 | 本项目方法 |
|---------|---------|-----------|
| 划分单位 | 按测站划分 | 按 IPP 点划分 |
| 训练集构建 | 选择 80% 测站的所有观测 | 每个历元保留前 80% IPP 点 |
| 验证集构建 | 选择 20% 测站的所有观测 | 每个历元保留后 20% IPP 点 |
| 优势 | 评估空间泛化能力 | 灵活性高，不依赖站点唯一性 |
| 适用场景 | 站点固定的网络 | 多星联合，站点可能重复 |

**本项目采用 IPP 点级划分的原因**：
1. 多星联合场景下，同一测站可能观测多颗卫星，站点名称不唯一
2. IPP 点级划分更灵活，支持任意点集
3. 通过确定性打乱 + 固定比例切分，保证可复现性

---

## 数据集划分策略

### 3.1 整体架构

```
原始数据（按测站物理分割）
│
├─── model_stations/ ────────────────────────────────────┐
│       │                                                 │
│       └─ 按历元文件组织                                  │
│           │                                             │
│           ├─ 每个历元文件确定性打乱 IPP 点                │
│           │                                             │
│           ├─ 训练集（前 80% IPP 点）                     │
│           │   - 用于模型参数学习                         │
│           │   - 统计归一化参数（μ, σ, lat/lon 范围）     │
│           │   - 训练时从中随机抽取 target（10%-50%）     │
│           │                                             │
│           └─ 验证集（所有 IPP 点）                       │
│               - 前 80% 作 context（预计算 indices）      │
│               - 后 20% 作 target（预计算 indices）       │
│               - 用于训练中的周期性评估                    │
│               - 指导模型选择（early stopping）           │
│                                                         │
└─── val_stations/ ──────────────────────────────────────┘
        │
        └─ 独立测试集（100%）
            - 完全独立的测站集合
            - 用于最终模型性能评估
            - 评估模型的空间泛化能力
```

### 3.2 详细划分流程

**步骤 1: 扫描历元文件**
```python
epoch_files = sorted(Path(model_stations_dir).glob("*.csv"))
# 示例: ['2024021900.csv', '2024021901.csv', ...]
```

**步骤 2: 确定性打乱（每个历元独立）**
```python
# 使用 seed + idx 确保可复现
rng = np.random.default_rng(seed=2026 + idx)
perm = rng.permutation(len(df))
df = df.iloc[perm].reset_index(drop=True)
```

**步骤 3: 按比例切分**
```python
n_total = len(df)  # 该历元的总 IPP 点数
n_train = int(n_total * 0.8)  # 前 80%

# 训练模式：只保留前 80% 作为训练点池
train_df = df.iloc[:n_train]

# 验证模式：保留所有点，但预计算 indices
context_indices = np.arange(n_train)      # 前 80%
target_indices = np.arange(n_train, n_total)  # 后 20%
```

**步骤 4: 过滤样本**
```python
# 过滤总 IPP 数 < min_total_ipps 的历元
if len(df) < min_total_ipps:
    skip_this_epoch()
```

---

## 三种数据模式

### 4.1 train_context_target 模式（训练集）

**用途**: 模型参数学习

**数据处理**:
```python
# 只保留前 80% IPP 点作为训练点池
df = df.iloc[:n_train].reset_index(drop=True)
```

**Context/Target 划分**:
- 训练时动态生成（每个 batch 随机）
- 从训练点池中随机抽取 10%-50% 作为 target
- 剩余点作为 context

**返回字段**:
```python
{
    "coords":        [N, 2],  # 归一化坐标
    "angles":        [N, 2],  # 归一化角度
    "stec":          [N, 1],  # 归一化 STEC
    "system_ids":    [N],     # 系统 ID
    "satellite_ids": [N],     # 卫星 ID
    "n_points":      int,     # 有效点数
    # 不返回 context_indices / target_indices
}
```

### 4.2 val_eval 模式（验证集）

**用途**: 训练中的周期性评估

**数据处理**:
```python
# 保留所有 IPP 点
context_indices = np.arange(n_train)      # 前 80%
target_indices = np.arange(n_train, n_total)  # 后 20%
```

**Context/Target 划分**:
- 固定划分（使用预计算的 indices）
- 前 80% 作 context
- 后 20% 作 target

**返回字段**:
```python
{
    "coords":          [N, 2],
    "angles":          [N, 2],
    "stec":            [N, 1],
    "system_ids":      [N],
    "satellite_ids":   [N],
    "n_points":        int,
    "context_indices": np.ndarray,  # 前 80% 的索引
    "target_indices":  np.ndarray,  # 后 20% 的索引
}
```

### 4.3 test_target 模式（测试集）

**用途**: 最终模型性能评估

**数据来源**: val_stations/ 目录（完全独立的测站）

**数据处理**:
```python
# 保留所有 IPP 点
target_indices = np.arange(n_total)  # 全部作为 target
```

**Context/Target 划分**:
- Context: 来自 model_stations/ 的同名历元文件（所有 IPP 点）
- Target: 来自 val_stations/ 的当前历元文件（所有 IPP 点）

**返回字段**:
```python
{
    "coords":         [N, 2],
    "angles":         [N, 2],
    "stec":           [N, 1],
    "system_ids":     [N],
    "satellite_ids":  [N],
    "n_points":       int,
    "target_indices": np.ndarray,  # 全部点的索引
}
```

---

## 归一化处理

### 5.1 归一化参数来源

**关键原则**: 归一化参数**仅从训练集统计**，验证集和测试集复用相同参数，防止数据泄露。

```python
# 从训练集统计归一化参数
train_ds = STECEpochDataset(
    epoch_files=epoch_files,
    mode="train_context_target",
    coord_normalizer=None,  # 自动统计
    stec_normalizer=None,   # 自动统计
    angle_normalizer=None,  # 自动统计
)

# 验证集和测试集复用
val_ds = STECEpochDataset(
    epoch_files=epoch_files,
    mode="val_eval",
    coord_normalizer=train_ds.coord_normalizer,  # 复用
    stec_normalizer=train_ds.stec_normalizer,    # 复用
    angle_normalizer=train_ds.angle_normalizer,  # 复用
)
```

### 5.2 坐标归一化（MinMax → [-1, 1]）

```python
lat_norm = 2 * (lat - lat_min) / (lat_max - lat_min + eps) - 1
lon_norm = 2 * (lon - lon_min) / (lon_max - lon_min + eps) - 1
```

**反归一化**:
```python
lat = (lat_norm + 1) * (lat_max - lat_min) / 2 + lat_min
lon = (lon_norm + 1) * (lon_max - lon_min) / 2 + lon_min
```

### 5.3 STEC 标准化（均值 0 方差 1）

```python
stec_norm = (stec - mean_train) / (std_train + eps)
```

**反归一化**:
```python
stec = stec_norm * std_train + mean_train
```

### 5.4 角度归一化（MinMax → [-1, 1]）

```python
azimuth_norm = 2 * (azimuth - 0) / (360 + eps) - 1
elevation_norm = 2 * (elevation - 0) / (90 + eps) - 1
```

---

## 训练与验证流程

### 6.1 训练阶段

```
训练集（前 80% IPP 点）
    │
    ├─ 每个 batch：
    │   ├─ 动态生成 context/target mask
    │   │   - mask_ratio 随机采样自 [10%, 50%]
    │   │   - 从训练点池中随机抽取 target
    │   │
    │   ├─ 构建条件均值 μ 和先验特征（Mu-REG 优化）
    │   │   - context 点: μ = context STEC 全局均值
    │   │   - target 点: μ = IDW 插值（基于最近 k 个 context 点）
    │   │   - 先验特征: prior_mu, prior_unc, prior_gap
    │   │
    │   ├─ 随机采样时间步 t ~ Uniform(1, T)
    │   │
    │   ├─ 前向加噪: xt = mu_bar(x0, t) + sigma_bar(t) * ε
    │   │
    │   ├─ 模型双分支预测（Mu-REG 优化）
    │   │   - 强条件分支: ε_strong = Transformer(..., weak_condition=False)
    │   │   - 弱条件分支: ε_weak = Transformer(..., weak_condition=True)
    │   │
    │   └─ 计算双分支损失（Mu-REG 优化）
    │       L_total = L_strong + λ_w·L_weak + λ_x·L_x0 + λ_j·L_jac
    │
    └─ 每 val_interval 步（默认 500 步）：
        └─ 在验证集上评估（快速单步估计）
```

### 6.2 验证阶段（训练中）

**目的**: 周期性评估模型性能，指导模型选择和 early stopping

**数据来源**: 验证集（所有 IPP 点，前 80% 作 context，后 20% 作 target）

**评估方式**:
1. 使用预计算的 context_indices 和 target_indices
2. 在固定时间步 t = T/2 处，单步估计 x0:
   ```python
   x0_pred = (xt - mu - sigma(t) * noise_pred) / (alpha(t) + eps) + mu
   ```
3. 反归一化到 TECU 单位，计算 MAE 和 RMSE
4. 若 MAE 改善，保存 `ckpt_best.pth`

**Early Stopping**:
- 连续 `early_stopping_patience` 次（默认 10 次）验证无改善，则停止训练

### 6.3 测试阶段

**数据来源**: val_stations/ 目录（完全独立的测站）

**测试流程**:
```
测试集（val_stations/ 目录）
    │
    ├─ 历元对齐：匹配 model_stations/ 中同名历元文件
    │
    ├─ 每个历元：
    │   ├─ Context: model_stations 的所有 IPP 点
    │   ├─ Target: val_stations 的所有 IPP 点
    │   │
    │   ├─ 构建条件均值 μ 和先验特征（Mu-REG 优化）
    │   │
    │   ├─ 初始化 x_T = μ + max_sigma * ε
    │   │
    │   ├─ 完整反向 SDE：循环 t = T, T-1, ..., 1
    │   │   每步：
    │   │     - 强条件分支: ε_strong = Transformer(..., weak_condition=False)
    │   │     - Guidance 修正（Mu-REG 优化，可选）:
    │   │       if use_guidance:
    │   │         ε_weak = Transformer(..., weak_condition=True)
    │   │         Δε = ε_strong - ε_weak
    │   │         w = w_max·s(t)·exp(-β·u)  # 时步+空间自适应
    │   │         ε_guided = ε_strong + w·Δε
    │   │     - x_{t-1} = mu_bar(x0_pred, t-1) + sigma(t-1) * ε_new
    │   │
    │   └─ 反归一化预测值与真实值 → 计算 MAE / RMSE
    │
    └─ 汇总所有历元的误差，保存至 JSON 文件
```

---

## 配置参数

### 7.1 数据划分参数（configs/default.yaml）

```yaml
data:
  model_stations_dir: "data/model_stations"
  val_stations_dir: "data/val_stations"
  split_ratio: 0.8                    # IPP 点划分比例
  seed: 2026                          # 随机种子
  min_total_ipps_per_sample: 20       # 历元总 IPP 数阈值
  min_test_context_ipps: 20           # 测试时 context IPP 数阈值
  min_test_target_ipps: 10            # 测试时 target IPP 数阈值
  max_points: 512                     # 单样本最大点数
  mask_ratio_min: 0.10                # 训练时 target 最小比例
  mask_ratio_max: 0.50                # 训练时 target 最大比例
```

### 7.2 训练与验证参数

```yaml
training:
  batch_size: 32
  num_epochs: 200
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  grad_clip: 1.0
  warmup_epochs: 10
  val_start_step: 1000                # 开始验证的步数
  val_interval: 500                   # 验证间隔（步数）
  early_stopping_patience: 10         # Early stopping 耐心值
```

### 7.3 SDE 参数

```yaml
sde:
  max_sigma: 50.0                     # 最大噪声标准差
  T: 100                              # 扩散总步数
  schedule: "cosine"                  # 噪声调度方式
  eps: 1.0e-8                         # 数值稳定小量

inference:
  idw_power: 2.0                      # IDW 插值幂次
  idw_k: 5                            # IDW 最近邻数量

mu_reg:                               # Mu-REG 优化参数
  prior_unc_a1: 1.0                   # 不确定度权重：距离项
  prior_unc_a2: 1.0                   # 不确定度权重：标准差项
  prior_unc_a3: 0.5                   # 不确定度权重：凸包外指示项
  prior_gap_k2: 2                     # prior_gap 计算的 k=2 邻居数
  lambda_w: 0.5                       # 弱条件损失权重
  lambda_x: 0.2                       # x0 重建损失权重
  lambda_j: 1.0e-4                    # Jacobian 稳定项权重
  guidance_scale_max: 2.0             # 最大 guidance 强度
  guidance_beta: 1.0                  # 不确定度抑制系数
  weak_context_dropout: 0.3           # 弱条件分支 context dropout 比例
  guidance_schedule: "sin2"           # 时步调度函数（sin2/linear/constant）
```

---

## 关键设计决策

### 8.1 为什么采用 IPP 点级划分？

**对比分析**:

| 划分方式 | 优点 | 缺点 | 适用场景 |
|---------|------|------|---------|
| **时间序列划分**<br>（前80%时间 vs 后20%时间） | 保持时序连续性<br>避免未来信息泄露 | 训练集和测试集包含相同测站<br>无法评估空间泛化能力 | 时间序列预测任务 |
| **站点级划分**<br>（80%站点 vs 20%站点） | 评估空间泛化能力<br>测试集包含未见过的测站 | 需要站点唯一性<br>多星场景下站点可能重复 | 单星固定网络 |
| **IPP 点级划分**<br>（80% IPP vs 20% IPP） | 灵活性高，不依赖站点<br>支持多星联合<br>可复现性强 | 同一历元的点被分割<br>需要确定性打乱 | 多星联合建模 |

**本项目选择 IPP 点级划分的原因**:
1. 多星联合场景下，同一测站可能观测多颗卫星，站点名称不唯一
2. IPP 点级划分更灵活，支持任意点集
3. 通过确定性打乱 + 固定比例切分，保证可复现性
4. 仍然保留空间泛化评估能力（通过 val_stations/ 独立测试集）

### 8.2 数据泄露防护

1. **归一化参数**: 仅从训练集统计，验证集和测试集复用
2. **确定性打乱**: 固定随机种子（seed + idx），保证可复现
3. **独立测试集**: val_stations/ 目录的数据完全独立，不参与训练和验证
4. **预计算 indices**: 验证集的 context/target 划分固定，避免随机性

### 8.3 为什么训练时遮挡比例是随机的？

**训练阶段**:
- mask_ratio 在 [10%, 50%] 范围内随机采样
- 增强模型泛化能力，适应不同的 context/target 比例
- 每个 batch 动态生成，避免过拟合

**验证/测试阶段**:
- 固定划分（前 80% context，后 20% target）
- 保证评估的一致性和可比性
- 便于跨实验对比

### 8.4 多星联合的优势

**设计特点**:
- 所有卫星系统的 IPP 点在同一样本中
- 通过 system_id 嵌入区分不同卫星系统
- 共享模型参数，提升泛化能力

**优势**:
1. 充分利用多源观测数据
2. 学习跨系统的共性特征
3. 提升模型在稀疏观测区域的性能
4. 减少模型数量，简化部署

---

## 整体流程总结

```
原始数据（按测站物理分割）
    │
    ├─── model_stations/ ────────────────────────────────────┐
    │       │                                                 │
    │       ├─ 扫描历元文件                                    │
    │       ├─ 每个历元确定性打乱 IPP 点                        │
    │       │                                                 │
    │       ├─ 训练集（前 80% IPP 点）                         │
    │       │   ├─ 统计归一化参数                             │
    │       │   ├─ 训练时动态生成 mask（10%-50%）             │
    │       │   ├─ 构建先验特征（Mu-REG 优化）                │
    │       │   ├─ 前向加噪 → 双分支预测 → 计算 loss          │
    │       │   └─ L_total = L_strong + λ_w·L_weak + λ_x·L_x0 + λ_j·L_jac
    │       │                                                 │
    │       └─ 验证集（所有 IPP 点）                           │
    │           ├─ 复用训练集归一化参数                        │
    │           ├─ 前 80% 作 context，后 20% 作 target        │
    │           └─ 单步去噪评估 → MAE 改善时保存 best model    │
    │               → Early stopping 判断                     │
    │                                                         │
    └─── val_stations/ ──────────────────────────────────────┘
            │
            ├─ 独立测试集（完全独立的测站）
            ├─ 历元对齐：匹配 model_stations/ 同名文件
            ├─ Context: model_stations 所有 IPP 点
            ├─ Target: val_stations 所有 IPP 点
            ├─ 完整 T=100 步反向 SDE（含 Guidance 优化）
            └─ 最终 MAE / RMSE
```

---

## 常见问题

### Q1: 为什么不采用站点级划分？

**A**: 多星联合场景下，同一测站可能观测多颗卫星，站点名称不唯一。IPP 点级划分更灵活，不依赖站点唯一性，同时通过 val_stations/ 独立测试集仍然可以评估空间泛化能力。

### Q2: val_stations/ 目录的数据何时使用？

**A**: 仅在**最终测试阶段**使用，用于评估模型在完全独立测站上的性能。训练和验证阶段只使用 model_stations/ 目录的数据。

### Q3: 如何保证验证集和测试集不泄露信息？

**A**:
1. 归一化参数仅从训练集统计
2. 确定性打乱使用固定随机种子，保证可复现
3. val_stations/ 目录的数据完全独立，不参与训练和验证
4. 验证集的 context/target 划分固定（预计算 indices）

### Q4: 为什么训练时遮挡比例是随机的，而验证/测试时是固定的？

**A**:
- **训练**: 随机遮挡比例增强模型泛化能力，适应不同的 context/target 比例
- **验证/测试**: 固定遮挡比例保证评估的一致性和可比性

### Q5: 如何处理变长样本？

**A**:
1. 数据集返回变长样本（不做 padding）
2. DataLoader 的 collate_fn 将 batch 内样本 padding 到相同长度
3. 生成 valid_mask 标识真实观测点
4. 模型内部使用 padding mask 忽略填充点

---

**文档版本**: v4.0
**最后更新**: 2026-04-16
**适用代码版本**: Mu-REG 优化版本（五阶段完整实现）
