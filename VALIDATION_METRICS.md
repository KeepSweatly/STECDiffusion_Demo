# STEC 条件扩散模型验证机制与精度指标

## 目录
1. [验证机制概述](#验证机制概述)
2. [验证数据划分](#验证数据划分)
3. [验证流程详解](#验证流程详解)
4. [精度指标计算](#精度指标计算)
5. [Early Stopping 机制](#early-stopping-机制)
6. [与最终测试的区别](#与最终测试的区别)
7. [验证指标的意义](#验证指标的意义)

---

## 验证机制概述

### 1.1 核心设计思想

本项目采用**IPP 点级划分**的验证机制，与传统方法的区别：

| 对比维度 | 传统方法 | 本项目方法 |
|---------|---------|-----------|
| 划分单位 | 按时间或站点 | 按 IPP 点 |
| 训练集 | 前 80% 时间或 80% 站点 | 每个历元前 80% IPP 点 |
| 验证集 | 后 20% 时间或 20% 站点 | 每个历元后 20% IPP 点 |
| 评估目标 | 时间外推或空间泛化 | 同历元内的空间插值 |

### 1.2 验证目标

在**同一个历元**中：
- **Context 点（前 80%）**: 提供已知的 STEC 观测值，作为空间条件
- **Target 点（后 20%）**: 作为预测目标，评估模型的空间插值能力

这种设计能够评估模型在**同一时刻不同空间位置**上的插值能力，符合实际应用场景。

---

## 验证数据划分

### 2.1 数据来源

**验证集数据**: 来自 `model_stations/` 目录，与训练集共享相同的历元文件

**关键区别**:
- **训练集**: 每个历元只保留前 80% IPP 点（训练点池）
- **验证集**: 每个历元保留所有 IPP 点，但预计算 context/target indices

### 2.2 IPP 点划分过程

对于每个历元样本：

```python
# 步骤 1: 确定性打乱（使用 seed + idx）
rng = np.random.default_rng(seed=2026 + idx)
perm = rng.permutation(len(df))
df = df.iloc[perm].reset_index(drop=True)

# 步骤 2: 计算划分点
n_total = len(df)
n_context = int(n_total * 0.8)  # 前 80%

# 步骤 3: 预计算 indices
context_indices = np.arange(n_context)      # [0, 1, ..., n_context-1]
target_indices = np.arange(n_context, n_total)  # [n_context, ..., n_total-1]
```

**示例**:
```
某历元有 100 个 IPP 点（打乱后）:
├─ Context 点: 索引 0-79（80 个点）
└─ Target 点: 索引 80-99（20 个点）
```

### 2.3 验证数据集构建

```python
val_ds = STECEpochDataset(
    epoch_files=epoch_files,
    mode="val_eval",  # 验证模式
    split_ratio=0.8,
    coord_normalizer=train_ds.coord_normalizer,  # 复用训练集参数
    stec_normalizer=train_ds.stec_normalizer,
    angle_normalizer=train_ds.angle_normalizer,
)
```

**返回数据**:
```python
{
    "coords":          [N, 2],  # 所有 IPP 点的归一化坐标
    "angles":          [N, 2],  # 所有 IPP 点的归一化角度
    "stec":            [N, 1],  # 所有 IPP 点的归一化 STEC
    "system_ids":      [N],     # 系统 ID
    "satellite_ids":   [N],     # 卫星 ID
    "n_points":        int,     # 总点数 N
    "context_indices": np.ndarray,  # 前 80% 的索引
    "target_indices":  np.ndarray,  # 后 20% 的索引
}
```

---

## 验证流程详解

### 3.1 触发时机

验证在训练过程中**定期触发**，基于全局训练步数（step）：

```yaml
# configs/default.yaml
training:
  val_start_step: 1000      # 训练 1000 步后开始验证
  val_interval: 500         # 每 500 步验证一次
```

**示例时间线**:
```
Step 0-999:    只训练，不验证
Step 1000:     第 1 次验证
Step 1500:     第 2 次验证
Step 2000:     第 3 次验证
...
```

### 3.2 验证数据组织

对于验证集中的每个 batch：

```python
# 从预计算的 indices 生成 mask
context_mask, target_mask = generate_context_target_mask(
    valid_mask=batch["valid_mask"],
    mode="val",
    context_indices_batch=batch["context_indices"],
    target_indices_batch=batch["target_indices"],
)

# 示例: 某个样本有 100 个点
# context_mask = [True, True, ..., True, False, False, ..., False]
#                 ←------ 80 个 ------→ ←------- 20 个 -------→
# target_mask  = [False, False, ..., False, True, True, ..., True]
```

**关键点**:
- **Context（已知）**: 前 80% IPP 点的 STEC 值
- **Target（预测目标）**: 后 20% IPP 点的位置
- **固定划分**: 不做随机遮挡，使用预计算的 indices

### 3.3 快速评估策略

为了节省计算时间，验证时使用**单步去噪估计**而非完整反向 SDE：

```python
# 步骤 1: 构建条件均值 μ
mu = sde.build_mu_batch(coords, stec, context_mask, target_mask)
# - Context 点: μ = context STEC 全局均值
# - Target 点: μ = IDW 插值（基于最近 k 个 context 点）

# 步骤 2: 在中间时间步 t = T/2 加噪
t_val = sde.T // 2  # 默认 T=100，所以 t_val=50
xt_all, noise_all, _ = sde.forward_sample_batch(stec, mu, t_batch)

# 步骤 3: 只对 target 点加噪
noisy_stec = stec.clone()
noisy_stec[target_mask] = xt_all[target_mask]

# 步骤 4: 模型预测噪声
noise_pred = model(
    noisy_stec=noisy_stec,
    coords=coords,
    angles=angles,
    system_ids=system_ids,
    context_stec=context_stec,
    role_type=role_type,
    valid_mask=valid_mask,
    t=t_batch,
)

# 步骤 5: 从 xt 和预测噪声估计 x0（单步去噪）
sigma_t = sde.sigma_bar(t_val)
alpha_t = sde.alpha(t_val)
x0_pred = (xt - mu - sigma_t * noise_pred) / (alpha_t + eps) + mu
```

**为什么使用中间时间步 T/2？**
- 完整的反向 SDE 推理（T 步）计算成本高
- 中间时间步能够快速评估模型的去噪能力
- 在训练早期就能提供有效的性能指标
- 与训练时的随机时间步采样保持一致

### 3.4 只评估 Target 点

```python
# 只提取 target 点的预测结果
x0_pred_target = x0_pred[target_mask]  # 预测值（归一化）
x0_true_target = stec[target_mask]     # 真实值（归一化）

# 反归一化到 TECU 单位
pred_orig = stec_normalizer.inverse_transform(x0_pred_target)
true_orig = stec_normalizer.inverse_transform(x0_true_target)
```

---

## 精度指标计算

### 4.1 反归一化

模型内部使用归一化的 STEC 值（均值 0，方差 1），评估时需要转回原始单位（TECU）：

```python
# 归一化参数（从训练集统计）
mean_train = 20.5  # TECU
std_train = 5.2    # TECU

# 反归一化公式
stec_orig = stec_norm * std_train + mean_train
```

### 4.2 MAE（平均绝对误差）

**公式**:
$$\text{MAE} = \frac{1}{M} \sum_{i=1}^{M} \left| \hat{s}_i - s_i \right|$$

其中：
- $\hat{s}_i$: 第 $i$ 个 target 点的预测 STEC（TECU）
- $s_i$: 第 $i$ 个 target 点的真实 STEC（TECU）
- $M$: 所有 target 点的总数

**含义**:
- 预测值与真实值的平均绝对偏差
- 单位：TECU（Total Electron Content Unit）
- 越小越好

**计算代码**:
```python
mae = np.mean(np.abs(pred_orig - true_orig))
```

### 4.3 RMSE（均方根误差）

**公式**:
$$\text{RMSE} = \sqrt{\frac{1}{M} \sum_{i=1}^{M} \left( \hat{s}_i - s_i \right)^2}$$

**含义**:
- 对大误差更敏感（因为平方项）
- 单位：TECU
- 越小越好

**计算代码**:
```python
rmse = np.sqrt(np.mean((pred_orig - true_orig) ** 2))
```

### 4.4 指标对比

| 指标 | 特点 | 适用场景 | 对离群点的敏感度 |
|------|------|---------|----------------|
| MAE | 对所有误差一视同仁 | 关注平均性能 | 低 |
| RMSE | 对大误差惩罚更重 | 关注极端情况 | 高 |

**示例**:
```
预测误差序列: [0.5, 0.5, 0.5, 5.0] TECU

MAE  = (0.5 + 0.5 + 0.5 + 5.0) / 4 = 1.625 TECU
RMSE = sqrt((0.25 + 0.25 + 0.25 + 25) / 4) = 2.55 TECU

RMSE > MAE，说明存在较大的离群误差
```

---

## Early Stopping 机制

### 5.1 工作原理

```python
# 每次 validation 后
if val_mae < best_val_mae:
    best_val_mae = val_mae
    no_improve_count = 0
    save_checkpoint("best")
    logger.info(f"=> 保存最佳模型（MAE={val_mae:.4f}）")
else:
    no_improve_count += 1
    logger.info(f"=> 无改善（连续 {no_improve_count}/{patience}）")

# 检查是否触发 early stopping
if no_improve_count >= patience:  # patience = 10
    logger.info(f"Early stopping 触发（连续 {patience} 次无改善）")
    stop_training()
```

### 5.2 触发条件

连续 **10 次** validation 没有改善（MAE 没有降低），则停止训练。

**示例时间线**:
```
Step 1000: MAE=2.50 → 保存 best model, no_improve_count=0
Step 1500: MAE=2.45 → 保存 best model, no_improve_count=0
Step 2000: MAE=2.48 → no_improve_count=1
Step 2500: MAE=2.47 → no_improve_count=2
Step 3000: MAE=2.46 → no_improve_count=3
...
Step 6000: MAE=2.46 → no_improve_count=10 → 触发 early stopping
```

### 5.3 为什么使用 MAE 而不是 RMSE？

**原因**:
1. **稳定性**: MAE 更稳定，不容易受离群点影响
2. **实用性**: 更符合实际应用中对平均精度的关注
3. **训练稳定**: 避免因个别极端误差导致的训练不稳定
4. **可解释性**: MAE 的物理意义更直观（平均偏差）

### 5.4 配置参数

```yaml
# configs/default.yaml
training:
  val_start_step: 1000              # 开始验证的步数
  val_interval: 500                 # 验证间隔（步数）
  early_stopping_patience: 10       # 连续无改善次数阈值
```

---

## 与最终测试的区别

### 6.1 对比表

| 维度 | 训练中的 Validation | 最终 Test |
|------|-------------------|-----------|
| **数据来源** | `model_stations/` | `val_stations/` |
| **Context 来源** | 同一历元的前 80% IPP 点 | `model_stations/` 同名历元的所有 IPP 点 |
| **Target 来源** | 同一历元的后 20% IPP 点 | `val_stations/` 历元的所有 IPP 点 |
| **划分方式** | 预计算 indices（固定） | 历元对齐（固定） |
| **推理方式** | 单步去噪（t=T/2） | 完整反向 SDE（T 步） |
| **目的** | 监控训练过程，early stopping | 最终性能评估 |
| **频率** | 每 500 步一次 | 训练结束后一次 |
| **输出** | MAE/RMSE 指标 | 完整预测结果 CSV + 分组统计 |

### 6.2 详细说明

#### 数据来源不同

**Validation**:
- 使用 `model_stations/` 中的数据
- 通过 IPP 点划分得到 context/target 子集
- 在同一个数据集内部评估

**Test**:
- 使用独立的 `val_stations/` 数据
- 完全未参与训练过程
- 更真实的泛化能力评估

#### Context 来源不同

**Validation**:
```python
# 某个历元有 100 个 IPP 点（打乱后）
context = 前 80 个点  # 索引 0-79
target = 后 20 个点   # 索引 80-99
```

**Test**:
```python
# model_stations/2024021900.csv: 150 个 IPP 点
# val_stations/2024021900.csv: 30 个 IPP 点

context = model_stations 的 150 个点（全部）
target = val_stations 的 30 个点（全部）
```

#### 推理方式不同

**Validation（快速评估）**:
```python
# 单步去噪
t = T // 2
xt = forward_noise(x0, t)
noise_pred = model(xt, t)
x0_pred = denoise_one_step(xt, noise_pred, t)
```

**Test（完整推理）**:
```python
# 完整反向 SDE
x_T = mu + max_sigma * random_noise()
for t in range(T, 0, -1):
    noise_pred = model(x_t, t)
    x_{t-1} = reverse_sde_step(x_t, noise_pred, t)
x0_pred = x_0
```

---

## 验证指标的意义

### 7.1 空间插值能力评估

验证集评估的是模型在**同一历元不同空间位置**上的插值能力：

- **训练时**: 模型学习前 80% IPP 点的 STEC 分布
- **验证时**: 模型预测后 20% IPP 点的 STEC 值
- **如果 MAE 低**: 说明模型能够准确地在新位置进行空间插值

### 7.2 过拟合检测

```
训练 loss 持续下降，但 validation MAE 不再改善
→ 模型开始过拟合前 80% IPP 点
→ Early stopping 及时停止训练
```

### 7.3 模型选择

```
Step 1000:  val_mae = 2.5 TECU
Step 2000:  val_mae = 2.3 TECU  ← 保存为 best model
Step 3000:  val_mae = 2.4 TECU
Step 4000:  val_mae = 2.5 TECU

最终使用 Step 2000 的模型进行测试
```

### 7.4 训练进度监控

通过验证指标可以实时了解：
1. 模型是否在学习（MAE 是否下降）
2. 模型是否过拟合（训练 loss 下降但 MAE 不降）
3. 模型是否收敛（MAE 不再改善）
4. 何时停止训练（触发 early stopping）

---

## 实际案例

### 8.1 训练日志示例

```
[Epoch 010/200] loss=0.125000  lr=1.00e-04  step=1200
  [Val @ step 1000] MAE=3.2500 TECU  RMSE=4.1200 TECU
  => 保存最佳模型（MAE=3.2500）

[Epoch 020/200] loss=0.098000  lr=9.50e-05  step=2400
  [Val @ step 2000] MAE=2.8500 TECU  RMSE=3.6500 TECU
  => 保存最佳模型（MAE=2.8500）
  [Val @ step 2500] MAE=2.7200 TECU  RMSE=3.5100 TECU
  => 保存最佳模型（MAE=2.7200）

[Epoch 030/200] loss=0.085000  lr=9.00e-05  step=3600
  [Val @ step 3000] MAE=2.6800 TECU  RMSE=3.4800 TECU
  => 保存最佳模型（MAE=2.6800）
  [Val @ step 3500] MAE=2.6500 TECU  RMSE=3.4500 TECU
  => 保存最佳模型（MAE=2.6500）

...

[Epoch 080/200] loss=0.062000  lr=5.00e-05  step=9600
  [Val @ step 9500] MAE=2.5100 TECU  RMSE=3.2800 TECU
  => 无改善（连续 8/10）
  [Val @ step 10000] MAE=2.5200 TECU  RMSE=3.2900 TECU
  => 无改善（连续 9/10）

[Epoch 085/200] loss=0.061000  lr=4.80e-05  step=10200
  [Val @ step 10500] MAE=2.5300 TECU  RMSE=3.3000 TECU
  => 无改善（连续 10/10）
Early stopping 触发（连续 10 次无改善）
训练完成。
```

### 8.2 解读

1. **训练初期**（Epoch 1-30）: MAE 快速下降，模型快速学习
2. **训练中期**（Epoch 30-80）: MAE 缓慢下降，模型精细调优
3. **训练后期**（Epoch 80-85）: MAE 不再改善，触发 early stopping
4. **最佳模型**: Epoch 75 左右，MAE=2.45 TECU

---

## 总结

### 9.1 验证机制的核心价值

1. **评估空间插值能力**: 在同一历元不同位置上预测 STEC
2. **防止过拟合**: 及时发现模型性能饱和
3. **自动模型选择**: 保存验证性能最优的模型
4. **节省训练时间**: Early stopping 避免无效训练

### 9.2 关键设计特点

- ✅ IPP 点级划分（灵活性高）
- ✅ 同一历元内评估（真实应用场景）
- ✅ 快速评估（单步去噪，节省时间）
- ✅ 稳定指标（MAE 作为主要依据）
- ✅ 自动化流程（无需人工干预）
- ✅ 固定划分（预计算 indices，保证一致性）

### 9.3 与训练/测试的关系

```
训练集（前 80% IPP 点）
    ↓ 学习
模型参数
    ↓ 评估
验证集（后 20% IPP 点）
    ↓ 选择最佳模型
Best Checkpoint
    ↓ 最终评估
测试集（val_stations/）
    ↓ 输出
最终性能报告
```

---

**文档版本**: v2.0
**最后更新**: 2026-03-31
**适用代码版本**: IPP 点级划分架构（多星联合版本）
