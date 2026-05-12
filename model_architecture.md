# STEC 条件扩散模型架构文档

## 目录
1. [总体架构概览](#总体架构概览)
2. [扩散 SDE 引擎](#扩散-sde-引擎)
3. [Transformer 噪声预测网络](#transformer-噪声预测网络)
4. [基础构建块](#基础构建块)
5. [位置编码模块](#位置编码模块)
6. [训练阶段数据流](#训练阶段数据流)
7. [推理阶段数据流](#推理阶段数据流)
8. [关键设计思路](#关键设计思路)

---

## 总体架构概览

### 1.1 核心组件

本项目实现了一个用于 **STEC（斜路径总电子含量）空间插值预测** 的条件扩散模型，针对不规则离散点集进行建模。核心架构由以下三部分组成：

```
┌─────────────────────────────────────────────────────────────┐
│                    STEC 条件扩散模型                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐   ┌──────────────────┐   ┌─────────┐ │
│  │  扩散 SDE 引擎   │ → │ Transformer 网络  │ → │ 输出头  │ │
│  │  (STEC_IRSDE)   │   │(STECDiffTransformer)│  │(噪声)  │ │
│  └─────────────────┘   └──────────────────┘   └─────────┘ │
│         ↑                      ↑                            │
│         │                      │                            │
│  ┌──────┴──────┐      ┌───────┴────────┐                  │
│  │ IDW 插值     │      │ 位置编码模块    │                  │
│  │ 条件均值构建  │      │ 多种嵌入策略    │                  │
│  └─────────────┘      └────────────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 设计理念

**参考来源**:
- **扩散框架**: EDiffSR（图像超分辨率条件扩散模型）
- **网络架构**: JustImageTransformer（JiT，简化的 DiT 架构）

**任务适配**:
- 从规则网格图像 → 不规则离散点集
- 从像素级预测 → IPP 点级预测
- 从图像条件 → 空间观测条件

---

## 扩散 SDE 引擎

### 2.1 核心类：STEC_IRSDE

**职责**: 实现扩散过程的前向（加噪）与反向（去噪）步骤

**数学基础**: Ornstein-Uhlenbeck（OU）随机微分方程

#### 前向扩散公式

$$x_t = \bar{\mu}(x_0, t) + \bar{\sigma}(t) \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

其中：

**条件均值轨迹**:
$$\bar{\mu}(x_0, t) = \mu + \alpha(t) \cdot (x_0 - \mu)$$
$$\alpha(t) = e^{-\theta t / T}$$

**条件噪声标准差（余弦调度）**:
$$\bar{\sigma}(t) = \sigma_{\max} \cdot \frac{1 - \cos(\pi t / T)}{2}$$

#### 关键参数

| 参数 | 默认值 | 含义 |
|------|-------|------|
| `max_sigma` | 50.0 | 最大噪声标准差 |
| `T` | 100 | 总扩散步数 |
| `theta` | 1.0 | OU 漂移系数（控制均值回归速度） |
| `schedule` | "cosine" | 噪声调度方式（cosine/linear/constant） |
| `idw_power` | 2.0 | IDW 插值幂次 |
| `idw_k` | 5 | IDW 最近邻数量 |
| `guidance_scale_max` | 2.0 | 最大 Guidance 强度（Mu-REG 优化） |
| `guidance_beta` | 1.0 | 不确定度抑制系数（Mu-REG 优化） |
| `guidance_schedule` | "sin2" | 时步调度函数（Mu-REG 优化） |
| `weak_context_dropout` | 0.3 | 弱条件分支 dropout 比例（Mu-REG 优化） |

### 2.2 条件均值构建（build_mu_batch）

**核心思想**: 为每个点构建条件均值 μ，引导扩散过程

**策略**:
- **Context 点**: μ = context 点 STEC 的全局均值（标量广播）
- **Target 点**: μ = IDW 距离加权插值结果（每个 target 点独立计算）

**IDW 插值公式**:
$$\mu_{\text{target}} = \frac{\sum_{i=1}^{k} w_i \cdot s_i}{\sum_{i=1}^{k} w_i}$$
$$w_i = \frac{1}{(d_i + \epsilon)^p}$$

其中：
- $s_i$: 第 $i$ 个最近 context 点的 STEC 值
- $d_i$: 到第 $i$ 个最近 context 点的距离
- $p$: IDW 幂次（默认 2.0）
- $k$: 最近邻数量（默认 5）

**Mu-REG 优化扩展**:
- 支持返回先验特征 `prior_features` [B, N, 3]
  - `prior_mu`: IDW 插值均值 μ_IDW(k=5)
  - `prior_unc`: 不确定度 u(p) = a1·d_kNN + a2·Std_kNN + a3·outside_hull
  - `prior_gap`: 局部变化指标 Δμ = μ_IDW(k=5) - μ_IDW(k=2)

### 2.3 核心方法

| 方法 | 说明 | 用途 |
|------|------|------|
| `build_mu_batch()` | 构造条件均值 μ 和先验特征 | 训练 + 推理 |
| `forward_sample_batch()` | 批量前向加噪 | 训练 |
| `reverse_sde_step()` | 单步反向去噪 | 推理 |
| `reverse_sde()` | 完整反向采样流程（含 Guidance） | 推理 |
| `guidance_timestep_schedule()` | 时步自适应调度（Mu-REG） | 推理 |
| `compute_spatial_adaptive_weights()` | 空间自适应权重（Mu-REG） | 推理 |

### 2.4 反向去噪公式（含 Mu-REG Guidance）

**单步去噪**（从 $x_t$ 到 $x_{t-1}$）:

```python
# 步骤 1: 预测噪声（强条件分支）
noise_pred_strong = model(..., weak_condition=False)

# 步骤 2: Guidance 修正（Mu-REG 优化，可选）
if use_guidance:
    # 预测弱条件分支噪声
    noise_pred_weak = model(..., weak_condition=True)

    # 计算 guidance 方向
    delta_noise = noise_pred_strong - noise_pred_weak

    # 空间自适应权重
    if prior_unc is not None:
        w = guidance_scale_max * s(t) * exp(-beta * prior_unc)
    else:
        w = guidance_scale_max * s(t)

    # 应用 guidance
    noise_pred = noise_pred_strong + w * delta_noise
else:
    noise_pred = noise_pred_strong

# 步骤 3: 从 xt 和预测噪声估计 x0
x0_pred = (xt - mu - sigma(t) * noise_pred) / (alpha(t) + eps) + mu

# 步骤 4: 计算 t-1 步的条件均值
mean_prev = mu + alpha(t-1) * (x0_pred - mu)

# 步骤 5: 添加随机噪声（DDPM 风格）
if t > 1:
    x_{t-1} = mean_prev + sigma(t-1) * randn()
else:
    x_{t-1} = x0_pred  # 最后一步直接返回去噪结果
```

**Guidance 调度函数**（Mu-REG 优化）:
- `sin2`: $s(t) = \sin^2(\pi t / 2T)$，早期弱后期强（推荐）
- `linear`: $s(t) = t / T$，线性增长
- `constant`: $s(t) = 1.0$，恒定强度

---

## Transformer 噪声预测网络

### 3.1 核心类：STECDiffTransformer

**职责**: 给定加噪 STEC、坐标、角度和时间步，预测每个 target 点上的噪声 ε

#### 输入张量规格

| 张量 | 形状 | 说明 |
|------|------|------|
| `noisy_stec` | `[B, N, 1]` | 加噪后的 STEC（target 处为 $x_t$，context 处为真实值） |
| `coords` | `[B, N, 2]` | 归一化经纬度坐标 |
| `angles` | `[B, N, 2]` | 归一化角度（azimuth, elevation） |
| `system_ids` | `[B, N]` | 卫星系统 ID（0=padding, 1=GPS, 2=GLONASS, ...） |
| `context_stec` | `[B, N, 1]` | context 处真实 STEC（target/padding 处为 0） |
| `role_type` | `[B, N]` | 点角色标签（0=padding, 1=context, 2=target） |
| `valid_mask` | `[B, N]` | 有效点掩码（True=有效, False=padding） |
| `t` | `[B]` | 扩散时间步（1 到 T） |
| `prior_features` | `[B, N, 3]` | 先验特征（Mu-REG 优化，可选） |
| `weak_condition` | `bool` | 是否使用弱条件分支（Mu-REG 优化，默认 False） |
| `context_dropout_rate` | `float` | 弱条件分支的 context dropout 比例（默认 0.3） |

#### 输出张量

| 张量 | 形状 | 说明 |
|------|------|------|
| `noise_pred` | `[B, N, 1]` | 预测噪声（仅 target 点有意义） |

### 3.2 前向传播流程

```
输入特征
    │
    ▼
① 点特征嵌入（2 层 MLP）
   [noisy_stec, context_stec, prior_features] → [B, N, 5] → [B, N, dim]
   （prior_features 包含 3 个先验特征：prior_mu, prior_unc, prior_gap）
    │
    ▼
② 弱条件分支处理（可选，Mu-REG 优化）
   if weak_condition and training:
       对 context_stec 进行 30% 随机 dropout
    │
    ▼
③ 位置编码叠加
   + 地理坐标傅里叶编码: coords → [B, N, dim]
   + 角度傅里叶编码: angles → [B, N, dim]
   + 系统 ID 嵌入: system_ids → [B, N, dim]
   + 角色类型嵌入: role_type → [B, N, dim]
   → 所有嵌入相加: [B, N, dim]
    │
    ▼
④ 条件构造（用于 AdaLN）
   时间步嵌入: t → 正弦编码 → MLP → [B, dim]
   Context 聚合: mean_pool(context_stec) → MLP → [B, dim]
   合并条件: t_emb + ctx_emb → [B, dim]
    │
    ▼
⑤ 6 × STECBlock（AdaLN-Transformer）
   每层: AdaLN → Self-Attention(RoPE) → AdaLN → SwiGLU FFN
    │
    ▼
⑥ 输出头
   最终 AdaLN 归一化 → Linear(dim → 1) → 掩码 padding 点
    │
    ▼
noise_pred: [B, N, 1]
```

### 3.3 嵌入策略详解

#### 1. 点特征嵌入（含先验特征）

```python
# 输入: [noisy_stec(1), context_stec(1), prior_features(3)] 拼接
# prior_features 包含: prior_mu, prior_unc, prior_gap
if prior_features is not None:
    point_features = torch.cat([noisy_stec, context_stec, prior_features], dim=-1)  # [B, N, 5]
else:
    point_features = torch.cat([noisy_stec, context_stec], dim=-1)  # [B, N, 2]
    # 自动填充零向量以保持向后兼容

# 2 层 MLP
x = self.point_embed(point_features)  # [B, N, dim]
```

**先验特征说明（Mu-REG 优化）**:
- `prior_mu`: IDW 插值均值 μ_IDW(k=5)
- `prior_unc`: 不确定度 u(p) = a1·d_kNN + a2·Std_kNN + a3·outside_hull
- `prior_gap`: 局部变化指标 Δμ = μ_IDW(k=5) - μ_IDW(k=2)

#### 2. 弱条件分支处理（Mu-REG 优化）

```python
# 训练时对 context_stec 进行随机 dropout（仅弱条件分支）
if weak_condition and self.training:
    context_mask = (role_type == 1)  # 识别 context 点
    dropout_mask = torch.rand(B, N, device=context_stec.device) > context_dropout_rate
    dropout_mask = dropout_mask | (~context_mask)  # 保留非 context 点
    context_stec = context_stec * dropout_mask.unsqueeze(-1).float()
```

**作用**: 训练弱条件分支，为 Guidance 机制做准备

#### 3. 地理坐标傅里叶编码

```python
# 多频率正弦特征（NeRF 风格）
pos_enc = self.pos_enc(coords)  # [B, N, dim]
x = x + pos_enc
```

#### 4. 角度傅里叶编码

```python
# 观测几何信息编码
angle_enc = self.angle_enc(angles)  # [B, N, dim]
x = x + angle_enc
```

#### 5. 系统 ID 嵌入

```python
# 区分不同卫星系统（GPS, GLONASS, Galileo, BDS）
sys_emb = self.system_embed(system_ids)  # [B, N, system_emb_dim]
sys_emb = self.system_proj(sys_emb)      # [B, N, dim]
x = x + sys_emb
```

#### 6. 角色类型编码

```python
# 区分 padding / context / target 点
role_emb = self.role_embed(role_type)  # [B, N, dim]
x = x + role_emb
```

#### 7. 时间步嵌入（条件）

```python
# DDPM 风格正弦编码 + MLP
t_emb = self.time_embed(t)  # [B, dim]
```

#### 8. Context 全局聚合（条件）

```python
# 对 context 点的 STEC 均值池化
ctx_mean = context_stec[context_mask].mean()  # 标量
ctx_emb = self.context_agg_mlp(ctx_mean)      # [B, dim]
```

#### 9. 合并条件

```python
# 用于 AdaLN 调制
cond = t_emb + ctx_emb  # [B, dim]
```

### 3.4 模型参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `dim` | 256 | 模型隐层维度 |
| `depth` | 6 | Transformer 层数 |
| `heads` | 8 | 注意力头数 |
| `mlp_ratio` | 4.0 | FFN 扩展比例 |
| `fourier_bands` | 64 | 地理坐标傅里叶频率数 |
| `angle_fourier_bands` | 32 | 角度傅里叶频率数 |
| `system_emb_dim` | 32 | 系统 ID 嵌入维度 |
| `time_emb_dim` | 256 | 时间步嵌入中间维度 |
| `max_seq_len` | 256 | RoPE 最大序列长度 |
| `dropout` | 0.0 | Dropout 概率 |

---

## 基础构建块

### 4.1 RMSNorm（均方根归一化）

**公式**:
$$\text{output} = \frac{x}{\text{RMS}(x) + \varepsilon} \cdot g$$
$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}$$

**特点**:
- 比 LayerNorm 更高效（无需计算均值）
- 现代大模型中广泛使用（LLaMA, GPT-4）

### 4.2 SwiGLU（门控前馈网络）

**公式**:
$$\text{output} = \text{Linear}_3\big(\text{SiLU}(\text{Linear}_1(x)) \odot \text{Linear}_2(x)\big)$$

**特点**:
- 门控机制增强非线性表达能力
- $\odot$ 表示逐元素乘法（门控）

### 4.3 STECAttention（多头自注意力）

**处理流程**:

```
输入 x: [B, N, dim]
    │
    ├─ 线性投影（无偏置）→ Q, K, V: [B, N, dim]
    │
    ├─ 拆分多头: [B, H, N, head_dim]
    │
    ├─ 对 Q, K 做 RMSNorm（QK-Norm）
    │
    ├─ 对 Q, K 应用 RoPE（旋转位置编码）
    │
    ├─ 缩放点积注意力（含 padding mask）
    │   Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    │
    └─ 重投影: [B, H, N, head_dim] → [B, N, dim]
```

**关键特性**:
- **QK-Norm**: 对 Q 和 K 做 RMSNorm，提升训练稳定性
- **RoPE**: 旋转位置编码，捕捉序列内相对位置关系
- **Padding Mask**: 忽略 padding 点，防止信息泄露

### 4.4 STECBlock（带 AdaLN 的 Transformer 块）

**AdaLN（自适应层归一化）机制**:

条件 MLP 输出 6 个参数：
```python
shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = adaLN_modulate(cond)
```

**块内计算结构**:

```python
# 注意力分支
x_norm = RMSNorm(x) * (1 + scale_attn) + shift_attn
x = x + gate_attn * Attention(x_norm)

# FFN 分支
x_norm = RMSNorm(x) * (1 + scale_ffn) + shift_ffn
x = x + gate_ffn * SwiGLU(x_norm)
```

**AdaLN 的作用**:
- **shift**: 偏移归一化后的特征
- **scale**: 缩放归一化后的特征
- **gate**: 控制残差连接的强度

---

## 位置编码模块

### 5.1 FourierPosEncoding（地理坐标编码）

**原理**: 对连续经纬度坐标做多频率正弦特征（NeRF 风格）

**公式**:
$$\text{feat} = [\sin(2\pi f_1 \cdot \text{lat}), \cos(2\pi f_1 \cdot \text{lat}), \ldots, \sin(2\pi f_k \cdot \text{lon}), \cos(2\pi f_k \cdot \text{lon})]$$

其中频率 $f_i = 2^i$，$i = 0, 1, \ldots, k-1$

**输出维度**: $4 \times \text{fourier\_bands}$，经线性投影到 `[B, N, dim]`

### 5.2 AngleFourierEncoding（角度编码）

**原理**: 与地理坐标编码类似，但用于方位角和高度角

**公式**:
$$\text{feat} = [\sin(2\pi f_1 \cdot \text{azimuth}), \cos(2\pi f_1 \cdot \text{azimuth}), \ldots]$$

**输出维度**: $4 \times \text{angle\_fourier\_bands}$，经线性投影到 `[B, N, dim]`

### 5.3 TimestepEmbedding（扩散时间步编码）

**原理**: DDPM 风格正弦编码 + 2 层 MLP

**公式**:
$$\text{sinusoidal}(t, i) = \begin{cases}
\sin(t / 10000^{2i/d}) & \text{if } i \text{ is even} \\
\cos(t / 10000^{2(i-1)/d}) & \text{if } i \text{ is odd}
\end{cases}$$

**输出**: `[B, dim]` 条件向量

### 5.4 RoPE（旋转位置编码）

**原理**: 通过复数旋转编码相对位置信息

**公式**:
$$\text{RoPE}(x, m) = x \cdot e^{im\theta}$$

其中 $m$ 是序列位置，$\theta$ 是预计算的频率

**特点**:
- 捕捉序列内相对位置关系
- 与绝对位置无关，支持外推
- 预计算频率，注册为 buffer（无梯度）

---

## 训练阶段数据流

```
原始 STEC 数据（历元文件）
    │
    ▼
数据集加载（STECEpochDataset）
    ├─ 确定性打乱 IPP 点
    ├─ 保留前 80% 作为训练点池
    └─ 归一化（coords, angles, stec）
    │
    ▼
DataLoader（collate_fn）
    ├─ Padding 到 batch 内最大长度
    └─ 生成 valid_mask
    │
    ▼
动态生成 context/target mask
    ├─ mask_ratio 随机采样自 [10%, 50%]
    └─ 从训练点池中随机抽取 target
    │
    ▼
构建条件均值 μ 和先验特征（build_mu_batch）
    ├─ Context 点: μ = mean(context STEC)
    ├─ Target 点: μ = IDW 插值
    └─ 先验特征: prior_mu, prior_unc, prior_gap（Mu-REG）
    │
    ▼
随机采样时间步 t ~ Uniform(1, T)
    │
    ▼
前向加噪（forward_sample_batch）
    xt = mu_bar(x0, t) + sigma(t) * noise
    │
    ▼
模型前向传播（双分支，Mu-REG 优化）
    ├─ 强条件分支: noise_pred_strong = Transformer(..., weak_condition=False)
    └─ 弱条件分支: noise_pred_weak = Transformer(..., weak_condition=True)
    │
    ▼
计算双分支损失（Mu-REG 优化）
    L_total = L_strong + λ_w·L_weak + λ_x·L_x0 + λ_j·L_jac
    │
    ▼
反向传播 & 参数更新
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
```

---

## 推理阶段数据流

```
输入数据
    ├─ Context 点（已知 STEC 值）
    └─ Target 点（未知位置）
    │
    ▼
构建条件均值 μ 和先验特征
    ├─ Context 点: μ = mean(context STEC)
    ├─ Target 点: μ = IDW 插值
    └─ 先验特征: prior_mu, prior_unc, prior_gap（Mu-REG）
    │
    ▼
初始化噪声状态
    x_T = stec.clone()
    x_T[target_mask] = mu[target_mask] + max_sigma * randn()
    │
    ▼
完整反向 SDE（循环 t = T, T-1, ..., 1）
    │
    ├─ 强条件分支预测噪声
    │   noise_pred_strong = Transformer(..., weak_condition=False)
    │
    ├─ Guidance 修正（Mu-REG 优化，可选）
    │   if use_guidance:
    │       noise_pred_weak = Transformer(..., weak_condition=True)
    │       delta_noise = noise_pred_strong - noise_pred_weak
    │
    │       # 时步自适应调度
    │       s_t = guidance_timestep_schedule(t)
    │
    │       # 空间自适应权重
    │       w = guidance_scale_max * s_t * exp(-beta * prior_unc)
    │
    │       # 应用 guidance
    │       noise_pred = noise_pred_strong + w * delta_noise
    │   else:
    │       noise_pred = noise_pred_strong
    │
    ├─ 估计 x0
    │   x0_pred = (x_t - mu - sigma(t)*noise_pred) / alpha(t) + mu
    │
    ├─ 计算 x_{t-1}
    │   if t > 1:
    │       mean_prev = mu + alpha(t-1) * (x0_pred - mu)
    │       x_{t-1} = mean_prev + sigma(t-1) * randn()
    │   else:
    │       x_{t-1} = x0_pred
    │
    └─ Context 点保持不变
        x_t[context_mask] = context_stec[context_mask]
    │
    ▼
输出去噪结果
    x0_pred = x_0
    │
    ▼
反归一化到 TECU 单位
    stec_orig = stec_norm * std_train + mean_train
    │
    ▼
计算评估指标
    MAE = mean(|pred - true|)
    RMSE = sqrt(mean((pred - true)^2))
```

---

## 关键设计思路

### 8.1 条件化策略

**空间条件**:
- Context 点作为空间锚点，提供局部观测约束
- IDW 插值生成平滑的条件均值 μ，引导扩散过程
- 无显式编码器-解码器，直接由 Transformer 完成预测

**时间条件**:
- 时间步嵌入提供扩散进度信息
- AdaLN 机制使时间步可以调制所有 Transformer 块的计算

**Mu-REG 优化（五阶段）**:
1. **显式先验特征**: 将 IDW 先验显式化为模型输入（prior_mu, prior_unc, prior_gap）
2. **双分支训练**: 强条件分支（100% context）+ 弱条件分支（30% dropout）
3. **Guidance 机制**: 利用双分支差异进行推理修正 Δε = ε_strong - ε_weak
4. **时步自适应**: 根据扩散时间步动态调整 guidance 强度 s(t)
5. **空间自适应**: 根据不确定度动态调整每个点的 guidance 权重 w = w_max·s(t)·exp(-β·u)

### 8.2 自适应归一化（AdaLN）

**优势**:
- 条件信息（时间步 + context 聚合）可以调制所有层的计算
- 门控机制（gate）实现可学习的残差强度控制
- 比简单的条件拼接更灵活、更强大

**实现**:
```python
# 条件 MLP 输出 6 个参数
shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = adaLN_modulate(cond)

# 调制归一化
x_norm = RMSNorm(x) * (1 + scale) + shift

# 门控残差
x = x + gate * layer(x_norm)
```

### 8.3 地理感知

**多尺度空间编码**:
- Fourier 编码捕捉多尺度空间结构（从局部到全局）
- RoPE 处理序列内相对位置关系
- 地理特征与语义特征分离建模

**观测几何编码**:
- 方位角和高度角的傅里叶编码
- 捕捉观测几何对 STEC 的影响

### 8.4 多星联合

**系统 ID 嵌入**:
- 通过可学习的嵌入区分不同卫星系统
- 共享模型参数，学习跨系统的共性特征
- 提升模型在稀疏观测区域的性能

**优势**:
1. 充分利用多源观测数据
2. 减少模型数量，简化部署
3. 提升泛化能力

### 8.5 掩码策略

**Padding 掩码**:
- 填充点的损失贡献为零
- 注意力掩码阻止填充点影响有效点的表示

**Context 固定**:
- 推理时 context 点值固定，不参与去噪更新
- 提供稳定的条件信息

### 8.6 与 JiT 的对应关系

| JiT 组件 | STEC 适配 | 说明 |
|---------|----------|------|
| `patch_embed` | 点特征嵌入（Linear） | 从图像 patch → 离散点 |
| `t_embedder` | TimestepEmbedding | 直接迁移 |
| `y_embedder`（类别） | Context 聚合嵌入 | 从类别查表 → 数据聚合 |
| `JiTBlock`（AdaLN） | STECBlock | 直接迁移并增强 |
| `final_linear` | 标量输出头（dim → 1） | 从 patch 像素 → 标量噪声 |
| RoPE | RoPE | 保留（序列位置） |
| 无 | FourierPosEncoding | 新增（地理位置） |
| 无 | AngleFourierEncoding | 新增（观测角度） |
| 无 | 系统 ID 嵌入 | 新增（多星联合） |
| 无 | 角色类型编码 | 新增（context/target 区分） |

---

## 总结

### 9.1 架构特点

- ✅ **条件扩散**: 基于 OU-SDE 的条件扩散框架
- ✅ **Transformer 主干**: 6 层 AdaLN-Transformer
- ✅ **多尺度编码**: 地理坐标 + 观测角度的傅里叶编码
- ✅ **多星联合**: 系统 ID 嵌入，支持多卫星系统
- ✅ **自适应归一化**: AdaLN 机制，条件调制所有层
- ✅ **空间感知**: IDW 插值构建条件均值
- ✅ **灵活性**: 支持变长样本，自动 padding
- ✅ **Mu-REG 优化**: 显式先验 + 双分支训练 + 自适应 Guidance

### 9.2 创新点

1. **离散点集扩散**: 从规则网格图像扩展到不规则离散点集
2. **IPP 点级建模**: 不依赖站点唯一性，支持多星联合
3. **空间条件构建**: IDW 插值生成平滑的条件均值
4. **多源信息融合**: 地理坐标 + 观测角度 + 系统 ID + 先验特征
5. **Mu-REG 优化**: 五阶段优化机制，提升预测精度和鲁棒性

### 9.3 性能优化

- **RMSNorm**: 比 LayerNorm 更高效
- **SwiGLU**: 门控 FFN，增强表达能力
- **QK-Norm**: 提升训练稳定性
- **预计算 RoPE**: 减少推理开销
- **双分支 Guidance**: 利用强弱条件差异修正预测
- **自适应权重**: 时步和空间双重自适应

---

**文档版本**: v3.0
**最后更新**: 2026-04-16
**适用代码版本**: Mu-REG 优化版本（五阶段完整实现）
