# 基于 REG 条件引导的 STEC 条件扩散优化技术路线（保留原均值回归加噪思路）

## 1. 文档目标

本文给出一套**面向不规则离散 IPP 点集的 STEC 条件扩散优化方案**。  
该方案**不改变你当前的“均值回归 + 条件均值 `μ` 引导的加噪扩散框架”**，只在以下几个方面进行增强：

1. 将当前仅隐式参与扩散过程的先验/条件均值 `μ`，改为**显式输入模型**；
2. 引入**强条件 / 弱条件双分支**，构造适合当前任务的 guidance 信号；
3. 借鉴 REG 思想，在**采样阶段**对噪声预测进行**梯度校正引导**；
4. 将 guidance 强度做成**时步自适应 + 空间自适应**；
5. 保留你现有的数据划分、训练 masking、外部独立测试站点和最终 PPP 应用链路。

---

## 2. 当前算法基础：哪些内容保持不变

## 2.1 数据与评估框架保持不变

你当前的数据组织方式可以概括为：

- `model_stations/`：训练 + 验证用建模站点
- `val_stations/`：独立外部测试站点
- 每个历元文件作为一个样本
- 训练阶段在训练点池中动态随机抽取 target
- 验证阶段固定前 80% 为 context、后 20% 为 target
- 最终测试阶段使用 `model_stations/` 作为 context，`val_stations/` 作为独立 target

这套框架本身是合理的，应继续保留。尤其是 `val_stations/` 作为独立空间外测集，对检验模型空间泛化能力非常重要。

## 2.2 当前扩散框架保持不变

你当前方法的核心扩散形式是一个**以条件均值 `μ` 为中心的 OU 型均值回归扩散过程**。  
设：

- x<sub>0</sub>：真实 STEC
- x<sub>t</sub>：第 t 个扩散步的随机变量
- μ：条件均值 / baseline / prior mean
- T：总扩散步数
- α(t)：均值回归系数
- σ(t)：第 t 步对应的噪声标准差
- ε ~ N(0, I)：标准高斯噪声

则你当前前向扩散为：

$$
x_t = \bar\mu(x_0,t) + \sigma(t)\,\varepsilon
$$

其中条件均值轨迹写为：

$$
\bar\mu(x_0,t)=\mu + \alpha(t)\,(x_0-\mu)
$$

你的当前反向估计式为：

$$
\hat x_0
=
\frac{x_t-\mu-\sigma(t)\,\hat\varepsilon_t}{\alpha(t)+\epsilon}
+\mu
$$

这里：

- ε̂<sub>t</sub> 是网络预测噪声
- ε 是数值稳定项，不是高斯噪声

**本次优化中，上述扩散主框架保持不变。**

---

## 3. 对当前模型的准确判断

## 3.1 当前模型已经是条件扩散模型

你当前模型并不是无条件扩散。  
因为噪声预测网络已经显式接收了：

- `noisy_stec`
- `context_stec`
- `coords`
- `angles`
- `system_ids`
- `role_type`
- `valid_mask`
- `t`

同时，`context_stec` 的聚合结果还进入了 AdaLN 条件支路。  
因此，从模型结构上看，你当前已经是一个**条件扩散模型**。

## 3.2 当前模型的主要不足：`μ` 仍然是“隐式条件”，不是“显式条件”

虽然当前模型已经用了条件信息，但有一个关键问题：

- **观测条件（context）是显式输入网络的**
- **条件均值 `μ` 主要只参与 SDE 轨迹与采样初始化**
- **`μ` 并没有作为一个独立的 query 级先验特征显式送入 Transformer**

也就是说，当前方法更准确地说是：

> **显式观测条件 + 隐式 prior mean 扩散**

而不是：

> **显式 prior mean 条件化扩散**

这就是本次最关键的结构性优化点。

---

## 4. REG 论文对你最有价值的思想

## 4.1 REG 的核心出发点

REG 论文指出：  
常规 guidance 在理论上并不是最优形式。  
理想情况下，采样阶段应该基于**joint scaling** 对应的目标来修正噪声预测。

设：

- y：条件变量
- R<sub>0</sub>(x<sub>0</sub>, y)：最终样本处的 reward / guidance target
- E<sub>t</sub>(x<sub>t</sub>, y)：从当前时刻 t 往未来完整去噪链传播后的“期望奖励”

则理论上的最优更新形式应写为：

$$
\bar\varepsilon_t^{\star}
=
\varepsilon_t
-
\sqrt{1-\bar\alpha_t}\nabla_{x_t}\log E_t(x_t,y)
$$

但 E<sub>t</sub> 不可直接计算，因此现有 guidance 实际上通常使用一个近似量 R<sub>t</sub>，即：

$$
\bar\varepsilon_t
=
\varepsilon_t
-
\sqrt{1-\bar\alpha_t}\nabla_{x_t}\log R_t(x_t,y)
$$

REG 的核心改进就是：  
在现有 guidance 基础上，再引入一个与雅可比相关的校正项，使它更接近理论上的最优引导方向。

## 4.2 REG 的实用意义

对你的任务而言，REG 不是要求你改变训练目标去“重新定义一个图像语义 reward”，而是提供一种思路：

1. 先构造两个不同强度的条件分支；
2. 用它们之间的差异形成 guidance；
3. 在采样阶段对 guidance 做 Jacobian-style 校正；
4. 从而提高 target/query 点恢复的稳定性与精度。

---

## 5. 面向当前任务的优化总路线

本次建议的优化方案命名为：

# **Mu-REG-STEC**
**Mean-reversion Prior-Explicit Rectified-Guided STEC Diffusion**

即：

> **保留原均值回归加噪机制 + 显式先验输入 + 强弱条件双分支 + REG 校正采样**

这个方案分为 5 个改进模块：

1. **保留原 `μ`-centered OU 扩散**
2. **将 `μ` 从隐式先验变成显式输入特征**
3. **构造强条件 / 弱条件双分支**
4. **在采样阶段引入 REG 风格的噪声梯度校正**
5. **让 guidance 强度随时步与空间先验质量自适应**

---

## 6. 改进 1：保留旧的均值回归扩散主干

## 6.1 条件均值 `μ` 的定义保持旧方案

对于一个样本中所有点，继续按你当前方式构造 `μ`：

### 对 context 点
设 context 点集合为 C，其真实观测 STEC 为 x(c)，则：

$$
\mu(c)=\frac{1}{|C|}\sum_{c'\in C}x(c')
$$

即：  
context 点仍采用 **context 全局均值广播** 的旧定义。

### 对 target 点
设 target 点为 q，则：

$$
\mu(q)=
\frac{\sum_{i=1}^{k} w_i(q)x(c_i)}
{\sum_{i=1}^{k} w_i(q)}
$$

其中：

- c<sub>i</sub>：距 q 最近的第 i 个 context 点
- x(c<sub>i</sub>)：对应 context 点的 STEC
- w<sub>i</sub>(q) = 1 / (d<sub>i</sub>(q) + δ)<sup>p</sup>
- d<sub>i</sub>(q)：target 点到第 i 个 context 点的空间距离
- p：IDW 幂次
- k：近邻数量
- δ：防止除零的小量

因此，本优化方案**不改变 `build_mu_batch()` 的基本思想**，只是把它进一步用于显式条件建模。

## 6.2 前向扩散仍围绕 `μ` 进行

前向扩散继续写成：

$$
x_t = \mu + \alpha_t(x_0-\mu)+\sigma_t\varepsilon
$$

其中：

- x<sub>0</sub>：目标点真实 STEC
- x<sub>t</sub>：第 t 步加噪后的状态
- μ：由上一步构造的条件均值
- α<sub>t</sub>：均值回归强度
- σ<sub>t</sub>：噪声标准差
- ε：标准高斯噪声

这就是你明确要求保留的“旧的均值回归加噪扩散思路”。

---

## 7. 改进 2：把 `μ` 变成显式模型输入，而不仅仅留在 SDE 里

## 7.1 现有问题

当前网络虽然能看到：

- noisy STEC
- context STEC
- 坐标
- 角度
- 系统 ID
- role type
- 时间步

但它并**看不到**：

- 当前点的 `μ` 到底是多少
- 该 `μ` 是否可靠
- 不同插值先验之间是否一致

这意味着网络虽然在一个“围绕 `μ` 的轨迹”上训练，但它对 `μ` 本身的感知是不直接的。

## 7.2 推荐新增的显式先验特征

对于每个点 p，建议构造如下附加特征：

### 1. 主先验均值
$$
\mu^{\text{IDW}}(p)
$$

即当前 `build_mu_batch()` 使用的 baseline。

### 2. 第二先验均值（可选）
$$
\mu^{\text{RBF}}(p)
$$

即使用 RBF 或 TPS 得到的备用平滑基线。

### 3. 先验差异
$$
\Delta\mu(p)=\mu^{\text{IDW}}(p)-\mu^{\text{RBF}}(p)
$$

它反映：
- 当前点上不同插值器是否一致
- 若差异很大，往往说明该点处 prior 不稳定

### 4. 先验不确定度
定义：

$$
u(p)=
a_1 d_{\text{kNN}}(p)
+
a_2 \operatorname{Std}_{\text{kNN}}(p)
+
a_3 \mathbf{1}_{\text{outside hull}}(p)
$$

其中：

- d<sub>kNN</sub>(p)：到最近邻 context 点的平均距离
- Std<sub>kNN</sub>(p)：邻域 context STEC 的加权方差
- 1<sub>outside hull</sub>(p)：是否位于 context 凸包外
- a<sub>1</sub>, a<sub>2</sub>, a<sub>3</sub>：权重系数

再把 u(p) 归一化到 `[0,1]`。

## 7.3 新的点级输入特征设计

原始点嵌入可以从：

$$
[\,x_t,\ x_{\text{ctx}}\,]
$$

扩展为：

$$
[\,x_t,\ x_{\text{ctx}},\ \mu^{\text{IDW}},\ u,\ \Delta\mu\,]
$$

其中：

- x<sub>t</sub>：当前点在第 t 步的状态  
  - 对 target 是当前采样值
  - 对 context 是真实值
- x<sub>ctx</sub>：仅 context 点填真实 STEC，target 点为 0
- μ<sup>IDW</sup>：当前点的 prior mean
- u：prior uncertainty
- Δμ：多 prior 分歧

再叠加你当前已有的：

- 坐标编码 `coords`
- 角度编码 `angles`
- 系统 ID 嵌入 `system_ids`
- 角色嵌入 `role_type`
- 时间步嵌入 `t`

这样网络不仅知道“当前值是多少”，还知道“当前值围绕哪个均值在演化”。

---

## 8. 改进 3：构造强条件 / 弱条件双分支，而不是无条件分支

## 8.1 为什么不能直接照搬 CFG 的“有条件 / 无条件”

图像 CFG 常用：

- 条件分支：ε<sub>θ</sub>(x<sub>t</sub>, t, y)
- 无条件分支：ε<sub>θ</sub>(x<sub>t</sub>, t, ∅)

但在你的任务里，如果完全去掉 context，模型就失去了：

- 空间观测锚点
- 本历元下的局部场信息
- 与当前卫星 / 当前几何相关的有效约束

因此，对你的空间插值任务，更合理的不是“有条件 vs 无条件”，而是：

> **强条件 vs 弱条件**

## 8.2 强条件分支 y<sup>+</sup>

强条件分支使用完整信息：

- 全部 context 点
- 原始 `μ_IDW`
- `u`
- `Δμ`
- 完整几何信息
- 完整系统 ID / role 信息

记其输出为：

$$
\varepsilon^{+}_t
=
\varepsilon_\theta(x_t,t,y^+)
$$

## 8.3 弱条件分支 y<sup>-</sup>

弱条件分支刻意降质，用来构造 guidance 差分信号。  
可采用以下降质方式之一或组合：

### 方式 A：额外 context dropout
对强条件中的 context 再随机丢弃一部分，例如 30%–50%。

### 方式 B：更弱的 prior
例如：
- 把 `IDW(k=5)` 改成 `IDW(k=2)`
- 或仅用局部均值
- 或不使用 `RBF` 辅助 prior

### 方式 C：去掉一部分质量特征
例如：
- 把 `u` 置零
- 把 `Δμ` 置零

于是弱条件分支输出为：

$$
\varepsilon^{-}_t
=
\varepsilon_\theta(x_t,t,y^-)
$$

## 8.4 guidance 差分信号

定义：

$$
g_t = \varepsilon^{+}_t - \varepsilon^{-}_t
$$

解释：

- 若 g<sub>t</sub> 很大，说明完整条件与弱条件对噪声方向判断差异明显；
- 该差异本身就是一个任务相关的 guidance 信号。

---

## 9. 改进 4：在旧 OU-SDE 反向采样中引入 REG 校正

## 9.1 当前反向采样基础

你当前对 x<sub>t</sub> 的反向估计是：

$$
\hat x_0
=
\frac{x_t-\mu-\sigma_t\hat\varepsilon_t}{\alpha_t+\epsilon}
+\mu
$$

这里：

- ε̂<sub>t</sub> 是当前用于采样的噪声预测
- μ 在当前步视为固定条件量，不对 x<sub>t</sub> 求导

## 9.2 不加 REG 时的 guidance 形式

如果只使用普通 guidance，可写成：

$$
\hat\varepsilon_t^{\text{guide}}
=
\varepsilon_t^{+}
+
w_t(\varepsilon_t^{+}-\varepsilon_t^{-})
$$

其中：

- ε<sub>t</sub><sup>+</sup>：强条件噪声预测
- ε<sub>t</sub><sup>-</sup>：弱条件噪声预测
- w<sub>t</sub>：guidance strength

## 9.3 REG 风格校正项

借鉴 REG 的思想，在你的 OU-SDE 里，可把校正项近似写为：

$$
J_t
=
\frac{\partial \left(\mathbf{1}^\top\varepsilon_t^{+}\right)}
{\partial x_t}
$$

其中：

- 1<sup>T</sup>ε<sub>t</sub><sup>+</sup> 表示把所有 target/query 点上的噪声预测分量求和
- J<sub>t</sub> 是对当前状态 x<sub>t</sub> 的梯度

由于：

$$
\hat x_0
=
\frac{x_t-\mu-\sigma_t\varepsilon_t}{\alpha_t}+\mu
$$

对 x<sub>t</sub> 求导得到：

$$
\frac{\partial \hat x_0}{\partial x_t}
=
\frac{1}{\alpha_t}
\left(
I-\sigma_t\frac{\partial \varepsilon_t}{\partial x_t}
\right)
$$

其中：

- I 为单位矩阵
- μ 在当前步视为常量，因此其导数为 0

与 REG 论文类似，可把 1/α<sub>t</sub> 吸收到 guidance scale 中，于是得到适合当前任务的近似校正项：

$$
c_t = 1 - \sigma_t J_t
$$

## 9.4 最终 REG 引导噪声

于是，建议在采样阶段使用：

$$
\varepsilon_t^{\text{REG}}
=
\varepsilon_t^{+}
+
w_t\,\big(\varepsilon_t^{+}-\varepsilon_t^{-}\big)\odot c_t
$$

其中：

- ⊙ 表示逐元素乘
- c<sub>t</sub> 是 REG correction term

注意：  
**该 REG 校正只对 target/query 点应用，不对 context 点应用。**

因为：

- context 点是已知观测
- 推理阶段 context 必须保持固定
- 不应被 guidance 更新

---

## 10. 改进 5：guidance 强度做成时步和空间联合自适应

## 10.1 为什么不能用固定常数

在你的任务里，guidance 强度若固定不变，容易出现两类问题：

1. 在稀疏区 / 外推区过度相信 prior，导致过平滑；
2. 在后期小噪声阶段 guidance 过强，破坏局部细节恢复。

因此建议把 guidance scale 写成：

$$
w_{t,j}
=
w_{\max}\,s(t)\,\exp(-\beta u_j)
$$

其中：

- w<sub>max</sub>：最大 guidance scale
- s(t)：时步调度函数
- u<sub>j</sub>：第 j 个 target/query 点的 prior uncertainty
- β：不确定度抑制系数

## 10.2 时步调度函数建议

可取：

$$
s(t)=\sin^2\left(\frac{\pi t}{2T}\right)
$$

它的直观意义是：

- 前期和中期 guidance 较强，用来确定整体结构
- 后期 guidance 自动减弱，保留局部细节

## 10.3 空间自适应意义

若某个 query 点的 prior uncertainty 很大，则：

$$
u_j \uparrow
\quad\Rightarrow\quad
\exp(-\beta u_j) \downarrow
$$

即 guidance 自动减弱。  
这样就不会在：

- 边界区
- 凸包外
- 超稀疏区

把结果生硬拉回到不可信的 IDW 先验上。

---

## 11. 训练阶段的优化后目标函数

## 11.1 主损失：强条件噪声预测损失

设真实噪声为 ε，则：

$$
L_{\text{full}}
=
\operatorname{MSE}(\varepsilon_t^{+},\varepsilon)
$$

## 11.2 弱条件辅助损失

$$
L_{\text{weak}}
=
\operatorname{MSE}(\varepsilon_t^{-},\varepsilon)
$$

## 11.3 `x_0` 重建损失

由强条件噪声预测可得到：

$$
\hat x_0^{+}
=
\frac{x_t-\mu-\sigma_t\varepsilon_t^{+}}{\alpha_t+\epsilon}
+\mu
$$

然后增加一个重建项：

$$
L_{x_0}
=
\operatorname{Huber}(\hat x_0^{+},x_0)
$$

## 11.4 Jacobian 稳定项（可选）

为了防止 REG correction 过激，增加：

$$
L_{\text{jac}}
=
\left\|
\operatorname{clip}(\sigma_t J_t,\,-c,\ c)
\right\|_2^2
$$

其中：

- c 为截断阈值

## 11.5 总损失

$$
L
=
L_{\text{full}}
+
\lambda_w L_{\text{weak}}
+
\lambda_x L_{x_0}
+
\lambda_j L_{\text{jac}}
$$

推荐初值：

- λ<sub>w</sub> = 0.5
- λ<sub>x</sub> = 0.2
- λ<sub>j</sub> = 10<sup>-4</sup>

---

## 12. 完整训练流程

## 12.1 训练阶段步骤

对一个样本（单历元文件）执行：

1. 从训练点池中动态划分 context / target；
2. 用当前旧方案构造 `μ`；
3. 随机采样时间步 t；
4. 按旧均值回归加噪公式构造 x<sub>t</sub>；
5. 构造强条件分支输入 y<sup>+</sup>；
6. 构造弱条件分支输入 y<sup>-</sup>；
7. 分别预测 ε<sub>t</sub><sup>+</sup> 和 ε<sub>t</sub><sup>-</sup>；
8. 计算 L<sub>full</sub>, L<sub>weak</sub>, L<sub>x_0</sub>, L<sub>jac</sub>；
9. 反向传播并更新参数。

---

## 13. 完整推理流程

## 13.1 推理输入

推理时输入包括：

- context 点：来自建模站点的已知 STEC
- query 点：可以是  
  1) 测试站点坐标  
  2) 规则格网点坐标

## 13.2 推理步骤

### Step 1：构造 prior mean
用旧方案构造：

- context 点：全局 context 均值
- query 点：IDW 插值均值

### Step 2：构造显式 prior 特征
为每个 query 构造：

- μ<sup>IDW</sup>
- Δμ
- u

### Step 3：初始化 target/query 状态
对 target/query 点：

$$
x_T = \mu + \sigma_T z,\qquad z\sim\mathcal N(0,I)
$$

context 点保持真实值不变。

### Step 4：反向采样
对 t = T, T-1, ..., 1：

1. 计算强条件预测 ε<sub>t</sub><sup>+</sup>
2. 若当前时步启用 guidance，则计算弱条件预测 ε<sub>t</sub><sup>-</sup>
3. 计算 Jacobian 校正项 J<sub>t</sub>
4. 得到
   $$
   \varepsilon_t^{REG}
   =
   \varepsilon_t^{+}
   +
   w_t(\varepsilon_t^{+}-\varepsilon_t^{-})\odot (1-\sigma_tJ_t)
$$
5. 用 ε<sub>t</sub><sup>REG</sup> 回代估计 x̂<sub>0</sub>
6. 计算 x<sub>t-1</sub>
7. 强制 context 点恢复为真实观测值

### Step 5：输出结果
最终得到：

- 测试站点预测值
- 或规则格网点处的 TEC / STEC 产品

---

## 14. 适合你当前任务的最小改动实施顺序

如果希望在现有工程基础上低风险落地，建议按以下顺序实施：

### 第一阶段：最小必做改动
1. 保留当前 `μ`-centered SDE 不变
2. 在 Transformer 输入中新增：
   - `prior_mu`
   - `prior_unc`
   - `prior_gap`
3. 在 AdaLN 条件分支中增加 prior 聚合信息

### 第二阶段：双分支条件训练
1. 增加 `full condition`
2. 增加 `weak condition`
3. 训练时同时回归 ε<sub>t</sub><sup>+</sup> 与 ε<sub>t</sub><sup>-</sup>

### 第三阶段：采样期 guidance
先加入普通 guidance：
$$
\varepsilon_t^{guide}
=
\varepsilon_t^{+}
+
w_t(\varepsilon_t^{+}-\varepsilon_t^{-})
$$
### 第四阶段：加入 REG 校正
再把上式升级为：
$$
\varepsilon_t^{REG}
=
\varepsilon_t^{+}
+
w_t(\varepsilon_t^{+}-\varepsilon_t^{-})\odot(1-\sigma_tJ_t)
$$
### 第五阶段：加入 uncertainty-aware adaptive guidance
最后再引入：
$$
w_{t,j}=w_{\max}s(t)\exp(-\beta u_j)
$$

---

## 15. 相比当前方法，最核心的改进点

## 改进点 1：保持原扩散主干不变
你要求保留的旧思路被完整保留：

- 仍然是围绕 `μ` 的均值回归加噪
- 不改成残差扩散
- 不改变当前 SDE 的物理逻辑

## 改进点 2：`μ` 从隐式条件变成显式条件
当前 `μ` 只在扩散轨迹里。  
优化后 `μ` 会直接进入 Transformer 输入和 AdaLN 条件支路。

## 改进点 3：构造适合空间插值任务的 guidance
不做“有条件 / 无条件”，而做：

- 强条件
- 弱条件

更适合不规则离散观测场景。

## 改进点 4：将 REG 迁移到你当前 OU-SDE
用 Jacobian 修正噪声 guidance，而不是只线性放大 guidance scale。

## 改进点 5：guidance 不再固定，而是时步和空间联合自适应
避免在：

- 高不确定区域
- 边界区
- 后期小噪声阶段

出现过强 guidance。

---

## 16. 最终建议：最值得先落地的版本

如果只做一版最实用、最贴近你当前工程的增强版，我建议优先落地下面这个版本：

# **Mu-REG-STEC (Lite)**

包含：

1. 保留当前 `μ`-centered OU-SDE
2. 显式加入 `prior_mu / prior_unc / prior_gap`
3. 训练 `full / weak` 双条件分支
4. 采样时仅对 target/query 点使用 REG guidance
5. guidance 只在中间时步区间启用
6. 最终仍用 `val_stations/` 做外测
7. 再单独做规则格网产品和 PPP 增强验证

这是因为该版本：

- 与当前代码兼容度最高
- 不改动你的核心扩散主线
- 可以最大程度吸收 REG 的有益思想
- 又不会把工程复杂度抬得过高

---

## 17. 一句话总结

这次优化的核心不是推翻你当前的旧扩散思路，而是在**保留原“以 `μ` 为中心的均值回归加噪扩散”**的前提下，把附件 3 中 REG 的思想转化为：

> **显式 prior mean 条件化 + 强弱条件双分支 + 采样阶段噪声梯度校正引导**

这条路线最符合你当前 STEC 空间建模任务的特点，也最适合在你现有工程上逐步落地。
