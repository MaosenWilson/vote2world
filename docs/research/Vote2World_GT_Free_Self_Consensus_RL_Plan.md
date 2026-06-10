# Vote2World：面向动作条件视觉世界模型的无 GT 自共识强化学习后训练方案

> **文档定位**：研究计划与最小可发表版本设计  
> **当前范围**：单步下一帧预测、RT-1 类机器人环境、自回归视频世界模型、GRPO、无未来 GT 奖励  
> **暂不纳入**：horizon guard、多步长时回溯、部署样本回流、扩散兼容 RL、联合训练验证器  
> **版本**：v0.1

---

## 0. 执行摘要

现有视觉世界模型通常使用未来真实观测作为监督信号。RLVR-World 进一步将解码预测帧与未来 GT 帧之间的感知误差作为可验证奖励，并使用 GRPO 对世界模型进行后训练。该方法有效，但在目标域适配或真实部署阶段，未来 GT 帧通常无法作为即时奖励使用。

本方案借鉴 TTRL 的核心思想：**不依赖外部真实标签，而是利用同一输入条件下的多次采样结果构造自标注奖励**。但是，图像属于连续高维空间，无法直接使用数学答案式的离散多数投票。因此，本方案提出：

> **在动作条件下，对候选预测帧相对于当前帧的视觉变化量进行邻域投票，以局部共识构造二值伪奖励，再使用 GRPO 微调视觉世界模型。**

最小方法链路为：

\[
\boxed{
(o_{t-m+1:t}, a_t)
\rightarrow
K \text{ 个候选下一帧}
\rightarrow
\text{变化量特征}
\rightarrow
\text{邻域共识投票}
\rightarrow
R_k \in \{0,1\}
\rightarrow
\text{GRPO}
}
\]

为了防止模型通过复制当前帧获得虚假的高共识，本方案增加两个轻量机制：

1. **低置信度组拒绝更新**：当候选结果过于分散，或全部样本奖励相同时，跳过该组；
2. **静态复制过滤**：当动作幅度明显非零而预测变化接近零时，强制将奖励置为 0。

第一篇工作不追求全面解决长时视频预测，而聚焦验证一个清晰问题：

> **在不读取未来 GT 帧的条件下，仅依靠模型自身候选输出之间的动作条件共识，是否可以改善下一帧预测质量，并缩小与 GT-based RLVR 后训练之间的差距？**

---

## 1. 暂定题目

### 1.1 推荐题目

**Vote2World: Ground-Truth-Free Self-Consensus Reinforcement Learning for Action-Conditioned Visual World Models**

中文可写为：

**Vote2World：面向动作条件视觉世界模型的无 GT 自共识强化学习方法**

### 1.2 备选题目

- **SC-VWM: Self-Consensus Reinforcement Learning for Ground-Truth-Free Visual World Model Adaptation**
- **Transition-Consensus RL for Ground-Truth-Free Visual World Model Post-Training**
- **Learning Visual World Models from Their Own Predictions: Action-Conditioned Consensus Reinforcement Learning**

### 1.3 命名建议

在实验尚未完成前，优先使用：

> **Ground-Truth-Free Post-Training**

而不是立即使用：

> **Test-Time Reinforcement Learning**

原因是：若初始实验仍使用离线 RT-1 数据中的无标签状态—动作输入做适配，更严谨的描述是“无未来 GT 后训练”。只有完成流式目标域适配或部署阶段在线更新实验后，才适合强化 `test-time adaptation` 叙事。

---

## 2. 研究背景

### 2.1 世界模型的基本目标

视觉世界模型学习动作干预下的状态转移：

\[
p_\theta(o_{t+1} \mid o_{t-m+1:t}, a_t),
\]

其中：

- \(o_{t-m+1:t}\)：历史视觉观测；
- \(a_t\)：当前动作；
- \(o_{t+1}\)：下一时刻视觉观测；
- \(\theta\)：世界模型参数。

在 RT-1 类机器人环境中，模型需要根据当前桌面场景、机械臂状态与动作向量，预测下一帧画面。

### 2.2 为什么传统 MLE 不够？

自回归视觉世界模型通常优化视觉 token 的 next-token prediction：

\[
\mathcal{L}_{\mathrm{MLE}}
=
-
\sum_n
\log
p_\theta
\left(
z_n \mid z_{<n}, c_t
\right),
\]

其中：

\[
c_t = (o_{t-m+1:t}, a_t).
\]

但 token-level MLE 与最终任务需求并不完全一致。模型可能获得较低的 token loss，却仍然出现：

- 画面重复；
- 动态区域不响应动作；
- 背景稳定但机械臂运动错误；
- 预测帧模糊；
- 目标物体漂移。

因此，需要更接近最终预测质量的优化目标。

---

## 3. 现有工作与切入点

### 3.1 RLVR-World：使用 GT 的视觉世界模型强化学习后训练

RLVR-World 将不同模态的世界模型统一为序列建模任务，并将任务相关预测指标作为可验证奖励。对于视频世界模型，其流程为：

\[
\text{当前状态与动作}
\rightarrow
\text{采样一组未来视觉 token}
\rightarrow
\text{解码为候选预测帧}
\rightarrow
\text{分别与未来 GT 帧比较}
\rightarrow
\text{GRPO 微调}.
\]

其视觉分支使用自回归视频世界模型，并以解码预测帧与真实未来观测之间的误差构造奖励。单步预测奖励可以概括为：

\[
R_k^{\mathrm{RLVR}}
=
-
\mathcal{L}
\left(
\hat{o}_{t+1}^{(k)},
o_{t+1}^{\mathrm{gt}}
\right).
\]

对于多步预测：

\[
R_k^{\mathrm{RLVR}}
=
-
\sum_{h=1}^{H}
\mathcal{L}
\left(
\hat{o}_{t+h}^{(k)},
o_{t+h}^{\mathrm{gt}}
\right).
\]

RLVR-World 的价值在于直接优化预测指标，而不是继续依赖 MLE 代理目标。但它仍需要未来 GT。

### 3.2 TTRL：不依赖 GT 的多数投票奖励

TTRL 面向无标签数学推理任务。对于同一道题，模型重复采样多个回答：

\[
y^{(1)},y^{(2)},\ldots,y^{(K)}.
\]

随后通过多数投票得到伪标签：

\[
\tilde{y}
=
\operatorname{Mode}
\left(
y^{(1)},y^{(2)},\ldots,y^{(K)}
\right),
\]

并定义二值奖励：

\[
R_k^{\mathrm{TTRL}}
=
\mathbf{1}
\left[
y^{(k)}=\tilde{y}
\right].
\]

其核心思想是：

\[
\boxed{
\text{不依赖真实标签，利用模型自身先验构造奖励}
}
\]

### 3.3 为什么不能直接将 TTRL 搬到图像空间？

数学答案是离散符号，可以直接统计众数。预测图像是连续高维对象，即使两张图语义一致，其像素通常也不会完全相同。

直接计算：

\[
\operatorname{Mode}
\left(
\hat{o}_{t+1}^{(1)},
\hat{o}_{t+1}^{(2)},
\ldots,
\hat{o}_{t+1}^{(K)}
\right)
\]

没有实际意义。

此外，像素级相似也可能被静态背景主导。若模型简单复制当前帧：

\[
\hat{o}_{t+1}^{(k)} \approx o_t,
\]

多个候选结果可能高度一致，却并不代表预测正确。

### 3.4 本工作的核心切入点

将离散答案多数投票推广为：

\[
\boxed{
\text{连续视觉变化空间中的邻域共识投票}
}
\]

并进一步强调：

\[
\boxed{
\text{不对原始帧投票，而对动作引起的视觉变化量投票}
}
\]

---

## 4. 问题定义

### 4.1 输入

第一篇工作只做单步预测。

给定历史帧与动作：

\[
c_t
=
\left(
o_{t-m+1:t},
a_t
\right).
\]

其中：

- \(o_t \in \mathbb{R}^{H \times W \times 3}\)；
- \(a_t \in \mathbb{R}^{d_a}\)；
- \(m\) 为历史窗口长度。

### 4.2 基础世界模型

采用预训练自回归视觉世界模型：

\[
\pi_\theta
\left(
z_{t+1}
\mid
c_t
\right),
\]

其中 \(z_{t+1}\) 是下一帧的离散视觉 token 序列。

视觉解码器记为：

\[
D(\cdot).
\]

### 4.3 适配阶段约束

在无 GT 适配阶段，程序只允许读取：

\[
\left(
o_{t-m+1:t},
a_t
\right).
\]

严格屏蔽：

\[
o_{t+1}^{\mathrm{gt}}.
\]

未来 GT 只能在独立评估阶段使用。

---

## 5. 方法总览

对于同一个条件输入 \(c_t\)，世界模型随机采样 \(K\) 次：

\[
\hat{z}_{t+1}^{(k)}
\sim
\pi_\theta(\cdot \mid c_t),
\qquad
k=1,\ldots,K.
\]

解码得到候选下一帧：

\[
\hat{o}_{t+1}^{(k)}
=
D
\left(
\hat{z}_{t+1}^{(k)}
\right).
\]

使用冻结视觉编码器 \(\phi(\cdot)\) 提取候选帧与当前帧特征，并计算候选状态变化量：

\[
\Delta_t^{(k)}
=
\phi
\left(
\hat{o}_{t+1}^{(k)}
\right)
-
\phi(o_t).
\]

随后：

1. 计算候选变化量之间的距离；
2. 统计每个候选的邻域支持率；
3. 根据阈值构造二值奖励；
4. 过滤静态复制候选；
5. 拒绝低置信度组；
6. 使用 GRPO 更新世界模型。

完整链路为：

```text
历史观测帧 o_{t-m+1:t} + 当前动作 a_t
                    │
                    ▼
          预训练视觉世界模型 πθ
                    │
      ┌─────────────┼─────────────┐
      ▼             ▼             ▼
 候选 token 1    候选 token 2    ... 候选 token K
      │             │                    │
      ▼             ▼                    ▼
 Visual Decoder  Visual Decoder       Visual Decoder
      │             │                    │
      ▼             ▼                    ▼
 候选帧 1         候选帧 2            候选帧 K
      │             │                    │
      └─────────────┼────────────────────┘
                    ▼
             冻结视觉编码器 φ
                    │
                    ▼
        计算变化量 Δ_t^(1),...,Δ_t^(K)
                    │
                    ▼
           计算两两距离矩阵 d_ij
                    │
                    ▼
        邻域投票：计算支持率 v_k
                    │
                    ▼
   静态复制过滤 + 低置信度组拒绝更新
                    │
                    ▼
            二值奖励 R_k ∈ {0,1}
                    │
                    ▼
              GRPO 参数更新
```

---

## 6. 核心方法设计

### 6.1 候选采样

对于每个条件输入：

\[
c_t
=
(o_{t-m+1:t},a_t),
\]

生成 \(K\) 个候选下一帧 token 序列：

\[
\hat{z}_{t+1}^{(1)},
\hat{z}_{t+1}^{(2)},
\ldots,
\hat{z}_{t+1}^{(K)}.
\]

再分别解码：

\[
\hat{o}_{t+1}^{(k)}
=
D
\left(
\hat{z}_{t+1}^{(k)}
\right).
\]

建议初始使用：

\[
K=16.
\]

后续消融：

\[
K \in \{8,16,32\}.
\]

### 6.2 冻结视觉特征提取器

定义冻结视觉编码器：

\[
\phi(\cdot).
\]

第一版优先使用世界模型自身 tokenizer 的 encoder，以降低系统复杂度。

可选对比：

- 模型自身 visual encoder；
- DINOv2；
- patch-level DINOv2；
- 仅取高变化 patch 的局部特征。

第一版不要联合训练 \(\phi\)，否则奖励空间会随训练漂移，增加不稳定性。

### 6.3 从图像共识改为变化量共识

基础图像特征为：

\[
f_t
=
\phi(o_t),
\]

\[
\hat{f}_{t+1}^{(k)}
=
\phi
\left(
\hat{o}_{t+1}^{(k)}
\right).
\]

定义候选状态变化量：

\[
\Delta_t^{(k)}
=
\hat{f}_{t+1}^{(k)}
-
f_t.
\]

该设计的动机是：

- 抵消大量静态背景；
- 突出动作引起的局部状态变化；
- 降低“复制当前帧即可获得高共识”的风险；
- 更贴近世界模型的状态转移建模目标。

### 6.4 候选间距离

优先使用 cosine distance：

\[
d_{ij}
=
1-
\cos
\left(
\Delta_t^{(i)},
\Delta_t^{(j)}
\right).
\]

其中：

\[
\cos(u,v)
=
\frac{u^\top v}
{\|u\|_2\|v\|_2+\varepsilon}.
\]

可选消融使用归一化欧氏距离：

\[
d_{ij}^{\mathrm{L2}}
=
\frac{
\left\|
\Delta_t^{(i)}-\Delta_t^{(j)}
\right\|_2
}{
\sqrt{d}
}.
\]

### 6.5 邻域支持率

对于第 \(k\) 个候选，统计其邻域支持率：

\[
v_k
=
\frac{1}{K-1}
\sum_{\substack{j=1\\j\neq k}}^{K}
\mathbf{1}
\left[
d_{kj}
\leq
\tau_d
\right].
\]

其中：

- \(d_{kj}\)：候选 \(k\) 与候选 \(j\) 的变化量距离；
- \(\tau_d\)：邻域距离阈值；
- \(v_k \in [0,1]\)：候选 \(k\) 的局部共识程度。

### 6.6 多值投票而不是唯一众数

本方案不强制选出唯一伪标签。多个局部高密度模式都可以获得正奖励。

例如：

```text
簇 A：机械臂向右移动较明显，共 6 个候选
簇 B：机械臂向右移动幅度略小，共 5 个候选
异常项：背景漂移或明显形变，共 5 个候选
```

若两个簇都满足支持率阈值，则簇 A 与簇 B 中的候选均可获得正奖励。

这比唯一众数更适合视觉预测的多模态特性。

---

## 7. 二值奖励

### 7.1 基础共识奖励

定义：

\[
R_k^{\mathrm{cons}}
=
\mathbf{1}
\left[
v_k
\geq
\tau_v
\right].
\]

其中 \(\tau_v\) 是最低支持率阈值。

建议初始使用：

\[
\tau_v = 0.5.
\]

消融范围：

\[
\tau_v \in \{0.3,0.5,0.7\}.
\]

### 7.2 静态复制过滤

仅使用共识奖励可能诱导模型生成静态帧。对非零动作，需要拒绝变化量过小的候选。

定义动作幅度：

\[
m_a
=
\|a_t\|_2.
\]

定义候选变化幅度：

\[
m_k
=
\left\|
\Delta_t^{(k)}
\right\|_2.
\]

静态复制门控为：

\[
G_k^{\mathrm{static}}
=
\mathbf{1}
\left[
m_a \leq \tau_a
\;\lor\;
m_k \geq \tau_m
\right].
\]

其含义是：

- 若动作本身接近零，允许预测帧保持基本不变；
- 若动作明显非零，但视觉变化接近零，则候选奖励置为 0。

### 7.3 最终样本级奖励

\[
\boxed{
R_k
=
R_k^{\mathrm{cons}}
\cdot
G_k^{\mathrm{static}}
}
\]

即：

\[
\boxed{
R_k
=
\mathbf{1}
\left[
v_k\geq\tau_v
\right]
\cdot
\mathbf{1}
\left[
\|a_t\|_2\leq\tau_a
\;\lor\;
\|\Delta_t^{(k)}\|_2\geq\tau_m
\right]
}
\]

满足：

\[
R_k \in \{0,1\}.
\]

---

## 8. 组级置信度与拒绝更新

### 8.1 为什么需要拒绝更新？

若候选组过于分散，无法形成可靠共识，则不应强行构造伪奖励。

此外，当组内奖励全部相同时：

\[
R_1=R_2=\cdots=R_K,
\]

GRPO 无法获得有效相对优势。

### 8.2 组级置信度

定义：

\[
C_t
=
\max_k v_k.
\]

当：

\[
C_t < \tau_c
\]

时，跳过该组。

### 8.3 奖励退化检测

定义：

\[
S_t
=
\sum_{k=1}^{K}R_k.
\]

当：

\[
S_t \in \{0,K\}
\]

时，跳过该组。

### 8.4 最终组级有效性条件

\[
M_t
=
\mathbf{1}
\left[
C_t\geq\tau_c
\right]
\cdot
\mathbf{1}
\left[
0<S_t<K
\right].
\]

仅当：

\[
M_t=1
\]

时执行 GRPO 更新。

---

## 9. 阈值设定

### 9.1 不应使用 GT 调整阈值

训练阶段不能根据：

\[
o_{t+1}^{\mathrm{gt}}
\]

选择 \(\tau_d\)、\(\tau_v\)、\(\tau_m\)。

否则会破坏无 GT 设定。

### 9.2 无标签 warm-up 统计

从适配输入中随机选取一批条件，使用初始模型采样候选组，统计：

\[
\mathcal{D}
=
\left\{
d_{ij}
\right\}.
\]

设置：

\[
\tau_d
=
Q_q(\mathcal{D}),
\]

其中 \(Q_q\) 表示第 \(q\) 分位数。

建议初始设置：

\[
q=0.3.
\]

消融：

\[
q \in \{0.2,0.3,0.4\}.
\]

### 9.3 动作幅度分桶的自适应阈值

固定 \(\tau_d\) 可能对大幅动作不公平。动作越大，合理候选之间的差异可能越明显。

定义动作幅度：

\[
m_a=\|a_t\|_2.
\]

将样本划分为动作幅度桶：

\[
b(a_t)\in\{1,2,\ldots,B\}.
\]

对每个桶单独计算：

\[
\tau_d^{(b)}
=
Q_q
\left(
\mathcal{D}^{(b)}
\right).
\]

最终使用：

\[
d_{ij}
\leq
\tau_d^{(b(a_t))}.
\]

建议：

- MVP：固定阈值；
- 完整版本：动作幅度分桶阈值；
- 消融：固定阈值 vs. 动作条件自适应阈值。

---

## 10. GRPO 更新

对于有效样本组，得到奖励：

\[
R_1,R_2,\ldots,R_K.
\]

计算组内均值：

\[
\mu_R
=
\frac{1}{K}
\sum_{k=1}^{K}
R_k.
\]

计算标准差：

\[
\sigma_R
=
\sqrt{
\frac{1}{K}
\sum_{k=1}^{K}
(R_k-\mu_R)^2
}.
\]

相对优势为：

\[
\hat{A}_k
=
\frac{
R_k-\mu_R
}{
\sigma_R+\varepsilon
}.
\]

对于候选 token 序列中的 token \(n\)，定义概率比：

\[
\rho_{k,n}(\theta)
=
\frac{
\pi_\theta
\left(
\hat{z}_{k,n}
\mid
c_t,\hat{z}_{k,<n}
\right)
}{
\pi_{\theta_{\mathrm{old}}}
\left(
\hat{z}_{k,n}
\mid
c_t,\hat{z}_{k,<n}
\right)
}.
\]

使用 clipped objective：

\[
\mathcal{L}_{\mathrm{GRPO}}
=
-
\frac{1}{K}
\sum_{k=1}^{K}
\frac{1}{|\hat{z}^{(k)}|}
\sum_n
\min
\left[
\rho_{k,n}\hat{A}_k,
\operatorname{clip}
\left(
\rho_{k,n},
1-\epsilon,
1+\epsilon
\right)
\hat{A}_k
\right]
+
\beta
D_{\mathrm{KL}}
\left(
\pi_\theta
\Vert
\pi_{\mathrm{ref}}
\right).
\]

其中：

- \(\pi_{\theta_{\mathrm{old}}}\)：采样时的行为策略；
- \(\pi_\theta\)：待更新策略；
- \(\pi_{\mathrm{ref}}\)：冻结参考策略；
- \(\epsilon\)：裁剪系数；
- \(\beta\)：KL 正则权重。

KL 正则非常重要。由于自共识奖励可能强化错误模式，需要限制模型偏离原始世界模型过快。

---

## 11. 最小算法伪代码

```python
def consensus_reward(
    current_frame,
    action,
    decoded_predictions,
    visual_encoder,
    tau_dist,
    tau_vote,
    tau_action,
    tau_motion,
    tau_group,
    eps=1e-8,
):
    """
    decoded_predictions: List[Tensor], length K
    returns:
        rewards: Tensor[K], binary rewards
        valid_group: bool
        diagnostics: dict
    """

    # 1. Frozen visual features
    current_feat = visual_encoder(current_frame).detach()
    pred_feats = [
        visual_encoder(pred).detach()
        for pred in decoded_predictions
    ]

    # 2. Transition features
    deltas = [
        pred_feat - current_feat
        for pred_feat in pred_feats
    ]

    # 3. Pairwise cosine distances
    K = len(deltas)
    distances = pairwise_cosine_distance(deltas, eps=eps)

    # 4. Neighborhood support ratios
    votes = []
    for k in range(K):
        support = 0
        for j in range(K):
            if j == k:
                continue
            support += int(distances[k, j] <= tau_dist)
        votes.append(support / (K - 1))

    # 5. Binary consensus reward
    consensus_reward = [
        int(v >= tau_vote)
        for v in votes
    ]

    # 6. Static-copy rejection
    action_mag = l2_norm(action)
    motion_mag = [
        l2_norm(delta)
        for delta in deltas
    ]

    static_gate = [
        int(action_mag <= tau_action or mag >= tau_motion)
        for mag in motion_mag
    ]

    rewards = [
        r_cons * gate
        for r_cons, gate in zip(consensus_reward, static_gate)
    ]

    # 7. Group-level abstention
    group_conf = max(votes)
    reward_sum = sum(rewards)

    valid_group = (
        group_conf >= tau_group
        and reward_sum > 0
        and reward_sum < K
    )

    diagnostics = {
        "group_confidence": group_conf,
        "reward_positive_ratio": reward_sum / K,
        "skip": not valid_group,
        "vote_scores": votes,
        "motion_magnitudes": motion_mag,
    }

    return rewards, valid_group, diagnostics
```

---

## 12. 数据划分与防止信息泄漏

### 12.1 推荐划分

| 数据子集 | 可读取内容 | 用途 |
|---|---|---|
| Base pre-training set | 完整轨迹 | 训练或直接使用已有 base checkpoint |
| Unlabeled adaptation set | 历史观测、当前动作 | 自共识 GRPO 后训练 |
| Held-out validation set | 完整轨迹 | 选择训练步数与诊断 |
| Held-out test set | 完整轨迹 | 最终报告指标 |

### 12.2 adaptation 阶段必须屏蔽的内容

禁止奖励模块访问：

\[
o_{t+1}^{\mathrm{gt}}.
\]

程序层面建议：

- adaptation dataloader 不返回 future frame；
- 或将 future frame 字段显式删除；
- reward module 接口中不允许出现 `ground_truth` 参数；
- 单独编写单元测试检查 GT 不可访问。

### 12.3 论文中建议使用的表述

> During self-consensus RL adaptation, future observations are strictly withheld and never accessed by the reward function. Ground-truth future frames are used only for held-out evaluation.

---

## 13. 实验路线

### 13.1 阶段 0：复现官方基线

优先使用 RLVR-World 官方 RT-1 单步模型作为起点。

需要完成：

1. 下载 RT-1 单步 tokenizer；
2. 下载 RT-1 单步 base world model；
3. 运行官方预测与评估脚本；
4. 复现 base 指标；
5. 运行官方 GT-based RLVR checkpoint；
6. 确认评估流程能够观察到性能差异。

此阶段不修改模型结构。

### 13.2 阶段 1：离线验证自共识是否具有代理价值

在不更新模型的条件下：

1. 随机选择 500—1000 个适配输入；
2. 每个输入采样 \(K=16\) 个候选下一帧；
3. 计算三种共识分数：
   - pixel-space consensus；
   - image-feature consensus；
   - transition-feature consensus；
4. 在分析脚本中使用 held-out GT 计算候选真实质量；
5. 检查自共识与真实质量之间的相关性。

这一阶段是生死线。若自共识与真实质量没有显著相关性，不应直接进行大规模 GRPO。

### 13.3 阶段 2：最小奖励版本

先只实现：

\[
R_k
=
\mathbf{1}
\left[
v_k\geq\tau_v
\right].
\]

并加入：

\[
S_t \in \{0,K\}
\Rightarrow
\text{skip update}.
\]

进行小规模 smoke test：

- 训练 20—50 个 GRPO step；
- 检查 reward ratio；
- 检查 skip rate；
- 检查 KL；
- 检查 entropy；
- 可视化候选预测。

### 13.4 阶段 3：加入变化量共识

比较：

| 方法 | 特征对象 |
|---|---|
| Pixel Consensus | 原始像素 |
| Image-Feature Consensus | 候选预测帧特征 |
| Transition Consensus | 候选预测帧相对当前帧的变化量 |

预期：

\[
\text{Transition Consensus}
>
\text{Image-Feature Consensus}
>
\text{Pixel Consensus}.
\]

### 13.5 阶段 4：加入静态复制过滤

加入：

\[
G_k^{\mathrm{static}}
=
\mathbf{1}
\left[
\|a_t\|_2\leq\tau_a
\;\lor\;
\|\Delta_t^{(k)}\|_2\geq\tau_m
\right].
\]

重点检查：

- 非零动作条件下 repetition rate 是否下降；
- 模型是否出现过度运动；
- 图像质量是否改善；
- 动态区域是否更加合理。

### 13.6 阶段 5：完整消融

完成：

- group size；
- 特征空间；
- binary reward vs. soft reward；
- 固定阈值 vs. 动作条件自适应阈值；
- 有无 abstention；
- 有无 static gate；
- Consensus-SFT vs. Consensus-GRPO。

---

## 14. 关键验证：伪奖励是否可信？

### 14.1 真实质量分数

GT 不进入训练，但可用于 held-out 分析。

对候选 \(k\)，定义：

\[
q_k^{\mathrm{gt}}
=
-
\operatorname{LPIPS}
\left(
\hat{o}_{t+1}^{(k)},
o_{t+1}^{\mathrm{gt}}
\right).
\]

也可以使用：

\[
q_k^{\mathrm{gt}}
=
-
\operatorname{MSE}
\left(
\hat{o}_{t+1}^{(k)},
o_{t+1}^{\mathrm{gt}}
\right).
\]

### 14.2 Spearman 相关性

统计：

\[
\rho
=
\operatorname{SpearmanCorr}
\left(
v_k,
q_k^{\mathrm{gt}}
\right).
\]

需要回答：

> 共识程度越高的候选，是否通常更接近真实下一帧？

### 14.3 高质量候选识别精度

将组内 GT 指标排名前 \(30\%\) 的候选定义为高质量样本：

\[
Y_k^{\mathrm{good}}
=
\mathbf{1}
\left[
q_k^{\mathrm{gt}}
\in
\text{Top-30\%}
\right].
\]

统计：

\[
\operatorname{Precision}
=
P
\left(
Y_k^{\mathrm{good}}=1
\mid
R_k=1
\right).
\]

### 14.4 组内 pairwise ranking accuracy

对任意候选对 \((i,j)\)，判断：

\[
v_i>v_j
\]

是否对应：

\[
q_i^{\mathrm{gt}}>q_j^{\mathrm{gt}}.
\]

定义：

\[
\operatorname{RankAcc}
=
\frac{
\sum_{i\neq j}
\mathbf{1}
\left[
(v_i-v_j)
(q_i^{\mathrm{gt}}-q_j^{\mathrm{gt}})
>0
\right]
}{
\sum_{i\neq j}
\mathbf{1}
\left[
v_i\neq v_j
\right]
}.
\]

### 14.5 决策条件

建议仅当 transition consensus 满足以下条件时进入大规模 GRPO：

- Spearman \(\rho\) 稳定为正；
- 高质量候选 precision 显著优于随机水平；
- pairwise ranking accuracy 高于随机水平；
- 对不同动作幅度子集均成立；
- 静态复制样本不会系统性获得高奖励。

---

## 15. 主实验对比

| 方法 | 奖励是否使用未来 GT | 目的 |
|---|---:|---|
| Base World Model | 否 | 原始预训练模型 |
| Continued MLE | 是 | 额外监督训练对比 |
| GT-based RLVR-World | 是 | 近似性能上界 |
| Random Binary Reward + GRPO | 否 | 排除“任意 RL 都有效” |
| Pixel Consensus + GRPO | 否 | 验证像素共识不足 |
| Image-Feature Consensus + GRPO | 否 | 验证特征空间共识 |
| Transition Consensus + GRPO | 否 | 核心方法 |
| Transition Consensus + Abstention | 否 | 验证拒绝更新机制 |
| Transition Consensus + Abstention + Static Gate | 否 | 完整方法 |
| Consensus-SFT | 否 | 验证 GRPO 相比伪标签 SFT 的必要性 |

核心目标不是强行超过使用 GT 的 RLVR-World，而是：

\[
\boxed{
\text{在完全不读取未来 GT 的条件下，显著优于 Base，并缩小与 GT-based RLVR 的差距}
}
\]

---

## 16. 评估指标

### 16.1 标准下一帧预测指标

使用 held-out test GT 计算：

\[
\operatorname{MSE},
\quad
\operatorname{PSNR},
\quad
\operatorname{SSIM},
\quad
\operatorname{LPIPS}.
\]

### 16.2 重复率

定义：

\[
\operatorname{RepRate}
=
\frac{1}{N}
\sum_{n=1}^{N}
\mathbf{1}
\left[
d
\left(
\hat{o}_{t+1}^{(n)},
o_t^{(n)}
\right)
<
\tau_{\mathrm{rep}}
\right].
\]

重点报告非零动作条件下：

\[
\operatorname{RepRate}_{a\neq 0}.
\]

### 16.3 动作响应性

对同一当前状态，分别输入真实动作与打乱动作：

\[
a_t,
\qquad
a_t^{\mathrm{shuffle}}.
\]

生成：

\[
\hat{o}_{t+1}(a_t),
\qquad
\hat{o}_{t+1}(a_t^{\mathrm{shuffle}}).
\]

定义动作敏感性：

\[
S_{\mathrm{action}}
=
d
\left(
\phi
\left(
\hat{o}_{t+1}(a_t)
\right),
\phi
\left(
\hat{o}_{t+1}(a_t^{\mathrm{shuffle}})
\right)
\right).
\]

若模型忽略动作，则：

\[
S_{\mathrm{action}}
\approx 0.
\]

### 16.4 动态区域指标

全图指标容易被静态背景主导。建议在评估阶段增加：

\[
\operatorname{LPIPS}_{\mathrm{robot}},
\quad
\operatorname{SSIM}_{\mathrm{robot}},
\quad
\operatorname{LPIPS}_{\mathrm{object}},
\quad
\operatorname{SSIM}_{\mathrm{object}}.
\]

第一版可以仅将 mask 用于评估，不放入奖励函数。

### 16.5 训练诊断指标

每个 step 建议记录：

- mean reward；
- positive reward ratio；
- skip rate；
- group confidence；
- pairwise distance mean/std；
- KL divergence；
- entropy；
- gradient norm；
- repetition rate；
- action sensitivity；
- 静态门控触发比例；
- 不同动作幅度桶的 reward ratio。

---

## 17. 消融实验矩阵

| 消融维度 | 设置 |
|---|---|
| Group size | 8、16、32 |
| 投票空间 | pixel、image feature、transition feature |
| 特征编码器 | tokenizer encoder、DINOv2 |
| 距离函数 | cosine、L2 |
| Reward | binary、soft score |
| 邻域阈值 | 20%、30%、40% 分位数 |
| 支持率阈值 \(\tau_v\) | 0.3、0.5、0.7 |
| 组置信度阈值 \(\tau_c\) | 0.3、0.5、0.7 |
| 阈值形式 | 固定阈值、动作幅度分桶阈值 |
| Abstention | 有、无 |
| Static Gate | 有、无 |
| Optimization | Consensus-SFT、Consensus-GRPO |
| 上界 | GT-based RLVR-World |

---

## 18. 失败模式与应对策略

### 18.1 错误共识

问题：

\[
\text{候选一致}
\neq
\text{候选正确}.
\]

多个候选可能一致地犯错。

应对：

- held-out GT 上验证共识分数与真实质量相关性；
- 设置低置信度拒绝更新；
- 使用 KL 限制策略漂移；
- 按动作幅度分桶统计；
- 将 GT-based RLVR 作为上界。

### 18.2 静态帧坍缩

问题：

\[
\hat{o}_{t+1}^{(k)}
\approx
o_t.
\]

模型可能通过输出静态帧获得高共识。

应对：

- 在 transition feature 上投票；
- 使用静态复制门控；
- 单独报告 \(\operatorname{RepRate}_{a\neq0}\)；
- 进行 action shuffle 测试。

### 18.3 奖励全部相同

问题：

\[
R_1=R_2=\cdots=R_K.
\]

此时：

\[
\sigma_R=0,
\]

GRPO 无有效排序信号。

应对：

\[
S_t \in \{0,K\}
\Rightarrow
\text{skip update}.
\]

### 18.4 多模态合理未来被误伤

问题：唯一多数伪标签会压制少数但合理的未来。

应对：

- 不选唯一 medoid；
- 使用邻域支持率；
- 允许多个局部稠密簇同时获得正奖励。

### 18.5 视觉编码器不敏感

问题：全局特征可能忽视机械臂局部小幅运动。

应对：

- 比较 tokenizer encoder 与 DINOv2；
- 增加 patch-level 特征；
- 仅聚合变化最大的局部 patch；
- 在评估阶段报告 robot/object masked metrics。

### 18.6 阈值过敏感

问题：固定阈值对不同动作幅度、不同任务分布可能不稳定。

应对：

- 使用无标签 warm-up 统计；
- 使用分位数阈值；
- 使用动作幅度分桶阈值；
- 报告阈值敏感性曲线。

---

## 19. 推荐的创新点表述

### 创新点一：无未来 GT 的视觉世界模型 RL 后训练

现有 GT-based RLVR 依赖未来观测构造奖励。本工作仅基于模型自身候选输出构造训练信号：

\[
\boxed{
\text{future-GT-free visual world model adaptation}
}
\]

### 创新点二：连续视觉空间中的多模态邻域投票

将 TTRL 的离散多数投票推广至连续视觉预测空间，并允许多个合理局部模式同时获得奖励：

\[
\boxed{
\text{neighborhood consensus instead of single-mode majority voting}
}
\]

### 创新点三：动作条件变化量共识

不直接比较候选预测帧，而比较相对于当前帧的视觉变化量：

\[
\Delta_t^{(k)}
=
\phi
\left(
\hat{o}_{t+1}^{(k)}
\right)
-
\phi(o_t).
\]

从而减少背景主导与静态复制捷径：

\[
\boxed{
\text{vote on predicted transitions, not raw frames}
}
\]

### 辅助稳定机制

- 低置信度组拒绝更新；
- 静态复制过滤；
- KL 正则；
- 动作幅度自适应阈值。

---

## 20. 论文结构建议

### 20.1 Introduction

依次回答：

1. 视觉世界模型需要更接近最终预测质量的训练目标；
2. RLVR-World 使用 GT 指标奖励有效，但未来 GT 在目标域适配时不可用；
3. TTRL 说明模型自身多次采样可以构造无标签奖励；
4. 图像不能直接多数投票；
5. 提出动作条件变化量邻域共识；
6. 使用二值奖励与 GRPO 完成无 GT 后训练。

### 20.2 Related Work

建议分为：

- Visual world models；
- Autoregressive video world models；
- RL post-training for generative models；
- RLVR-World；
- TTRL and self-labeled RL；
- Consensus learning and self-training。

### 20.3 Method

建议结构：

1. Problem formulation；
2. Candidate rollout sampling；
3. Transition-space neighborhood voting；
4. Binary consensus reward；
5. Confidence abstention；
6. Static-copy rejection；
7. GRPO optimization。

### 20.4 Experiments

建议结构：

1. Setup；
2. Can self-consensus predict GT quality?；
3. Main GT-free adaptation results；
4. Comparison with GT-based RLVR；
5. Ablations；
6. Failure analysis；
7. Qualitative visualization。

---

## 21. 摘要草稿

> Visual world models are typically trained or post-trained with access to ground-truth future observations. However, such supervision is unavailable during target-domain adaptation or deployment. Inspired by test-time reinforcement learning with self-labeled rewards, we propose Vote2World, a ground-truth-free reinforcement learning framework for action-conditioned visual world models. Given the same observation history and action, the world model repeatedly samples multiple candidate next-frame predictions. Instead of comparing predictions against future ground truth, we construct binary pseudo-rewards through neighborhood voting in a frozen visual transition space. Specifically, we measure the consensus of predicted feature changes relative to the current observation, allowing multiple plausible future modes to receive positive rewards. To prevent static-frame collapse and noisy self-reinforcement, we further introduce action-aware static-copy rejection and confidence-based group abstention. The resulting rewards are used to post-train an autoregressive visual world model with group relative policy optimization. Experiments on RT-1-style robot manipulation environments evaluate whether self-consensus rewards correlate with true prediction quality and whether ground-truth-free adaptation improves perceptual fidelity, action responsiveness, and repetition artifacts while narrowing the gap to GT-based RLVR post-training.

---

## 22. 最小可发表版本

第一篇建议严格控制为：

\[
\boxed{
\text{RT-1 单步预测}
+
\text{iVideoGPT / RLVR-World 单步 base}
+
\text{GT-free transition consensus}
+
\text{binary reward}
+
\text{GRPO}
+
\text{abstention}
+
\text{static gate}
}
\]

暂不加入：

- horizon-aware reward；
- 多步长时优化；
- 轨迹回溯；
- 动态验证器；
- 联合训练视觉编码器；
- diffusion-compatible RL；
- DIAMOND；
- Vid2World；
- 部署闭环；
- VLM/VLA 打分器。

---

## 23. 推荐初始超参数

以下参数仅用于首轮实验，不应视为最终配置。

| 参数 | 初始值 | 消融范围 |
|---|---:|---:|
| Group size \(K\) | 16 | 8、16、32 |
| 邻域阈值分位数 \(q\) | 0.30 | 0.20、0.30、0.40 |
| 支持率阈值 \(\tau_v\) | 0.50 | 0.30、0.50、0.70 |
| 组置信度阈值 \(\tau_c\) | 0.50 | 0.30、0.50、0.70 |
| 距离函数 | cosine | cosine、L2 |
| 特征编码器 | tokenizer encoder | tokenizer encoder、DINOv2 |
| Reward | binary | binary、soft |
| Tokenizer | frozen | 不更新 |
| Policy reference | frozen base model | 固定 |
| KL coefficient | 沿用 RLVR-World 初始值 | 小范围搜索 |
| 训练步数 | 先 20—50 step smoke test | 再逐步扩大 |

---

## 24. 工程实施清单

### 24.1 阶段 0：环境与官方基线

- [ ] 克隆 RLVR-World 官方仓库；
- [ ] 获取 `vid_wm` 模块；
- [ ] 下载 RT-1 单步 tokenizer；
- [ ] 下载 RT-1 单步 base checkpoint；
- [ ] 下载 RT-1 单步 RLVR checkpoint；
- [ ] 运行官方 inference；
- [ ] 复现 MSE、PSNR、SSIM、LPIPS；
- [ ] 确认候选采样与 visual decoder 流程；
- [ ] 确认 tokenizer 在 RLVR 微调阶段冻结。

### 24.2 阶段 1：候选缓存与离线诊断

- [ ] 选择 500—1000 个输入；
- [ ] 每个输入采样 \(K=16\) 个候选；
- [ ] 保存 token、解码帧、action、当前帧；
- [ ] 保存 GT，仅供独立分析脚本使用；
- [ ] 提取 image feature；
- [ ] 提取 transition feature；
- [ ] 计算 pairwise distance；
- [ ] 计算 consensus score；
- [ ] 计算与 GT 质量的相关性；
- [ ] 生成可视化样例。

### 24.3 阶段 2：奖励模块

- [ ] 实现 cosine distance；
- [ ] 实现邻域支持率；
- [ ] 实现 binary reward；
- [ ] 实现全 0 / 全 1 skip；
- [ ] 实现 group confidence skip；
- [ ] 实现静态复制 gate；
- [ ] 实现日志；
- [ ] 编写无 GT 泄漏单元测试。

### 24.4 阶段 3：GRPO smoke test

- [ ] 运行 20—50 step；
- [ ] 检查 reward ratio；
- [ ] 检查 skip rate；
- [ ] 检查 KL；
- [ ] 检查 entropy；
- [ ] 检查梯度；
- [ ] 可视化训练前后候选；
- [ ] 确认没有静态帧坍缩。

### 24.5 阶段 4：主实验

- [ ] Base；
- [ ] Continued MLE；
- [ ] GT-based RLVR；
- [ ] Random Reward；
- [ ] Pixel Consensus；
- [ ] Image-Feature Consensus；
- [ ] Transition Consensus；
- [ ] Transition Consensus + Abstention；
- [ ] Transition Consensus + Abstention + Static Gate；
- [ ] Consensus-SFT。

### 24.6 阶段 5：写作材料

- [ ] 方法总图；
- [ ] 自共识与 GT 质量相关性图；
- [ ] 主实验表；
- [ ] 消融表；
- [ ] reward ratio / skip rate 曲线；
- [ ] 动作幅度分桶分析；
- [ ] 重复率分析；
- [ ] 成功案例；
- [ ] 错误共识案例；
- [ ] 静态复制案例。

---

## 25. Codex 执行指令草案

```text
目标：
在 RLVR-World 官方 vid_wm 单步 RT-1 基线上，实现 GT-free transition-consensus binary reward，并用于 GRPO 后训练。

约束：
1. 不修改 visual tokenizer；
2. 首先只做 single-step prediction；
3. adaptation dataloader 不得向 reward module 返回 future ground-truth frame；
4. future GT 只能由独立 evaluation / analysis 脚本读取；
5. 保留原始 GT-based reward，作为对照；
6. 新增日志：reward_positive_ratio、skip_rate、group_confidence、pairwise_distance_mean/std、static_gate_ratio、KL、entropy。

执行顺序：
A. 检查服务器是否已存在 RLVR-World 项目与 RT-1 checkpoint；
B. 若不存在则克隆官方仓库并下载单步 tokenizer、single-step-base、single-step-rlvr；
C. 运行官方单步 inference 与 evaluation，记录 base 和 RLVR 指标；
D. 定位 vid_wm 中 reward_fn、rollout sampling、visual decoder、GRPO trainer 接口；
E. 新增 consensus_reward.py：
   - frozen visual encoder feature extraction
   - transition delta feature
   - pairwise cosine distance
   - neighborhood support ratio
   - binary reward
   - group abstention
   - static-copy gate
F. 新增 analysis_proxy_quality.py：
   - 采样 K=16 候选
   - 比较 pixel / image-feature / transition-feature consensus
   - 仅在分析脚本读取 future GT
   - 计算 Spearman、Precision@positive、PairwiseRankAcc
G. 新增 smoke-test 配置：
   - group_size=16
   - 训练 20—50 steps
   - tokenizer frozen
   - reference policy frozen
H. 输出目录：
   outputs/
     baseline_eval/
     proxy_quality_analysis/
     grpo_smoke_test/
     visualizations/
     logs/
I. 在任何大规模训练前，先输出 proxy-quality 报告。
```

---

## 26. 结果判定标准

### 26.1 可以继续推进的信号

- transition consensus 与 GT 质量存在稳定正相关；
- transition consensus 优于 pixel consensus；
- 正奖励比例不接近 0 或 1；
- skip rate 可控；
- KL 没有快速爆炸；
- 非零动作 repetition rate 没有上升；
- GRPO 后 held-out LPIPS、SSIM 或 MSE 至少有一项稳定改善；
- 定性样例显示机械臂或目标物体变化更合理。

### 26.2 需要暂停并调整的信号

- 共识分数与 GT 质量无相关；
- 静态帧获得大量正奖励；
- skip rate 长期超过较高比例；
- 所有奖励趋于一致；
- KL 快速增加；
- 模型忽略动作；
- 感知指标改善但动态区域指标变差；
- 仅背景更稳定，机械臂预测没有改善。

### 26.3 调整优先级

若实验失败，按以下顺序调整：

1. 从 raw image feature 切换至 transition feature；
2. 加入 static gate；
3. 使用 patch-level feature；
4. 按动作幅度分桶设置阈值；
5. 增强 abstention；
6. 使用 DINOv2 替代 tokenizer encoder；
7. 增加 action-shuffle contrastive gate；
8. 最后才考虑训练独立验证器。

---

## 27. 可选扩展：结果不足时再加入

### 27.1 Action-shuffle 对比门控

构造负动作：

\[
a_t^{-}
=
\operatorname{Shuffle}(a_t).
\]

分别生成：

\[
\hat{o}_{t+1}(a_t),
\qquad
\hat{o}_{t+1}(a_t^-).
\]

若两者变化量几乎相同，则说明模型可能忽视动作。

定义：

\[
G_k^{\mathrm{action}}
=
\mathbf{1}
\left[
d
\left(
\Delta_t^{(k)}(a_t),
\Delta_t^{(k)}(a_t^-)
\right)
\geq
\tau_{\mathrm{act}}
\right].
\]

最终奖励可扩展为：

\[
R_k
=
R_k^{\mathrm{cons}}
\cdot
G_k^{\mathrm{static}}
\cdot
G_k^{\mathrm{action}}.
\]

该机制会增加采样成本，因此不放入 MVP。

### 27.2 Soft reward 消融

使用连续支持率：

\[
R_k^{\mathrm{soft}}
=
v_k.
\]

或：

\[
R_k^{\mathrm{soft}}
=
v_k
\cdot
G_k^{\mathrm{static}}.
\]

主线仍建议使用 binary reward，以保持方法简洁并贴近 TTRL。

### 27.3 Patch-level transition consensus

将全局特征替换为 patch 特征：

\[
\phi(o)
=
\left[
\phi_1(o),
\phi_2(o),
\ldots,
\phi_P(o)
\right].
\]

对变化最大的 top-\(q\) patch 聚合：

\[
\Delta_t^{(k)}
=
\operatorname{AggregateTopQ}
\left(
\phi_p(\hat{o}_{t+1}^{(k)})
-
\phi_p(o_t)
\right).
\]

该方案可能更适合机械臂局部运动。

---

## 28. 与后续工作的衔接

本方案刻意不加入 horizon guard，但为后续工作保留接口。

第一篇：

\[
\text{单步 GT-free self-consensus GRPO}
\]

第二篇可扩展为：

\[
\text{多步 diffusion world model}
+
\text{horizon-aware reward}
+
\text{一致性验证器}
+
\text{回溯重生成}
+
\text{部署样本回流}.
\]

两篇工作的逻辑关系为：

| 工作 | 核心问题 |
|---|---|
| 第一篇 | 没有未来 GT 时，模型能否依靠自身多次采样形成有效奖励？ |
| 第二篇 | 在长时多帧扩散世界模型中，如何控制误差累积并执行可验证回溯？ |

---

## 29. 相关工作竞争态势

### 29.1 必须对比的工作

#### TTRL

核心作用：证明多数投票可以为无标签 RL 构造奖励。

#### RLVR-World

核心作用：证明视觉世界模型可以使用解码预测指标作为 verifiable reward，并通过 GRPO 后训练。

#### iVideoGPT

核心作用：提供自回归视觉世界模型与视觉 tokenization 基础。

### 29.2 需要关注的新近工作

#### Distribution-Aware Reward Estimation for Test-Time Reinforcement Learning

该方向指出单一 majority outcome 可能丢失非多数但正确的候选，并造成奖励偏差。本方案使用邻域支持率而不是唯一众数，可作为视觉连续空间中的自然回应。

#### Persistent Robot World Models: Stabilizing Multi-Step Rollouts via Reinforcement Learning

该工作针对机器人扩散世界模型的多步 rollout 稳定性，比较同一 rollout 状态下的多个候选未来，并使用 GT 视觉保真奖励强化更优候选。其重点是多步扩散模型与长时稳定性。本方案应明确区分：

\[
\boxed{
\text{本方案的主创新是 future-GT-free self-consensus reward}
}
\]

---

## 30. 参考文献与官方资源

### [1] TTRL

Zuo, Y., Zhang, K., Sheng, L., Qu, S., Cui, G., et al.  
**TTRL: Test-Time Reinforcement Learning.** NeurIPS 2025.  
Paper: https://arxiv.org/abs/2504.16084  
Code: https://github.com/PRIME-RL/TTRL

### [2] RLVR-World

Wu, J., Yin, S., Feng, N., and Long, M.  
**RLVR-World: Training World Models with Reinforcement Learning.** NeurIPS 2025.  
Paper: https://arxiv.org/abs/2505.13934  
Project: https://thuml.github.io/RLVR-World/  
Code: https://github.com/thuml/RLVR-World

### [3] iVideoGPT

Wu, J., Yin, S., Feng, N., He, X., Li, D., Hao, J., and Long, M.  
**iVideoGPT: Interactive VideoGPTs are Scalable World Models.** NeurIPS 2024.  
Paper: https://arxiv.org/abs/2405.15223  
Project: https://thuml.github.io/iVideoGPT/

### [4] DeepSeekMath / GRPO

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., et al.  
**DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.** 2024.  
Paper: https://arxiv.org/abs/2402.03300

### [5] RT-1

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J., et al.  
**RT-1: Robotics Transformer for Real-World Control at Scale.** 2022.  
Paper: https://arxiv.org/abs/2212.06817  
Project: https://robotics-transformer1.github.io/

### [6] DINOv2

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., et al.  
**DINOv2: Learning Robust Visual Features without Supervision.** 2023.  
Paper: https://arxiv.org/abs/2304.07193

### [7] Distribution-Aware Reward Estimation for TTRL

Du, B., Huang, X., and Li, X.  
**Distribution-Aware Reward Estimation for Test-Time Reinforcement Learning.** 2026.  
Paper: https://arxiv.org/abs/2601.21804

### [8] Persistent Robot World Models

Bardhan, J., Drozdik, P., Sivic, J., and Petrik, V.  
**Persistent Robot World Models: Stabilizing Multi-Step Rollouts via Reinforcement Learning.** 2026.  
Paper: https://arxiv.org/abs/2603.25685

---

## 31. 最终执行顺序

严格按照以下顺序推进：

1. 复现 RLVR-World 官方 RT-1 单步 base；
2. 复现官方 GT-based RLVR 评估；
3. 缓存 500—1000 个输入的 \(K=16\) 候选；
4. 不训练模型，验证自共识与 GT 质量相关性；
5. 对比 pixel、image feature、transition feature；
6. 仅当 transition consensus 有效时实现 binary reward；
7. 加入全 0 / 全 1 skip；
8. 运行 20—50 step smoke test；
9. 加入 static gate；
10. 检查 repetition rate、action sensitivity、KL、skip rate；
11. 运行主实验；
12. 完成 Consensus-SFT 对比；
13. 完成 group size 与阈值消融；
14. 输出成功与失败案例；
15. 再决定是否扩展至多步预测。

---

## 32. 一句话总结

\[
\boxed{
\text{同一状态—动作条件下生成多个候选下一帧}
\rightarrow
\text{在视觉变化空间中执行邻域投票}
\rightarrow
\text{用二值自共识奖励替代未来 GT 奖励}
\rightarrow
\text{通过 GRPO 完成无 GT 世界模型后训练}
}
\]

该方案足够简洁，能够与 TTRL 和 RLVR-World 建立明确联系，同时保留清晰的独立创新点。
