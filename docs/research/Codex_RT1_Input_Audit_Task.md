# RLVR-World / RT-1 输入核查任务说明

> **用途**：交给 Codex 执行  
> **目标**：先下载少量 RT-1 数据并核查数据结构、动作字段、官方预处理结果与 RLVR-World 单步输入链路，为后续 GT-free 自共识奖励设计提供可靠依据。  
> **要求**：本阶段只做环境检查、少量数据下载、字段核查与结果汇总，不进行模型训练，不修改 RLVR-World 代码，不开始 GRPO。

---

## 1. 背景

我们计划基于 RLVR-World 官方提供的 RT-1 单步视觉世界模型权重，研究一种不依赖未来 GT 帧的自共识强化学习后训练方法。

当前最小研究路线为：

\[
(o_{t-3:t}, a_{t-3:t})
\rightarrow
K \text{ 个候选下一帧}
\rightarrow
\text{候选之间自共识投票}
\rightarrow
R_k \in \{0,1\}
\rightarrow
\text{GRPO 微调}
\]

在开始修改奖励函数之前，必须先确认：

1. RT-1 数据集中动作字段的真实结构；
2. RLVR-World 官方预处理脚本如何展平动作；
3. 展平后的动作维度是否为 13；
4. 单步模型实际读取多少张历史图像和多少步动作；
5. 图中标注的  
   \[
   (x,y,z,\mathrm{roll},\mathrm{pitch},\mathrm{yaw},\mathrm{gripper\ openness})
   \]
   与数据集中字段如何对应；
6. 后续静态复制过滤、动作幅度分桶与动作一致性奖励应该基于哪些字段。

---

## 2. 本阶段任务边界

### 2.1 本阶段需要完成

- 检查服务器中是否已存在 RLVR-World 项目；
- 检查服务器中是否已存在 RT-1 / Open X-Embodiment 数据；
- 若不存在，则下载少量 RT-1 数据；
- 核查原始数据结构；
- 使用 RLVR-World 官方预处理脚本转换少量 episode；
- 核查转换后的 `.npz` 文件；
- 核查 RLVR-World 单步 processor、训练配置与动作量化逻辑；
- 输出一份结构化报告；
- 给出后续奖励设计应采用的动作字段拆分建议。

### 2.2 本阶段禁止执行

- 不进行 tokenizer 预训练；
- 不进行 Transformer 预训练；
- 不运行完整 GRPO；
- 不修改 reward function；
- 不下载完整 RT-1 数据集；
- 不下载不必要的大模型；
- 不删除已有项目或数据；
- 不对服务器目录进行大规模清理；
- 不修改系统 CUDA、PyTorch 或 Conda 全局环境；
- 不将未来 GT 用于任何训练。

---

## 3. 目标数据集

目标数据集为 RT-1 对应的 Open X-Embodiment 数据：

```text
fractal20220817_data
```

该数据集来自 RT-1 机器人操作轨迹。

本阶段只需要下载少量 episode，用于核查结构和跑通预处理流程。建议优先下载或保留：

```text
5—20 个 episode
```

如果官方数据下载工具无法直接限制 episode 数量，则先下载最小可用 split，随后仅转换前若干个 episode。

---

## 4. 需要确认的核心问题

### 4.1 原始动作字段

请检查 `fractal20220817_data` 中每个 `step["action"]` 的结构。

需要确认是否包含以下字段，以及每个字段的 shape、dtype、样例值和数值范围：

| 字段 | 预期含义 | 预期 shape |
|---|---|---:|
| `world_vector` | 末端执行器平移增量 \(x,y,z\) | `(3,)` |
| `rotation_delta` | 末端执行器姿态增量 `roll, pitch, yaw` | `(3,)` |
| `gripper_closedness_action` | 夹爪开合 | `(1,)` |
| `base_displacement_vector` | 底盘平移 | `(2,)` |
| `base_displacement_vertical_rotation` | 底盘旋转 | `(1,)` |
| `terminate_episode` | episode 模式或终止相关表示 | `(3,)` |

需要确认图中展示的 7 维动作：

\[
(x,y,z,\mathrm{roll},\mathrm{pitch},\mathrm{yaw},\mathrm{gripper\ openness})
\]

是否对应：

\[
\texttt{world\_vector}
+
\texttt{rotation\_delta}
+
\texttt{gripper\_closedness\_action}.
\]

### 4.2 展平后的动作维度

请确认 RLVR-World 官方 `oxe_data_converter.py` 对结构化动作字段执行拼接后，最终动作维度是否为：

\[
13.
\]

需要输出：

```text
flattened_action.shape
```

以及至少 3 个时间步的展平动作样例。

### 4.3 动作字段拼接顺序

这一点非常重要。

RLVR-World 官方转换脚本可能按：

```text
step["action"].keys()
```

返回顺序进行拼接。

请明确输出：

1. 原始 `step["action"].keys()` 的顺序；
2. 每个字段在展平向量中的索引范围；
3. 展平后 13 维向量的字段映射表。

最终报告中应给出类似表格：

| 索引范围 | 字段 | 维度 |
|---|---|---:|
| `[0:3]` | 待确认 | 3 |
| `[3:6]` | 待确认 | 3 |
| `...` | `...` | `...` |

不要根据文档猜测，必须根据实际下载的数据与官方转换脚本确认。

### 4.4 图像字段

请确认 RT-1 中用于 RLVR-World 的图像字段名称。

预期：

```text
observation["image"]
```

需要输出：

- shape；
- dtype；
- 分辨率；
- 值域；
- episode 中帧数量；
- 是否存在其他相机视角；
- RLVR-World 官方转换脚本默认使用哪个图像字段。

### 4.5 转换后的 `.npz` 文件

请使用 RLVR-World 官方：

```text
vid_wm/oxe_data_converter.py
```

转换少量 episode。

需要检查 `.npz` 文件中：

```text
keys
image.shape
action.shape
image.dtype
action.dtype
episode length
```

预期结构类似：

```text
keys = ["image", "action"]
image.shape = (T, H, W, 3)
action.shape = (T, 13)
```

但必须以实际结果为准。

---

## 5. RLVR-World 官方输入链路核查

### 5.1 核查单步训练配置

请检查 RLVR-World 视频世界模型单步代码与配置，确认：

- `context_length`；
- `segment_length`；
- `action_dim`；
- `action_bins`；
- `visual_token_num`；
- `action_ranges_path`；
- processor 类型；
- 单步模型输出 token 数量；
- 单步模型输入序列拼接顺序。

需要重点核查：

```text
vid_wm/ivideogpt/train_vgpt.py
vid_wm/ivideogpt/ivideogpt/processor.py
```

### 5.2 需要确认的预期值

请验证以下预期是否成立：

| 参数 | 预期值 |
|---|---:|
| `context_length` | 4 |
| `segment_length` | 5 |
| `action_dim` | 13 |
| `action_bins` | 256 |

如果实际配置不同，请明确指出。

### 5.3 单步输入输出关系

请确认官方单步模型是否实现：

\[
(o_{t-3:t},a_{t-3:t})
\rightarrow
\hat{o}_{t+1}.
\]

需要回答：

1. 输入是否包含 4 张历史帧；
2. 输入是否包含对应的 4 个动作向量；
3. 每个动作向量是否为 13 维；
4. 图像是否先经过 tokenizer 转换为视觉 token；
5. 动作是否逐维量化为离散 token；
6. 历史视觉 token 与动作 token 是否在每个时间步拼接；
7. 输出是否为下一帧视觉 token；
8. decoder 是否将下一帧 token 还原为图像。

### 5.4 动作量化方式

请核查官方 processor 中动作量化逻辑，输出清晰说明：

\[
a_t^{(d)}
\rightarrow
q_t^{(d)}
\in
\{0,\ldots,255\}.
\]

需要确认：

- 每个动作维度是否独立量化；
- 数值范围是否从 `action_ranges_path` 读取；
- 是否执行 clip；
- 是否执行 floor；
- 是否添加视觉 token offset；
- 是否存在特殊 token 或额外处理。

---

## 6. 与后续奖励设计直接相关的核查

### 6.1 静态复制过滤不能直接使用完整 13 维范数

后续我们计划过滤：

\[
\hat{o}_{t+1}\approx o_t
\]

但不能直接对完整动作向量计算：

\[
\|a_t\|_2.
\]

原因是 13 维动作混合了：

- 机械臂平移；
- 机械臂旋转；
- 夹爪开合；
- 底盘平移；
- 底盘旋转；
- 终止模式。

不同字段尺度和语义不同。

### 6.2 请输出动作分组建议

根据实际字段与索引顺序，输出以下分组：

#### 机械臂动作

\[
a_t^{\mathrm{arm}}
=
[
\texttt{world\_vector},
\texttt{rotation\_delta},
\texttt{gripper\_closedness\_action}
].
\]

#### 底盘动作

\[
a_t^{\mathrm{base}}
=
[
\texttt{base\_displacement\_vector},
\texttt{base\_displacement\_vertical\_rotation}
].
\]

#### 模式或终止字段

\[
a_t^{\mathrm{mode}}
=
[
\texttt{terminate\_episode}
].
\]

需要明确：

- 每一组在展平向量中的索引；
- 是否应参与静态复制过滤；
- 是否需要单独设置阈值；
- 哪些字段不应进入连续运动幅度计算。

### 6.3 统计动作分布

请基于少量 episode 输出以下统计：

| 统计项 | 要求 |
|---|---|
| 每个动作字段的 min / max | 必须输出 |
| 每个动作字段的 mean / std | 必须输出 |
| 每个动作字段零值比例 | 必须输出 |
| 机械臂平移范数分布 | 必须输出 |
| 机械臂旋转范数分布 | 必须输出 |
| 夹爪动作分布 | 必须输出 |
| 底盘运动范数分布 | 必须输出 |
| `terminate_episode` 取值模式 | 必须输出 |

### 6.4 判断是否适合动作幅度分桶

请根据统计结果给出建议：

- 是否适合按照机械臂平移幅度分桶；
- 是否需要单独按旋转幅度分桶；
- 夹爪动作是否应单独处理；
- 底盘动作是否应单独处理；
- `terminate_episode` 是否应从运动幅度中排除；
- 是否存在大量近零动作；
- 近零动作比例是否会影响静态复制过滤。

---

## 7. 少量数据下载与转换要求

### 7.1 下载原则

- 优先检查服务器是否已有数据；
- 不重复下载；
- 只下载或转换最少量 episode；
- 不进行完整数据集同步；
- 输出下载路径；
- 输出磁盘占用；
- 输出数据来源；
- 输出下载是否完整；
- 记录任何报错。

### 7.2 推荐目录结构

可根据服务器实际情况调整，但请保持结构清晰：

```text
project_root/
├── RLVR-World/
├── data/
│   ├── raw/
│   │   └── fractal20220817_data/
│   └── converted/
│       └── fractal20220817_data/
└── reports/
    └── rt1_input_audit/
```

### 7.3 转换范围

只转换：

```text
5—20 个 episode
```

如果转换脚本支持：

```text
--max_num_episodes
```

则使用该参数。

---

## 8. 输出文件要求

请在：

```text
reports/rt1_input_audit/
```

中生成以下文件。

### 8.1 `README.md`

简要说明：

- 检查目的；
- 数据来源；
- 下载范围；
- 项目路径；
- 数据路径；
- 执行步骤；
- 结果摘要；
- 遇到的问题。

### 8.2 `rt1_action_schema.md`

必须包含：

1. 原始 `step["action"]` 字段列表；
2. 每个字段 shape；
3. 每个字段 dtype；
4. 每个字段样例；
5. 拼接顺序；
6. 展平后的索引映射；
7. 最终维度；
8. 与图中 7 维动作的对应关系；
9. 与后续奖励设计相关的分组建议。

### 8.3 `rt1_npz_audit.md`

必须包含：

- 转换后的 `.npz` 文件列表；
- 每个文件的 `keys`；
- `image.shape`；
- `action.shape`；
- dtype；
- episode length；
- 至少 3 个样例动作；
- 异常项。

### 8.4 `rlvr_world_input_pipeline.md`

必须包含：

- 单步输入形式；
- 单步输出形式；
- `context_length`；
- `segment_length`；
- `action_dim`；
- `action_bins`；
- 动作量化过程；
- token 拼接过程；
- 输入序列示意图；
- 后续自共识奖励模块应接收哪些变量。

### 8.5 `rt1_action_statistics.md`

必须包含：

- 每个字段 min / max / mean / std；
- 零值比例；
- 机械臂平移范数；
- 机械臂旋转范数；
- 夹爪分布；
- 底盘运动分布；
- `terminate_episode` 分布；
- 动作幅度分桶建议；
- 静态复制过滤建议。

### 8.6 `audit_summary.md`

最终结论必须回答：

1. 图中 7 维动作是否来自 RT-1 数据；
2. RLVR-World 实际使用多少维动作；
3. 13 维动作的具体字段和索引；
4. 单步模型到底输入几张图像；
5. 单步模型输入几步动作；
6. 单步模型输出什么；
7. 动作如何量化；
8. 后续奖励设计应该使用哪些动作字段；
9. 是否可以直接使用官方 single-step base checkpoint；
10. 是否存在需要修改当前研究思路的地方。

---

## 9. 输出中必须附带的表格

### 9.1 原始动作字段表

| 字段 | shape | dtype | 样例 | min | max | mean | std | 零值比例 |
|---|---:|---|---|---:|---:|---:|---:|---:|

### 9.2 展平动作索引映射表

| 索引范围 | 字段 | 维度 | 是否参与静态复制过滤 | 建议处理方式 |
|---|---|---:|---:|---|

### 9.3 单步模型输入输出表

| 项目 | 实际值 | 证据位置 |
|---|---|---|
| context length |  |  |
| segment length |  |  |
| action dim |  |  |
| action bins |  |  |
| 图像字段 |  |  |
| 动作字段 |  |  |
| 输入 token 拼接方式 |  |  |
| 输出 token |  |  |
| decoder 输出 |  |  |

### 9.4 后续奖励模块输入表

| 输入 | 是否需要 | 用途 |
|---|---:|---|
| 当前帧 \(o_t\) | 是 | 计算视觉变化量 |
| 历史帧 \(o_{t-3:t-1}\) | 由世界模型使用 | 生成候选下一帧 |
| 当前动作 \(a_t\) | 是 | 静态复制过滤、动作分桶 |
| 历史动作 \(a_{t-3:t-1}\) | 由世界模型使用 | 生成候选下一帧 |
| 候选预测帧 \(\hat{o}_{t+1}^{(1:K)}\) | 是 | 共识投票 |
| 未来 GT \(o_{t+1}^{\mathrm{gt}}\) | 训练阶段禁止使用 | 仅用于独立评估 |

---

## 10. 证据要求

所有结论都必须注明证据来源。

优先证据顺序：

1. 实际下载的数据；
2. RLVR-World 官方代码；
3. Open X-Embodiment / TensorFlow Datasets 元数据；
4. RLVR-World 官方 README；
5. RT-1 官方文档。

不要只依据论文图示推断。

对于关键结论，请给出：

- 文件路径；
- 行号；
- 实际命令输出；
- 数据样例；
- shape；
- dtype。

---

## 11. Codex 执行顺序

请严格按以下顺序执行。

### 第一步：检查已有环境

确认：

- 项目目录；
- Conda 环境；
- Python 版本；
- 磁盘空间；
- 是否已有 RLVR-World；
- 是否已有 `fractal20220817_data`；
- 是否已有转换后的 `.npz`；
- 是否已有官方 checkpoint。

### 第二步：准备 RLVR-World 项目

若不存在：

- 克隆 RLVR-World 官方仓库；
- 不修改代码；
- 记录 commit SHA；
- 检查 `vid_wm` 目录；
- 检查 `oxe_data_converter.py`；
- 检查单步 processor；
- 检查单步训练脚本。

### 第三步：下载少量 RT-1 数据

目标：

```text
fractal20220817_data
```

只获取足以检查结构的最小数据量。

如果无法限制下载规模：

- 记录原因；
- 选择最小 split；
- 避免完整同步；
- 若下载量仍然过大，暂停并汇报。

### 第四步：检查原始结构

检查：

```text
episode["steps"]
step["observation"]
step["action"]
```

输出字段、shape、dtype、样例与顺序。

### 第五步：转换少量 episode

使用官方：

```text
vid_wm/oxe_data_converter.py
```

转换 5—20 个 episode。

### 第六步：检查 `.npz`

确认：

```text
image.shape
action.shape
```

并输出动作索引映射。

### 第七步：核查官方输入链路

检查：

```text
train_vgpt.py
ivideogpt/processor.py
```

确认：

- 4 帧历史；
- 4 步动作；
- 13 维动作；
- 256 bins；
- 图像 token；
- 动作 token；
- 拼接方式；
- 下一帧 token 生成方式。

### 第八步：统计动作分布

对少量 episode 生成统计表和结论。

### 第九步：生成报告

输出本任务要求的全部 `.md` 文件。

### 第十步：停止

完成报告后停止，不进入训练。

---

## 12. 验收标准

任务完成时，必须能够明确回答：

\[
\boxed{
\text{RLVR-World 单步模型的真实输入到底是什么？}
}
\]

并且给出：

\[
\boxed{
\text{RT-1 动作字段}
\rightarrow
\text{13 维展平动作}
\rightarrow
\text{256-bin 离散 token}
\rightarrow
\text{与视觉 token 拼接}
\rightarrow
\text{预测下一帧视觉 token}
}
\]

同时，必须给出后续无 GT 自共识奖励设计的动作字段拆分建议，避免将不兼容的动作维度直接混合计算。

---

## 13. 最终要求

请不要直接开始训练，也不要扩大任务范围。

本阶段只完成：

```text
少量数据下载
→ 原始结构检查
→ 官方预处理检查
→ 动作字段核查
→ 单步输入链路核查
→ 动作统计
→ 报告输出
```

完成后，将 `reports/rt1_input_audit/` 目录整体汇总，并给出：

1. 执行摘要；
2. 已确认事实；
3. 未确认问题；
4. 对当前研究思路的影响；
5. 下一步建议。
