# 实验：<简短标题> (<exp-id>, <日期>)

> 三段递进填充：运行前写【运行配置】→ 运行后写【结果】→ 阶段完成写【分析】。
> 参数有改动就更新【运行配置】并在末尾【变更记录】留痕。

## 1. 运行配置（运行前填写）

**目的 / 假设**：<这个实验要回答什么；预期结果>

**运行环境**
- 服务器：AutoDL `root@connect.westd.seetacloud.com:<port>`（端口每次开机变）
- 解释器：`/root/miniconda3/bin/python`
- 项目：`/root/autodl-tmp/vote2world`
- 模型 ckpt：base=`thuml/rt1-world-model-single-step-base`；tokenizer=`CNNFSQModel256`
- 代码版本：<grpo.py/train_grpo.py 是否含改动；本地备份 server_code_backup/>

**超参（论文要写的，逐项列全）**
| 参数 | 值 | 说明 |
|---|---|---|
| reward (D) | | pixel / code / hybrid |
| mode (shaping) | | gt_only / hybrid_add / hybrid_mult |
| steps | | |
| K (候选数) | | |
| temperature | | generate 采样温度 |
| batch_windows | | |
| train_windows / eval_windows | | |
| lr | | |
| seed(s) | | |
| alpha / lam / beta | | hybrid / 共识权重 |

**启动命令**：`<完整可复现命令>`

## 2. 结果（运行后填写）

**来源**：`<服务器 log / json 路径>`，已拉本地：`<本地路径>`

<结果表格 / 曲线数字>

## 3. 分析（阶段完成填写）

- **结论**：<是否支持假设，先结论>
- **数据支撑**：<关键数字>
- **审稿人视角 / 漏洞**：<margin、seed 方差、显著性、混淆变量…>
- **下一步**：<>

## 变更记录

- <日期> <改了什么参数 / 为什么>
