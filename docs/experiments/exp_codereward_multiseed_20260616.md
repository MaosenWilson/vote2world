# 实验：code-reward 多 seed 稳健性验证 (multiseed, 2026-06-16)

## 1. 运行配置（运行前登记）

**目的 / 假设**：首轮 (n=1, seed0) code reward 三指标全面 ≥ pixel baseline，但 margin ~2%、单 seed。
本实验跑 5 个 seed 验证 **code 对 pixel 的优势是否带误差棒稳住**（排除 seed 噪声）。
预期：若 code 在 5 seed 上均值优于 pixel 且 std 不掩盖差异 → 稳健性成立，标题级 claim 立住；否则回到"小规模占优但不稳"的诚实结论。

**运行环境**
- 服务器：AutoDL `root@connect.westd.seetacloud.com:39929`（端口每次开机变；本次 39929）
- 解释器：`/root/miniconda3/bin/python`
- 项目：`/root/autodl-tmp/vote2world`
- 模型 ckpt：base=`thuml/rt1-world-model-single-step-base`；tokenizer=`CNNFSQModel256`（fsq_levels [7,5,5,5,5]）
- 代码版本：`grpo.py`(gt_reward/code_rms) + `train_grpo.py`(--rewards) 已部署并 import 通过；本地备份 `server_code_backup/`

**超参（论文要写的，逐项列全）**
| 参数 | 值 | 说明 |
|---|---|---|
| reward (D) | pixel, code | 两臂对比；pixel=-LPIPS(baseline)，code=-FSQ code RMS(不 decode) |
| mode (shaping) | gt_only | 纯 GT 可验证 reward，不叠加共识 |
| steps | 40 | 与首轮严格可比 |
| K (候选数) | 16 | |
| temperature | 1.0 | grpo generate 默认；gt_only 不涉共识，两臂同 temp 公平 |
| batch_windows | 2 | |
| train_windows / eval_windows | 24 / 12 | |
| lr | 1e-5 | AdamW |
| seeds | 0, 1, 2, 3, 4 | **本实验唯一新增维度** |
| eval_every | 10 | eval at step 0,10,20,30,40 |
| alpha / lam / beta | n/a | gt_only 不用 |

**启动命令**（5 seed × pixel,code 两臂，nohup 后台）：

    cd /root/autodl-tmp/vote2world
    for s in 0 1 2 3 4; do
      python scripts/train_grpo.py --rewards pixel,code --modes gt_only \
        --steps 40 --K 16 --batch_windows 2 --train_windows 24 --eval_windows 12 \
        --eval_every 10 --lr 1e-5 --seed $s --out outputs/grpo/curves_seed${s}.json
    done

预计 ~70min（10 runs × ~7min）。日志 `outputs/grpo/train_multiseed.log`。

## 2. 结果（运行后填写）

来源：服务器 `outputs/grpo/train_multiseed.log` + `curves_seed{0..4}.json`（10 runs 全完成，13:53→~15:05）。

**每 seed step40（pixel / code）**
| seed | LPIPS px/code | codeRMS px/code | PSNR px/code | repeat |
|------|---------------|------------------|--------------|--------|
| 0 | 0.0838 / 0.0817 | 0.3467 / 0.3359 | 24.59 / 24.78 | 0 |
| 1 | 0.0844 / 0.0830 | 0.3479 / 0.3385 | 24.54 / 24.66 | 0 |
| 2 | 0.0830 / 0.0813 | 0.3438 / 0.3340 | 24.66 / 24.83 | 0 |
| 3 | 0.0825 / 0.0802 | 0.3450 / 0.3326 | 24.66 / 24.86 | 0 |
| 4 | 0.0834 / 0.0801 | 0.3461 / 0.3295 | 24.62 / 24.98 | 0 |

**汇总（n=5，mean±std）与配对差（code−pixel）**
| 指标 | pixel | code | Δ mean | code 胜 |
|------|-------|------|--------|---------|
| evalLPIPS ↓ | 0.0834±0.0007 | 0.0813±0.0011 | −0.0022 | 5/5 |
| codeRMS ↓ | 0.3459±0.0014 | 0.3341±0.0031 | −0.0118 | 5/5 |
| PSNR ↑ | 24.61±0.05 | 24.82±0.11 | +0.21 | 5/5 |

逐 seed 配对 dLPIPS = [−0.0021,−0.0014,−0.0017,−0.0023,−0.0033]（全负）。

## 3. 分析（阶段完成填写）

- **结论**：**code 对 pixel 的优势带误差棒稳住——稳健性确认。** n=1 升级为 n=5，code 在 3 指标 × 5 seed = 15/15 配对全胜，方向 100% 一致；首轮的"会不会是 seed 噪声"被排除。
- **数据支撑**：① 三指标误差棒不重叠（LPIPS gap 0.0022 ≈ 2–3×std；codeRMS gap 0.0118 ≫ std 0.001–0.003；PSNR gap 0.21 > std）。② 逐 seed 配对差全部同号。单指标符号检验 p=(1/2)^5≈0.031，三指标（相关）联合更强。
- **审稿人视角 / 漏洞**：
  - margin 绝对值仍小（LPIPS ~2.6% 相对）——是"**稳健的小提升**"，写作时不可夸大为大幅领先。
  - 40 步未收敛（codeRMS 仍单调降，seed4 到 0.3295）；temp=1.0 未对齐探针 temp0.5。更长训练 / 调温 可能放大差距，也可能不——待测。
  - 统计可补：配对 t 检验 / Wilcoxon 给 p 值；seed 数可加到 8–10 上发表级误差棒。
  - 同一 base 起点、同 testbed、同 eval；公平。未含 hybrid、未接共识 shaping。
- **下一步**：① 延长步数(80–120)看分化是否扩大 + 收敛点；② 补 hybrid 第三臂；③ 训练加 `--temperature` 对齐 0.5；④ 接共识 hybrid_mult 做正交叠加；⑤ 画 5-seed 训练曲线带误差带（论文 figure）。

## 变更记录

- 2026-06-16 创建并按 5 seeds × 40 步运行；结果回填：code 三指标 5/5 全胜，稳健性确认。
