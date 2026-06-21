# 首轮 GRPO 训练结果（2026-06-15，本地存档）

设计：reward 距离 D ∈ {pixel LPIPS, FSQ code-space RMS}，advantage shaping = gt_only，
40 步，K=16，batch_windows=2，train_windows=24，eval_windows=12，seed=0，temp=1.0(默认)。
同一 base 起点。源：服务器 `outputs/grpo/train_full.log`。

## pixel 臂（RLVR-World baseline，D = -LPIPS）

| step | evalLPIPS | codeRMS | PSNR | repeat |
|------|-----------|---------|------|--------|
| 0    | 0.0964    | 0.3787  | 23.94| 0.000  |
| 10   | 0.0921    | 0.3653  | 24.19| 0.000  |
| 20   | 0.0869    | 0.3565  | 24.40| 0.000  |
| 30   | 0.0849    | 0.3492  | 24.54| 0.000  |
| 40   | 0.0833    | 0.3458  | 24.60| 0.000  |

## code 臂（D = -code RMS，不 decode）

| step | evalLPIPS | codeRMS | PSNR | repeat |
|------|-----------|---------|------|--------|
| 0    | 0.0964    | 0.3787  | 23.94| 0.000  |
| 10   | 0.0867    | 0.3570  | 24.35| 0.000  |
| 20   | 0.0848    | 0.3468  | 24.52| 0.000  |
| 30   | 0.0827    | 0.3399  | 24.69| 0.000  |
| 40   | **0.0817**| **0.3359**| **24.78**| 0.000 |

## 结论

- **code reward 全面 ≥ pixel baseline**：step40 evalLPIPS 0.0817<0.0833、codeRMS 0.3359<0.3458、
  PSNR 24.78>24.60；两臂 repeat=0 无塌缩，均单调改善。
- **关键**：code reward 连像素 LPIPS 都赢过 pixel reward → 支撑 *Dynamics over Reconstruction*
  （code 信号剥离 decode 重建噪声 → 梯度更干净 → 泛化到像素指标也更好）。
- 诚实：margin 小（LPIPS ~2%）、single seed/40 步小规模；稳健性待多 seed + 更长训练。
- 每步 K16 bw2 ≈ 9.5s。
- hybrid 臂 + `curves_full.json` 因午夜关机时限未必跑完（json 三臂全完才写）。

## 复现/续跑命令（服务器）

    cd /root/autodl-tmp/vote2world
    python scripts/train_grpo.py --rewards pixel,code,hybrid --modes gt_only \
      --steps 40 --K 16 --batch_windows 2 --train_windows 24 --eval_windows 12 \
      --eval_every 10 --alpha 0.5 --lr 1e-5 --seed 0 \
      --out outputs/grpo/curves_full.json
