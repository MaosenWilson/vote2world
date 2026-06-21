# DoR 实验台账

记录约定见 `_TEMPLATE.md` 与记忆 `experiment-logging-workflow`。
每个实验：运行前写运行配置 → 运行后写结果 → 阶段写分析；此处一行索引。

| 日期 | 实验 | 状态 | 文件 | 一句话结论 |
|------|------|------|------|-----------|
| 2026-06-15 | GRPO 首轮 pixel vs code (gt_only, 40步, K16, seed0) | 完成 (n=1) | `grpo_train_results_20260615.md` | code reward 三指标全面 ≥ pixel baseline，概念验证级，待多 seed |
| 2026-06-16 | code-reward 多 seed 稳健性验证 (5 seed × 40步) | 完成 | `exp_codereward_multiseed_20260616.md` | code 三指标 5/5 全胜、误差棒不重叠，稳健性确认 |
