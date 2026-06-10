# Experiment Roadmap

## Phase 0: Official Baselines

- Reproduce RLVR-World RT-1 single-step base evaluation.
- Reproduce GT-based RLVR evaluation.
- Confirm tokenizer, decoder, candidate sampling, and metrics.

## Phase 1: Proxy-Quality Analysis

- Sample 500-1000 adaptation inputs.
- Generate `K=16` next-frame candidates per input.
- Compare pixel, image-feature, and transition-feature consensus.
- Use held-out GT only in the analysis script.
- Report Spearman correlation, positive precision, and pairwise ranking accuracy.

## Phase 2: Minimal Reward

- Implement binary neighborhood consensus reward.
- Add all-zero and all-one reward group skip.
- Add group confidence abstention.
- Add no-GT leakage tests.

## Phase 3: GRPO Smoke Test

- Run 20-50 GRPO steps.
- Track reward ratio, skip rate, KL, entropy, gradient norm, and repetition rate.
- Save qualitative samples before scaling.

## Phase 4: Full Method

- Add transition-feature reward.
- Add action-aware static-copy gate.
- Run main comparison and ablations.

