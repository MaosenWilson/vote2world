# DoR — Dynamics over Reconstruction

**Calibrating verifiable rewards for video world models.** (Project handle: `DoR`, formerly `Vote2World`.)

Video world-model RL post-training (RLVR-World style) scores candidate next-frame
predictions against a held-out ground-truth frame and turns that score into a GRPO
reward. The problem: a pixel-space metric like LPIPS is dominated by **tokenizer
reconstruction noise**, not dynamics — frame-to-frame motion in this setting
(≈0.040 LPIPS) is *smaller* than the tokenizer's own encode/decode floor
(≈0.053 LPIPS). The model ends up being rewarded for texture fidelity, not for
predicting the right motion.

**DoR's contribution:** move the verifiable reward out of pixel space and into the
visual tokenizer's pre-decode code space (FSQ `indices_to_codes`), where there is no
reconstruction floor to fight. A secondary, demoted contribution (originally the
project's starting point) reshapes the GRPO advantage with intra-group consensus —
kept as an ablation, not the headline.

## Headline result (2026-06-16, 5 seeds × 40 GRPO steps, RT-1 single-step)

Code-space reward beats the pixel-LPIPS baseline on all three held-out metrics,
every seed:

| metric (step 40) | pixel reward | code reward | paired wins |
|---|---|---|---|
| eval LPIPS  | 0.0834 ± 0.0007 | **0.0813 ± 0.0011** | 5/5 |
| code RMS    | 0.3459 ± 0.0014 | **0.3341 ± 0.0031** | 5/5 |
| PSNR        | 24.61 ± 0.05    | **24.82 ± 0.11**    | 5/5 |

Full method and derivation: [docs/DoR_method.md](docs/DoR_method.md). Experiment
ledger and raw run logs: [docs/experiments/](docs/experiments/).

## Method sketch

Skeleton = RLVR-World (GT-verifiable reward + GRPO). Two independent axes:

1. **Reward distance `D`** — `pixel` (-LPIPS, baseline) / `code` (FSQ code-space
   RMS, no decode — the DoR contribution) / `hybrid` (z-scored fusion).
2. **Advantage shaping** — `gt_only` (unmodified GRPO) / `hybrid_add` / `hybrid_mult`
   (consensus-shaped advantage, ablation; degrades to `gt_only` when β=0, and never
   lets group consensus override a GT-bad candidate).

Both axes are orthogonal and live in `src/dor/rewards.py` / `src/dor/grpo.py`.

## Repository layout

```text
src/dor/             Core package: rewards, GRPO loop, consensus, generation, metrics, tokenization.
  data/              RT-1 episode loading, window sampling, no-GT adaptation contracts.
  processors/        Adaptation-side preprocessing.
scripts/             Entry points: smoke_test / cache_candidates / analyze_consensus / train_grpo,
                     diagnostic probes (probe_phi_dino, probe_phi_latent, probe_tokenizer_floor),
                     figures/ (plotting), rt1/ (RT-1 input-pipeline audit & download).
configs/             YAML configs (grpo, analysis, ablations, baseline).
tests/               Unit tests, incl. no-future-GT leakage checks.
docs/
  DoR_method.md      Canonical method write-up (this is the paper-writing source of truth).
  experiments/       Experiment ledger (EXPERIMENTS.md) + per-run result logs + template.
  research/          Earlier research plans (pre-rename, retained for context).
  engineering/       Workspace conventions.
  论文写作操作流_SOP.md   Paper-writing tool/skill SOP (nature-skills vs paperspine).
reports/             RT-1 input-pipeline / action-schema audit reports.
extended_abstract/   Separate ICICIC 2026 extended-abstract deliverable (multi-level
                     consistency reward for DIAMOND CS:GO) — independent of the DoR main paper.
third_party/RLVR-World/   Upstream checkout, git-ignored.
data/ checkpoints/ outputs/  Git-ignored; live on the training server only.
```

## Quickstart (training server)

```bash
pip install -e .
python scripts/smoke_test.py                 # load + generate + score sanity check
python scripts/cache_candidates.py --n_windows 200 --K 16
python scripts/analyze_consensus.py           # Q1 consensus<->GT correlation, Q2 selection
python scripts/train_grpo.py --rewards pixel,code,hybrid --modes gt_only \
       --steps 40 --K 16 --seed 0 --out outputs/grpo/curves.json
```

Engineering notes:
- Blackwell (sm_120) cannot run upstream verl/vllm 0.6.3 — uses HF `.generate()` + a
  lightweight custom GRPO instead.
- `src/dor/compat.py` shims `huggingface_hub.cached_download` for diffusers 0.27.
- Every reward mode starts from the same pre-RL base checkpoint
  (`thuml/rt1-world-model-single-step-base`) for a fair comparison.
- Server-internal paths (`configs/vote2world/`, `reports/vote2world_*`, the
  `/root/autodl-tmp/vote2world` project directory) intentionally still say
  `vote2world` — renaming them is pure busywork with no payoff, the importable
  package is `dor`.

## Status

Renamed from `Vote2World` to `DoR` on 2026-06-15 after the tokenizer-floor diagnosis
above. Target has been raised from an ICICIC 2026 extended abstract to a top-venue
submission; method is locked, multi-seed robustness for the code-reward result is
confirmed, next steps are longer training runs, a third (hybrid) arm, and a
consensus × code-reward interaction ablation — see
[docs/experiments/EXPERIMENTS.md](docs/experiments/EXPERIMENTS.md).

## References

- Wu, Yin, Feng, Long. *RLVR-World: Training World Models with Reinforcement
  Learning.* arXiv:2505.13934, 2025.
- *TTRL: Test-Time Reinforcement Learning.* NeurIPS 2025.
- *ToolRL: Reward is All Tool Learning Needs.* NeurIPS 2025.

Large-file policy and the full reference list with verified citations live in
[docs/DoR_method.md](docs/DoR_method.md).
