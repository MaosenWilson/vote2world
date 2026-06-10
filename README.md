# Vote2World

Vote2World is a research workspace for **ground-truth-free self-consensus reinforcement learning for action-conditioned visual world models**.

The first target is a minimal publishable setting:

- RT-1-style single-step next-frame prediction.
- Autoregressive visual world model based on the RLVR-World / iVideoGPT line.
- No future ground-truth frame access during adaptation.
- Candidate next-frame sampling, transition-feature neighborhood voting, binary pseudo-reward, group abstention, static-copy filtering, and GRPO post-training.

The detailed research plan is in [docs/research/Vote2World_GT_Free_Self_Consensus_RL_Plan.md](docs/research/Vote2World_GT_Free_Self_Consensus_RL_Plan.md).

## Current Status

This repository is currently a workspace scaffold. It is prepared for later execution on a rented large-memory GPU server. Local scripts are not expected to run until the upstream RLVR-World code, RT-1 data, tokenizer, and checkpoints are available on the server.

## Workspace Layout

```text
vote2world/
  docs/
    research/        Research plan and method notes.
    engineering/     Engineering runbooks and workspace conventions.
    experiments/     Experiment plans, result templates, and analysis notes.
  src/vote2world/
    rewards/         GT-free consensus reward modules.
    analysis/        Proxy-quality analysis and diagnostics.
    data/            Dataset wrappers and no-GT adaptation loaders.
    models/          Model adapters around upstream visual world models.
    training/        GRPO / SFT training entry points.
    evaluation/      Held-out GT evaluation only.
    utils/           Shared logging, config, and metric helpers.
  configs/
    baseline/        Official baseline evaluation configs.
    analysis/        Candidate cache and proxy-quality configs.
    grpo/            Smoke-test and full GRPO configs.
    ablations/       Ablation config variants.
  scripts/
    setup/           Server setup and environment checks.
    download/        Dataset and checkpoint download helpers.
    run/             Main experiment launchers.
    analysis/        Offline analysis launchers.
    server/          Server-specific notes and commands.
  third_party/
    RLVR-World/      Placeholder for the upstream RLVR-World checkout.
  data/              Local or server data mount points. Ignored by git.
  checkpoints/       Tokenizers and model checkpoints. Ignored by git.
  outputs/           Logs, reports, visualizations, and caches. Ignored by git.
  tests/             Unit tests, especially no-GT leakage checks.
```

## Execution Roadmap

1. Reproduce official RLVR-World RT-1 single-step base and GT-based RLVR evaluation.
2. Cache 500-1000 adaptation inputs with `K=16` candidate predictions.
3. Without training, test whether pixel, image-feature, and transition-feature consensus correlate with held-out GT quality.
4. Implement GT-free binary consensus reward only after transition consensus shows positive proxy value.
5. Run 20-50 GRPO smoke-test steps and inspect reward ratio, skip rate, KL, entropy, repetition rate, and visual samples.
6. Add static-copy gate, run main experiments, then run ablations.

## No-GT Adaptation Rule

During self-consensus RL adaptation, future observations must be withheld from the reward path.

Engineering constraints:

- Adaptation dataloaders should return only observation history and current action.
- Reward modules must not accept `ground_truth`, `future_frame`, or equivalent fields.
- Held-out GT can be used only by independent evaluation and proxy-quality analysis scripts.
- Tests should explicitly check that adaptation batches cannot expose future observations.

## Large File Policy

The following paths are intentionally ignored by git:

- `data/`
- `checkpoints/`
- `outputs/`
- `third_party/RLVR-World/`

Keep only lightweight metadata, manifests, config files, and reports in git. Put RT-1 data, generated candidate frames, decoded videos, logs, and checkpoints on the training server or external storage.

## Server Bring-Up Plan

On the GPU server:

1. Clone this repository.
2. Clone RLVR-World into `third_party/RLVR-World/`.
3. Install the upstream environment according to RLVR-World first.
4. Download RT-1 tokenizer, base checkpoint, and GT-based RLVR checkpoint into `checkpoints/`.
5. Mount or download RT-1 data into `data/rt1/`.
6. Run official baseline inference and evaluation before adding Vote2World code.

## GitHub Sync

This workspace can be initialized as a git repository immediately, but GitHub sync requires a target remote.

Use one of these:

```bash
git remote add origin git@github.com:<owner>/vote2world.git
git push -u origin main
```

or create a new GitHub repository first, then add its remote URL. If this directory is initialized locally before the remote exists, keep the first commit limited to the scaffold and documentation.
