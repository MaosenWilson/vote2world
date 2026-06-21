"""Train GRPO under one or more (reward, mode) designs and save held-out curves.

Reward distance D (--rewards):
  pixel   RLVR-World perceptual GT (-LPIPS)            baseline
  code    FSQ code-space RMS (no decode)               Dynamics-over-Reconstruction
  hybrid  alpha*z(pixel) + (1-alpha)*z(code)           fusion
Advantage shaping (--modes, dor.rewards.MODES):
  gt_only / hybrid_add / hybrid_mult  (consensus only reshapes advantage).

Example:
  python scripts/train_grpo.py --rewards pixel,code,hybrid --modes gt_only \
         --steps 40 --K 16 --temperature 0.5 --alpha 0.5 --seed 0
"""
import argparse
import json
import os

from dor.constants import ROOT
from dor.grpo import train
from dor.rewards import MODES

REWARDS = ("pixel", "code", "hybrid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewards", default="pixel,code,hybrid")
    ap.add_argument("--modes", default="gt_only")
    ap.add_argument("--alpha", type=float, default=0.5, help="hybrid weight on z(pixel)")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--batch_windows", type=int, default=2)
    ap.add_argument("--train_windows", type=int, default=24)
    ap.add_argument("--eval_windows", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lam", type=float, default=0.5, help="additive consensus weight (hybrid_add)")
    ap.add_argument("--beta", type=float, default=0.5, help="multiplicative consensus weight (hybrid_mult)")
    ap.add_argument("--kl", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--out", default=f"{ROOT}/outputs/grpo/curves.json")
    args = ap.parse_args()

    rewards = [r.strip() for r in args.rewards.split(",") if r.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad_r = set(rewards) - set(REWARDS)
    bad_m = set(modes) - set(MODES)
    if bad_r:
        raise SystemExit(f"unknown rewards {bad_r}; valid: {REWARDS}")
    if bad_m:
        raise SystemExit(f"unknown modes {bad_m}; valid: {MODES}")

    runs = {}
    for reward in rewards:
        for mode in modes:
            runs[f"{reward}-{mode}"] = train(
                mode, reward=reward, alpha=args.alpha, steps=args.steps, K=args.K,
                batch_windows=args.batch_windows, train_windows=args.train_windows,
                eval_windows=args.eval_windows, lr=args.lr, lam=args.lam, beta=args.beta,
                kl=args.kl, eval_every=args.eval_every, seed=args.seed,
            )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(args=vars(args), runs=runs), f, indent=2)
    print(f"[done] saved {args.out}", flush=True)
    print("GRPO_OK", flush=True)


if __name__ == "__main__":
    main()
