"""Offline validation of the consensus-shaped reward design (no training).
  Q1: does transition-feature consensus correlate with held-out GT quality?
  Q2: does each ranker pick better candidates than pure-GT / consensus-only / random?
  Q3: static-copy / repetition diagnostics.
Held-out GT used for evaluation only. Consumes the cache from cache_candidates.py."""
import argparse
import json
import os

import numpy as np

from dor.constants import ROOT
from dor.metrics import pearson, spearman
from dor.rewards import shape_advantage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=f"{ROOT}/outputs/analysis/cache.npz")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--out", default=f"{ROOT}/outputs/analysis/results.json")
    args = ap.parse_args()

    d = np.load(args.cache)
    N, K = d["lpips"].shape
    quality = -d["lpips"]
    qpsnr = d["psnr"]
    r_gt = -d["lpips"]                       # GT reward (RLVR-World perceptual), GT kept
    v, copy_sim, motion, tok_rep = d["v"], d["copy_sim"], d["motion"], d["tok_repeat"]

    # ---- Q1: within-group correlation, consensus vs GT quality ----
    sp = np.array([spearman(v[i], quality[i]) for i in range(N)])
    pe = np.array([pearson(v[i], quality[i]) for i in range(N)])
    q1 = dict(
        spearman_mean=float(np.nanmean(sp)),
        spearman_se=float(np.nanstd(sp) / np.sqrt(max(1, np.sum(~np.isnan(sp))))),
        pearson_mean=float(np.nanmean(pe)),
        frac_groups_positive=float(np.nanmean(sp > 0)),
    )

    # ---- Q2: candidate selection by ranker ----
    def pick(scores):
        idx = scores.argmax(axis=1)
        return d["lpips"][np.arange(N), idx], qpsnr[np.arange(N), idx]

    def hybrid_scores(mode):
        out = np.zeros((N, K))
        for i in range(N):
            adv, _ = shape_advantage(r_gt[i], v[i], mode=mode, lam=args.lam, beta=args.beta,
                                     motion=float(motion[i]), copy_sim=copy_sim[i])
            out[i] = adv
        return out

    rng = np.random.default_rng(0)
    rand_idx = rng.integers(0, K, size=N)
    oracle, worst, meanlp = d["lpips"].min(1), d["lpips"].max(1), d["lpips"].mean(1)

    def regret(lp):
        return float(np.mean((lp - oracle) / (worst - oracle + 1e-8)))

    rankers = {
        "gt_only": r_gt,
        "consensus_only": v,
        "hybrid_add": hybrid_scores("hybrid_add"),
        "hybrid_mult": hybrid_scores("hybrid_mult"),
    }
    q2 = {}
    for name, sc in rankers.items():
        lp, ps = pick(sc)
        q2[name] = dict(mean_lpips=float(lp.mean()), mean_psnr=float(ps.mean()), norm_regret=regret(lp))
    q2["random"] = dict(
        mean_lpips=float(d["lpips"][np.arange(N), rand_idx].mean()),
        mean_psnr=float(qpsnr[np.arange(N), rand_idx].mean()),
        norm_regret=regret(d["lpips"][np.arange(N), rand_idx]),
    )
    q2["oracle"] = dict(mean_lpips=float(oracle.mean()), mean_psnr=float(qpsnr.max(1).mean()), norm_regret=0.0)
    q2["group_mean"] = dict(mean_lpips=float(meanlp.mean()), mean_psnr=float(qpsnr.mean()), norm_regret=regret(meanlp))

    # ---- Q3: static copy / repetition ----
    hi = motion > np.median(motion)
    q3 = dict(
        token_repeat_rate=float(tok_rep.mean()),
        token_repeat_rate_high_motion=float(tok_rep[hi].mean()) if hi.any() else float("nan"),
        mean_copy_sim_lpips=float(copy_sim.mean()),
        copy_current_mean_lpips=float(d["copy_q"][:, 2].mean()),
        model_mean_lpips=float(d["lpips"].mean()),
    )

    res = dict(N=int(N), K=int(K), lam=args.lam, beta=args.beta,
               Q1_consensus_correlation=q1, Q2_selection=q2, Q3_static_copy=q3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print("ANALYZE_OK", flush=True)


if __name__ == "__main__":
    main()
