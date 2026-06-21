"""Diagnose whether the consensus signal is drowned (by temperature / task regime)
or genuinely absent. For one or more candidate caches it reports:
  (A) candidate-distribution health: model vs copy-current, within-group spread;
  (B) Q1 consensus<->GT correlation, overall and stratified by motion.
"""
import argparse

import numpy as np

from dor.metrics import spearman


def per_group_spearman(v, quality, idx):
    sp = np.array([spearman(v[i], quality[i]) for i in idx])
    n = max(1, int(np.sum(~np.isnan(sp))))
    return float(np.nanmean(sp)), float(np.nanstd(sp) / np.sqrt(n)), float(np.nanmean(sp > 0))


def report(cache_path):
    d = np.load(cache_path)
    lpips = d["lpips"]                      # [N,K]
    N, K = lpips.shape
    quality = -lpips
    v = d["v"]
    motion = d["motion"]
    copy_lpips = d["copy_q"][:, 2]          # copy-current vs GT, per window

    gmin, gmax, gmean = lpips.min(1), lpips.max(1), lpips.mean(1)

    print(f"\n##### {cache_path}   (N={N}, K={K})")
    print("  --- (A) candidate distribution health ---")
    print(f"  model_mean_lpips        = {lpips.mean():.4f}")
    print(f"  copy_current_lpips      = {copy_lpips.mean():.4f}   (baseline; lower=closer to GT)")
    print(f"  frac windows best-cand beats copy = {(gmin < copy_lpips).mean():.3f}")
    print(f"  within-group spread (max-min)     = {(gmax - gmin).mean():.4f}")
    print(f"  within-group std                  = {lpips.std(1).mean():.4f}")
    print(f"  oracle/group_mean lpips           = {gmin.mean():.4f} / {gmean.mean():.4f}")

    print("  --- (B) Q1 consensus<->GT correlation (per-group spearman) ---")
    med = np.median(motion)
    lo = np.where(motion <= med)[0]
    hi = np.where(motion > med)[0]
    for name, idx in (("ALL", np.arange(N)), ("low-motion", lo), ("high-motion", hi)):
        m, se, fpos = per_group_spearman(v, quality, idx)
        print(f"  {name:<12} n={len(idx):>3}  spearman={m:+.4f} ± {se:.4f}   frac_pos={fpos:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("caches", nargs="+", help="paths to cache .npz files")
    args = ap.parse_args()
    for c in args.caches:
        report(c)
    print("\nDIAGNOSE_OK")


if __name__ == "__main__":
    main()
