"""Plot held-out LPIPS / PSNR curves for each GRPO reward mode in curves.json."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dor.constants import ROOT  # noqa: E402

CURVES = f"{ROOT}/outputs/grpo/curves.json"
OUT = f"{ROOT}/outputs/figures/grpo_curves.png"
COLORS = {"gt_only": "#888888", "hybrid_add": "#4C9F70", "hybrid_mult": "#7B61A8"}


def main():
    with open(CURVES) as f:
        runs = json.load(f)["runs"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=200)
    for mode, r in runs.items():
        c = COLORS.get(mode)
        ax[0].plot(r["step"], r["eval_lpips"], "-o", color=c, label=mode, lw=1.6, ms=4)
        ax[1].plot(r["step"], r["eval_psnr"], "-o", color=c, label=mode, lw=1.6, ms=4)
    ax[0].set_title("Held-out LPIPS (lower better)")
    ax[1].set_title("Held-out PSNR (higher better)")
    for a in ax:
        a.set_xlabel("GRPO step")
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
