"""Render the method figure: Vote2World consensus-shaped advantage pipeline."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(figsize=(9.2, 3.5), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.axis("off")


def box(x, y, w, h, text, fc, ec="#333333", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(x1, y1, x2, y2, color="#444444"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 lw=1.2, color=color))


box(1, 16, 15, 9, "History\n$o_{t-3:t},\\ a_{t-3:t}$\n(4 frames, 13-d action)", "#E8F0FE", bold=True)
box(19, 14.5, 17, 12, "RLVR-World\nAR world model\n(frozen FSQ tokenizer\n+ Llama)\n$\\Rightarrow$ sample $K$ candidates\n$\\hat{o}^{(1..K)}_{t+1}$", "#FCE8E6", bold=True)
box(40, 27, 24, 9.5, "GT reward (kept):\n$r^{gt}_k=-\\mathrm{LPIPS}(\\hat{o}_k, o^{gt}_{t+1})$", "#E6F4EA")
box(40, 14.5, 24, 10.5, "Consensus (vote):\n$\\Delta_k=\\phi(\\hat{o}_k)-\\phi(o_t)$\nneighborhood support $c_k$\n+ static-copy gate", "#FEF7E0")
box(67, 18.5, 16, 13, "Advantage shaping\n$\\tilde{A}_k=z(r^{gt}_k)$\n$\\times\\max(\\delta,1{+}\\beta z(c_k))$", "#F3E8FD", bold=True)
box(86, 19, 13, 12, "GRPO\nupdate world\nmodel with\n$\\tilde{A}_k$", "#E8F0FE", bold=True)

arrow(16, 20.5, 19, 20.5)
arrow(36, 22, 40, 31.5)        # to GT reward
arrow(36, 19, 40, 19.5)        # to consensus
arrow(64, 31.5, 67, 27)        # GT -> shaping
arrow(64, 19.5, 67, 24)        # consensus -> shaping
arrow(83, 25, 86, 25)          # shaping -> grpo
ax.add_patch(FancyArrowPatch((92, 19), (27.5, 14.5), connectionstyle="arc3,rad=0.32",
                             arrowstyle="-|>", mutation_scale=11, lw=1.1, color="#7B61A8", ls="--"))
ax.text(60, 6.5, "policy-gradient update (world model only; tokenizer & $\\phi$ frozen)",
        ha="center", fontsize=7.5, color="#7B61A8", style="italic")
ax.text(50, 38.7, "Vote2World: consensus-shaped advantage for verifiable world-model RL",
        ha="center", fontsize=10, fontweight="bold")

out = sys.argv[1] if len(sys.argv) > 1 else "fig1_method.png"
plt.savefig(out, bbox_inches="tight", dpi=300)
print("saved", out)
