"""Route-B probe: measure candidates in FSQ *code* space WITHOUT decode.

tok.indices_to_codes maps token ids -> post-quant continuous codes, bypassing the
decoder, so there is no reconstruction-noise floor. Hardened version:
  - --which base|rlvr   (does RL training sharpen the code-space advantage?)
  - within-group spread (max-min, std) in code & pixel -> GRPO usability
  - per-window spearman(code_dist, pixel_lpips) -> are the two metrics orthogonal?

Reports over N windows (K cands @ temp 0.5):
  DISCR   oracle=min_i d(cand_i,gt) | copy=d(cur,gt) | model=mean   (code & pixel)
  SPREAD  within-group (max-min), std                                (code & pixel)
"""
import argparse

import numpy as np

from dor.constants import CTX, GRID
from dor.episodes import get_window_tensors, list_episodes, sample_windows
from dor.generation import generate_candidates
from dor.metrics import Metrics, spearman
from dor.models import load_action_ranges, load_tokenizer, load_world_model
from dor.tokenization import build_prompt, decode_tokens, encode_indices


def codes(tok, idx):
    """idx [B,16,20] long -> flat post-quant code [B, D] float (no decode)."""
    return tok.indices_to_codes(idx).float().reshape(idx.shape[0], -1)


def cdist(a, b):
    """per-element RMS distance; a [K,D], b [1,D] -> [K]."""
    return (a - b).pow(2).mean(1).sqrt().detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=184)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top_k", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--which", default="base", choices=["base", "rlvr"])
    args = ap.parse_args()
    dev = "cuda"
    tok = load_tokenizer(dev)
    model = load_world_model(dev, args.which)
    ar = load_action_ranges(dev)
    M = Metrics(dev)
    print(f"[setup] which={args.which} fsq_levels={getattr(tok, 'fsq_levels', '?')}", flush=True)

    rows = {k: [] for k in ("orc", "cpy", "mdl", "sprd", "std",
                            "orc_lp", "cpy_lp", "mdl_lp", "sprd_lp", "std_lp", "sp_cp")}
    for wi, (p, s) in enumerate(sample_windows(list_episodes(), args.n, seed=args.seed)):
        frames, actions = get_window_tensors(p, s, dev)
        gt, cur = frames[CTX], frames[CTX - 1]
        gt_idx = encode_indices(tok, gt.unsqueeze(0))
        cur_idx = encode_indices(tok, cur.unsqueeze(0))
        prompt = build_prompt(tok, frames, actions, ar)
        cand = generate_candidates(model, prompt, args.K, temperature=args.temperature,
                                   top_k=args.top_k, seed=args.seed * 100000 + wi)
        gc, cc = codes(tok, gt_idx), codes(tok, cur_idx)
        kc = codes(tok, cand.reshape(args.K, GRID[0], GRID[1]))
        dc = cdist(kc, gc)
        cq = float(cdist(cc, gc)[0])
        rows["orc"].append(dc.min()); rows["cpy"].append(cq); rows["mdl"].append(dc.mean())
        rows["sprd"].append(dc.max() - dc.min()); rows["std"].append(dc.std())

        imgs = decode_tokens(tok, cand)
        lp = M.eval_batch(imgs, gt)["lpips"]
        cqp = float(M.eval_batch(cur.unsqueeze(0), gt)["lpips"][0])
        rows["orc_lp"].append(lp.min()); rows["cpy_lp"].append(cqp); rows["mdl_lp"].append(lp.mean())
        rows["sprd_lp"].append(lp.max() - lp.min()); rows["std_lp"].append(lp.std())
        rows["sp_cp"].append(spearman(dc, lp))
        if (wi + 1) % 30 == 0:
            print(f"[{wi + 1}/{args.n}]", flush=True)

    R = {k: np.asarray(v, float) for k, v in rows.items()}
    print(f"\n##### which={args.which}  N={args.n}  K={args.K} #####")
    print("=== DISCRIMINATION (lower=closer to GT; want oracle<copy) ===")
    print(f"  code   oracle={R['orc'].mean():.4f} copy={R['cpy'].mean():.4f} "
          f"model={R['mdl'].mean():.4f}  frac oracle<copy={np.mean(R['orc'] < R['cpy']):.3f}")
    print(f"  pixel  oracle={R['orc_lp'].mean():.4f} copy={R['cpy_lp'].mean():.4f} "
          f"model={R['mdl_lp'].mean():.4f}  frac oracle<copy={np.mean(R['orc_lp'] < R['cpy_lp']):.3f}")
    print("=== WITHIN-GROUP SPREAD (GRPO usability; rel = spread/oracle) ===")
    print(f"  code   max-min={R['sprd'].mean():.4f}  std={R['std'].mean():.4f}  "
          f"rel_std={R['std'].mean() / max(1e-9, R['orc'].mean()):.3f}")
    print(f"  pixel  max-min={R['sprd_lp'].mean():.4f}  std={R['std_lp'].mean():.4f}  "
          f"rel_std={R['std_lp'].mean() / max(1e-9, R['orc_lp'].mean()):.3f}")
    print(f"=== code-vs-pixel per-window spearman = {np.nanmean(R['sp_cp']):+.4f} "
          f"(low=orthogonal: code measures something pixel doesn't) ===")
    print("LATENT_PROBE_OK")


if __name__ == "__main__":
    main()
