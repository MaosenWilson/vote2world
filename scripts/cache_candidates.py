"""Cache K candidate next-frames per window with held-out GT quality, transition
consensus, motion, and static-copy signals for offline proxy-quality analysis.
GT here is for evaluation/analysis only (no training)."""
import argparse
import os
import time

import numpy as np

from dor.consensus import consensus_support, motion_magnitude
from dor.constants import CTX, ROOT
from dor.episodes import get_window_tensors, list_episodes, sample_windows
from dor.generation import generate_candidates
from dor.metrics import Metrics
from dor.models import load_action_ranges, load_tokenizer, load_world_model
from dor.tokenization import build_prompt, decode_tokens, encode_features, encode_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_windows", type=int, default=200)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=2000)
    ap.add_argument("--which", default="base", choices=["base", "rlvr"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{ROOT}/outputs/analysis/cache.npz")
    args = ap.parse_args()

    dev = "cuda"
    tok = load_tokenizer(dev)
    model = load_world_model(dev, args.which)
    ar = load_action_ranges(dev)
    M = Metrics(dev)
    wins = sample_windows(list_episodes(), args.n_windows, seed=args.seed)
    print(f"[setup] windows={len(wins)} K={args.K} model={args.which}", flush=True)

    agg = {k: [] for k in ("mae", "mse", "psnr", "lpips", "v", "copy_sim", "tok_repeat")}
    motion, tau_list, pair_list, copy_q = [], [], [], []
    t0 = time.time()
    for wi, (p, s) in enumerate(wins):
        frames, actions = get_window_tensors(p, s, dev)
        gt, cur = frames[CTX], frames[CTX - 1]
        prompt = build_prompt(tok, frames, actions, ar)
        cand = generate_candidates(model, prompt, args.K, temperature=args.temperature,
                                   top_k=args.top_k, seed=args.seed * 100000 + wi)
        imgs = decode_tokens(tok, cand)
        q = M.eval_batch(imgs, gt)
        copy_sim = M.eval_batch(imgs, cur)["lpips"]
        delta = encode_features(tok, imgs) - encode_features(tok, cur.unsqueeze(0))
        v, tau, pmean = consensus_support(delta)
        cur_tok = encode_indices(tok, frames[:CTX])[CTX - 1].reshape(-1)
        tok_repeat = (cand == cur_tok.unsqueeze(0)).all(dim=1).float().cpu().numpy()
        cq = M.eval_batch(cur.unsqueeze(0), gt)

        for k in ("mae", "mse", "psnr", "lpips"):
            agg[k].append(q[k])
        agg["v"].append(v)
        agg["copy_sim"].append(copy_sim)
        agg["tok_repeat"].append(tok_repeat)
        motion.append(motion_magnitude(actions[CTX - 1], ar))
        tau_list.append(tau)
        pair_list.append(pmean)
        copy_q.append([float(cq["mae"][0]), float(cq["psnr"][0]), float(cq["lpips"][0])])
        if (wi + 1) % 20 == 0:
            print(f"[{wi + 1}/{len(wins)}] {time.time() - t0:.0f}s", flush=True)

    out = {k: np.stack(v).astype(np.float32) for k, v in agg.items()}  # [N,K]
    out["motion"] = np.array(motion, np.float32)
    out["tau"] = np.array(tau_list, np.float32)
    out["pair_mean"] = np.array(pair_list, np.float32)
    out["copy_q"] = np.array(copy_q, np.float32)  # [N,3] mae,psnr,lpips of copy-current
    out["meta"] = np.array([args.K, len(wins), args.seed])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **out)
    print(f"[done] saved {args.out}  N={len(wins)} K={args.K} total={time.time() - t0:.0f}s", flush=True)
    print("CACHE_OK", flush=True)


if __name__ == "__main__":
    main()
