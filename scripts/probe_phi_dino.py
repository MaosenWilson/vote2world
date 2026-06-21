"""phi-space probe (DINOv2): does a tokenizer-decode-robust feature space reflect
dynamics better than pixel LPIPS, before we commit to a phi reward?

Self-contained (like probe_tokenizer_floor): generate K candidates per window,
decode, embed with DINOv2. Reports over N windows (K cands each, temp 0.5):
  (1) FLOOR  phi_floor = phid(decode(encode(gt)), gt)  vs  phi_motion = phid(gt, cur)
             pixel side is inverted (floor 0.053 > motion 0.040); phi must un-invert.
  (2) CORR   spearman(phid(cand,gt), token_err)  vs  spearman(lpips, token_err)
  (3) DISCR  oracle(min phid) < copy(phid(cur,gt)) < model(mean phid) ?
Two phi variants: 'frame' (DINO(pred) vs DINO(gt)) and 'motion' (delta vs cur in phi).
"""
import argparse
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import numpy as np
import torch
import torch.nn.functional as Fnn

from dor.constants import CTX
from dor.episodes import get_window_tensors, list_episodes, sample_windows
from dor.generation import generate_candidates
from dor.metrics import Metrics, spearman
from dor.models import load_action_ranges, load_tokenizer, load_world_model
from dor.tokenization import build_prompt, decode_tokens, encode_indices

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Phi:
    """DINOv2 CLS embedder. imgs in [0,1] [B,3,H,W] -> [B,D]."""

    def __init__(self, dev, name="facebook/dinov2-small"):
        from transformers import Dinov2Model
        self.m = Dinov2Model.from_pretrained(name).to(dev).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.mean, self.std = _MEAN.to(dev), _STD.to(dev)

    @torch.no_grad()
    def __call__(self, imgs):
        x = Fnn.interpolate(imgs, size=224, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        h = self.m(pixel_values=x).last_hidden_state  # [B,1+P,D]
        return h[:, 0], h[:, 1:]  # cls [B,D], patch [B,P,D]


def cos_d(a, b):
    """row-wise cosine distance 1-cos; a [B,D], b [B,D] or [1,D] -> [B]."""
    a, b = Fnn.normalize(a, dim=-1), Fnn.normalize(b, dim=-1)
    return (1 - (a * b).sum(-1)).cpu().numpy()


def patch_d(a, b):
    """mean over patches of cosine distance; a [B,P,D], b [1,P,D] -> [B]."""
    a, b = Fnn.normalize(a, dim=-1), Fnn.normalize(b, dim=-1)
    return (1 - (a * b).sum(-1)).mean(-1).cpu().numpy()


def msd(x):
    x = np.asarray(x, float)
    n = max(1, int(np.sum(~np.isnan(x))))
    return f"{np.nanmean(x):+.4f} ± {np.nanstd(x) / np.sqrt(n):.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top_k", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda"
    tok = load_tokenizer(dev)
    model = load_world_model(dev, "base")
    ar = load_action_ranges(dev)
    M = Metrics(dev)
    phi = Phi(dev)
    print("[setup] DINOv2 + world model loaded", flush=True)

    # per-variant accumulators: 'cls', 'patch'
    floor = {k: [] for k in ("cls", "patch")}
    motion = {k: [] for k in ("cls", "patch")}
    sp = {k: [] for k in ("cls", "patch", "lpips")}
    orc = {k: [] for k in ("cls", "patch")}
    cpy = {k: [] for k in ("cls", "patch")}
    for wi, (p, s) in enumerate(sample_windows(list_episodes(), args.n, seed=args.seed)):
        frames, actions = get_window_tensors(p, s, dev)
        gt, cur = frames[CTX], frames[CTX - 1]
        gt_tok = encode_indices(tok, gt.unsqueeze(0)).reshape(-1)
        rt_gt = decode_tokens(tok, gt_tok.unsqueeze(0))
        prompt = build_prompt(tok, frames, actions, ar)
        cand = generate_candidates(model, prompt, args.K, temperature=args.temperature,
                                   top_k=args.top_k, seed=args.seed * 100000 + wi)
        imgs = decode_tokens(tok, cand)

        c_gt, p_gt = phi(gt.unsqueeze(0))
        c_cur, p_cur = phi(cur.unsqueeze(0))
        c_rt, p_rt = phi(rt_gt)
        c_im, p_im = phi(imgs)
        lp = M.eval_batch(imgs, gt)["lpips"]
        tok_err = (cand != gt_tok.unsqueeze(0)).float().mean(1).cpu().numpy()
        sp["lpips"].append(spearman(lp, tok_err))

        for k, (d_rt, d_gt, d_cur, d_im, df) in (
            ("cls", (c_rt, c_gt, c_cur, c_im, cos_d)),
            ("patch", (p_rt, p_gt, p_cur, p_im, patch_d)),
        ):
            floor[k].append(float(df(d_rt, d_gt)[0]))
            motion[k].append(float(df(d_gt, d_cur)[0]))
            phid = df(d_im, d_gt)
            sp[k].append(spearman(phid, tok_err))
            orc[k].append(float(phid.min()))
            cpy[k].append(float(df(d_cur, d_gt)[0]))
        if (wi + 1) % 10 == 0:
            print(f"[{wi + 1}/{args.n}]", flush=True)

    print("\n=== (1) FLOOR  ratio = phi_floor / phi_motion  (<1 = phi un-inverts pixel pathology) ===")
    for k in ("cls", "patch"):
        ff, mf = np.mean(floor[k]), np.mean(motion[k])
        print(f"  {k:<6} floor={ff:.4f}  motion={mf:.4f}  ratio={ff / max(1e-9, mf):.3f}")
    print("\n=== (2) CORR with token_err (per-window spearman, higher=better proxy) ===")
    for k in ("cls", "patch", "lpips"):
        print(f"  {k:<6} {msd(sp[k])}   frac_pos={np.mean(np.asarray(sp[k]) > 0):.3f}")
    print("\n=== (3) DISCRIMINATION  oracle(min phid) vs copy(phid(cur,gt)) ===")
    for k in ("cls", "patch"):
        fr = np.mean(np.asarray(orc[k]) < np.asarray(cpy[k]))
        print(f"  {k:<6} oracle={np.mean(orc[k]):.4f}  copy={np.mean(cpy[k]):.4f}  frac oracle<copy={fr:.3f}")
    print("PHI_PROBE_OK")


if __name__ == "__main__":
    main()
