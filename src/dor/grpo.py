"""Lean GRPO training/eval loop (bypasses verl/vllm; runs on Blackwell sm_120).

The verifiable GT distance D is selectable via `reward`:
  pixel   -LPIPS(decode(cand), gt)                RLVR-World baseline
  code    -RMS(codes(cand) - codes(gt))           FSQ code space, no decode
  hybrid  alpha*z(pixel) + (1-alpha)*z(code)       z-score fusion
Intra-group consensus only reshapes the advantage (see dor.rewards.MODES); the
GT reward r_gt itself is never modified.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from dor.consensus import consensus_support, motion_magnitude
from dor.constants import CTX, GRID, TPF
from dor.episodes import get_window_tensors, list_episodes, sample_windows
from dor.generation import generate_candidates
from dor.metrics import Metrics
from dor.models import load_action_ranges, load_tokenizer, load_world_model
from dor.rewards import _zscore, shape_advantage
from dor.tokenization import build_prompt, decode_tokens, encode_features, encode_indices


def code_vec(tok, idx):
    """idx [B,16,20] long -> flat post-quant FSQ code [B, 16*20*5] float (no decode)."""
    return tok.indices_to_codes(idx).float().reshape(idx.shape[0], -1)


def code_rms(tok, cand, gt_idx):
    """cand [K,TPF] long, gt_idx [1,16,20] -> per-candidate code-space RMS [K] np."""
    cc = code_vec(tok, cand.reshape(cand.shape[0], GRID[0], GRID[1]))
    gc = code_vec(tok, gt_idx)
    return (cc - gc).pow(2).mean(1).sqrt().detach().cpu().numpy()


def gt_reward(kind, metrics, tok, cand, imgs, gt, gt_idx, alpha=0.5):
    """Verifiable GT reward r_gt [K] (higher=closer to GT). Never consensus-shaped."""
    if kind == "pixel":
        return -metrics.eval_batch(imgs, gt)["lpips"]
    if kind == "code":
        return -code_rms(tok, cand, gt_idx)
    if kind == "hybrid":
        r_pix = -metrics.eval_batch(imgs, gt)["lpips"]
        r_cod = -code_rms(tok, cand, gt_idx)
        return alpha * _zscore(r_pix) + (1.0 - alpha) * _zscore(r_cod)
    raise ValueError(f"unknown reward kind: {kind!r}")


def seq_logp(model, prompt, gen, autocast=True):
    """prompt [P], gen [K, TPF] -> (per-seq summed logprob [K], per-token logprob [K, TPF])."""
    K = gen.shape[0]
    full = torch.cat([prompt.unsqueeze(0).expand(K, -1), gen], dim=1)  # [K, P+TPF]
    P = prompt.shape[0]
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast else torch.enable_grad()
    with ctx:
        logits = model(full).logits
    pred = logits[:, P - 1:P - 1 + TPF, :].float()
    logp = F.log_softmax(pred, dim=-1)
    tok_logp = logp.gather(-1, gen.unsqueeze(-1)).squeeze(-1)  # [K, TPF]
    return tok_logp.sum(dim=1), tok_logp


@torch.no_grad()
def eval_model(model, tok, metrics, wins, device, K=8, seed=999):
    """Held-out LPIPS / PSNR / code-RMS / token-repeat over eval windows."""
    lps, pss, crm, reps = [], [], [], []
    ar = load_action_ranges(device)
    for wi, (p, s) in enumerate(wins):
        frames, actions = get_window_tensors(p, s, device)
        gt = frames[CTX]
        prompt = build_prompt(tok, frames, actions, ar)
        cand = generate_candidates(model, prompt, K, seed=seed + wi)
        q = metrics.eval_batch(decode_tokens(tok, cand), gt)
        lps.append(q["lpips"].mean())
        pss.append(q["psnr"].mean())
        gt_idx = encode_indices(tok, gt.unsqueeze(0))
        crm.append(code_rms(tok, cand, gt_idx).mean())
        cur_tok = encode_indices(tok, frames[:CTX])[CTX - 1].reshape(-1)
        reps.append((cand == cur_tok.unsqueeze(0)).all(dim=1).float().mean().item())
    return float(np.mean(lps)), float(np.mean(pss)), float(np.mean(crm)), float(np.mean(reps))


def train(mode, *, reward="pixel", alpha=0.5, steps=40, K=8, batch_windows=2,
          train_windows=24, eval_windows=12, lr=1e-5, lam=0.5, beta=0.5, kl=0.0,
          eval_every=10, seed=0, device="cuda"):
    """Run GRPO for one (reward, mode) design; returns a log dict of curves."""
    tok = load_tokenizer(device)
    model = load_world_model(device, "base", dtype=torch.float32)
    model.config.use_cache = False
    model.train()
    ref = load_world_model(device, "base", dtype=torch.float32).eval() if kl > 0 else None
    ar = load_action_ranges(device)
    metrics = Metrics(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    allw = sample_windows(list_episodes(), train_windows + eval_windows, seed=1)
    train_w, eval_w = allw[:train_windows], allw[train_windows:]

    log = {"step": [], "reward_mean": [], "adv_abs_mean": [], "frac_pos": [],
           "eval_lpips": [], "eval_psnr": [], "eval_code_rms": [], "eval_repeat": []}

    def _log_eval(step, r_mean=0.0, adv_abs=0.0, frac_pos=0.0):
        lp, ps, cr, rp = eval_model(model, tok, metrics, eval_w, device, K=K)
        log["step"].append(step)
        log["reward_mean"].append(r_mean)
        log["adv_abs_mean"].append(adv_abs)
        log["frac_pos"].append(frac_pos)
        log["eval_lpips"].append(lp)
        log["eval_psnr"].append(ps)
        log["eval_code_rms"].append(cr)
        log["eval_repeat"].append(rp)
        print(f"[{reward}/{mode}] step {step} evalLPIPS={lp:.4f} codeRMS={cr:.4f} "
              f"PSNR={ps:.2f} repeat={rp:.3f}", flush=True)

    _log_eval(0)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(train_w), size=batch_windows)
        opt.zero_grad()
        rwd_acc, adv_acc, pos_acc, nseq = 0.0, 0.0, 0.0, 0
        for bi in idx:
            p, s = train_w[bi]
            frames, actions = get_window_tensors(p, s, device)
            gt, cur = frames[CTX], frames[CTX - 1]
            prompt = build_prompt(tok, frames, actions, ar)
            with torch.no_grad():
                cand = generate_candidates(model, prompt, K, seed=step * 1000 + int(bi))
                imgs = decode_tokens(tok, cand)
                gt_idx = encode_indices(tok, gt.unsqueeze(0))
                r_gt = gt_reward(reward, metrics, tok, cand, imgs, gt, gt_idx, alpha)
                if mode == "gt_only":
                    adv, _ = shape_advantage(r_gt, mode=mode)
                else:
                    copy_sim = metrics.eval_batch(imgs, cur)["lpips"]
                    delta = encode_features(tok, imgs) - encode_features(tok, cur.unsqueeze(0))
                    v, _, _ = consensus_support(delta)
                    mm = motion_magnitude(actions[CTX - 1], ar)
                    adv, _ = shape_advantage(r_gt, v, mode=mode, lam=lam, beta=beta,
                                             motion=mm, copy_sim=copy_sim)
                adv_t = torch.tensor(adv, device=device, dtype=torch.float32)
            logp_sum, tok_logp = seq_logp(model, prompt, cand)
            pg = -(adv_t * logp_sum).mean()
            if ref is not None:
                with torch.no_grad():
                    _, ref_tok = seq_logp(ref, prompt, cand)
                pg = pg + kl * (tok_logp - ref_tok).mean()
            (pg / len(idx)).backward()
            rwd_acc += float(np.mean(r_gt))
            adv_acc += float(np.mean(np.abs(adv)))
            pos_acc += float(np.mean(adv > 0))
            nseq += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0 or step == steps:
            _log_eval(step, rwd_acc / nseq, adv_acc / nseq, pos_acc / nseq)
            print(f"[{reward}/{mode}] {step}/{steps} {time.time() - t0:.0f}s r_gt={rwd_acc / nseq:.4f}", flush=True)

    del model, ref, tok
    torch.cuda.empty_cache()
    return log
