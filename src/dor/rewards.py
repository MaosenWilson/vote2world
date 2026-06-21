"""Reward and advantage shaping for GRPO.

The reward stays a pure RLVR-World verifiable GT signal (negative LPIPS vs the
ground-truth next frame). Intra-group consensus only *shapes the advantage*:

  gt_only      A = z(r_gt)                                  RLVR-World baseline
  hybrid_add   A = z( z(r_gt) + lam * z(c) )                additive reward shaping  (ablation arm A)
  hybrid_mult  A = z( z(r_gt) * max(delta, 1 + beta*z(c)) ) advantage modulation     (main method B)

Why (B) multiplies the consensus weight onto the GT-signed advantage:
  * Collective-hallucination guard: with w = max(delta, 1 + beta*z(c)) > 0 the
    sign of z(r_gt) is preserved, so a high-consensus-but-GT-bad candidate cannot
    be flipped into a positive update.
  * Graceful degradation: beta = 0  =>  w = 1  =>  exactly gt_only.
The reward r_gt is never modified, so the verifiable objective is preserved.
"""
import numpy as np

MODES = ("gt_only", "hybrid_add", "hybrid_mult")


def _zscore(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + eps)


def static_copy_gate(motion, copy_sim, tau_motion=0.5, tau_copy=0.10):
    """1.0 keep / 0.0 gate. Under non-trivial action, gate near-static-copy candidates.

    copy_sim [K] = LPIPS(candidate, current frame); small => barely changed.
    """
    copy_sim = np.asarray(copy_sim, float)
    if motion <= tau_motion:
        return np.ones_like(copy_sim)
    return np.where(copy_sim < tau_copy, 0.0, 1.0)


def shape_advantage(r_gt, consensus=None, mode="gt_only", *, lam=0.5, beta=0.5,
                    delta=0.1, motion=None, copy_sim=None, gate_penalty=2.0,
                    tau_motion=0.5, tau_copy=0.10):
    """Compute the group-normalised GRPO advantage. r_gt is GT-based and never modified.

    Args:
      r_gt:      [K] GT reward (e.g. -LPIPS vs GT next frame).
      consensus: [K] transition-feature consensus support (required for hybrid_*).
      mode:      one of MODES.
      motion, copy_sim: enable the static-copy gate when both are provided.
    Returns:
      (advantage [K] np.float64, info dict).
    """
    r_gt = np.asarray(r_gt, float)
    zr = _zscore(r_gt)
    info = {"zr": zr}

    if mode == "gt_only":
        shaped = zr
    elif mode in ("hybrid_add", "hybrid_mult"):
        if consensus is None:
            raise ValueError(f"mode {mode!r} requires `consensus`")
        zc = _zscore(consensus)
        info["zc"] = zc
        if mode == "hybrid_add":
            shaped = zr + lam * zc
        else:  # hybrid_mult (B)
            w = np.maximum(delta, 1.0 + beta * zc)
            info["w"] = w
            shaped = zr * w
    else:
        raise ValueError(f"unknown mode: {mode!r} (expected one of {MODES})")

    if motion is not None and copy_sim is not None:
        gate = static_copy_gate(motion, copy_sim, tau_motion, tau_copy)
        shaped = shaped - gate_penalty * (1.0 - gate)
        info["gate"] = gate

    adv = _zscore(shaped)  # group-normalise so step scale is comparable across modes
    info["adv"] = adv
    return adv, info
