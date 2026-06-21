"""Intra-group consensus in transition-feature space (the redefined 'vote').

Unlike TTRL, the vote does not produce a pseudo-label that replaces GT. It
produces a continuous per-candidate consensus-support score that later reshapes
the GRPO advantage (see `dor.rewards`).
"""
import torch
import torch.nn.functional as F

from dor.constants import MOTION_DIMS


def consensus_support(delta, tau_quantile=0.30):
    """Leave-one-out neighborhood support in transition-feature space.

    delta [K, C] = phi(candidate) - phi(current) transition features.
    Returns (support[K] in [0,1], tau, pairwise_dist_mean). support[i] = fraction
    of peers whose cosine distance to i is within the tau-quantile threshold.
    """
    d = F.normalize(delta, dim=-1)
    dist = 1.0 - d @ d.t()                       # [K,K] cosine distance
    K = dist.shape[0]
    off = dist[~torch.eye(K, dtype=torch.bool, device=dist.device)].reshape(K, K - 1)
    tau = torch.quantile(off.reshape(-1), tau_quantile)
    support = (off <= tau).float().mean(dim=1)   # [K]
    return support.detach().cpu().numpy(), float(tau), float(off.mean())


def motion_magnitude(action, ar):
    """Normalized |a_t| over rotation_delta + world_vector dims. action [13] -> float."""
    span = ar[:, 1] - ar[:, 0] + 1e-8
    a = (action / span)[MOTION_DIMS]
    return float(a.norm().item())
