"""RT-1 episode loading and context-window sampling.

A window is frames[start : start + CTX + 1]; the target (held-out GT) next frame
is frame start + CTX. GT is used by the reward (RLVR-World), not withheld.
"""
import glob

import numpy as np
import torch

from dor.constants import CTX, DATA_DIR


def list_episodes():
    return sorted(glob.glob(f"{DATA_DIR}/*.npz"))


def load_episode(path):
    d = np.load(path, allow_pickle=True)
    return d["image"], d["action"].astype(np.float32)  # [T,256,320,3] uint8, [T,13]


def sample_windows(paths, n_windows, seed=0, stride=5):
    """Return list of (episode_path, start_idx), shuffled, truncated to n_windows."""
    rng = np.random.default_rng(seed)
    wins = []
    for p in paths:
        T = np.load(p, allow_pickle=True)["image"].shape[0]
        wins += [(p, s) for s in range(0, T - (CTX + 1), stride)]
    rng.shuffle(wins)
    return wins[:n_windows]


def get_window_tensors(path, start, device):
    """Return (frames [CTX+1,3,256,320] in [0,1], actions [CTX+1,13])."""
    img, act = load_episode(path)
    seg = img[start:start + CTX + 1]
    acts = act[start:start + CTX + 1]
    frames = torch.from_numpy(seg).float().div(255.0).permute(0, 3, 1, 2).to(device)
    actions = torch.from_numpy(acts).to(device)
    return frames, actions
