"""Measure the tokenizer reconstruction floor.

Any token-based prediction is bounded below by decode(encode(.)) error, so
comparing the model to a *pixel*-copy baseline is unfair. This reports, over
sampled windows:
  floor          LPIPS(decode(encode(gt)),  gt)  -> best any token prediction can do
  tokenized_copy LPIPS(decode(encode(cur)), gt)  -> fair copy baseline (token space)
  pixel_copy     LPIPS(cur,                 gt)  -> unfair baseline (pixel space)
"""
import argparse

import numpy as np

from dor.constants import CTX
from dor.episodes import get_window_tensors, list_episodes, sample_windows
from dor.metrics import Metrics
from dor.models import load_tokenizer
from dor.tokenization import decode_tokens, encode_indices


def roundtrip(tok, frame):
    """frame [3,H,W] -> decode(encode(frame)) [3,H,W]."""
    idx = encode_indices(tok, frame.unsqueeze(0)).reshape(1, -1)  # [1,320]
    return decode_tokens(tok, idx)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    dev = "cuda"
    tok = load_tokenizer(dev)
    M = Metrics(dev)
    floor, tcopy, pcopy = [], [], []
    for p, s in sample_windows(list_episodes(), args.n, seed=0):
        fr, _ = get_window_tensors(p, s, dev)
        gt, cur = fr[CTX], fr[CTX - 1]
        floor.append(float(M.eval_batch(roundtrip(tok, gt).unsqueeze(0), gt)["lpips"][0]))
        tcopy.append(float(M.eval_batch(roundtrip(tok, cur).unsqueeze(0), gt)["lpips"][0]))
        pcopy.append(float(M.eval_batch(cur.unsqueeze(0), gt)["lpips"][0]))
    print(f"floor          LPIPS(decode(encode(gt)),  gt) = {np.mean(floor):.4f} +- {np.std(floor):.4f}")
    print(f"tokenized_copy LPIPS(decode(encode(cur)), gt) = {np.mean(tcopy):.4f} +- {np.std(tcopy):.4f}")
    print(f"pixel_copy     LPIPS(cur,                 gt) = {np.mean(pcopy):.4f} +- {np.std(pcopy):.4f}")
    print("FLOOR_OK")


if __name__ == "__main__":
    main()
