"""End-to-end smoke test: load models, build one prompt, generate K candidates,
decode, score vs held-out GT, and report transition-feature consensus."""
import time

import numpy as np

from dor.consensus import consensus_support, motion_magnitude
from dor.constants import CTX, PROMPT_LEN
from dor.episodes import get_window_tensors, list_episodes
from dor.generation import generate_candidates
from dor.metrics import Metrics
from dor.models import load_action_ranges, load_tokenizer, load_world_model
from dor.tokenization import build_prompt, decode_tokens, encode_features


def main():
    dev = "cuda"
    t0 = time.time()
    tok = load_tokenizer(dev)
    model = load_world_model(dev, "base")
    ar = load_action_ranges(dev)
    print(f"[load] {time.time() - t0:.1f}s  tok+model+ar ok", flush=True)

    paths = list_episodes()
    print(f"[data] {len(paths)} episodes", flush=True)
    frames, actions = get_window_tensors(paths[0], 0, dev)
    prompt = build_prompt(tok, frames, actions, ar)
    print(f"[prompt] len={prompt.shape[0]} (expect {PROMPT_LEN})", flush=True)
    assert prompt.shape[0] == PROMPT_LEN

    K = 6
    t0 = time.time()
    cand = generate_candidates(model, prompt, K, seed=0)
    print(f"[gen] K={K} {time.time() - t0:.1f}s tokens {tuple(cand.shape)}", flush=True)

    imgs = decode_tokens(tok, cand)
    gt, cur = frames[CTX], frames[CTX - 1]
    q = Metrics(dev).eval_batch(imgs, gt)
    print(f"[quality vs GT] PSNR={np.round(q['psnr'], 2)} LPIPS={np.round(q['lpips'], 4)}", flush=True)

    delta = encode_features(tok, imgs) - encode_features(tok, cur.unsqueeze(0))
    v, tau, pmean = consensus_support(delta)
    print(f"[consensus] support={np.round(v, 3)} tau={tau:.4f} pair_mean={pmean:.4f}", flush=True)
    print(f"[motion] {motion_magnitude(actions[CTX - 1], ar):.4f}", flush=True)
    print("SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
