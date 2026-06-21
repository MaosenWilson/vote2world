"""Candidate next-frame sampling via HF .generate() (bypasses vllm/verl on Blackwell)."""
import torch

from dor.constants import EOS, TPF, VTOK


@torch.no_grad()
def generate_candidates(model, prompt, K, temperature=1.0, top_k=2000, seed=0):
    """prompt [PROMPT_LEN] -> candidate visual tokens [K, TPF] long."""
    torch.manual_seed(seed)
    out = model.generate(
        input_ids=prompt.unsqueeze(0),
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        num_return_sequences=K,
        max_new_tokens=TPF,      # only the visual tokens
        min_new_tokens=TPF,
        eos_token_id=None,       # ignore EOS like the official eval
        pad_token_id=EOS,
    )
    gen = out[:, prompt.shape[0]:prompt.shape[0] + TPF]  # [K, TPF]
    return gen.clamp(0, VTOK - 1).long()
