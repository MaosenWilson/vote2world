"""Loaders for the frozen visual tokenizer and the autoregressive world model."""
import torch

import dor.compat  # noqa: F401  applies hf shim + puts ivideogpt on sys.path
from dor.constants import AR_PATH, BASE_DIR, RLVR_DIR, TOK_DIR

_WORLD_MODEL_DIRS = {"base": BASE_DIR, "rlvr": RLVR_DIR}


def load_tokenizer(device, dtype=torch.float32):
    """Frozen FSQ-CNN visual tokenizer (pixels <-> visual tokens)."""
    from ivideogpt.tokenizer import CNNFSQModel256

    tok = CNNFSQModel256.from_pretrained(TOK_DIR).to(device=device, dtype=dtype).eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    return tok


def load_world_model(device, which="base", dtype=torch.float32):
    """Autoregressive Llama world model. `which` in {'base', 'rlvr'}."""
    from transformers import AutoModelForCausalLM

    path = _WORLD_MODEL_DIRS[which]
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device).eval()


def load_action_ranges(device):
    """Per-dim action min/max used for discretization. Returns [13, 2]."""
    return torch.load(AR_PATH, map_location=device).float()
