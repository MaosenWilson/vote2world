"""Export real RT-1 frames for the method-figure replica:
 - CTX history/context frames (the dataset input filmstrip)
 - the held-out GT next frame
 - K decoded candidate next-frame predictions
Picks a high-motion window so candidates look distinct."""
import os

from PIL import Image

from dor.constants import CTX, ROOT
from dor.consensus import motion_magnitude
from dor.episodes import get_window_tensors, list_episodes, sample_windows
from dor.generation import generate_candidates
from dor.models import load_action_ranges, load_tokenizer, load_world_model
from dor.tokenization import build_prompt, decode_tokens


def save(t, path):
    a = (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(a).save(path)


def main():
    dev = "cuda"
    tok = load_tokenizer(dev)
    model = load_world_model(dev, "base")
    ar = load_action_ranges(dev)

    best = None
    for p, s in sample_windows(list_episodes(), 60, seed=7):
        _, ac = get_window_tensors(p, s, dev)
        mm = motion_magnitude(ac[CTX - 1], ar)
        if best is None or mm > best[0]:
            best = (mm, p, s)
    mm, p, s = best

    fr, ac = get_window_tensors(p, s, dev)
    prompt = build_prompt(tok, fr, ac, ar)
    cand = generate_candidates(model, prompt, 8, temperature=1.0, top_k=2000, seed=123)
    imgs = decode_tokens(tok, cand)

    out = f"{ROOT}/outputs/figures/frames"
    os.makedirs(out, exist_ok=True)
    for i in range(CTX):
        save(fr[i], f"{out}/hist_{i}.png")
    save(fr[CTX], f"{out}/gt.png")
    for i in range(5):
        save(imgs[i], f"{out}/cand_{i}.png")
    print(f"motion={mm:.3f} window=({os.path.basename(p)},{s}) saved to {out}", flush=True)
    print("EXPORT_OK", flush=True)


if __name__ == "__main__":
    main()
