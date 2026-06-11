#!/usr/bin/env python3
"""Convert a small RT-1 TFDS TFRecord subset to RLVR-World-style NPZ files.

This fallback avoids TensorFlow / TFDS. It parses TFRecord Example messages with
the stdlib parser from `scripts.analysis.audit_rt1_tfrecord_stdlib`, decodes
JPEG frames with Pillow, and writes:

  {split}_eps_{index:08d}.npz

with keys:

  image:  (T, H, W, 3), uint8
  action: (T, 13), float32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analysis.audit_rt1_tfrecord_stdlib import (  # noqa: E402
    ACTION_DIMS,
    ACTION_FIELDS,
    iter_tfrecord,
    maybe_action_key,
    parse_record,
    reshape_flat,
)


def decode_jpeg(data: bytes) -> np.ndarray:
    from io import BytesIO

    image = Image.open(BytesIO(data)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def convert_record(record: bytes) -> tuple[np.ndarray, np.ndarray] | None:
    parsed = parse_record(record)
    features = parsed.get("features", {})
    keys = sorted(features.keys())
    image_key = next((key for key in keys if key == "steps/observation/image"), None)
    if image_key is None:
        return None
    image_values = features[image_key].get("bytes", [])
    if not image_values:
        return None

    action_rows = {}
    for field in ACTION_FIELDS:
        key = maybe_action_key(keys, field)
        if key is None:
            raise KeyError(f"missing action field {field}")
        values = []
        feature = features[key]
        if "float" in feature:
            values = [float(x) for x in feature["float"]]
        elif "int" in feature:
            values = [float(x) for x in feature["int"]]
        rows = reshape_flat(values, ACTION_DIMS[field])
        action_rows[field] = rows

    length = len(image_values)
    for field, rows in action_rows.items():
        if len(rows) != length:
            raise ValueError(f"{field} length {len(rows)} != image length {length}")

    frames = np.stack([decode_jpeg(x) for x in image_values], axis=0)
    actions = []
    for i in range(length):
        row = []
        for field in ACTION_FIELDS:
            row.extend(action_rows[field][i])
        actions.append(row)
    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.shape != (length, 13):
        raise ValueError(f"action shape mismatch: {action_array.shape}")
    return frames, action_array


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("fractal20220817_data-train.tfrecord-*"))
    if not files:
        raise FileNotFoundError(f"no train shards found under {args.data_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for shard in files:
        for record in iter_tfrecord(shard):
            result = convert_record(record)
            if result is None:
                continue
            frames, actions = result
            out = args.output_dir / f"{args.split}_eps_{converted:08d}.npz"
            if out.exists():
                converted += 1
                if converted >= args.max_episodes:
                    print(f"converted={converted} output_dir={args.output_dir}")
                    return 0
                continue
            np.savez_compressed(out, image=frames, action=actions)
            print(f"wrote {out} image={frames.shape} action={actions.shape}")
            converted += 1
            if converted >= args.max_episodes:
                print(f"converted={converted} output_dir={args.output_dir}")
                return 0
    print(f"converted={converted} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
