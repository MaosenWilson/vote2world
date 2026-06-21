#!/usr/bin/env python3
"""Download RT-1 fractal20220817_data shards through public HTTPS URLs.

This is a fallback for machines where gsutil/gcloud storage cannot be installed
or Google apt repositories are unavailable. It uses only the Python standard
library and writes files into the TensorFlow Datasets directory layout expected
by RLVR-World.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = "https://storage.googleapis.com/gresearch/robotics/fractal20220817_data/0.1.0"
DATASET_NAME = "fractal20220817_data"
VERSION = "0.1.0"
NUM_SHARDS = 1024


def download(url: str, dst: Path, retries: int = 8, timeout: int = 120) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] {dst}")
        return

    for attempt in range(1, retries + 1):
        try:
            print(f"[download] {url} -> {dst}")
            with urllib.request.urlopen(url, timeout=timeout) as response:
                with tmp.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            tmp.replace(dst)
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            if tmp.exists():
                tmp.unlink()
            if attempt == retries:
                raise
            sleep_s = min(60, 2**attempt)
            print(f"[retry {attempt}/{retries}] {exc}; sleeping {sleep_s}s", file=sys.stderr)
            time.sleep(sleep_s)


def shard_name(index: int) -> str:
    return f"{DATASET_NAME}-train.tfrecord-{index:05d}-of-{NUM_SHARDS:05d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="data/raw",
        help="Directory that will contain fractal20220817_data/0.1.0.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of TFRecord shards to download from shard 0. Use 1024 for the full dataset.",
    )
    parser.add_argument(
        "--start-shard",
        type=int,
        default=0,
        help="First shard index to download.",
    )
    args = parser.parse_args()

    if args.num_shards < 0 or args.start_shard < 0:
        raise ValueError("shard counts must be non-negative")
    if args.start_shard + args.num_shards > NUM_SHARDS:
        raise ValueError(f"requested shards exceed available range 0..{NUM_SHARDS - 1}")

    root = Path(args.output_root) / DATASET_NAME / VERSION
    download(f"{BASE_URL}/dataset_info.json", root / "dataset_info.json")
    download(f"{BASE_URL}/features.json", root / "features.json")

    for index in range(args.start_shard, args.start_shard + args.num_shards):
        name = shard_name(index)
        download(f"{BASE_URL}/{name}", root / name)

    print(f"[done] wrote files under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
