#!/usr/bin/env python3
"""Download public Google Cloud Storage objects under a prefix.

This avoids gsutil/gcloud installation and uses only the Python standard library.
It is intended for public buckets such as:
  gs://gresearch/robotics/fractal20220817_data/0.1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


API_ROOT = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
MEDIA_ROOT = "https://storage.googleapis.com/{bucket}/{name}"


def fetch_json(url: str, retries: int = 8) -> dict:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            if attempt == retries - 1:
                raise
            sleep_s = min(2 ** attempt, 30)
            print(f"list retry {attempt + 1}/{retries}: {exc}; sleeping {sleep_s}s", file=sys.stderr)
            time.sleep(sleep_s)
    raise RuntimeError("unreachable")


def list_objects(bucket: str, prefix: str, limit: int | None = None) -> list[dict]:
    objects: list[dict] = []
    page_token = None
    while True:
        query = {
            "prefix": prefix.rstrip("/") + "/",
            "fields": "items(name,size),nextPageToken",
        }
        if page_token:
            query["pageToken"] = page_token
        url = API_ROOT.format(bucket=bucket) + "?" + urllib.parse.urlencode(query)
        payload = fetch_json(url)
        for item in payload.get("items", []):
            if item["name"].endswith("/"):
                continue
            objects.append(item)
            if limit is not None and len(objects) >= limit:
                return objects
        page_token = payload.get("nextPageToken")
        if not page_token:
            return objects


def download_one(bucket: str, obj: dict, prefix: str, out_dir: Path, retries: int = 8) -> None:
    rel = obj["name"][len(prefix.rstrip("/") + "/") :]
    dst = out_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(obj.get("size", 0))
    if dst.exists() and dst.stat().st_size == expected_size:
        print(f"skip {rel}")
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    url = MEDIA_ROOT.format(bucket=bucket, name=urllib.parse.quote(obj["name"]))
    for attempt in range(retries):
        try:
            print(f"download {rel} ({expected_size} bytes)")
            with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, dst)
            return
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            if attempt == retries - 1:
                raise
            sleep_s = min(2 ** attempt, 30)
            print(f"download retry {attempt + 1}/{retries}: {exc}; sleeping {sleep_s}s", file=sys.stderr)
            time.sleep(sleep_s)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="gresearch")
    parser.add_argument("--prefix", default="robotics/fractal20220817_data/0.1.0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None, help="download only first N objects")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    objects = list_objects(args.bucket, args.prefix, args.limit)
    total = sum(int(obj.get("size", 0)) for obj in objects)
    print(f"objects={len(objects)} bytes={total} gib={total / 1024**3:.2f}")
    for obj in objects[:20]:
        print(f"{obj['name']} {obj.get('size', '0')}")
    if len(objects) > 20:
        print(f"... {len(objects) - 20} more")
    if args.list_only:
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for obj in objects:
        download_one(args.bucket, obj, args.prefix, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
