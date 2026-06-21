#!/usr/bin/env python3
"""Audit RT-1 TFDS TFRecord files with only the Python standard library.

The script parses TFRecord framing and the TensorFlow Example/SequenceExample
protobuf wire format directly. It is intentionally narrow: enough to inspect
RT-1 action/image fields without installing TensorFlow on the audit server.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ACTION_FIELDS = [
    "world_vector",
    "rotation_delta",
    "gripper_closedness_action",
    "base_displacement_vector",
    "base_displacement_vertical_rotation",
    "terminate_episode",
]

ACTION_DIMS = {
    "world_vector": 3,
    "rotation_delta": 3,
    "gripper_closedness_action": 1,
    "base_displacement_vector": 2,
    "base_displacement_vertical_rotation": 1,
    "terminate_episode": 3,
}


class Stat:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min = math.inf
        self.max = -math.inf
        self.zero = 0

    def add(self, value: float) -> None:
        self.n += 1
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        if abs(value) < 1e-12:
            self.zero += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    def as_dict(self) -> dict[str, float | int | None]:
        if self.n == 0:
            return {"n": 0, "min": None, "max": None, "mean": None, "std": None, "zero_ratio": None}
        return {
            "n": self.n,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "std": math.sqrt(self.m2 / self.n),
            "zero_ratio": self.zero / self.n,
        }


def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        b = buf[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not b & 0x80:
            return value, pos
        shift += 7


def parse_fields(buf: bytes) -> list[tuple[int, int, Any]]:
    pos = 0
    out: list[tuple[int, int, Any]] = []
    while pos < len(buf):
        key, pos = read_varint(buf, pos)
        field = key >> 3
        wire = key & 7
        if wire == 0:
            value, pos = read_varint(buf, pos)
        elif wire == 1:
            value = buf[pos : pos + 8]
            pos += 8
        elif wire == 2:
            length, pos = read_varint(buf, pos)
            value = buf[pos : pos + length]
            pos += length
        elif wire == 5:
            value = buf[pos : pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        out.append((field, wire, value))
    return out


def parse_feature(feature_msg: bytes) -> dict[str, list[Any]]:
    values: dict[str, list[Any]] = {}
    for field, wire, value in parse_fields(feature_msg):
        if wire != 2:
            continue
        if field == 1:  # bytes_list
            vals = []
            for f, w, v in parse_fields(value):
                if f == 1 and w == 2:
                    vals.append(v)
            values["bytes"] = vals
        elif field == 2:  # float_list
            vals = []
            for f, w, v in parse_fields(value):
                if f == 1 and w == 5:
                    vals.append(struct.unpack("<f", v)[0])
                elif f == 1 and w == 2:
                    vals.extend(struct.unpack("<" + "f" * (len(v) // 4), v))
            values["float"] = vals
        elif field == 3:  # int64_list
            vals = []
            for f, w, v in parse_fields(value):
                if f == 1 and w == 0:
                    vals.append(v)
                elif f == 1 and w == 2:
                    p = 0
                    while p < len(v):
                        item, p = read_varint(v, p)
                        vals.append(item)
            values["int"] = vals
    return values


def parse_features(features_msg: bytes) -> dict[str, dict[str, list[Any]]]:
    features = {}
    for field, wire, entry in parse_fields(features_msg):
        if field != 1 or wire != 2:
            continue
        key = None
        feature = None
        for f, w, v in parse_fields(entry):
            if f == 1 and w == 2:
                key = v.decode("utf-8", errors="replace")
            elif f == 2 and w == 2:
                feature = parse_feature(v)
        if key is not None and feature is not None:
            features[key] = feature
    return features


def parse_feature_lists(feature_lists_msg: bytes) -> dict[str, list[dict[str, list[Any]]]]:
    feature_lists = {}
    for field, wire, entry in parse_fields(feature_lists_msg):
        if field != 1 or wire != 2:
            continue
        key = None
        feature_list = []
        for f, w, v in parse_fields(entry):
            if f == 1 and w == 2:
                key = v.decode("utf-8", errors="replace")
            elif f == 2 and w == 2:
                for ff, ww, vv in parse_fields(v):
                    if ff == 1 and ww == 2:
                        feature_list.append(parse_feature(vv))
        if key is not None:
            feature_lists[key] = feature_list
    return feature_lists


def parse_record(record: bytes) -> dict[str, Any]:
    top = parse_fields(record)
    parsed: dict[str, Any] = {"kind": "unknown", "context": {}, "feature_lists": {}, "features": {}}
    for field, wire, value in top:
        if wire != 2:
            continue
        if field == 1:
            parsed["context"] = parse_features(value)
        elif field == 2:
            parsed["feature_lists"] = parse_feature_lists(value)
    if parsed["feature_lists"]:
        parsed["kind"] = "SequenceExample"
        return parsed
    if top and top[0][0] == 1 and top[0][1] == 2:
        parsed["kind"] = "Example"
        parsed["features"] = parse_features(top[0][2])
    return parsed


def iter_tfrecord(path: Path, max_records: int | None = None):
    with path.open("rb") as f:
        count = 0
        while True:
            header = f.read(12)
            if not header:
                break
            if len(header) != 12:
                raise ValueError(f"truncated tfrecord header in {path}")
            length = struct.unpack("<Q", header[:8])[0]
            data = f.read(length)
            f.read(4)  # data crc
            if len(data) != length:
                raise ValueError(f"truncated tfrecord payload in {path}")
            yield data
            count += 1
            if max_records is not None and count >= max_records:
                break


def first_values(feature: dict[str, list[Any]], limit: int = 8) -> list[Any]:
    for kind in ("float", "int", "bytes"):
        if kind in feature:
            vals = feature[kind]
            if kind == "bytes":
                return [f"<bytes:{len(v)}>" for v in vals[:limit]]
            return vals[:limit]
    return []


def feature_values(feature: dict[str, list[Any]]) -> list[float]:
    if "float" in feature:
        return [float(x) for x in feature["float"]]
    if "int" in feature:
        return [float(x) for x in feature["int"]]
    return []


def reshape_flat(values: list[float], dim: int) -> list[list[float]]:
    if dim <= 0:
        return []
    usable = len(values) - (len(values) % dim)
    return [values[i : i + dim] for i in range(0, usable, dim)]


def maybe_action_key(keys: list[str], field: str) -> str | None:
    suffixes = [f"/action/{field}", f"action/{field}", field]
    for suffix in suffixes:
        matches = [key for key in keys if key.endswith(suffix)]
        if matches:
            return matches[0]
    return None


def audit(data_dir: Path, max_episodes: int) -> dict[str, Any]:
    files = sorted(data_dir.glob("fractal20220817_data-train.tfrecord-*"))
    if not files:
        raise FileNotFoundError(f"no train tfrecord files under {data_dir}")

    action_stats = defaultdict(Stat)
    norm_stats = defaultdict(Stat)
    terminate_counter = Counter()
    samples = []
    schema = None
    episodes = 0
    steps = 0

    for path in files:
        for raw in iter_tfrecord(path):
            parsed = parse_record(raw)
            feature_lists = parsed.get("feature_lists", {})
            flat_features = parsed.get("features", {})
            keys = sorted(feature_lists.keys() or flat_features.keys())
            if schema is None:
                schema = {
                    "record_kind": parsed["kind"],
                    "first_file": str(path),
                    "feature_list_keys": sorted(feature_lists.keys()),
                    "feature_keys": sorted(flat_features.keys()),
                    "context_keys": sorted(parsed.get("context", {}).keys()),
                }

            action_keys = {field: maybe_action_key(keys, field) for field in ACTION_FIELDS}
            image_key = None
            for candidate in keys:
                if candidate.endswith("/observation/image") or candidate.endswith("observation/image") or candidate == "image":
                    image_key = candidate
                    break

            ep_steps = 0
            if image_key and image_key in feature_lists:
                ep_steps = len(feature_lists[image_key])
            elif any(k and k in feature_lists for k in action_keys.values()):
                ep_steps = max(len(feature_lists[k]) for k in action_keys.values() if k)
            elif image_key and image_key in flat_features:
                ep_steps = len(flat_features[image_key].get("bytes", []))
            elif any(k and k in flat_features for k in action_keys.values()):
                candidates = []
                for field, key in action_keys.items():
                    if key and key in flat_features:
                        candidates.append(len(feature_values(flat_features[key])) // ACTION_DIMS[field])
                ep_steps = max(candidates) if candidates else 0

            episode_sample = {
                "file": str(path),
                "kind": parsed["kind"],
                "steps": ep_steps,
                "image_key": image_key,
                "action_keys": action_keys,
                "first_step": {},
            }

            for field, key in action_keys.items():
                if not key:
                    continue
                seq = feature_lists.get(key, [])
                if seq:
                    vals0 = feature_values(seq[0])
                    episode_sample["first_step"][field] = vals0
                    for step_feature in seq:
                        vals = feature_values(step_feature)
                        for i, value in enumerate(vals):
                            action_stats[f"{field}[{i}]"].add(value)
                        if field == "world_vector":
                            norm_stats["arm_translation_norm"].add(math.sqrt(sum(v * v for v in vals)))
                        elif field == "rotation_delta":
                            norm_stats["arm_rotation_norm"].add(math.sqrt(sum(v * v for v in vals)))
                        elif field == "base_displacement_vector":
                            norm_stats["base_translation_norm"].add(math.sqrt(sum(v * v for v in vals)))
                        elif field == "base_displacement_vertical_rotation":
                            norm_stats["base_rotation_abs"].add(abs(vals[0]) if vals else 0.0)
                        elif field == "gripper_closedness_action":
                            norm_stats["gripper_abs"].add(abs(vals[0]) if vals else 0.0)
                        elif field == "terminate_episode":
                            terminate_counter[tuple(int(v) for v in vals)] += 1
                else:
                    feature = flat_features.get(key)
                    if feature:
                        vals = feature_values(feature)
                        rows = reshape_flat(vals, ACTION_DIMS[field])
                        if rows:
                            episode_sample["first_step"][field] = rows[0]
                        else:
                            episode_sample["first_step"][field] = first_values(feature)
                        for row in rows:
                            for i, value in enumerate(row):
                                action_stats[f"{field}[{i}]"].add(value)
                            if field == "world_vector":
                                norm_stats["arm_translation_norm"].add(math.sqrt(sum(v * v for v in row)))
                            elif field == "rotation_delta":
                                norm_stats["arm_rotation_norm"].add(math.sqrt(sum(v * v for v in row)))
                            elif field == "base_displacement_vector":
                                norm_stats["base_translation_norm"].add(math.sqrt(sum(v * v for v in row)))
                            elif field == "base_displacement_vertical_rotation":
                                norm_stats["base_rotation_abs"].add(abs(row[0]) if row else 0.0)
                            elif field == "gripper_closedness_action":
                                norm_stats["gripper_abs"].add(abs(row[0]) if row else 0.0)
                            elif field == "terminate_episode":
                                terminate_counter[tuple(int(v) for v in row)] += 1

            samples.append(episode_sample)
            episodes += 1
            steps += ep_steps
            if episodes >= max_episodes:
                return {
                    "data_dir": str(data_dir),
                    "episodes": episodes,
                    "steps": steps,
                    "schema": schema,
                    "samples": samples,
                    "action_stats": {k: v.as_dict() for k, v in sorted(action_stats.items())},
                    "norm_stats": {k: v.as_dict() for k, v in sorted(norm_stats.items())},
                    "terminate_episode_counts": {str(k): v for k, v in terminate_counter.items()},
                }
    return {
        "data_dir": str(data_dir),
        "episodes": episodes,
        "steps": steps,
        "schema": schema,
        "samples": samples,
        "action_stats": {k: v.as_dict() for k, v in sorted(action_stats.items())},
        "norm_stats": {k: v.as_dict() for k, v in sorted(norm_stats.items())},
        "terminate_episode_counts": {str(k): v for k, v in terminate_counter.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-episodes", type=int, default=20)
    args = parser.parse_args()

    result = audit(Path(args.data_dir), args.max_episodes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "episodes": result["episodes"],
        "steps": result["steps"],
        "kind": result["schema"]["record_kind"] if result["schema"] else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
