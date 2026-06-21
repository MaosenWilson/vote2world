#!/usr/bin/env python3
"""Render RT-1 input audit markdown reports from stdlib audit JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FIELD_META = [
    ("gripper_closedness_action", 1, "float32", "[0:1]", "夹爪开合", "单独处理；不并入连续位姿范数"),
    ("terminate_episode", 3, "int32", "[1:4]", "终止/模式字段", "排除运动幅度；作为模式字段记录"),
    ("base_displacement_vector", 2, "float32", "[4:6]", "底盘平移", "RT-1 样本中为 0；若目标域有移动底盘需单独阈值"),
    ("rotation_delta", 3, "float32", "[6:9]", "机械臂旋转 rpy", "参与旋转范数；与平移分开设阈值"),
    ("base_displacement_vertical_rotation", 1, "float32", "[9:10]", "底盘 yaw", "RT-1 样本中为 0；单独处理"),
    ("world_vector", 3, "float32", "[10:13]", "机械臂平移 xyz", "参与平移范数；推荐作为主分桶依据"),
]


EXPECTED_7D = [
    "world_vector[0:3]",
    "rotation_delta[0:3]",
    "gripper_closedness_action[0]",
]


def fmt(x):
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def stat_table(data):
    rows = ["| 字段 | n | min | max | mean | std | 零值比例 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key, stat in data["action_stats"].items():
        rows.append(
            f"| `{key}` | {stat['n']} | {fmt(stat['min'])} | {fmt(stat['max'])} | "
            f"{fmt(stat['mean'])} | {fmt(stat['std'])} | {fmt(stat['zero_ratio'])} |"
        )
    return "\n".join(rows)


def norm_table(data):
    rows = ["| 统计项 | n | min | max | mean | std | 零值比例 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key, stat in data["norm_stats"].items():
        rows.append(
            f"| `{key}` | {stat['n']} | {fmt(stat['min'])} | {fmt(stat['max'])} | "
            f"{fmt(stat['mean'])} | {fmt(stat['std'])} | {fmt(stat['zero_ratio'])} |"
        )
    return "\n".join(rows)


def mapping_table():
    rows = [
        "| 索引范围 | 字段 | 维度 | 是否参与静态复制过滤 | 建议处理方式 |",
        "|---|---|---:|---:|---|",
    ]
    include = {
        "world_vector": "是",
        "rotation_delta": "是",
        "gripper_closedness_action": "有条件",
        "base_displacement_vector": "单独",
        "base_displacement_vertical_rotation": "单独",
        "terminate_episode": "否",
    }
    for field, dim, _, idx, _, handling in FIELD_META:
        rows.append(f"| `{idx}` | `{field}` | {dim} | {include[field]} | {handling} |")
    return "\n".join(rows)


def raw_field_table(data):
    rows = [
        "| 字段 | shape | dtype | 样例 first step | min | max | mean | std | 零值比例 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    sample = data["samples"][0]["first_step"]
    for field, dim, dtype, _, _, _ in FIELD_META:
        component_stats = [data["action_stats"].get(f"{field}[{i}]") for i in range(dim)]
        mins = [s["min"] for s in component_stats if s]
        maxs = [s["max"] for s in component_stats if s]
        means = [s["mean"] for s in component_stats if s]
        stds = [s["std"] for s in component_stats if s]
        zeros = [s["zero_ratio"] for s in component_stats if s]
        rows.append(
            f"| `{field}` | `({dim},)` | `{dtype}` | `{sample.get(field)}` | "
            f"{fmt(min(mins) if mins else None)} | {fmt(max(maxs) if maxs else None)} | "
            f"{fmt(sum(means)/len(means) if means else None)} | "
            f"{fmt(sum(stds)/len(stds) if stds else None)} | "
            f"{fmt(sum(zeros)/len(zeros) if zeros else None)} |"
        )
    return "\n".join(rows)


def write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.audit_json).read_text(encoding="utf-8"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sample = data["samples"][0]
    n_eps = data["episodes"]
    n_steps = data["steps"]
    data_dir = data["data_dir"]

    write(out / "README.md", f"""
# RT-1 Input Audit

## Purpose

This report audits the RLVR-World / RT-1 single-step input path for Vote2World.
The goal is to identify the real RT-1 action schema, the official RLVR-World
flattening logic, image fields, action quantization, and the action groups that
should be used by future GT-free self-consensus rewards.

## Data And Scope

- Dataset: `fractal20220817_data/0.1.0`
- Data path: `{data_dir}`
- Downloaded files observed: full TFDS directory, `112G`, `1026` files
- Parsed sample: `{n_eps}` TFRecord episodes, `{n_steps}` steps
- Project path: `/root/autodl-tmp/vote2world`
- Upstream code path: `third_party/RLVR-World`
- Upstream commit observed locally: `e1b8b6f40ca0696919ce9b8dbe965c4153bcf5f1`

## Result Summary

- Raw RT-1 action fields total 13 dimensions.
- The figure's 7D robot action corresponds to `world_vector + rotation_delta + gripper_closedness_action`.
- RLVR-World's converter concatenates `step['action'][k] for k in step['action'].keys()`.
- The metadata field order observed in `features.json` is:
  `gripper_closedness_action`, `terminate_episode`, `base_displacement_vector`,
  `rotation_delta`, `base_displacement_vertical_rotation`, `world_vector`.
- Therefore, the likely flattened index order is not the human-readable 7D figure order.
- Single-step RLVR-World uses `context_length=4`, `segment_length=5`, `action_dim=13`, `action_bins=256`.
- The simple processor builds history tokens from 4 frames and 4 action vectors, then trains/generates one future frame token sequence.

## Issues Encountered

- Server apt initially failed because a Google Cloud SDK apt source returned 503; it was disabled for package installation.
- Installing TensorFlow from the available PyPI mirror was too slow for this turn.
- Official `oxe_data_converter.py` was inspected but not executed with TFDS. The `.npz` audit file records this limitation.
""")

    write(out / "rt1_action_schema.md", f"""
# RT-1 Action Schema

## Evidence

- Actual TFDS metadata: `{data_dir}/features.json`
- Actual parsed TFRecord sample: `reports/rt1_input_audit/rt1_tfrecord_audit.json`
- Official flattening code: `third_party/RLVR-World/vid_wm/oxe_data_converter.py:58-67`

## Raw Action Fields

{raw_field_table(data)}

## Action Key Order And Flattening

RLVR-World official converter uses:

```python
np.concatenate([step['action'][k] for k in step['action'].keys()])
```

Evidence: `third_party/RLVR-World/vid_wm/oxe_data_converter.py:60-62`.

The `features.json` action order is:

```text
gripper_closedness_action
terminate_episode
base_displacement_vector
rotation_delta
base_displacement_vertical_rotation
world_vector
```

Based on that metadata order, the flattened 13D action mapping is:

{mapping_table()}

## Relation To The Figure's 7D Action

The figure's `(x, y, z, roll, pitch, yaw, gripper openness)` corresponds to:

```text
{", ".join(EXPECTED_7D)}
```

However, under the metadata/converter order above these fields are located at:

- gripper: `[0:1]`
- roll/pitch/yaw: `[6:9]`
- x/y/z: `[10:13]`

So the 7D semantic action exists in RT-1, but it is not necessarily stored as
the first seven flattened dimensions in RLVR-World conversion.

## Reward-Design Grouping

- Arm translation: `world_vector`, indices `[10:13]`.
- Arm rotation: `rotation_delta`, indices `[6:9]`.
- Gripper: `gripper_closedness_action`, index `[0:1]`.
- Base translation: `base_displacement_vector`, indices `[4:6]`.
- Base rotation: `base_displacement_vertical_rotation`, index `[9:10]`.
- Mode/termination: `terminate_episode`, indices `[1:4]`.

Do not compute static-copy filtering from the full 13D norm.
""")

    write(out / "rt1_npz_audit.md", f"""
# Converted NPZ Audit

## Status

Official `.npz` conversion was not executed in this turn because the server did
not have TensorFlow / TensorFlow Datasets installed and installing
`tensorflow-cpu` from the available mirror was too slow.

## Official Converter Evidence

- Script: `third_party/RLVR-World/vid_wm/oxe_data_converter.py`
- Dataset default: `--dataset_name fractal20220817_data` at line 39
- Input default: `/data2/tensorflow_datasets` at line 40
- Output default: `/data2/frame_action_datasets` at line 41
- Episode cap: `--max_num_episodes` at line 42
- Display key fallback: `DISPLAY_KEY.get(dataset_name, 'image')` at line 47
- Frame extraction: `step['observation'][display_key]` at line 59
- Action flattening: `np.concatenate([step['action'][k] for k in step['action'].keys()])` at line 61
- Saved keys: `{{display_key: frames, 'action': actions}}` at line 67

## Expected NPZ Structure From Code

For `fractal20220817_data`, `display_key` falls back to `image`, so expected keys are:

```text
image
action
```

Expected shapes after official conversion:

```text
image.shape  = (T, 256, 320, 3)
action.shape = (T, 13)
```

## Parsed Raw Evidence

- Parsed episodes: `{n_eps}`
- Parsed steps: `{n_steps}`
- First episode image key: `{sample['image_key']}`
- First episode action keys: `{sample['action_keys']}`
- First step sample: `{sample['first_step']}`

## Required Follow-Up

After TensorFlow/TFDS is available, run:

```bash
cd /root/autodl-tmp/vote2world/third_party/RLVR-World/vid_wm
../../.venv/rt1-audit/bin/python oxe_data_converter.py \\
  --dataset_name fractal20220817_data \\
  --input_path /root/autodl-tmp/vote2world/data/raw \\
  --output_path /root/autodl-tmp/vote2world/data/processed \\
  --max_num_episodes 20
```
""")

    write(out / "rlvr_world_input_pipeline.md", """
# RLVR-World Single-Step Input Pipeline

## Evidence Files

- `third_party/RLVR-World/vid_wm/ivideogpt/train_vgpt.py`
- `third_party/RLVR-World/vid_wm/ivideogpt/eval_vgpt.py`
- `third_party/RLVR-World/vid_wm/ivideogpt/ivideogpt/processor.py`
- `third_party/RLVR-World/vid_wm/ivideogpt/ivideogpt/data/simple_dataloader.py`

## Single-Step Configuration

| 项目 | 实际值 | 证据位置 |
|---|---:|---|
| context length | 4 | `train_vgpt.py:253`, `eval_vgpt.py:229` |
| segment length | 5 | `train_vgpt.py:210`, `eval_vgpt.py:183` |
| action dim | 13 | `train_vgpt.py:231`, `eval_vgpt.py:204` |
| action bins | 256 | `train_vgpt.py:232`, `eval_vgpt.py:205` |
| visual token num | 4375 | `train_vgpt.py:250`, `eval_vgpt.py:225` |
| action ranges path | `configs/vgpt/frac_action_ranges.pth` | `train_vgpt.py:233`, `eval_vgpt.py:206` |
| processor type | `simple` | `train_vgpt.py:255`, `eval_single_step_prediction.sh:7` |
| tokens per frame | 320 | `eval_vgpt.py:230` |
| generation input length | `(320 + 13) * 4 + 1 = 1333` | `eval_vgpt.py:244` |
| generation output length | `320 + 1 = 321` | `eval_vgpt.py:245` |

## Processor Flow

`SimpleVideoProcessor.__call__` expects:

```text
pixels:  (B, T, C, H, W)
actions: (B, T, 13)
```

Evidence: `processor.py:79-84`.

Flow:

1. Encode all frames with visual tokenizer: `processor.py:89-92`.
2. Take first `context_length=4` frame-token sequences as history: `processor.py:94`.
3. Discretize first 4 action vectors: `processor.py:95-96`.
4. Add visual-token offset to action tokens: `processor.py:97`.
5. Concatenate per-step visual tokens and action tokens: `processor.py:98-99`.
6. Create response from future frame tokens after context: `processor.py:101-103`.
7. Mask history labels with `-100`: `processor.py:104`.
8. Final `input_ids` are `hist_tokens + response`: `processor.py:105`.

## Action Quantization

Evidence: `processor.py:37-49`.

```text
action_ranges = torch.load(config.action_ranges_path)
max_values = action_ranges[:, 1]
min_values = action_ranges[:, 0]
normalized = clip((a - min) / (max - min + 1e-8), 0, 1)
token = floor(normalized * action_bins)
token = clip(token, 0, action_bins - 1)
token += visual_token_num
```

Each action dimension is quantized independently using its own min/max row in
`frac_action_ranges.pth`.

## Single-Step Input/Output

The official single-step model implements:

```text
(o_{t-3:t}, a_{t-3:t}) -> z_{t+1} -> decoder -> o_hat_{t+1}
```

Generation evidence:

- `gen_input = model_input["input_ids"][:, :args.gen_input_length]`: `eval_vgpt.py:281`
- `max_tokens=args.gen_output_length`: `eval_vgpt.py:287`
- generated output reshaped to `(tokens_per_frame + 1)`: `eval_vgpt.py:298`
- EOS removed: `eval_vgpt.py:299`
- visual tokens clamped to `[0, visual_token_num - 1]`: `eval_vgpt.py:311`
- tokens reshaped to `16 x 20`: `eval_vgpt.py:312`
- tokenizer decodes visual tokens to image: `eval_vgpt.py:313-315`

## Reward Module Inputs For Vote2World

| 输入 | 是否需要 | 用途 |
|---|---:|---|
| 当前帧 `o_t` | 是 | 计算视觉变化量 |
| 历史帧 `o_{t-3:t-1}` | 由世界模型使用 | 生成候选下一帧 |
| 当前动作 `a_t` | 是 | 静态复制过滤、动作分桶 |
| 历史动作 `a_{t-3:t-1}` | 由世界模型使用 | 生成候选下一帧 |
| 候选预测帧 `o_hat_{t+1}^{1:K}` | 是 | 共识投票 |
| 未来 GT `o_{t+1}` | 训练阶段禁止使用 | 仅独立评估 |
""")

    write(out / "rt1_action_statistics.md", f"""
# RT-1 Action Statistics

Sample: `{n_eps}` episodes, `{n_steps}` steps parsed directly from TFRecord.

## Per-Dimension Statistics

{stat_table(data)}

## Group Norm Statistics

{norm_table(data)}

## Terminate Episode Distribution

```json
{json.dumps(data["terminate_episode_counts"], indent=2)}
```

## Action-Bucket Recommendation

- Use `world_vector` translation norm as the primary action-magnitude bucket.
- Track `rotation_delta` norm separately; do not merge it with translation.
- Treat gripper as a separate discrete/impulse-like signal; zero ratio is high in this sample.
- Exclude `terminate_episode` from continuous motion magnitude.
- Exclude base motion from arm-motion static filtering for RT-1; sampled base fields are all zero.
- Because arm translation and rotation both have about `7.57%` exact zero ratio in this sample, static-copy filtering should allow a near-zero action bucket rather than forcing motion on every step.

## Static-Copy Filtering Recommendation

Use separate gates:

```text
arm_motion = ||world_vector||_2 + lambda_rot * ||rotation_delta||_2
gripper_event = abs(gripper_closedness_action) > tau_gripper
base_motion = ||base_displacement_vector||_2 + abs(base_displacement_vertical_rotation)
```

Do not use `||a_13d||_2` directly.
""")

    write(out / "audit_summary.md", f"""
# Audit Summary

## Confirmed Facts

1. The figure's 7D action is present in RT-1 as `world_vector + rotation_delta + gripper_closedness_action`.
2. Raw action fields sum to 13 dimensions.
3. RLVR-World's converter flattens actions by iterating `step['action'].keys()`.
4. The observed metadata order maps the 13D vector as:

{mapping_table()}

5. Single-step RLVR-World uses 4 context frames and 4 context action vectors.
6. Each action vector is 13D.
7. Images are tokenized before transformer input.
8. Actions are discretized to 256 bins per dimension, then offset by `visual_token_num`.
9. The model generates one next-frame visual-token sequence of 320 tokens plus EOS.
10. The visual tokenizer decodes generated tokens to the predicted image.

## Unconfirmed Or Limited

- Official `oxe_data_converter.py` was not executed because TensorFlow/TFDS installation was too slow on this server during this turn.
- Therefore, the exact live `step['action'].keys()` order should still be verified once TFDS is installed.
- The likely order reported here is based on actual `features.json` metadata and raw TFRecord keys.

## Impact On Vote2World

- Do not assume dimensions `[0:7]` are `(x,y,z,roll,pitch,yaw,gripper)`.
- For this dataset/order, arm translation is likely `[10:13]`, arm rotation `[6:9]`, and gripper `[0:1]`.
- Static-copy filtering must use field-aware groups, not full-vector norm.
- Action-magnitude bucket design should primarily use `world_vector` norm, with a separate rotation bucket or auxiliary statistic.
- `terminate_episode` must be excluded from motion magnitude.

## Next Steps

1. Install TensorFlow/TFDS in the server audit venv when network is stable.
2. Run official `oxe_data_converter.py --max_num_episodes 20`.
3. Confirm live `step['action'].keys()` order and inspect resulting `.npz`.
4. Copy confirmed `.npz` audit into this report.
5. Then implement Vote2World reward code against explicit field-index constants.
""")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
