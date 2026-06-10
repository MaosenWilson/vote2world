# RT-1 Action Statistics

Sample: `20` episodes, `978` steps parsed directly from TFRecord.

## Per-Dimension Statistics

| 字段 | n | min | max | mean | std | 零值比例 |
|---|---:|---:|---:|---:|---:|---:|
| `base_displacement_vector[0]` | 978 | 0 | 0 | 0 | 0 | 1 |
| `base_displacement_vector[1]` | 978 | 0 | 0 | 0 | 0 | 1 |
| `base_displacement_vertical_rotation[0]` | 978 | 0 | 0 | 0 | 0 | 1 |
| `gripper_closedness_action[0]` | 978 | -1 | 1 | 0.0236907 | 0.355828 | 0.725971 |
| `rotation_delta[0]` | 978 | -0.631637 | 0.779248 | 0.045123 | 0.152496 | 0.0756646 |
| `rotation_delta[1]` | 978 | -0.631898 | 0.438662 | -0.0104302 | 0.12806 | 0.0756646 |
| `rotation_delta[2]` | 978 | -0.883984 | 0.641255 | 0.00299235 | 0.130804 | 0.0756646 |
| `terminate_episode[0]` | 978 | 0 | 1 | 0.0204499 | 0.141533 | 0.97955 |
| `terminate_episode[1]` | 978 | 0 | 1 | 0.9591 | 0.198058 | 0.0408998 |
| `terminate_episode[2]` | 978 | 0 | 0 | 0 | 0 | 1 |
| `world_vector[0]` | 978 | -0.229348 | 0.22064 | 0.00901573 | 0.0524137 | 0.0756646 |
| `world_vector[1]` | 978 | -0.171165 | 0.194903 | 0.00419171 | 0.0405698 | 0.0756646 |
| `world_vector[2]` | 978 | -0.397654 | 0.266149 | -0.0137086 | 0.0750358 | 0.0756646 |

## Group Norm Statistics

| 统计项 | n | min | max | mean | std | 零值比例 |
|---|---:|---:|---:|---:|---:|---:|
| `arm_rotation_norm` | 978 | 0 | 0.976472 | 0.175151 | 0.168048 | 0.0756646 |
| `arm_translation_norm` | 978 | 0 | 0.403633 | 0.0777991 | 0.06525 | 0.0756646 |
| `base_rotation_abs` | 978 | 0 | 0 | 0 | 0 | 1 |
| `base_translation_norm` | 978 | 0 | 0 | 0 | 0 | 1 |
| `gripper_abs` | 978 | 0 | 1 | 0.16075 | 0.318331 | 0.725971 |

## Terminate Episode Distribution

```json
{
  "(0, 1, 0)": 938,
  "(1, 0, 0)": 20,
  "(0, 0, 0)": 20
}
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
