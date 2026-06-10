# Audit Summary

## Confirmed Facts

1. The figure's 7D action is present in RT-1 as `world_vector + rotation_delta + gripper_closedness_action`.
2. Raw action fields sum to 13 dimensions.
3. RLVR-World's converter flattens actions by iterating `step['action'].keys()`.
4. The observed metadata order maps the 13D vector as:

| 索引范围 | 字段 | 维度 | 是否参与静态复制过滤 | 建议处理方式 |
|---|---|---:|---:|---|
| `[0:1]` | `gripper_closedness_action` | 1 | 有条件 | 单独处理；不并入连续位姿范数 |
| `[1:4]` | `terminate_episode` | 3 | 否 | 排除运动幅度；作为模式字段记录 |
| `[4:6]` | `base_displacement_vector` | 2 | 单独 | RT-1 样本中为 0；若目标域有移动底盘需单独阈值 |
| `[6:9]` | `rotation_delta` | 3 | 是 | 参与旋转范数；与平移分开设阈值 |
| `[9:10]` | `base_displacement_vertical_rotation` | 1 | 单独 | RT-1 样本中为 0；单独处理 |
| `[10:13]` | `world_vector` | 3 | 是 | 参与平移范数；推荐作为主分桶依据 |

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
