# RT-1 Action Schema

## Evidence

- Actual TFDS metadata: `data/raw/fractal20220817_data/0.1.0/features.json`
- Actual parsed TFRecord sample: `reports/rt1_input_audit/rt1_tfrecord_audit.json`
- Official flattening code: `third_party/RLVR-World/vid_wm/oxe_data_converter.py:58-67`

## Raw Action Fields

| 字段 | shape | dtype | 样例 first step | min | max | mean | std | 零值比例 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| `gripper_closedness_action` | `(1,)` | `float32` | `[0.0]` | -1 | 1 | 0.0236907 | 0.355828 | 0.725971 |
| `terminate_episode` | `(3,)` | `int32` | `[0.0, 1.0, 0.0]` | 0 | 1 | 0.326517 | 0.113197 | 0.673483 |
| `base_displacement_vector` | `(2,)` | `float32` | `[0.0, 0.0]` | 0 | 0 | 0 | 0 | 1 |
| `rotation_delta` | `(3,)` | `float32` | `[-0.0692676231265068, -0.044923778623342514, 0.013373391702771187]` | -0.883984 | 0.779248 | 0.0125617 | 0.13712 | 0.0756646 |
| `base_displacement_vertical_rotation` | `(1,)` | `float32` | `[0.0]` | 0 | 0 | 0 | 0 | 1 |
| `world_vector` | `(3,)` | `float32` | `[-0.014691736549139023, -0.00486538652330637, -0.01381668820977211]` | -0.397654 | 0.266149 | -0.00016705 | 0.0560064 | 0.0756646 |

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

| 索引范围 | 字段 | 维度 | 是否参与静态复制过滤 | 建议处理方式 |
|---|---|---:|---:|---|
| `[0:1]` | `gripper_closedness_action` | 1 | 有条件 | 单独处理；不并入连续位姿范数 |
| `[1:4]` | `terminate_episode` | 3 | 否 | 排除运动幅度；作为模式字段记录 |
| `[4:6]` | `base_displacement_vector` | 2 | 单独 | RT-1 样本中为 0；若目标域有移动底盘需单独阈值 |
| `[6:9]` | `rotation_delta` | 3 | 是 | 参与旋转范数；与平移分开设阈值 |
| `[9:10]` | `base_displacement_vertical_rotation` | 1 | 单独 | RT-1 样本中为 0；单独处理 |
| `[10:13]` | `world_vector` | 3 | 是 | 参与平移范数；推荐作为主分桶依据 |

## Relation To The Figure's 7D Action

The figure's `(x, y, z, roll, pitch, yaw, gripper openness)` corresponds to:

```text
world_vector[0:3], rotation_delta[0:3], gripper_closedness_action[0]
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
