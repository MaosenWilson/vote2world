# RT-1 Action Schema Audit

- source: `data/processed/fractal20220817_data/train_eps_00000000.npz`
- status: `confirmed`
- action_dim: `13`
- flatten_key_order: `['gripper_closedness_action', 'terminate_episode', 'base_displacement_vector', 'rotation_delta', 'base_displacement_vertical_rotation', 'world_vector']`

| field | slice | dim | group | motion magnitude |
|---|---:|---:|---|---:|
| `gripper_closedness_action` | `[0:1]` | 1 | `gripper` | False |
| `terminate_episode` | `[1:4]` | 3 | `mode` | False |
| `base_displacement_vector` | `[4:6]` | 2 | `base` | False |
| `rotation_delta` | `[6:9]` | 3 | `rotation` | True |
| `base_displacement_vertical_rotation` | `[9:10]` | 1 | `base` | False |
| `world_vector` | `[10:13]` | 3 | `translation` | True |
