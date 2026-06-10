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
- Saved keys: `{display_key: frames, 'action': actions}` at line 67

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

- Parsed episodes: `20`
- Parsed steps: `978`
- First episode image key: `steps/observation/image`
- First episode action keys: `{'world_vector': 'steps/action/world_vector', 'rotation_delta': 'steps/action/rotation_delta', 'gripper_closedness_action': 'steps/action/gripper_closedness_action', 'base_displacement_vector': 'steps/action/base_displacement_vector', 'base_displacement_vertical_rotation': 'steps/action/base_displacement_vertical_rotation', 'terminate_episode': 'steps/action/terminate_episode'}`
- First step sample: `{'world_vector': [-0.014691736549139023, -0.00486538652330637, -0.01381668820977211], 'rotation_delta': [-0.0692676231265068, -0.044923778623342514, 0.013373391702771187], 'gripper_closedness_action': [0.0], 'base_displacement_vector': [0.0, 0.0], 'base_displacement_vertical_rotation': [0.0], 'terminate_episode': [0.0, 1.0, 0.0]}`

## Required Follow-Up

After TensorFlow/TFDS is available, run:

```bash
cd /root/autodl-tmp/vote2world/third_party/RLVR-World/vid_wm
../../.venv/rt1-audit/bin/python oxe_data_converter.py \
  --dataset_name fractal20220817_data \
  --input_path /root/autodl-tmp/vote2world/data/raw \
  --output_path /root/autodl-tmp/vote2world/data/processed \
  --max_num_episodes 20
```
