# RT-1 Input Audit

## Purpose

This report audits the RLVR-World / RT-1 single-step input path for Vote2World.
The goal is to identify the real RT-1 action schema, the official RLVR-World
flattening logic, image fields, action quantization, and the action groups that
should be used by future GT-free self-consensus rewards.

## Data And Scope

- Dataset: `fractal20220817_data/0.1.0`
- Data path: `data/raw/fractal20220817_data/0.1.0`
- Downloaded files observed: full TFDS directory, `112G`, `1026` files
- Parsed sample: `20` TFRecord episodes, `978` steps
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
