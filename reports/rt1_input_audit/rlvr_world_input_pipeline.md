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
