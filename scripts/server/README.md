# Server Notes

This repository does not assume local execution.

Recommended server order:

1. Clone this repository.
2. Clone `https://github.com/thuml/RLVR-World` into `third_party/RLVR-World/`.
3. Create the upstream RLVR-World environment first.
4. Download tokenizer and checkpoints into `checkpoints/`.
5. Mount RT-1 data under `data/rt1/`.
6. Reproduce official baseline metrics before implementing Vote2World reward code.

