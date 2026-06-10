# Workspace Conventions

This repository separates lightweight project code from large research artifacts.

## Tracked

- Research and engineering documentation.
- Config templates.
- Source code under `src/vote2world/`.
- Lightweight scripts and tests.
- Small reports or tables that are needed for paper writing.

## Not Tracked

- RT-1 data.
- Downloaded upstream checkpoints.
- Candidate frame caches.
- Decoded videos and images.
- Training logs and temporary experiment outputs.
- A full clone of upstream RLVR-World.

## Server Assumption

Local development prepares structure and documentation only. Real execution is expected on a GPU server after the upstream RLVR-World environment and assets are installed.

