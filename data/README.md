# Data Directory

This directory is a mount point for server-side datasets.

Expected layout:

```text
data/
  rt1/             RT-1 data or symlinks.
  raw/             Raw downloaded data.
  processed/       Preprocessed samples.
  cache/
    candidates/    Candidate token/frame caches.
  manifests/       Lightweight dataset split manifests.
```

Large files are ignored by git.

