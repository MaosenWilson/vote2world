"""Runtime compatibility shims for the upstream RLVR-World / iVideoGPT stack.

diffusers==0.27.0 imports `cached_download`, which modern huggingface_hub has
removed. We alias it at runtime instead of editing site-packages. Import this
module before importing diffusers or ivideogpt; the loaders in
`dor.models` do so automatically.
"""
import sys

import huggingface_hub as _hh

from dor.constants import IVG

if not hasattr(_hh, "cached_download"):
    _hh.cached_download = _hh.hf_hub_download

if IVG not in sys.path:
    sys.path.insert(0, IVG)
