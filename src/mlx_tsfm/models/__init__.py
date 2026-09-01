"""TSFM model backends. Importing this package registers the built-in backends."""
from ..tsfm import register_model
from .timesfm import TimesFM3


def _build_timesfm3(**kw) -> TimesFM3:
    """Load real TimesFM 3.0 weights (downloads from HF on first use). Non-commercial license."""
    return TimesFM3.from_pretrained(dtype=kw.get("dtype", "fp32"), compile=kw.get("compile", True))


register_model("timesfm-3.0", _build_timesfm3)
