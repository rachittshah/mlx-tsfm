"""mlx-tsfm: Time-Series Foundation Model inference on Apple Silicon (built on mlx / mlx-lm)."""
from .tsfm import Forecast, TSFMConfig, TSFMModel, load, register_model

__version__ = "0.0.1"


def forecast(model_id: str, context, horizon: int, quantiles=None, **load_kwargs) -> Forecast:
    """One-shot convenience: ``load`` a backend and ``forecast`` a single series."""
    return load(model_id, **load_kwargs).forecast(context, horizon, quantiles=quantiles)


__all__ = [
    "load",
    "forecast",
    "TSFMModel",
    "TSFMConfig",
    "Forecast",
    "register_model",
    "__version__",
]
