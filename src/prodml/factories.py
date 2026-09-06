from collections.abc import Callable
from typing import Any, Protocol

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

from prodml.mlp import MLPWrapper


class Trainable(Protocol):
    """Anything with a .fit(X, y) method — sklearn models satisfy it structurally."""

    def fit(self, X: Any, y: Any) -> Any: ...
    

MODEL_FACTORIES: dict[str, Callable[..., Trainable]] = {
    "linear": LinearRegression,
    "xgboost": XGBRegressor,
    "mlp": MLPWrapper,
}

METRICS: dict[str, Callable[[Any, Any], float]] = {
    "rmse": root_mean_squared_error,
    "mae": mean_absolute_error,
    "r2": r2_score,
}