from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from typing import Protocol, Any, Callable


class Fittable(Protocol):
    """Anything with a .fit(X, y) method — sklearn models satisfy it structurally."""
    def fit(self, X: Any, y: Any) -> Any: ...


MODEL_FACTORIES: dict[str, Callable[..., Fittable]] = {
    "linear": LinearRegression,
}

METRICS: dict[str, Callable[[Any, Any], float]] = {
    "rmse": root_mean_squared_error,
    "mae":  mean_absolute_error,
}