import hashlib, os, subprocess, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
import mlflow, mlflow.sklearn
import pickle
from typing import Any, Protocol
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer

from prodml.config import TrainingSettings
from prodml.data import clean_data, load_training_data, load_validation_data
from prodml.features import compose_features
from prodml.logging_config import setup_logger
from prodml.registry import METRICS, MODEL_FACTORIES

def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()

def git_user() -> str:
    return subprocess.run(["git", "config", "--local", "user.name"],
                          capture_output=True, text=True).stdout.strip()

def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def log_figures(trainer, model_name: str) -> None:
    """Residual plot (all) + feature importance (linear/xgboost) — 2.3."""
    y_val = getattr(trainer, "_y_val", None)
    preds = getattr(trainer, "_preds", None)
    if y_val is not None and preds is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(preds, y_val - preds, s=8, alpha=0.3)
        ax.axhline(0.0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted duration (min)")
        ax.set_ylabel("Residual (min)")
        ax.set_title(f"{model_name} — residuals")
        mlflow.log_figure(fig, "diagnostics/residuals.png")
        plt.close(fig)

    model = trainer.model
    names = trainer._vec.get_feature_names_out()
    if model_name == "linear":
        importance = np.abs(np.asarray(model.coef_).ravel())
    elif model_name == "xgboost":
        importance = np.asarray(model.feature_importances_)
    else:
        return                                       # MLP: no native importance
    top = np.argsort(importance)[-20:]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([names[i] for i in top], importance[top])
    ax.set_xlabel("importance")
    ax.set_title(f"{model_name} — top feature importance")
    mlflow.log_figure(fig, "diagnostics/feature_importance.png")
    plt.close(fig)

class Fittable(Protocol):
    """Anything with a .fit(X, y) method — sklearn models satisfy it structurally."""

    def fit(self, X: Any, y: Any) -> Any: ...


class Trainer:
    """Train and validate a machine learning model for trip-duration prediction.

    The trainer loads training and validation data, preprocesses features, fits a
    configured model, validates it using one or more metrics, and persists the
    trained model together with the fitted DictVectorizer.
    """

    def __init__(
        self,
        settings: TrainingSettings,
        model_params: dict[str, Any] | None = None,
        model_name: str | None = None,
    ) -> None:
        self.settings = settings
        self._vec: DictVectorizer | None = None
        self.metrics: dict[str, float] = {}

        if model_name is None:
            self.model_name = self.settings.model_name
            self.model_params = self.settings.model_params
        else:
            self.model_name = model_name
            self.model_params = model_params if model_params is not None else {}

        if self.model_name not in MODEL_FACTORIES:
            raise ValueError(f"Model {self.model_name} is not supported.")
        self.model: Fittable = MODEL_FACTORIES[self.model_name](**self.model_params)

    def save_model(self) -> None:
        """Save the trained model and vectorizer to disk. Separated to avoid overriding the model before validation."""
        if self._vec is None or self.model is None:
            raise RuntimeError("call train() before validate()")
        with open(self.settings.pkl_model_path, "wb") as f_out:
            pickle.dump({"model": self.model, "vectorizer": self._vec}, f_out)

    def validate(self, metrics: list[str] | None = None) -> tuple[float, ...]:
        """Validate the model on the validation set."""
        if metrics is None:
            metrics = list(METRICS.keys())
        if not all(metric in METRICS for metric in metrics):
            raise ValueError(
                f"Metrics {metrics} are not supported. Supported metrics: {list(METRICS.keys())}"
            )
        if self._vec is None or self.model is None:
            raise RuntimeError("call train() before validate()")
        df_val_raw = load_validation_data(self.settings.validation_set)
        df_val = clean_data(compose_features(df_val_raw))
        X_dicts = df_val[["Trip_Distance"] + ["PU_DO"]].to_dict(orient="records")
        X_val = self._vec.transform(X_dicts)
        Y_val = df_val["Trip_Duration"].values
        preds = self.model.predict(X_val)
        self.metrics = {metric: METRICS[metric](Y_val, preds) for metric in metrics}
        self._y_val = Y_val
        self._preds = preds

    def train(self) -> None:
        """Run the training workflow."""

        if self._vec is None:
            self._vec = DictVectorizer()

        df_train_raw = load_training_data(self.settings.training_set)
        df_train = clean_data(compose_features(df_train_raw))
        X_dicts = df_train[["Trip_Distance"] + ["PU_DO"]].to_dict(orient="records")
        X_train = self._vec.fit_transform(X_dicts)
        Y_train = df_train["Trip_Duration"].values

        self.model.fit(X_train, Y_train)


def main() -> None:
    matplotlib.use("Agg")
    settings = TrainingSettings()
    setup_logger(log_level=settings.log_level, log_format=settings.log_format)
    _log = logging.getLogger(__name__)
    
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("Ride Duration Model")

    if settings.model_name == "xgboost":
        mlflow.xgboost.autolog()                     # 2.4 experiment: free vs manual

    _log.info("Starting training")
    with mlflow.start_run(run_name=settings.model_name):
        trainer = Trainer(settings)

        # params — identity + lineage
        data_hash = file_md5(settings.training_set)
        mlflow.log_params({
            "model_family": settings.model_name,
            "training_set": settings.training_set,
            "validation_set": settings.validation_set,
            "data_version": data_hash,
        })
        if settings.model_params:
            mlflow.log_params(settings.model_params)

        # train + duration metric
        t0 = time.monotonic()
        trainer.train()
        mlflow.log_metric("train_duration_s", time.monotonic() - t0)

        # validate + metrics
        trainer.validate()
        mlflow.log_metrics(trainer.metrics)

        # artifacts — model bundle, size, plots, requirements
        trainer.save_model()
        mlflow.log_artifact(settings.pkl_model_path, artifact_path="model")
        mlflow.log_metric("model_size_mb", os.path.getsize(settings.pkl_model_path) / 1e6)
        log_figures(trainer, settings.model_name)
        if Path("requirements.txt").exists():
            mlflow.log_artifact("requirements.txt")
        else:
            _log.warning("requirements.txt missing — run: uv export --frozen --no-hashes --no-dev --group train -o requirements.txt")

        # tags
        mlflow.set_tags({
            "git_commit": os.getenv("GIT_COMMIT", git_commit()),
            "data_version": data_hash,
            "author": os.getenv("GIT_USER", git_user()),
            "framework": {"linear": "scikit-learn", "xgboost": "xgboost", "mlp": "pytorch"}[settings.model_name],
        })

    _log.info("Run complete")