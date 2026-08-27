import logging
import pickle
from typing import Any, Protocol

from sklearn.feature_extraction import DictVectorizer

from prodml.config import TrainingSettings
from prodml.data import clean_data, load_training_data, load_validation_data
from prodml.features import compose_features
from prodml.logging_config import setup_logger
from prodml.registry import METRICS, MODEL_FACTORIES


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

        if model_name is None:
            self.model_name = self.settings.model_name
            self.model_params = self.settings.model_params
        else:
            self.model_name = model_name
            self.model_params = model_params if model_params is not None else {}

        if self.model_name not in MODEL_FACTORIES:
            raise ValueError(f"Model {self.model_name} is not supported.")
        self._model: Fittable = MODEL_FACTORIES[self.model_name](**self.model_params)

    def save_model(self) -> None:
        """Save the trained model and vectorizer to disk. Separated to avoid overriding the model before validation."""
        if self._vec is None or self._model is None:
            raise RuntimeError("call train() before validate()")
        with open(self.settings.model_path, "wb") as f_out:
            pickle.dump({"model": self._model, "vectorizer": self._vec}, f_out)

    def validate(self, metrics: list[str] | None = None) -> tuple[float, ...]:
        """Validate the model on the validation set."""
        if metrics is None:
            metrics = list(METRICS.keys())
        if not all(metric in METRICS for metric in metrics):
            raise ValueError(
                f"Metrics {metrics} are not supported. Supported metrics: {list(METRICS.keys())}"
            )
        if self._vec is None or self._model is None:
            raise RuntimeError("call train() before validate()")
        df_val_raw = load_validation_data(self.settings.validation_set)
        df_val = clean_data(compose_features(df_val_raw))
        X_dicts = df_val[["Trip_Distance"] + ["PU_DO"]].to_dict(orient="records")
        X_val = self._vec.transform(X_dicts)
        Y_val = df_val["Trip_Duration"].values
        preds = self._model.predict(X_val)
        return {metric: METRICS[metric](Y_val, preds) for metric in metrics}

    def train(self) -> None:
        """Run the training workflow."""

        if self._vec is None:
            self._vec = DictVectorizer()

        df_train_raw = load_training_data(self.settings.training_set)
        df_train = clean_data(compose_features(df_train_raw))
        X_dicts = df_train[["Trip_Distance"] + ["PU_DO"]].to_dict(orient="records")
        X_train = self._vec.fit_transform(X_dicts)
        Y_train = df_train["Trip_Duration"].values

        self._model.fit(X_train, Y_train)


def main() -> None:
    settings = TrainingSettings()
    trainer = Trainer(settings)

    setup_logger(log_level=settings.log_level, log_format=settings.log_format)
    _log = logging.getLogger(__name__)

    _log.info("Starting training")
    trainer.train()
    _log.info("Training completed")

    _log.info("Starting validation")
    metrics = trainer.validate()
    _log.info(f"Validation completed. Metrics: {metrics}")

    _log.info("Saving model")
    trainer.save_model()
