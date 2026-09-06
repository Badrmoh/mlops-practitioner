import functools
import hashlib
import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd

from prodml.config import PredictSettings
from prodml.logging_config import setup_logger

settings = PredictSettings()
_log = logging.getLogger(__name__)


def timed(fn):
    """Log the execution time of the decorated function."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.monotonic()  # NOT time.time — immune to clock jumps
        result = fn(*args, **kwargs)
        duration = time.monotonic() - start
        _log.info(f"Prediction served with latency {duration*1000:.1f} ms")
        return result

    return wrapper


class DurationPredictor:
    """Predict trip duration; loads from a local bundle or the MLflow registry."""

    def __init__(self, settings: PredictSettings):
        self.settings = settings

    def load(self) -> None:
        if self.settings.model_source == "mlflow":
            self._load_from_registry()
        else:
            self._load_from_file()

    def _load_from_file(self) -> None:
        """Local pickle bundle — the hermetic path used by tests/dev."""
        try:
            with open(self.settings.pkl_model_path, "rb") as f_in:
                model_data = pickle.load(f_in)
                self._model = model_data["model"]
                self._vec = model_data["vectorizer"]
                self.features = self._vec.get_feature_names_out()
        except FileNotFoundError:
            _log.error(f"Model file not found at {self.settings.pkl_model_path}")
            raise
        except Exception as e:
            _log.error(f"Error loading model: {e}")
            raise

    def _load_from_registry(self) -> None:
        """Load by stage URI — the served model follows the registry, not a path."""
        uri = f"models:/{self.settings.registry_model_name}/{self.settings.registry_stage}"
        _log.info(f"Loading model from {uri}")
        self._pyfunc = mlflow.pyfunc.load_model(uri)

        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(
            self.settings.registry_model_name, stages=[self.settings.registry_stage]
        )
        if not versions:
            raise RuntimeError(
                f"No version in stage {self.settings.registry_stage} "
                f"for {self.settings.registry_model_name}"
            )
        self._version = max(versions, key=lambda v: v.version)
        self._run = client.get_run(self._version.run_id)

        # vectorizer from the same run's bundle artifact — powers feature_names
        local = mlflow.artifacts.download_artifacts(run_id=self._run.info.run_id, artifact_path="model")
        pkl = next(Path(local).glob("*.pkl"))
        with open(pkl, "rb") as f:
            self._vec = pickle.load(f)["vectorizer"]
        self.features = self._vec.get_feature_names_out()

    @timed
    def predict(self, features: dict) -> dict:
        """Predict one trip; returns {**features, "Prediction": minutes}."""
        _log.debug("predict.features", features=features)
        if self.settings.model_source == "mlflow":
            prediction = self._pyfunc.predict(pd.DataFrame([features]))
            return {**features, "Prediction": float(prediction.ravel()[0])}
        X = self._vec.transform([features])
        prediction = self._model.predict(X)
        return {**features, "Prediction": prediction[0]}

    @timed
    def predict_batch(self, features_list: list[dict]) -> list[dict]:
        if self.settings.model_source == "mlflow":
            predictions = self._pyfunc.predict(pd.DataFrame(features_list))
            return [
                {**features, "Prediction": pred}
                for features, pred in zip(features_list, predictions.ravel())
            ]
        X = self._vec.transform(features_list)
        predictions = self._model.predict(X)
        return [{**features, "Prediction": pred} for features, pred in zip(features_list, predictions)]

    @property
    def metadata(self) -> dict:
        if self.settings.model_source == "mlflow":
            tags = self._run.data.tags
            ts_ms = self._run.info.end_time or self._run.info.start_time
            return {
                "model_name": self.settings.registry_model_name,
                "model_version": str(self._version.version),
                "training_framework": tags.get("framework", "unknown"),
                "training_date": datetime.fromtimestamp(ts_ms / 1000),
                "artifact_hash": tags.get("data_version", ""),
                "feature_names": self.features.tolist(),
            }

        with open(self.settings.pkl_model_path, "rb") as f:
            model_hash = hashlib.file_digest(f, "md5").hexdigest()

        return {
            "model_name": Path(self.settings.pkl_model_path).stem,
            "model_version": "0.1.0",
            "training_framework": "scikit-learn",
            "training_date": datetime.fromtimestamp(os.path.getmtime(self.settings.pkl_model_path)),
            "artifact_hash": model_hash,
            "feature_names": self._vec.get_feature_names_out().tolist(),
        }


def main() -> None:
    """Main function to load the model and make a prediction."""
    setup_logger(log_level=settings.log_level, log_format=settings.log_format)
    predictor = DurationPredictor(settings)
    predictor.load()

    example_features = [
        {"PU_DO": "260_193", "Trip_Distance": 2.740000009536743},
        {"PU_DO": "260_226", "Trip_Distance": 1.4299999475479126},
        {"PU_DO": "181_249", "Trip_Distance": 3.700000047683716},
        {"PU_DO": "260_260", "Trip_Distance": 0.4000000059604645},
        {"PU_DO": "74_244", "Trip_Distance": 2.740000009536743},
    ]

    predicted_duration = predictor.predict(example_features[0])
    _log.info(f"Predicted trip duration: {predicted_duration['Prediction']:.2f} minutes")

    predicted_durations = predictor.predict_batch(example_features)
    _log.info(f"Predicted trip durations: {predicted_durations}")


if __name__ == "__main__":
    main()