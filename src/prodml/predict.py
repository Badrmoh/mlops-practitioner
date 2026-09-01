import functools
import logging
import time
import os
import hashlib
from datetime import datetime

from prodml.config import PredictSettings
from prodml.logging_config import setup_logger

settings = PredictSettings()
#setup_logger(log_level=settings.log_level, log_format=settings.log_format)
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
    """A class for predicting the duration of a trip based on input features."""

    def __init__(self, settings: PredictSettings):
        self.settings = settings

    def load(self) -> None:
        """Load a trained model and vectorizer from disk."""
        import pickle

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

    @timed
    def predict(self, features: dict) -> float:
        """
        Predict the duration of a trip given its features.

        Args:
            features (dict): A dictionary containing the features of the trip.

        Returns:
            dict: A dictionary containing the input features and the predicted duration of the trip in minutes.
        """
        _log.debug("predict.features", features=features)
        X = self._vec.transform([features])
        prediction = self._model.predict(X)
        return {**features, "Prediction": prediction[0]}

    @timed
    def predict_batch(self, features_list: list[dict]) -> list[dict]:
        """
        Predict the duration of multiple trips given their features.

        Args:
            features_list (list[dict]): A list of dictionaries, each containing the features of a trip.

        Returns:
            list[dict]: A list of dictionaries containing the features and predicted durations for each trip in minutes.
        """
        X = self._vec.transform(features_list)
        predictions = self._model.predict(X)
        return [{**features, "Prediction": pred} for features, pred in zip(features_list, predictions)]

    @property
    def metadata(self) -> "dict":
        """
        Get the metadata of the trained model.
        """

        with open(self.settings.pkl_model_path, "rb") as f:
            model_hash = hashlib.file_digest(f, "md5").hexdigest()

        return {
            "model_name": self.settings.pkl_model_path.split('/')[-1].split('.')[0],
            "model_version": "0.1.0",
            "training_framework": "scikit-learn",
            "training_date": datetime.fromtimestamp(os.path.getmtime(self.settings.pkl_model_path)),
            "artifact_hash": model_hash,
            "feature_names": self._vec.get_feature_names_out().tolist(),
        }


def main() -> None:
    """Main function to load the model and make a prediction."""
    predictor = DurationPredictor(settings)
    predictor.load()

    # Example features for prediction
    example_features = [
        {"PU_DO": "260_193", "Trip_Distance": 2.740000009536743},
        {"PU_DO": "260_226", "Trip_Distance": 1.4299999475479126},
        {"PU_DO": "181_249", "Trip_Distance": 3.700000047683716},
        {"PU_DO": "260_260", "Trip_Distance": 0.4000000059604645},
        {"PU_DO": "74_244", "Trip_Distance": 2.740000009536743},
    ]

    # Make a prediction
    predicted_duration = predictor.predict(example_features[0])
    _log.info(f"Predicted trip duration: {predicted_duration['Prediction']:.2f} minutes")

    # Make batch prediction
    predicted_durations = predictor.predict_batch(example_features)
    _log.info(f"Predicted trip durations: {predicted_durations}")
