import functools
import logging
import time

from prodml.config import PredictSettings
from prodml.logging_config import setup_logger

settings = PredictSettings()
setup_logger(log_level=settings.log_level, log_format=settings.log_format)
_log = logging.getLogger(__name__)


def timed(fn):
    """Log the execution time of the decorated function."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.monotonic()  # NOT time.time — immune to clock jumps
        result = fn(*args, **kwargs)
        duration = time.monotonic() - start
        _log.info(f"{fn.__name__} took {duration*1000:.1f} ms")
        return result

    return wrapper


class DurationPredictor:
    """A class for predicting the duration of a trip based on input features."""

    def __init__(self, settings: PredictSettings):
        self.settings = settings

    def load(self) -> None:
        """Load a trained model and vectorizer from disk."""
        import pickle

        with open(self.settings.model_path, "rb") as f_in:
            model_data = pickle.load(f_in)
            self._model = model_data["model"]
            self._vec = model_data["vectorizer"]

    @timed
    def predict(self, features: dict) -> float:
        """
        Predict the duration of a trip given its features.

        Args:
            features (dict): A dictionary containing the features of the trip.

        Returns:
            float: The predicted duration of the trip in minutes.
        """
        X = self._vec.transform([features])
        prediction = self._model.predict(X)
        return prediction[0]

    @timed
    def predict_batch(self, features_list: list[dict]) -> list[float]:
        """
        Predict the duration of multiple trips given their features.

        Args:
            features_list (list[dict]): A list of dictionaries, each containing the features of a trip.

        Returns:
            list[float]: A list of predicted durations for each trip in minutes.
        """
        X = self._vec.transform(features_list)
        predictions = self._model.predict(X)
        return predictions.tolist()


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
    _log.info(f"Predicted trip duration: {predicted_duration:.2f} minutes")

    # Make batch prediction
    predicted_durations = predictor.predict_batch(example_features)
    _log.info(f"Predicted trip durations: {predicted_durations}")
