import pickle

import pytest
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from prodml.config import PredictSettings
from prodml.predict import DurationPredictor


@pytest.fixture
def sample_features() -> list[dict]:
    """Realistic trip dicts matching the training keys exactly."""
    return [
        {"PU_DO": "260_193", "Trip_Distance": 2.74},
        {"PU_DO": "74_244", "Trip_Distance": 1.43},
        {"PU_DO": "181_249", "Trip_Distance": 3.70},
    ]


@pytest.fixture(scope="session")
def model_bundle(tmp_path_factory) -> dict:
    """Raw trained artifact {"model", "vectorizer"} — trained ONCE per run."""
    trips = [
        {"PU_DO": "260_193", "Trip_Distance": 2.7},
        {"PU_DO": "260_193", "Trip_Distance": 3.1},
        {"PU_DO": "74_244", "Trip_Distance": 1.4},
        {"PU_DO": "181_249", "Trip_Distance": 3.7},
        {"PU_DO": "260_260", "Trip_Distance": 0.4},
    ]
    durations = [12.0, 13.0, 8.0, 15.0, 5.0]

    vec = DictVectorizer()
    X = vec.fit_transform(trips)  # fit ONCE — the fitted vectorizer IS the vocabulary
    model = LinearRegression().fit(X, durations)
    return {"model": model, "vectorizer": vec}


@pytest.fixture(scope="session")
def trained_model(model_bundle, tmp_path_factory) -> DurationPredictor:
    """DurationPredictor loaded through the REAL load() production path."""
    bundle_path = tmp_path_factory.mktemp("models") / "model.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump(model_bundle, f)

    predictor = DurationPredictor(PredictSettings(pkl_model_path=str(bundle_path)))
    predictor.load()
    return predictor


@pytest.fixture
def client(trained_model, monkeypatch):
    """TestClient with the real lifespan, but the model injected — no real files."""
    from fastapi.testclient import TestClient

    from prodml.api import main

    monkeypatch.setattr(main, "load_predictor", lambda: trained_model)

    with TestClient(main.app) as test_client:
        yield test_client