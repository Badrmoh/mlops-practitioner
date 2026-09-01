import pytest

def test_prediction_returns_float(trained_model, sample_features):
    result = trained_model.predict(sample_features[0])
    assert isinstance(result["Prediction"], float)


def test_prediction_in_sane_range(trained_model, sample_features):
    result = trained_model.predict(sample_features[0])
    pred = result["Prediction"]
    assert 0.0 < pred < 60.0  # minutes: positive, plausible taxi duration


def test_prediction_deterministic(trained_model, sample_features):
    row = sample_features[0]
    assert trained_model.predict(row) == trained_model.predict(row)


@pytest.mark.parametrize("pu_do, distance", [
    ("260_193", 2.74),   # case 1
    ("999_999", 2.5),    # case 2 — unseen pair
    ("74_244", 0.0),     # case 3 — zero distance
])
def test_edge_cases(trained_model, pu_do, distance):
    result = trained_model.predict({"PU_DO": pu_do, "Trip_Distance": distance})
    assert isinstance(result["Prediction"], float)