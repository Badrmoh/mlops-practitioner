import pandas as pd

from prodml.features import compose_features


def test_compose_features_normal():
    df = pd.DataFrame({"Start_Zone": ["260", "74"], "End_Zone": ["193", "244"]})
    result = compose_features(df)
    assert result["PU_DO"].tolist() == ["260_193", "74_244"]


def test_missing_category(trained_model):
    # missing zone → compose produces NA → handled without crash
    df = pd.DataFrame({"Start_Zone": [None, "74"], "End_Zone": ["193", "244"]})
    result = compose_features(df)
    assert result["PU_DO"].iloc[0] is pd.NA
    assert result["PU_DO"].iloc[1] == "74_244"


def test_zero_distance(trained_model):
    # feature edge case: distance 0 → still produces a sane prediction
    row = {"PU_DO": "260_193", "Trip_Distance": 0.0}
    result = trained_model.predict(row)
    assert isinstance(result["Prediction"], float)


def test_unseen_pu_do_pair(trained_model):
    # PU_DO not in training vocabulary → vectorizer silently zeroes it
    row = {"PU_DO": "999_999", "Trip_Distance": 2.5}
    result = trained_model.predict(row)
    assert isinstance(result["Prediction"], float)