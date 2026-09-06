"""DVC train: features parquet (train month) + params -> model bundle."""
import pickle
from pathlib import Path

import pandas as pd
import yaml
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "pipelines/params.yaml") as f:
    cfg = yaml.safe_load(f)["train"]

df = pd.read_parquet(ROOT / "data/features/green_tripdata_2024-11.parquet")
X_dicts = df[["Trip_Distance", "PU_DO"]].to_dict(orient="records")
y = df["Trip_Duration"].values

vec = DictVectorizer()
X = vec.fit_transform(X_dicts)                    # fit ONCE, on train

model = XGBRegressor(**cfg["model_params"])
model.fit(X, y)

out = ROOT / "models/model.pkl"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "wb") as f:
    pickle.dump({"model": model, "vectorizer": vec}, f)
print(f"train xgboost {cfg['model_params']} -> {out.relative_to(ROOT)}")