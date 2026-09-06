"""DVC evaluate: model bundle + features (val month) -> metrics.json + predictions."""
import json
import pickle
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "pipelines/params.yaml") as f:
    cfg = yaml.safe_load(f)
val_month = cfg["prepare"]["months"][1]

with open(ROOT / "models/model.pkl", "rb") as f:
    bundle = pickle.load(f)
vec, model = bundle["vectorizer"], bundle["model"]

df = pd.read_parquet(ROOT / "data/features" / f"green_tripdata_{val_month}.parquet")
X_dicts = df[["Trip_Distance", "PU_DO"]].to_dict(orient="records")
y = df["Trip_Duration"].values

preds = model.predict(vec.transform(X_dicts))    # transform only — never re-fit

metrics = {
    "rmse": float(root_mean_squared_error(y, preds)),
    "mae": float(mean_absolute_error(y, preds)),
    "r2": float(r2_score(y, preds)),
}

out_dir = ROOT / "data/evaluate"
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"actual": y, "predicted": preds}).to_parquet(
    out_dir / f"predictions_{val_month}.parquet")

with open(ROOT / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"evaluate: {metrics}")