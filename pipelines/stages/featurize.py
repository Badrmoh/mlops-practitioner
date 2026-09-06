"""DVC stage: cleaned parquet -> features parquet (PU_DO + model columns)."""
from pathlib import Path

import pandas as pd

from prodml.features import compose_features

ROOT = Path(__file__).resolve().parents[2]
MONTHS = ["2024-11", "2024-12"]
COLS = ["Trip_Distance", "PU_DO", "Trip_Duration"]

for month in MONTHS:
    src = ROOT / "data/prepared" / f"green_tripdata_{month}.parquet"
    out = ROOT / "data/features" / f"green_tripdata_{month}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = compose_features(pd.read_parquet(src))[COLS]
    df.to_parquet(out)
    print(f"featurize {month}: {len(df)} rows -> {out.relative_to(ROOT)}")