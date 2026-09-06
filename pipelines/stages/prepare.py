"""DVC stage: raw parquet -> cleaned parquet (duration in minutes + filters)."""
from pathlib import Path

import pandas as pd

from prodml.data import clean_data

ROOT = Path(__file__).resolve().parents[2]
MONTHS = ["2024-11", "2024-12"]

for month in MONTHS:
    raw = ROOT / "data/raw" / f"green_tripdata_{month}.parquet"
    out = ROOT / "data/prepared" / f"green_tripdata_{month}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = clean_data(pd.read_parquet(raw))
    df.to_parquet(out)
    print(f"prepare {month}: {len(df)} rows -> {out.relative_to(ROOT)}")