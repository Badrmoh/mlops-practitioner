import pandas as pd


def compose_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compose features for the input DataFrame."""
    df["PU_DO"] = (
        df["Start_Zone"].astype("string") + "_" + df["End_Zone"].astype("string")
    )
    return df
