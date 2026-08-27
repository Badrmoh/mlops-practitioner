import pandas as pd

def load_training_data(data_path: str) -> pd.DataFrame:
    """
    Load the training data from the specified parquet file.
    
    Args:
        data_path (str): The path to the parquet file containing the training data.
    Returns:
        pd.DataFrame: The loaded training data as a pandas DataFrame.
    """
    return pd.read_parquet(data_path)

def load_validation_data(data_path: str) -> pd.DataFrame:
    """
    Load the validation data from the specified parquet file.
    
    Args:
        data_path (str): The path to the parquet file containing the validation data.
    Returns:
        pd.DataFrame: The loaded validation data as a pandas DataFrame.
    """
    return pd.read_parquet(data_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df["Trip_Start_dt"] = pd.to_datetime(df["Trip_Start"])
    df["Trip_End_dt"]   = pd.to_datetime(df["Trip_End"])
    df["Trip_Duration"] = (df["Trip_Duration"] / 60).astype("float")
    return df[(df["Trip_Duration"] >= 1) & (df["Trip_Duration"] <= 60)]