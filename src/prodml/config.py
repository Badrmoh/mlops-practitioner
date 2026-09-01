from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class TrainingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PRODML_TRAIN_",
        env_file=REPO_ROOT / ".env",
        extra="ignore",
    )
    training_set: str
    validation_set: str
    pkl_model_path: str = "models/model.pkl"
    model_name: str = "linear"
    model_params: dict[str, Any] = {}
    log_level: str = "info"
    log_format: str = "default"


class PredictSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PRODML_PREDICT_",
        env_file=REPO_ROOT / ".env",
        extra="ignore",
    )
    pkl_model_path: str = "models/model.pkl"
    log_level: str = "info"
    log_format: str = "default"


class BenchmarkSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PRODML_BENCHMARK_",
        env_file=REPO_ROOT / ".env",
        extra="ignore",
    )
    pkl_model_path: str = "models/prodml_model.pkl"
    onnx_model_path: str = "models/prodml_model.onnx"
    log_level: str = "info"
    log_format: str = "default"