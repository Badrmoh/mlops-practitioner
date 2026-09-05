"""MP2 2.5 — XGBoost hyperparameter sweep as nested MLflow runs."""

import itertools
import logging
import os
import time

import mlflow

from prodml.config import TrainingSettings
from prodml.logging_config import setup_logger
from prodml.train import Trainer, file_md5, git_commit, git_user

GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 6],
    "learning_rate": [0.05, 0.2],
}

TRIALS = [
    dict(zip(GRID, combo)) for combo in itertools.product(*GRID.values())
]  # 3 x 2 x 2 = 12 trials


def main() -> None:
    settings = TrainingSettings()
    setup_logger(log_level=settings.log_level, log_format=settings.log_format)
    _log = logging.getLogger(__name__)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("Ride Duration Model")

    data_hash = file_md5(settings.training_set)
    author = os.getenv("GIT_USER", git_user())
    commit = os.getenv("GIT_COMMIT", git_commit())

    _log.info(f"Starting sweep: {len(TRIALS)} trials")
    with mlflow.start_run(run_name="xgboost-sweep"):
        for i, params in enumerate(TRIALS):
            with mlflow.start_run(nested=True, run_name=f"xgboost-trial-{i:02d}"):
                mlflow.log_params(params)
                trainer = Trainer(settings, model_name="xgboost", model_params=params)

                t0 = time.monotonic()
                trainer.train()
                mlflow.log_metric("train_duration_s", time.monotonic() - t0)

                trainer.validate()
                mlflow.log_metrics(trainer.metrics)

                trainer.save_model()
                mlflow.log_artifact(settings.pkl_model_path, artifact_path="model")
                mlflow.log_metric("model_size_mb", os.path.getsize(settings.pkl_model_path) / 1e6)

                mlflow.set_tags({
                    "git_commit": os.getenv("GIT_COMMIT", git_commit()),
                    "data_version": data_hash,
                    "author": os.getenv("GIT_USER", git_user()),
                    "framework": "xgboost",
                })
                _log.info(f"trial {i:02d} {params} -> {trainer.metrics}")


if __name__ == "__main__":
    main()