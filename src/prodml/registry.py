import argparse
import logging
import os
import pickle
from pathlib import Path

import mlflow

_log = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "ride-duration-predictor"
STAGES = ("None", "Staging", "Production", "Archived")


class RideDurationModel(mlflow.pyfunc.PythonModel):
    """pyfunc wrapper around the {"model", "vectorizer"} bundle.

    Loaded via mlflow.pyfunc.load_model("models:/..."), the vectorizer
    travels inside the model — vocabulary and weights are ONE artifact.
    """

    def load_context(self, context):
        with open(context.artifacts["bundle"], "rb") as f:
            bundle = pickle.load(f)
        self._model = bundle["model"]
        self._vec = bundle["vectorizer"]

    def predict(self, context, model_input):
        """model_input: pandas DataFrame (pyfunc convention) -> predictions."""
        records = model_input.to_dict(orient="records")
        return self._model.predict(self._vec.transform(records))


def package_and_register(run_id: str, model_name: str = DEFAULT_MODEL_NAME, stage: str | None = None):
    """Download a run's bundle, log it as a pyfunc model in the same run, register it."""
    local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
    pkls = list(Path(local).glob("*.pkl"))
    if not pkls:
        raise RuntimeError(f"No bundle pickle under artifact path 'model' in run {run_id}")
    bundle_path = pkls[0]

    with mlflow.start_run(run_id=run_id):
        mlflow.pyfunc.log_model(
            artifact_path="model_pyfunc",
            python_model=RideDurationModel(),
            artifacts={"bundle": str(bundle_path)},
        )

    version = mlflow.register_model(f"runs:/{run_id}/model_pyfunc", model_name)
    _log.info(f"Registered {model_name} version {version.version} from run {run_id}")
    if stage:
        transition(model_name, version.version, stage)
    return version


def transition(model_name: str, version: str | int, stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    client = mlflow.MlflowClient()
    client.transition_model_version_stage(model_name, str(version), stage)
    _log.info(f"{model_name} v{version} -> {stage}")


def _production_metric(client, model_name: str, metric: str) -> tuple[float, int] | None:
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        return None
    prod = max(versions, key=lambda v: v.version)
    return client.get_run(prod.run_id).data.metrics.get(metric), prod.version


def promote_if_better(candidate_run_id: str, model_name: str = DEFAULT_MODEL_NAME, metric: str = "mae"):
    """Gate: promote the candidate to Production only if it beats the current model."""
    client = mlflow.MlflowClient()
    candidate = client.get_run(candidate_run_id).data.metrics.get(metric)
    if candidate is None:
        raise ValueError(f"Run {candidate_run_id} has no metric '{metric}'")

    current = _production_metric(client, model_name, metric)
    if current is None:
        _log.info(f"No Production model yet — promoting candidate ({metric}={candidate})")
        return package_and_register(candidate_run_id, model_name, stage="Production")

    current_value, current_version = current
    if candidate < current_value:
        _log.info(f"Candidate {metric}={candidate} beats Production {metric}={current_value} — promoting")
        new_version = package_and_register(candidate_run_id, model_name, stage="Production")
        for v in client.get_latest_versions(model_name, stages=["Production"]):
            if v.version != new_version.version:
                transition(model_name, v.version, "Archived")
        return new_version

    _log.info(f"Candidate {metric}={candidate} does NOT beat Production {metric}={current_value} — skipped")
    return None


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(prog="python -m prodml.registry")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_register = sub.add_parser("register", help="package a run's bundle and register it")
    p_register.add_argument("run_id")
    p_register.add_argument("--name", default=DEFAULT_MODEL_NAME)
    p_register.add_argument("--stage", choices=STAGES)

    p_stage = sub.add_parser("stage", help="move a registered version between stages")
    p_stage.add_argument("version", type=int)
    p_stage.add_argument("stage", choices=STAGES)
    p_stage.add_argument("--name", default=DEFAULT_MODEL_NAME)

    p_gate = sub.add_parser("promote-if-better", help="promote only if the candidate beats Production")
    p_gate.add_argument("run_id")
    p_gate.add_argument("--name", default=DEFAULT_MODEL_NAME)
    p_gate.add_argument("--metric", default="mae")

    args = parser.parse_args(argv)

    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        parser.error("MLFLOW_TRACKING_URI is not set")
    mlflow.set_tracking_uri(uri)

    if args.cmd == "register":
        package_and_register(args.run_id, args.name, args.stage)
    elif args.cmd == "stage":
        transition(args.name, args.version, args.stage)
    elif args.cmd == "promote-if-better":
        version = promote_if_better(args.run_id, args.name, args.metric)
        print(f"promoted: version {version.version}" if version else "not promoted")


if __name__ == "__main__":
    main()