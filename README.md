# MLOps Practitioner Mini Projects

This project serves as a notebook for all projects in the course

## Quick Setup

## MP1

0. Install dependencies

```
uv sync --group dev
```

1. Configuration

```
cp .env.example .env
```

add your configuration in the .env file


2. Run training:

```
uv run prodml-train
source .env
ls $PRODML_TRAIN_MODEL_PATH
```

3. Run Validation

Make sure that PRODML_PREDICT_MODEL_PATH is set in `.env` correctly

```
uv run prodml-predict
```

4. Run Parity Test

```
uv run tests test_serialization_latency.py
```


5. Run Benchmarking

Compare model performance between pickle and ONNX formats:

```
uv run optimizations/benchmark_pk_onnx.py
```


---

## Configurations

| Variable | Used By | Required | Default | Valid Values | Description |
| --- | --- | --- | --- | --- | --- |
| `PRODML_TRAIN_TRAINING_SET` | `prodml-train` | Yes | - | File path | Path to the training dataset. |
| `PRODML_TRAIN_VALIDATION_SET` | `prodml-train` | Yes | - | File path | Path to the validation dataset used during training/evaluation. |
| `PRODML_TRAIN_MODEL_PATH` | `prodml-train` | No | `models/model.pkl` | File path | Path where the trained model artifact is saved. |
| `PRODML_TRAIN_ONNX_PATH` | `prodml-train` | No | - | File path | Path where the ONNX model export is saved. |
| `PRODML_TRAIN_MODEL_NAME` | `prodml-train` | No | `linear` | String | Logical model name. |
| `PRODML_TRAIN_LOG_LEVEL` | `prodml-train` | No | `info` | `debug`, `info`, `warning`, `error`, `critical` | Logging level for training. |
| `PRODML_TRAIN_LOG_FORMAT` | `prodml-train` | No | `default` | `default`, `json` | Logging format for training. |
| `PRODML_PREDICT_MODEL_PATH` | `prodml-predict` | No | `models/model.pkl` | File path | Path to the trained model artifact loaded for prediction. |
| `PRODML_PREDICT_LOG_LEVEL` | `prodml-predict` | No | `info` | `debug`, `info`, `warning`, `error`, `critical` | Logging level for prediction. |
| `PRODML_PREDICT_LOG_FORMAT` | `prodml-predict` | No | `default` | `default`, `json` | Logging format for prediction. |
| `PRODML_BENCHMARK_MODEL_PATH` | `prodml-benchmark` | No | `models/model.pkl` | File path | Path to the trained model artifact for benchmarking. |
| `PRODML_BENCHMARK_ONNX_PATH` | `prodml-benchmark` | No | - | File path | Path to the ONNX model for benchmarking. |
| `PRODML_BENCHMARK_LOG_LEVEL` | `prodml-benchmark` | No | `info` | `debug`, `info`, `warning`, `error`, `critical` | Logging level for benchmarking. |
| `PRODML_BENCHMARK_LOG_FORMAT` | `prodml-benchmark` | No | `default` | `default`, `json` | Logging format for benchmarking. |