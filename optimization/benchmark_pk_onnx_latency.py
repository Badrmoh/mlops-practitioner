# https://superfastpython.com/python-benchmarking-best-practices/

import logging
import pickle
import time

import numpy as np
import onnxruntime as ort

from prodml.config import BenchmarkSettings
from prodml.logging_config import setup_logger

settings = BenchmarkSettings()
setup_logger(settings.log_level, settings.log_format)
_log = logging.getLogger(__name__)

# load both models once
with open(settings.pkl_path, "rb") as f:
    artifact = pickle.load(f)
model, vec = artifact["model"], artifact["vectorizer"]
session = ort.InferenceSession(settings.onnx_path)
input_name = session.get_inputs()[0].name
n_features = len(vec.get_feature_names_out())

# same 500 random rows for both — shape matters, values don't
X = np.random.default_rng(42).random((500, n_features)).astype(np.float32)

# convert a vector to 2-D array, and auto detect number of features by "-1".
pkl_fn = lambda row: model.predict(row.reshape(1, -1))
onnx_fn = lambda row: session.run(None, {input_name: row.reshape(1, -1)})


def measure(fn):
    """
    Measure the mean and 95th percentile latency of `fn` over `X`.
    Benchmark best practices:
    - use time.monotonic() instead of time.time() to avoid clock jumps.
      Normal time.time() can jump forward or backward if the system clock is adjusted,
      which can lead to negative durations or inflated timings. time.monotonic() is guaranteed to always move forward.
    - warmup the function with 50 calls before timing to avoid cold-start effects
    - in order to make the benchmark reproducible, we use a seed "42" to get same random numbers every time.
    Returns:
        mean_ms (float): Mean latency in milliseconds.
        p95_ms (float): 95th percentile latency in milliseconds.
    """
    for row in X[:50]:  # warmup — caches, lazy init
        fn(row)
    times = []
    for row in X:  # 500 timed calls, same rows for both
        t0 = time.monotonic()
        fn(row)
        times.append(time.monotonic() - t0)
    ms = np.array(times) * 1000
    return ms.mean(), np.percentile(ms, 95)


pkl_mean, pkl_p95 = measure(pkl_fn)
onnx_mean, onnx_p95 = measure(onnx_fn)

_log.info(f"pickle  mean {pkl_mean:.3f} ms | p95 {pkl_p95:.3f} ms")
_log.info(f"onnx    mean {onnx_mean:.3f} ms | p95 {onnx_p95:.3f} ms")
