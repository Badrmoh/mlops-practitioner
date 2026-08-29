import pickle

import numpy as np
import onnxruntime as ort

from prodml.config import TrainingSettings
from prodml.export import export_to_onnx


def test_export_produces_faithful_model(tmp_path):
    """
    Test that exporting a model to ONNX format produces predictions
    that are close to the original model's predictions.

    Background:
    - Pickle computes in float64, ONNX in float32 → weights are rounded at export.
    - The predictions therefore drift by a fixed ABSOLUTE amount (~1e-3),
      independent of the prediction's size (the drift is the sum of weight
      roundings — a property of the model, not of the prediction).
    - Naturally, small predictions need lesser tolerance than large ones,
          but errors are the same at all prediction sizes.

    Tolerance: |f32_onnx − f64_pickle| ≤ atol + rtol × |f64_pickle|
    - rtol × |b|: relative allowance — scales with the prediction. It makes
      the comparison scale-aware: large predictions aren't judged with a
      microscope. But at small |b| it collapses to ~0.
    - atol: the absolute floor — the ONLY term governing small predictions.

    """
    settings = TrainingSettings()
    with open(settings.pkl_model_path, "rb") as f:
        bundle = pickle.load(f)

    onnx_path = tmp_path / "model.onnx"
    export_to_onnx(bundle["model"], bundle["vectorizer"], onnx_path)

    n_features = len(bundle["vectorizer"].get_feature_names_out())
    X = np.random.default_rng(42).random((500, n_features)).astype(np.float32)

    pred_pickle = bundle["model"].predict(X)

    sess = ort.InferenceSession(str(onnx_path))
    pred_onnx = sess.run(None, {"features": X})[0]
    pred_onnx = pred_onnx.ravel()  # ← flatten to (500,)

    # diff = np.abs(pred_pickle - pred_onnx)
    # print("max:", diff.max(), "| mean:", diff.mean())

    assert np.allclose(pred_pickle, pred_onnx, atol=1e-2)
