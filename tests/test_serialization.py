import numpy as np
import onnxruntime as ort

from prodml.export import export_to_onnx


def test_export_produces_faithful_model(tmp_path, model_bundle):
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

    Uses the hermetic model_bundle fixture (not the real artifact) so the
    test runs anywhere — no .env, no data files.
    """
    onnx_path = tmp_path / "model.onnx"
    export_to_onnx(model_bundle["model"], model_bundle["vectorizer"], onnx_path)

    n_features = len(model_bundle["vectorizer"].get_feature_names_out())
    X = np.random.default_rng(42).random((500, n_features)).astype(np.float32)

    pred_pickle = model_bundle["model"].predict(X)

    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name  # don't hardcode skl2onnx's naming
    pred_onnx = sess.run(None, {input_name: X})[0]
    pred_onnx = pred_onnx.ravel()  # ← flatten to (500,)

    assert np.allclose(pred_pickle, pred_onnx, atol=1e-2)