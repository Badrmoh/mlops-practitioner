from pathlib import Path

import onnx
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.base import BaseEstimator
from sklearn.feature_extraction import DictVectorizer


def export_to_onnx(
    model: BaseEstimator, vectorizer: DictVectorizer, path: Path
) -> None:
    """Export the trained model to ONNX format."""

    pickle_vectorizer_len = len(vectorizer.get_feature_names_out())

    onnx_model = to_onnx(
        model,
        initial_types=[("features", FloatTensorType([None, pickle_vectorizer_len]))],
    )

    # validate, then write the file
    onnx.checker.check_model(onnx_model)

    onnx_path = path.with_suffix(".onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

if __name__ == "__main__":
    import pickle
    from pathlib import Path

    from prodml.config import TrainingSettings

    settings = TrainingSettings()
    with open(settings.model_path, "rb") as f:
        model = pickle.load(f)


    onnx_path = Path(settings.model_path).parent / "model.onnx"
    export_to_onnx(model["model"], model["vectorizer"], onnx_path)
