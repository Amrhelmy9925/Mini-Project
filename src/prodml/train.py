"""Train linear models on the taxi duration data and save artifacts.

Notebook-style flow: get features from features.py, fit models, print
metrics, persist artifacts.
"""

import logging
import os

import joblib
import numpy as np
import skl2onnx
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from .config import settings
from .features import get_features
from .api.logging_conf import setup_logging

logger = logging.getLogger(__name__)


def train_models(X_train_dv, X_test_dv, y_train, y_test):
    """Fit both models, print metrics, return the fitted dict.

    root_mean_squared_error is used instead of
    mean_squared_error(squared=False), which was removed in sklearn >= 1.6.
    """
    models = {
        "lr": LinearRegression(),
        # Ridge = LinearRegression + L2 penalty. No random_state: default
        # solver is deterministic and would ignore it anyway.
        "rd": Ridge(alpha=1.0),
        # "ls": Lasso(alpha=0.1),
    }

    for name, model in models.items():
        # fit on the vectorized training matrix (sparse one-hot features)
        # .toarray().astype(np.float32) ensures dense float32 input,
        # avoiding sklearn sparse int64-indices incompatibility.
        model.fit(X_train_dv.astype(np.float32), y_train.astype(np.float32))
        y_pred = model.predict(X_test_dv)

        rmse = root_mean_squared_error(y_test, y_pred)  # in minutes
        mae = mean_absolute_error(y_test, y_pred)       # in minutes
        r2 = r2_score(y_test, y_pred)                   # variance explained

        logger.info(
            "model trained",
            extra={"model": name, "rmse": float(rmse), "mae": float(mae), "r2": float(r2)},
        )

    return models


def save_artifacts(dv, models, X_train_dv) -> None:
    """Persist everything inference needs.

    - vectorizer as joblib: ONNX does not embed preprocessing, so predict
      must apply the same dv before feeding the model.
    - the "lr" model converted to ONNX at settings.model_path — this is
      what predict.DurationPredictor loads. skl2onnx converts sklearn
      estimators to an ONNX graph; float input type matches what
      predict.py feeds it.
    """
    models_dir = os.path.dirname(settings.model_path) or "."
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(dv, settings.vectorizer_path)
    logger.info("saved vectorizer", extra={"path": settings.vectorizer_path})

    # float32 sample tells skl2onnx to build a single-precision graph,
    # matching the float32 matrix predict.py feeds at inference.
    onnx_model = skl2onnx.to_onnx(
        models["rd"],
        X_train_dv[:1].toarray().astype(np.float32),
        target_opset=settings.onnx_opset,
    )
    with open(settings.model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logger.info("saved ONNX model", extra={"path": settings.model_path})

    # All fitted sklearn models as one joblib dict.
    joblib.dump(models, settings.models_path)
    logger.info("saved models", extra={"path": settings.models_path})


def main() -> None:
    setup_logging(level=logging.INFO)
    dv, X_train_dv, X_test_dv, y_train, y_test = get_features()
    models = train_models(X_train_dv, X_test_dv, y_train, y_test)
    save_artifacts(dv, models, X_train_dv)


if __name__ == "__main__":
    main()
