"""Production-ready model inference for NYC taxi duration prediction.

Seam between the model artifact, the BentoML runner, and the API layer
(Modules 3-5). Everything is typed, timed, and configured via environment
variables so there are zero hardcoded paths.

Expects an ONNX model at settings.model_path and the fitted
DictVectorizer at settings.vectorizer_path (both written by train.py).
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import joblib
import numpy as np
import onnxruntime as rt

from .config import settings

logger = logging.getLogger(__name__)


def timed(fn: Callable) -> Callable:
    """Decorator that logs runtime of the wrapped function."""

    @wraps(fn)  # preserves fn's name/docstring — without it introspection breaks
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%s: %.4fs", fn.__name__, elapsed)
        return result

    return wrapper




# ---------------------------------------------------------------------------
# Core class — sits behind every deployment flavor (pickle, ONNX, BentoML)
# ---------------------------------------------------------------------------

class DurationPredictor:
    """Predict trip duration from ride-feature dicts using an ONNX model.

    The public API is intentionally small so it can be dropped into any
    framework (Flask, FastAPI, BentoML, CLI) without change.
    """

    def __init__(self) -> None:
        self.session: Optional[rt.InferenceSession] = None
        self.dv: Optional[Any] = None
        self._input_name: Optional[str] = None

    # ------------------------------------------------------------------
    # Persistence — .load() is the single entry point for every deployment
    # ------------------------------------------------------------------

    def load(self, path: Optional[str] = None, dv: Optional[Any] = None) -> None:
        """Load the ONNX model and the fitted DictVectorizer.

        Parameters
        ----------
        path : str | None
            Override the model path from config (useful for tests).
        dv : Any | None
            Pre-fitted DictVectorizer. If None, loads it from
            settings.vectorizer_path (the joblib train.py saved) — ONNX does
            not embed preprocessing, so the vectorizer always travels
            alongside the model.
        """
        model_path = path or settings.model_path

        if dv is None:
            dv = joblib.load(settings.vectorizer_path)

        sess_options = rt.SessionOptions()
        sess_options.intra_op_num_threads = settings.intra_op_threads

        # providers=None lets ONNX Runtime pick its own defaults (CUDA when
        # available); settings.onnx_provider overrides via env var.
        self.session = rt.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=[settings.onnx_provider] if settings.onnx_provider else None,
        )
        self.dv = dv
        self._input_name = self.session.get_inputs()[0].name

    def _vectorize(self, features: List[Dict[str, Any]]) -> np.ndarray:
        """Transform raw feature dicts into the float32 dense matrix the
        ONNX graph expects. Shared by predict_one and predict_batch so the
        preprocessing can never drift between them."""
        if self.session is None or self.dv is None:
            raise RuntimeError("Call .load() before predicting")

        # Cast on the sparse side first: avoids materializing a float64
        # dense buffer just to copy it down to float32.
        X_vec = self.dv.transform(features).astype(np.float32)
        return X_vec.toarray()

    # ------------------------------------------------------------------
    # Single-ride prediction
    # ------------------------------------------------------------------

    @timed
    def predict_one(self, features: Dict[str, Any]) -> float:
        """Predict duration for one ride.

        Parameters
        ----------
        features : dict[str, Any]
            Ride features as they came from the API / ingestion layer.
            Must contain the same keys the DictVectorizer was fitted on.

        Returns
        -------
        float
            Predicted trip duration in minutes.
        """
        return self.predict_batch([features])[0]

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    @timed
    def predict_batch(self, features: List[Dict[str, Any]]) -> List[float]:
        """Predict duration for many rides.

        Parameters
        ----------
        features : list[dict[str, Any]]
            List of ride-feature dicts (same schema as .predict_one).

        Returns
        -------
        list[float]
            Predicted durations in minutes, one per input row.
        """
        if self.session is None:
            raise RuntimeError("Call .load() before predicting")
        X_dense = self._vectorize(features)
        preds = self.session.run(None, {self._input_name: X_dense})[0]
        # skl2onnx regression output has shape (N,1) — ravel to 1-D.
        return preds.ravel().tolist()
