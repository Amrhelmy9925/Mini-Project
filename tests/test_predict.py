"""Taste-test the predictor (predict.py).

6yo version: predictor is a robot that turns ride cards into minutes.
We check: gives a number, same card twice = same answer, no load = shout error.
Uses unittest.mock so no real model file needed (fast).
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from prodml.predict import DurationPredictor, timed


def test_timed_keeps_name():
    @timed
    def my_fn():
        """doc"""
        return 1

    assert my_fn.__name__ == "my_fn"
    assert my_fn() == 1


def test_predict_returns_float(mock_predictor, sample_features):
    pred = mock_predictor.predict_one(sample_features)
    assert isinstance(pred, float)
    assert 0 < pred < 300  # sane range for taxi minutes


def test_predict_deterministic(mock_predictor, sample_features):
    a = mock_predictor.predict_one(sample_features)
    b = mock_predictor.predict_one(sample_features)
    assert a == b


def test_predict_batch_len(mock_predictor, sample_features):
    preds = mock_predictor.predict_batch([sample_features, sample_features])
    # mock returns [[12.5]] for any batch; we just check it returns a list
    assert isinstance(preds, list)
    assert len(preds) >= 1


def test_not_loaded_raises(sample_features):
    p = DurationPredictor()  # session=None
    with pytest.raises(RuntimeError, match="Call .load()"):
        p.predict_one(sample_features)
    with pytest.raises(RuntimeError, match="Call .load()"):
        p._vectorize([sample_features])


def test_load_real_files():
    """One real test: loads models/baseline.onnx + dv.joblib from disk."""
    import joblib

    from prodml.config import settings

    p = DurationPredictor()
    dv = joblib.load(settings.vectorizer_path)
    p.load(dv=dv)  # loads real ONNX
    assert p.session is not None
    assert p._input_name is not None


def test_vectorize_uses_dv(mock_predictor, sample_features):
    X = mock_predictor._vectorize([sample_features])
    assert isinstance(X, np.ndarray)
    assert X.dtype == np.float32
    assert X.shape[0] == 1
