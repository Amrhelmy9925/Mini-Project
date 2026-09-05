"""Shared fixtures: one cookie dough for all taste-testers.

Why this file exists:
- pytest loads conftest.py automatically before any test.
- Fixtures here avoid reloading the big parquet (131s) in every test.
- Uses tiny fake data + mocks so tests are fast (<1s) and don't need real training run.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction import DictVectorizer


@pytest.fixture
def sample_features() -> dict:
    """One ride, like API receives. Minimal valid payload."""
    return {
        "VendorID": "1",
        "passenger_count": 1.0,
        "trip_distance": 2.5,
        "RatecodeID": "1",
        "store_and_fwd_flag": "N",
        "PULocationID": "100",
        "DOLocationID": "200",
        "payment_type": "1",
        "fare_amount": 10.0,
        "extra": 0.5,
        "mta_tax": 0.5,
        "tip_amount": 2.0,
        "tolls_amount": 0.0,
        "improvement_surcharge": 0.3,
        "total_amount": 13.3,
        "congestion_surcharge": 2.5,
        "Airport_fee": 0.0,
        "cbd_congestion_fee": 0.0,
    }


@pytest.fixture
def tiny_dv():
    """Tiny DictVectorizer fitted on 2 rows — no parquet needed."""
    train_dicts = [
        {"VendorID": "1", "trip_distance": 1.0, "PULocationID": "100", "DOLocationID": "200"},
        {"VendorID": "2", "trip_distance": 5.0, "PULocationID": "101", "DOLocationID": "201"},
    ]
    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(train_dicts)
    return dv, X_train


@pytest.fixture
def mock_predictor(sample_features, tiny_dv):
    """DurationPredictor with fake ONNX session — no model file needed."""
    from unittest.mock import MagicMock

    from prodml.predict import DurationPredictor

    dv, _ = tiny_dv
    p = DurationPredictor()
    # Fake session: predict_batch calls session.run() -> returns [[12.5]] etc.
    fake_session = MagicMock()
    fake_session.run.return_value = [np.array([[12.5]])]
    fake_session.get_inputs.return_value = [MagicMock(name="input")]
    p.session = fake_session
    p.dv = dv
    p._input_name = "input"
    return p


@pytest.fixture
def client(monkeypatch):
    """FastAPI TestClient with mocked model load — no real ONNX needed."""
    from unittest.mock import MagicMock

    import numpy as np
    from fastapi.testclient import TestClient

    import prodml.api.main as main_module

    # Avoid loading real files on startup: replace predictor.load with no-op
    # that sets a fake session so /health returns ok.
    def _fake_load(self, path=None, dv=None):
        from sklearn.feature_extraction import DictVectorizer

        self.dv = DictVectorizer()
        self.dv.fit([{"trip_distance": 1.0}])
        self.session = MagicMock()
        # return one 9.9 per input row, so batch of N -> N predictions
        def _run(_, inputs):
            X = list(inputs.values())[0]
            n = X.shape[0]
            return [np.array([[9.9]] * n)]
        self.session.run.side_effect = _run
        self.session.get_inputs.return_value = [MagicMock(name="input")]
        self._input_name = "input"

    monkeypatch.setattr(main_module.DurationPredictor, "load", _fake_load)

    # Build client and manually trigger fake load (startup event may not fire in TestClient without context manager)
    c = TestClient(main_module.app)
    try:
        main_module.predictor.load()
    except Exception:
        pass
    return c
