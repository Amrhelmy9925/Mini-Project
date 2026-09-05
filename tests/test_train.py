"""Taste-test the bakery (train.py + export.py).

6yo version: bakery mixes tiny dough (5 rows), bakes 2 cakes (lr, rd),
wraps them (joblib + ONNX). We use tmp_path so we never overwrite real models/.
"""

import numpy as np
import scipy.sparse as sp


def _tiny_matrices():
    from sklearn.feature_extraction import DictVectorizer

    dicts = [{"a": float(i), "b": "x" if i % 2 == 0 else "y"} for i in range(10)]
    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(dicts)
    y = np.array([float(i) + 1.0 for i in range(10)], dtype=np.float32)
    X_train, X_test = X[:8], X[8:]
    y_train, y_test = y[:8], y[8:]
    return dv, X_train, X_test, y_train, y_test


def test_train_models_returns_dict():
    from prodml.train import train_models

    dv, X_train, X_test, y_train, y_test = _tiny_matrices()
    models = train_models(X_train, X_test, y_train, y_test)
    assert set(models.keys()) == {"lr", "rd"}
    preds = models["rd"].predict(X_test)
    assert len(preds) == 2


def test_save_artifacts_tmp(monkeypatch, tmp_path):
    import joblib

    from prodml import config as config_module
    from prodml.train import save_artifacts

    dv, X_train, X_test, y_train, y_test = _tiny_matrices()
    from prodml.train import train_models

    models = train_models(X_train, X_test, y_train, y_test)

    # Redirect all 3 artifact paths into tmp dir so real models/ untouched
    monkeypatch.setattr(config_module.settings, "model_path", str(tmp_path / "model.onnx"))
    monkeypatch.setattr(config_module.settings, "vectorizer_path", str(tmp_path / "dv.joblib"))
    monkeypatch.setattr(config_module.settings, "models_path", str(tmp_path / "models.joblib"))

    # export.py reads settings.model_path at call time via default arg?
    # save_artifacts reads settings.* inside function, so monkeypatch works.
    save_artifacts(dv, models, X_train)

    assert (tmp_path / "model.onnx").exists()
    assert (tmp_path / "dv.joblib").exists()
    assert (tmp_path / "models.joblib").exists()
    loaded = joblib.load(tmp_path / "models.joblib")
    assert "rd" in loaded


def test_export_onnx_tmp(monkeypatch, tmp_path):
    from prodml import config as config_module
    from prodml.export import export_onnx, parity_and_benchmark

    dv, X_train, X_test, y_train, y_test = _tiny_matrices()
    from prodml.train import train_models

    models = train_models(X_train, X_test, y_train, y_test)
    out = str(tmp_path / "tiny.onnx")

    # export_onnx has out=settings.model_path as default — pass explicit out
    res = export_onnx(models["rd"], X_train, out=out)
    assert res == out

    # parity_and_benchmark reads settings.model_path — point it at our tmp file
    monkeypatch.setattr(config_module.settings, "model_path", out)
    (mean_pkl, p95_pkl), (mean_onnx, p95_onnx) = parity_and_benchmark(models["rd"], X_test)
    assert mean_pkl > 0 and mean_onnx > 0
