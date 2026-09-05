from pathlib import Path
import time, numpy as np, joblib, onnxruntime as rt
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from .config import settings

def export_onnx(model, X_sample, out=settings.model_path, opset=settings.onnx_opset):
    # [None, n_features] -> dynamic batch
    initial_type = [("input", FloatTensorType([None, X_sample.shape[1]]))]
    onx = to_onnx(model, X_sample[:1].toarray().astype(np.float32), target_opset=opset, initial_types=initial_type)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(onx.SerializeToString())
    return out

def parity_and_benchmark(model, X_val_500):
    X = X_val_500.toarray().astype(np.float32)
    pred_pkl = model.predict(X)
    sess = rt.InferenceSession(settings.model_path)
    name = sess.get_inputs()[0].name
    pred_onnx = sess.run(None, {name: X})[0].ravel()
    assert np.allclose(pred_pkl, pred_onnx, atol=1e-4)
    # benchmark mean/p95
    def _bench(fn, n=100):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return float(np.mean(ts)), float(np.percentile(ts, 95))

    mean_pkl, p95_pkl = _bench(lambda: model.predict(X))
    mean_onnx, p95_onnx = _bench(lambda: sess.run(None, {name: X}))
    return (mean_pkl, p95_pkl), (mean_onnx, p95_onnx)