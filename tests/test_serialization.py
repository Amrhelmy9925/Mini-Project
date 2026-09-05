import numpy as np, joblib, onnxruntime as rt
from prodml.features import get_features

def test_parity():
    dv, X_train_dv, X_test_dv, y_train, y_test = get_features()
    X = X_test_dv[:500].toarray().astype(np.float32) # 500 rows required
    models = joblib.load("models/models.joblib")
    model = models["rd"] # or ["lr"]
    pred_pkl = model.predict(X)

    sess = rt.InferenceSession("models/baseline.onnx")
    name = sess.get_inputs()[0].name
    pred_onnx = sess.run(None, {name: X})[0].ravel()

    assert np.allclose(pred_pkl, pred_onnx, atol=1e-4)

def test_benchmark():
    import time
    dv, X_train_dv, X_test_dv, _, _ = get_features()
    X = X_test_dv[:500].toarray().astype(np.float32)
    model = joblib.load("models/models.joblib")["rd"]
    sess = rt.InferenceSession("models/baseline.onnx")
    name = sess.get_inputs()[0].name

    def bench(fn, n=100):
        ts=[]
        for _ in range(n):
            t0=time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
        return float(np.mean(ts)), float(np.percentile(ts,95))

    mean_pkl, p95_pkl = bench(lambda: model.predict(X))
    mean_onnx, p95_onnx = bench(lambda: sess.run(None, {name: X}))
    print(f"pickle mean {mean_pkl*1000:.2f}ms p95 {p95_pkl*1000:.2f}ms")
    print(f"onnx   mean {mean_onnx*1000:.2f}ms p95 {p95_onnx*1000:.2f}ms")
    # copy 4 numbers to reports/module-1.md