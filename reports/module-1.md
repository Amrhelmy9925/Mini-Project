# Module 1 — From notebook to production-ready service

## Validation scores (top of report, per Step 01)

- Before (notebook `notebook/00-baseline.ipynb`): reported "MAE" **55.97 minutes**
  - HONEST NOTE: that cell runs `mae = mean_squared_error(...)`, so 55.97 is
    actually **MSE mislabeled as MAE**. Implied RMSE ≈ √55.97 ≈ 7.48.
  - Consequence: the ±0.05 parity acceptance check is **not directly
    comparable** — different metric and likely different cleaning/filters.
- After (package `python -m prodml.train` equivalent path, Ridge `rd`, n=700956):
  - MAE **4.1523** (LinearRegression `lr`: 4.1684), RMSE **7.1963**

## Serialization: pickle vs ONNX (Step 04)

Parity (500 validation rows, `np.allclose(atol=1e-4)`): **True**

| Engine | Mean latency | p95 latency |
|--------|-------------|-------------|
| Pickle (`models.joblib["rd"]`) | 0.20 ms | 0.27 ms |
| ONNX (`baseline.onnx` via onnxruntime) | 0.06 ms | 0.11 ms |

ONNX is ~3× faster on identical 500-row batches.

### Format comparison

| Format | Human-readable | Cross-language | Schema-enforced | Safe to load untrusted |
|--------|---------------|----------------|-----------------|------------------------|
| JSON | Yes | Yes | Only with external schema | Yes (data only) |
| Protobuf | No | Yes | Yes | Yes (typed parse) |
| Pickle | No | No (Python only) | No | **No** |
| ONNX | No | Yes | Yes (graph + types) | Yes (no code exec) |

**Pickle executes arbitrary code on load. Never load a `.pkl` you did not
produce.**

**We serve ONNX**, because parity holds, inference is ~3× faster, and the
graph is portable and safe to load, while pickle stays only as the training
artifact.

## Containerization (Step 07)

| Image | Size |
|-------|------|
| Single-stage (`docker/Dockerfile.single`, temp file, deleted after measure) | 1.43 GB (`prodml-api:single`) |
| Multi-stage (`docker/Dockerfile`) | 1.49 GB (`prodml-api:multi`) |

Docker Hub image URL: <!-- TODO(you): paste, e.g. https://hub.docker.com/r/<you>/prodml-api -->

Non-root check: `docker exec <id> whoami` → `appuser`.

## Maturity self-assessment (Step 08, Lesson 1 five-level model)

We place this repo at **level 2 (automated pipeline, versioned artifacts)**:
packaged code, tested API, versioned image — but no CI enforcement, no
experiment tracking in the loop, no monitoring/retraining.

To reach level 3 we need Module 2: CI gates (tests + lint + image build) on
every PR plus tracked experiments (MLflow) backing each published artifact.

## Test evidence

- `pytest -v`: 22+ tests pass (features edge cases via `@parametrize`,
  predictor determinism, API happy/422/schema, serialization parity,
  train/export round-trip with `tmp_path` + `monkeypatch` mocks).
- Coverage: **92%** (gate `--cov-fail-under=70`).
- `print()` in `src/`: none (grep verified).
- Endpoints: `/health`, `/metadata`, `/predict`, `/predict/batch` respond;
  `trip_distance: -5` → `422`, not a stack trace.
- Model loads once at startup (`lifespan` + `/health` checks `session`),
  never per request.
