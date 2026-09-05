"""Taxi duration API - rebuilt simple so you own every line.

Cookie shop version:
- Block 1: tools (imports)
- Block 2: turn on JSON logs
- Block 3: one robot (predictor) shared by all requests
- Block 4: menus (Pydantic schemas = what door accepts/returns)
- Block 5: open shop (lifespan loads model ONCE at startup, not per request)
- Block 6: name tag per visitor (middleware correlation_id)
- Block 7: polite errors (422 vs 500)
- Block 8: doors (/health, /metadata, /predict, /predict/batch)
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from prodml.api.logging_conf import correlation_id_var, setup_logging
from prodml.config import settings
from prodml.predict import DurationPredictor

from .schemas import (PredictResponse,MetadataResponse,MODEL_VERSION,RideFeatures,BatchRequest,BatchResponse)

# ---- Block 2: logs must start before any logger.info ----
setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)



# ---- Block 3: one robot for whole shop ----
predictor = DurationPredictor()




# ---- helpers (no decorator = not a door) ----
def _clean(features: dict) -> dict:
    """Make API dict look like training dict: strings stay strings, None -> 0."""
    for col in ["VendorID", "RatecodeID", "store_and_fwd_flag", "PULocationID", "DOLocationID", "payment_type"]:
        if features.get(col) is not None:
            features[col] = str(features[col])
    return {k: (0 if v is None else v) for k, v in features.items()}


def _hash(path: str) -> str:
    """First 16 chars of sha256 of model file = ID card number."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except Exception:
        return "missing"


# ---- Block 5: open shop - load ONCE ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        predictor.load()
        logger.info("model loaded", extra={"model_path": settings.model_path})
    except Exception as exc:
        logger.error("model load failure", exc_info=True, extra={"error": str(exc)})
    yield  # shop stays open here
    # no cleanup needed on close


app = FastAPI(title="prodml - taxi duration", lifespan=lifespan)


# keep old startup hook too so TestClient without lifespan still loads
@app.on_event("startup")
def _load_model() -> None:
    try:
        if predictor.session is None:
            predictor.load()
    except Exception:
        pass


# ---- Block 6: name tag per visitor ----
@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    cid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    correlation_id_var.set(cid)
    start = time.perf_counter()
    logger.info("request started", extra={"method": request.method, "path": request.url.path})
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error("unhandled exception", exc_info=True, extra={"error": str(exc)})
        raise
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = cid
    logger.info(
        "request completed",
        extra={"method": request.method, "path": request.url.path,
               "status_code": response.status_code, "latency_ms": latency_ms},
    )
    return response


# ---- Block 7: polite errors ----
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    cid = correlation_id_var.get()
    logger.error("validation rejection", extra={"error": str(exc.errors())})
    return JSONResponse(status_code=422, content={"detail": exc.errors()},
                        headers={"X-Request-ID": cid})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    cid = correlation_id_var.get()
    logger.error("request failed", exc_info=True, extra={"error": str(exc)})
    return JSONResponse(status_code=500, content={"detail": "internal server error"},
                        headers={"X-Request-ID": cid})


# ---- Block 8: doors ----
@app.get("/health")
def health():
    loaded = predictor.session is not None
    return {"status": "ok" if loaded else "model_not_loaded","version":MODEL_VERSION}


@app.get("/metadata", response_model=MetadataResponse)
def metadata():
    names: list[str] = []
    try:
        if predictor.dv is not None and hasattr(predictor.dv, "get_feature_names_out"):
            names = list(predictor.dv.get_feature_names_out())
    except Exception:
        names = []
    return MetadataResponse(
        model_version=MODEL_VERSION,
        framework="sklearn+onnxruntime",
        feature_names=names,
        artifact_hash=_hash(settings.model_path),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(body: RideFeatures):
    features = body.model_dump(exclude_none=False)
    logger.debug("feature vector", extra={"feature_vector": features})

    td = float(features.get("trip_distance") or 0)
    if td > 100:
        logger.warning("input outside the training range (trip_distance > 100)",
                       extra={"trip_distance": td})

    start = time.perf_counter()
    try:
        pred = predictor.predict_one(_clean(features))
    except Exception as exc:
        logger.error("prediction failed", exc_info=True, extra={"error": str(exc)})
        raise
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info("prediction served",
                extra={"prediction": float(pred), "latency_ms": latency_ms})
    return PredictResponse(prediction=float(pred), latency_ms=latency_ms,
                           model_version=MODEL_VERSION,
                           correlation_id=correlation_id_var.get())


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(body: BatchRequest):
    start = time.perf_counter()
    rows = [_clean(r.model_dump(exclude_none=False)) for r in body.rides]
    preds = predictor.predict_batch(rows)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return BatchResponse(predictions=[float(p) for p in preds],
                         latency_ms=latency_ms, model_version=MODEL_VERSION)
