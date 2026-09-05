from pydantic import BaseModel, Field

MODEL_VERSION = "0.1.0"
# ---- Block 4: menus ----
class RideFeatures(BaseModel):
    """One ride. trip_distance is required, rest optional."""

    trip_distance: float = Field(
        gt=0, lt=200, description="trip distance in miles"
    )
    VendorID: str | None = None
    passenger_count: float | None = None
    RatecodeID: str | None = None
    store_and_fwd_flag: str | None = None
    PULocationID: str | None = None
    DOLocationID: str | None = None
    payment_type: str | None = None
    fare_amount: float | None = None
    extra: float | None = None
    mta_tax: float | None = None
    tip_amount: float | None = None
    tolls_amount: float | None = None
    improvement_surcharge: float | None = None
    total_amount: float | None = None
    congestion_surcharge: float | None = None
    Airport_fee: float | None = None
    cbd_congestion_fee: float | None = None

    model_config = {
        "extra": "allow",  # ignore unknown columns, don't crash
        "json_schema_extra": {
            "example": {"trip_distance": 2.5, "PULocationID": "100", "DOLocationID": "200"}
        },
    }
    
# class PredictRequest(BaseModel):
#     trip_distance:float = Field(gt=0,lt=200)
#     PULocationID:str |None =None
#     DOLocationID:str |None =None

class PredictResponse(BaseModel):
    prediction: float
    latency_ms: float
    model_version: str = MODEL_VERSION
    correlation_id: str = "-"


class MetadataResponse(BaseModel):
    model_version: str
    framework: str
    feature_names: list[str]
    artifact_hash: str    


class BatchRequest(BaseModel):
    rides: list[RideFeatures]


class BatchResponse(BaseModel):
    predictions: list[float]
    latency_ms: float
    model_version: str = MODEL_VERSION    