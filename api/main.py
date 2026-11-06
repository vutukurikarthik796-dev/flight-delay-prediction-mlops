from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from api.schemas import FlightInput, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "flight_delay_final_model.pkl"
FEATURES = [
    "MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE", "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT", "SCHEDULED_TIME", "DISTANCE", "DEP_HOUR",
    "DEP_MINUTE", "DEP_PERIOD",
]
model = None


def _feature_row(flight: FlightInput) -> dict:
    data = flight.model_dump(exclude={"scheduled_departure"})
    hour, minute = divmod(flight.scheduled_departure, 100)
    if hour < 6:
        period = "Night"
    elif hour < 12:
        period = "Morning"
    elif hour < 18:
        period = "Afternoon"
    else:
        period = "Evening"
    data["DEP_HOUR"] = hour
    data["DEP_MINUTE"] = minute
    data["DEP_PERIOD"] = period
    return {name: data[name] for name in FEATURES}


def _predict_one(flight: FlightInput) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Prediction model is not loaded")
    frame = pd.DataFrame([_feature_row(flight)], columns=FEATURES)
    try:
        prediction = int(model.predict(frame)[0])
        probability = float(model.predict_proba(frame)[0][1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {type(exc).__name__}") from exc
    return PredictionResponse(
        prediction=prediction,
        prediction_label="FLIGHT DELAYED" if prediction == 1 else "FLIGHT NOT DELAYED",
        delay_probability=probability,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"[SUCCESS] Loaded flight-delay model from {MODEL_PATH}")
    yield
    print("[INFO] Flight Delay Prediction API shutting down")


app = FastAPI(
    title="Flight Delay Prediction API",
    description="Predicts whether a flight's arrival delay will exceed 15 minutes using pre-departure features.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"service": "Flight Delay Prediction API", "status": "running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy" if model is not None else "not_ready", "model_loaded": model is not None}


@app.get("/metadata")
def metadata():
    return {
        "service": "Flight Delay Prediction API",
        "version": "1.0.0",
        "model_file": MODEL_PATH.name,
        "model_loaded": model is not None,
        "target": "DELAYED",
        "target_definition": "1 if ARRIVAL_DELAY > 15 minutes, otherwise 0",
        "input_features": FEATURES,
        "scheduled_departure_format": "HHMM integer, e.g. 1830",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(flight: FlightInput):
    return _predict_one(flight)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest):
    results = [_predict_one(flight) for flight in payload.flights]
    return BatchPredictionResponse(count=len(results), predictions=results)
