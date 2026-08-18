from __future__ import annotations

import os
import time
import uuid
from typing import Any
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference import SemantIQ


# ============================================================
# Application Initialization & Hardening
# ============================================================

app = FastAPI(
    title="SemantIQ Inference API",
    description="Cost-Aware Semantic Deduplication and Verification API",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000")
allowed_origins = [orig.strip() for orig in allowed_origins_env.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global engine singleton
_engine: SemantIQ | None = None


def get_engine() -> SemantIQ:
    global _engine
    if _engine is None:
        _engine = SemantIQ()
    return _engine


@app.middleware("http")
async def add_request_metadata(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    
    response: Response = await call_next(request)
    
    process_time = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-MS"] = f"{process_time:.2f}"
    return response


# ============================================================
# Schemas
# ============================================================

class QuestionPairRequest(BaseModel):
    question1: str = Field(..., min_length=1, max_length=1000, example="How can I learn Python for data science?")
    question2: str = Field(..., min_length=1, max_length=1000, example="What is the best way to study Python for data analysis?")


class BatchPairRequest(BaseModel):
    pairs: list[QuestionPairRequest] = Field(..., min_items=1, max_items=100)


class PredictResponse(BaseModel):
    question1: str
    question2: str
    score: float
    decision: str
    confidence: str
    thresholds: dict[str, float]
    contradiction_warning: bool
    critical_tokens: dict[str, Any]
    evidence: dict[str, float]
    latency_ms: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    architecture: str
    num_features: int
    features: list[str]
    threshold_policy: dict[str, Any]
    calibration: dict[str, Any]
    status: str


# ============================================================
# Health & Readiness Probes
# ============================================================

@app.get("/live", tags=["Health"])
def liveness_probe():
    """Kubernetes / Docker liveness probe."""
    return {"status": "alive"}


@app.get("/ready", tags=["Health"])
def readiness_probe():
    """Kubernetes / Docker readiness probe ensuring model is in memory."""
    engine = get_engine()
    return {
        "status": "ready",
        "model_loaded": engine is not None,
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Legacy service health check."""
    return readiness_probe()


# ============================================================
# Model Metadata & Versioning
# ============================================================

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
@app.get("/api/v1/model-info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """Retrieve metadata about active model, features, and decision policy."""
    engine = get_engine()
    return {
        "model_name": "SemantIQ",
        "model_version": "1.1.0",
        "architecture": "19-Feature Hybrid Fusion + Contradiction Verifier + XGBoost",
        "num_features": len(engine.feature_columns),
        "features": engine.feature_columns,
        "threshold_policy": {
            "high_precision_T_star": engine.high_precision_threshold,
            "default_T": engine.default_threshold,
            "decision_tiers": {
                "DUPLICATE": f"score >= {engine.high_precision_threshold:.4f} (High confidence, target >=90% val precision)",
                "NEEDS_REVIEW": f"{engine.default_threshold:.2f} <= score < {engine.high_precision_threshold:.4f} (Uncertain zone)",
                "DISTINCT": f"score < {engine.default_threshold:.2f} (Non-duplicate)",
            },
        },
        "calibration": {
            "expected_calibration_error": 0.01202,
            "brier_score": 0.1172,
        },
        "status": "online",
    }


# ============================================================
# Versioned Inference Endpoints
# ============================================================

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
@app.post("/api/v1/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: QuestionPairRequest):
    """Predict semantic duplicate probability and 3-tier decision for a single question pair."""
    try:
        engine = get_engine()
        return engine.predict_pair(request.question1, request.question2)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/batch-predict", response_model=list[PredictResponse], tags=["Inference"])
@app.post("/api/v1/batch-predict", response_model=list[PredictResponse], tags=["Inference"])
def batch_predict(request: BatchPairRequest):
    """High-throughput vectorized batch prediction for up to 100 question pairs."""
    try:
        engine = get_engine()
        pairs = [(p.question1, p.question2) for p in request.pairs]
        return engine.predict_batch(pairs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")
