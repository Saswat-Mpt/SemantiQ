from __future__ import annotations

import os
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference import SemantIQ


app = FastAPI(
    title="SemantIQ Inference API",
    description="Cost-Aware Semantic Deduplication and Verification API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine singleton
_engine: SemantIQ | None = None


def get_engine() -> SemantIQ:
    global _engine
    if _engine is None:
        _engine = SemantIQ()
    return _engine


# ============================================================
# Schemas
# ============================================================

class QuestionPairRequest(BaseModel):
    question1: str = Field(..., min_length=1, example="How can I learn Python for data science?")
    question2: str = Field(..., min_length=1, example="What is the best way to study Python for data analysis?")


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
    architecture: str
    num_features: int
    features: list[str]
    threshold_policy: dict[str, Any]
    status: str


# ============================================================
# Endpoints
# ============================================================

@app.on_event("startup")
def startup_event():
    get_engine()


@app.get("/health", tags=["Health"])
def health_check():
    """Service health and model readiness probe."""
    engine = get_engine()
    return {
        "status": "healthy",
        "engine_ready": engine is not None,
        "active_model": "XGBoost_Experiment_E",
    }


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """Retrieve metadata about the active model and decision policy."""
    engine = get_engine()
    return {
        "model_name": "SemantIQ",
        "architecture": "19-Feature Hybrid Fusion + XGBoost",
        "num_features": len(engine.feature_columns),
        "features": engine.feature_columns,
        "threshold_policy": {
            "high_precision_T_star": engine.high_precision_threshold,
            "default_T": engine.default_threshold,
            "decision_tiers": {
                "DUPLICATE": f"score >= {engine.high_precision_threshold:.4f} (High confidence, >=90% val precision target)",
                "NEEDS_REVIEW": f"{engine.default_threshold:.2f} <= score < {engine.high_precision_threshold:.4f} (Uncertain zone)",
                "DISTINCT": f"score < {engine.default_threshold:.2f} (Non-duplicate)",
            },
        },
        "status": "online",
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: QuestionPairRequest):
    """Predict semantic duplicate probability and 3-tier decision for a single question pair."""
    try:
        engine = get_engine()
        result = engine.predict_pair(request.question1, request.question2)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/batch-predict", response_model=list[PredictResponse], tags=["Inference"])
def batch_predict(request: BatchPairRequest):
    """Batch prediction for multiple question pairs."""
    try:
        engine = get_engine()
        pairs = [(p.question1, p.question2) for p in request.pairs]
        return engine.predict_batch(pairs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")
