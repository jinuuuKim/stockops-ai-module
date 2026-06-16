"""FastAPI application for demand forecasting with Facebook Prophet."""

import os
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Path, Request, status
from fastapi.security import APIKeyHeader
from opentelemetry import trace
from pydantic import BaseModel, Field

from metrics import setup_metrics
from services.forecasting import (
    evaluate_async,
    fetch_evaluation_history,
    forecast_async,
)
from telemetry import setup_tracing

load_dotenv()

app = FastAPI(title="StockOps AI Forecasting Service", version="1.0.0")

# Enable OpenTelemetry distributed tracing (no-op unless OTEL_EXPORTER_OTLP_ENDPOINT
# is set). Must run after the app is created and before requests are served.
setup_tracing(app)

# Expose Prometheus metrics at /metrics (pull model; always on, mirroring the
# api-server's /actuator/prometheus). The metrics half of the observability split.
setup_metrics(app)
tracer = trace.get_tracer(__name__)

# --- API Key Authentication ---
AI_MODULE_API_KEY = os.environ.get("AI_MODULE_API_KEY", "")
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Depends(API_KEY_HEADER)) -> None:
    """Validate the X-API-Key header. Raises 401 if missing or invalid."""
    if not AI_MODULE_API_KEY:
        # If no key is configured, allow all requests (dev mode)
        return
    if api_key is None or api_key != AI_MODULE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


# --- Request / Response Models ---

class PredictRequest(BaseModel):
    """Request body for single-product demand forecasting."""

    product_id: int = Field(..., ge=1, description="Product identifier")
    days: int = Field(..., ge=1, le=365, description="Number of days to forecast")


class PredictResponse(BaseModel):
    """Response body for single-product demand forecasting."""

    product_id: int
    days: int
    forecast: list[dict[str, Any]]


class BulkPredictRequest(BaseModel):
    """Request body for bulk demand forecasting."""

    products: list[PredictRequest] = Field(..., min_length=1, max_length=50)


class EvaluateResponse(BaseModel):
    """Response body for forecast evaluation metrics."""

    product_id: int
    mae: float
    rmse: float
    mape: float
    model_version: str = "prophet"


class EvaluateHistoryResponse(BaseModel):
    """Response body for evaluation history."""

    product_id: int
    history: list[dict[str, Any]]


# --- Public Endpoints (no auth) ---

@app.get("/health", response_model=dict[str, str])
def health_check() -> dict[str, str]:
    """Return the health status of the service."""
    return {"status": "ok"}


# --- Protected Endpoints (API Key required) ---

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, _=Depends(verify_api_key)) -> PredictResponse:
    """Generate a demand forecast for a single product."""
    try:
        result = await forecast_async(request.product_id, request.days)
        return PredictResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@app.post("/predict/bulk", response_model=list[PredictResponse])
async def predict_bulk(request: BulkPredictRequest, _=Depends(verify_api_key)) -> list[PredictResponse]:
    """Generate demand forecasts for multiple products in one request."""
    with tracer.start_as_current_span("ai.predict_bulk") as span:
        span.set_attribute("bulk.product_count", len(request.products))
        results: list[PredictResponse] = []
        errors: list[str] = []

        for item in request.products:
            try:
                result = await forecast_async(item.product_id, item.days)
                results.append(PredictResponse(**result))
            except ValueError as exc:
                errors.append(f"product_id={item.product_id}: {exc}")
            except RuntimeError as exc:
                errors.append(f"product_id={item.product_id}: {exc}")

        span.set_attribute("bulk.success_count", len(results))
        span.set_attribute("bulk.error_count", len(errors))
        if errors:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"errors": errors, "successful": len(results)},
            )

        return results


@app.get("/evaluate/{product_id}", response_model=EvaluateResponse)
async def evaluate_model(
    product_id: int = Path(..., ge=1),
    model_version: str = "prophet",
    _=Depends(verify_api_key),
) -> EvaluateResponse:
    """Evaluate forecast accuracy for a product using historical data."""
    try:
        result = await evaluate_async(product_id, model_version)
        return EvaluateResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@app.get("/evaluate/history/{product_id}", response_model=EvaluateHistoryResponse)
async def evaluation_history(
    product_id: int = Path(..., ge=1),
    limit: int = 20,
    _=Depends(verify_api_key),
) -> EvaluateHistoryResponse:
    """Fetch past evaluation results for a product."""
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="limit must be between 1 and 100",
        )
    try:
        history = fetch_evaluation_history(product_id, limit)
        return EvaluateHistoryResponse(product_id=product_id, history=history)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
