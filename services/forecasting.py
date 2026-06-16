"""Forecasting service: data fetching, model training, evaluation."""

import asyncio
import logging
import os
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.pool
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from psycopg2.extras import RealDictCursor

from metrics import (
    EVALUATION_MAPE,
    FORECAST_DURATION,
    FORECAST_REQUESTS,
    MODEL_CACHE_EVENTS,
    MODEL_TRAIN_DURATION,
)
from models.prophet_model import (
    ProphetModelCache,
    create_prophet_model,
    fit_prophet,
    predict_future,
)

logger = logging.getLogger(__name__)

# Module tracer. trace.get_tracer() returns a ProxyTracer that is a no-op until
# setup_tracing() installs a real TracerProvider, so importing this at module load
# is safe and keeps tracing opt-in (no spans exported when no OTLP endpoint is set).
tracer = trace.get_tracer(__name__)

MIN_HISTORY_DAYS = int(os.environ.get("MIN_HISTORY_DAYS", "14"))
MODEL_CACHE_TTL_SECONDS = int(os.environ.get("MODEL_CACHE_TTL_SECONDS", "3600"))
MODEL_CACHE_MAX_SIZE = int(os.environ.get("MODEL_CACHE_MAX_SIZE", "10"))
MAPE_ALERT_THRESHOLD = float(os.environ.get("MAPE_ALERT_THRESHOLD", "30.0"))
EVALUATION_TRAIN_RATIO = float(os.environ.get("EVALUATION_TRAIN_RATIO", "0.8"))

DB_POOL_MIN = int(os.environ.get("DB_POOL_MIN", "2"))
DB_POOL_MAX = int(os.environ.get("DB_POOL_MAX", "10"))
DB_CONNECT_TIMEOUT = int(os.environ.get("DB_CONNECT_TIMEOUT", "10"))

# Lazy-initialized connection pool
_db_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def _get_db_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """Return (or create) the shared thread-safe connection pool."""
    global _db_pool
    if _db_pool is None:
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            raise RuntimeError("DATABASE_URL environment variable is not set")
        try:
            _db_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=DB_POOL_MIN,
                maxconn=DB_POOL_MAX,
                dsn=db_url,
                connect_timeout=DB_CONNECT_TIMEOUT,
            )
        except psycopg2.Error as exc:
            raise RuntimeError(f"Failed to create connection pool: {exc}") from exc
    return _db_pool


def _get_db_connection() -> psycopg2.extensions.connection:
    """Obtain a connection from the pool."""
    pool = _get_db_pool()
    try:
        conn = pool.getconn()
        # Set statement timeout (15s) at session level
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '15000'")
        return conn
    except psycopg2.Error as exc:
        raise RuntimeError(f"Failed to get pooled connection: {exc}") from exc


def _release_db_connection(conn: psycopg2.extensions.connection) -> None:
    """Return a connection to the pool."""
    pool = _get_db_pool()
    pool.putconn(conn)


_model_cache = ProphetModelCache(ttl_seconds=MODEL_CACHE_TTL_SECONDS, max_size=MODEL_CACHE_MAX_SIZE)


def _record_span_error(span: trace.Span, exc: Exception) -> None:
    """Attach exception details to the active span without changing app errors."""
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


def fetch_outbound_history(product_id: int) -> pd.DataFrame:
    """Fetch daily aggregated outbound quantities for a product."""
    # The psycopg2 instrumentation emits its own DB-query span, which nests under
    # this span automatically (this runs inside the active context, propagated
    # across asyncio.to_thread which copies contextvars).
    with tracer.start_as_current_span("ai.fetch_outbound_history") as span:
        span.set_attribute("product.id", product_id)
        query = """
            SELECT o.outbound_date AS ds, SUM(oi.quantity) AS y
            FROM outbounds o
            JOIN outbound_items oi ON o.id = oi.outbound_id
            WHERE oi.product_id = %s
              AND o.status = 'CONFIRMED'
            GROUP BY o.outbound_date
            ORDER BY o.outbound_date
        """
        conn = _get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (product_id,))
                rows = cur.fetchall()
        except psycopg2.Error as exc:
            raise RuntimeError(f"Database query failed: {exc}") from exc
        finally:
            _release_db_connection(conn)

        span.set_attribute("history.rows", len(rows))
        if not rows:
            return pd.DataFrame(columns=["ds", "y"])

        df = pd.DataFrame(rows)
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = df["y"].astype(float)
        return df


def _fill_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing dates with zero quantity to create a complete daily series."""
    if df.empty:
        return df.copy()
    full_range = pd.date_range(start=df["ds"].min(), end=df["ds"].max(), freq="D")
    df = df.set_index("ds").reindex(full_range, fill_value=0.0).reset_index()
    df = df.rename(columns={"index": "ds"})
    return df


async def train_or_load_model_async(product_id: int) -> Any:
    """Return a trained Prophet model for the given product, using cache if available.

    Runs the blocking Prophet fit() in a thread executor so the FastAPI event loop
    is not blocked during model training.
    """
    with tracer.start_as_current_span("ai.train_or_load_model") as span:
        span.set_attribute("product.id", product_id)

        cached = _model_cache.get(product_id)
        if cached is not None:
            span.set_attribute("model.cache_hit", True)
            MODEL_CACHE_EVENTS.labels(result="hit").inc()
            return cached
        span.set_attribute("model.cache_hit", False)
        MODEL_CACHE_EVENTS.labels(result="miss").inc()

        df = await asyncio.to_thread(fetch_outbound_history, product_id)
        df = _fill_missing_dates(df)
        span.set_attribute("history.days", len(df))

        if len(df) < MIN_HISTORY_DAYS:
            raise ValueError(
                f"Insufficient historical data for product {product_id}. "
                f"Found {len(df)} days, minimum required is {MIN_HISTORY_DAYS}."
            )

        model = create_prophet_model(yearly_seasonality=True)
        with tracer.start_as_current_span("ai.fit_prophet"), MODEL_TRAIN_DURATION.time():
            model = await asyncio.to_thread(fit_prophet, model, df)
        _model_cache.set(product_id, model)
        return model


def train_or_load_model(product_id: int) -> Any:
    """Synchronous wrapper for backward compatibility."""
    with tracer.start_as_current_span("ai.train_or_load_model") as span:
        span.set_attribute("product.id", product_id)

        cached = _model_cache.get(product_id)
        if cached is not None:
            span.set_attribute("model.cache_hit", True)
            MODEL_CACHE_EVENTS.labels(result="hit").inc()
            return cached
        span.set_attribute("model.cache_hit", False)
        MODEL_CACHE_EVENTS.labels(result="miss").inc()

        try:
            df = fetch_outbound_history(product_id)
            df = _fill_missing_dates(df)
            span.set_attribute("history.days", len(df))

            if len(df) < MIN_HISTORY_DAYS:
                raise ValueError(
                    f"Insufficient historical data for product {product_id}. "
                    f"Found {len(df)} days, minimum required is {MIN_HISTORY_DAYS}."
                )

            model = create_prophet_model(yearly_seasonality=True)
            with tracer.start_as_current_span("ai.fit_prophet"), MODEL_TRAIN_DURATION.time():
                model = fit_prophet(model, df)
            _model_cache.set(product_id, model)
            return model
        except Exception as exc:
            _record_span_error(span, exc)
            raise


async def forecast_async(product_id: int, days: int) -> dict[str, Any]:
    """Async variant of forecast() that does not block the event loop."""
    if days <= 0:
        raise ValueError("days must be a positive integer")

    with tracer.start_as_current_span("ai.forecast") as span, FORECAST_DURATION.time():
        span.set_attribute("product.id", product_id)
        span.set_attribute("forecast.days", days)
        try:
            model = await train_or_load_model_async(product_id)
            with tracer.start_as_current_span("ai.predict_future"):
                forecast_df = await asyncio.to_thread(predict_future, model, days)

            future_df = forecast_df.tail(days)
            records = []
            for _, row in future_df.iterrows():
                records.append({
                    "ds": row["ds"].strftime("%Y-%m-%d"),
                    "yhat": round(float(row["yhat"]), 2),
                    "yhat_lower": round(float(row["yhat_lower"]), 2),
                    "yhat_upper": round(float(row["yhat_upper"]), 2),
                })
        except Exception as exc:
            FORECAST_REQUESTS.labels(outcome="error").inc()
            _record_span_error(span, exc)
            raise

        FORECAST_REQUESTS.labels(outcome="success").inc()
        span.set_attribute("forecast.points", len(records))
        return {
            "product_id": product_id,
            "days": days,
            "forecast": records,
        }


def forecast(product_id: int, days: int) -> dict[str, Any]:
    """Generate a demand forecast for a product."""
    if days <= 0:
        raise ValueError("days must be a positive integer")

    with tracer.start_as_current_span("ai.forecast") as span, FORECAST_DURATION.time():
        span.set_attribute("product.id", product_id)
        span.set_attribute("forecast.days", days)
        try:
            model = train_or_load_model(product_id)
            with tracer.start_as_current_span("ai.predict_future"):
                forecast_df = predict_future(model, days)

            future_df = forecast_df.tail(days)
            records = []
            for _, row in future_df.iterrows():
                records.append({
                    "ds": row["ds"].strftime("%Y-%m-%d"),
                    "yhat": round(float(row["yhat"]), 2),
                    "yhat_lower": round(float(row["yhat_lower"]), 2),
                    "yhat_upper": round(float(row["yhat_upper"]), 2),
                })
        except Exception as exc:
            FORECAST_REQUESTS.labels(outcome="error").inc()
            _record_span_error(span, exc)
            raise

        FORECAST_REQUESTS.labels(outcome="success").inc()
        span.set_attribute("forecast.points", len(records))
        return {
            "product_id": product_id,
            "days": days,
            "forecast": records,
        }


def _store_evaluation(product_id: int, mae: float, rmse: float, mape: float, model_version: str) -> None:
    """Persist evaluation metrics into the analytics.ai_model_evaluations table."""
    with tracer.start_as_current_span("ai.store_evaluation") as span:
        span.set_attribute("product.id", product_id)
        span.set_attribute("model.version", model_version)
        span.set_attribute("evaluation.mape", mape)
        query = """
            INSERT INTO analytics.ai_model_evaluations (product_id, mae, rmse, mape, model_version)
            VALUES (%s, %s, %s, %s, %s)
        """
        conn = _get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(query, (product_id, mae, rmse, mape, model_version))
                conn.commit()
        except psycopg2.Error as exc:
            wrapped = RuntimeError(f"Failed to store evaluation result: {exc}")
            _record_span_error(span, wrapped)
            raise wrapped from exc
        finally:
            _release_db_connection(conn)


def fetch_evaluation_history(product_id: int, limit: int = 20) -> list[dict[str, Any]]:
    """Fetch past evaluation results for a product, ordered by most recent first."""
    with tracer.start_as_current_span("ai.fetch_evaluation_history") as span:
        span.set_attribute("product.id", product_id)
        span.set_attribute("history.limit", limit)
        query = """
            SELECT id, mae, rmse, mape, evaluated_at, model_version
            FROM analytics.ai_model_evaluations
            WHERE product_id = %s
            ORDER BY evaluated_at DESC
            LIMIT %s
        """
        conn = _get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (product_id, limit))
                rows = cur.fetchall()
        except psycopg2.Error as exc:
            wrapped = RuntimeError(f"Database query failed: {exc}")
            _record_span_error(span, wrapped)
            raise wrapped from exc
        finally:
            _release_db_connection(conn)
        span.set_attribute("history.rows", len(rows))
        return [dict(row) for row in rows]


async def evaluate_async(product_id: int, model_version: str = "prophet") -> dict[str, Any]:
    """Async variant of evaluate() that does not block the event loop."""
    with tracer.start_as_current_span("ai.evaluate") as span:
        span.set_attribute("product.id", product_id)
        span.set_attribute("model.version", model_version)

        try:
            df = await asyncio.to_thread(fetch_outbound_history, product_id)
            df = _fill_missing_dates(df)

            if len(df) < MIN_HISTORY_DAYS * 2:
                raise ValueError(
                    f"Insufficient data for evaluation for product {product_id}. "
                    f"Found {len(df)} days, minimum required is {MIN_HISTORY_DAYS * 2}."
                )

            split_idx = int(len(df) * EVALUATION_TRAIN_RATIO)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]

            model = create_prophet_model(yearly_seasonality=True)
            with tracer.start_as_current_span("ai.fit_prophet"), MODEL_TRAIN_DURATION.time():
                await asyncio.to_thread(model.fit, train_df)

            with tracer.start_as_current_span("ai.predict"):
                future = model.make_future_dataframe(periods=len(test_df))
                forecast_df = model.predict(future)
            pred_df = forecast_df.tail(len(test_df)).reset_index(drop=True)
            actuals = test_df["y"].values
            predictions = pred_df["yhat"].values

            mae = float(np.mean(np.abs(actuals - predictions)))
            rmse = float(np.sqrt(np.mean((actuals - predictions) ** 2)))
            mape = float(np.mean(np.abs((actuals - predictions) / np.where(actuals == 0, 1, actuals))) * 100)
        except Exception as exc:
            _record_span_error(span, exc)
            raise

        EVALUATION_MAPE.observe(mape)
        span.set_attribute("evaluation.mape", round(mape, 4))

        result = {
            "product_id": product_id,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 4),
            "model_version": model_version,
        }

        _store_evaluation(product_id, result["mae"], result["rmse"], result["mape"], model_version)

        if mape > MAPE_ALERT_THRESHOLD:
            span.set_attribute("evaluation.alert", True)
            logger.warning(
                "AI model evaluation alert: product_id=%s MAPE=%.2f%% exceeds %.1f%% threshold (model=%s)",
                product_id, mape, MAPE_ALERT_THRESHOLD, model_version,
            )

        return result


def evaluate(product_id: int, model_version: str = "prophet") -> dict[str, Any]:
    """Evaluate forecast accuracy using cross-validation on historical data."""
    with tracer.start_as_current_span("ai.evaluate") as span:
        span.set_attribute("product.id", product_id)
        span.set_attribute("model.version", model_version)

        try:
            df = fetch_outbound_history(product_id)
            df = _fill_missing_dates(df)
            span.set_attribute("history.days", len(df))

            if len(df) < MIN_HISTORY_DAYS * 2:
                raise ValueError(
                    f"Insufficient data for evaluation for product {product_id}. "
                    f"Found {len(df)} days, minimum required is {MIN_HISTORY_DAYS * 2}."
                )

            split_idx = int(len(df) * EVALUATION_TRAIN_RATIO)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]

            model = create_prophet_model(yearly_seasonality=True)
            with tracer.start_as_current_span("ai.fit_prophet"), MODEL_TRAIN_DURATION.time():
                model.fit(train_df)

            with tracer.start_as_current_span("ai.predict"):
                future = model.make_future_dataframe(periods=len(test_df))
                forecast_df = model.predict(future)
            pred_df = forecast_df.tail(len(test_df)).reset_index(drop=True)
            actuals = test_df["y"].values
            predictions = pred_df["yhat"].values

            mae = float(np.mean(np.abs(actuals - predictions)))
            rmse = float(np.sqrt(np.mean((actuals - predictions) ** 2)))
            mape = float(np.mean(np.abs((actuals - predictions) / np.where(actuals == 0, 1, actuals))) * 100)
        except Exception as exc:
            _record_span_error(span, exc)
            raise

        EVALUATION_MAPE.observe(mape)
        span.set_attribute("evaluation.mape", round(mape, 4))

        result = {
            "product_id": product_id,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 4),
            "model_version": model_version,
        }

        _store_evaluation(product_id, result["mae"], result["rmse"], result["mape"], model_version)

        if mape > MAPE_ALERT_THRESHOLD:
            span.set_attribute("evaluation.alert", True)
            logger.warning(
                "AI model evaluation alert: product_id=%s MAPE=%.2f%% exceeds %.1f%% threshold (model=%s)",
                product_id, mape, MAPE_ALERT_THRESHOLD, model_version,
            )

        return result
