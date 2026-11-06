"""Validated ETA REST service. Run from the repository root with uvicorn."""
import logging
import math
import os
import time
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.exceptions import HTTPException

bundle = joblib.load(os.getenv("ETA_MODEL", "data/eta_v1.joblib"))
model = bundle["model"]
registry = CollectorRegistry()
requests_total = Counter("eta_requests_total", "HTTP requests excluding metrics", ["path", "status"], registry=registry)
latency = Histogram("eta_request_latency_seconds", "HTTP time excluding metrics", registry=registry,
                    buckets=(.001, .005, .01, .025, .05, .08, .1, .25, .5, 1))
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


class Delivery(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    distance_km: float = Field(ge=.1, le=500)
    n_stops: int = Field(ge=1, le=60, strict=True)
    hour: int = Field(ge=0, le=23, strict=True)
    weekday: int = Field(ge=0, le=6, strict=True)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1, strict=True)
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def feasible(self):
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery: distance > 150 km with > 20 stops")
        return self


class Batch(BaseModel):
    instances: list[Delivery] = Field(min_length=1, max_length=500)


@app.middleware("http")
async def trace_and_measure(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID", str(uuid4()))
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logging.exception("ETA request failed: %s", request.state.request_id)
        response = JSONResponse({"error": "internal_server_error", "request_id": request.state.request_id}, status_code=500)
    response.headers["X-Request-ID"] = request.state.request_id
    if request.url.path != "/metrics":
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        requests_total.labels(path, str(response.status_code)).inc()
        latency.observe(time.perf_counter() - started)
    return response


@app.exception_handler(RequestValidationError)
async def invalid(request: Request, exc: RequestValidationError):
    details = [{"field": ".".join(map(str, e["loc"][1:])), "msg": e["msg"]} for e in exc.errors()]
    return JSONResponse({"error": "validation_failed", "details": details,
                         "request_id": request.state.request_id}, status_code=422)


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return JSONResponse({"error": str(exc.detail), "request_id": request.state.request_id},
                        status_code=exc.status_code, headers=exc.headers)


@app.post("/v1/models/eta:predict")
def predict(batch: Batch, request: Request):
    frame = pd.DataFrame([row.model_dump() for row in batch.instances], columns=bundle["feature_columns"])
    predictions = [round(float(v), 1) for v in model.predict(frame)]
    if len(predictions) != len(frame) or not all(math.isfinite(v) for v in predictions):
        raise ValueError("invalid model output")
    return {"predictions": predictions, "model_version": bundle["version"],
            "request_id": request.state.request_id, "served_by": str(os.getpid())}


@app.get("/healthz")
def health(request: Request):
    return {"status": "ok", "model_version": bundle["version"], "request_id": request.state.request_id}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(registry), media_type="text/plain; version=0.0.4; charset=utf-8")
