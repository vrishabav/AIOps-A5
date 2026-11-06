"""Stable customer bucketing and a fractional-CPU Ray Serve deployment graph."""
import hashlib
import math
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Header
from pydantic import BaseModel, Field
from ray import serve

ROOT = Path(__file__).resolve().parents[1]
api = FastAPI()


def bucket(customer_id):
    return int.from_bytes(hashlib.sha256(customer_id.encode()).digest()[:8], "big") / 2**64


class CanaryRequest(BaseModel):
    customer_id: str = Field(min_length=1)
    instances: list[dict] = Field(min_length=1, max_length=500)


@serve.deployment(ray_actor_options={"num_cpus": .25}, max_ongoing_requests=4)
class EtaModel:
    def __init__(self, path, extra_work_ms=0.0):
        self.bundle = joblib.load(path)
        self.extra_work_ms = extra_work_ms

    def predict(self, rows):
        until = time.perf_counter() + self.extra_work_ms / 1000
        while time.perf_counter() < until:
            pass
        b = self.bundle
        values = b["model"].predict(pd.DataFrame(rows)[b["feature_columns"]])
        return {"predictions": [round(float(v), 1) for v in values], "model_version": b["version"],
                "served_by": str(os.getpid())}


@serve.deployment(ray_actor_options={"num_cpus": .25}, max_ongoing_requests=16)
@serve.ingress(api)
class Router:
    def __init__(self, prod, candidate, weight):
        self.prod, self.candidate, self.weight = prod, candidate, weight

    @api.post("/v1/models/eta:predict")
    async def predict(self, body: CanaryRequest, x_beta: str | None = Header(default=None)):
        candidate = x_beta == "1" or bucket(body.customer_id) < self.weight
        handle = self.candidate if candidate else self.prod
        output = await handle.predict.remote(body.instances)
        return {**output, "arm": "v2" if candidate else "v1"}


def build(weight=0.1, extra_work_ms=0.0):
    weight, extra_work_ms = float(weight), float(extra_work_ms)
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("weight must be between 0 and 1")
    if not math.isfinite(extra_work_ms) or extra_work_ms < 0:
        raise ValueError("extra_work_ms must be nonnegative")
    prod = EtaModel.options(name="EtaProd", num_replicas=2).bind(str(ROOT / "data/eta_v1.joblib"), extra_work_ms)
    candidate = EtaModel.options(name="EtaCandidate", autoscaling_config={
        "min_replicas": 1, "max_replicas": 3, "target_ongoing_requests": 2,
        "upscale_delay_s": 2, "downscale_delay_s": 15,
    }).bind(str(ROOT / "data/eta_v2.joblib"), extra_work_ms)
    return Router.bind(prod, candidate, weight)
