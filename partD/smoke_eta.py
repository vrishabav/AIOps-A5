"""Release smoke gate for the validated Part A endpoint (exit 0/1)."""
import argparse
import json
import math
import time

import numpy as np
import requests

ROW = dict(distance_km=10.0, n_stops=3, hour=11, weekday=2, traffic_index=.2, rain=0, weight_kg=5.0)


def smoke(url, expect_version, max_p95_ms=80):
    if not math.isfinite(max_p95_ms) or max_p95_ms <= 0:
        raise ValueError("max-p95-ms must be positive and finite")
    checks = {}
    with requests.Session() as session:
        def predict(rows):
            response = session.post(url, json={"instances": rows}, timeout=10)
            response.raise_for_status()
            body = response.json()
            values = body["predictions"]
            assert body["model_version"] == expect_version, f"expected version {expect_version}, got {body['model_version']}"
            assert len(values) == len(rows) and all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values)
            return body

        base = predict([ROW])
        checks["version_schema_finite_positive"] = True
        distance = predict([dict(ROW, distance_km=40)])["predictions"][0]
        assert distance > base["predictions"][0], "longer distance must increase ETA"
        checks["distance_sanity"] = True
        traffic = predict([dict(ROW, traffic_index=.9)])["predictions"][0]
        assert traffic >= base["predictions"][0], "heavier traffic must not shorten ETA"
        checks["traffic_sanity"] = True
        batch = predict([ROW, dict(ROW, distance_km=40)])
        assert batch["predictions"] == [base["predictions"][0], distance]
        checks["batch_order"] = True
        invalid = session.post(url, json={"instances": [dict(ROW, distance_km=-1)]}, timeout=10)
        assert invalid.status_code == 422, f"invalid input status={invalid.status_code}"
        checks["invalid_input_422"] = True
        assert base["request_id"]
        checks["request_id"] = True
        times = []
        for _ in range(30):
            started = time.perf_counter()
            predict([ROW])
            times.append((time.perf_counter() - started) * 1000)
        p95 = float(np.percentile(times, 95))
        assert p95 <= max_p95_ms, f"p95={p95:.3f} ms exceeds {max_p95_ms} ms"
        checks["latency_slo"] = True
    return {"checks": checks, "p95_ms": p95, "latency_requests": 30, "model_version": expect_version}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--max-p95-ms", type=float, default=80)
    args = parser.parse_args()
    try:
        result = smoke(args.url, args.expect_version, args.max_p95_ms)
    except Exception as exc:
        print(json.dumps({"pass": False, "reason": str(exc)}))
        raise SystemExit(1)
    print(json.dumps({"pass": True, **result}, indent=2))
