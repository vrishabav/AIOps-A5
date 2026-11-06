from uuid import UUID

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from partA import eta_app as api

URL = "/v1/models/eta:predict"
ROW = dict(distance_km=12.0, n_stops=3, hour=10, weekday=2, traffic_index=.3, rain=0, weight_kg=4.0)


@pytest.fixture
def client():
    with TestClient(api.app) as c:
        yield c


def post(client, rows=None, **kwargs):
    return client.post(URL, json={"instances": [ROW] if rows is None else rows}, **kwargs)


def test_valid_and_health(client):
    response = post(client)
    assert response.status_code == 200
    expected = round(float(api.model.predict(pd.DataFrame([ROW])[api.bundle["feature_columns"]])[0]), 1)
    assert response.json()["predictions"] == [expected]
    assert response.json()["model_version"] == api.bundle["version"]
    health = client.get("/healthz")
    assert health.json()["status"] == "ok"
    assert health.json()["model_version"] == api.bundle["version"]


def test_batch_order(client):
    rows = [dict(ROW, distance_km=d) for d in [60.0, 1.0, 30.0]]
    expected = [post(client, [row]).json()["predictions"][0] for row in rows]
    assert len(set(expected)) == 3
    assert post(client, rows).json()["predictions"] == expected


@pytest.mark.parametrize("field,value", [
    ("distance_km", .09), ("distance_km", 501), ("n_stops", 0), ("n_stops", 61),
    ("hour", -1), ("hour", 24), ("weekday", -1), ("weekday", 7),
    ("traffic_index", -.01), ("traffic_index", 1.01), ("rain", -1), ("rain", 2),
    ("weight_kg", 0), ("weight_kg", 1001), ("distance_km", "no"),
    ("n_stops", 1.5), ("rain", True), ("hour", None),
])
def test_invalid_fields(client, field, value):
    response = post(client, [dict(ROW, **{field: value})])
    assert response.status_code == 422
    assert response.json()["error"] == "validation_failed"
    assert any(field in d["field"] for d in response.json()["details"])


def test_missing_field(client):
    row = ROW.copy()
    del row["rain"]
    assert post(client, [row]).status_code == 422


def test_malformed_json(client):
    response = client.post(URL, content="{", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["error"] == "validation_failed"


def test_infeasible_and_boundaries(client):
    response = post(client, [dict(ROW, distance_km=151, n_stops=21)])
    assert response.status_code == 422
    assert "infeasible" in response.text
    for distance, stops in [(150, 21), (151, 20)]:
        assert post(client, [dict(ROW, distance_km=distance, n_stops=stops)]).status_code == 200


@pytest.mark.parametrize("size,code", [(0, 422), (1, 200), (500, 200), (501, 422)])
def test_batch_limits(client, size, code):
    assert post(client, [ROW] * size).status_code == code


@pytest.mark.parametrize("rows", [[ROW], []])
@pytest.mark.parametrize("supplied", [True, False])
def test_request_ids_on_success_and_error(client, rows, supplied):
    response = post(client, rows, headers={"X-Request-ID": "client-123"} if supplied else {})
    rid = response.headers["X-Request-ID"]
    assert response.json()["request_id"] == rid
    assert rid == "client-123" if supplied else str(UUID(rid)) == rid


def test_metrics_exclude_scrapes(client):
    post(client)
    post(client, [])
    before = client.get("/metrics")
    after = client.get("/metrics")
    assert before.status_code == 200
    assert before.text == after.text
    assert "text/plain" in before.headers["content-type"]
    assert before.headers["X-Request-ID"]
    samples = [s for f in text_string_to_metric_families(before.text) for s in f.samples]
    assert any(s.name == "eta_requests_total" and s.labels == {"path": URL, "status": "422"} for s in samples)
    assert any(s.name == "eta_requests_total" and s.labels == {"path": URL, "status": "200"} for s in samples)
    buckets = [s for s in samples if s.name == "eta_request_latency_seconds_bucket"]
    assert len(buckets) >= 5 and buckets[-1].labels["le"] == "+Inf"
    assert any(s.name == "eta_request_latency_seconds_sum" for s in samples)
    assert any(s.name == "eta_request_latency_seconds_count" for s in samples)


def test_generic_500(client, monkeypatch):
    def broken(*args):
        raise RuntimeError("PRIVATE_DATABASE_PASSWORD")
    monkeypatch.setattr(api.model, "predict", broken)
    response = post(client, headers={"X-Request-ID": "failed-123"})
    assert response.status_code == 500
    assert response.json() == {"error": "internal_server_error", "request_id": "failed-123"}
    assert response.headers["X-Request-ID"] == "failed-123"
    assert "PRIVATE" not in response.text
    assert 'status="500"' in client.get("/metrics").text


def test_http_error_request_id(client):
    response = client.get("/missing")
    assert response.status_code == 404
    assert str(UUID(response.json()["request_id"])) == response.headers["X-Request-ID"]
