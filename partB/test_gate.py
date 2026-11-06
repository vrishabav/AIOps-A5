import copy

import pytest

from partB.eta_mlflow import decide

PROD = dict(mae=10.0, mae_rush=12.0, mae_rain=8.0, p95_ms=2.0)
CAND = dict(mae=9.0, mae_rush=11.0, mae_rain=7.0, p95_ms=3.0)


def test_all_pass_and_pure():
    p, c = copy.deepcopy(PROD), copy.deepcopy(CAND)
    result = decide(p, c)
    assert result["promote"] and all(result["criteria"].values())
    assert p == PROD and c == CAND


@pytest.mark.parametrize("key,at,inside,outside,criterion", [
    ("mae", 9.7, 9.6999, 9.7001, "overall_gain"),
    ("p95_ms", 10.0, 9.9999, 10.0001, "latency"),
    ("mae_rush", 12.6, 12.5999, 12.6001, "rush_slice"),
    ("mae_rain", 8.4, 8.3999, 8.4001, "rain_slice"),
])
def test_inclusive_boundaries(key, at, inside, outside, criterion):
    for value in (at, inside):
        assert decide(PROD, dict(CAND, **{key: value}))["promote"]
    result = decide(PROD, dict(CAND, **{key: outside}))
    assert not result["promote"]
    assert not result["criteria"][criterion]
    assert sum(result["criteria"].values()) == 3


def test_strict_relative_gain():
    assert not decide(PROD, CAND, min_rel_gain=.60)["promote"]
    assert decide(PROD, dict(CAND, mae=4), min_rel_gain=.60)["promote"]


def test_custom_thresholds():
    assert decide(PROD, dict(CAND, p95_ms=20, mae_rush=13.2),
                  max_p95_ms=20, max_slice_regression=.1)["promote"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_invalid_metrics_fail_closed(bad):
    assert not decide(PROD, dict(CAND, mae=bad))["promote"]


def test_missing_metric_and_zero_baseline():
    assert not decide(PROD, {"mae": 0})["promote"]
    assert not decide(dict(PROD, mae=0), dict(CAND, mae=0))["promote"]
    assert decide(dict(PROD, mae_rain=0), dict(CAND, mae_rain=0))["promote"]
    assert not decide(dict(PROD, mae_rain=0), CAND)["promote"]
    assert not decide(dict(PROD, mae_rain=0), dict(CAND, mae_rain=1e-13))["promote"]


def test_bad_threshold_rejected():
    with pytest.raises(ValueError):
        decide(PROD, CAND, min_rel_gain=-.1)
