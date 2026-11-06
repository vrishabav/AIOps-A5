"""SQLite registry, pure release decision, timed evaluation and HTTP shadowing."""
import argparse
import json
import math
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eta_data import FEATURES, make_data


def tracking():
    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    return mlflow


def register():
    mlflow = tracking()

    # Local class is serialized by value: model servers need no project imports.
    class EtaPyfunc(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            import joblib
            self.bundle = joblib.load(context.artifacts["bundle"])

        def predict(self, context, model_input, params=None):
            import pandas as pd
            b = self.bundle
            values = b["model"].predict(model_input[b["feature_columns"]])
            return pd.DataFrame({"eta_minutes": values.round(1), "model_version": b["version"]})

    from mlflow.models import infer_signature
    mlflow.set_experiment("swiftparcel-eta")
    client = mlflow.MlflowClient()
    sample = make_data(5, seed=99)[FEATURES].astype("float64")
    requirements = [f"{p}=={version(p)}" for p in
                    ("mlflow", "cloudpickle", "scikit-learn", "pandas", "numpy", "scipy", "joblib")]
    table = []
    for semantic, alias in [("v1", "prod"), ("v2", "candidate")]:
        path = Path(f"data/eta_{semantic}.joblib").resolve()
        b = joblib.load(path)
        output = pd.DataFrame({"eta_minutes": b["model"].predict(sample).round(1), "model_version": semantic})
        with mlflow.start_run(run_name=semantic):
            mlflow.set_tag("bundle_version", semantic)
            info = mlflow.pyfunc.log_model(
                name=f"eta-{semantic}", python_model=EtaPyfunc(), artifacts={"bundle": str(path)},
                signature=infer_signature(sample, output), input_example=sample,
                pip_requirements=requirements, registered_model_name="eta-model")
        registry_version = str(info.registered_model_version)
        client.set_model_version_tag("eta-model", registry_version, "bundle_version", semantic)
        client.set_registered_model_alias("eta-model", alias, registry_version)
        table.append({"version": registry_version, "alias": alias, "bundle": semantic})
    return table


def decide(prod, cand, min_rel_gain=.03, max_p95_ms=10.0, max_slice_regression=.05):
    """All criteria are inclusive. Invalid/nonfinite metrics fail closed."""
    limits = (min_rel_gain, max_p95_ms, max_slice_regression)
    if not all(math.isfinite(v) and v >= 0 for v in limits) or min_rel_gain > 1:
        raise ValueError("thresholds must be finite and nonnegative; min_rel_gain <= 1")
    keys = ("mae", "mae_rush", "mae_rain", "p95_ms")
    criteria = dict.fromkeys(("overall_gain", "latency", "rush_slice", "rain_slice"), False)
    if not all(isinstance(m.get(k), (int, float)) and math.isfinite(m[k]) and m[k] >= 0
               for m in (prod, cand) for k in keys):
        return {"criteria": criteria, "promote": False}

    def at_most(value, bound):
        return value <= bound or math.isclose(value, bound, rel_tol=1e-12, abs_tol=0)

    criteria = {
        "overall_gain": (prod["mae"] > 0 or min_rel_gain == 0) and at_most(cand["mae"], prod["mae"] * (1 - min_rel_gain)),
        "latency": at_most(cand["p95_ms"], max_p95_ms),
        "rush_slice": at_most(cand["mae_rush"], prod["mae_rush"] * (1 + max_slice_regression)),
        "rain_slice": at_most(cand["mae_rain"], prod["mae_rain"] * (1 + max_slice_regression)),
    }
    return {"criteria": criteria, "promote": all(criteria.values())}


def evaluate(model, data):
    from sklearn.metrics import mean_absolute_error
    x, y = data[FEATURES].astype("float64"), data.eta_minutes
    predictions = model.predict(x)["eta_minutes"].to_numpy()
    rush = data.hour.isin([7, 8, 9, 16, 17, 18, 19])
    rain = data.rain == 1
    metrics = {"mae": float(mean_absolute_error(y, predictions)),
               "mae_rush": float(mean_absolute_error(y[rush], predictions[rush])),
               "mae_rain": float(mean_absolute_error(y[rain], predictions[rain]))}
    for _ in range(10):
        model.predict(x.iloc[:1])
    times = []
    for i in range(200):
        started = time.perf_counter()
        model.predict(x.iloc[i:i + 1])
        times.append((time.perf_counter() - started) * 1000)
    metrics["p95_ms"] = float(np.percentile(times, 95))
    return metrics


def promotion_gate(**thresholds):
    thresholds = {"min_rel_gain": .03, "max_p95_ms": 10.0, "max_slice_regression": .05, **thresholds}
    mlflow = tracking()
    client = mlflow.MlflowClient()
    data = make_data(2000, seed=99)
    metrics, versions = {}, {}
    for alias in ("prod", "candidate"):
        v = client.get_model_version_by_alias("eta-model", alias).version
        versions[alias] = str(v)
        metrics[alias] = evaluate(mlflow.pyfunc.load_model(f"models:/eta-model/{v}"), data)
    return {"versions": versions, **metrics, "thresholds": thresholds,
            **decide(metrics["prod"], metrics["candidate"], **thresholds)}


def shadow(prod_url="http://127.0.0.1:5101/invocations", cand_url="http://127.0.0.1:5102/invocations"):
    import requests
    rows = pd.read_csv("data/eta_fresh.csv")[FEATURES].head(200).to_dict("records")
    latencies = [[], []]
    predictions = [[], []]
    versions = [set(), set()]
    with requests.Session() as session:
        for url in (prod_url, cand_url):
            for _ in range(10):
                session.post(url, json={"dataframe_records": [rows[0]]}, timeout=30).raise_for_status()
        for i, row in enumerate(rows):
            # Alternate order to avoid always giving one server the warmer CPU.
            for arm in ([0, 1] if i % 2 == 0 else [1, 0]):
                started = time.perf_counter()
                response = session.post((prod_url, cand_url)[arm], json={"dataframe_records": [row]}, timeout=30)
                response.raise_for_status()
                output = response.json()["predictions"][0]
                latencies[arm].append((time.perf_counter() - started) * 1000)
                predictions[arm].append(float(output["eta_minutes"]))
                versions[arm].add(output["model_version"])
    assert versions == [{"v1"}, {"v2"}], f"unexpected shadow versions: {versions}"
    diff = np.abs(np.array(predictions[0]) - predictions[1])
    return {"requests_per_server": len(rows), "mean_absolute_difference_minutes": float(diff.mean()),
            "percent_difference_over_10_minutes": float(100 * np.mean(diff > 10)),
            "prod_p95_ms": float(np.percentile(latencies[0], 95)),
            "candidate_p95_ms": float(np.percentile(latencies[1], 95))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["register", "gate", "shadow"])
    args = parser.parse_args()
    if args.action == "register":
        result = register()
    elif args.action == "gate":
        result = {"default": promotion_gate(), "strict": promotion_gate(min_rel_gain=.60)}
    else:
        result = shadow()
    print(json.dumps(result, indent=2))
