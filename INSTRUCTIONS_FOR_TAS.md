# Instructions for TAs — Assignment 5, da24b033

The solution follows the assignment's `partA/`–`partD/` layout. Run Python
commands from the directory containing `run.py`, `eta_data.py` and those folders.
This guide gives the current verification procedure; the report format is
**Typst/PDF**, replacing the earlier Markdown-report workflow.

## 1. Files and environment

In the working directory, the report is in the sibling folder:

```text
Assignment 5/
  report/REPORT.pdf
  report/REPORT.typ
  Module4_da24b033/
    INSTRUCTIONS_FOR_TAS.md
    run.py, eta_data.py, load_test.py, requirements.txt, README.md
    partA/, partB/, partC/, partD/
```

For a submission/extracted copy, `REPORT.pdf` and `REPORT.typ` may instead be
placed directly beside `run.py`. The runner supports both locations. The final
submission should put `REPORT.pdf` at the archive root, as the brief specifies.

- CPU only; tested on Linux with Python **3.14.4** and the pinned requirements.
- Ray verification starts its own local **2-CPU** cluster. Run it without other
  active Ray work on the same machine.
- Ports **8000, 5101, 5102 and 8100** must be available. No Docker or cloud account
  is needed. Typst is optional for execution; the compiled report is supplied.
- Generated model bundles, data and SQLite state are intentionally omitted.

From the solution root, use a separate environment:

```bash
python3 --version
mkdir -p .runtime
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/.runtime" PIP_CACHE_DIR="$PWD/.runtime/pip-cache"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ETA_MODEL=data/eta_v1.joblib
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

Use a Python 3.14 interpreter for `python3` above to match the tested pins. Keep
this environment active in every terminal used below. Run one stage at a time.

## 2. Complete verification

The bundled evidence records runs from **5 November 2026**. Verification will
overwrite raw result files with the evaluator's measurements. Use an extracted
copy, or preserve the bundled evidence first:

```bash
mkdir -p .runtime/before-verification
cp -a partA partB partC .runtime/before-verification/
```

The complete automated sequence is:

```bash
python run.py all
echo "verification exit code=$?"
```

Expected: exit **0**, 46 passing tests, load/MLflow completion, five Ray PASS
lines and a passing smoke result. The runner continues independent stages after
a failure and returns exit **1** if any stage failed. Inspect the saved stage
status and logs; do not infer success from a report PDF or old result file alone.

The equivalent stage-by-stage route is given below. Use either the complete
sequence or the individual stages; both routes perform the same checks.

### A1/A2 — regenerate models and test the service

```bash
python run.py data
python run.py tests
```

The first command creates the train/fresh CSVs, v1/v2 bundles and the 200-row
`data/eta_sample.csv`. It must run before the REST tests, because the service
loads its bundle during import.

Expected: **46 passed** in total: **34 REST cases** and **12 pure-gate cases**.
For a separate count or targeted diagnosis:

```bash
python -m pytest -q -o cache_dir=.runtime/pytest partA/test_eta_app.py
python -m pytest -q -o cache_dir=.runtime/pytest partB/test_gate.py
```

The REST cases cover a correct prediction, batch order, ranges/types/missing
fields/malformed JSON, the infeasible rule and its boundaries, batches
0/1/500/501, request-ID echo/generation on success and errors, health, metrics
including scrape exclusion, and generic 500 handling via a monkeypatched model.
The gate cases cover inclusive/inside/outside thresholds, slice regressions,
custom thresholds, invalid metrics, zero baselines and input immutability.

### A3 — load, batching and capacity change

```bash
python run.py a
```

This starts v1 on port 8000, warms it up, runs concurrency **1, 2, 4, 8, 16** for
**4 seconds each**, measures batch **100** at concurrency **1**, then repeats
the sweep with **two workers**. Native numerical-library thread counts stay at
one. It stops its own servers after measurement.

Inspect the measured tables and batch cost:

```bash
python - <<'PY'
import json
import pandas as pd
a = json.load(open("partA/loadtest_results.json"))
t = json.load(open("partA/tuning_results.json"))
columns = ["concurrency", "rps", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "errors"]
print("ONE WORKER")
print(pd.DataFrame(a)[columns].to_string(index=False))
print("TWO WORKERS")
print(pd.DataFrame(t["2"])[columns].to_string(index=False))
print("Batch 1 ms/row:", a[0]["mean_ms"])
print("Batch 100 ms/row:", t["batch100"]["mean_ms"] / 100)
PY
```

Expected: five rows per sweep, v1 responses and normally zero errors. Evaluate
capacity as the highest measured error-free throughput with **p95 <= 80 ms**.
The report explains saturation using throughput and mean latency through
Little's Law, batch overhead amortization, and the actual worker-change result.
The recorded two-worker trial did **not** double capacity; the brief permits an
honest measured failure to reach 2x. There is no fixed expected req/s score.

### B1/B2/B3 — registry, both gates and real model-server shadowing

```bash
python run.py b
python -m json.tool partB/gate_output.txt
python -m json.tool partB/shadow_output.txt
```

This registers both custom pyfunc models in SQLite, with signatures, examples
and pinned serving requirements. It assigns `eta-model@prod` to the v1 bundle
and `eta-model@candidate` to v2. Numeric registry versions can change on reruns;
the saved report's versions 3/4 are not hardcoded expectations.

The gates evaluate `make_data(2000, seed=99)`, including the specified rush/rain
slices and 200 timed single-row predictions after warm-up. `decide()` is pure;
promotion requires **all four** criteria. The default gate uses 3% minimum
relative MAE gain, 10 ms maximum p95 and 5% permitted slice regression. The
second gate raises minimum gain to 60%.

Expected for the recorded models: MAE approximately **11.955** for prod and
**6.105** for candidate; about **48.94%** improvement. The recorded default gate
passes, while the 60% gate rejects. A strict-gate `promote: false` is an expected
decision, not a failed verification stage. Timing criteria must use the actual
measurements rather than forcing the previous decision.

The same stage launches `mlflow models serve` on **5101** and **5102**, sends the
same first **200 fresh rows** to both servers, one row per measured request, and
saves prediction disagreement and each server's HTTP p95. Local gate timing
excludes HTTP; shadow timing includes it. Both servers are stopped afterwards.
Server logs are `.runtime/mlflow-prod.log` and `.runtime/mlflow-candidate.log`.

### C1/C2 — Ray canary and all five assertions

```bash
python run.py c
cat partC/verify_output.txt
```

Allow several minutes; the helper collects output before displaying it. Do not
start another copy while it is quiet. Expected: exit 0 and **five PASS lines**:

1. 1,000 distinct customers at weight 0.2; candidate share in **[0.16, 0.24]**.
2. 200 customers retain their arm on a second request.
3. 200 retain their arm after deleting and redeploying the identical app.
4. Raising weight to 0.5 moves no existing candidate customer back to prod.
5. All 50 forced-candidate requests work; normal routing resumes afterwards.

The fixed IDs recorded a candidate share of **0.213**. Source inspection confirms
SHA-256 bucketing, `X-Beta: 1`, two prod replicas, candidate autoscaling **1–3**,
and 0.25 CPU reservations for each replica/router. At maximum scale the total
reservation is **1.5 CPUs**, within the 2-CPU Ray allocation. Ray shuts down when
verification finishes. Detailed output is `.runtime/ray-driver.log`.

### D1 — release smoke success and deliberate rejection

```bash
python run.py d
python -m json.tool .runtime/smoke-output.txt
python -m json.tool .runtime/smoke-negative.txt
```

Expected: seven checks pass, the positive CLI returns **0**, and deliberately
expecting the wrong version returns **1**. The stage asserts the negative
subprocess exit code itself. Checks cover version/finite positive outputs,
distance sanity, traffic sanity, batch order, invalid-input 422, request ID and
HTTP p95 over 30 timed requests. The report gives four abort criteria with
metrics, thresholds and windows; those production monitoring rules are proposed,
not claims of an exercised live rollback system.

## 3. Confirm this verification run completed

After `all`, or after every individual stage above, run:

```bash
python - <<'PY'
import json
from pathlib import Path
s = json.load(open(".runtime/status.json"))
for key in ("data", "tests", "a", "b", "c", "d"):
    assert s.get(key) == "PASS", (key, s.get(key))
print("All six execution stages: PASS")
g = json.load(open("partB/gate_output.txt"))
from partB.eta_mlflow import decide
for label in ("default", "strict"):
    d = g[label]
    expected = decide(d["prod"], d["candidate"], **d["thresholds"])
    assert d["criteria"] == expected["criteria"]
    assert d["promote"] == expected["promote"]
    print(label, "promotion:", d["promote"])
assert json.load(open("partB/shadow_output.txt"))["requests_per_server"] == 200
lines = Path("partC/verify_output.txt").read_text().splitlines()
assert len(lines) == 5 and all(line.startswith("PASS ") for line in lines)
positive = json.load(open(".runtime/smoke-output.txt"))
negative = json.load(open(".runtime/smoke-negative.txt"))
assert positive["pass"] and all(positive["checks"].values())
assert len(positive["checks"]) == 7 and positive["latency_requests"] == 30
assert negative["pass"] is False
print("Gate decisions, shadow count, five Ray checks and both smoke paths: PASS")
PY
```

The checks above are in addition to reading the process exit codes and current
logs. The JSON status file is a record of execution, not a live health check.

## 4. Optional direct HTTP inspection

After data generation, start the service in **terminal 1** from the solution
root, with the same environment active:

```bash
python -m uvicorn partA.eta_app:app --host 127.0.0.1 --port 8000
```

In **terminal 2**, activate the same environment and run:

```bash
curl -i http://127.0.0.1:8000/healthz
python - <<'PY'
import requests
url = "http://127.0.0.1:8000/v1/models/eta:predict"
row = dict(distance_km=12.0, n_stops=3, hour=10, weekday=2,
           traffic_index=0.3, rain=0, weight_kg=4.0)
for name, delivery, status in [
    ("valid", row, 200),
    ("infeasible", dict(row, distance_km=151, n_stops=21), 422),
]:
    rid = "ta-" + name
    r = requests.post(url, json={"instances": [delivery]},
                      headers={"X-Request-ID": rid}, timeout=10)
    print(name, r.status_code, r.json())
    assert r.status_code == status
    assert r.headers["X-Request-ID"] == r.json()["request_id"] == rid
    if status == 422:
        assert "infeasible" in r.text
print("Direct HTTP checks: PASS")
PY
curl -sS http://127.0.0.1:8000/metrics
```

Check the counter labels, histogram buckets including `+Inf`, `_sum` and
`_count`. The prediction list retains input order and one-decimal rounding.
Stop terminal 1 with **Ctrl+C** before rerunning stages A or D. `/docs` is
disabled; inspect the API through requests, health, metrics and tests.

## 5. Rubric and evidence map

| Item | Code / verification | Evidence / written answer |
|---|---|---|
| A1: REST contract | `partA/eta_app.py`, direct HTTP checks | 34 REST test cases |
| A2: own tests | `partA/test_eta_app.py` | `.runtime/tests.txt` after a fresh run |
| A3: load, batching, worker change | `load_test.py`, `run.py a` | `partA/loadtest_results.json`, `partA/tuning_results.json`; report A |
| B1: registration | `register()` in `partB/eta_mlflow.py` | `partB/gate_output.txt`; report B1 |
| B2: multi-criteria gate | `decide()`, `promotion_gate()`, `partB/test_gate.py` | Both gate outputs; report B2 |
| B3: paired shadow | `shadow()`, actual MLflow servers via `run.py b` | `partB/shadow_output.txt`; report B3 |
| C1: canary app | `build()` in `partC/eta_ray.py` | Config, stable hash and header routing |
| C2: five required checks | `partC/verify_ray.py` | `partC/verify_output.txt` |
| C3: hash, stickiness, scaling | Report C3 | Concise written answers |
| D1: smoke and abort policy | `partD/smoke_eta.py`, `run.py d` | Positive/negative smoke logs; report D |

## 6. Report editing and reproducibility notes

`REPORT.typ` is self-contained and editable. From the Assignment 5 directory:

```bash
typst compile report/REPORT.typ report/REPORT.pdf
```

From the code directory in the current working layout:

```bash
typst compile ../report/REPORT.typ ../report/REPORT.pdf
```

If the report files are beside `run.py` in an extracted submission:

```bash
typst compile REPORT.typ REPORT.pdf
```

`python run.py report` also compiles the source if Typst is installed; otherwise
it uses the supplied PDF. It does not regenerate Markdown or overwrite the
editable report text. Report figures describe the recorded 5 November run;
rerunning experiments updates the raw evidence but does **not** automatically
rewrite the report's tables/date. Update those in Typst when preparing a new
recorded report. The PDF can be read without installing Typst or the Lato font.

Latency and throughput depend on the evaluator's machine and current load.
Assess the measurement method and reasoning rather than requiring identical
timings. The supplied report includes machine details, actual results, an
explicit untested list, starter attribution and AI disclosure. No production
traffic, sustained-load autoscaling or live alert/rollback drills are claimed.

## 7. Retry and cleanup

- A missing model means data generation has not completed: run `python run.py data`.
- Inspect `.runtime/status.json` and the relevant `.runtime/` log after a failure;
  rerun the failed stage. Do not run stages concurrently.
- If a port is occupied, stop only the known demo service, usually with Ctrl+C.
  Avoid broad process-kill commands. Completed stages stop their own servers.
- If Ray was interrupted, allow its cleanup before retrying. Do not start a
  second verifier against an existing local Ray job.
- The runtime folder, environment, generated CSV/joblib files, `mlruns/`,
  `mlartifacts/` and `mlflow.db*` are local state, not required submission files.
  Keep raw result JSON/text evidence and the report in the submission.
