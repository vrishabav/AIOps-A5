# TA instructions - Assignment 5 (da24b033)

Layout follows the brief: `partA/` to `partD/`, `REPORT.pdf` (source `REPORT.typ`), plus `run.py`, `eta_data.py`, `load_test.py`.
Run everything from the folder containing `run.py`. CPU only; tested on Linux, Python 3.14.4.
Ports 8000, 5101, 5102 and 8100 must be free; Ray starts its own 2-CPU cluster.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python run.py all      # about 5 minutes; exit 0 = every stage passed
```

`run.py` overwrites the raw result files with your own measurements.
Stages can also be run one at a time: `data`, `tests`, `a`, `b`, `c`, `d`
(run `data` first; it creates the data and model bundles, which are not submitted).
Per-stage status is in `.runtime/status.json`, logs are in `.runtime/`.

## What to check

| Item | Command / file | Expected |
|---|---|---|
| A1, A2 | `python run.py tests` (or `pytest partA partB`) | 46 passed: 34 REST + 12 gate |
| A3 | `python run.py a` → `partA/loadtest_results.json`, `partA/tuning_results.json` | 1/2/4/8/16-client table, batch 1 vs 100, two-worker repeat |
| B1, B2 | `python run.py b` → `partB/gate_output.txt` | default gate promotes; `min_rel_gain=0.60` rejects |
| B3 | same stage → `partB/shadow_output.txt` | 200 requests per server, difference and p95 values |
| C1, C2 | `python run.py c` → `partC/verify_output.txt` | five `PASS` lines |
| D1 | `python run.py d` | smoke exits 0; wrong `--expect-version` exits 1 |
| C3, D1 answers, machine, not-tested list | `REPORT.pdf` | written answers |

Notes:
- Latency and throughput depend on the machine; judge the method and reasoning, not fixed values.
  The report's tables are from the recorded run; reruns update the raw JSON/text files only.
- A strict-gate `promote: false` is the expected decision, not a failure.
- Numeric registry versions can differ on reruns.
- Optional manual check: `python -m uvicorn partA.eta_app:app --port 8000`, then
  `curl localhost:8000/healthz` and `curl localhost:8000/metrics`.
- If a stage fails, read its log in `.runtime/`, free the port, and rerun that stage alone.
