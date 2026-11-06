# Assignment 5

## Files

| Files | Purpose |
|---|---|
| `partA/`, `partB/`, `partC/`, `partD/` | Implementations, required tests and evidence |
| `partA/tuning_results.json` | Additional raw evidence for the required batch-100 and two-worker experiments; baseline is stored only in `loadtest_results.json` |
| `REPORT.pdf` / `REPORT.typ` | Tables, written answers, machine details, not-tested list |
| `eta_data.py`, `load_test.py` | Supplied shared data/model generator and adapted load generator |
| `requirements.txt` | Tested dependency versions |
| `conftest.py` | Empty; puts the repo root on the import path for `pytest` |
| `run.py` | One helper for running all stages and packaging |

## Setup and run

Run from this folder:

```bash
mkdir -p .runtime
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/.runtime" PIP_CACHE_DIR="$PWD/.runtime/pip-cache"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py all
```

`all` generates data, runs pytest, measures REST load/batching/two workers,
registers and gates MLflow models, shadows both model servers, verifies Ray
routing and checks smoke success/failure. Failures produce exit 1 and an
incomplete status. See `INSTRUCTIONS_FOR_TAS.md` for verification.

For individual stages, first run `python run.py data`, then:

```bash
python run.py tests
python run.py a        # REST and capacity experiments
python run.py b        # MLflow registry, both gates and shadow
python run.py c        # Ray's five HTTP verification checks
python run.py d        # smoke success and wrong-version rejection
python run.py report   # compile REPORT.typ if Typst is installed
```

Run one stage at a time; ports 8000, 5101, 5102 and 8100 must be free. The helper stops its own servers. Ray verification starts and shuts down its own 2-CPU cluster; run it without an existing Ray cluster. On Linux it uses a short `/proc/<pid>/cwd` path for Unix sockets, with actual files in `.runtime/ray/`. All models use CPU; native BLAS/OpenMP threads are limited to one.

## AI disclosure
AI Assistants and Agents were used for tasks including:
- Generating boilerplate and syntax for implementations; writing tests
- Drafting README, INSTRUCTIONS_FOR_TAS.md
- Typesetting syntax, refining language for the report
- Verification and debugging