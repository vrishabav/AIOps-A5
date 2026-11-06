"""Collect rubric evidence, compile the editable Typst report, or package submission."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import socket
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
RUNTIME = ROOT / ".runtime"
RUNTIME.mkdir(exist_ok=True)
os.environ.update({"PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "MLFLOW_ENABLE_ASYNC_LOGGING": "false",
                   "RAY_USAGE_STATS_ENABLED": "0", "TMPDIR": str(RUNTIME), "MPLCONFIGDIR": str(RUNTIME / "mpl")})
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
os.environ.setdefault("ETA_MODEL", "data/eta_v1.joblib")
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def command(*args, output=None, timeout=300, quiet=False):
    result = subprocess.run([sys.executable, *args], text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout)
    if output:
        Path(output).write_text(result.stdout)
    if not quiet:
        print(result.stdout, end="", flush=True)
    if result.returncode:
        raise RuntimeError(f"command exited {result.returncode}: {' '.join(args)}")
    return result.stdout


@contextmanager
def server(args, port, health, name):
    import requests
    # Refuse to measure somebody else's process or stop a process we did not start.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    with (RUNTIME / f"{name}.log").open("w") as log:
        process = subprocess.Popen([sys.executable, *args], stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} stopped; inspect .runtime/{name}.log")
                try:
                    if requests.get(f"http://127.0.0.1:{port}{health}", timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(.25)
            else:
                raise TimeoutError(f"{name} not ready after 120 seconds")
            yield
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def data():
    command("eta_data.py")
    import pandas as pd
    from eta_data import FEATURES
    pd.read_csv("data/eta_fresh.csv")[FEATURES].head(200).to_csv("data/eta_sample.csv", index=False)


def tests():
    command("-m", "pytest", "-q", "-o", "cache_dir=.runtime/pytest", "partA/test_eta_app.py", "partB/test_gate.py",
            output=RUNTIME / "tests.txt")


def part_a():
    import pandas as pd
    import requests
    from load_test import run
    rows = pd.read_csv("data/eta_sample.csv").to_dict("records")
    url = "http://127.0.0.1:8000/v1/models/eta:predict"
    results = {}
    for workers in (1, 2):
        args = ["-m", "uvicorn", "partA.eta_app:app", "--host", "127.0.0.1", "--port", "8000",
                "--workers", str(workers), "--no-access-log"]
        with server(args, 8000, "/healthz", f"rest-{workers}"):
            for _ in range(20):
                requests.post(url, json={"instances": rows[:1]}, timeout=10).raise_for_status()
            table = [run(url, "rest", c, 4, rows, 1) for c in (1, 2, 4, 8, 16)]
            if workers == 2:
                results["2"] = table
            if workers == 1:
                save("partA/loadtest_results.json", table)
                results["batch100"] = run(url, "rest", 1, 4, rows, 100)
                command("partD/smoke_eta.py", "--url", url, "--expect-version", "v1",
                        output=RUNTIME / "smoke-output.txt")
        print(f"Part A: completed {workers}-worker measurements", flush=True)
    save("partA/tuning_results.json", results)


def part_b():
    from partB.eta_mlflow import register, promotion_gate, shadow
    registry = register()
    gate = {"registry": registry, "default": promotion_gate(), "strict": promotion_gate(min_rel_gain=.60)}
    save("partB/gate_output.txt", gate)
    args = ["-m", "mlflow", "models", "serve", "--env-manager", "local", "--host", "127.0.0.1"]
    with server([*args, "-m", "models:/eta-model@prod", "-p", "5101"], 5101, "/ping", "mlflow-prod"):
        with server([*args, "-m", "models:/eta-model@candidate", "-p", "5102"], 5102, "/ping", "mlflow-candidate"):
            save("partB/shadow_output.txt", shadow())


def part_c():
    output = command("-m", "partC.verify_ray", output=RUNTIME / "ray-driver.log", timeout=600, quiet=True)
    lines = [line for line in output.splitlines() if line.startswith("PASS ")]
    assert len(lines) == 5, "missing Ray verification evidence"
    Path("partC/verify_output.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def part_d():
    url = "http://127.0.0.1:8000/v1/models/eta:predict"
    with server(["-m", "uvicorn", "partA.eta_app:app", "--host", "127.0.0.1", "--port", "8000", "--no-access-log"],
                8000, "/healthz", "smoke"):
        command("partD/smoke_eta.py", "--url", url, "--expect-version", "v1", output=RUNTIME / "smoke-output.txt")
        # Prove the release gate fails closed when a wrong version is expected.
        result = subprocess.run([sys.executable, "partD/smoke_eta.py", "--url", url, "--expect-version", "wrong"],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 1, "wrong-version release must fail"
        (RUNTIME / "smoke-negative.txt").write_text(result.stdout)



def report_paths():
    for folder in (ROOT, ROOT.parent / "report"):
        source, pdf = folder / "REPORT.typ", folder / "REPORT.pdf"
        if source.exists() or pdf.exists():
            return source, pdf
    raise FileNotFoundError("Keep REPORT.pdf/REPORT.typ at the solution root or in ../report/")


def report():
    source, pdf = report_paths()
    if source.exists() and shutil.which("typst"):
        subprocess.run(["typst", "compile", str(source), str(pdf)], check=True)
    elif not pdf.exists():
        raise RuntimeError("Compile REPORT.typ with Typst, or include the compiled REPORT.pdf")
    print(f"Report PDF: {pdf}")
    print("Recorded figures are edited in REPORT.typ; new measurements do not rewrite them.")


def package(rollno):
    if not rollno or not all(c.isalnum() or c in "_-" for c in rollno):
        raise ValueError("pass --rollno with your actual roll number")
    report()
    source, pdf = report_paths()
    files = ["README.md", "INSTRUCTIONS_FOR_TAS.md", "requirements.txt", "eta_data.py", "load_test.py", "run.py"]
    for folder in ("partA", "partB", "partC", "partD"):
        files += [str(p) for p in Path(folder).iterdir() if p.suffix in (".py", ".txt", ".json")]
    target = f"Module4_{rollno}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path)
        archive.write(pdf, "REPORT.pdf")
        if source.exists():
            archive.write(source, "REPORT.typ")
    print(f"Created {target}; check the report roll number and recorded figures before submitting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["all", "data", "tests", "a", "b", "c", "d", "report", "zip"])
    parser.add_argument("--rollno")
    args = parser.parse_args()
    status = read(RUNTIME / "status.json", {})
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor()) if Path("/proc/cpuinfo").exists() else platform.processor()
    ram = round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3, 1)
    status["machine"] = {"cpu": cpu, "logical_cpus": os.cpu_count(), "ram_gib": ram, "os": platform.platform()}
    save(RUNTIME / "status.json", status)
    if args.action == "report":
        report()
    elif args.action == "zip":
        package(args.rollno)
    else:
        actions = {"data": data, "tests": tests, "a": part_a, "b": part_b, "c": part_c, "d": part_d}
        selected = list(actions) if args.action == "all" else [args.action]
        failed = False
        for name in selected:
            print(f"Running {name}...", flush=True)
            try:
                actions[name]()
                status[name] = "PASS"
            except Exception as exc:
                status[name] = f"NOT COMPLETED: {exc}"
                failed = True
                print(status[name], flush=True)
            latest = read(RUNTIME / "status.json", {})
            latest[name] = status[name]
            save(RUNTIME / "status.json", latest)
        report()
        raise SystemExit(1 if failed else 0)
