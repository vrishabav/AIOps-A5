"""Run all five canary checks over real HTTP; owns and shuts down its local Ray."""
import json
import os
import sys
from pathlib import Path

import ray
import requests
from ray import serve

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eta_data import FEATURES, make_data
from partC.eta_ray import build


def verify(port=8100):
    storage = Path(".runtime/ray").resolve()
    storage.mkdir(parents=True, exist_ok=True)
    # /proc keeps Unix socket paths short; all files still live inside this repo.
    temp = f"/proc/{os.getpid()}/cwd/.runtime/ray" if sys.platform.startswith("linux") else str(storage)
    ray.init(num_cpus=2, include_dashboard=False, object_store_memory=100 * 1024**2,
             _temp_dir=temp, _plasma_directory=str(storage), _node_ip_address="127.0.0.1")
    try:
        serve.start(http_options={"host": "127.0.0.1", "port": port})
        url = f"http://127.0.0.1:{port}/v1/models/eta:predict"
        row = make_data(1, seed=99)[FEATURES].to_dict("records")[0]
        ids = [f"customer-{i:04d}" for i in range(1000)]
        evidence = {}
        with requests.Session() as session:
            def arm(customer, override=False):
                response = session.post(url, json={"customer_id": customer, "instances": [row]},
                                        headers={"X-Beta": "1"} if override else {}, timeout=30)
                response.raise_for_status()
                body = response.json()
                assert body["model_version"] == body["arm"]
                assert len(body["predictions"]) == 1
                return body["arm"]

            serve.run(build(weight=.2), name="eta", route_prefix="/")
            original = {customer: arm(customer) for customer in ids}
            share = sum(v == "v2" for v in original.values()) / len(ids)
            assert abs(share - .2) <= .04, share
            evidence["candidate_share_1000"] = share
            print(f"PASS a: 1000 customers, candidate share={share:.3f} (required 0.16..0.24)", flush=True)
            assert all(arm(c) == original[c] for c in ids[:200])
            evidence["second_call_sticky_200"] = True
            print("PASS b: 200/200 customers sticky on second HTTP call", flush=True)
            serve.delete("eta")
            serve.run(build(weight=.2), name="eta", route_prefix="/")
            assert all(arm(c) == original[c] for c in ids[:200])
            evidence["redeploy_sticky_200"] = True
            print("PASS c: 200/200 customers sticky after identical redeployment", flush=True)
            serve.run(build(weight=.5), name="eta", route_prefix="/")
            increased = {customer: arm(customer) for customer in ids}
            assert all(increased[c] == "v2" for c in ids if original[c] == "v2")
            evidence["monotonic_1000"] = True
            evidence["candidate_share_at_half"] = sum(v == "v2" for v in increased.values()) / len(ids)
            print("PASS d: no original candidate customer moved to prod at weight=0.5", flush=True)
            prod_ids = [c for c in ids if increased[c] == "v1"][:50]
            assert len(prod_ids) == 50
            assert all(arm(c, override=True) == "v2" for c in prod_ids)
            assert all(arm(c) == "v1" for c in prod_ids)
            evidence["override_50"] = True
            print("PASS e: 50/50 prod customers forced to v2; next normal call remains v1", flush=True)
        return evidence
    finally:
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
