"""Serial fresh-process systems checks; optionally wait for the quality job."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--wait-pid", type=int)
p.add_argument("--repeats", type=int, default=2)
p.add_argument("--methods", nargs="+", default=["full", "svdd", "svdd_prerope", "keydiff", "leverage", "snapkv", "streaming"])
p.add_argument("--fraction", type=float, default=.2)
a = p.parse_args()
if a.wait_pid:
    print(f"Waiting for quality process {a.wait_pid}; no GPU work starts until it exits.", flush=True)
    while True:
        try:
            os.kill(a.wait_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(3)
    rows = [json.loads(s) for s in Path("results/pilot/predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 1350, "Quality run did not complete; systems suite not launched"
out = Path("results/systems"); out.mkdir(parents=True, exist_ok=True)
order = [(method, repeat) for repeat in range(a.repeats) for method in a.methods]
random.Random(20260927).shuffle(order)
(out / "suite.json").write_text(json.dumps({"arguments":vars(a), "order":order,
    "policy":"serial, fresh process per measurement, fixed 32 decode forwards; no simultaneous quality job",
    "started_local":time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2)+"\n")
for number, (method, repeat) in enumerate(order, 1):
    print(f"SYSTEMS {number}/{len(order)} {method} repeat={repeat}", flush=True)
    subprocess.run([sys.executable, "scripts/benchmark_systems.py", "--method", method,
                    "--fraction", str(a.fraction), "--skip-weight-hash", "--output",
                    str(out / f"{method}_r{repeat}.json")], check=True)
print("Systems suite complete", flush=True)
