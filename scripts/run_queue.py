"""Run a list of training jobs as subprocesses (N in parallel), one log per job.

  python scripts/run_queue.py jobs.json --parallel 2 --tag stage4

jobs.json: [{"name": "lr5e-4", "seed": 1, "set": {"lr": 5e-4}}, ...]
Each job must pass a smoke test first unless "--no-smoke" is given.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "results" / "logs"


def cmd(job, smoke=False):
    c = [sys.executable, "-m", "src.train", "--name", job["name"], "--seed", str(job.get("seed", 1))]
    if smoke:
        c.append("--smoke")
    if job.get("set"):
        c += ["--set"] + [f"{k}={json.dumps(v)}" for k, v in job["set"].items()]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--tag", default="queue")
    ap.add_argument("--no-smoke", action="store_true")
    ap.add_argument("--gpus", type=int, default=0, help="spread parallel jobs over this many CUDA GPUs")
    a = ap.parse_args()
    jobs = json.loads(Path(a.jobs).read_text())
    LOGS.mkdir(parents=True, exist_ok=True)
    env = {"PYTHONUNBUFFERED": "1", **__import__("os").environ}
    if not a.no_smoke:
        seen = set()
        for j in jobs:
            key = json.dumps(j.get("set", {}), sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            r = subprocess.run(cmd(j, smoke=True), cwd=ROOT, env=env, capture_output=True, text=True)
            ok = r.returncode == 0 and "SMOKE OK" in r.stdout
            print(f"[{a.tag}] smoke {j['name']}: {'OK' if ok else 'FAILED'}", flush=True)
            if not ok:
                print(r.stdout[-2000:], r.stderr[-3000:], flush=True)
                sys.exit(1)
    pending, running = list(jobs), []
    while pending or running:
        while pending and len(running) < a.parallel:
            j = pending.pop(0)
            log = LOGS / f"{a.tag}_{j['name']}_s{j.get('seed', 1)}.log"
            f = open(log, "w")
            jenv = dict(env)
            if a.gpus:
                load = [sum(1 for r in running if r[4] == g) for g in range(a.gpus)]
                gpu = load.index(min(load))                     # least-loaded GPU
                jenv["CUDA_VISIBLE_DEVICES"] = str(gpu)
            else:
                gpu = None
            p = subprocess.Popen(cmd(j), cwd=ROOT, env=jenv, stdout=f, stderr=subprocess.STDOUT)
            running.append((p, j, f, time.time(), gpu))
            print(f"[{a.tag}] started {j['name']} s{j.get('seed', 1)} -> {log.name}", flush=True)
        time.sleep(5)
        for item in running[:]:
            p, j, f, t0, _ = item
            if p.poll() is not None:
                f.close()
                running.remove(item)
                print(f"[{a.tag}] finished {j['name']} s{j.get('seed', 1)} rc={p.returncode} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"[{a.tag}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
