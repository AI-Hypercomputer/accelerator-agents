#!/usr/bin/env python3
"""Render a live status page for the nightly MaxKernel sweep.

Everything here is derived from disk and from `ps` -- results.csv for finished
problems, each run's state.json for in-flight progress, the log filenames for
the problem <-> run_id mapping. Nothing is remembered between invocations, so
running it twice on an unchanged tree gives the same answer.

    tools/nightly/status.py            # write workspace/_nightly/STATUS.md
    tools/nightly/status.py --stdout   # print instead
"""

import argparse
import csv
import datetime
import glob
import json
import os
import re
import shutil
import subprocess

MK = os.environ.get("MK", "/home/cathygao_google_com/maxkernel")
NIGHTLY = os.path.join(MK, "workspace", "_nightly")
RESULTS = os.path.join(NIGHTLY, "results.csv")
LIST = os.path.join(NIGHTLY, "kernelbench_list.txt")
OUT = os.path.join(NIGHTLY, "STATUS.md")


# A run that exits 0 without committing an iteration did not actually do the
# work -- the ENOSPC failures on 2026-09-23 all looked like this. Treat it as a
# failure regardless of exit code.
def is_real_success(row):
  try:
    return int(row["iterations"]) > 0 and row["best_speedup"] not in ("NA", "")
  except (ValueError, KeyError):
    return False


def read_list():
  if not os.path.exists(LIST):
    return []
  with open(LIST) as f:
    return [l.strip() for l in f if l.strip()]


def read_results():
  if not os.path.exists(RESULTS):
    return []
  with open(RESULTS) as f:
    return list(csv.DictReader(f))


def log_index():
  """problem -> [(run_id, log_size_bytes)], newest last."""
  idx = {}
  for path in sorted(glob.glob(os.path.join(NIGHTLY, "logs", "*", "*.log"))):
    m = re.match(r"^(.*)\.(run_[0-9a-f]+)\.log$", os.path.basename(path))
    if not m:
      continue
    idx.setdefault(m.group(1), []).append((m.group(2), os.path.getsize(path)))
  return idx


def running():
  """problem -> elapsed string, from the live run_one.sh processes."""
  out = {}
  try:
    ps = subprocess.run(
      ["ps", "-eo", "etime=,args="], capture_output=True, text=True, timeout=10
    ).stdout
  except Exception:
    return out
  for line in ps.splitlines():
    m = re.search(r"run_one\.sh\s+(\S+)$", line.strip())
    if m and "grep" not in line:
      out[m.group(1)] = line.strip().split()[0]
  return out


def state_of(run_id):
  p = os.path.join(MK, "workspace", run_id, "state.json")
  try:
    with open(p) as f:
      return json.load(f)
  except Exception:
    return None


def sweep_alive():
  try:
    subprocess.run(
      ["pgrep", "-f", "run_batch.sh KernelBench"],
      capture_output=True,
      check=True,
      timeout=10,
    )
    return True
  except Exception:
    return False


def render():
  problems = read_list()
  rows = read_results()
  logs = log_index()
  live = running()

  by_problem = {}
  for r in rows:
    by_problem[r["problem"]] = r  # last row wins on a re-run

  ok = [
    p for p in problems if p in by_problem and is_real_success(by_problem[p])
  ]
  inflight = [p for p in problems if p in live]
  # A problem being retried right now belongs under "Running", not under the
  # failure list its previous attempt put it on.
  bad = [
    p
    for p in problems
    if p in by_problem and p not in live and not is_real_success(by_problem[p])
  ]
  # A problem xargs already consumed but that never wrote a results row is
  # NOT pending -- it will never be retried on its own. It died between
  # `claude -p` starting and run_one.sh's harvest step.
  orphaned = [
    p
    for p in problems
    if p not in by_problem and p not in live and logs.get(p.replace("/", "_"))
  ]
  pending = [
    p
    for p in problems
    if p not in by_problem and p not in live and p not in orphaned
  ]

  du = shutil.disk_usage("/")
  free_gb = du.free / 2**30
  now = datetime.datetime.now(datetime.timezone.utc).astimezone()

  L = []
  a = L.append
  a("# KernelBench sweep — live status")
  a("")
  a(
    f"_Generated {now.strftime('%Y-%m-%d %H:%M:%S %Z')} by `tools/nightly/status.py`._"
  )
  a("")
  a(
    f"**{len(ok)} / {len(problems)} done** · {len(inflight)} running · "
    f"{len(pending)} pending · **{len(bad) + len(orphaned)} need a re-run**"
  )
  a("")
  a(f"- sweep process: {'**alive**' if sweep_alive() else '**NOT RUNNING**'}")
  a(
    f"- disk free on `/`: **{free_gb:.1f} GB** ({du.used / du.total:.0%} used)"
    + ("  ← **LOW: runs fail with ENOSPC below ~2GB**" if free_gb < 8 else "")
  )
  a("")

  if bad or orphaned:
    a("## Need a re-run")
    a("")
    a("| op | run | exit | min | why | partial |")
    a("|---|---|---|---|---|---|")
    for p in bad:
      r = by_problem[p]
      sizes = dict(logs.get(p.replace("/", "_"), []))
      why = (
        "empty log — `claude -p` never started"
        if sizes.get(r["run_id"], 1) == 0
        else "exited without committing an iteration"
      )
      s = state_of(r["run_id"]) or {}
      sp = s.get("best_speedup")
      a(
        f"| {p.split('/')[-1]} | `{r['run_id']}` | {r['exit_code']} | "
        f"{r['minutes']} | {why} | "
        f"{('%.3gx @ iter %s' % (sp, s.get('iteration'))) if isinstance(sp, (int, float)) else '—'} |"
      )
    for p in orphaned:
      rid, size = logs[p.replace("/", "_")][-1]
      s = state_of(rid) or {}
      sp = s.get("best_speedup")
      why = (
        "empty log — `claude -p` never started"
        if size == 0
        else "killed before run_one.sh harvested the result"
      )
      a(
        f"| {p.split('/')[-1]} | `{rid}` | — | — | {why} | "
        f"{('%.3gx @ iter %s' % (sp, s.get('iteration'))) if isinstance(sp, (int, float)) else '—'} |"
      )
    a("")
    a("Re-run any of these with:")
    a("")
    a("```bash")
    a("MK_TIMEOUT=6h tools/nightly/run_one.sh KernelBench/<op>")
    a("```")
    a("")

  a("## Running")
  a("")
  if inflight:
    a("| op | elapsed | iter | best so far |")
    a("|---|---|---|---|")
    for p in inflight:
      rid = (logs.get(p.replace("/", "_")) or [(None, 0)])[-1][0]
      s = state_of(rid) if rid else None
      it = f"{s.get('iteration')}/{s.get('max_iterations')}" if s else "?"
      sp = s.get("best_speedup") if s else None
      a(
        f"| {p.split('/')[-1]} | {live[p]} | {it} | "
        f"{('%.3gx' % sp) if isinstance(sp, (int, float)) else '—'} |"
      )
  else:
    a("_none_")
  a("")

  a("## Finished")
  a("")
  a("| # | op | iters | base ms | best ms | speedup | min |")
  a("|---|---|---|---|---|---|---|")
  done_sorted = sorted(ok, key=lambda p: -float(by_problem[p]["best_speedup"]))
  for i, p in enumerate(done_sorted, 1):
    r = by_problem[p]
    a(
      f"| {i} | {p.split('/')[-1]} | {r['iterations']} | "
      f"{float(r['base_time_ms']):.4f} | {float(r['best_optimized_time']):.4f} | "
      f"**{float(r['best_speedup']):.2f}x** | {r['minutes']} |"
    )
  if ok:
    sp = [float(by_problem[p]["best_speedup"]) for p in ok]
    a("")
    a(
      f"median **{sorted(sp)[len(sp) // 2]:.2f}x** · "
      f"max **{max(sp):.2f}x** · min **{min(sp):.2f}x**"
    )
  a("")

  a("## Pending")
  a("")
  a(", ".join(p.split("/")[-1] for p in pending) if pending else "_none_")
  a("")
  return "\n".join(L) + "\n"


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--stdout", action="store_true")
  args = ap.parse_args()
  text = render()
  if args.stdout:
    print(text)
  else:
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
      f.write(text)
    os.replace(tmp, OUT)  # atomic: a reader never sees a half-written page
    print(f"wrote {OUT}")
