#!/usr/bin/env python3
"""Render the live status of the jaxbench_level2 sweep.

Writes `_nightly/JAXBENCH_LEVEL2_STATUS.md`. Progress is read from each run's
own `state.json` rather than from anything the model says about itself, and the
per-op lifecycle (queued / running / finished) from the marker files
`run_jaxbench_level2.sh` maintains under `_nightly/jaxbench_level2_runs/`.

    python3 tools/nightly/jaxbench_status.py            # write once
    python3 tools/nightly/jaxbench_status.py --watch 60  # rewrite every 60s
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import time
import urllib.request

MK = pathlib.Path(__file__).resolve().parents[2]
BENCH = MK / "jaxbench_level2"
NIGHTLY = MK / "_nightly"
MARKERS = NIGHTLY / "jaxbench_level2_runs"
STATUS = NIGHTLY / "JAXBENCH_LEVEL2_STATUS.md"
TPU_PORT = 8000


def ops() -> list[str]:
  return sorted(
      d.name for d in BENCH.iterdir() if (d / "kernel_task.yaml").is_file()
  )


def n_configs(op: str) -> int:
  text = (BENCH / op / "kernel_task.yaml").read_text()
  return len(re.findall(r"'name':", text))


def load_json(path: pathlib.Path):
  try:
    return json.loads(path.read_text())
  except Exception:  # pylint: disable=broad-exception-caught
    return None


def fmt_dur(seconds: float | None) -> str:
  if seconds is None or seconds < 0:
    return "-"
  seconds = int(seconds)
  h, rem = divmod(seconds, 3600)
  m = rem // 60
  return f"{h}h{m:02d}m" if h else f"{m}m"


def fmt_num(v, spec="{:.3f}") -> str:
  if v is None or v == "NA" or v == "":
    return "-"
  try:
    return spec.format(float(v))
  except (TypeError, ValueError):
    return str(v)


def tpu_queue() -> str:
  try:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{TPU_PORT}/queue", timeout=5
    ) as r:
      q = json.loads(r.read().decode())
    return (
        f"reachable — {q.get('running_count', '?')} running, "
        f"{q.get('queued_count', '?')} queued, "
        f"{q.get('total_jobs', '?')} total"
    )
  except Exception:  # pylint: disable=broad-exception-caught
    return "**UNREACHABLE** — run `tools/nightly/gke_tpu_forward.sh ensure`"


def row_for(op: str, now: float) -> dict:
  marker = load_json(MARKERS / f"{op}.json") or {}
  state = {}
  run_dir = marker.get("run_dir")
  if run_dir:
    state = load_json(pathlib.Path(run_dir) / "state.json") or {}

  status = marker.get("status", "pending")
  started = marker.get("started_at")
  finished = marker.get("finished_at")
  if status == "running" and started:
    elapsed = now - started
  elif started and finished:
    elapsed = finished - started
  else:
    elapsed = None

  # A run whose marker says "running" but whose pid is gone died with its
  # parent; say so rather than showing it as live forever.
  if status == "running":
    pid = marker.get("pid")
    if pid and not pathlib.Path(f"/proc/{pid}").exists():
      status = "orphaned"

  return {
      "op": op,
      "status": status,
      "configs": n_configs(op),
      "iter": state.get("iteration", 0) if state else "-",
      "base_ms": state.get("base_time_ms") if state else None,
      "best_ms": state.get("best_optimized_time") if state else None,
      "speedup": state.get("best_speedup") if state else None,
      "elapsed": elapsed,
      "rc": marker.get("exit_code"),
      "run_id": marker.get("run_id", "-"),
      "log": marker.get("log", ""),
  }


ICON = {
    "done": "done",
    "running": "RUNNING",
    "failed": "FAILED",
    "timeout": "TIMEOUT",
    "orphaned": "ORPHANED",
    "incomplete": "INCOMPLETE",
    "stopped": "stopped",
    "pending": "pending",
}

# Statuses that need a human to look, even though some exited with rc=0.
NEEDS_ATTENTION = ("failed", "timeout", "orphaned", "incomplete")

# Operator-initiated halt; not a fault, but not a result either.
HALTED = ("stopped",)


def render() -> str:
  now = time.time()
  rows = [row_for(op, now) for op in ops()]

  counts: dict[str, int] = {}
  for r in rows:
    counts[r["status"]] = counts.get(r["status"], 0) + 1
  order = ["done", "running", "pending", "stopped", "incomplete",
           "failed", "timeout", "orphaned"]
  tally = "  ".join(
      f"{k}: {counts[k]}" for k in order if counts.get(k)
  )

  done = [r for r in rows if r["status"] == "done" and r["speedup"]]
  wins = [r for r in done if _f(r["speedup"]) and _f(r["speedup"]) > 1.05]

  out = [
      "# jaxbench_level2 sweep status",
      "",
      f"_Updated {dt.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}_",
      "",
      f"**{len(rows)} ops** — {tally or 'nothing started'}",
      "",
      f"- TPU backend: GKE `maxkernel-v6e-cluster` (us-east5-b) via "
      f"127.0.0.1:{TPU_PORT} — {tpu_queue()}",
      "- Each cluster job sees **1 v6e chip** "
      "(`TPU_VISIBLE_CHIPS=2`, `TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1`), "
      "not the local VM's 8.",
      "- Baselines are already-optimized Pallas kernels, so a speedup near "
      "1.00x is a legitimate result, not a failure.",
      "",
      "| op | status | iter | cfgs | base ms | best ms | speedup | elapsed | run |",
      "|---|---|---|---|---|---|---|---|---|",
  ]
  for r in rows:
    out.append(
        f"| {r['op']} | {ICON.get(r['status'], r['status'])} | {r['iter']} |"
        f" {r['configs']} | {fmt_num(r['base_ms'])} | {fmt_num(r['best_ms'])} |"
        f" {fmt_num(r['speedup'], '{:.4f}x')} | {fmt_dur(r['elapsed'])} |"
        f" {r['run_id']} |"
    )

  if done:
    best = sorted(done, key=lambda r: -(_f(r["speedup"]) or 0))
    out += [
        "",
        f"## Finished ({len(done)}) — {len(wins)} above 1.05x",
        "",
    ]
    for r in best:
      out.append(f"- **{r['op']}** {fmt_num(r['speedup'], '{:.4f}x')} "
                 f"({fmt_num(r['base_ms'])} ms -> {fmt_num(r['best_ms'])} ms)")

  failed = [r for r in rows if r["status"] in NEEDS_ATTENTION]
  if failed:
    out += ["", "## Needs attention", ""]
    for r in failed:
      out.append(
          f"- **{r['op']}** — {r['status']} (rc={r['rc']}), log: `{r['log']}`"
      )

  out += [
      "",
      "---",
      "",
      f"Results CSV: `{NIGHTLY / 'jaxbench_level2_results.csv'}`  ",
      f"Logs: `{NIGHTLY / 'logs' / 'jaxbench_level2'}`  ",
      f"Forward log: `{NIGHTLY / f'gke_forward.{TPU_PORT}.log'}`",
      "",
  ]
  return "\n".join(out)


def _f(v):
  try:
    return float(v)
  except (TypeError, ValueError):
    return None


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument(
      "--watch",
      type=int,
      default=0,
      metavar="SECONDS",
      help="rewrite the status file on this interval instead of once",
  )
  args = ap.parse_args()

  NIGHTLY.mkdir(parents=True, exist_ok=True)
  MARKERS.mkdir(parents=True, exist_ok=True)

  while True:
    STATUS.write_text(render())
    if not args.watch:
      print(f"wrote {STATUS}")
      return 0
    time.sleep(args.watch)


if __name__ == "__main__":
  raise SystemExit(main())
