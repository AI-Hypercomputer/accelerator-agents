#!/usr/bin/env python3
"""Reduces a raw autotune sweep into the documented best_config result shape.

The gap this closes
-------------------
`tpu_server.py`'s autotune endpoint emits, per trial:

    {"all_results": [{"cfg": {...}, "exit_code": 0, "output": "...", ...}, ...]}

It does NOT rank the trials or nominate a winner. But both downstream
consumers require one:

  * `apply_best_config.py` reads `results["best_config"]` and hard-fails with
    "No 'best_config' found" otherwise.
  * `autotune_summary_agent.md` is told to "Extract the `best_config` and
    `best_time_ms` from the results file".

Nothing in the pipeline produced those keys, so Phase 4 could never complete
as documented. Picking the winner is a pure reduction over measured numbers --
exactly the kind of step MaxKernel deliberately keeps out of an LLM's hands
(see `apply_best_config.py`'s module docstring) -- so it belongs here.

Selection rule: among trials that exited 0 AND printed `CORRECTNESS: True`,
pick the lowest `PERF_METRICS` (optimized-kernel latency in ms, as emitted by
`test_harness_template.py`). A trial that failed, timed out, or produced a
wrong answer is never eligible, no matter how fast it was.
"""

import argparse
import json
import re
import sys

# Emitted by test_harness_template.py: "PERF_METRICS: 0.057231"
_PERF_RE = re.compile(r"^PERF_METRICS:\s*([0-9eE+.\-]+)\s*$", re.M)
_CORRECT_RE = re.compile(r"^CORRECTNESS:\s*(True|False)\s*$", re.M)


def _trial_metrics(trial: dict):
  """Returns (time_ms, correct, reason) for one trial entry."""
  if trial.get("status") == "timeout":
    return None, False, "timeout"
  if trial.get("exit_code", 1) != 0:
    return None, False, f"exit_code={trial.get('exit_code')}"

  output = trial.get("output") or ""

  correct_m = _CORRECT_RE.search(output)
  if not correct_m:
    return None, False, "no CORRECTNESS line"
  if correct_m.group(1) != "True":
    return None, False, "incorrect result"

  perf_m = _PERF_RE.search(output)
  if not perf_m:
    return None, True, "no PERF_METRICS line"
  try:
    return float(perf_m.group(1)), True, ""
  except ValueError:
    return None, True, f"unparsable PERF_METRICS {perf_m.group(1)!r}"


def select_best(results: dict) -> dict:
  """Ranks an autotune sweep and returns the documented result shape."""
  if "best_config" in results and results["best_config"]:
    # Already reduced -- pass through untouched so this step is idempotent.
    return results

  trials = results.get("all_results")
  if not trials:
    raise ValueError(
        "No 'all_results' array found. Expected the autotune output of "
        "`tpu_client.py --action autotune`."
    )

  ranked = []
  rejected = []
  for trial in trials:
    cfg = trial.get("cfg", {})
    time_ms, correct, reason = _trial_metrics(trial)
    if time_ms is None:
      rejected.append({"cfg": cfg, "reason": reason, "correct": correct})
      continue
    ranked.append({"cfg": cfg, "time_ms": time_ms})

  ranked.sort(key=lambda r: r["time_ms"])

  if not ranked:
    raise ValueError(
        "No autotune trial produced a correct, timed result. Rejected: "
        + json.dumps(rejected)
    )

  return {
      "best_config": ranked[0]["cfg"],
      "best_time_ms": ranked[0]["time_ms"],
      "ranked_results": ranked,
      "rejected_results": rejected,
      "all_results": trials,
  }


def main():
  parser = argparse.ArgumentParser(
      description=(
          "Select the best autotune configuration from a raw sweep result."
      )
  )
  parser.add_argument(
      "results_path", help="Raw autotune results JSON (with 'all_results')."
  )
  parser.add_argument(
      "output_path",
      nargs="?",
      default=None,
      help="Where to write the reduced result (defaults to results_path).",
  )
  args = parser.parse_args()

  try:
    with open(args.results_path, "r") as f:
      results = json.load(f)
    reduced = select_best(results)
    out = args.output_path or args.results_path
    with open(out, "w") as f:
      json.dump(reduced, f, indent=2)
    print(
        f"Best config {reduced['best_config']} at"
        f" {reduced['best_time_ms']} ms -> {out}"
    )
    if reduced.get("rejected_results"):
      print(f"Rejected {len(reduced['rejected_results'])} trial(s):")
      for r in reduced["rejected_results"]:
        print(f"  - {r['cfg']}: {r['reason']}")
  except Exception as e:  # pylint: disable=broad-except
    print(f"Failed to select best config: {e}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()
