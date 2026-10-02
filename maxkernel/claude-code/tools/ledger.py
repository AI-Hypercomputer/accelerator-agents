#!/usr/bin/env python3
"""The ideas-ledger state machine: agents propose, this tool records.

`ideas_ledger.json` is the single junction between the run's two chains of
custody. The semantics spine (primary source -> golden values -> base.py ->
harness) decides what is *correct*. The advisory spine (reference kernel ->
brief -> reconciliation -> this ledger) only ever proposes what might be
*fast*. An idea crosses from the second into the first exactly once, when a
plan adopts it, and it is adjudicated against a real XProf trace afterwards.

Status transitions are a state machine, not prose, so they live in a tool
rather than in an agent's judgement. An agent that could write
`"status": "confirmed"` directly could also write it for an idea that was
never adopted, or re-propose one the profile already refuted -- and the whole
value of the ledger is that at the end of the run you can say which borrowed
ideas actually produced speedups instead of asserting that the reference
helped.

  proposed ──adopt──▶ adopted ──verdict──▶ confirmed
      │                                └──▶ refuted
      │                                └──▶ inconclusive
      └──drop──▶ dropped

A NON_PORTABLE idea can never be adopted: it is recorded so the planner can
see the mechanism was considered and discarded, which is a stronger defence
against transliteration than leaving it out.

Exit codes:
  0  operation applied
  2  illegal transition, unknown id, or schema violation
  4  ledger file missing or unreadable
"""

import argparse
import json
import sys
from pathlib import Path

VALID_CLASSES = {"ALGORITHMIC", "STRUCTURAL", "NON_PORTABLE"}
VALID_STATUSES = {
  "proposed",
  "adopted",
  "confirmed",
  "refuted",
  "inconclusive",
  "dropped",
}
VALID_VERDICTS = {"confirmed", "refuted", "inconclusive"}
VALID_TRUST = {"aligned", "partial", "divergent", "rejected"}

REQUIRED_FIELDS = (
  "id",
  "claim",
  "evidence",
  "class",
  "tpu_translation",
  "mechanism",
  "falsifiable_as",
)


class LedgerError(Exception):
  pass


def load(path):
  p = Path(path)
  if not p.is_file():
    raise LedgerError(f"ledger not found: {path}")
  try:
    data = json.loads(p.read_text())
  except json.JSONDecodeError as e:
    raise LedgerError(f"ledger is not valid JSON: {e}") from e
  data.setdefault("ideas", [])
  data.setdefault("reference_trust", "rejected")
  return data


def save(path, data):
  Path(path).write_text(json.dumps(data, indent=2) + "\n")


def find(data, idea_id):
  for idea in data["ideas"]:
    if idea["id"] == idea_id:
      return idea
  raise LedgerError(f"no such idea: {idea_id}")


def validate(data):
  """Checks the whole ledger. Returns a list of problems (empty == valid)."""
  problems = []
  trust = data.get("reference_trust")
  if trust not in VALID_TRUST:
    problems.append(
      f"reference_trust must be one of {sorted(VALID_TRUST)}, got {trust!r}"
    )

  seen = set()
  for i, idea in enumerate(data.get("ideas", [])):
    where = idea.get("id", f"#{i}")
    for field in REQUIRED_FIELDS:
      if not idea.get(field):
        problems.append(f"{where}: missing required field {field!r}")
    if idea.get("id") in seen:
      problems.append(f"{where}: duplicate id")
    seen.add(idea.get("id"))

    if idea.get("class") not in VALID_CLASSES:
      problems.append(f"{where}: class must be one of {sorted(VALID_CLASSES)}")
    status = idea.get("status", "proposed")
    if status not in VALID_STATUSES:
      problems.append(
        f"{where}: status must be one of {sorted(VALID_STATUSES)}"
      )

    if idea.get("class") == "NON_PORTABLE" and status not in (
      "proposed",
      "dropped",
    ):
      problems.append(
        f"{where}: NON_PORTABLE ideas can never be adopted or adjudicated "
        f"(status={status!r}). They are recorded so the planner can see the "
        "mechanism was considered and discarded."
      )
    if status in VALID_VERDICTS and idea.get("adopted_in") is None:
      problems.append(f"{where}: has verdict {status!r} but was never adopted")
    if status == "adopted" and idea.get("adopted_in") is None:
      problems.append(f"{where}: status 'adopted' requires adopted_in")

  return problems


def cmd_init(args):
  """Creates an empty ledger, or validates an agent-written one in place."""
  path = Path(args.path)
  if path.is_file() and not args.force:
    data = load(path)
    problems = validate(data)
    if problems:
      for p in problems:
        print(f"INVALID: {p}", file=sys.stderr)
      return 2
    # Normalize: every idea gets the bookkeeping fields the loop relies on.
    for idea in data["ideas"]:
      idea.setdefault("status", "proposed")
      idea.setdefault("adopted_in", None)
      idea.setdefault("verdict_evidence", None)
      idea.setdefault("depends_on_difference", None)
      idea.setdefault("history", [])
    save(path, data)
    print(f"Validated and normalized {len(data['ideas'])} ideas in {path}")
    return 0

  data = {
    "reference_trust": args.trust,
    "alignment_path": args.alignment,
    "ideas": [],
  }
  save(path, data)
  print(f"Initialized empty ledger at {path} (trust={args.trust})")
  return 0


def cmd_list(args):
  data = load(args.path)
  ideas = data["ideas"]
  if args.status:
    ideas = [i for i in ideas if i.get("status", "proposed") == args.status]
  if args.klass:
    ideas = [i for i in ideas if i.get("class") == args.klass]
  if args.adoptable:
    ideas = [
      i
      for i in ideas
      if i.get("class") != "NON_PORTABLE"
      and i.get("status", "proposed") == "proposed"
    ]

  if args.json:
    print(
      json.dumps(
        {"reference_trust": data["reference_trust"], "ideas": ideas}, indent=2
      )
    )
    return 0

  print(f"reference_trust: {data['reference_trust']}")
  if not ideas:
    print("(no matching ideas)")
    return 0
  for idea in ideas:
    status = idea.get("status", "proposed")
    adopted = f" @iter{idea['adopted_in']}" if idea.get("adopted_in") else ""
    dep = idea.get("depends_on_difference")
    dep_note = f"  depends_on={dep}" if dep else ""
    print(f"{idea['id']:<12} {idea['class']:<13} {status}{adopted}{dep_note}")
    print(f"             {idea['claim'].strip().splitlines()[0][:96]}")
  return 0


def cmd_adopt(args):
  data = load(args.path)
  idea = find(data, args.id)
  status = idea.get("status", "proposed")

  if idea.get("class") == "NON_PORTABLE":
    raise LedgerError(
      f"{args.id} is NON_PORTABLE and can never be adopted. It is in the "
      "ledger so the plan can show the mechanism was considered and "
      "discarded, not so it can be ported."
    )
  if status == "refuted":
    raise LedgerError(
      f"{args.id} was refuted by the profile in iteration "
      f"{idea.get('adopted_in')}; it must not be re-proposed. Evidence: "
      f"{idea.get('verdict_evidence')}"
    )
  if status not in ("proposed", "inconclusive"):
    raise LedgerError(f"{args.id} cannot move from {status!r} to 'adopted'")
  if data["reference_trust"] == "rejected":
    raise LedgerError(
      "reference_trust is 'rejected' -- the reference implements a "
      "different operation and no idea from it may be adopted."
    )

  idea["status"] = "adopted"
  idea["adopted_in"] = args.iteration
  idea.setdefault("history", []).append(
    {"event": "adopted", "iteration": args.iteration, "note": args.note}
  )
  save(args.path, data)
  print(f"{args.id} -> adopted @iter{args.iteration}")
  return 0


def cmd_verdict(args):
  data = load(args.path)
  idea = find(data, args.id)
  status = idea.get("status", "proposed")
  if status != "adopted":
    raise LedgerError(
      f"{args.id} is {status!r}; only an adopted idea can be adjudicated. "
      "A claim that was never put into a kernel has nothing to check "
      "against the trace."
    )
  if args.result not in VALID_VERDICTS:
    raise LedgerError(f"result must be one of {sorted(VALID_VERDICTS)}")
  if not args.evidence:
    raise LedgerError(
      "a verdict needs evidence from the trace -- the falsifiable claim is "
      "the whole point of the entry"
    )

  idea["status"] = args.result
  idea["verdict_evidence"] = args.evidence
  idea.setdefault("history", []).append(
    {
      "event": args.result,
      "iteration": idea.get("adopted_in"),
      "evidence": args.evidence,
    }
  )
  save(args.path, data)
  print(f"{args.id} -> {args.result}")
  return 0


def cmd_drop(args):
  data = load(args.path)
  idea = find(data, args.id)
  if idea.get("status") in VALID_VERDICTS:
    raise LedgerError(f"{args.id} is already adjudicated ({idea['status']})")
  idea["status"] = "dropped"
  idea.setdefault("history", []).append({"event": "dropped", "note": args.note})
  save(args.path, data)
  print(f"{args.id} -> dropped")
  return 0


def cmd_report(args):
  """The reference-contribution table for the orchestrator's final report."""
  data = load(args.path)
  rows = []
  for idea in data["ideas"]:
    status = idea.get("status", "proposed")
    if status == "adopted":
      outcome = "adopted, not yet adjudicated"
    elif status in VALID_VERDICTS:
      outcome = status
    elif idea.get("class") == "NON_PORTABLE":
      outcome = "dropped (no TPU counterpart)"
    elif idea.get("depends_on_difference"):
      outcome = f"not adopted (depends on {idea['depends_on_difference']})"
    else:
      outcome = "not adopted"
    rows.append(
      {
        "id": idea["id"],
        "class": idea["class"],
        "summary": idea["claim"].strip().splitlines()[0][:60],
        "adopted_in": idea.get("adopted_in"),
        "outcome": outcome,
        "evidence": idea.get("verdict_evidence"),
      }
    )

  if args.json:
    print(
      json.dumps(
        {"reference_trust": data["reference_trust"], "rows": rows}, indent=2
      )
    )
    return 0

  print(f"Reference contribution (reference_trust = {data['reference_trust']})")
  if not rows:
    print("  (no ideas were extracted from the reference)")
    return 0
  for r in rows:
    where = f"iter{r['adopted_in']}" if r["adopted_in"] else "--"
    print(f"  {r['id']:<12} {r['summary']:<60} {where:<7} {r['outcome']}")
    if r["evidence"]:
      print(f"  {'':<12} evidence: {r['evidence']}")
  return 0


def cmd_validate(args):
  data = load(args.path)
  problems = validate(data)
  if problems:
    for p in problems:
      print(f"INVALID: {p}", file=sys.stderr)
    return 2
  print(
    f"Ledger valid: {len(data['ideas'])} ideas, trust={data['reference_trust']}"
  )
  return 0


def main():
  parser = argparse.ArgumentParser(
    description="Read and transition the ideas ledger."
  )
  sub = parser.add_subparsers(dest="command", required=True)

  p = sub.add_parser(
    "init", help="create an empty ledger, or validate one in place"
  )
  p.add_argument("path")
  p.add_argument("--trust", default="rejected", choices=sorted(VALID_TRUST))
  p.add_argument(
    "--alignment", default=None, help="path to reference_alignment.md"
  )
  p.add_argument(
    "--force", action="store_true", help="overwrite an existing ledger"
  )
  p.set_defaults(func=cmd_init)

  p = sub.add_parser("list", help="list ideas, optionally filtered")
  p.add_argument("path")
  p.add_argument("--status", choices=sorted(VALID_STATUSES))
  p.add_argument("--class", dest="klass", choices=sorted(VALID_CLASSES))
  p.add_argument(
    "--adoptable", action="store_true", help="only ideas a plan may still adopt"
  )
  p.add_argument("--json", action="store_true")
  p.set_defaults(func=cmd_list)

  p = sub.add_parser("adopt", help="record that a plan adopted an idea")
  p.add_argument("path")
  p.add_argument("--id", required=True)
  p.add_argument("--iteration", type=int, required=True)
  p.add_argument("--note", default=None)
  p.set_defaults(func=cmd_adopt)

  p = sub.add_parser(
    "verdict", help="adjudicate an adopted idea against the trace"
  )
  p.add_argument("path")
  p.add_argument("--id", required=True)
  p.add_argument("--result", required=True, choices=sorted(VALID_VERDICTS))
  p.add_argument("--evidence", required=True)
  p.set_defaults(func=cmd_verdict)

  p = sub.add_parser("drop", help="retire an idea without adopting it")
  p.add_argument("path")
  p.add_argument("--id", required=True)
  p.add_argument("--note", default=None)
  p.set_defaults(func=cmd_drop)

  p = sub.add_parser("report", help="the reference-contribution table")
  p.add_argument("path")
  p.add_argument("--json", action="store_true")
  p.set_defaults(func=cmd_report)

  p = sub.add_parser(
    "validate", help="check the whole ledger against the schema"
  )
  p.add_argument("path")
  p.set_defaults(func=cmd_validate)

  args = parser.parse_args()
  try:
    sys.exit(args.func(args))
  except LedgerError as e:
    print(f"LEDGER ERROR: {e}", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
  main()
