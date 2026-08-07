#!/usr/bin/env python3
"""
Prints a per-agent token footprint table for the current Claude Code session.

Reads the real Claude Code transcript at ~/.claude/projects/<slug>/<id>.jsonl.
Subagent turns are marked with isSidechain=true; the parent session's own turns
are everything else. Claude Code does not stamp a subagent NAME on each entry,
so sidechain usage is reported in aggregate rather than invented per agent.

If no transcript is readable, prints a clearly labelled mock baseline instead of
silently reporting zeros.
"""
import glob
import json
import os
import sys

PROJECTS_ROOT = os.path.expanduser("~/.claude/projects")


def latest_transcript():
    candidates = glob.glob(os.path.join(PROJECTS_ROOT, "**", "*.jsonl"), recursive=True)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def collect(transcript):
    buckets = {
        "primary session (nexus_meta)": {"input": 0, "output": 0, "total": 0},
        "subagents (sidechain turns)": {"input": 0, "output": 0, "total": 0},
    }
    found = False

    with open(transcript, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except Exception:
                continue

            usage = (entry.get("message") or {}).get("usage") or {}
            if not usage:
                continue

            inp = (
                usage.get("input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
            )
            out = usage.get("output_tokens", 0)
            if not (inp or out):
                continue

            key = (
                "subagents (sidechain turns)"
                if entry.get("isSidechain")
                else "primary session (nexus_meta)"
            )
            buckets[key]["input"] += inp
            buckets[key]["output"] += out
            buckets[key]["total"] += inp + out
            found = True

    return buckets, found


MOCK_BASELINE = {
    "nexus_meta": {"input": 1250, "output": 340, "total": 1590},
    "nexus_kb_retriever": {"input": 820, "output": 150, "total": 970},
    "nexus_profiler": {"input": 910, "output": 180, "total": 1090},
    "nexus_kernel_author": {"input": 2100, "output": 480, "total": 2580},
}


def main():
    transcript = sys.argv[1] if len(sys.argv) > 1 else latest_transcript()

    table, found = (None, False)
    if transcript and os.path.exists(transcript):
        try:
            table, found = collect(transcript)
        except Exception:
            found = False

    if found:
        label = "[Nexus Token Footprint (live Claude Code session transcript)]"
        source = os.path.basename(transcript)
    else:
        table = MOCK_BASELINE
        label = "[Nexus Token Footprint (SIMULATED / MOCK BASELINE - no transcript read)]"
        source = "mock baseline"

    print("=" * 78)
    print(label)
    print(f"Source: {source}")
    print("=" * 78)
    print(f"{'Agent':<30} | {'Input':<13} | {'Output':<13} | {'Total':<12}")
    print("-" * 78)
    for name, usage in table.items():
        print(
            f"{name:<30} | {usage['input']:<13} | {usage['output']:<13} | {usage['total']:<12}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
