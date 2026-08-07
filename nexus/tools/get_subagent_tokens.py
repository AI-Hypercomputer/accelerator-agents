#!/usr/bin/env python3
"""
Utility tool to count and print token usage per subagent from Claude session logs.
"""
import os
import sys
import glob
import json

def get_latest_brain_dir():
    brain_root = os.path.expanduser("~/.gemini/Claude/brain")
    if not os.path.exists(brain_root):
        return None
    dirs = sorted(glob.glob(os.path.join(brain_root, "*")), key=os.path.getmtime, reverse=True)
    for d in dirs:
        if os.path.isdir(d):
            return d
    return None

def count_tokens(brain_dir=None):
    if not brain_dir:
        brain_dir = get_latest_brain_dir()

    token_table = {
        "nexus_meta": {"input": 0, "output": 0, "total": 0},
        "nexus_kb_retriever": {"input": 0, "output": 0, "total": 0},
        "nexus_profiler": {"input": 0, "output": 0, "total": 0},
        "nexus_kernel_author": {"input": 0, "output": 0, "total": 0}
    }
    found_real_tokens = False

    if brain_dir and os.path.exists(brain_dir):
        log_files = glob.glob(os.path.join(brain_dir, "**/*.jsonl"), recursive=True)
        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        agent_name = data.get("agent_name") or data.get("agent") or "nexus_meta"
                        if agent_name not in token_table:
                            token_table[agent_name] = {"input": 0, "output": 0, "total": 0}

                        usage = data.get("token_usage") or data.get("usage") or {}
                        if isinstance(usage, dict) and ("input_tokens" in usage or "prompt_tokens" in usage or "total_tokens" in usage):
                            inp = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                            out = usage.get("output_tokens") or usage.get("completion_tokens") or 0
                            tot = usage.get("total_tokens") or (inp + out)
                            token_table[agent_name]["input"] += inp
                            token_table[agent_name]["output"] += out
                            token_table[agent_name]["total"] += tot
                            found_real_tokens = True
            except Exception:
                continue

    # Fallback baseline table if running without active log transcript metadata
    if not found_real_tokens:
        label = "[Nexus Subagent Token Footprint Summary (Simulated / Mock Baseline)]"
        token_table = {
            "nexus_meta": {"input": 1250, "output": 340, "total": 1590},
            "nexus_kb_retriever": {"input": 820, "output": 150, "total": 970},
            "nexus_profiler": {"input": 910, "output": 180, "total": 1090},
            "nexus_kernel_author": {"input": 2100, "output": 480, "total": 2580}
        }
    else:
        label = "[Nexus Subagent Token Footprint Summary (Live Claude Session Logs)]"

    print("=" * 65)
    print(label)
    print("=" * 65)
    print(f"{'Subagent Name':<22} | {'Input Tokens':<13} | {'Output Tokens':<13} | {'Total Tokens':<12}")
    print("-" * 65)
    for agent, usage in token_table.items():
        print(f"{agent:<22} | {usage['input']:<13} | {usage['output']:<13} | {usage['total']:<12}")
    print("=" * 65)


if __name__ == "__main__":
    main_dir = sys.argv[1] if len(sys.argv) > 1 else None
    count_tokens(main_dir)
