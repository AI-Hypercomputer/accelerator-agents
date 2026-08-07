#!/usr/bin/env python3
"""
Production-ready conditional PreToolUse hook for enforcing subagent token budget caps.
Checks incoming Claude JSON stdin payload and environment triggers.
"""
import json
import os
import sys

def main():
    # Read payload from stdin if available
    payload = {}
    try:
        if not sys.stdin.isatty():
            raw_input = sys.stdin.read()
            if raw_input.strip():
                payload = json.loads(raw_input)
    except Exception:
        payload = {}

    # Check environment variable simulation trigger
    simulate_exceeded = os.environ.get("NEXUS_SIMULATE_BUDGET_EXCEEDED") == "1"

    # Extract agent name and token usage from hook payload if present
    agent_name = payload.get("agent_name") or payload.get("agent") or payload.get("subagent") or ""
    token_usage = payload.get("token_usage", {})
    if isinstance(token_usage, dict):
        total_tokens = token_usage.get("total", 0)
    elif isinstance(token_usage, (int, float)):
        total_tokens = int(token_usage)
    else:
        total_tokens = 0

    # Condition: simulate env var set OR (nexus_kernel_author and total_tokens > 3000)
    budget_exceeded = simulate_exceeded or (agent_name == "nexus_kernel_author" and total_tokens > 3000)

    if budget_exceeded:
        response = {
            "decision": "deny",
            "reason": "EXECUTOR_TERMINATION_REASON_MAX_TOKEN_BUDGET_EXCEEDED",
            "message": "Subagent execution halted: Token budget exceeded."
        }
        print(json.dumps(response))
        sys.stdout.flush()
        sys.exit(0)
    else:
        response = {
            "decision": "allow"
        }
        print(json.dumps(response))
        sys.stdout.flush()
        sys.exit(0)

if __name__ == "__main__":
    main()
