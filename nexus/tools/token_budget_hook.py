#!/usr/bin/env python3
"""
PreToolUse circuit breaker enforcing the Nexus subagent token budget.

Halts tool execution when either trigger fires:
  * NEXUS_SIMULATE_BUDGET_EXCEEDED=1 is set in the environment, or
  * the running agent is nexus_kernel_author and its cumulative session
    tokens exceed NEXUS_TOKEN_BUDGET (default 3000).

Deliberately silent in the passing case: emitting permissionDecision "allow"
would bypass Claude Code's permission system for EVERY tool call, so this hook
only speaks up when it is denying.
"""
import json
import os
import sys

DEFAULT_BUDGET = 3000
GUARDED_AGENT = "nexus_kernel_author"


def read_payload():
    try:
        raw = sys.stdin.read()
        if raw.strip():
            return json.loads(raw)
    except Exception:
        pass
    return {}


def session_tokens(payload):
    """Best-effort read of cumulative token usage from the hook payload."""
    usage = payload.get("token_usage") or payload.get("usage") or {}
    if isinstance(usage, (int, float)):
        return int(usage)
    if isinstance(usage, dict):
        total = usage.get("total") or usage.get("total_tokens")
        if total is not None:
            return int(total)
        inp = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        out = usage.get("output_tokens") or usage.get("completion_tokens") or 0
        return int(inp) + int(out)
    return 0


def main():
    payload = read_payload()

    try:
        budget = int(os.environ.get("NEXUS_TOKEN_BUDGET", DEFAULT_BUDGET))
    except ValueError:
        budget = DEFAULT_BUDGET

    simulate = os.environ.get("NEXUS_SIMULATE_BUDGET_EXCEEDED") == "1"
    agent = payload.get("agent_name") or payload.get("agent") or payload.get("subagent") or ""
    total = session_tokens(payload)

    over_budget = agent == GUARDED_AGENT and total > budget

    if not (simulate or over_budget):
        sys.exit(0)

    reason = (
        "EXECUTOR_TERMINATION_REASON_MAX_TOKEN_BUDGET_EXCEEDED -- "
        "Subagent execution halted: token budget exceeded"
    )
    if over_budget:
        reason += f" ({total} > {budget} tokens for {agent})."
    else:
        reason += " (simulated via NEXUS_SIMULATE_BUDGET_EXCEEDED=1)."

    response = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }
    print(json.dumps(response))
    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()


