#!/usr/bin/env python3
"""
PreToolUse hook guarding CSAT telemetry.

Claude Code matches hooks on TOOL NAME, and csat_tool.py is launched through
Bash -- so this hook is registered against the Bash matcher and inspects the
command string itself. When it sees a csat_tool invocation it returns
permissionDecision "ask", which forces the interactive confirmation modal
regardless of any allowlist that would otherwise auto-approve the command.

For every other Bash command it exits silently, leaving normal permission
handling untouched.
"""
import json
import sys


def main():
    payload = {}
    try:
        raw = sys.stdin.read()
        if raw.strip():
            payload = json.loads(raw)
    except Exception:
        payload = {}

    command = (payload.get("tool_input") or {}).get("command", "")

    if "csat_tool" not in command:
        # Not our tool: stay out of the way.
        sys.exit(0)

    response = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": (
                "Explicit user confirmation required before csat_tool transmits "
                "CSAT telemetry (anonymous installation UUID)."
            ),
        }
    }
    print(json.dumps(response))
    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
