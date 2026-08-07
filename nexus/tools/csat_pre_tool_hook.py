#!/usr/bin/env python3
"""
PreToolUse hook for csat_tool in Claude.
Returns 'force_ask' to guarantee that Claude pops up an interactive
confirmation modal in the UI before executing telemetry, regardless of
global IDE settings.
"""
import json
import sys

def main():
    payload = {
        "decision": "force_ask",
        "reason": "Explicit user confirmation required before executing CSAT telemetry (csat_tool)."
    }
    print(json.dumps(payload))
    sys.stdout.flush()

if __name__ == "__main__":
    main()
