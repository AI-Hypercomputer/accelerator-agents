#!/usr/bin/env python3
"""
Reports a Nexus agent's configured model tier and config source.

Reads the YAML frontmatter of agents/<agent_name>.md, which is the real
Claude Code subagent descriptor format. nexus_meta has no descriptor: it is the
primary session agent, not a subagent, so it runs on whatever model the session
is using.
"""
import argparse
import os
import sys

PLUGIN_ROOT = os.environ.get(
    "CLAUDE_PLUGIN_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)


def parse_frontmatter(path):
    """Minimal frontmatter reader -- avoids a PyYAML dependency."""
    fields = {}
    with open(path, "r") as f:
        if f.readline().strip() != "---":
            return fields
        for line in f:
            if line.strip() == "---":
                break
            if ":" in line and not line.startswith((" ", "\t", "#")):
                key, _, value = line.partition(":")
                fields[key.strip()] = value.strip()
    return fields


def main():
    parser = argparse.ArgumentParser(description="Get Nexus agent runtime info")
    parser.add_argument("--agent", default="nexus_meta", help="Name of the agent")
    args, _ = parser.parse_known_args()

    agent = args.agent
    descriptor = os.path.join(PLUGIN_ROOT, "agents", f"{agent}.md")

    if os.path.exists(descriptor):
        fields = parse_frontmatter(descriptor)
        model = fields.get("model", "(unset - inherits session model)")
        config_source = os.path.relpath(descriptor, PLUGIN_ROOT)
    elif agent == "nexus_meta":
        model = "(primary session agent - inherits the session model)"
        config_source = "skills/nexus_meta/SKILL.md"
    else:
        print(f"[Nexus Runtime Info] Unknown agent: {agent}", file=sys.stderr)
        sys.exit(1)

    print("=" * 65)
    print(f"[Nexus Runtime Info: Model = {model} | Config = {config_source}]")
    print("=" * 65)


if __name__ == "__main__":
    main()
