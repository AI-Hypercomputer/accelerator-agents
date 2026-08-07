#!/usr/bin/env python3
"""
Utility tool for Nexus subagents to self-report their active model tier and config source.
Reads the agent descriptor from _agents/agents/<agent_name>/agent.json if --agent is provided.
"""
import os
import sys
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Get agent runtime info")
    parser.add_argument("--agent", type=str, default="nexus_meta", help="Name of the agent")
    args, _ = parser.parse_known_args()

    agent_name = args.agent
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidate_paths = [
        os.path.join(workspace_root, "agents", agent_name, "agent.json"),
        os.path.join(workspace_root, "_agents", "agents", agent_name, "agent.json"),
        os.path.join(workspace_root, ".agents", "agents", agent_name, "agent.json"),
    ]


    model_name = "gemini-2.5-pro"
    config_source = None

    for candidate in candidate_paths:
        if os.path.exists(candidate):
            config_source = os.path.relpath(candidate, workspace_root)
            try:
                with open(candidate, "r") as f:
                    data = json.load(f)
                    model_name = data.get("model", model_name)
                    env_source = data.get("env", {}).get("NEXUS_CONFIG_SOURCE")
                    if env_source:
                        config_source = env_source
                break
            except Exception:
                pass

    if not config_source:
        model_name = os.environ.get("NEXUS_MODEL_NAME", "gemini-2.5-pro")
        config_source = os.environ.get("NEXUS_CONFIG_SOURCE", f"skills/{agent_name}/SKILL.md")


    print("=" * 65)
    print(f"[Nexus Runtime Info: Model = {model_name} | Config = {config_source}]")
    print("=" * 65)

if __name__ == "__main__":
    main()
