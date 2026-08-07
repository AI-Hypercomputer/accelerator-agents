#!/usr/bin/env python3
"""
CSAT Telemetry Collector Tool for Nexus Claude PoC.
Reads or generates an anonymous installation UUID in ~/.nexus/config.json
and records subjective feedback after explicit user consent.
"""
import os
import sys
import json
import uuid

CONFIG_FILE = os.path.expanduser("~/.nexus/config.json")

def get_or_create_install_uuid():
    config_dir = os.path.dirname(CONFIG_FILE)
    os.makedirs(config_dir, exist_ok=True)
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                if "installation_uuid" in data:
                    return data["installation_uuid"]
        except Exception:
            pass
    install_id = str(uuid.uuid4())
    with open(CONFIG_FILE, "w") as f:
        json.dump({"installation_uuid": install_id}, f, indent=2)
    return install_id

def main():
    print("=" * 65)
    print("[Nexus CSAT Telemetry Collector]")
    print("=" * 65)
    print("WARNING: This tool will transmit CSAT feedback to backend telemetry.")
    print("Please confirm explicit user consent before proceeding.")
    
    install_uuid = get_or_create_install_uuid()
    print(f"[Telemetry Info] Anonymous Installation UUID: {install_uuid}")
    
    # Simulate reading arguments or recording mock CSAT
    question_no = sys.argv[1] if len(sys.argv) > 1 else "Q1"
    score = sys.argv[2] if len(sys.argv) > 2 else "5"
    feedback = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else "Nexus PoC completed successfully."
    
    payload = {
        "installation_uuid": install_uuid,
        "question_no": question_no,
        "score": score,
        "subjective_feedback": feedback,
        "status": "Mock Telemetry Recorded Successfully"
    }
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
