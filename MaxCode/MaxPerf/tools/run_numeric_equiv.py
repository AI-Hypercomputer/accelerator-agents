#!/usr/bin/env python3
# pylint: skip-file
"""run_numeric_equiv.py — Run 100-prompt corpus to generate references or compare outputs.

Usage:
  # To generate references from a baseline server:
  python tools/run_numeric_equiv.py --generate --port 8000

  # To compare current server outputs against references:
  python tools/run_numeric_equiv.py --compare --port 8000
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
import urllib.error
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

# Fallback layout: if prompts.jsonl is in the same directory as the script
if os.path.isfile(os.path.join(SCRIPT_DIR, "prompts.jsonl")):
  PROMPTS_PATH = os.path.join(SCRIPT_DIR, "prompts.jsonl")
  REF_DIR = os.path.join(SCRIPT_DIR, "reference_outputs")
else:
  PROMPTS_PATH = os.path.join(ROOT, "raw", "numeric_ref", "prompts.jsonl")
  REF_DIR = os.path.join(ROOT, "raw", "numeric_ref", "reference_outputs")


def get_model_name(host, port):
  url = f"http://{host}:{port}/v1/models"
  try:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
      data = json.loads(response.read().decode())
      return data["data"][0]["id"]
  except Exception as e:
    print(f"Error connecting to server at {url}: {e}", file=sys.stderr)
    sys.exit(1)


def query_completion(host, port, model, prompt_text, max_tokens):
  url = f"http://{host}:{port}/v1/completions"
  payload = {
      "model": model,
      "prompt": prompt_text,
      "max_tokens": max_tokens,
      "temperature": 0.0,
  }
  headers = {"Content-Type": "application/json"}
  data = json.dumps(payload).encode("utf-8")

  req = urllib.request.Request(url, data=data, headers=headers)
  try:
    with urllib.request.urlopen(req) as response:
      res_body = json.loads(response.read().decode())
      text = res_body["choices"][0]["text"]
      return {"status": "success", "text": text}
  except Exception as e:
    return {"status": "error", "error": str(e)}


def load_prompts():
  prompts = []
  if not os.path.isfile(PROMPTS_PATH):
    print(f"ERROR: Prompts file not found at {PROMPTS_PATH}", file=sys.stderr)
    sys.exit(1)
  with open(PROMPTS_PATH, "r") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      prompts.append(json.loads(line))
  return prompts


def main():
  parser = argparse.ArgumentParser(description="Numeric Equivalence Tool")
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument(
      "--generate",
      action="store_true",
      help="Generate reference outputs from the server",
  )
  group.add_argument(
      "--compare",
      action="store_true",
      help="Compare current server outputs to reference outputs",
  )

  parser.add_argument("--host", default="localhost", help="Server host")
  parser.add_argument("--port", type=int, default=8000, help="Server port")
  parser.add_argument(
      "--concurrency",
      type=int,
      default=10,
      help="Number of concurrent requests",
  )
  args = parser.parse_args()

  os.makedirs(REF_DIR, exist_ok=True)
  prompts = load_prompts()
  print(f"Loaded {len(prompts)} prompts from {PROMPTS_PATH}")

  model = get_model_name(args.host, args.port)
  print(f"Connected to server. Model: {model}")

  results = {}
  with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
    futures = {
        executor.submit(
            query_completion,
            args.host,
            args.port,
            model,
            p["text"],
            p["max_tokens"],
        ): p
        for p in prompts
    }

    for future in as_completed(futures):
      p = futures[future]
      pid = p["prompt_id"]
      res = future.result()
      results[pid] = res

  if args.generate:
    print("Writing reference files...")
    for pid, res in results.items():
      if res["status"] != "success":
        print(f"WARNING: Prompt {pid} failed: {res.get('error')}")
      ref_file = os.path.join(REF_DIR, f"prompt_{pid}.json")
      with open(ref_file, "w") as f:
        json.dump(res, f, indent=2)
    print("References generated successfully.")

  elif args.compare:
    print("Comparing results against references...")
    mismatches = 0
    failures = 0

    for pid in sorted(results.keys()):
      res = results[pid]
      ref_file = os.path.join(REF_DIR, f"prompt_{pid}.json")

      if not os.path.isfile(ref_file):
        print(f"ERROR: Reference file for prompt {pid} is missing!")
        failures += 1
        continue

      with open(ref_file, "r") as f:
        ref_res = json.load(f)

      if res["status"] != "success":
        print(f"Prompt {pid} failed at runtime: {res.get('error')}")
        failures += 1
        continue

      if ref_res["status"] != "success":
        print(f"Prompt {pid} reference was a failure. Skipping comparison.")
        continue

      if res["text"] != ref_res["text"]:
        print(f"MISMATCH for Prompt {pid}!")
        print(f"  Reference text: {ref_res['text']!r}")
        print(f"  Current text:   {res['text']!r}")
        mismatches += 1

    if mismatches > 0 or failures > 0:
      print(
          f"Numeric equivalence failed. {mismatches} mismatches, {failures}"
          " failures.",
          file=sys.stderr,
      )
      sys.exit(1)
    else:
      print("Numeric equivalence passed! All 100 prompts match exactly.")
      sys.exit(0)


if __name__ == "__main__":
  main()
