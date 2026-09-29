#!/usr/bin/env python3
# LLMWiki toolset for MaxKernel autonomous TPU kernel optimization (Google3).

import argparse
import datetime
import glob
import logging
import os
import re
import shutil
import subprocess
import sys
from typing import List, Optional

# Resolve WIKI_DIR dynamically inside Google3 or standalone environment
DEFAULT_WIKI_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wiki"
)
WIKI_DIR = os.environ.get("WIKI_DIR", DEFAULT_WIKI_DIR)
DEFAULT_MODE = os.environ.get("WIKI_MODE", "full")

SUPPRESSED_PATTERNS = ["classes", "briefs", "tokamax", "distilled"]


def _is_suppressed(path_or_str: str) -> bool:
  path_lower = path_or_str.lower()
  return any(p in path_lower for p in SUPPRESSED_PATTERNS)


def _sanitize_path(relative_path: str) -> Optional[str]:
  clean_rel = relative_path.strip("/").strip("\\")
  full_path = os.path.abspath(os.path.join(WIKI_DIR, clean_rel))
  if not full_path.startswith(os.path.abspath(WIKI_DIR)):
    return None
  return full_path


def _python_wiki_search(
    query: str, search_dir: str, mode: str = DEFAULT_MODE, max_matches: int = 5
) -> str:
  terms = [t.lower() for t in query.split() if len(t) > 1]
  if not terms:
    terms = [query.lower()]

  results = []
  for root, _, files in os.walk(search_dir):
    for f in sorted(files):
      if not f.endswith(".md"):
        continue

      if mode == "suppressed" and (_is_suppressed(root) or _is_suppressed(f)):
        continue

      full_path = os.path.join(root, f)
      rel_path = os.path.relpath(full_path, WIKI_DIR)
      try:
        with open(full_path, "r", encoding="utf-8", errors="ignore") as fp:
          lines = fp.readlines()
      except Exception:
        continue

      matched_blocks = []
      for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(term in line_lower for term in terms):
          start = max(0, i - 1)
          end = min(len(lines), i + 3)
          snippet = "".join(lines[start:end]).strip()
          matched_blocks.append("  Line " + str(i + 1) + ": " + snippet)
          if len(matched_blocks) >= 2:
            break

      if matched_blocks:
        formatted = (
            "### ["
            + rel_path
            + "](wiki/"
            + rel_path
            + ")\n"
            + "\n".join(matched_blocks)
        )
        results.append(formatted)
        if len(results) >= max_matches:
          break

    if len(results) >= max_matches:
      break

  return "\n\n".join(results)


def query_wiki(
    query: str, category: str = "all", mode: str = DEFAULT_MODE
) -> dict:
  if not os.path.exists(WIKI_DIR):
    return {
        "status": "error",
        "message": "Wiki directory not found at: " + str(WIKI_DIR),
    }

  search_dir = (
      WIKI_DIR if category == "all" else os.path.join(WIKI_DIR, category)
  )
  if not os.path.exists(search_dir):
    search_dir = WIKI_DIR

  # 1. Tier 1: Ripgrep with strict suppression glob filtering
  if shutil.which("rg"):
    cmd = [
        "rg",
        "-i",
        "-C",
        "2",
        "--max-count",
        "3",
        "--heading",
    ]
    if mode == "suppressed":
      for p in SUPPRESSED_PATTERNS:
        cmd.extend(["--glob", f"!**/{p}/**", "--glob", f"!**/*{p}*"])

    cmd.extend([query, search_dir])
    try:
      res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
      if res.returncode == 0 and res.stdout.strip():
        output = res.stdout.replace(WIKI_DIR.rstrip("/") + "/", "")
        return {
            "status": "success",
            "tier": "Tier 1 (Ripgrep Exact Match)",
            "results": output[:4000],
            "message": "Found direct matches in wiki for " + str(query),
        }
    except Exception as e:
      logging.debug(f"Ripgrep execution skipped: {e}")

  # 2. Tier 2: Pure-Python Lexical Matcher with suppression
  py_results = _python_wiki_search(query, search_dir, mode=mode)
  if py_results.strip():
    return {
        "status": "success",
        "tier": "Tier 2 (Lexical Token Matcher)",
        "results": py_results[:4000],
        "message": "Found direct matches in wiki for " + str(query),
    }

  # 3. Tier 3: Index fallback
  index_path = os.path.join(WIKI_DIR, "kernel-optimization-index.md")
  if os.path.exists(index_path):
    try:
      with open(index_path, "r", encoding="utf-8") as f:
        index_content = f.read()

      paragraphs = index_content.split("## ")
      matched = [
          "## " + p
          for p in paragraphs
          if any(t in p.lower() for t in query.lower().split())
      ]
      if matched:
        return {
            "status": "success",
            "tier": "Tier 3 (Master Index Fallback)",
            "results": "\n\n".join(matched[:3])[:4000],
            "message": "Matched index sections for " + str(query),
        }

      return {
          "status": "success",
          "tier": "Tier 3 (Root Index Fallback)",
          "results": index_content[:2500],
          "message": (
              "No direct match for "
              + str(query)
              + ". Provided root kernel optimization index."
          ),
      }
    except Exception as e:
      logging.debug(f"Index read error: {e}")

  return {
      "status": "error",
      "message": "No matches found for " + str(query) + " in wiki.",
  }


def read_wiki_page(
    file_path: str, max_chars: int = 8000, mode: str = DEFAULT_MODE
) -> dict:
  if mode == "suppressed" and _is_suppressed(file_path):
    return {
        "status": "error",
        "message": (
            f"Access to distilled / tokamax page suppressed in mode: {mode}"
        ),
    }

  full_path = _sanitize_path(file_path)
  if not full_path or not os.path.exists(full_path):
    matches = glob.glob(
        WIKI_DIR + "/**/" + os.path.basename(file_path), recursive=True
    )
    if matches:
      full_path = matches[0]
      if mode == "suppressed" and _is_suppressed(full_path):
        return {
            "status": "error",
            "message": (
                f"Access to distilled / tokamax page suppressed in mode: {mode}"
            ),
        }
    else:
      return {
          "status": "error",
          "message": "Wiki file not found: " + str(file_path),
      }

  try:
    with open(full_path, "r", encoding="utf-8", errors="ignore") as fp:
      content = fp.read()
    rel_path = os.path.relpath(full_path, WIKI_DIR)
    return {
        "status": "success",
        "file_path": rel_path,
        "content": content[:max_chars],
        "truncated": len(content) > max_chars,
    }
  except Exception as e:
    return {"status": "error", "message": "Failed to read wiki file: " + str(e)}


def get_wiki_index() -> dict:
  index_file = os.path.join(WIKI_DIR, "kernel-optimization-index.md")
  if not os.path.exists(index_file):
    index_file = os.path.join(WIKI_DIR, "index.md")
  if not os.path.exists(index_file):
    return {"status": "error", "message": "No index file found in wiki."}

  try:
    with open(index_file, "r", encoding="utf-8", errors="ignore") as fp:
      return {"status": "success", "content": fp.read()[:8000]}
  except Exception as e:
    return {"status": "error", "message": "Failed to read index: " + str(e)}


def record_wiki_observation(
    title: str, content: str, author: str = "MaxKernel"
) -> dict:
  obs_dir = os.path.join(WIKI_DIR, "observations")
  os.makedirs(obs_dir, exist_ok=True)
  slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
  timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  fname = slug + "-" + timestamp + ".md"
  fpath = os.path.join(obs_dir, fname)

  doc = (
      "---\ntitle: "
      + title
      + "\nauthor: "
      + author
      + "\ndate: "
      + datetime.datetime.now().strftime("%Y-%m-%d")
      + "\n---\n\n# "
      + title
      + "\n\n"
      + content
      + "\n"
  )
  try:
    with open(fpath, "w", encoding="utf-8") as fp:
      fp.write(doc)
    return {"status": "success", "file_path": "observations/" + fname}
  except Exception as e:
    return {
        "status": "error",
        "message": "Failed to write observation: " + str(e),
    }


def main():
  parser = argparse.ArgumentParser(
      description="Query the MaxKernel LLMWiki knowledge base."
  )
  subparsers = parser.add_subparsers(
      dest="command", help="Commands: query, read, index, record"
  )

  q_parser = subparsers.add_parser(
      "query", help="Search the wiki for concepts or formulas"
  )
  q_parser.add_argument("query", help="The search query")
  q_parser.add_argument(
      "--category", default="all", help="Subdirectory to search in"
  )
  q_parser.add_argument(
      "--mode",
      default=DEFAULT_MODE,
      choices=["full", "suppressed"],
      help="Wiki mode",
  )

  r_parser = subparsers.add_parser("read", help="Read a specific wiki document")
  r_parser.add_argument("file_path", help="Relative path to the markdown file")
  r_parser.add_argument(
      "--mode",
      default=DEFAULT_MODE,
      choices=["full", "suppressed"],
      help="Wiki mode",
  )

  subparsers.add_parser(
      "index", help="Print the master kernel optimization index"
  )

  rec_parser = subparsers.add_parser("record", help="Record a new observation")
  rec_parser.add_argument("title", help="Observation title")
  rec_parser.add_argument("content", help="Markdown content")

  args = parser.parse_args()

  if args.command == "query":
    res = query_wiki(args.query, category=args.category, mode=args.mode)
    if res.get("status") == "success":
      print("[" + str(res.get("tier", "Search Match")) + "]:")
      print(res.get("results", ""))
    else:
      print("Error: " + str(res.get("message", "No matches")))
  elif args.command == "read":
    res = read_wiki_page(args.file_path, mode=args.mode)
    if res.get("status") == "success":
      print(res.get("content", ""))
    else:
      print("Error: " + str(res.get("message", "File read error")))
  elif args.command == "index":
    res = get_wiki_index()
    print(res.get("content", ""))
  elif args.command == "record":
    res = record_wiki_observation(args.title, args.content)
    print(res)
  else:
    parser.print_help()


if __name__ == "__main__":
  main()
