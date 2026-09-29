#!/usr/bin/env python3
# Standalone CLI for querying the MaxKernel LLMWiki knowledge base in Google3.
# Drop-in replacement for the previous Vertex AI RAG retrieval tool.

import argparse
import os
import sys

# Support relative and absolute imports
tools_dir = os.path.dirname(os.path.abspath(__file__))
if tools_dir not in sys.path:
  sys.path.insert(0, tools_dir)

try:
  from wiki_tool import query_wiki, read_wiki_page, get_wiki_index, DEFAULT_MODE
except ImportError:
  try:
    from experimental.MaxKernel.tools.wiki_tool import query_wiki, read_wiki_page, get_wiki_index, DEFAULT_MODE
  except ImportError:
    from third_party.py.accelerator_agents.MaxKernel.tools.wiki_tool import query_wiki, read_wiki_page, get_wiki_index, DEFAULT_MODE


def retrieve(
    query: str,
    category: str = "all",
    mode: str = DEFAULT_MODE,
):
  """Runs a 3-tiered retrieval query against the LLMWiki knowledge base."""
  return query_wiki(query=query, category=category, mode=mode)


def main():
  parser = argparse.ArgumentParser(
      description="Query the MaxKernel LLMWiki knowledge base."
  )
  parser.add_argument(
      "query", help="Text query to retrieve relevant context for."
  )
  parser.add_argument(
      "--category", default="all", help="Subfolder category to search in."
  )
  parser.add_argument(
      "--mode",
      default=DEFAULT_MODE,
      choices=["full", "suppressed"],
      help="LLMWiki mode.",
  )

  args = parser.parse_args()

  result = retrieve(query=args.query, category=args.category, mode=args.mode)
  if result.get("status") == "success":
    print("[" + str(result.get("tier", "LLMWiki Match")) + "]:")
    print(result.get("results", ""))
  else:
    print("Error: " + str(result.get("message", "No matches found.")))


if __name__ == "__main__":
  main()

