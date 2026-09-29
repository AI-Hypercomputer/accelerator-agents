#!/usr/bin/env python3
"""Checks that every file, tool and subagent a prompt references actually exists.

This repository has already shipped this bug once. Commit `a0f3615` merged a
`SKILL.md` that calls `tools/detect_input_language.py` and a worker that
dispatches `maxkernel-analyze-source`, while `157fc9b` had reverted the agent
away and the tool was never committed at all. Both references were dead on
`main`, and nothing caught it -- a prompt is not compiled, so a path that does
not resolve fails at run time, inside an agent, on a TPU, five minutes into a
run.

This is the cheap check that would have caught it. It resolves:

  * `{{MAXKERNEL_ROOT}}/...` paths in any prompt      -> must exist on disk
  * `maxkernel-*` subagent names in a dispatch        -> must have agents/<name>.md
  * `tools/<name>.py` mentioned anywhere              -> must exist
  * every registered agent's frontmatter              -> name must match filename

Exit codes:
  0  every reference resolves
  1  at least one dead reference
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Files that describe the design rather than drive it. A design document is
# allowed to name something that does not exist yet -- that is what a proposal
# is -- so its dead references are not failures.
ADVISORY = {"docs/", "README.md", "user_guide.md", "wiki/"}

PROMPT_GLOBS = ("agents/maxkernel-*.md", "skill/SKILL.md", "skill/general_rules.md")

RE_MK_PATH = re.compile(r"\{\{MAXKERNEL_ROOT\}\}/([\w./-]+)")
RE_TOOLS = re.compile(r"\btools/([\w-]+\.py)\b")
RE_SUBAGENT = re.compile(r"subagent_type\s*=\s*[\"']([\w-]+)[\"']")
RE_AGENT_BACKTICK = re.compile(r"`(maxkernel-[\w-]+)`")
RE_FRONTMATTER_NAME = re.compile(r"^---\s*\nname:\s*([\w-]+)", re.M)

# Names that appear in prose as a category rather than a dispatch target.
NOT_AGENTS = {"maxkernel-worker.md", "maxkernel-debug_history.md"}

# Paths created at install or run time, so absent from a fresh checkout. A
# prompt referencing one is correct; the file's absence is not a dead link.
GENERATED = {
    "tpu_config.json",      # written by install.py / tpu_client.py --add_tpu
    "workspace",            # run directories
    "maxkernel_debug_history.md",
}


def strip_trailing_punct(path):
    """Paths quoted mid-sentence pick up the sentence's punctuation."""
    return path.rstrip(".,;:)]`\'\"")


def is_advisory(rel):
    return any(rel.startswith(a) for a in ADVISORY)


def known_agents():
    return {p.stem for p in (ROOT / "agents").glob("maxkernel-*.md")}


def check():
    problems = []
    agents = known_agents()

    files = []
    for pattern in PROMPT_GLOBS:
        files.extend(sorted(ROOT.glob(pattern)))

    for path in files:
        rel = str(path.relative_to(ROOT))
        text = path.read_text(errors="replace")

        # 1. {{MAXKERNEL_ROOT}}-rooted paths
        for m in RE_MK_PATH.finditer(text):
            target = strip_trailing_punct(m.group(1))
            # Placeholders inside an example are not real paths.
            if "<" in target or target.endswith("/") or not target:
                continue
            if target.split("/")[0] in GENERATED or target in GENERATED:
                continue
            if not (ROOT / target).exists():
                problems.append(f"{rel}: {{{{MAXKERNEL_ROOT}}}}/{target} does not exist")

        # 2. tools/<name>.py
        for m in RE_TOOLS.finditer(text):
            tool = m.group(1)
            if not (ROOT / "tools" / tool).exists():
                problems.append(f"{rel}: tools/{tool} does not exist")

        # 3. dispatched subagents
        for regex in (RE_SUBAGENT, RE_AGENT_BACKTICK):
            for m in regex.finditer(text):
                name = m.group(1)
                if not name.startswith("maxkernel-"):
                    continue
                if name in NOT_AGENTS:
                    continue
                if name not in agents:
                    problems.append(f"{rel}: dispatches `{name}` but agents/{name}.md does not exist")

    # 4. frontmatter name must match filename
    for path in sorted((ROOT / "agents").glob("maxkernel-*.md")):
        text = path.read_text(errors="replace")
        m = RE_FRONTMATTER_NAME.search(text)
        if not m:
            problems.append(f"agents/{path.name}: no `name:` in frontmatter")
        elif m.group(1) != path.stem:
            problems.append(
                f"agents/{path.name}: frontmatter name `{m.group(1)}` != filename `{path.stem}`"
            )

    return sorted(set(problems))


def main():
    problems = check()
    if problems:
        print(f"{len(problems)} dead reference(s):\n", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print("All prompt references resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
