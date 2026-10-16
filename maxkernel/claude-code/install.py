#!/usr/bin/env python3
"""Install MaxKernel's skill, subagents and (optionally) guard hooks into Claude Code.

Nothing in this repository is machine-specific: the prompt files under
``agents/`` and ``skill/`` are templates containing ``{{PLACEHOLDER}}`` tokens.
This script resolves those tokens against *your* paths and writes the rendered
result into your Claude Code config directory.

    python3 install.py                 # install for the current user
    python3 install.py --with-guard    # also install the workspace guard hooks
    python3 install.py --dry-run       # show what would happen, change nothing
    python3 install.py --uninstall     # remove everything this script installed

Re-run it any time you edit a template, or after `git pull`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent
SKILL_NAME = "maxkernel"
AGENT_GLOB = "maxkernel-*.md"

# Steps maxkernel-worker now follows inline, from the rendered copies under
# skills/maxkernel/references/maxkernel-worker/. Earlier versions installed
# each as a subagent; remove those so a stale prompt cannot be dispatched.
RETIRED_AGENTS = [
  "maxkernel-analyze-torch-source.md",
  "maxkernel-autotune-planner.md",
  "maxkernel-autotune-summary.md",
  "maxkernel-compilation-summary.md",
  "maxkernel-fix-port.md",
  "maxkernel-fix-test-script.md",
  "maxkernel-generate-test-file.md",
  "maxkernel-reconcile-reference.md",
  "maxkernel-summarize-test-results.md",
  "maxkernel-synthesize-baseline.md",
  "maxkernel-test-script-validation-summary.md",
  "maxkernel-write-jnp-reference.md",
]

PLACEHOLDER = re.compile(r"\{\{([A-Z_]+)\}\}")

# The PreToolUse guard is opt-in; these are the settings.json edits it needs.
HOOK_COMMAND_MARKER = "workspace-guard.py"
GUARD_DENY_RULES = [
  "Read(//**/pallas/**)",
  "Read(//**/jax/experimental/**)",
  "Read(//**/jax/_src/**)",
  "Read(//**/site-packages/**)",
  "Read(//**/dist-packages/**)",
  "Edit(//**/site-packages/**)",
]

# Claude Code applies path-pattern permission rules only to the file-reading
# and file-writing tools -- a Read(...) rule already covers Grep and Glob, and
# an explicit Grep(...) or Glob(...) rule is ignored with a warning on every
# launch. Earlier versions of this installer wrote both, so strip them from
# settings.json rather than just dropping them from the list above; otherwise
# they linger in every existing install. workspace-guard.py blocks those tools
# itself, so nothing is lost.
STALE_DENY_RULES = [
  "Grep(//**/pallas/**)",
  "Glob(//**/pallas/**)",
]


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def build_vars(claude_dir: pathlib.Path, venv: pathlib.Path) -> dict[str, str]:
  return {
    "HOME": str(pathlib.Path.home()),
    "MAXKERNEL_ROOT": str(REPO),
    "CLAUDE_DIR": str(claude_dir),
    "VENV": str(venv),
    "VENV_NAME": venv.name,
    "VENV_PYTHON": str(venv / "bin" / "python"),
  }


def render(text: str, variables: dict[str, str], source: pathlib.Path) -> str:
  unknown: set[str] = set()

  def sub(match: re.Match) -> str:
    key = match.group(1)
    if key not in variables:
      unknown.add(key)
      return match.group(0)
    return variables[key]

  out = PLACEHOLDER.sub(sub, text)
  if unknown:
    raise SystemExit(
      f"{source}: unknown placeholder(s) {sorted(unknown)}. "
      "Add them to build_vars() or fix the template."
    )
  return out


# --------------------------------------------------------------------------
# small fs helpers
# --------------------------------------------------------------------------


class Runner:
  """Applies (or, with --dry-run, only narrates) filesystem changes."""

  def __init__(self, dry_run: bool) -> None:
    self.dry_run = dry_run
    self.changed = 0

  def say(self, verb: str, target) -> None:
    prefix = "would " if self.dry_run else ""
    print(f"  {prefix}{verb:<9} {target}")

  def write(self, path: pathlib.Path, text: str) -> None:
    if path.exists() and path.read_text() == text:
      self.say("unchanged", path)
      return
    self.say("write", path)
    if not self.dry_run:
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text(text)
    self.changed += 1

  def remove(self, path: pathlib.Path) -> None:
    if not path.exists():
      return
    self.say("remove", path)
    if not self.dry_run:
      shutil.rmtree(path) if path.is_dir() else path.unlink()
    self.changed += 1

  def mkdir(self, path: pathlib.Path) -> None:
    if path.is_dir():
      return
    self.say("mkdir", path)
    if not self.dry_run:
      path.mkdir(parents=True, exist_ok=True)
    self.changed += 1

  def backup(self, path: pathlib.Path) -> None:
    if not path.exists():
      return
    dest = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
    self.say("backup", dest)
    if not self.dry_run:
      shutil.copy2(path, dest)


# --------------------------------------------------------------------------
# install steps
# --------------------------------------------------------------------------


def install_prompts(run: Runner, claude_dir: pathlib.Path, variables) -> None:
  print("Subagents and skill:")
  agents_dir = claude_dir / "agents"
  skill_dir = claude_dir / "skills" / SKILL_NAME

  sources = sorted((REPO / "agents").glob(AGENT_GLOB))
  if not sources:
    raise SystemExit(f"No agent templates found under {REPO / 'agents'}.")
  for src in sources:
    run.write(agents_dir / src.name, render(src.read_text(), variables, src))
  for name in RETIRED_AGENTS:
    run.remove(agents_dir / name)

  for name in ("SKILL.md", "general_rules.md"):
    src = REPO / "skill" / name
    run.write(skill_dir / name, render(src.read_text(), variables, src))

  for src in sorted((REPO / "skill" / "references").rglob("*.md")):
    rel = src.relative_to(REPO / "skill")
    run.write(skill_dir / rel, render(src.read_text(), variables, src))


def install_workspace(run: Runner) -> None:
  print("Workspace and TPU config:")
  run.mkdir(REPO / "workspace")
  config = REPO / "tpu_config.json"
  if config.exists():
    run.say("keep", f"{config} (already configured)")
  else:
    example = REPO / "tpu_config.example.json"
    run.write(config, example.read_text())


def install_guard(run: Runner, claude_dir: pathlib.Path, variables) -> None:
  print("Workspace guard hooks:")
  hooks_dir = claude_dir / "hooks"
  for src in sorted((REPO / "hooks").glob("*.py")):
    run.write(hooks_dir / src.name, render(src.read_text(), variables, src))

  settings_path = claude_dir / "settings.json"
  settings = {}
  if settings_path.exists():
    try:
      settings = json.loads(settings_path.read_text())
    except json.JSONDecodeError as exc:
      raise SystemExit(
        f"{settings_path} is not valid JSON ({exc}); fix it first."
      )

  guard = str(hooks_dir / "workspace-guard.py")
  entry = {
    "matcher": "*",
    "hooks": [
      {
        "type": "command",
        "command": f"python3 {guard} || exit 2",
        "timeout": 10,
        "statusMessage": "Checking workspace boundary...",
      }
    ],
  }

  pre = settings.setdefault("hooks", {}).setdefault("PreToolUse", [])
  pre = [e for e in pre if HOOK_COMMAND_MARKER not in json.dumps(e)]
  pre.append(entry)
  settings["hooks"]["PreToolUse"] = pre

  deny = settings.setdefault("permissions", {}).setdefault("deny", [])
  deny[:] = [rule for rule in deny if rule not in STALE_DENY_RULES]
  # Claude Code spells an absolute path in a permission rule with a leading
  # "//", and VENV already starts with "/" -- hence the single slash here.
  for rule in GUARD_DENY_RULES + [
    f"Read(/{variables['VENV']}/**)",
    f"Edit(/{variables['VENV']}/**)",
  ]:
    if rule not in deny:
      deny.append(rule)

  run.backup(settings_path)
  run.write(settings_path, json.dumps(settings, indent=2) + "\n")


def uninstall(run: Runner, claude_dir: pathlib.Path) -> None:
  print("Removing installed files:")
  for path in sorted((claude_dir / "agents").glob(AGENT_GLOB)):
    run.remove(path)
  run.remove(claude_dir / "skills" / SKILL_NAME)
  for name in ("workspace-guard.py", "guard-tests.py"):
    run.remove(claude_dir / "hooks" / name)

  settings_path = claude_dir / "settings.json"
  if not settings_path.exists():
    return
  try:
    settings = json.loads(settings_path.read_text())
  except json.JSONDecodeError:
    print(f"  skipped   {settings_path} (not valid JSON; edit it by hand)")
    return

  pre = settings.get("hooks", {}).get("PreToolUse", [])
  kept = [e for e in pre if HOOK_COMMAND_MARKER not in json.dumps(e)]
  if len(kept) != len(pre):
    settings["hooks"]["PreToolUse"] = kept
    if not kept:
      del settings["hooks"]["PreToolUse"]
    if not settings["hooks"]:
      del settings["hooks"]
    run.backup(settings_path)
    run.write(settings_path, json.dumps(settings, indent=2) + "\n")
  print(
    "  note      permission deny-rules were left in settings.json; "
    "remove them by hand if you no longer want them."
  )


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------


def preflight(venv: pathlib.Path) -> None:
  print("Environment check:")
  python = venv / "bin" / "python"
  if not python.exists():
    print(f"  MISSING   {python}")
    print("            The agents run every tool through this interpreter.")
    print(
      f"            Create it:  python3.12 -m venv {venv} && "
      f"{python} -m pip install -r {REPO / 'requirements.txt'}"
    )
    print("            See the Install section of README.md.")
    return
  print(f"  ok        {python}")

  config = REPO / "tpu_config.json"
  example = REPO / "tpu_config.example.json"
  if config.exists() and config.read_text() != example.read_text():
    print(f"  ok        {config}")
  else:
    print(
      f"  todo      {config} still holds the example (local-TPU) config. "
      "Register your TPU with"
    )
    print(
      "            tools/tpu_client.py --add_tpu — see docs/tpu-setup.md, Part 1."
    )


# --------------------------------------------------------------------------


def main() -> None:
  default_claude = pathlib.Path(
    os.environ.get("CLAUDE_CONFIG_DIR", pathlib.Path.home() / ".claude")
  )

  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
    "--claude-dir",
    type=pathlib.Path,
    default=default_claude,
    help=f"Claude Code config directory (default: {default_claude})",
  )
  parser.add_argument(
    "--venv",
    type=pathlib.Path,
    default=pathlib.Path.home() / "maxkernel_venv",
    help="Python virtualenv the agents run tools with "
    "(default: ~/maxkernel_venv)",
  )
  parser.add_argument(
    "--with-guard",
    action="store_true",
    help="also install the PreToolUse workspace guard and "
    "wire it into settings.json",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="print the changes without making them",
  )
  parser.add_argument(
    "--uninstall",
    action="store_true",
    help="remove the skill, subagents and guard hooks",
  )
  args = parser.parse_args()

  claude_dir = args.claude_dir.expanduser().resolve()
  venv = args.venv.expanduser().resolve()
  run = Runner(args.dry_run)

  if args.uninstall:
    uninstall(run, claude_dir)
    print(
      f"\nDone ({run.changed} change(s)). Restart Claude Code to pick it up."
    )
    return

  variables = build_vars(claude_dir, venv)
  print(f"MaxKernel repo:   {REPO}")
  print(f"Claude Code dir:  {claude_dir}")
  print(f"Python venv:      {venv}\n")

  install_prompts(run, claude_dir, variables)
  install_workspace(run)
  if args.with_guard:
    install_guard(run, claude_dir, variables)
  print()
  preflight(venv)

  print(f"\nDone ({run.changed} change(s)).")
  if not args.dry_run:
    print(
      "Restart Claude Code, then ask it to optimize a Pallas TPU kernel — "
      "or invoke the skill by name with /maxkernel."
    )


if __name__ == "__main__":
  sys.exit(main())
