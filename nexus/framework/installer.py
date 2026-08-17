#!/usr/bin/env python3
"""Installer module for Coworker framework.

Handles installing, updating, and cleanly uninstalling translated distributions
in destination workspaces with atomic ownership tracking and rollback support.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Any, Sequence

try:
  from accelerator_agents.tpu_nexus.framework import utils
except ImportError:
  try:
    from . import utils
  except ImportError:
    import utils

Path = pathlib.Path

CoworkerError = utils.CoworkerError
FRAMEWORK_VERSION = utils.FRAMEWORK_VERSION
JsonConfigManager = utils.JsonConfigManager
SemVer = utils.SemVer
TomlConfigManager = utils.TomlConfigManager
compute_digest = utils.compute_digest
load_json = utils.load_json
rooted = utils.rooted
safe_relative = utils.safe_relative
validate_instance = utils.validate_instance
validate_package_name = utils.validate_package_name
version_satisfies = utils.version_satisfies
write_json = utils.write_json


# =====================================================================
# Environment Prompting & Question Coercion
# =====================================================================


def coerce_answer_value(question: dict[str, Any], raw: Any) -> Any:
  """Coerce raw input into the declared question type and validate choices."""
  key = question["name"]
  if question.get("type") == "boolean" and isinstance(raw, str):
    if raw.lower() not in {"yes", "no", "true", "false", "y", "n"}:
      raise CoworkerError(f"{key} must be yes/no")
    raw = raw.lower() in {"yes", "true", "y"}

  if question.get("type") == "integer" and isinstance(raw, str):
    try:
      raw = int(raw)
    except ValueError as exc:
      raise CoworkerError(f"{key} must be an integer") from exc

  if "choices" in question and raw not in question["choices"]:
    raise CoworkerError(f"{key} must be one of {question['choices']}")

  return raw


def prompt_question_value(
    question: dict[str, Any],
    supplied: dict[str, Any],
    current: dict[str, Any],
    non_interactive: bool,
    destination: Path,
) -> Any:
  """Resolve the value for an environment question interactively or from answers."""
  key = question["name"]
  if question.get("value_from") == "destination":
    return str(destination.resolve())

  if key in supplied:
    return coerce_answer_value(question, supplied[key])

  fallback = current.get(key, question.get("default"))

  if non_interactive:
    if fallback is None:
      raise CoworkerError(f"missing required answer: {key}")
    return coerce_answer_value(question, fallback)

  choices = (
      f" ({' | '.join(map(str, question['choices']))})"
      if "choices" in question
      else ""
  )
  suffix = f" [{fallback}]" if fallback is not None else ""
  entered = input(f"{question['prompt']}{choices}{suffix}: ").strip()

  if not entered and fallback is None:
    raise CoworkerError(f"missing required answer: {key}")

  return coerce_answer_value(question, entered if entered else fallback)


prompt_value = prompt_question_value


def validate_question_constraints(
    questions: list[dict[str, Any]], answers: dict[str, Any]
) -> None:
  """Validate filesystem and relational constraints on environment answers."""
  for question in questions:
    value = answers[question["name"]]
    if question.get("relative_path") and Path(str(value)).is_absolute():
      raise CoworkerError(f"{question['name']} must be relative")

    if "must_exist" in question:
      candidate = Path(str(value))
      if "relative_to" in question:
        candidate = Path(str(answers[question["relative_to"]])) / candidate
      expected = question["must_exist"]
      if expected == "file" and not candidate.is_file():
        raise CoworkerError(
            f"{question['name']} does not identify a file: {candidate}"
        )
      if expected == "directory" and not candidate.is_dir():
        raise CoworkerError(
            f"{question['name']} does not identify a directory: {candidate}"
        )


# =====================================================================
# Harness Detection & File Expansion
# =====================================================================


def check_harness(plan: dict[str, Any]) -> str:
  """Execute harness version command and assert compatibility against plan version range."""
  try:
    result = subprocess.run(
        plan["version_command"],
        text=True,
        capture_output=True,
        check=True,
    )
  except (OSError, subprocess.CalledProcessError) as exc:
    raise CoworkerError(f"cannot detect {plan['harness']}: {exc}") from exc

  output = (result.stdout or result.stderr).strip()
  if not version_satisfies(SemVer.parse(output), plan["version_range"]):
    raise CoworkerError(
        f"installed {plan['harness']} version {output!r} is outside"
        f" {plan['version_range']}"
    )
  return output


def expand_copies(
    distribution: Path,
    destination: Path,
    plan: dict[str, Any],
) -> list[tuple[Path, Path]]:
  """Expand copy specifications into explicit (source_file, dest_file) pairs."""
  expanded: list[tuple[Path, Path]] = []
  seen: set[Path] = set()

  for mapping in plan["copies"]:
    source_rel = safe_relative(mapping["source"], "copy source")
    destination_rel = safe_relative(mapping["destination"], "copy destination")

    source = (
        distribution
        if str(source_rel) == "."
        else rooted(distribution, str(source_rel), "copy source")
    )
    target = rooted(destination, str(destination_rel), "copy destination")

    if source.is_file():
      pairs = [(source, target)]
    elif source.is_dir():
      pairs = [
          (item, target / item.relative_to(source))
          for item in source.rglob("*")
          if item.is_file()
      ]
    else:
      raise CoworkerError(f"install source does not exist: {source}")

    for _, installed in pairs:
      if installed in seen:
        raise CoworkerError(
            f"install plan writes the same file twice: {installed}"
        )
      seen.add(installed)

    expanded.extend(pairs)

  return expanded


# =====================================================================
# Ownership Management & Rollback
# =====================================================================


def clean_empty_directories(root: Path) -> None:
  """Remove empty directories under root in bottom-up order."""
  for directory in sorted(
      (p for p in root.rglob("*") if p.is_dir()), reverse=True
  ):
    try:
      directory.rmdir()
    except OSError:
      pass


def check_action_modification(
    destination: Path, action: dict[str, Any]
) -> str | None:
  """Check if an ownership action target has been modified by the user."""
  path = rooted(destination, action["path"], "owned path")
  kind = action["kind"]

  if kind == "file":
    if path.is_file() and compute_digest(path) != action["sha256"]:
      return action["path"]
  elif kind == "json_value" and path.is_file():
    exists, value = JsonConfigManager.get_at_path(
        load_json(path), action["keys"]
    )
    if exists and value != action["value"]:
      return f"{action['path']}:{'/'.join(action['keys'])}"
  elif kind == "json_container" and path.is_file():
    exists, value = JsonConfigManager.get_at_path(
        load_json(path), action["keys"]
    )
    if exists and not isinstance(value, dict):
      return f"{action['path']}:{'/'.join(action['keys'])}"
  elif kind == "toml_value" and path.is_file():
    lines = path.read_text(encoding="utf-8").splitlines()
    _, value = TomlConfigManager.find_key(
        lines, action["section"], action["key"]
    )
    if value is not None and value != action["value"]:
      return f"{action['path']}:[{action['section']}]/{action['key']}"

  return None


def detect_owned_modifications(
    destination: Path, ownership: dict[str, Any]
) -> list[str]:
  """Detect any files or configuration entries modified since installation."""
  modified: list[str] = []
  for action in ownership.get("actions", []):
    mod = check_action_modification(destination, action)
    if mod:
      modified.append(mod)
  return modified


def rollback_json_action(
    path: Path,
    keys: Sequence[str],
    expected_val: Any,
    is_container: bool = False,
) -> None:
  """Roll back an added JSON key or container from a config document."""
  if not path.is_file():
    return
  document = load_json(path)
  exists, value = JsonConfigManager.get_at_path(document, keys)
  if not exists:
    return
  if (is_container and not value) or (
      not is_container and value == expected_val
  ):
    parent: Any = document
    for key in keys[:-1]:
      if isinstance(parent, dict) and key in parent:
        parent = parent[key]
      else:
        parent = None
        break
    if isinstance(parent, dict) and keys[-1] in parent:
      del parent[keys[-1]]
      if document:
        write_json(path, document)
      else:
        path.unlink(missing_ok=True)


def rollback_toml_action(
    path: Path,
    section: str,
    key: str,
    expected_val: Any,
    section_created: bool = False,
) -> None:
  """Roll back an added TOML key and optional section from a config file."""
  if not path.is_file():
    return
  lines = path.read_text(encoding="utf-8").splitlines()
  index, value = TomlConfigManager.find_key(lines, section, key)
  if index is not None and value == expected_val:
    del lines[index]
    if section_created:
      section_index = next(
          (i for i, line in enumerate(lines) if line.strip() == f"[{section}]"),
          None,
      )
      if section_index is not None:
        next_section = next(
            (
                i
                for i in range(section_index + 1, len(lines))
                if re.match(r"^\s*\[[^]]+]\s*$", lines[i])
            ),
            len(lines),
        )
        if not any(
            line.strip() and not line.lstrip().startswith("#")
            for line in lines[section_index + 1 : next_section]
        ):
          del lines[section_index:next_section]

    content = "\n".join(lines).strip()
    if content:
      path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
      path.unlink(missing_ok=True)


def rollback_action(destination: Path, action: dict[str, Any]) -> None:
  """Revert a single recorded installation action if unchanged."""
  path = rooted(destination, action["path"], "owned path")
  kind = action["kind"]

  if kind == "file":
    if path.is_file() and compute_digest(path) == action["sha256"]:
      path.unlink()
  elif kind == "json_value":
    rollback_json_action(
        path, action["keys"], action["value"], is_container=False
    )
  elif kind == "json_container":
    rollback_json_action(path, action["keys"], {}, is_container=True)
  elif kind == "toml_value":
    rollback_toml_action(
        path,
        action["section"],
        action["key"],
        action["value"],
        action.get("section_created", False),
    )


def remove_owned(destination: Path, ownership: dict[str, Any]) -> list[str]:
  """Remove files and rollback config settings tracked in ownership record."""
  modified = detect_owned_modifications(destination, ownership)

  for action in reversed(ownership.get("actions", [])):
    rollback_action(destination, action)

  clean_empty_directories(destination)
  return modified


# =====================================================================
# Install & Uninstall Operations
# =====================================================================


def validate_install_preconditions(
    destination: Path,
    previous_ownership: dict[str, Any] | None,
    upgrade: bool,
) -> None:
  """Validate destination state and upgrade requirements before installation."""
  if destination == Path(destination.anchor):
    raise CoworkerError("refusing to install into a filesystem root")

  if previous_ownership and not upgrade:
    raise CoworkerError(
        f"installation already exists in {destination}; use --upgrade"
    )
  if upgrade and not previous_ownership:
    raise CoworkerError(f"cannot upgrade missing installation in {destination}")
  if previous_ownership:
    modified = detect_owned_modifications(destination, previous_ownership)
    if modified:
      raise CoworkerError(
          "refusing upgrade because managed content changed:\n"
          + "\n".join(modified)
      )


def check_file_collisions(
    copies: Sequence[tuple[Path, Path]],
    destination: Path,
    previous_ownership: dict[str, Any] | None,
) -> None:
  """Ensure newly copied files do not overwrite unmanaged user files."""
  old_files = {
      action["path"]
      for action in (previous_ownership or {}).get("actions", [])
      if action["kind"] == "file"
  }
  collisions = [
      str(target)
      for _, target in copies
      if target.exists()
      and str(target.relative_to(destination)) not in old_files
  ]
  if collisions:
    raise CoworkerError(
        "refusing to overwrite existing files:\n" + "\n".join(collisions)
    )


def prevalidate_configuration_mutations(
    destination: Path, plan: dict[str, Any]
) -> None:
  """Verify that JSON merges and TOML settings can be applied without conflict."""
  for merge in plan.get("json_merges", []):
    path = rooted(destination, merge["path"], "JSON config path")
    document = load_json(path) if path.exists() else {}
    JsonConfigManager.merge(document, merge["value"], [], [])

  for setting in plan.get("toml_sets", []):
    path = rooted(destination, setting["path"], "TOML config path")
    lines = (
        path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    )
    _, existing = TomlConfigManager.find_key(
        lines, setting["section"], setting["key"]
    )
    if existing is not None and existing != TomlConfigManager.render_value(
        setting["value"]
    ):
      raise CoworkerError(
          f"refusing to replace existing TOML setting: [{setting['section']}]"
          f" {setting['key']}"
      )


def collect_environment_answers(
    distribution: Path,
    destination: Path,
    plan: dict[str, Any],
    answers_path: Path | None,
    previous_environment: dict[str, Any],
    non_interactive: bool,
) -> dict[str, Any]:
  """Prompt, validate, and return environment configuration answers."""
  spec = load_json(
      rooted(
          distribution, plan["environment_questions"], "environment questions"
      )
  )
  supplied = load_json(answers_path) if answers_path else {}
  unknown_answers = set(supplied) - {
      question["name"] for question in spec["questions"]
  }
  if unknown_answers:
    raise CoworkerError(
        f"unknown answer keys: {', '.join(sorted(unknown_answers))}"
    )

  current = {
      key: value
      for key, value in previous_environment.items()
      if key != "revision"
  }
  answers = {
      q["name"]: prompt_question_value(
          q, supplied, current, non_interactive, destination
      )
      for q in spec["questions"]
  }
  validate_question_constraints(spec["questions"], answers)

  env_schema = load_json(
      rooted(distribution, plan["environment_schema"], "environment schema")
  )
  errors = validate_instance(answers, env_schema)
  if errors:
    raise CoworkerError("invalid environment:\n" + "\n".join(errors))

  return answers


def install(
    distribution: Path,
    destination: Path,
    answers_path: Path | None = None,
    non_interactive: bool = False,
    upgrade: bool = False,
    skip_harness_check: bool = False,
) -> None:
  """Install or upgrade a translated distribution package into a workspace."""
  build = load_json(distribution / "coworker-build.json")
  plan = load_json(distribution / "coworker-install.json")

  if (
      build["harness"] != plan["harness"]
      or build["version_range"] != plan["version_range"]
  ):
    raise CoworkerError("build metadata and install plan disagree")

  if not skip_harness_check:
    check_harness(plan)

  destination = destination.resolve()
  ownership_path = rooted(destination, plan["ownership_path"], "ownership path")
  environment_path = rooted(
      destination, plan["environment_path"], "environment path"
  )

  previous_ownership = (
      load_json(ownership_path) if ownership_path.exists() else None
  )
  previous_environment = (
      load_json(environment_path) if environment_path.exists() else {}
  )

  validate_install_preconditions(destination, previous_ownership, upgrade)
  answers = collect_environment_answers(
      distribution,
      destination,
      plan,
      answers_path,
      previous_environment,
      non_interactive,
  )

  copies = expand_copies(distribution, destination, plan)
  check_file_collisions(copies, destination, previous_ownership)
  prevalidate_configuration_mutations(destination, plan)

  # Rollback prior installation if upgrading
  if previous_ownership:
    remove_owned(destination, previous_ownership)
    if ownership_path.exists():
      ownership_path.unlink()

  # Perform installation
  destination.mkdir(parents=True, exist_ok=True)
  actions: list[dict[str, Any]] = []

  # Copy files
  for source, target in copies:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    actions.append({
        "kind": "file",
        "path": str(target.relative_to(destination)),
        "sha256": compute_digest(target),
    })

  # Apply JSON merges
  for merge in plan.get("json_merges", []):
    path = rooted(destination, merge["path"], "JSON config path")
    document = load_json(path) if path.exists() else {}
    records: list[dict[str, Any]] = []
    JsonConfigManager.merge(document, merge["value"], [], records)
    write_json(path, document)
    actions.extend({**record, "path": merge["path"]} for record in records)

  # Apply TOML sets
  for setting in plan.get("toml_sets", []):
    path = rooted(destination, setting["path"], "TOML config path")
    record = TomlConfigManager.apply_set(
        path, setting["section"], setting["key"], setting["value"]
    )
    if record:
      actions.append({**record, "path": setting["path"]})

  # Update environment revision
  current = {k: v for k, v in previous_environment.items() if k != "revision"}
  previous_revision = previous_environment.get("revision", 0)
  revision = (
      previous_revision + 1 if current != answers else max(previous_revision, 1)
  )
  write_json(environment_path, {"revision": revision, **answers})
  actions.append({
      "kind": "file",
      "path": str(environment_path.relative_to(destination)),
      "sha256": compute_digest(environment_path),
  })

  # Record ownership
  ownership_path.parent.mkdir(parents=True, exist_ok=True)
  write_json(
      ownership_path,
      {
          "framework_version": FRAMEWORK_VERSION,
          "package": build["source"],
          "version": build["source_version"],
          "harness": build["harness"],
          "compatibility_target": build["compatibility_target"],
          "actions": actions,
      },
  )


def uninstall(destination: Path, package: str) -> list[str]:
  """Uninstall an installed package from destination directory, preserving user-modified files."""
  destination = destination.resolve()
  validate_package_name(package)

  ownership_path = rooted(
      destination, f".coworker/{package}/ownership.json", "ownership path"
  )
  ownership = load_json(ownership_path)
  modified = remove_owned(destination, ownership)
  ownership_path.unlink()

  clean_empty_directories(destination)
  return modified


def build_installer_parser() -> argparse.ArgumentParser:
  """Construct parser for standalone installer commands."""
  parser = argparse.ArgumentParser(
      prog="coworker-installer",
      description="Installation and uninstallation operations for Coworker packages.",
  )
  sub = parser.add_subparsers(dest="command", required=True)

  # install
  p_install = sub.add_parser(
      "install", help="Install translated package into workspace"
  )
  p_install.add_argument(
      "distribution",
      type=Path,
      help="Path to translated distribution directory",
  )
  p_install.add_argument(
      "--destination",
      type=Path,
      required=True,
      help="Workspace destination path",
  )
  p_install.add_argument(
      "--answers", type=Path, help="Path to JSON file with pre-supplied answers"
  )
  p_install.add_argument(
      "--non-interactive",
      action="store_true",
      help="Do not prompt for missing answers",
  )
  p_install.add_argument(
      "--upgrade", action="store_true", help="Upgrade existing installation"
  )
  p_install.add_argument(
      "--skip-harness-check", action="store_true", help=argparse.SUPPRESS
  )

  # uninstall
  p_uninstall = sub.add_parser(
      "uninstall", help="Uninstall package from workspace"
  )
  p_uninstall.add_argument(
      "destination", type=Path, help="Workspace destination path"
  )
  p_uninstall.add_argument(
      "--package", required=True, help="Package name to uninstall"
  )

  return parser


def main(argv: list[str] | None = None) -> int:
  """Main entrypoint for standalone installer operations."""
  parser = build_installer_parser()
  args = parser.parse_args(argv)

  try:
    if args.command == "install":
      install(
          distribution=args.distribution,
          destination=args.destination,
          answers_path=args.answers,
          non_interactive=args.non_interactive,
          upgrade=args.upgrade,
          skip_harness_check=args.skip_harness_check,
      )

    elif args.command == "uninstall":
      modified = uninstall(args.destination, args.package)
      if modified:
        print(
            "Preserved modified managed content:\n" + "\n".join(modified),
            file=sys.stderr,
        )

    return 0
  except CoworkerError as exc:
    print(f"installer: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
