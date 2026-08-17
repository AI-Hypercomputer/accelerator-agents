#!/usr/bin/env python3
"""Compiler module for Coworker framework.

Handles package manifest verification, structural integrity checks, target
harness compatibility validation, document generation, and compilation into
distribution formats (Claude Code and Codex).
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import sys
from typing import Any, Callable, Sequence

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
GENERATED_MARKER = utils.GENERATED_MARKER
PORTABLE_TOOLS = utils.PORTABLE_TOOLS
load_json = utils.load_json
parse_version_range = utils.parse_version_range
rooted = utils.rooted
safe_relative = utils.safe_relative
validate_package_name = utils.validate_package_name
validate_schema_definition = utils.validate_schema_definition
write_json = utils.write_json


# =====================================================================
# Package Validation & Graph Checking
# =====================================================================


def render_agent_references(
    text: str, manifest: dict[str, Any], prefix: str = ""
) -> str:
  """Replace {{agent:NAME}} template tokens with prefixed agent identifiers."""
  known_agents = {agent["name"] for agent in manifest["agents"]}

  def replace(match: re.Match[str]) -> str:
    name = match.group(1)
    if name not in known_agents:
      raise CoworkerError(f"instructions reference unknown agent: {name}")
    return prefix + name

  rendered = re.sub(r"\{\{agent:([a-z0-9]+(?:-[a-z0-9]+)*)}}", replace, text)
  if "{{agent:" in rendered:
    raise CoworkerError("instructions contain a malformed agent reference")
  return rendered


def validate_delegation_graph(
    entrypoint: dict[str, Any],
    agents_by_name: dict[str, dict[str, Any]],
    expected_agents: set[str],
) -> None:
  """Validate delegation graph for cycle detection, max depth <= 2, and reachability."""
  visited: set[str] = set()

  def visit(agent_name: str, depth: int, path: tuple[str, ...]) -> None:
    if agent_name in path:
      cycle = " -> ".join((*path, agent_name))
      raise CoworkerError(f"delegation graph contains a cycle: {cycle}")
    if depth > 2:
      chain = " -> ".join((*path, agent_name))
      raise CoworkerError(
          f"delegation exceeds two levels below entrypoint: {chain}"
      )

    visited.add(agent_name)
    for delegated in agents_by_name[agent_name].get("delegates", []):
      visit(delegated, depth + 1, (*path, agent_name))

  for delegated in entrypoint["delegates"]:
    visit(delegated, 1, (entrypoint["name"],))

  unreachable = expected_agents - visited
  if unreachable:
    raise CoworkerError(
        f"agents are unreachable from entrypoint: {sorted(unreachable)}"
    )


def load_package(package_dir: Path) -> dict[str, Any]:
  """Load, validate, and verify the structural integrity of a package manifest and its graph."""
  manifest_path = package_dir / "package.json"
  manifest = load_json(manifest_path)

  # 1. Manifest level requirements
  required_manifest_fields = {
      "name",
      "version",
      "description",
      "entrypoint",
      "agents",
      "environment",
      "compatibility_targets",
  }
  missing_manifest = required_manifest_fields - manifest.keys()
  if missing_manifest:
    raise CoworkerError(
        f"package.json missing: {', '.join(sorted(missing_manifest))}"
    )
  manifest_name = manifest["name"]

  validate_package_name(manifest_name)

  # 2. Agent presence & uniqueness
  agent_names = [agent.get("name") for agent in manifest["agents"]]
  if None in agent_names or len(agent_names) != len(set(agent_names)):
    raise CoworkerError(
        f"Agent names are not unique in package or are missing: {manifest_name}"
    )
  name_set = set(agent_names)

  # 3. Entrypoint validation
  entrypoint = manifest["entrypoint"]
  entry_point_is_missing = {
      "name",
      "instructions",
      "delegates",
  } - entrypoint.keys()
  if entry_point_is_missing:
    raise CoworkerError(
        f"Entrypoint {manifest_name} missing fields:"
        f" {sorted(entry_point_is_missing)}"
    )
  if entrypoint["name"] in name_set:
    raise CoworkerError("Entrypoint and agent names must be distinct")

  unknown_entry_delegates = set(entrypoint["delegates"]) - name_set
  if unknown_entry_delegates:
    raise CoworkerError(
        "entrypoint delegates to unknown agents:"
        f" {sorted(unknown_entry_delegates)}"
    )

  # 4. Agent definitions & files
  agents_by_name = {agent["name"]: agent for agent in manifest["agents"]}
  for agent in manifest["agents"]:
    required_agent_fields = {
        "name",
        "description",
        "instructions",
        "accepts",
        "produces",
        "delegates",
        "tools",
    }
    missing_agent = required_agent_fields - agent.keys()
    if missing_agent:
      agent_label = agent.get("name", "<unnamed>")
      raise CoworkerError(
          f"agent {agent_label} missing: {sorted(missing_agent)}"
      )

    unknown_tools = set(agent["tools"]) - PORTABLE_TOOLS
    if unknown_tools:
      raise CoworkerError(
          f"agent {agent['name']} has unsupported portable tools:"
          f" {sorted(unknown_tools)}"
      )

    unknown_delegates = set(agent.get("delegates", [])) - name_set
    if unknown_delegates:
      raise CoworkerError(
          f"agent {agent['name']} delegates to unknown agents:"
          f" {unknown_delegates}"
      )

    source_inst = rooted(
        package_dir, agent["instructions"], "agent instructions"
    )
    if not source_inst.is_file():
      raise CoworkerError(f"missing agent instructions: {source_inst}")
    render_agent_references(
        source_inst.read_text(encoding="utf-8"), manifest, ""
    )

    for schema_key in ("accepts", "produces"):
      schema_path = rooted(
          package_dir, agent[schema_key], f"agent {schema_key}"
      )
      if not schema_path.is_file():
        raise CoworkerError(f"missing agent {schema_key} schema: {schema_path}")

  # 5. Delegation graph traversal: cycle detection, max depth <= 2, reachability
  validate_delegation_graph(entrypoint, agents_by_name, name_set)

  # 6. Entrypoint and environment files verification
  entry_inst = rooted(
      package_dir, entrypoint["instructions"], "entrypoint instructions"
  )
  if not entry_inst.is_file():
    raise CoworkerError(f"missing entrypoint instructions: {entry_inst}")
  render_agent_references(entry_inst.read_text(encoding="utf-8"), manifest, "")

  questions = rooted(
      package_dir, manifest["environment"]["questions"], "environment questions"
  )
  env_schema = rooted(
      package_dir, manifest["environment"]["schema"], "environment schema"
  )
  if not questions.is_file() or not env_schema.is_file():
    raise CoworkerError("environment questions or schema is missing")

  return manifest


def target_for(manifest: dict[str, Any], harness: str) -> dict[str, Any]:
  """Retrieve and validate the compatibility target configuration for a given harness."""
  target = next(
      (
          t
          for t in manifest.get("compatibility_targets", [])
          if t.get("harness") == harness
      ),
      None,
  )
  if not target:
    raise CoworkerError(f"no compatibility target for harness {harness}")

  # Validate version constraint syntax
  parse_version_range(target["versions"])

  # Verify target satisfies all required capabilities
  required = set(manifest.get("required_capabilities", []))
  available = {k for k, v in target.get("capabilities", {}).items() if v}
  missing = required - available
  if missing:
    raise CoworkerError(
        f"target {target['name']} cannot preserve required capabilities:"
        f" {', '.join(sorted(missing))}"
    )

  return target


def verify_package(package_dir: Path) -> None:
  """Verify package manifest, schema definitions, and target compatibility."""
  manifest = load_package(package_dir)
  for schema_rel in manifest.get("schemas", []):
    schema_path = rooted(package_dir, schema_rel, "schema path")
    schema = load_json(schema_path)
    errors = validate_schema_definition(schema)
    if errors:
      raise CoworkerError(f"invalid schema {schema_rel}:\n" + "\n".join(errors))

  harnesses = {
      target["harness"] for target in manifest["compatibility_targets"]
  }
  for harness in harnesses:
    target_for(manifest, harness)


# =====================================================================
# Document & Prompt Generators
# =====================================================================


def generated_header(manifest: dict[str, Any], target: dict[str, Any]) -> str:
  """Generate the standard comment header for translated assets."""
  return (
      f"<!-- {GENERATED_MARKER} Source"
      f" {manifest['name']}@{manifest['version']}; compatibility-target"
      f" {target['name']}. -->\n\n"
  )


def skill_frontmatter(name: str, description: str) -> str:
  """Generate YAML frontmatter for an entrypoint skill."""
  escaped = description.replace('"', "'")
  return f'---\nname: {name}\ndescription: "{escaped}"\n---\n\n'


def skill_package_contract(package_name: str, delegates: list[str]) -> str:
  """Build the runtime execution contract section for an installed skill package."""
  targets = ", ".join(delegates) or "none"
  runtime = f"python3 .coworker/{package_name}/runtime/runtime.py"
  return f"""

## Installed package contract

- Read `.coworker/{package_name}/environment.json` before delegating.
- Before the first delegation, run `{runtime} start-run --workspace . --package {package_name}` exactly once.
- Retain the returned `run_id`, include it in every delegation request, and use only that run's artifact references.
- Direct delegation targets: {targets}. Do not invoke any other package agent directly.
- Run one branch at a time. A subagent may wait for its one nested helper; do not run sibling branches in parallel.
- Validate messages with `{runtime} validate`.
- Store durable outputs below `.coworker/{package_name}/runs/<run_id>/artifacts/`; never overwrite an existing artifact.
- Create descriptors with `{runtime} describe-artifact --workspace . --package {package_name} --run-id <run_id> --file <path-relative-to-artifacts> --schema <path-relative-to-package> --media-type <type>` and pass descriptors, not payload copies.
"""


def skill_document(
    package_dir: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
    agent_prefix: str = "",
) -> str:
  """Generate SKILL.md markdown document for entrypoint workflow."""
  entry = manifest["entrypoint"]
  frontmatter = skill_frontmatter(entry["name"], manifest["description"])
  header = generated_header(manifest, target)

  source_path = rooted(
      package_dir, entry["instructions"], "entrypoint instructions"
  )
  rendered_body = render_agent_references(
      source_path.read_text(encoding="utf-8"), manifest, agent_prefix
  ).rstrip()

  delegates = [agent_prefix + name for name in entry["delegates"]]
  contract = skill_package_contract(manifest["name"], delegates)

  return f"{frontmatter}{header}{rendered_body}{contract}"


def agent_execution_contract(
    package_name: str, agent: dict[str, Any], delegates: list[str]
) -> str:
  """Build the runtime execution contract section for an agent prompt."""
  targets = ", ".join(delegates) or "none"
  runtime = f"python3 .coworker/{package_name}/runtime/runtime.py"
  return f"""

## Generated execution contract

- Validate the request against `.coworker/{package_name}/{agent['accepts']}` before work.
- Require the request's `run_id` and keep every output under `.coworker/{package_name}/runs/<run_id>/artifacts/`.
- Validate the result against `.coworker/{package_name}/{agent['produces']}` before returning.
- Use `{runtime} validate --schema SCHEMA INSTANCE`.
- Return the common result envelope with `completed`, `invalid_input`, `needs_input`, or `failed`.
- Never overwrite an existing artifact; create its descriptor with `{runtime} describe-artifact --workspace . --package {package_name} --run-id <run_id> --file <path-relative-to-artifacts> --schema <path-relative-to-package> --media-type <type>`.
- Reject artifact references whose URI does not contain the request's exact package and `run_id`.
- Materialize durable outputs and return artifact descriptors; do not copy payloads into messages.
- Allowed delegation targets: {targets}.
- Run sequentially. Never weaken inherited permissions or approvals.
"""


def agent_prompt(
    package_dir: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
    agent: dict[str, Any],
    agent_prefix: str = "",
) -> str:
  """Generate agent instruction prompt document with generated execution contract."""
  source_path = rooted(package_dir, agent["instructions"], "agent instructions")
  rendered_body = render_agent_references(
      source_path.read_text(encoding="utf-8"), manifest, agent_prefix
  ).rstrip()

  delegates = [agent_prefix + name for name in agent.get("delegates", [])]
  contract = agent_execution_contract(manifest["name"], agent, delegates)

  return f"{generated_header(manifest, target)}{rendered_body}{contract}"


def claude_frontmatter(agent: dict[str, Any]) -> str:
  """Generate Claude agent YAML frontmatter with mapped native tools."""
  tool_map = {
      "read": "Read",
      "search": "Grep, Glob",
      "shell": "Bash",
      "write": "Write",
      "edit": "Edit",
  }
  tools: list[str] = []
  for portable in agent["tools"]:
    for native in tool_map[portable].split(", "):
      if native not in tools:
        tools.append(native)
  if agent.get("delegates"):
    tools.append("Agent")

  return (
      "---\n"
      f"name: {agent['name']}\n"
      f"description: {agent['description']}\n"
      f"tools: {', '.join(tools)}\n"
      "background: false\n"
      "---\n\n"
  )


# =====================================================================
# Harness Translation
# =====================================================================


def copy_asset(source: Path, destination: Path) -> None:
  """Copy a file or directory tree to destination, creating parent directories as needed."""
  if source.is_dir():
    shutil.copytree(source, destination, dirs_exist_ok=True)
  elif source.is_file():
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
  else:
    raise CoworkerError(f"copy path does not exist: {source}")


def bundle_runtime_scripts(runtime_dir: Path) -> None:
  """Bundle framework runtime worker scripts into the distribution runtime directory."""
  runtime_dir.mkdir(parents=True, exist_ok=True)
  framework_dir = Path(__file__).resolve().parent

  for script_name in ("runtime.py", "utils.py"):
    script_path = framework_dir / script_name
    if script_path.is_file():
      shutil.copy2(script_path, runtime_dir / script_name)


def copy_runtime_assets(
    package_dir: Path, output: Path, manifest: dict[str, Any]
) -> None:
  """Copy configured package assets and bundle framework runtime worker scripts."""
  for item in manifest.get("copy", []):
    rel = safe_relative(item, "copy path")
    source = rooted(package_dir, str(rel), "copy path")
    copy_asset(source, output / rel)

  bundle_runtime_scripts(output / "runtime")


def runtime_copy_mappings(
    package_name: str, copy_items: Sequence[str]
) -> list[dict[str, str]]:
  """Build list of runtime and schema copy mappings for an installed package."""
  mappings = [
      {"source": "runtime", "destination": f".coworker/{package_name}/runtime"},
      {"source": "schemas", "destination": f".coworker/{package_name}/schemas"},
  ]
  if any(Path(item).parts[:1] == ("scripts",) for item in copy_items):
    mappings.append({
        "source": "scripts",
        "destination": f".coworker/{package_name}/scripts",
    })
  mappings.append({
      "source": "coworker-build.json",
      "destination": f".coworker/{package_name}/coworker-build.json",
  })
  return mappings


def harness_install_specs(entry_name: str) -> dict[str, dict[str, Any]]:
  """Return harness-specific copy rules, version command, and configuration settings."""
  return {
      "claude-code": {
          "version_command": ["claude", "--version"],
          "copies": [
              {
                  "source": f"project/skills/{entry_name}",
                  "destination": f".claude/skills/{entry_name}",
              },
              {"source": "project/agents", "destination": ".claude/agents"},
          ],
          "json_merges": [{
              "path": ".claude/settings.local.json",
              "value": {
                  "env": {
                      "CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH": "2",
                      "CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS": "2",
                  }
              },
          }],
          "toml_sets": [],
      },
      "codex": {
          "version_command": ["codex", "--version"],
          "copies": [
              {
                  "source": f"skills/{entry_name}",
                  "destination": f".agents/skills/{entry_name}",
              },
              {"source": "agents", "destination": ".codex/agents"},
          ],
          "json_merges": [],
          "toml_sets": [
              {
                  "path": ".codex/config.toml",
                  "section": "agents",
                  "key": "enabled",
                  "value": True,
              },
              {
                  "path": ".codex/config.toml",
                  "section": "agents",
                  "key": "max_concurrent_threads_per_session",
                  "value": 2,
              },
          ],
      },
  }


def build_install_plan(
    manifest: dict[str, Any],
    target: dict[str, Any],
    harness: str,
) -> dict[str, Any]:
  """Generate coworker-install.json installation plan for target harness."""
  name = manifest["name"]
  entry = manifest["entrypoint"]["name"]
  spec = harness_install_specs(entry)[harness]

  return {
      "harness": harness,
      "version_range": target["versions"],
      "environment_questions": "environment/questions.json",
      "environment_schema": manifest["environment"]["schema"],
      "environment_path": f".coworker/{name}/environment.json",
      "ownership_path": f".coworker/{name}/ownership.json",
      "version_command": spec["version_command"],
      "copies": (
          spec["copies"] + runtime_copy_mappings(name, manifest.get("copy", []))
      ),
      "json_merges": spec["json_merges"],
      "toml_sets": spec["toml_sets"],
  }


def translate_claude(
    package_dir: Path,
    output: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
) -> None:
  """Translate package into Claude Code plugin and project layouts."""
  plugin = output / ".claude-plugin"
  write_json(
      plugin / "plugin.json",
      {
          "name": manifest["name"],
          "displayName": manifest.get("display_name", manifest["name"]),
          "version": manifest["version"],
          "description": manifest["description"],
          "author": {"name": manifest.get("author", "Coworker Framework")},
      },
  )

  entry_name = manifest["entrypoint"]["name"]
  plugin_agent_prefix = f"{manifest['name']}:"

  # Write plugin skill and standalone project skill
  skill = output / "skills" / entry_name / "SKILL.md"
  skill.parent.mkdir(parents=True, exist_ok=True)
  skill.write_text(
      skill_document(package_dir, manifest, target, plugin_agent_prefix),
      encoding="utf-8",
  )

  project_skill = output / "project" / "skills" / entry_name / "SKILL.md"
  project_skill.parent.mkdir(parents=True, exist_ok=True)
  project_skill.write_text(
      skill_document(package_dir, manifest, target), encoding="utf-8"
  )

  # Write plugin agents and project agents
  for agent in manifest["agents"]:
    fm = claude_frontmatter(agent)
    path = output / "agents" / f"{agent['name']}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        fm
        + agent_prompt(
            package_dir, manifest, target, agent, plugin_agent_prefix
        ),
        encoding="utf-8",
    )

    project_agent = output / "project" / "agents" / f"{agent['name']}.md"
    project_agent.parent.mkdir(parents=True, exist_ok=True)
    project_agent.write_text(
        fm + agent_prompt(package_dir, manifest, target, agent),
        encoding="utf-8",
    )

  write_json(
      output / "coworker-install.json",
      build_install_plan(manifest, target, "claude-code"),
  )


def translate_codex(
    package_dir: Path,
    output: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
) -> None:
  """Translate package into Codex plugin and agents layouts."""
  plugin = output / ".codex-plugin"
  write_json(
      plugin / "plugin.json",
      {
          "name": manifest["name"],
          "version": manifest["version"],
          "description": manifest["description"],
          "author": {"name": manifest.get("author", "Coworker Framework")},
          "skills": "./skills/",
          "interface": {
              "displayName": manifest.get("display_name", manifest["name"]),
              "shortDescription": manifest["description"][:80],
              "longDescription": manifest["description"],
              "developerName": manifest.get("author", "Coworker Framework"),
              "category": "Productivity",
              "capabilities": ["Interactive", "Write"],
              "defaultPrompt": [
                  manifest["entrypoint"].get(
                      "example_prompt", "Run this workflow."
                  )[:128]
              ],
          },
      },
  )

  entry_name = manifest["entrypoint"]["name"]
  skill = output / "skills" / entry_name / "SKILL.md"
  skill.parent.mkdir(parents=True, exist_ok=True)
  skill.write_text(
      skill_document(package_dir, manifest, target), encoding="utf-8"
  )

  for agent in manifest["agents"]:
    config = output / "agents" / f"{agent['name']}.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    prompt = agent_prompt(package_dir, manifest, target, agent).replace(
        '"""', "'''"
    )
    sandbox = (
        "workspace-write"
        if any(tool in agent["tools"] for tool in ("write", "edit"))
        else "read-only"
    )
    config.write_text(
        f'name = "{agent["name"]}"\n'
        f'description = "{agent["description"]}"\n'
        f'sandbox_mode = "{sandbox}"\n'
        f'developer_instructions = """\n{prompt}\n"""\n',
        encoding="utf-8",
    )

  write_json(
      output / "coworker-install.json",
      build_install_plan(manifest, target, "codex"),
  )


def translate(package_dir: Path, harness: str, output: Path) -> None:
  """Translate a package into a target harness distribution."""
  manifest = load_package(package_dir)
  target = target_for(manifest, harness)

  if output.exists():
    marker = output / "coworker-build.json"
    if not marker.is_file() or not load_json(marker).get("generated"):
      raise CoworkerError(
          f"refusing to replace non-generated output directory: {output}"
      )
    shutil.rmtree(output)

  output.mkdir(parents=True)

  translators: dict[
      str, Callable[[Path, Path, dict[str, Any], dict[str, Any]], None]
  ] = {
      "claude-code": translate_claude,
      "codex": translate_codex,
  }
  translators[harness](package_dir, output, manifest, target)

  copy_runtime_assets(package_dir, output, manifest)

  write_json(
      output / "coworker-build.json",
      {
          "framework_version": FRAMEWORK_VERSION,
          "source": manifest["name"],
          "source_version": manifest["version"],
          "compatibility_target": target["name"],
          "harness": harness,
          "version_range": target["versions"],
          "generated": True,
      },
  )


def build_compiler_parser() -> argparse.ArgumentParser:
  """Construct parser for standalone compiler commands."""
  parser = argparse.ArgumentParser(
      prog="coworker-compiler",
      description="Verification and translation compiler for Coworker packages.",
  )
  sub = parser.add_subparsers(dest="command", required=True)

  # verify
  p_verify = sub.add_parser(
      "verify", help="Verify package structure and schemas"
  )
  p_verify.add_argument("package", type=Path, help="Path to package directory")

  # translate
  p_translate = sub.add_parser(
      "translate", help="Translate package to target harness"
  )
  p_translate.add_argument(
      "package", type=Path, help="Path to package directory"
  )
  p_translate.add_argument(
      "--harness",
      choices=["claude-code", "codex"],
      required=True,
      help="Target harness",
  )
  p_translate.add_argument(
      "--output", type=Path, required=True, help="Output directory"
  )

  return parser


def main(argv: list[str] | None = None) -> int:
  """Main entrypoint for standalone compiler operations."""
  parser = build_compiler_parser()
  args = parser.parse_args(argv)

  try:
    if args.command == "verify":
      verify_package(args.package)

    elif args.command == "translate":
      translate(args.package, args.harness, args.output)

    return 0
  except CoworkerError as exc:
    print(f"compiler: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
