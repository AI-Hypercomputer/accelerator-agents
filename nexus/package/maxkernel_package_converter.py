#!/usr/bin/env python3
"""Converts raw MaxKernel / TPU multi-skill plugins into a Coworker Canonical Package."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil

Path = pathlib.Path


def write_json(path: Path, data: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=3, sort_keys=True) + "\n")


def strip_frontmatter(content: str) -> str:
  """Remove YAML frontmatter (--- ... ---) from Markdown files."""
  if content.startswith("---"):
    parts = content.split("---", 2)
    if len(parts) >= 3:
      return parts[2].lstrip()
  return content


def create_directories(target_dir: Path) -> None:
  for subdir in [
      "instructions",
      "tools",
      "references",
      "schemas",
      "environment",
  ]:
    (target_dir / subdir).mkdir(parents=True, exist_ok=True)


def convert_instructions(source_dir: Path, target_dir: Path) -> None:
  """Convert legacy MaxKernel subagents and SKILL.md into templated Coworker instructions."""
  instructions_map = {
      source_dir / "SKILL.md": target_dir / "instructions/entrypoint.md",
      source_dir / "subagents/worker.md": (
          target_dir / "instructions/worker.md"
      ),
      source_dir / "subagents/plan_kernel_agent.md": (
          target_dir / "instructions/plan-kernel.md"
      ),
      source_dir / "subagents/implement_kernel_agent.md": (
          target_dir / "instructions/implement-kernel.md"
      ),
      source_dir / "subagents/generate_test_file_agent.md": (
          target_dir / "instructions/generate-test-file.md"
      ),
      source_dir / "subagents/fix_test_script_agent.md": (
          target_dir / "instructions/fix-test-script.md"
      ),
      source_dir / "subagents/test_script_validation_summary_agent.md": (
          target_dir / "instructions/test-script-validation-summary.md"
      ),
      source_dir / "subagents/fix_kernel_compilation_agent.md": (
          target_dir / "instructions/fix-kernel-compilation.md"
      ),
      source_dir / "subagents/compilation_summary_agent.md": (
          target_dir / "instructions/compilation-summary.md"
      ),
      source_dir / "subagents/summarize_test_results_agent.md": (
          target_dir / "instructions/summarize-test-results.md"
      ),
      source_dir / "subagents/autotune_planner_agent.md": (
          target_dir / "instructions/autotune-planner.md"
      ),
      source_dir / "subagents/autotune_summary_agent.md": (
          target_dir / "instructions/autotune-summary.md"
      ),
      source_dir / "subagents/generate_profile_script_agent.md": (
          target_dir / "instructions/generate-profile-script.md"
      ),
      source_dir / "subagents/summarize_profile_agent.md": (
          target_dir / "instructions/summarize-profile.md"
      ),
  }

  agent_replacements = [
      ("subagents/worker.md", "{{agent:worker}}"),
      ("subagents/plan_kernel_agent.md", "{{agent:plan-kernel}}"),
      ("subagents/implement_kernel_agent.md", "{{agent:implement-kernel}}"),
      ("subagents/generate_test_file_agent.md", "{{agent:generate-test-file}}"),
      ("subagents/fix_test_script_agent.md", "{{agent:fix-test-script}}"),
      (
          "subagents/test_script_validation_summary_agent.md",
          "{{agent:test-script-validation-summary}}",
      ),
      (
          "subagents/fix_kernel_compilation_agent.md",
          "{{agent:fix-kernel-compilation}}",
      ),
      ("subagents/compilation_summary_agent.md", "{{agent:compilation-summary}}"),
      (
          "subagents/summarize_test_results_agent.md",
          "{{agent:summarize-test-results}}",
      ),
      ("subagents/autotune_planner_agent.md", "{{agent:autotune-planner}}"),
      ("subagents/autotune_summary_agent.md", "{{agent:autotune-summary}}"),
      (
          "subagents/generate_profile_script_agent.md",
          "{{agent:generate-profile-script}}",
      ),
      ("subagents/summarize_profile_agent.md", "{{agent:summarize-profile}}"),
  ]

  for src, dst in instructions_map.items():
    if src.exists():
      content = strip_frontmatter(src.read_text(encoding="utf-8"))
      for old_ref, new_ref in agent_replacements:
        content = content.replace(old_ref, new_ref)
      dst.write_text(content, encoding="utf-8")


def copy_tools(source_dir: Path, target_dir: Path) -> None:
  """Copy MaxKernel helper tools to the package tools directory."""
  tools_src = source_dir / "tools"
  if tools_src.exists():
    for script in tools_src.glob("*.py"):
      dst = target_dir / "tools" / script.name
      shutil.copy2(script, dst)
      os.chmod(dst, 0o755)


def copy_references(source_dir: Path, target_dir: Path) -> None:
  """Copy reference markdown documents (setup, VM guide, etc.)."""
  ref_files = [
      source_dir / "setup.md",
      source_dir / "tpu_vm.md",
      source_dir / "README.md",
  ]
  for ref in ref_files:
    if ref.exists():
      shutil.copy2(ref, target_dir / "references" / ref.name)


def generate_environment(
    target_dir: Path, package_name: str, display_name: str
) -> None:
  """Generate environment questionnaire for TPU connection & tolerances."""
  questions = {
      "$schema": "https://json-schema.org/draft/2020-12/schema",
      "description": f"{display_name} Environment Questionnaire",
      "questions": [
          {
              "default": "local-tpu-vm",
              "name": "tpu_target",
              "prompt": "Remote TPU VM hostname or SSH target address",
              "type": "string",
          },
          {
              "default": 5,
              "name": "max_iterations",
              "prompt": "Maximum number of optimization iterations",
              "type": "integer",
          },
          {
              "default": 0.01,
              "name": "atol",
              "prompt": "Absolute tolerance (atol) for kernel correctness",
              "type": "float",
          },
          {
              "default": 0.01,
              "name": "rtol",
              "prompt": "Relative tolerance (rtol) for kernel correctness",
              "type": "float",
          },
      ],
  }
  write_json(target_dir / "environment/questions.json", questions)


def generate_schemas(target_dir: Path, package_name: str) -> None:
  """Generate formal JSON contracts for message passing and artifact descriptors."""
  artifact_prop = {
      "type": "object",
      "additionalProperties": False,
      "required": [
          "file",
          "media_type",
          "run_id",
          "schema",
          "sha256",
          "size_bytes",
      ],
      "properties": {
          "file": {"type": "string"},
          "media_type": {"type": "string"},
          "run_id": {
              "type": "string",
              "pattern": "^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$",
          },
          "schema": {"type": "string"},
          "sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
          "size_bytes": {"type": "integer", "minimum": 0},
      },
  }

  schemas = {
      "environment.json": {
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "type": "object",
          "additionalProperties": False,
          "required": ["atol", "max_iterations", "rtol", "tpu_target"],
          "properties": {
              "atol": {"type": "number"},
              "max_iterations": {"type": "integer", "minimum": 1, "maximum": 20},
              "rtol": {"type": "number"},
              "tpu_target": {"type": "string"},
          },
      },
      "artifact.json": {
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "type": "object",
          "additionalProperties": False,
          "required": [
              "file",
              "media_type",
              "run_id",
              "schema",
              "sha256",
              "size_bytes",
          ],
          "properties": {
              "file": {"type": "string"},
              "media_type": {"type": "string"},
              "run_id": {
                  "type": "string",
                  "pattern": "^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$",
              },
              "schema": {"type": "string"},
              "sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
              "size_bytes": {"type": "integer", "minimum": 0},
          },
      },
      "result-envelope.json": {
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "type": "object",
          "additionalProperties": False,
          "required": ["status"],
          "properties": {
              "artifacts": {"type": "array", "items": {"type": "object"}},
              "errors": {"type": "array", "items": {"type": "object"}},
              "result": {"type": "object"},
              "status": {
                  "type": "string",
                  "enum": [
                      "completed",
                      "invalid_input",
                      "needs_input",
                      "failed",
                  ],
              },
          },
      },
      "worker-request.json": {
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "type": "object",
          "additionalProperties": False,
          "required": ["base_kernel_artifact", "iteration", "run_id"],
          "properties": {
              "base_kernel_artifact": artifact_prop,
              "iteration": {"type": "integer", "minimum": 1},
              "run_id": {
                  "type": "string",
                  "pattern": "^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$",
              },
          },
      },
      "worker-result.json": {
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "type": "object",
          "additionalProperties": False,
          "required": [
              "iteration",
              "optimized_kernel_artifact",
              "run_id",
              "status",
          ],
          "properties": {
              "iteration": {"type": "integer", "minimum": 1},
              "latency_ms": {"type": "number"},
              "optimized_kernel_artifact": artifact_prop,
              "run_id": {
                  "type": "string",
                  "pattern": "^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$",
              },
              "speedup_factor": {"type": "number"},
              "status": {
                  "type": "string",
                  "enum": ["success", "failed", "completed"],
              },
          },
      },
      "plan-request.json": {
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "type": "object",
          "additionalProperties": False,
          "required": ["base_kernel_artifact", "iteration", "run_id"],
          "properties": {
              "base_kernel_artifact": artifact_prop,
              "iteration": {"type": "integer", "minimum": 1},
              "previous_profile_artifact": artifact_prop,
              "run_id": {
                  "type": "string",
                  "pattern": "^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$",
              },
          },
      },
      "plan-result.json": {
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "type": "object",
          "additionalProperties": False,
          "required": ["plan_artifact", "proposed_strategy", "run_id"],
          "properties": {
              "plan_artifact": artifact_prop,
              "proposed_strategy": {"type": "string"},
              "run_id": {
                  "type": "string",
                  "pattern": "^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$",
              },
          },
      },
      "implement-request.json": {
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "type": "object",
          "additionalProperties": False,
          "required": ["base_kernel_artifact", "plan_artifact", "run_id"],
          "properties": {
              "base_kernel_artifact": artifact_prop,
              "plan_artifact": artifact_prop,
              "run_id": {
                  "type": "string",
                  "pattern": "^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$",
              },
          },
      },
      "implement-result.json": {
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "type": "object",
          "additionalProperties": False,
          "required": ["code_artifact", "run_id", "status"],
          "properties": {
              "code_artifact": artifact_prop,
              "notes": {"type": "string"},
              "run_id": {
                  "type": "string",
                  "pattern": "^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$",
              },
              "status": {
                  "type": "string",
                  "enum": ["success", "compilation_error"],
              },
          },
      },
      "autotune-request.json": {
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "type": "object",
          "additionalProperties": False,
          "required": ["code_artifact", "run_id"],
          "properties": {
              "code_artifact": artifact_prop,
              "run_id": {
                  "type": "string",
                  "pattern": "^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$",
              },
          },
      },
      "autotune-result.json": {
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "type": "object",
          "additionalProperties": False,
          "required": ["best_config_artifact", "best_latency_ms", "run_id"],
          "properties": {
              "best_config_artifact": artifact_prop,
              "best_latency_ms": {"type": "number"},
              "run_id": {
                  "type": "string",
                  "pattern": "^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$",
              },
          },
      },
      "profile-request.json": {
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "type": "object",
          "additionalProperties": False,
          "required": ["code_artifact", "run_id"],
          "properties": {
              "code_artifact": artifact_prop,
              "run_id": {
                  "type": "string",
                  "pattern": "^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$",
              },
          },
      },
      "profile-result.json": {
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "type": "object",
          "additionalProperties": False,
          "required": ["bottleneck_summary", "profile_artifact", "run_id"],
          "properties": {
              "bottleneck_summary": {"type": "string"},
              "profile_artifact": artifact_prop,
              "run_id": {
                  "type": "string",
                  "pattern": "^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$",
              },
              "vpu_utilization": {"type": "number"},
          },
      },
      "test-request.json": {
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "type": "object",
          "additionalProperties": False,
          "required": ["code_artifact", "run_id"],
          "properties": {
              "code_artifact": artifact_prop,
              "run_id": {
                  "type": "string",
                  "pattern": "^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$",
              },
          },
      },
      "test-result.json": {
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "type": "object",
          "additionalProperties": False,
          "required": ["correct", "latency_ms", "run_id"],
          "properties": {
              "correct": {"type": "boolean"},
              "error_message": {"type": "string"},
              "latency_ms": {"type": "number"},
              "run_id": {
                  "type": "string",
                  "pattern": "^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$",
              },
          },
      },
  }

  for filename, schema in schemas.items():
    write_json(target_dir / "schemas" / filename, schema)


def generate_package_manifest(
    target_dir: Path,
    package_name: str,
    display_name: str,
    description: str,
    author: str,
) -> None:
  """Generate canonical package manifest registering all 13 subagents and contracts."""
  leaf_agents = [
      (
          "plan-kernel",
          "Formulates Pallas kernel optimization strategies.",
          "instructions/plan-kernel.md",
          "schemas/plan-request.json",
          "schemas/plan-result.json",
          ["read", "search"],
      ),
      (
          "implement-kernel",
          "Implements Pallas TPU kernel code from optimization plans.",
          "instructions/implement-kernel.md",
          "schemas/implement-request.json",
          "schemas/implement-result.json",
          ["read", "write", "edit"],
      ),
      (
          "generate-test-file",
          "Generates correctness and benchmark harness test inputs.",
          "instructions/generate-test-file.md",
          "schemas/test-request.json",
          "schemas/test-result.json",
          ["read", "write", "edit"],
      ),
      (
          "fix-test-script",
          "Repairs and adjusts harness scripts and input generation.",
          "instructions/fix-test-script.md",
          "schemas/test-request.json",
          "schemas/test-result.json",
          ["read", "write", "edit"],
      ),
      (
          "test-script-validation-summary",
          "Summarizes test script validation feedback.",
          "instructions/test-script-validation-summary.md",
          "schemas/test-request.json",
          "schemas/test-result.json",
          ["read"],
      ),
      (
          "fix-kernel-compilation",
          "Fixes TPU kernel compilation errors.",
          "instructions/fix-kernel-compilation.md",
          "schemas/implement-request.json",
          "schemas/implement-result.json",
          ["read", "write", "edit"],
      ),
      (
          "compilation-summary",
          "Summarizes TPU kernel compilation status.",
          "instructions/compilation-summary.md",
          "schemas/implement-request.json",
          "schemas/implement-result.json",
          ["read"],
      ),
      (
          "summarize-test-results",
          "Summarizes numerical correctness and benchmark latency.",
          "instructions/summarize-test-results.md",
          "schemas/test-request.json",
          "schemas/test-result.json",
          ["read"],
      ),
      (
          "autotune-planner",
          "Plans parameter search grid for block and tile dimensions.",
          "instructions/autotune-planner.md",
          "schemas/autotune-request.json",
          "schemas/autotune-result.json",
          ["read", "write", "edit", "shell"],
      ),
      (
          "autotune-summary",
          "Extracts best configuration from autotuning search.",
          "instructions/autotune-summary.md",
          "schemas/autotune-request.json",
          "schemas/autotune-result.json",
          ["read"],
      ),
      (
          "generate-profile-script",
          "Generates TPU profile execution script for XPlane traces.",
          "instructions/generate-profile-script.md",
          "schemas/profile-request.json",
          "schemas/profile-result.json",
          ["read", "write", "edit"],
      ),
      (
          "summarize-profile",
          "Parses XPlane traces to summarize TPU compute/memory metrics.",
          "instructions/summarize-profile.md",
          "schemas/profile-request.json",
          "schemas/profile-result.json",
          ["read", "shell"],
      ),
  ]

  worker_delegates = [name for name, _, _, _, _, _ in leaf_agents]

  agents = [
      {
          "name": "worker",
          "description": (
              "Coordinates one full iteration cycle of kernel optimization."
          ),
          "instructions": "instructions/worker.md",
          "accepts": "schemas/worker-request.json",
          "produces": "schemas/worker-result.json",
          "tools": ["read", "write", "edit", "shell"],
          "delegates": worker_delegates,
      }
  ]

  for name, desc, inst, accepts, produces, tools in leaf_agents:
    agents.append({
        "name": name,
        "description": desc,
        "instructions": inst,
        "accepts": accepts,
        "produces": produces,
        "tools": tools,
        "delegates": [],
    })

  manifest = {
      "name": package_name,
      "version": "1.0.0",
      "display_name": display_name,
      "description": description,
      "author": author,
      "copy": ["schemas", "environment", "tools", "references"],
      "required_capabilities": [
          "skills",
          "isolated_agents",
          "sequential_execution",
          "named_tools",
      ],
      "environment": {
          "questions": "environment/questions.json",
          "schema": "schemas/environment.json",
      },
      "entrypoint": {
          "name": package_name,
          "instructions": "instructions/entrypoint.md",
          "example_prompt": (
              "Optimize the FlashAttention Pallas kernel for TPU v5e."
          ),
          "delegates": ["worker"],
      },
      "agents": agents,
      "compatibility_targets": [
          {
              "capabilities": {
                  "isolated_agents": True,
                  "named_tools": True,
                  "nested_delegation": True,
                  "sequential_execution": True,
                  "skills": True,
              },
              "harness": "claude-code",
              "name": "claude-code-2.1",
              "versions": ">=2.1.219,<3.0.0",
          },
          {
              "capabilities": {
                  "isolated_agents": True,
                  "named_tools": True,
                  "nested_delegation": True,
                  "sequential_execution": True,
                  "skills": True,
              },
              "harness": "codex",
              "name": "codex-0.147",
              "versions": ">=0.145.0,<0.160.0",
          },
      ],
      "schemas": [
          "schemas/environment.json",
          "schemas/artifact.json",
          "schemas/result-envelope.json",
          "schemas/worker-request.json",
          "schemas/worker-result.json",
          "schemas/plan-request.json",
          "schemas/plan-result.json",
          "schemas/implement-request.json",
          "schemas/implement-result.json",
          "schemas/autotune-request.json",
          "schemas/autotune-result.json",
          "schemas/profile-request.json",
          "schemas/profile-result.json",
          "schemas/test-request.json",
          "schemas/test-result.json",
      ],
  }

  write_json(target_dir / "package.json", manifest)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Convert MaxKernel / TPU multi-skill plugin to a Coworker package."
  )
  parser.add_argument(
      "source_dir",
      type=Path,
      help="Source directory of the legacy MaxKernel / plugin repository.",
  )
  parser.add_argument(
      "target_dir",
      type=Path,
      help="Target directory for the canonical Coworker package.",
  )
  parser.add_argument(
      "--package-name",
      default=None,
      help="Canonical package name (default: derived from target_dir).",
  )
  parser.add_argument(
      "--display-name",
      default=None,
      help="Human-readable display name for the package.",
  )
  parser.add_argument(
      "--description",
      default=(
          "Orchestrate Pallas TPU kernel development and self-refinement"
          " optimization loops."
      ),
      help="Package description.",
  )
  parser.add_argument(
      "--author",
      default="Accelerator Agents Team",
      help="Package author or team.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  source_dir = args.source_dir.resolve()
  target_dir = args.target_dir.resolve()
  package_name = (
      args.package_name or target_dir.name.replace("_", "-").lower()
  )
  display_name = (
      args.display_name
      or package_name.replace("-", " ").replace("_", " ").title()
  )

  print(
      f"Converting {source_dir} -> {target_dir} for package"
      f" '{package_name}'..."
  )
  create_directories(target_dir)
  convert_instructions(source_dir, target_dir)
  copy_tools(source_dir, target_dir)
  copy_references(source_dir, target_dir)
  generate_environment(target_dir, package_name, display_name)
  generate_schemas(target_dir, package_name)
  generate_package_manifest(
      target_dir, package_name, display_name, args.description, args.author
  )

  print("\nConversion complete! You can verify the package with:")
  print(f"  python3 framework/compiler.py verify {target_dir}")


if __name__ == "__main__":
  main()
