#!/usr/bin/env python3
"""Embedded runtime for Coworker agent packages.

Provides runtime execution support for run namespace allocation, artifact
provenance and descriptor generation, and deterministic payload validation.
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import sys
from typing import Any
import uuid

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
compute_digest = utils.compute_digest
format_json = utils.format_json
load_json = utils.load_json
rooted = utils.rooted
validate_instance = utils.validate_instance
validate_package_name = utils.validate_package_name
validate_run_id = utils.validate_run_id
write_json = utils.write_json


def generate_run_id() -> str:
  """Generate a collision-resistant timestamped run identifier."""
  timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
      "%Y%m%dT%H%M%SZ"
  )
  return f"{timestamp}-{uuid.uuid4().hex[:12]}"


def get_installed_package_root(workspace: Path, package: str) -> Path:
  """Validate package and return its resolved installation root in the workspace."""
  validate_package_name(package)
  package_root = rooted(
      workspace.resolve(), f".coworker/{package}", "installed package path"
  )
  environment_path = rooted(
      package_root, "environment.json", "environment path"
  )
  if not environment_path.is_file():
    raise CoworkerError(
        f"package {package} is not installed in {workspace.resolve()}"
    )
  return package_root


def get_installed_run_context(
    workspace: Path, package: str, run_id: str
) -> tuple[Path, Path, dict[str, Any]]:
  """Retrieve and validate the package root, run root directory, and run metadata document."""
  validate_run_id(run_id)
  package_root = get_installed_package_root(workspace, package)
  run_root = rooted(package_root, f"runs/{run_id}", "run path")
  run_document = load_json(run_root / "run.json")

  if (
      run_document.get("run_id") != run_id
      or run_document.get("package") != package
  ):
    raise CoworkerError(
        "run metadata does not match the requested package and run ID"
    )

  return package_root, run_root, run_document


def start_run(workspace: Path, package: str) -> dict[str, Any]:
  """Create a collision-resistant run namespace inside an installed package."""
  package_root = get_installed_package_root(workspace, package)
  environment = load_json(package_root / "environment.json")
  runs_root = rooted(package_root, "runs", "runs path")
  runs_root.mkdir(parents=True, exist_ok=True)

  for _ in range(10):
    run_id = generate_run_id()
    run_root = rooted(runs_root, run_id, "run path")
    try:
      run_root.mkdir()
    except FileExistsError:
      continue

    (run_root / "artifacts").mkdir()
    document = {
        "created_at": (
            datetime.datetime.now(datetime.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        ),
        "environment_revision": environment.get("revision", 1),
        "framework_version": FRAMEWORK_VERSION,
        "package": package,
        "run_id": run_id,
    }
    write_json(run_root / "run.json", document)
    return document

  raise CoworkerError("could not allocate a unique run ID")


def describe_artifact(
    workspace: Path,
    package: str,
    run_id: str,
    file: str,
    schema: str,
    media_type: str,
) -> dict[str, Any]:
  """Describe an artifact belonging to an active or completed package run."""
  if not media_type.strip():
    raise CoworkerError("media type must not be empty")

  package_root, run_root, run_document = get_installed_run_context(
      workspace, package, run_id
  )

  artifact_root = rooted(run_root, "artifacts", "artifact root")
  artifact_file = rooted(artifact_root, file, "artifact file")
  if not artifact_file.is_file():
    raise CoworkerError(f"artifact does not identify a file: {artifact_file}")

  schema_path = rooted(package_root, schema, "artifact schema")
  if not schema_path.is_file():
    raise CoworkerError(
        f"artifact schema does not identify a file: {schema_path}"
    )

  relative = artifact_file.relative_to(artifact_root).as_posix()
  return {
      "environment_revision": run_document["environment_revision"],
      "media_type": media_type,
      "run_id": run_id,
      "schema": schema,
      "sha256": compute_digest(artifact_file),
      "uri": f"workspace://{package}/runs/{run_id}/artifacts/{relative}",
  }


def validate_json_file(schema_path: Path, instance_path: Path) -> list[str]:
  """Validate a JSON instance against a JSON schema file."""
  instance = load_json(instance_path)
  schema = load_json(schema_path)
  return validate_instance(instance, schema)


def build_runtime_parser() -> argparse.ArgumentParser:
  """Construct parser for standalone runtime commands."""
  parser = argparse.ArgumentParser(
      prog="coworker-runtime",
      description="Embedded runtime execution operations for Coworker packages.",
  )
  sub = parser.add_subparsers(dest="command", required=True)

  # start-run
  p_start = sub.add_parser(
      "start-run", help="Initialize a new run namespace for a package"
  )
  p_start.add_argument(
      "--workspace", type=Path, required=True, help="Workspace root directory"
  )
  p_start.add_argument(
      "--package", required=True, help="Installed package name"
  )

  # describe-artifact
  p_artifact = sub.add_parser(
      "describe-artifact", help="Generate an artifact provenance descriptor"
  )
  p_artifact.add_argument(
      "--workspace", type=Path, required=True, help="Workspace root directory"
  )
  p_artifact.add_argument(
      "--package", required=True, help="Installed package name"
  )
  p_artifact.add_argument("--run-id", required=True, help="Run ID")
  p_artifact.add_argument(
      "--file",
      required=True,
      help="Artifact file path relative to run artifacts/",
  )
  p_artifact.add_argument(
      "--schema", required=True, help="Schema path relative to package root"
  )
  p_artifact.add_argument(
      "--media-type", required=True, help="Media type of the artifact"
  )

  # validate
  p_validate = sub.add_parser(
      "validate", help="Validate a JSON instance against a schema"
  )
  p_validate.add_argument(
      "--schema", type=Path, required=True, help="Path to schema JSON file"
  )
  p_validate.add_argument(
      "instance", type=Path, help="Path to JSON instance file to validate"
  )

  return parser


def main(argv: list[str] | None = None) -> int:
  """Main entrypoint for standalone runtime operations."""
  parser = build_runtime_parser()
  args = parser.parse_args(argv)

  try:
    if args.command == "start-run":
      run_doc = start_run(args.workspace, args.package)
      print(format_json(run_doc))

    elif args.command == "describe-artifact":
      descriptor = describe_artifact(
          workspace=args.workspace,
          package=args.package,
          run_id=args.run_id,
          file=args.file,
          schema=args.schema,
          media_type=args.media_type,
      )
      print(format_json(descriptor))

    elif args.command == "validate":
      errors = validate_json_file(args.schema, args.instance)
      if errors:
        print("\n".join(errors), file=sys.stderr)
        return 2

    return 0
  except CoworkerError as exc:
    print(f"runtime: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
