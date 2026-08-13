#!/usr/bin/env python3
"""Standalone CLI for looking up a Python API's signature/docstring/source."""

import argparse
import difflib
import importlib
import inspect
import sys
import textwrap

import docstring_parser


def resolve_api(api_str: str):
  """Dynamically resolve an API object from its dotted string name."""
  parts = api_str.split(".")

  module = None
  remaining_parts = []
  for i in range(len(parts), 0, -1):
    module_path = ".".join(parts[:i])
    try:
      module = importlib.import_module(module_path)
      remaining_parts = parts[i:]
      break
    except ImportError:
      continue

  if module is None:
    raise ImportError(f"Could not resolve any part of {api_str} as a module.")

  obj = module
  for attr in remaining_parts:
    try:
      obj = getattr(obj, attr)
    except AttributeError as e:
      available_attrs = [a for a in dir(obj) if not a.startswith("_")]
      matches = difflib.get_close_matches(attr, available_attrs)
      if matches:
        raise ImportError(
            f"Could not resolve attribute '{attr}' in"
            f" '{getattr(obj, '__name__', obj)}'. Did you mean one of these:"
            f" {matches}? Original error: {e}"
        ) from e
      raise ImportError(
          f"Could not resolve attribute '{attr}' in '{obj}': {e}"
      ) from e

  return inspect.unwrap(obj)  # pytype: disable=bad-argument-type  # pylint: disable=g-blanket-type-suppression


def get_signature(obj) -> str:
  try:
    return str(inspect.signature(obj))
  except (TypeError, ValueError):
    return "Signature not available"


def get_methods(obj):
  if inspect.isclass(obj):
    return [name for name, _ in inspect.getmembers(obj, inspect.isfunction)]
  return []


def get_attributes(obj):
  if inspect.isclass(obj):
    return [
        name
        for name, _ in inspect.getmembers(obj)
        if not name.startswith("_")
        and not inspect.isroutine(getattr(obj, name, None))
    ]
  return []


def format_docstring_sections(doc):
  """Parses docstring sections and returns formatted descriptions."""
  if not doc:
    return "", [], "", ""

  parsed = docstring_parser.parse(doc)
  description = parsed.description
  parameters = parsed.params
  returns = parsed.returns
  examples = parsed.examples

  param_strs = (
      [f"  - **{p.arg_name}**: {p.description}" for p in parameters]
      if parameters
      else []
  )
  return_str = f"{returns.description}" if returns else ""
  example_str = (
      "\n".join(ex.description for ex in examples if ex.description is not None)
      if examples
      else ""
  )
  return description, param_strs, return_str, example_str


def generate_definition(api_str: str) -> str:
  """Resolves an API, parses its documentation, and returns a formatted definition."""
  obj = resolve_api(api_str)
  doc = inspect.getdoc(obj)
  signature = get_signature(obj)
  methods = get_methods(obj)
  attributes = get_attributes(obj)
  desc, param_strs, return_str, example_str = format_docstring_sections(doc)

  lines = []

  def add_line(text=""):
    lines.append(text)

  add_line("-" * 80)
  add_line(f"### API: {api_str}")
  add_line(f"\n**Signature**:\n`{api_str}{signature}`")
  if desc:
    add_line(f"\n**Description**:\n{textwrap.fill(desc, width=80)}")

  if param_strs:
    add_line("\n**Parameters**:")
    for p in param_strs:
      add_line(p)

  if attributes:
    add_line("\n**Attributes**:")
    for a in attributes:
      add_line(f"  - `{a}`")

  if methods:
    add_line("\n**Methods**:")
    for m in methods:
      add_line(f"  - `{m}`")

  if return_str:
    add_line(f"\n**Returns**:\n  {return_str}")

  if example_str:
    add_line(f"\n**Examples**:\n```python\n{example_str}\n```")

  try:
    source_code = inspect.getsource(obj)
    if len(source_code) > 2000:
      trunc_point = source_code.rfind("\n", 0, 2000)
      if trunc_point == -1:
        trunc_point = 2000
      source_code = (
          source_code[:trunc_point]
          + "\n... [Source code truncated due to length] ..."
      )
    add_line(f"\n**Source Code**:\n```python\n{source_code}\n```")
  except (TypeError, OSError) as e:
    add_line(f"\n**Source Code**: Not available ({e})")

  add_line("-" * 80)
  return "\n".join(lines)


def main():
  parser = argparse.ArgumentParser(
      description="Look up a Python API's signature, docstring, and source."
  )
  parser.add_argument(
      "api_name",
      help=(
          "Fully-qualified dotted API name, e.g."
          " jax.experimental.pallas.pallas_call"
      ),
  )
  args = parser.parse_args()

  print(generate_definition(args.api_name))


if __name__ == "__main__":
  try:
    main()
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"Failed to resolve API '{sys.argv[-1]}': {e}", file=sys.stderr)
    sys.exit(1)
