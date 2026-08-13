#!/usr/bin/env python3
"""Deterministically applies an autotuning best_config to a kernel file.

Replaces `apply_best_config_agent` (an LLM agent that read the autotune spec
and results, then rewrote the kernel by hand). Substituting `{PLACEHOLDER}`
values back into a code template is a mechanical string operation, so it is
done here as plain code instead of asking a model to transcribe values -- the
same fix applied upstream to ApplyBestConfigAgent (it stopped being a
CustomLlmAgent and became a plain deterministic step).
"""

import argparse
import ast
import json
import re
import sys

# ALL_CAPS only, matching the naming convention autotune_planner_agent.md
# requires for tunable params (e.g. {BLOCK_M}, {BLOCK_N}). This is a
# deliberate restriction, not an oversight: a permissive "{identifier}" regex
# also matches incidental code that happens to look the same, e.g. an
# f-string "f'{x}'" or a one-element set literal "{x}" -- lowercase variable
# names Pallas kernels use constantly. Requiring ALL_CAPS placeholders keeps
# this detectable at all while staying disjoint from normal kernel code.
PLACEHOLDER_RE = re.compile(r"\{([A-Z_][A-Z0-9_]*)\}")

# Matches the same ALL_CAPS convention as PLACEHOLDER_RE, but without the
# curly braces -- used to catch a template that wrote a bare placeholder-
# looking name (e.g. `BLOCK_Q_VAL`) instead of the required `{BLOCK_Q}`. Such
# a name is invisible to PLACEHOLDER_RE (no braces to match) so it silently
# survives substitution untouched and only fails later, as a NameError, when
# someone actually runs the kernel.
_ALL_CAPS_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _bound_names(target: ast.expr):
  """Yields bare names a single assignment-like target binds (recursively)."""
  if isinstance(target, ast.Name):
    yield target.id
  elif isinstance(target, (ast.Tuple, ast.List)):
    for elt in target.elts:
      yield from _bound_names(elt)
  elif isinstance(target, ast.Starred):
    yield from _bound_names(target.value)
  # ast.Attribute / ast.Subscript targets (e.g. `o_ref[0] = ...`) don't bind
  # a new bare name, so they're intentionally not yielded here.


def find_undefined_all_caps_names(code: str) -> set[str]:
  """Finds ALL_CAPS names read in `code` that are never bound anywhere in it.

  This is a whole-file, scope-insensitive approximation (a name bound in one
  function counts as "defined" for the whole file) -- deliberately coarse to
  keep false positives near zero, since the only thing this is meant to
  catch is a leftover, never-substituted autotune placeholder (which reads
  as a name that is never bound ANYWHERE in the file, in any scope).

  Args:
    code: Python source code string.

  Returns:
    A set of ALL_CAPS variable names that are used but undefined.
  """
  tree = ast.parse(code)

  bound = set()
  used = set()

  for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
      bound.add(node.name)
      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        for a in (
            list(args.posonlyargs)
            + list(args.args)
            + list(args.kwonlyargs)
            + ([args.vararg] if args.vararg else [])
            + ([args.kwarg] if args.kwarg else [])
        ):
          bound.add(a.arg)
    elif isinstance(node, ast.Lambda):
      args = node.args
      for a in (
          list(args.posonlyargs)
          + list(args.args)
          + list(args.kwonlyargs)
          + ([args.vararg] if args.vararg else [])
          + ([args.kwarg] if args.kwarg else [])
      ):
        bound.add(a.arg)
    elif isinstance(node, (ast.Assign,)):
      for t in node.targets:
        bound.update(_bound_names(t))
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
      bound.update(_bound_names(node.target))
    elif isinstance(node, getattr(ast, "NamedExpr", ())):
      bound.update(_bound_names(node.target))
    elif isinstance(node, (ast.For, ast.AsyncFor)):
      bound.update(_bound_names(node.target))
    elif isinstance(node, (ast.With, ast.AsyncWith)):
      for item in node.items:
        if item.optional_vars is not None:
          bound.update(_bound_names(item.optional_vars))
    elif isinstance(node, (ast.comprehension,)):
      bound.update(_bound_names(node.target))
    elif isinstance(node, ast.ExceptHandler):
      if node.name:
        bound.add(node.name)
    elif isinstance(node, ast.Import):
      for alias in node.names:
        bound.add(alias.asname or alias.name.split(".")[0])
    elif isinstance(node, ast.ImportFrom):
      for alias in node.names:
        bound.add(alias.asname or alias.name)
    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
      if _ALL_CAPS_NAME_RE.match(node.id):
        used.add(node.id)

  return used - bound


def apply_best_config(
    spec_path: str, results_path: str, output_path: str
) -> None:
  """Applies best_config parameters from results_path to code template in spec_path.

  Args:
    spec_path: Path to autotune spec JSON file.
    results_path: Path to autotune results JSON file.
    output_path: Path where the resulting kernel file will be written.
  """
  with open(spec_path, "r") as f:
    spec = json.load(f)
  code_template = spec.get("code_template", "")
  if not code_template:
    raise ValueError(f"No 'code_template' found in {spec_path}")

  with open(results_path, "r") as f:
    results = json.load(f)
  best_config = results.get("best_config", {})
  if not best_config:
    raise ValueError(f"No 'best_config' found in {results_path}")

  # Placeholder names as they existed in the ORIGINAL template, before
  # substitution -- not a scan of the final text. Scanning the final text for
  # any "{identifier}"-shaped string would false-positive on legitimate code
  # that happens to look like one (e.g. an f-string "f'{x}'" or a one-element
  # set literal "{x}"); this instead only flags names the template itself
  # declared as placeholders.
  template_placeholders = set(PLACEHOLDER_RE.findall(code_template))
  uncovered = template_placeholders - set(best_config.keys())

  final_code = code_template
  for key, value in best_config.items():
    final_code = final_code.replace("{" + str(key) + "}", str(value))

  # A name in `uncovered` only matters if it's still literally present in the
  # output -- e.g. best_config partially overlapping code_template shouldn't
  # false-positive on names that happened to get substituted some other way.
  remaining = sorted(p for p in uncovered if "{" + p + "}" in final_code)
  if remaining:
    raise ValueError(
        f"Placeholders {remaining} appear in code_template but have no "
        f"matching key in best_config {sorted(best_config)} -- they were "
        "left as literal text in the output, which will fail to compile."
    )

  # Belt-and-suspenders check independent of curly-brace syntax: after
  # substitution, the output should not reference any ALL_CAPS name that
  # isn't actually defined anywhere in the file. A template that wrote a
  # bare placeholder-looking name (e.g. `BLOCK_Q_VAL` instead of the
  # required `{BLOCK_Q}`) produces exactly this -- invisible to the
  # curly-brace check above, but a live NameError the moment someone runs
  # the kernel. Catch it here, at generation time, instead.
  undefined = sorted(find_undefined_all_caps_names(final_code))
  if undefined:
    raise ValueError(
        f"{output_path} would reference undefined name(s) {undefined} -- "
        "these look like unresolved autotune placeholders that were "
        "written as bare identifiers instead of the required "
        "`{PLACEHOLDER}` curly-brace syntax (so they never matched "
        "PLACEHOLDER_RE and silently survived substitution untouched). "
        "Fix code_template in the autotune spec to wrap them in curly "
        f"braces, e.g. `{{{undefined[0]}}}`, and re-run."
    )

  with open(output_path, "w") as f:
    f.write(final_code)

  print(f"Applied best_config {best_config} to {output_path}")


def main():
  parser = argparse.ArgumentParser(
      description="Apply an autotuning best_config to a kernel file."
  )
  parser.add_argument(
      "spec_path", help="Path to the autotune spec JSON (code_template)."
  )
  parser.add_argument(
      "results_path", help="Path to the autotune results JSON (best_config)."
  )
  parser.add_argument(
      "output_path", help="Path to write the resulting kernel file to."
  )
  args = parser.parse_args()

  try:
    apply_best_config(args.spec_path, args.results_path, args.output_path)
  except Exception as e:  # pylint: disable=broad-except
    print(f"Failed to apply best config: {e}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()
