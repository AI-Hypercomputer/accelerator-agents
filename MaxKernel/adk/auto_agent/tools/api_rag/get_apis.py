# Imports
import argparse
import difflib
import importlib
import inspect
import pkgutil
import textwrap

from docstring_parser import parse as parse_docstring

# --- API Discovery (Now using pkgutil for reliability) ---


def discover_apis(api_prefix, recursive):
  """
  Discovers APIs. If recursive, performs a deep walk of all submodules
  using the robust pkgutil method.
  """
  if not recursive:
    try:
      resolve_api(api_prefix)
      return [api_prefix]
    except ImportError as e:
      print(
        f"❌ Error: Could not resolve the single API '{api_prefix}'. Details: {e}"
      )
      return []

  print(
    f"🔎 Deep-recursively discovering APIs under '{api_prefix}' using pkgutil..."
  )
  found_apis = set()
  root_package = api_prefix.split(".")[0]

  try:
    # Import the top-level package to get its filesystem path for pkgutil
    prefix_module = importlib.import_module(api_prefix)
    # __path__ is a list of paths where submodules are located
    search_path = prefix_module.__path__
  except (ImportError, AttributeError):
    print(
      f"❌ Error: '{api_prefix}' is not a valid package or could not be found."
    )
    return []

  # Use a set to hold all module names to inspect, starting with the top-level one
  modules_to_inspect = {api_prefix}

  # pkgutil.walk_packages finds all submodules recursively
  for module_info in pkgutil.walk_packages(
    path=search_path, prefix=api_prefix + "."
  ):
    modules_to_inspect.add(module_info.name)

  # Now, iterate through the complete list of discovered modules
  for module_name in sorted(list(modules_to_inspect)):
    try:
      module = importlib.import_module(module_name)
      for member_name, member in inspect.getmembers(module):
        if member_name.startswith("_"):
          continue

        # Add functions or classes that belong to the root package
        if inspect.isfunction(member) or inspect.isclass(member):
          if (
            hasattr(member, "__module__")
            and member.__module__
            and member.__module__.startswith(root_package)
          ):
            full_api_path = f"{module_name}.{member_name}"
            found_apis.add(full_api_path)
    except Exception:
      # Silently skip any modules that fail to import or be inspected
      continue

  if not found_apis:
    print(
      f"⚠️ Warning: No public functions or classes found under '{api_prefix}'."
    )
  else:
    print(f"✅ Found {len(found_apis)} APIs to document.")

  return sorted(list(found_apis))


# --- Documentation Generation (Unchanged) ---


def resolve_api(api_str):
  """Dynamically resolve an API object from its string name."""
  parts = api_str.split(".")

  # Try to find the longest valid module path
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

  # Traverse the attributes
  obj = module
  for attr in remaining_parts:
    try:
      obj = getattr(obj, attr)
    except AttributeError as e:
      # Suggest close matches if attribute lookup failed
      available_attrs = [a for a in dir(obj) if not a.startswith("_")]
      matches = difflib.get_close_matches(attr, available_attrs)
      if matches:
        raise ImportError(
          f"Could not resolve attribute '{attr}' in '{getattr(obj, '__name__', obj)}'. Did you mean one of these: {matches}? Original error: {e}"
        )
      raise ImportError(f"Could not resolve attribute '{attr}' in '{obj}': {e}")

  return inspect.unwrap(obj)


def get_signature(obj):
  """Gets the signature of a callable object."""
  try:
    return str(inspect.signature(obj))
  except (TypeError, ValueError):
    return "Signature not available"


def get_methods(obj):
  """Gets the public methods of a class."""
  if inspect.isclass(obj):
    return [name for name, _ in inspect.getmembers(obj, inspect.isfunction)]
  return []


def get_attributes(obj):
  """Gets the public attributes of a class."""
  if inspect.isclass(obj):
    return [
      name
      for name, _ in inspect.getmembers(obj)
      if not name.startswith("_")
      and not inspect.isroutine(getattr(obj, name, None))
    ]
  return []


def format_docstring_sections(doc):
  """Parses a docstring into its constituent sections."""
  if not doc:
    return "", [], "", ""

  parsed = parse_docstring(doc)
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
  example_str = "\n".join(ex.description for ex in examples) if examples else ""
  return description, param_strs, return_str, example_str


def generate_definition(api_str):
  """
  Resolves a JAX API, parses its documentation, and returns the
  formatted definition as a string.
  """
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
  if desc != "":
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

  # Add source code extraction
  try:
    source_code = inspect.getsource(obj)
    # Truncate if too long to avoid overwhelming context
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


# --- Main Execution Block (Unchanged) ---

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Parse JAX API definitions and write to a file."
  )
  parser.add_argument(
    "--api",
    type=str,
    required=True,
    help="JAX API string (e.g., jax.numpy.dot) or prefix for recursion (e.g., jax.numpy).",
  )
  parser.add_argument(
    "--output", type=str, required=True, help="Path to the output file."
  )
  parser.add_argument(
    "--recursive",
    action="store_true",
    help="If enabled, finds all public APIs under the given --api prefix.",
  )
  args = parser.parse_args()

  try:
    api_list = discover_apis(args.api, args.recursive)

    if not api_list:
      print("No APIs to process. Exiting.")
      exit()

    with open(args.output, "w", encoding="utf-8") as f:
      total = len(api_list)
      print(f"\nWriting {total} API definitions to '{args.output}'...")
      for i, api_str in enumerate(api_list, 1):
        try:
          print(f"  ({i}/{total}) Processing: {api_str}")
          definition = generate_definition(api_str)
          f.write(definition)
          f.write("\n\n\n")
        except Exception as e:
          print(f"    -> ❌ Skipping '{api_str}': {e}")

    print(f"\n✨ Done. All processed definitions are in '{args.output}'.")

  except IOError as e:
    print(f"❌ A critical file error occurred: {e}")
