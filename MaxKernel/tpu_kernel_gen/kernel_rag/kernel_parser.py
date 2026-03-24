#!/usr/bin/env python3
"""
Pallas Kernel Parser

A script that uses tree-sitter to parse Python files and identify JAX Pallas kernels.
This script searches for:
- @pallas_call decorators
- pallas.pallas_call function calls
- jax.experimental.pallas imports and usage
- pl.* function calls (common Pallas alias)
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
  import tree_sitter_python
  from tree_sitter import Language, Node, Parser
except ImportError:
  print("Error: tree-sitter-python is required. Install with:")
  print("pip install tree-sitter tree-sitter-python")
  sys.exit(1)


class CodeLines:
  """Represents a range of lines in source code."""

  def __init__(self, start: int, end: int):
    self.start = start
    self.end = end

  def __str__(self):
    return f"{self.start}-{self.end}"

  def __repr__(self):
    return f"CodeLines({self.start}, {self.end})"


@dataclass
class PallasKernel:
  """Represents a discovered Pallas kernel."""

  file_path: Path
  function_name: str
  call_lines: CodeLines  # The lines where the kernel is called
  call_code: Optional[str] = None  # The full pallas_call invocation
  decorator_args: Optional[str] = None
  definition_lines: List[CodeLines] = None  # The lines where the kernel function is defined
  definition_code: Optional[str] = None  # The actual kernel function definition


class PallasKernelFinder:
  """Finds Pallas kernels in Python source code using tree-sitter."""

  def __init__(self):
    self.language = Language(tree_sitter_python.language())
    self.parser = Parser(self.language)

    # Patterns to identify Pallas usage
    self.pallas_imports = {
      "jax.experimental.pallas",
      "jax.experimental.pallas.pallas_call",
      "jax.experimental.pallas as pallas",
      "jax.experimental.pallas as pl",
    }

    self.pallas_functions = {"pallas_call", "pallas.pallas_call", "pl.pallas_call"}

  def parse_file(self, file_path: Path) -> List[PallasKernel]:
    """Parse a single Python file and extract Pallas kernels."""
    try:
      with open(file_path, "r", encoding="utf-8") as f:
        source_code = f.read()
    except (UnicodeDecodeError, IOError) as e:
      print(f"Warning: Could not read {file_path}: {e}")
      return []

    tree = self.parser.parse(bytes(source_code, "utf8"))
    root_node = tree.root_node

    kernels = []

    # Check if file has Pallas imports
    if not self._has_pallas_imports(root_node, source_code):
      return kernels

    # Find kernels
    kernels.extend(self._find_kernels(root_node, source_code, file_path))

    return kernels

  def find_kernels_in_directory(self, directory: Path, recursive: bool = True) -> List[PallasKernel]:
    """Find all Pallas kernels in a directory."""
    all_kernels = []

    if not directory.exists():
      print(f"Error: Directory {directory} does not exist")
      return all_kernels

    if not directory.is_dir():
      print(f"Error: {directory} is not a directory")
      return all_kernels

    # Find all Python files
    pattern = "**/*.py" if recursive else "*.py"
    python_files = list(directory.glob(pattern))

    print(f"Scanning {len(python_files)} Python files...")

    for py_file in python_files:
      if py_file.is_file():
        kernels = self.parse_file(py_file)
        all_kernels.extend(kernels)
        if kernels:
          print(f"Found {len(kernels)} kernel(s) in {py_file}")

    return all_kernels

  def _has_pallas_imports(self, root_node: Node, source_code: str) -> bool:
    """Check if the file imports Pallas modules."""
    query = self.language.query(
      """
        (import_statement
          name: (dotted_name) @import_name)
        (import_from_statement
          module_name: (dotted_name) @module_name)
        (import_from_statement
          module_name: (dotted_name) @module_name
          name: (dotted_name) @import_name)
        (import_from_statement
          module_name: (dotted_name) @module_name
          name: (aliased_import 
            name: (dotted_name) @import_name
            alias: (identifier) @alias))
        """
    )

    captures = query.captures(root_node)
    for _, nodes in captures.items():
      for node in nodes:
        text = source_code[node.start_byte : node.end_byte]
        if "pallas" in text:
          print("Found Pallas import:", text)
          return True

    return False

  def _find_kernels(self, root_node: Node, source_code: str, file_path: Path) -> List[PallasKernel]:
    """Find direct calls to pallas_call functions."""
    kernels = []

    query = self.language.query(
      """
        (call
          function: (identifier) @function_name
          arguments: (argument_list) @arguments) @call_node
          
        (call
          function: (attribute
            object: (identifier) @module_name
            attribute: (identifier) @function_name)
          arguments: (argument_list) @arguments) @call_node
        """
    )

    captures = query.captures(root_node)

    # Group captures by call node
    call_nodes = {}
    for capture_name, nodes in captures.items():
      for node in nodes:
        if capture_name == "call_node":
          call_nodes[node] = {}
        else:
          # Find the parent call node
          parent_call = node
          while parent_call and parent_call not in call_nodes:
            parent_call = parent_call.parent
            if parent_call and parent_call.type == "call":
              break

          if parent_call and parent_call in call_nodes:
            if capture_name not in call_nodes[parent_call]:
              call_nodes[parent_call][capture_name] = []
            call_nodes[parent_call][capture_name].append(node)

    for call_node, captures_dict in call_nodes.items():
      function_names = captures_dict.get("function_name", [])
      module_names = captures_dict.get("module_name", [])
      arguments_list = captures_dict.get("arguments", [])

      if not function_names:
        continue

      function_name = source_code[function_names[0].start_byte : function_names[0].end_byte]

      # Build full function name with module if present
      full_function_name = function_name
      for module_node in module_names:
        module_name = source_code[module_node.start_byte : module_node.end_byte]
        full_function_name = f"{module_name}.{function_name}"
        break

      # Check if this is a pallas_call
      if (
        function_name in self.pallas_functions
        or full_function_name in self.pallas_functions
        or "pallas_call" in function_name
      ):
        start_point = call_node.start_point
        end_point = call_node.end_point

        # Extract kernel function name by manually parsing the arguments
        kernel_func_name = "unknown_kernel"

        if arguments_list:
          arguments_node = arguments_list[0]

          # Look through all argument children to find the first meaningful argument
          for i, arg_child in enumerate(arguments_node.children):
            if arg_child.type == "identifier":
              # Direct identifier argument - check if it's a variable or function name
              potential_name = source_code[arg_child.start_byte : arg_child.end_byte]
              # Try to resolve if this is a variable assignment
              resolved_name = self._resolve_kernel_name(root_node, source_code, potential_name, call_node)
              kernel_func_name = resolved_name if resolved_name else potential_name
              break
            elif arg_child.type == "call":
              # Call argument (like functools.partial)
              kernel_func_name = self._extract_function_from_call(arg_child, source_code)
              break
            elif arg_child.type == "keyword_argument":
              # Check if this is kernel= keyword argument
              keyword_name_node = None
              keyword_value_node = None
              for grandchild in arg_child.children:
                if grandchild.type == "identifier" and not keyword_name_node:
                  keyword_name_node = grandchild
                elif grandchild.type in ["identifier", "call"] and keyword_name_node:
                  keyword_value_node = grandchild
                  break

              if keyword_name_node:
                keyword_name = source_code[keyword_name_node.start_byte : keyword_name_node.end_byte]
                if keyword_name == "kernel" and keyword_value_node:
                  if keyword_value_node.type == "identifier":
                    potential_name = source_code[keyword_value_node.start_byte : keyword_value_node.end_byte]
                    resolved_name = self._resolve_kernel_name(
                      root_node,
                      source_code,
                      potential_name,
                      call_node,
                    )
                    kernel_func_name = resolved_name if resolved_name else potential_name
                  elif keyword_value_node.type == "call":
                    kernel_func_name = self._extract_function_from_call(keyword_value_node, source_code)
                  break

        # Extract the complete pallas call body (improved formatting)
        pallas_call_text = source_code[call_node.start_byte : call_node.end_byte]

        # Find complete kernel definition recursively
        definition_lines, definition_code = self._find_complete_kernel_definition(
          root_node, source_code, kernel_func_name, call_node
        )

        kernel = PallasKernel(
          file_path=file_path,
          function_name=kernel_func_name,
          call_lines=CodeLines(start_point.row + 1, end_point.row + 1),
          call_code=pallas_call_text,
          definition_lines=definition_lines,
          definition_code=definition_code,
        )

        kernels.append(kernel)

    return kernels

  def _find_complete_kernel_definition(
    self, root_node: Node, source_code: str, kernel_name: str, call_context: Node
  ) -> tuple[List[CodeLines], Optional[str]]:
    """Recursively find all parts of a kernel definition."""
    visited_functions = set()
    all_definitions = []

    def find_function_definition_recursive(func_name: str, context_node: Node = None) -> None:
      if func_name in visited_functions:
        return
      visited_functions.add(func_name)

      # Find the function definition - try assignment first, then regular function
      func_def_node, func_def_lines = self._find_function_definition(root_node, source_code, func_name, context_node)

      # If we didn't find an assignment, try to find a regular function definition
      if not func_def_node:
        func_def_node, func_def_lines = self._find_function_definition(
          root_node,
          source_code,
          func_name,
          None,  # No context for regular functions
        )

      if func_def_node and func_def_lines:
        # Only include actual function definitions or assignments that create functions
        # Skip individual variable assignments unless they're function-creating assignments
        should_include = False

        if func_def_node.type == "function_definition":
          should_include = True
        elif func_def_node.type == "assignment":
          # Check if this assignment creates a function (like functools.partial)
          assignment_value = None
          for child in func_def_node.children:
            if child.type != "identifier" and child.type != "=":
              assignment_value = child
              break

          if assignment_value:
            assignment_text = source_code[assignment_value.start_byte : assignment_value.end_byte]
            # Include if it's a function-creating assignment
            if any(keyword in assignment_text for keyword in ["functools.partial", "partial", "lambda"]):
              should_include = True

        if should_include:
          # Extract the function code
          func_code = source_code[func_def_node.start_byte : func_def_node.end_byte]
          all_definitions.append((func_def_lines, func_code, func_def_node))

          # Find all function calls within this definition
          called_functions = self._extract_function_calls(func_def_node, source_code)

          # Recursively process called functions
          for called_func in called_functions:
            find_function_definition_recursive(called_func, func_def_node)

    # Start the recursive search
    find_function_definition_recursive(kernel_name, call_context)

    # If we didn't find anything and the kernel_name looks like it might have come from
    # functools.partial extraction, also try searching without context
    if not all_definitions and call_context:
      find_function_definition_recursive(kernel_name, None)

    if not all_definitions:
      return [], None

    # Sort definitions by their position in the file (top to bottom)
    all_definitions.sort(key=lambda x: x[0].start)

    # Extract relevant imports
    imports_code = self._extract_relevant_imports(root_node, source_code, all_definitions)

    # Combine imports and definition codes
    combined_code_parts = []
    if imports_code:
      combined_code_parts.append(imports_code)

    definition_code = "\n\n".join(def_code for _, def_code, _ in all_definitions)
    combined_code_parts.append(definition_code)

    combined_code = "\n\n".join(combined_code_parts)
    definition_lines = [def_lines for def_lines, _, _ in all_definitions]

    return definition_lines, combined_code

  def _extract_relevant_imports(self, root_node: Node, source_code: str, definitions: list) -> str:
    """Extract imports that are relevant to the kernel definitions."""
    # Get all imports from the file
    import_query = self.language.query(
      """
        (import_statement) @import
        (import_from_statement) @import
        """
    )

    captures = import_query.captures(root_node)
    all_imports = []

    for capture_name, nodes in captures.items():
      if capture_name == "import":
        for node in nodes:
          import_text = source_code[node.start_byte : node.end_byte]
          all_imports.append(import_text)

    # Collect all identifiers used in the kernel definitions
    used_identifiers = set()
    for _, def_code, def_node in definitions:
      self._collect_identifiers_from_node(def_node, source_code, used_identifiers)

    # Filter imports to include only those that are likely relevant
    relevant_imports = []
    for import_text in all_imports:
      if self._is_import_relevant(import_text, used_identifiers):
        relevant_imports.append(import_text)

    return "\n".join(relevant_imports) if relevant_imports else ""

  def _collect_identifiers_from_node(self, node: Node, source_code: str, identifiers: set):
    """Recursively collect all identifiers from a node."""
    if node.type == "identifier":
      identifier = source_code[node.start_byte : node.end_byte]
      identifiers.add(identifier)

    for child in node.children:
      self._collect_identifiers_from_node(child, source_code, identifiers)

  def _is_import_relevant(self, import_text: str, used_identifiers: set) -> bool:
    """Check if an import is relevant to the kernel definitions."""
    # Always include these essential imports for Pallas kernels
    essential_modules = {
      "jax",
      "jnp",
      "numpy",
      "np",
      "functools",
      "typing",
      "collections",
      "pallas",
      "pl",
      "pltpu",
      "lax",
    }

    # Check if import mentions essential modules
    for module in essential_modules:
      if module in import_text:
        return True

    # Check if any used identifiers match import names
    # Extract potential module aliases and names from import
    import_parts = import_text.split()
    for part in import_parts:
      # Remove common keywords and punctuation
      clean_part = part.strip("(),").split(".")[0]
      if clean_part in ["import", "from", "as"]:
        continue
      if clean_part in used_identifiers:
        return True

    return False

  def _find_function_definition(
    self,
    root_node: Node,
    source_code: str,
    func_name: str,
    context_node: Node = None,
  ) -> tuple[Optional[Node], Optional[CodeLines]]:
    """Find the definition of a function, considering context for variable assignments."""

    # First, check if this is a variable assignment (like functools.partial)
    if context_node:
      assignment_def = self._find_variable_assignment(root_node, source_code, func_name, context_node)
      if assignment_def:
        return assignment_def

    # Search for regular function definitions
    query = self.language.query(
      """
        (function_definition
          name: (identifier) @func_name) @func_def
        """
    )

    captures = query.captures(root_node)

    for capture_name, nodes in captures.items():
      for node in nodes:
        if capture_name == "func_name":
          name = source_code[node.start_byte : node.end_byte]
          if name == func_name:
            # Find the parent function definition
            func_def_node = node.parent
            while func_def_node and func_def_node.type != "function_definition":
              func_def_node = func_def_node.parent

            if func_def_node:
              start_point = func_def_node.start_point
              end_point = func_def_node.end_point
              return func_def_node, CodeLines(start_point.row + 1, end_point.row + 1)

    return None, None

  def _find_variable_assignment(
    self, root_node: Node, source_code: str, var_name: str, context_node: Node
  ) -> tuple[Optional[Node], Optional[CodeLines]]:
    """Find variable assignments like 'kernel = functools.partial(...)'."""

    query = self.language.query(
      """
        (assignment
          left: (identifier) @var_name
          right: (_) @assignment_value) @assignment
        """
    )

    captures = query.captures(root_node)
    assignments = {}

    for capture_name, nodes in captures.items():
      for node in nodes:
        if capture_name == "assignment":
          assignments[node] = {}
        else:
          parent_assignment = node
          while parent_assignment and parent_assignment not in assignments:
            parent_assignment = parent_assignment.parent
            if parent_assignment and parent_assignment.type == "assignment":
              break

          if parent_assignment and parent_assignment in assignments:
            if capture_name not in assignments[parent_assignment]:
              assignments[parent_assignment][capture_name] = []
            assignments[parent_assignment][capture_name].append(node)

    # Look for assignments to our variable name that are before the context
    # Find the closest assignment before the context
    context_line = context_node.start_point.row if context_node else float("inf")
    closest_assignment = None
    closest_distance = float("inf")

    for assignment_node, assignment_captures in assignments.items():
      var_names = assignment_captures.get("var_name", [])
      if not var_names:
        continue

      assigned_var_name = source_code[var_names[0].start_byte : var_names[0].end_byte]
      assignment_line = assignment_node.start_point.row

      # Check if this assignment matches our variable and is before the context
      if assigned_var_name == var_name and assignment_line < context_line:
        distance = context_line - assignment_line
        if distance < closest_distance:
          closest_distance = distance
          closest_assignment = assignment_node

    if closest_assignment:
      start_point = closest_assignment.start_point
      end_point = closest_assignment.end_point
      return closest_assignment, CodeLines(start_point.row + 1, end_point.row + 1)

    return None, None

  def _extract_function_calls(self, func_def_node: Node, source_code: str) -> List[str]:
    """Extract all function calls within a function definition."""
    function_calls = set()

    # Extract regular function calls
    query = self.language.query(
      """
        (call
          function: (identifier) @func_name)
        (call
          function: (attribute
            attribute: (identifier) @func_name))
        """
    )

    captures = query.captures(func_def_node)

    for capture_name, nodes in captures.items():
      if capture_name == "func_name":
        for node in nodes:
          func_name = source_code[node.start_byte : node.end_byte]
          # Filter out built-in functions and common modules
          if not self._is_builtin_or_standard_lib(func_name):
            function_calls.add(func_name)

    # Extract function references from functools.partial and other function references
    # This query finds all identifiers that could be function references
    ref_query = self.language.query(
      """
        (call
          arguments: (argument_list
            (identifier) @arg_func))
        """
    )

    ref_captures = ref_query.captures(func_def_node)

    for capture_name, nodes in ref_captures.items():
      if capture_name == "arg_func":
        for node in nodes:
          func_name = source_code[node.start_byte : node.end_byte]
          # Check if this looks like a function name and isn't a builtin
          if (
            func_name.replace("_", "")
            .replace("0", "")
            .replace("1", "")
            .replace("2", "")
            .replace("3", "")
            .replace("4", "")
            .replace("5", "")
            .replace("6", "")
            .replace("7", "")
            .replace("8", "")
            .replace("9", "")
            .isalpha()
            and not self._is_builtin_or_standard_lib(func_name)
            and func_name not in ["True", "False", "None"]
          ):
            # Check if this is used in a context that suggests it's a function reference
            # Look at the parent call to see if it's functools.partial or similar
            parent = node.parent
            while parent and parent.type != "call":
              parent = parent.parent

            if parent:
              # Check if this is the first argument to a call (likely a function reference)
              arg_list = None
              for child in parent.children:
                if child.type == "argument_list":
                  arg_list = child
                  break

              if arg_list and arg_list.children:
                # Find the first actual argument (skip parentheses and commas)
                first_arg = None
                for child in arg_list.children:
                  if child.type == "identifier":
                    first_arg = child
                    break

                # If this identifier is the first argument, it's likely a function reference
                if first_arg and first_arg == node:
                  function_calls.add(func_name)

    return list(function_calls)

  def _is_builtin_or_standard_lib(self, func_name: str) -> bool:
    """Check if a function name is a built-in or standard library function."""
    builtins = {
      "print",
      "len",
      "range",
      "enumerate",
      "zip",
      "map",
      "filter",
      "sum",
      "max",
      "min",
      "abs",
      "round",
      "int",
      "float",
      "str",
      "bool",
      "list",
      "dict",
      "set",
      "tuple",
      "isinstance",
      "issubclass",
      "hasattr",
      "getattr",
      "setattr",
      "delattr",
      "all",
      "any",
      "ord",
      "chr",
      "hex",
      "oct",
      "bin",
      "divmod",
      "pow",
    }

    standard_modules = {
      "jnp",
      "jax",
      "lax",
      "pl",
      "pltpu",
      "math",
      "functools",
      "dataclasses",
    }

    return func_name in builtins or any(func_name.startswith(mod + ".") for mod in standard_modules)

  def _extract_function_from_call(self, call_node: Node, source_code: str) -> str:
    """Extract the actual function name from a call expression like functools.partial(func, ...)."""
    # Check if this is a functools.partial call
    call_text = source_code[call_node.start_byte : call_node.end_byte]

    if "partial" in call_text:
      # Look for the first argument to partial, which should be the function
      # Use a more comprehensive query to find the first argument
      query = self.language.query(
        """
            (call
              arguments: (argument_list
                (identifier) @first_arg))
            """
      )

      captures = query.captures(call_node)

      for capture_name, nodes in captures.items():
        if capture_name == "first_arg":
          for i, node in enumerate(nodes):
            func_name = source_code[node.start_byte : node.end_byte]
            # Return the first non-module identifier
            if func_name not in ["functools", "partial"]:
              return func_name

      # If query approach doesn't work, try manual parsing
      # Find the argument list and extract the first identifier
      for child in call_node.children:
        if child.type == "argument_list":
          for i, arg_child in enumerate(child.children):
            if arg_child.type == "identifier":
              func_name = source_code[arg_child.start_byte : arg_child.end_byte]
              if func_name not in ["functools", "partial"]:
                return func_name

    # For other types of calls, try to extract any function-like identifier
    # Look for identifiers within the call that could be function names
    identifiers = []

    def collect_identifiers(node):
      if node.type == "identifier":
        identifier_text = source_code[node.start_byte : node.end_byte]
        # Filter out obvious non-function names
        if identifier_text not in [
          "functools",
          "partial",
          "jax",
          "pl",
          "jnp",
          "lax",
        ]:
          identifiers.append(identifier_text)
      for child in node.children:
        collect_identifiers(child)

    collect_identifiers(call_node)

    # Return the first reasonable identifier, or fall back to the whole call text
    if identifiers:
      return identifiers[0]

    # If all else fails, return the whole call text
    return call_text

  def _resolve_kernel_name(self, root_node: Node, source_code: str, var_name: str, call_context: Node) -> Optional[str]:
    """Resolve a variable name to the actual kernel function name if it's an assignment."""

    # Find variable assignments before the call context
    query = self.language.query(
      """
        (assignment
          left: (identifier) @var_name
          right: (_) @assignment_value) @assignment
        """
    )

    captures = query.captures(root_node)
    assignments = {}

    for capture_name, nodes in captures.items():
      for node in nodes:
        if capture_name == "assignment":
          assignments[node] = {}
        else:
          parent_assignment = node
          while parent_assignment and parent_assignment not in assignments:
            parent_assignment = parent_assignment.parent
            if parent_assignment and parent_assignment.type == "assignment":
              break

          if parent_assignment and parent_assignment in assignments:
            if capture_name not in assignments[parent_assignment]:
              assignments[parent_assignment][capture_name] = []
            assignments[parent_assignment][capture_name].append(node)

    # Look for the most recent assignment to var_name before the call context
    context_line = call_context.start_point.row if call_context else float("inf")
    closest_assignment = None
    closest_distance = float("inf")

    for assignment_node, assignment_captures in assignments.items():
      var_names = assignment_captures.get("var_name", [])
      assignment_values = assignment_captures.get("assignment_value", [])

      if not var_names:
        continue

      assigned_var_name = source_code[var_names[0].start_byte : var_names[0].end_byte]
      assignment_line = assignment_node.start_point.row

      # Check if this assignment matches our variable and is before the context
      if assigned_var_name == var_name and assignment_line < context_line:
        distance = context_line - assignment_line
        if distance < closest_distance:
          closest_distance = distance
          closest_assignment = (
            assignment_node,
            assignment_values[0] if assignment_values else None,
          )

    if closest_assignment:
      assignment_node, assignment_value = closest_assignment
      if assignment_value:
        assignment_text = source_code[assignment_value.start_byte : assignment_value.end_byte]

        # Handle functools.partial case
        if "functools.partial" in assignment_text or "partial" in assignment_text:
          return self._extract_function_from_call(assignment_value, source_code)

        # Handle direct function assignment
        elif assignment_value.type == "identifier":
          return source_code[assignment_value.start_byte : assignment_value.end_byte]

        # Handle other call expressions
        elif assignment_value.type == "call":
          return self._extract_function_from_call(assignment_value, source_code)

    return None


def print_kernel_summary(kernels: List[PallasKernel]):
  """Print a summary of found kernels."""
  if not kernels:
    print("No Pallas kernels found.")
    return

  print(f"\n=== Found {len(kernels)} Pallas Kernels ===\n")

  # Group by file
  kernels_by_file = {}
  for kernel in kernels:
    if kernel.file_path not in kernels_by_file:
      kernels_by_file[kernel.file_path] = []
    kernels_by_file[kernel.file_path].append(kernel)

  for file_path, file_kernels in kernels_by_file.items():
    print(f"📁 {file_path}")
    for kernel in file_kernels:
      print(f"  🔧 {kernel.function_name}")
      if kernel.definition_lines:
        definition_ranges = ", ".join([f"lines {lines.start}-{lines.end}" for lines in kernel.definition_lines])
        print(f"     Kernel definition: {definition_ranges}")
      print(f"     Pallas call usage: lines {kernel.call_lines.start}-{kernel.call_lines.end}")
      print()


def main():
  parser = argparse.ArgumentParser(
    description="Find JAX Pallas kernels in Python source code",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python kernel_parser.py /path/to/project
  python kernel_parser.py /path/to/project --no-recursive
  python kernel_parser.py /path/to/project --output kernels.txt
        """,
  )

  parser.add_argument("directory", type=Path, help="Directory to search for Pallas kernels")

  parser.add_argument(
    "--no-recursive",
    action="store_true",
    help="Do not search subdirectories recursively",
  )

  parser.add_argument("--output", type=Path, help="Output file to write results (optional)")

  parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

  args = parser.parse_args()

  finder = PallasKernelFinder()
  kernels = finder.find_kernels_in_directory(args.directory, recursive=not args.no_recursive)

  if args.output:
    # Write results to file
    with open(args.output, "w") as f:
      f.write(f"Pallas Kernels found in {args.directory}\n")
      f.write("=" * 50 + "\n\n")

      for kernel in kernels:
        f.write(f"File: {kernel.file_path}\n")
        f.write(f"Kernel Function: {kernel.function_name}\n")
        f.write(f"Pallas Call Lines: {kernel.call_line_start}-{kernel.call_line_end}\n")
        if kernel.def_line_start:
          f.write(f"Kernel Definition Lines: {kernel.def_line_start}-{kernel.def_line_end}\n")
        if kernel.decorator_args:
          f.write(f"Pallas Call: {kernel.decorator_args}\n")

        if kernel.kernel_definition:
          f.write("\nKernel Definition:\n")
          f.write(f"{kernel.kernel_definition}\n")

        if kernel.pallas_call_body:
          f.write("\nPallas Call:\n")
          f.write(f"{kernel.pallas_call_body}\n")

        f.write("\n" + "-" * 40 + "\n\n")

    print(f"Results written to {args.output}")

  print_kernel_summary(kernels)


if __name__ == "__main__":
  main()
