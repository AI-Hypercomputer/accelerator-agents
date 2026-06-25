"""Utility functions for Beam Search Orchestrator."""

import ast
import logging

logger = logging.getLogger(__name__)


class PallasASTNormalizer(ast.NodeTransformer):
  """AST NodeTransformer that normalizes variable names and strips metadata."""

  def __init__(self):
    super().__init__()
    self.var_map = {}
    self.var_counter = 0
    # Reserved names that should not be renamed (modules, main entrypoint, key JAX APIs)
    self.reserved_names = {
        "jax",
        "jnp",
        "lax",
        "pl",
        "pltpu",
        "solution",
        "mean",
        "square",
        "rsqrt",
        "astype",
        "dtype",
        "float32",
        "bfloat16",
    }

  def _get_replacement_name(self, name: str) -> str:
    if name in self.reserved_names:
      return name
    if name not in self.var_map:
      self.var_map[name] = f"v{self.var_counter}"
      self.var_counter += 1
    return self.var_map[name]

  def visit_Name(self, node):
    node.id = self._get_replacement_name(node.id)
    return self.generic_visit(node)

  def visit_arg(self, node):
    node.arg = self._get_replacement_name(node.arg)
    return self.generic_visit(node)

  def visit_FunctionDef(self, node):
    # Do not rename the primary solution entrypoint function
    if node.name != "solution":
      node.name = self._get_replacement_name(node.name)
    # Remove docstring expressions if they exist at the start of the function body
    if ast.get_docstring(node):
      if isinstance(node.body[0], ast.Expr) and isinstance(
          node.body[0].value, ast.Constant
      ):
        node.body.pop(0)
    return self.generic_visit(node)


def normalize_ast_code(source_code: str) -> str:
  """Parses, normalizes, and unparses Python code to compare structural equivalence."""
  try:
    tree = ast.parse(source_code)
    normalizer = PallasASTNormalizer()
    normalized_tree = normalizer.visit(tree)
    ast.fix_missing_locations(normalized_tree)
    return ast.unparse(normalized_tree)
  except Exception as e:
    logger.error(f"AST Normalization failed: {e}")
    # Return basic whitespace-stripped string fallback on error
    return "".join(source_code.split())
