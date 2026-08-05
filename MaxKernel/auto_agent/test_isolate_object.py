"""
Unit tests and demonstration script for isolate_object.py.

This test file illustrates how isolate_object.py:
1. Distinguishes between external libraries (e.g. jax, jax.numpy) and local workspace modules.
2. Recursively crawls local workspace modules to inline helper functions and constants.
3. Topologically sorts extracted dependencies so execution order is valid.
4. Identifies required local import files for automated dependency discovery.

Run this test directly via:
    python3 MaxKernel/auto_agent/test_isolate_object.py
or:
    python3 -m unittest MaxKernel/auto_agent/test_isolate_object.py
"""

import ast
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from auto_agent.isolate_object import (
  ImportCollector,
  ObjectExtractor,
  isolate_object,
)


class TestIsolateObjectDemo(unittest.TestCase):
  def setUp(self):
    """Create a temporary workspace with local helper packages and kernel files."""
    self.test_dir = tempfile.mkdtemp(prefix="isolate_object_demo_")
    self.workspace_root = Path(self.test_dir)

    # Create a dummy marker so isolate_object recognizes this as a workspace root
    (self.workspace_root / "pyproject.toml").touch()

    # Create local package directory: tpu_commons (a known local package in common ML workspaces)
    self.pkg_dir = self.workspace_root / "tpu_commons"
    self.pkg_dir.mkdir()
    (self.pkg_dir / "__init__.py").touch()

  def tearDown(self):
    """Clean up the temporary workspace directory."""
    shutil.rmtree(self.test_dir, ignore_errors=True)

  def test_01_external_vs_local_imports(self):
    """
    Demonstrates how isolate_object keeps external imports (jax/numpy) as-is
    while inlining local workspace helper functions.
    """
    helpers_file = self.pkg_dir / "helpers.py"
    helpers_file.write_text(
      "def local_scale(x, factor):\n"
      "    '''Local helper function to scale tensor.'''\n"
      "    return x * factor\n"
    )

    kernel_file = self.pkg_dir / "kernel.py"
    kernel_file.write_text(
      "import jax\n"
      "import jax.numpy as jnp\n"
      "from tpu_commons.helpers import local_scale\n"
      "\n"
      "def my_kernel(x):\n"
      "    '''Target kernel function.'''\n"
      "    scaled = local_scale(x, 2.0)\n"
      "    return jnp.exp(scaled)\n"
    )

    # Isolate 'my_kernel' from kernel_file
    standalone_code = isolate_object(str(kernel_file), "my_kernel")

    print("\n" + "=" * 70)
    print("=== TEST 1: External vs. Local Imports ===")
    print("=" * 70)
    print("ORIGINAL KERNEL FILE (kernel.py):")
    print(kernel_file.read_text())
    print("-" * 70)
    print("STANDALONE INLINED OUTPUT (isolate_object result):")
    print(standalone_code)
    print("=" * 70)

    # 1. External imports must be preserved verbatim at the top
    self.assertIn("import jax", standalone_code)
    self.assertIn("import jax.numpy as jnp", standalone_code)

    # 2. Local import statement 'from tpu_commons.helpers import local_scale' should be replaced
    #    by the actual inlined function definition
    self.assertIn("def local_scale(x, factor):", standalone_code)
    self.assertIn("def my_kernel(x):", standalone_code)

    # 3. Execution order check: local_scale must be defined BEFORE my_kernel
    scale_idx = standalone_code.index("def local_scale")
    kernel_idx = standalone_code.index("def my_kernel")
    self.assertLess(
      scale_idx,
      kernel_idx,
      "Dependency local_scale must appear before my_kernel in standalone output.",
    )

  def test_02_recursive_dependencies(self):
    """
    Demonstrates recursive AST crawling across multiple files:
    target.py -> level1.py -> level2.py
    """
    # level2.py defines a foundational constant
    level2_file = self.pkg_dir / "level2.py"
    level2_file.write_text("BASE_CONSTANT = 42\n")

    # level1.py imports BASE_CONSTANT from level2 and defines a helper
    level1_file = self.pkg_dir / "level1.py"
    level1_file.write_text(
      "from tpu_commons.level2 import BASE_CONSTANT\n"
      "\n"
      "def add_base(x):\n"
      "    return x + BASE_CONSTANT\n"
    )

    # target.py imports add_base from level1
    target_file = self.pkg_dir / "target.py"
    target_file.write_text(
      "from tpu_commons.level1 import add_base\n"
      "\n"
      "def compute(x):\n"
      "    return add_base(x) * 2\n"
    )

    standalone_code = isolate_object(str(target_file), "compute")

    print("\n" + "=" * 70)
    print("=== TEST 2: Recursive Dependencies (target -> level1 -> level2) ===")
    print("=" * 70)
    print("STANDALONE INLINED OUTPUT (isolate_object result):")
    print(standalone_code)
    print("=" * 70)

    # All three levels should be present in the output
    self.assertIn("BASE_CONSTANT = 42", standalone_code)
    self.assertIn("def add_base(x):", standalone_code)
    self.assertIn("def compute(x):", standalone_code)

    # Verify topological order: BASE_CONSTANT -> add_base -> compute
    const_idx = standalone_code.index("BASE_CONSTANT = 42")
    add_idx = standalone_code.index("def add_base(x):")
    compute_idx = standalone_code.index("def compute(x):")
    self.assertLess(const_idx, add_idx)
    self.assertLess(add_idx, compute_idx)

  def test_03_discover_local_import_files(self):
    """
    Demonstrates how ObjectExtractor.get_local_import_files discovers
    local dependency file paths without including external libraries.
    This powers discover_kernel_dependencies_tool.
    """
    helpers_file = self.pkg_dir / "helpers.py"
    helpers_file.write_text("def helper_fn():\n    return 10\n")

    kernel_file = self.pkg_dir / "kernel.py"
    kernel_file.write_text(
      "import os\n"
      "import jax.numpy as jnp\n"
      "from tpu_commons.helpers import helper_fn\n"
      "\n"
      "def my_func():\n"
      "    return helper_fn()\n"
    )

    extractor = ObjectExtractor(str(kernel_file), debug=False)

    # Collect all top-level imports in the kernel file
    with open(kernel_file, "r", encoding="utf-8") as f:
      tree = ast.parse(f.read())
    collector = ImportCollector()
    collector.visit(tree)

    # Discover local dependency file paths
    local_files = extractor.get_local_import_files(list(collector.imports))

    print("\n" + "=" * 70)
    print("=== TEST 3: Discovered Local Dependency Files ===")
    print("=" * 70)
    print(f"KERNEL IMPORTS: {list(collector.imports)}")
    print(f"DISCOVERED LOCAL DEPENDENCY FILES: {local_files}")
    print("=" * 70)

    # Should include my_pkg/helpers.py
    self.assertEqual(len(local_files), 1)
    self.assertEqual(os.path.abspath(local_files[0]), str(helpers_file))

  def test_04_pattern_a_source_dir_vs_session_dir(self):
    """
    Demonstrates Pattern A: discovering dependencies when base_kernel.py
    lives in an isolated session directory (workdir) while dependency
    packages live in the source repository (source_dir).
    """
    helpers_file = self.pkg_dir / "helpers.py"
    helpers_file.write_text("def helper_fn():\n    return 42\n")

    # Create an isolated session subdirectory inside workspace_root
    session_dir = self.workspace_root / "session_123"
    session_dir.mkdir()

    # Place base_kernel.py inside session_dir
    kernel_file = session_dir / "base_kernel.py"
    kernel_file.write_text(
      "import jax\n"
      "from tpu_commons.helpers import helper_fn\n"
      "\n"
      "def my_kernel():\n"
      "    return helper_fn()\n"
    )

    # Initialize ObjectExtractor directly
    extractor = ObjectExtractor(str(kernel_file), debug=False)
    extractor._find_workspace_root = lambda start_dir: str(self.workspace_root)

    with open(kernel_file, "r", encoding="utf-8") as f:
      tree = ast.parse(f.read())
    collector = ImportCollector()
    collector.visit(tree)

    local_files = extractor.get_local_import_files(list(collector.imports))

    print("\n" + "=" * 70)
    print("=== TEST 4: Pattern A (source_dir vs session_dir) ===")
    print("=" * 70)
    print(f"KERNEL IN SESSION DIR: {kernel_file}")
    print(f"SOURCE WORKSPACE ROOT: {self.workspace_root}")
    print(f"DISCOVERED LOCAL DEPENDENCY FILES: {local_files}")
    print("=" * 70)

    self.assertEqual(len(local_files), 1)
    self.assertEqual(os.path.abspath(local_files[0]), str(helpers_file))


if __name__ == "__main__":
  unittest.main(verbosity=2)
