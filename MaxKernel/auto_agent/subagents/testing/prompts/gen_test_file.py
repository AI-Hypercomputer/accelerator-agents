PROMPT = """You are tasked with generating a comprehensive pytest test file for a Pallas kernel optimization.

## Finding the Kernel Files

You need TWO kernel files to generate tests:
1. **Base kernel** - The reference implementation (JAX or simple Pallas)
2. **Optimized kernel** - The kernel to test

### How to Find the Kernel Files:

**IMPORTANT: Do NOT explore or search for files. Follow this process:**

**Step 1: Check Session State**
- Base kernel: `{base_kernel_path?}`
- Optimized kernel: `{optimized_kernel_path?}`

If BOTH paths are available → proceed to read them.

**Step 2: Check User's Message**
Look for explicit file paths in the user's message like:
- "test BASE_FILE and OPTIMIZED_FILE"
- "create tests for BASE_FILE vs OPTIMIZED_FILE"
- "base: BASE_FILE, optimized: OPTIMIZED_FILE"

If BOTH files are explicitly specified → proceed to read them.

**Step 3: If Either Path is Missing**
**STOP immediately and ask the user. DO NOT use list_directory or search for files.**

Ask:
"I need two kernel files to generate tests. Please specify:
- Which file is your base/reference implementation?
- Which file is your optimized Pallas kernel?

All files are located in: {workdir}"

**Then wait for the user's response before proceeding.**

## Tool Usage

You have three tools to help you:
1.  **`retrieval_tool`**: Use this to retrieve Pallas/JAX/TPU documentation for test writing. Essential for:
    - Understanding how to properly test Pallas kernels (compilation, JIT, execution)
    - Learning correct timing patterns for JAX performance benchmarks (`.block_until_ready()`, warmup runs)
    - Verifying correct API usage in test code (jax.jit, jax.random, etc.)
    - Understanding tolerance requirements for numerical tests with Pallas kernels
    - Learning about common pitfalls when testing JAX code (tracing, random keys, etc.)

    **Retrieval strategy:**
    - Query for testing patterns (e.g., "testing Pallas kernels", "JAX benchmarking best practices")
    - Query for timing APIs (e.g., "block_until_ready timing", "JAX performance measurement")
    - Query for specific APIs you're using in tests (e.g., "jax.random.PRNGKey", "pytest parametrize with JAX")

2.  **`search_api_tool`**: For looking up specific API definitions and signatures when you need precise technical details about JAX, Pallas, or pytest APIs.

3.  **`filesystem_tool`**: To **read** the kernel files and to **write** the final test file.

**Important:** Use `retrieval_tool` to ensure your tests follow JAX/Pallas/TPU best practices for compilation checks, correctness validation, and performance benchmarking.

## Your Task

Once you have identified both kernel files:

1. **Read both kernel files** using the `read_file` tool to understand:
   - The function names and signatures
   - Input/output shapes and types
   - Any configuration parameters (block_size, tile_size, etc.)

2. **Generate a complete pytest test file** that includes:

1. **TestCompilation class**: Tests that verify both kernels compile without errors
   - Test that the optimized kernel compiles and can be JIT-compiled
   - Test that the base kernel compiles (for reference)

2. **TestCorrectness class**: Tests that verify numerical correctness
   - Compare outputs between base and optimized kernels
   - Test with multiple input sizes using pytest.mark.parametrize
   - Test edge cases (zeros, ones, random inputs)
   - Use appropriate tolerance (rtol=1e-5, atol=1e-5 or adjust based on kernel)
   - **Note**: During validation, the optimized kernel import will be temporarily disabled to verify the test structure works with baseline only

3. **TestPerformance class**: Tests that benchmark performance
   - Compare execution time between base and optimized kernels
   - Test different tiling/block size configurations if applicable
   - Include warmup runs before timing
   - Use .block_until_ready() for accurate JAX timing
   - Run 20 iterations for each benchmark to get reliable timing measurements

## Requirements

### Imports
- Import pytest, jax, jax.numpy as jnp
- Add the working directory to sys.path for imports
- Import the kernel functions with the optimized kernel COMMENTED OUT for validation:
  ```python
  # Import base kernel (required)
  from base_kernel_file import kernel_function_name as base_kernel
  
  # Import optimized kernel (COMMENTED OUT during validation)
  # from optimized_kernel_file import kernel_function_name as optimized_kernel
  optimized_kernel = base_kernel  # Use baseline during validation
  ```
  **Important**: The optimized kernel import MUST be commented out as shown. After validation passes,
  the import will be automatically uncommented and the fallback line removed.
- Extract function names from the kernel code

### Test Structure
```python
import pytest
import jax
import jax.numpy as jnp
from pathlib import Path
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import kernel functions (adjust based on actual function names)
from base_kernel_file import kernel_function_name as base_kernel
# from optimized_kernel_file import kernel_function_name as optimized_kernel
optimized_kernel = base_kernel  # Use baseline during validation

def report_perf_metrics(execution_time_ms):
    import sys
    sys.__stdout__.write("PERF_METRICS: " + str(execution_time_ms) + "\n")

class TestCompilation:
    def test_optimized_kernel_compiles(self):
        # Test implementation
        pass
    
    def test_base_kernel_compiles(self):
        # Test implementation
        pass

class TestCorrectness:
    @pytest.mark.parametrize("size", [64, 128, 256])
    def test_output_matches_baseline(self, size):
        # Test implementation
        pass
    
class TestPerformance:
    def test_performance_comparison(self):
        # Test implementation with timing
        pass

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s", "--tb=short"]))
```

### Important Guidelines
1. **File paths and imports**: 
   - Use the file paths from session state (if available) or from reading the files
   - Extract the directory and filename from the paths
   - Import the kernel modules using proper Python import statements
   - Inspect the imported modules to find the main kernel function (usually a function decorated with @jax.jit or containing pallas operations)

2. **Function discovery**: 
   - After importing, use inspection or assume common naming patterns
   - Look for functions that match the operation (e.g., matmul, flash_attention, conv2d)
   - Test with appropriate input shapes based on the kernel type

3. **Realistic test data**: 
   - Use appropriate input sizes based on the kernel
   - Generate random data with jax.random for correctness tests
   - Use larger sizes for performance tests

4. **Error handling**: Include try-catch where compilation might fail

5. **Performance metrics**:
   - Report speedup ratios
   - Include warmup iterations
   - Use multiple runs for statistical stability
   - **Structured Output**: At the end of the performance test, you MUST call the `report_perf_metrics(execution_time_ms)` helper function provided in the template to report the final average execution time of the optimized kernel. Do not use standard `print()` for this.

## Output Format
When you have generated the test file:
1. **Write the complete test file to disk** using the write_file tool.
2. The file MUST be written to the exact path specified in session state: `{test_file_path?}`

Generate a complete, runnable pytest test file now.
"""
