import jax
import jax.numpy as jnp
import numpy as np


# 1. Define the core computation as a pure function.
# This is the function that will be compiled by JAX.
def computation(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
  """Adds two JAX arrays element-wise."""
  return A + B


# 2. Create a JIT-compiled version of the function for performance.
jit_computation = jax.jit(computation)

# 3. Set up the test data.
N = 1024

# Create host-side data using NumPy to establish a ground truth.
h_A = np.arange(N, dtype=np.float32)
h_B = np.arange(N, dtype=np.float32) * 2

# Calculate the expected result using NumPy.
expected_C = h_A + h_B

# Convert NumPy arrays to JAX arrays. JAX will handle moving them to the device.
d_A = jnp.array(h_A)
d_B = jnp.array(h_B)

# 4. Execute the JAX computation.
# We call the JIT-compiled version.
d_C = jit_computation(d_A, d_B)

# 5. Verify the result.
# block_until_ready ensures the computation is finished before we inspect the result.
d_C.block_until_ready()

# Print the first 10 results for visual inspection, similar to the original C++ code.
print("Verifying JAX computation against NumPy...")
print("First 10 results:")
for i in range(10):
  print(f"C[{i}] = {d_C[i]:.1f} (expected {expected_C[i]:.1f})")

# Use a robust numerical comparison to verify the entire array.
# This will raise an error if the arrays do not match.
np.testing.assert_allclose(d_C, expected_C, rtol=1e-6)

print("\nVerification successful: The JAX output matches the NumPy output.")
