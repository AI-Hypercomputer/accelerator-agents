import argparse
import time

import jax
import jax.numpy as jnp
import jax.random as random


def simple_compute(x):
  return jax.nn.softmax(x, axis=1)


def check_compilation():
  # Initialization
  batch_size = 16
  dim = 16384
  key = random.PRNGKey(0)
  x = random.normal(key, (batch_size, dim))

  # Computation
  result = optimized_compute(x).block_until_ready()


# EVOLVE-BLOCK-START
def optimized_compute(x: jnp.ndarray) -> jnp.ndarray:
  return jax.nn.softmax(x, axis=1)


# EVOLVE-BLOCK-END


def check_correctness():
  # Check if the output is correct
  batch_size = 16
  dim = 16384
  key = random.PRNGKey(0)
  x = random.normal(key, (batch_size, dim))

  o1 = simple_compute(x).block_until_ready()

  o2 = optimized_compute(x).block_until_ready()

  # Check if the outputs are close
  are_equal = jnp.allclose(o1, o2, rtol=1e-02, atol=1e-02)
  if are_equal:
    return True
  else:
    return False


def check_performance(num_iter: int = 5):
  # Measure performance of the optimized compute function
  batch_size = 16
  dim = 16384
  key = random.PRNGKey(0)
  x = random.normal(key, (batch_size, dim))

  # Measure time taken for the optimized compute function

  simple_compute_times = []
  optimized_compute_times = []

  # Warmup:
  simple_compute(x).block_until_ready()
  optimized_compute(x).block_until_ready()

  for i in range(num_iter):
    print(f"Iteration {i + 1}/{num_iter}")
    s = time.time()
    simple_compute(x).block_until_ready()
    e = time.time()
    simple_compute_times.append(e - s)
    s = time.time()
    optimized_compute(x).block_until_ready()
    e = time.time()
    optimized_compute_times.append(e - s)

  simple_avg = jnp.mean(jnp.array(simple_compute_times))
  optimized_avg = jnp.mean(jnp.array(optimized_compute_times))
  return simple_avg, optimized_avg


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run TPU kernel tests")
  parser.add_argument("--compilation", action="store_true", help="Run compilation test")
  parser.add_argument("--correctness", action="store_true", help="Run correctness test")
  parser.add_argument("--performance", action="store_true", help="Run performance test")

  args = parser.parse_args()

  if args.compilation:
    print("Running compilation test...")
    check_compilation()
    print("Compilation test: Success")

  if args.correctness:
    print("Running correctness test...")
    if check_correctness():
      print("Correctness test: Success")
      exit(0)
    else:
      print("Correctness test: Failed")
      exit(1)

  if args.performance:
    print("Running performance test...")
    simple_avg, optimized_avg = check_performance()
    print(f"Simple compute average time: {simple_avg:.6f} s")
    print(f"Optimized compute average time: {optimized_avg:.6f} s")
