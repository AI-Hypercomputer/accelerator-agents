import string

HARNESS_TEMPLATE = string.Template("""
import time
import json
import jax
import jax.numpy as jnp
import importlib
import importlib.util
import os
import traceback


def load_module_from_path(module_name, file_path):
  spec = importlib.util.spec_from_file_location(module_name, file_path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {module_name} from {file_path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def benchmark(func, args, static_argnums, num_runs=20, num_warmups=5):
  # 1. Identify dynamic args for the compiled function call.
  dynamic_args = tuple(
      arg for i, arg in enumerate(args) if i not in static_argnums)

  # 2. Compile the function to an executable to eliminate dispatch overhead.
  compiled_func = jax.jit(func,
                          static_argnums=static_argnums).lower(*args).compile()

  # 3. Warm up
  for _ in range(num_warmups):
    res = compiled_func(*dynamic_args)
  jax.block_until_ready(res)

  # 4. Benchmark
  start = time.perf_counter()
  for _ in range(num_runs):
    res = compiled_func(*dynamic_args)
  jax.block_until_ready(res)
  end = time.perf_counter()

  return (end - start) / num_runs


def main():
  try:
    # Load task configuration from task.json
    if not os.path.exists("task.json"):
      raise FileNotFoundError(
          "task.json not found. It should contain input_gen_code.")

    with open("task.json", "r") as f:
      task_data = json.load(f)

    input_gen_code = task_data.get("input_gen_code")

    if input_gen_code:
      exec_globals = {"jax": jax, "jnp": jnp, "__builtins__": __builtins__}
      ldict = {}
      try:
        exec(input_gen_code, exec_globals, ldict)
      except Exception as e:
        raise RuntimeError(f"Failed to execute input_gen_code: {e}")

      if "get_inputs" not in ldict:
        raise RuntimeError("input_gen_code must define get_inputs()")
      dynamic_args, static_args = ldict["get_inputs"]()

      try:
        inputs = ldict["get_inputs"]()
      except Exception as e:
        raise RuntimeError(f"Error while running get_inputs(): {e}")

      if not isinstance(inputs, tuple) or len(inputs) != 2:
        raise ValueError(
            f"get_inputs() must return exactly 2 elements: (dynamic_args, static_args). Got: {type(inputs)}"
        )

      dynamic_args, static_args = inputs
      if not isinstance(dynamic_args, (list, tuple)) or not isinstance(static_args, (list, tuple)):
        raise TypeError("Both dynamic_args and static_args must be lists or tuples.")

      args = tuple(dynamic_args) + tuple(static_args)
      static_argnums = tuple(range(len(dynamic_args), len(args)))
    else:
      raise ValueError("input_gen_code must be provided.")

    # Import the uploaded scripts as modules
    base_mod = load_module_from_path("reference", "reference.py")
    optimized_mod = load_module_from_path("optimized", "optimized.py")

    # 1. Correctness Check
    out_base = jax.block_until_ready(base_mod.computation(*args))
    out_base_cpu = jax.device_get(out_base)
    del out_base

    try:
      out_optimized = jax.block_until_ready(optimized_mod.computation(*args))
      out_optimized_cpu = jax.device_get(out_optimized)
      del out_optimized
    except Exception as e:
      result = {
          "compiled_successfully": False,
          "error": str(e),
          "traceback": traceback.format_exc()
      }
      with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f)
      return

    is_correct = jnp.allclose(out_base_cpu,
                              out_optimized_cpu,
                              atol={atol},
                              rtol={rtol})
    max_abs_diff = float(jnp.max(jnp.abs(out_base_cpu - out_optimized_cpu)))
    max_rel_diff = float(jnp.max(jnp.abs((out_base_cpu - out_optimized_cpu) / out_base_cpu)))

    if not is_correct:
      result = {
          "compiled_successfully": True,
          "numerically_correct": False,
          "max_abs_diff": max_abs_diff,
          "max_rel_diff": max_rel_diff,
      }
      with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f)
      return

    # 2. Speed Benchmark
    time_base = benchmark(base_mod.computation, args, static_argnums)
    time_optimized = benchmark(optimized_mod.computation, args, static_argnums)

    result = {
        "compiled_successfully": True,
        "numerically_correct": bool(is_correct),
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "reference_time_ms": time_base * 1000,
        "optimized_time_ms": time_optimized * 1000,
    }
    with open("result.json", "w", encoding="utf-8") as f:
      json.dump(result, f)
  except Exception as e:
    with open("result.json", "w", encoding="utf-8") as f:
      json.dump({
          "error": str(e),
          "traceback": traceback.format_exc()
      }, f)


if __name__ == "__main__":
  main()
""")
