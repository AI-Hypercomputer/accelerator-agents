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
import xprof_utils


def load_module_from_path(module_name, file_path):
  spec = importlib.util.spec_from_file_location(module_name, file_path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {module_name} from {file_path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def benchmark(func, args, static_argnums, trace_dir=None, num_runs=20, num_warmups=5):
  # 1. Identify dynamic args for the compiled function call.
  dynamic_args = tuple(
      arg for i, arg in enumerate(args) if i not in static_argnums)

  def benchmark_func(*f_args):
    with jax.named_scope('benchmark_func'):
      res = func(*f_args)
      return jax.block_until_ready(res)

  # 2. Compile the function to an executable to eliminate dispatch overhead.
  # Attempt to use proper sharding if available
  try:
    from jax.experimental import layout as jax_layout
    in_shardings = jax_layout.Format(jax_layout.Layout.AUTO)
    out_shardings = jax_layout.Format(jax_layout.Layout.AUTO)
    compiled_func = jax.jit(
        benchmark_func,
        static_argnums=static_argnums,
        in_shardings=in_shardings,
        out_shardings=out_shardings
    ).lower(*args).compile()

    # Layout Alignment Trick
    if hasattr(compiled_func, 'input_formats'):
      arg_formats, _ = compiled_func.input_formats
      
      @jax.jit
      def enforce_layout(*xs):
        return xs
        
      # Compile helper with desired output formats
      enforce_layout_compiled = jax.jit(
        enforce_layout,
        out_shardings=arg_formats
      ).lower(*dynamic_args).compile()
      
      # Apply alignment
      dynamic_args = enforce_layout_compiled(*dynamic_args)
  except Exception as e:
    compiled_func = jax.jit(benchmark_func, static_argnums=static_argnums).lower(*args).compile()

  # 3. Warm up
  for _ in range(num_warmups):
    res = compiled_func(*dynamic_args)
  jax.block_until_ready(res)

  # 4. Benchmark
  def run_wall_time():
    start = time.perf_counter()
    for _ in range(num_runs):
      res = compiled_func(*dynamic_args)
    jax.block_until_ready(res)
    end = time.perf_counter()
    return (end - start) / num_runs
  
  def run_xprof():
    with jax.profiler.trace(trace_dir):
      for _ in range(num_runs):
        res = compiled_func(*dynamic_args)
        jax.block_until_ready(res)
        # Inject dummy op to separate trace events
        jnp.sum(jax.random.normal(jax.random.key(0), (128, 128), jnp.float32)).block_until_ready()

  avg_wall_time = run_wall_time()

  xprof_time = 0.0
  if trace_dir:
    run_xprof()
    try:
      xprof_time = xprof_utils.extract_xprof_time(trace_dir, 'benchmark_func')
    except Exception as e:
      raise RuntimeError(f"Failed to extract xprof time: {e}")

  return avg_wall_time, xprof_time


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
      ldict = {}
      try:
        exec(input_gen_code, globals(), ldict)
      except Exception as e:
        raise RuntimeError(f"Failed to execute input_gen_code: {e}")

      if "get_inputs" not in ldict:
        raise RuntimeError("input_gen_code must define get_inputs()")

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
    jit_base = jax.jit(base_mod.computation, static_argnums=static_argnums)
    out_base = jax.block_until_ready(jit_base(*args))
    out_base_cpu = jax.device_get(out_base)
    del out_base

    try:
      jit_optimized = jax.jit(optimized_mod.computation, static_argnums=static_argnums)
      out_optimized = jax.block_until_ready(jit_optimized(*args))
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

    time_base, xprof_time_base = benchmark(base_mod.computation, args, static_argnums, trace_dir="trace_base")
    time_optimized, xprof_time_optimized = benchmark(optimized_mod.computation, args, static_argnums, trace_dir="trace_opt")

    result = {
        "compiled_successfully": True,
        "numerically_correct": bool(is_correct),
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "reference_time_ms": time_base * 1000,
        "optimized_time_ms": time_optimized * 1000,
        "xprof_reference_time_ms": xprof_time_base,
        "xprof_optimized_time_ms": xprof_time_optimized,
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
