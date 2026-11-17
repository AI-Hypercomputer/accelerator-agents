import string

HARNESS_TEMPLATE = string.Template("""
import time
import json
import jax
import jax.numpy as jnp
import numpy as np
import importlib
import importlib.util
import os
import shutil
import sys
import traceback
import xprof_utils


def load_module_from_path(module_name, file_path):
  spec = importlib.util.spec_from_file_location(module_name, file_path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {module_name} from {file_path}")
  module = importlib.util.module_from_spec(spec)
  # Register before exec_module: modules using `from __future__ import
  # annotations` turn every annotation into a string, and dataclasses then
  # resolves them via sys.modules[cls.__module__], which raises
  # AttributeError on None if the module was never registered.
  sys.modules[module_name] = module
  try:
    spec.loader.exec_module(module)
  except BaseException:
    sys.modules.pop(module_name, None)
    raise
  return module


def normalize_donate_map(donate_map):
  \"\"\"Returns `donate_map` as {int: int}; JSON round-trips turn the keys into strings.\"\"\"
  return {int(k): int(v) for k, v in dict(donate_map or {}).items()}


def check_donate_map(donate_map, args, static_argnums, out_leaves, kernel_name):
  \"\"\"Validates a task's `donate_args` against a kernel's actual output structure.

  Every entry maps a dynamic argument to the flattened output leaf that holds
  the same buffer after the call. Donation only pays off (and recycling the
  output as the next input only works) when the two have identical shape and
  dtype, so anything else is rejected before a buffer is donated.
  \"\"\"
  dynamic_positions = [i for i in range(len(args)) if i not in static_argnums]
  for in_idx, out_idx in donate_map.items():
    if not 0 <= in_idx < len(dynamic_positions):
      raise ValueError(
          f"donate_args: dynamic arg index {in_idx} is out of range "
          f"(the task has {len(dynamic_positions)} dynamic args).")
    if not 0 <= out_idx < len(out_leaves):
      raise ValueError(
          f"donate_args: output leaf index {out_idx} is out of range "
          f"({kernel_name} returns {len(out_leaves)} leaves).")
    arg = args[dynamic_positions[in_idx]]
    out = out_leaves[out_idx]
    arg_shape = tuple(getattr(arg, "shape", ()))
    arg_dtype = jnp.dtype(getattr(arg, "dtype", type(arg)))
    if arg_shape != tuple(out.shape) or arg_dtype != jnp.dtype(out.dtype):
      raise ValueError(
          f"donate_args: dynamic arg {in_idx} ({arg_dtype}{list(arg_shape)}) "
          f"cannot alias output leaf {out_idx} of {kernel_name} "
          f"({jnp.dtype(out.dtype)}{list(out.shape)}); shape and dtype must match.")


def benchmark(func, args, static_argnums, trace_dir=None, num_runs=20,
              num_warmups=5, donate_map=None):
  \"\"\"Times `func(*args)` over `num_runs` calls of one compiled executable.

  `donate_map` ({dynamic_arg_index: output_leaf_index}, from the task's
  `donate_args`) names the buffers the kernel updates in place, e.g. a KV
  cache or a recurrent state. Each listed input is donated to the compiled
  call, which lets XLA alias it with that output instead of materializing a
  full copy, and after every call the input slot is rebound to the returned
  leaf so the next call sees a live buffer. Donation has to happen on this
  outer `jax.jit`: a `donate_argnums` on a jitted `computation` is dropped
  when it is traced inside another jit. The caller's `args` are never
  consumed, the donated buffers are private copies.
  \"\"\"
  donate_map = normalize_donate_map(donate_map)
  dynamic_positions = [i for i in range(len(args)) if i not in static_argnums]

  # 1. Identify dynamic args for the compiled function call.
  dynamic_args_list = [args[i] for i in dynamic_positions]

  if donate_map:
    def with_static_args(*dyn_args):
      full_args = list(args)
      for pos, arg in zip(dynamic_positions, dyn_args):
        full_args[pos] = arg
      return func(*full_args)

    # Check the contract against this kernel's outputs before donating
    # anything; a mismatch would otherwise surface as a confusing argument
    # type error on the second call.
    out_leaves = jax.tree_util.tree_leaves(
        jax.eval_shape(with_static_args, *dynamic_args_list))
    check_donate_map(donate_map, args, static_argnums, out_leaves, "the kernel")

    # Donate private copies: the caller's `args` are benchmarked again with
    # the other kernel (and must stay intact for the layout fallback below).
    for in_idx in donate_map:
      dynamic_args_list[in_idx] = jnp.copy(dynamic_args_list[in_idx])

  # Positional indices for jax.jit (static args may precede dynamic ones).
  donate_argnums = tuple(dynamic_positions[i] for i in sorted(donate_map))

  def recycle_donated(res):
    # The donated inputs are dead after the call; the in-place-updated
    # buffers now live in the outputs the task mapped them to.
    if donate_map:
      leaves = jax.tree_util.tree_leaves(res)
      for in_idx, out_idx in donate_map.items():
        dynamic_args_list[in_idx] = leaves[out_idx]

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
        donate_argnums=donate_argnums,
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
      ).lower(*dynamic_args_list).compile()
      
      # Apply alignment
      dynamic_args_list = list(enforce_layout_compiled(*dynamic_args_list))
  except Exception as e:
    compiled_func = jax.jit(
        benchmark_func,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums
    ).lower(*args).compile()

  # 3. Warm up
  for _ in range(num_warmups):
    res = compiled_func(*dynamic_args_list)
    recycle_donated(res)
  jax.block_until_ready(res)

  # 4. Benchmark
  def run_wall_time():
    start = time.perf_counter()
    for _ in range(num_runs):
      res = compiled_func(*dynamic_args_list)
      recycle_donated(res)
    jax.block_until_ready(res)
    end = time.perf_counter()
    return (end - start) / num_runs
  
  def run_xprof():
    # Clear stale traces so events from earlier runs aren't counted.
    shutil.rmtree(trace_dir, ignore_errors=True)

    with jax.profiler.trace(trace_dir):
      for _ in range(num_runs):
        res = compiled_func(*dynamic_args_list)
        recycle_donated(res)
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


def diff_metrics(b, o, chunk_elems=1 << 24):
  \"\"\"Max absolute and max relative difference between two outputs.

  Computed on the host in bounded-size chunks. `b` and `o` have already been
  pulled off the device by `jax.device_get`, so the naive
  `jnp.max(jnp.abs((b - o) / b))` ships them straight back: the true division
  promotes to float, and for a large integer output (e.g. a 4 GiB uint8 paged
  KV cache) that is a 16 GiB argument plus a 16 GiB result, which does not fit
  in HBM. These are diagnostic metrics only -- `is_correct` comes from
  `jnp.allclose` -- but raising here used to fail the whole case.

  Chunking also fixes two latent issues with the old expression: the
  subtraction no longer wraps around for unsigned dtypes, and a NaN in one
  chunk no longer suppresses the maximum found in the others.
  \"\"\"
  fb = np.asarray(b).reshape(-1)
  fo = np.asarray(o).reshape(-1)
  max_abs = 0.0
  max_rel = 0.0
  for i in range(0, fb.size, chunk_elems):
    x = fb[i:i + chunk_elems].astype(np.float64)
    y = fo[i:i + chunk_elems].astype(np.float64)
    d = np.abs(x - y)
    # max() over an empty slice is undefined; size is never 0 here but guard
    # anyway so a zero-sized output cannot take down the comparison.
    if d.size == 0:
      continue
    max_abs = max(max_abs, float(np.max(d)))
    with np.errstate(divide="ignore", invalid="ignore"):
      r = d / np.abs(x)
    # Matches the previous definition |(b - o) / b|: division by a zero
    # reference stays +inf, and 0/0 stays NaN rather than being counted.
    r = r[~np.isnan(r)]
    if r.size:
      max_rel = max(max_rel, float(np.max(r)))
  return max_abs, max_rel


def main():
  try:
    # Load task configuration from task.json
    if not os.path.exists("task.json"):
      raise FileNotFoundError(
          "task.json not found. It should contain input_gen_code.")

    with open("task.json", "r") as f:
      task_data = json.load(f)

    input_gen_code = task_data.get("input_gen_code")
    task_atol = task_data.get("atol")
    task_rtol = task_data.get("rtol")
    sort_before_compare = task_data.get("sort_before_compare", False)
    donate_args = normalize_donate_map(task_data.get("donate_args"))

    if input_gen_code:
      ldict = {}
      try:
        exec(input_gen_code, globals(), ldict)
      except Exception as e:
        raise RuntimeError(f"Failed to execute input_gen_code: {e}")

      if "get_inputs" not in ldict:
        raise RuntimeError("input_gen_code must define get_inputs()")

      try:
        raw_inputs = ldict["get_inputs"]()
      except Exception as e:
        raise RuntimeError(f"Error while running get_inputs(): {e}")

      # Check if the input is a list of tuples or a single tuple
      if isinstance(raw_inputs, list):
        inputs_list = raw_inputs
      elif isinstance(raw_inputs, tuple) and len(raw_inputs) == 2:
        inputs_list = [raw_inputs]
      else:
        raise ValueError(
            f"get_inputs() must return a list of tuples or a single (dynamic_args, static_args) tuple. Got: {type(raw_inputs)}"
        )
    else:
      raise ValueError("input_gen_code must be provided.")

    # Import the uploaded scripts as modules
    base_mod = load_module_from_path("reference", "reference.py")
    optimized_mod = load_module_from_path("optimized", "optimized.py")

    harness_logs = []

    result = {
        "compiled_successfully": [],
        "numerically_correct": [],
        "max_abs_diff": [],
        "max_rel_diff": [],
        "reference_time_ms": [],
        "optimized_time_ms": [],
        "xprof_reference_time_ms": [],
        "xprof_optimized_time_ms": [],
        "error_trace": [],
    }

    # Iterate over all input configurations
    for idx, inputs in enumerate(inputs_list):
      if not isinstance(inputs, tuple) or len(inputs) != 2:
        raise ValueError(
            f"Each input config must return exactly 2 elements: (dynamic_args, static_args). Got: {type(inputs)}"
        )

      dynamic_args, static_args = inputs
      if not isinstance(dynamic_args, (list, tuple)) or not isinstance(static_args, (list, tuple)):
        raise TypeError("Both dynamic_args and static_args must be lists or tuples.")

      args = tuple(dynamic_args) + tuple(static_args)
      static_argnums = tuple(range(len(dynamic_args), len(args)))

      if isinstance(task_atol, list):
        if idx < len(task_atol):
          curr_atol = task_atol[idx]
        else:
          curr_atol = task_atol[-1]
          harness_logs.append(
              f"atol list length ({len(task_atol)}) is shorter than input "
              f"count ({len(inputs_list)}). Reusing last atol ({curr_atol}) "
              f"for input {idx}."
          )
      else:
        curr_atol = float(task_atol) if task_atol is not None else 1e-3

      if isinstance(task_rtol, list):
        if idx < len(task_rtol):
          curr_rtol = task_rtol[idx]
        else:
          curr_rtol = task_rtol[-1]
          harness_logs.append(
              f"rtol list length ({len(task_rtol)}) is shorter than input "
              f"count ({len(inputs_list)}). Reusing last rtol ({curr_rtol}) "
              f"for input {idx}."
          )
      else:
        curr_rtol = float(task_rtol) if task_rtol is not None else 1e-3

      # 1. Correctness Check
      try:
        jit_base = jax.jit(base_mod.computation, static_argnums=static_argnums)
        out_base = jax.block_until_ready(jit_base(*args))
        out_base_cpu = jax.device_get(out_base)
        del out_base
      except Exception as e:
        result["compiled_successfully"].append(False)
        result["error_trace"].append(f"Reference model failed: {traceback.format_exc()}")
        result["numerically_correct"].append(False)
        result["max_abs_diff"].append(None)
        result["max_rel_diff"].append(None)
        result["reference_time_ms"].append(0.0)
        result["optimized_time_ms"].append(0.0)
        result["xprof_reference_time_ms"].append(0.0)
        result["xprof_optimized_time_ms"].append(0.0)
        continue

      # `donate_args` is part of the task contract, so a map that does not fit
      # the reference's own outputs is a dataset bug: fail the task rather
      # than quietly benchmarking without donation.
      if donate_args:
        check_donate_map(
            donate_args, args, static_argnums,
            jax.tree_util.tree_leaves(out_base_cpu), "the reference")

      # Dirty all HBM memory leaves with NaN / Sentinel values to prevent cache reuse
      try:
        for leaf in jax.tree_util.tree_leaves(out_base_cpu):
          if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            if jnp.issubdtype(leaf.dtype, jnp.floating) or jnp.issubdtype(leaf.dtype, jnp.complexfloating):
              val = jnp.nan
            elif jnp.issubdtype(leaf.dtype, jnp.bool_):
              val = True
            else:
              val = 123  # Fits within int8/uint8 and all larger integer dtypes

            dummy = jnp.full(leaf.shape, val, dtype=leaf.dtype)
            dummy.block_until_ready()
            del dummy
      except Exception as e:
        harness_logs.append(f"Failed to dirty HBM memory: {e}")

      try:
        jit_optimized = jax.jit(optimized_mod.computation, static_argnums=static_argnums)
        out_optimized = jax.block_until_ready(jit_optimized(*args))
        out_optimized_cpu = jax.device_get(out_optimized)
        del out_optimized
      except Exception as e:
        result["compiled_successfully"].append(False)
        result["error_trace"].append(traceback.format_exc())
        result["numerically_correct"].append(False)
        result["max_abs_diff"].append(None)
        result["max_rel_diff"].append(None)
        result["reference_time_ms"].append(0.0)
        result["optimized_time_ms"].append(0.0)
        result["xprof_reference_time_ms"].append(0.0)
        result["xprof_optimized_time_ms"].append(0.0)
        continue

      result["compiled_successfully"].append(True)
      result["error_trace"].append(None)

      out_base_flat = jax.tree_util.tree_leaves(out_base_cpu)
      out_optimized_flat = jax.tree_util.tree_leaves(out_optimized_cpu)

      is_correct = True
      max_abs_diff = 0.0
      max_rel_diff = 0.0

      try:
        if len(out_base_flat) != len(out_optimized_flat):
           raise ValueError(f"Output count mismatch: {len(out_base_flat)} vs {len(out_optimized_flat)}")
        for b, o in zip(out_base_flat, out_optimized_flat):
          if b.shape != o.shape:
             raise ValueError(f"Shape mismatch: {b.shape} vs {o.shape}")
          if sort_before_compare and np.ndim(b) > 0:
            # Order along the last axis is not part of this task's output
            # contract (e.g. top-k index sets): compare rows as sorted
            # multisets. Done on the host, so it never affects timing.
            b = np.sort(b, axis=-1)
            o = np.sort(o, axis=-1)
          is_correct = is_correct and bool(jnp.allclose(b, o, atol=curr_atol, rtol=curr_rtol))
          leaf_abs, leaf_rel = diff_metrics(b, o)
          max_abs_diff = max(max_abs_diff, leaf_abs)
          max_rel_diff = max(max_rel_diff, leaf_rel)
      except Exception as e:
        harness_logs.append(f"Correctness check failed for input {idx}: {e}")
        is_correct = False

      result["numerically_correct"].append(bool(is_correct))
      result["max_abs_diff"].append(max_abs_diff)
      result["max_rel_diff"].append(max_rel_diff)

      if not is_correct:
        result["reference_time_ms"].append(0.0)
        result["optimized_time_ms"].append(0.0)
        result["xprof_reference_time_ms"].append(0.0)
        result["xprof_optimized_time_ms"].append(0.0)
        continue

      # Benchmark and collect timing results
      try:
        time_base, xprof_time_base = benchmark(base_mod.computation, args, static_argnums, trace_dir=f"trace_base_{idx}", donate_map=donate_args)
        time_optimized, xprof_time_optimized = benchmark(optimized_mod.computation, args, static_argnums, trace_dir=f"trace_opt_{idx}", donate_map=donate_args)
        result["reference_time_ms"].append(time_base * 1000)
        result["optimized_time_ms"].append(time_optimized * 1000)
        result["xprof_reference_time_ms"].append(xprof_time_base)
        result["xprof_optimized_time_ms"].append(xprof_time_optimized)
      except Exception as e:
        harness_logs.append(f"Benchmarking failed for input {idx}: {e}")
        result["reference_time_ms"].append(0.0)
        result["optimized_time_ms"].append(0.0)
        result["xprof_reference_time_ms"].append(0.0)
        result["xprof_optimized_time_ms"].append(0.0)

    if harness_logs:
      result["logs"] = harness_logs
    with open("result.json", "w", encoding="utf-8") as f:
      json.dump(result, f)
  except Exception as e:
    with open("result.json", "w", encoding="utf-8") as f:
      json.dump({
          "error_trace": [traceback.format_exc()]
      }, f)


if __name__ == "__main__":
  main()
""")
