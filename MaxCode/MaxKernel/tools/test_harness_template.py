"""Fixed, framework-owned test harness for Pallas kernel correctness + benchmarking.

This file is a TEMPLATE. It is never hand-copied by an LLM and never run
standalone -- `tools/assemble_test_harness.py` reads it and concatenates its
contents programmatically (plain file I/O, no LLM in the loop) onto the
`get_inputs()` snippet an LLM *does* write, plus the base kernel's source
embedded as an isolated, exec()'d namespace (see that script for why: two
kernels that both define a module-level `kernel` helper would silently
collide if pasted into one shared namespace).

This file intentionally:
  - ends right after `def main(): ...` -- no trailing
    `if __name__ == "__main__": main()`. That line is appended by
    `tools/assemble_test_run.py` once `opt_computation` has been bound, since
    `main()` cannot run before `opt_computation` exists.
  - never defines `opt_computation`. That is the one piece of this workflow
    that changes every iteration; `tools/assemble_test_run.py` binds it in
    (again via exec()'d namespace isolation, not text substitution) once per
    iteration and once per autotune trial, while everything below is
    generated ONCE (`tools/assemble_test_harness.py`) and reused unchanged
    for the rest of the run.
"""

# `get_inputs`, `base_computation`, `opt_computation` are never defined in
# this file -- see the docstring above. `assemble_test_harness.py` and
# `assemble_test_run.py` bind them into the *assembled* file's namespace via
# exec(), a mechanism static analysis can't see across the file boundary.
# This is expected, not a bug; suppressed here rather than left as IDE noise.
# pyright: reportUndefinedVariable=false

import math
import sys
import time
import traceback

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Tolerances -- the only part of this file a generator/fixer agent may edit.
# Raise these (e.g. 1e-1) for bf16/low-precision kernels.
# ---------------------------------------------------------------------------
ATOL = 1e-2
RTOL = 1e-2

# Dynamic globals injected at assembly time
get_inputs = None  # pylint: disable=invalid-name
base_computation = None  # pylint: disable=invalid-name
opt_computation = None  # pylint: disable=invalid-name


def benchmark(func, args, static_argnums, num_runs=20, num_warmups=5):
  """JIT-compiles `func` (with layout alignment where possible) and times it."""
  dynamic_args = tuple(
      arg for i, arg in enumerate(args) if i not in static_argnums
  )

  def benchmark_func(*f_args):
    all_args = list(args)
    dyn_idx = 0
    for i in range(len(args)):
      if i not in static_argnums:
        all_args[i] = f_args[dyn_idx]
        dyn_idx += 1
    res = func(*all_args)
    if isinstance(res, tuple):
      return tuple(jax.block_until_ready(r) for r in res)
    return jax.block_until_ready(res)

  try:
    from jax.experimental import layout as jax_layout  # pylint: disable=g-import-not-at-top

    in_shardings = jax_layout.Format(jax_layout.Layout.AUTO)
    out_shardings = jax_layout.Format(jax_layout.Layout.AUTO)
    compiled_func = (
        jax.jit(
            benchmark_func,
            static_argnums=static_argnums,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
        )
        .lower(*args)
        .compile()
    )

    if hasattr(compiled_func, "input_formats"):
      arg_formats, _ = compiled_func.input_formats

      @jax.jit
      def enforce_layout(*xs):
        return xs

      enforce_layout_compiled = (
          jax.jit(enforce_layout, out_shardings=arg_formats)
          .lower(*dynamic_args)
          .compile()
      )
      dynamic_args = enforce_layout_compiled(*dynamic_args)
  except Exception:  # pylint: disable=broad-except
    compiled_func = (
        jax.jit(benchmark_func, static_argnums=static_argnums)
        .lower(*args)
        .compile()
    )

  res = None
  for _ in range(num_warmups):
    res = compiled_func(*dynamic_args)
  if res is not None:
    jax.block_until_ready(res)

  start = time.perf_counter()
  res = None
  for _ in range(num_runs):
    res = compiled_func(*dynamic_args)
  if res is not None:
    jax.block_until_ready(res)
  end = time.perf_counter()
  return (end - start) / num_runs


def main():
  try:
    raw_inputs = get_inputs()  # noqa: F821 -- prepended above this template

    if isinstance(raw_inputs, list):
      inputs_list = raw_inputs
    elif isinstance(raw_inputs, tuple) and len(raw_inputs) == 2:
      inputs_list = [raw_inputs]
    else:
      raise ValueError(
          "get_inputs() must return a list of tuples or a single "
          "(dynamic_args, static_args) tuple."
      )

    all_correct = True
    times_opt = []
    speedups = []

    for idx, inputs in enumerate(inputs_list):
      if not isinstance(inputs, tuple) or len(inputs) != 2:
        raise ValueError(
            "Each input config must return exactly 2 elements: "
            f"(dynamic_args, static_args). Got: {type(inputs)}"
        )

      dynamic_args, static_args = inputs
      args = tuple(dynamic_args) + tuple(static_args)
      static_argnums = tuple(range(len(dynamic_args), len(args)))

      jit_base = jax.jit(base_computation, static_argnums=static_argnums)
      out_base = jax.device_get(jax.block_until_ready(jit_base(*args)))

      jit_opt = jax.jit(opt_computation, static_argnums=static_argnums)
      out_opt = jax.device_get(jax.block_until_ready(jit_opt(*args)))

      base_leaves = jax.tree_util.tree_leaves(out_base)
      opt_leaves = jax.tree_util.tree_leaves(out_opt)

      is_correct = True
      if len(base_leaves) != len(opt_leaves):
        is_correct = False
        print(
            f"Output count mismatch for input config {idx}: expected "
            f"{len(base_leaves)}, got {len(opt_leaves)}"
        )
      else:
        for i, (b, o) in enumerate(zip(base_leaves, opt_leaves)):
          if b.shape != o.shape:
            is_correct = False
            print(
                f"Mismatch in output tensor {i} for input config {idx}: "
                f"expected shape {b.shape}, got shape {o.shape}"
            )
            continue
          match = bool(jnp.allclose(b, o, atol=ATOL, rtol=RTOL))
          if not match:
            is_correct = False
            max_diff = jnp.max(jnp.abs(b - o))
            print(
                f"Mismatch in output tensor {i} for input config {idx}: "
                f"max absolute difference {max_diff}"
            )

      if not is_correct:
        all_correct = False
        continue

      time_base = benchmark(base_computation, args, static_argnums)
      time_opt = benchmark(opt_computation, args, static_argnums)
      times_opt.append(time_opt)
      speedup = (time_base / time_opt) if time_opt > 0 else 0
      speedups.append(speedup)
      print(f"SPEEDUP_CASE_{idx}: {speedup:.4f}")

    print(f"CORRECTNESS: {all_correct}")

    if not all_correct:
      sys.exit(1)

    valid_opt = [t for t in times_opt if t > 0]
    valid_speedups = [s for s in speedups if s > 0]
    if valid_opt:
      geo_mean_time_opt = math.exp(
          sum(math.log(t) for t in valid_opt) / len(valid_opt)
      )
      # RESULT_TIME/PERF_METRICS report absolute optimized-kernel latency --
      # not speedup. SPEEDUP is reported
      # separately below for human-readable summaries.
      print(f"RESULT_TIME: {geo_mean_time_opt * 1000:.6f} ms")
      print(f"PERF_METRICS: {geo_mean_time_opt * 1000:.6f}")
    if valid_speedups:
      geo_mean_speedup = math.exp(
          sum(math.log(s) for s in valid_speedups) / len(valid_speedups)
      )
      print(f"SPEEDUP: {geo_mean_speedup:.4f}")

  except Exception as e:  # pylint: disable=broad-except
    print(f"ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)


# NOTE: no `if __name__ == "__main__": main()` here on purpose -- see the
# module docstring. `tools/assemble_test_run.py` appends it after binding
# `opt_computation`.
