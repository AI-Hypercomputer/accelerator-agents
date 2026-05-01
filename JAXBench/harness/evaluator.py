"""Agent-facing kernel evaluation: compile -> correctness -> benchmark -> report.

This is the primary interface for agents optimizing JAX kernels. Given a workload
name and a path to a kernel file (which must export a workload(*inputs) function),
this module compiles the kernel, checks correctness against the baseline, benchmarks
it, and returns a structured comparison.
"""

import json
import os
import sys
import traceback

import jax
import jax.numpy as jnp

from JAXBench.benchmark import get_workload_dir, has_optimized
from JAXBench.harness.loader import load_module, load_kernel
from JAXBench.harness.profiler import benchmark_fn
from JAXBench.harness.correctness import check_correctness
from JAXBench.harness.runner import get_flop_count
from JAXBench.harness.tpu_specs import get_tpu_spec


def evaluate_kernel(workload_name, kernel_path, tpu='auto', num_warmup=5,
                    num_iters=50):
    """Evaluate an agent-generated kernel against a workload.

    Steps:
        1. Load baseline module, create inputs, run baseline to get reference output
        2. Load the agent's kernel file (must have a workload() function)
        3. Run agent's kernel with the same inputs
        4. Check correctness (atol=1e-2, rtol=1e-2)
        5. If correct, benchmark both with device profiling
        6. If optimized.py exists, benchmark that too for three-way comparison

    Args:
        workload_name: Name of the workload (e.g. '1p_Flash_Attention')
        kernel_path: Path to a Python file with a workload(*inputs) function
        tpu: TPU target ('v5e', 'v6e', or 'auto')
        num_warmup: Number of warmup iterations
        num_iters: Number of benchmark iterations

    Returns:
        dict with evaluation results (see below for schema).
    """
    tpu_name, tpu_spec = get_tpu_spec(tpu)
    peak_tflops = tpu_spec['peak_tflops_bf16']

    result = {
        'workload': workload_name,
        'status': 'error',
        'tpu': tpu_name,
    }

    try:
        # --- Step 1: Load baseline ---
        workload_dir = get_workload_dir(workload_name)
        baseline_path = os.path.join(workload_dir, 'baseline.py')
        baseline_mod = load_module(baseline_path, f'{workload_name}.baseline')

        config = getattr(baseline_mod, 'CONFIG', {})
        result['config'] = {k: v for k, v in config.items()
                            if isinstance(v, (int, float, str, bool))}

        skip_jit = getattr(baseline_mod, '_skip_jit', False)

        create_fn = getattr(baseline_mod, 'create_inputs')
        if 'dtype' in create_fn.__code__.co_varnames:
            inputs = create_fn(dtype=jnp.bfloat16)
        else:
            inputs = create_fn()
        if not isinstance(inputs, (list, tuple)):
            inputs = (inputs,)

        # Get FLOP count from baseline
        baseline_fn = baseline_mod.workload
        xla_flops = get_flop_count(baseline_mod, baseline_fn, inputs, skip_jit)

        # Run baseline to get reference output
        run_baseline = baseline_fn if skip_jit else jax.jit(baseline_fn)
        ref_output = run_baseline(*inputs)
        if hasattr(ref_output, 'block_until_ready'):
            ref_output.block_until_ready()

        # --- Step 2: Load agent kernel ---
        try:
            kernel_mod = load_kernel(kernel_path)
        except (FileNotFoundError, ValueError) as e:
            result['status'] = 'compile_error'
            result['error'] = str(e)
            return result

        kernel_fn = kernel_mod.workload
        kernel_skip_jit = getattr(kernel_mod, '_skip_jit', False)

        # --- Step 3: Run agent kernel ---
        try:
            run_kernel = kernel_fn if kernel_skip_jit else jax.jit(kernel_fn)
            test_output = run_kernel(*inputs)
            if hasattr(test_output, 'block_until_ready'):
                test_output.block_until_ready()
        except Exception as e:
            result['status'] = 'runtime_error'
            result['error'] = str(e)[:300]
            result['traceback'] = traceback.format_exc()[-500:]
            return result

        # --- Step 4: Check correctness ---
        correctness = check_correctness(ref_output, test_output)
        result['correctness'] = correctness

        if not correctness['correct']:
            result['status'] = 'incorrect'
            return result

        # --- Step 5: Benchmark both ---
        jax.clear_caches()

        baseline_bench = benchmark_fn(
            baseline_fn, inputs,
            num_warmup=num_warmup, num_iters=num_iters,
            skip_jit=skip_jit, label=f'{workload_name}_baseline',
        )
        baseline_out = baseline_bench.pop('output')

        jax.clear_caches()

        kernel_bench = benchmark_fn(
            kernel_fn, inputs,
            num_warmup=num_warmup, num_iters=num_iters,
            skip_jit=kernel_skip_jit, label=f'{workload_name}_kernel',
        )
        kernel_out = kernel_bench.pop('output')

        def _perf_dict(bench):
            median = bench['median_ms']
            tflops = 0.0
            util = 0.0
            if xla_flops > 0 and median > 0:
                tflops = xla_flops / (median / 1000) / 1e12
                util = (tflops / peak_tflops) * 100
            return {
                'median_ms': bench['median_ms'],
                'mean_ms': bench['mean_ms'],
                'std_ms': bench['std_ms'],
                'min_ms': bench['min_ms'],
                'timing_method': bench['timing_method'],
                'tflops': round(tflops, 2),
                'utilization_pct': round(util, 1),
            }

        result['baseline'] = _perf_dict(baseline_bench)
        result['kernel'] = _perf_dict(kernel_bench)

        speedup = (baseline_bench['median_ms'] / kernel_bench['median_ms']
                    if kernel_bench['median_ms'] > 0 else 0)
        result['speedup_vs_baseline'] = round(speedup, 2)

        # --- Step 6: Benchmark pallas reference if present ---
        result['pallas_reference'] = None
        result['speedup_vs_pallas'] = None

        if has_optimized(workload_name):
            try:
                jax.clear_caches()
                opt_path = os.path.join(workload_dir, 'optimized.py')
                opt_mod = load_module(opt_path, f'{workload_name}.optimized')
                opt_fn = opt_mod.workload
                opt_skip_jit = getattr(opt_mod, '_skip_jit', False)

                opt_bench = benchmark_fn(
                    opt_fn, inputs,
                    num_warmup=num_warmup, num_iters=num_iters,
                    skip_jit=opt_skip_jit, label=f'{workload_name}_pallas',
                )
                opt_bench.pop('output')

                result['pallas_reference'] = _perf_dict(opt_bench)
                pallas_speedup = (opt_bench['median_ms'] / kernel_bench['median_ms']
                                  if kernel_bench['median_ms'] > 0 else 0)
                result['speedup_vs_pallas'] = round(pallas_speedup, 2)
            except Exception:
                pass  # Pallas reference is optional

        result['status'] = 'correct'

    except Exception as e:
        result['error'] = str(e)[:300]
        result['traceback'] = traceback.format_exc()[-500:]

    return result


def format_eval_result(result):
    """Format an evaluation result as human-readable text."""
    lines = []
    lines.append(f"Workload: {result['workload']}")
    lines.append(f"Status:   {result['status']}")
    lines.append(f"TPU:      {result.get('tpu', '?')}")
    lines.append("")

    if 'correctness' in result:
        c = result['correctness']
        lines.append(f"Correctness: {'PASS' if c['correct'] else 'FAIL'} "
                      f"(max_diff={c['max_diff']:.6f}, {c['reason']})")
        lines.append("")

    if result['status'] == 'correct':
        b = result['baseline']
        k = result['kernel']
        lines.append(f"{'':>20} {'Median(ms)':>12} {'TFLOPS':>10} {'Util%':>8}")
        lines.append(f"{'Baseline (XLA)':>20} {b['median_ms']:>12.3f} "
                      f"{b['tflops']:>10.1f} {b['utilization_pct']:>7.1f}%")
        lines.append(f"{'Your Kernel':>20} {k['median_ms']:>12.3f} "
                      f"{k['tflops']:>10.1f} {k['utilization_pct']:>7.1f}%")

        if result.get('pallas_reference'):
            p = result['pallas_reference']
            lines.append(f"{'Pallas Reference':>20} {p['median_ms']:>12.3f} "
                          f"{p['tflops']:>10.1f} {p['utilization_pct']:>7.1f}%")

        lines.append("")
        lines.append(f"Speedup vs Baseline: {result['speedup_vs_baseline']:.2f}x")
        if result.get('speedup_vs_pallas') is not None:
            lines.append(f"Speedup vs Pallas:   {result['speedup_vs_pallas']:.2f}x")

    elif 'error' in result:
        lines.append(f"Error: {result['error']}")

    return '\n'.join(lines)
