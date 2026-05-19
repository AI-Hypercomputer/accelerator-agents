"""Core benchmark runner: run individual or all workloads."""

import csv
import json
import os
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from JAXBench.benchmark import list_workloads, get_workload_dir, has_optimized, BENCHMARK_DIR
from JAXBench.harness.loader import load_module
from JAXBench.harness.profiler import benchmark_fn
from JAXBench.harness.tpu_specs import get_tpu_spec


def get_flop_count(mod, workload_fn, inputs, skip_jit):
    """Get FLOP count: prefer module's get_flops(), fall back to XLA cost_analysis."""
    get_flops_fn = getattr(mod, 'get_flops', None)
    if get_flops_fn is not None:
        return int(get_flops_fn())

    if not skip_jit:
        try:
            compiled = jax.jit(workload_fn).lower(*inputs).compile()
            cost = compiled.cost_analysis()
            if isinstance(cost, list):
                cost = cost[0] if cost else {}
            return int(cost.get('flops', 0))
        except Exception:
            pass

    return 0


def run_workload(name, variant='baseline', tpu='auto', num_warmup=5,
                 num_iters=50):
    """Run a single workload variant with device profiling.

    Args:
        name: Workload name (e.g. '1p_Flash_Attention')
        variant: 'baseline' or 'optimized'
        tpu: TPU target ('v5e', 'v6e', or 'auto')
        num_warmup: Number of warmup iterations
        num_iters: Number of benchmark iterations

    Returns:
        dict with structured results including timing, FLOPS, utilization.
    """
    tpu_name, tpu_spec = get_tpu_spec(tpu)
    peak_tflops = tpu_spec['peak_tflops_bf16']

    workload_dir = get_workload_dir(name)
    module_path = os.path.join(workload_dir, f'{variant}.py')

    if not os.path.exists(module_path):
        return None

    result = {
        'name': name,
        'variant': variant,
        'status': 'error',
        'tpu': tpu_name,
    }

    try:
        mod = load_module(module_path, f'{name}.{variant}')

        config = getattr(mod, 'CONFIG', {})
        result['config'] = {k: v for k, v in config.items()
                            if isinstance(v, (int, float, str, bool))}

        skip_jit = getattr(mod, '_skip_jit', False)

        create_fn = getattr(mod, 'create_inputs')
        if 'dtype' in create_fn.__code__.co_varnames:
            inputs = create_fn(dtype=jnp.bfloat16)
        else:
            inputs = create_fn()
        if not isinstance(inputs, (list, tuple)):
            inputs = (inputs,)

        result['input_shapes'] = [
            list(x.shape) if hasattr(x, 'shape') else str(type(x))
            for x in inputs
        ]

        workload_fn = getattr(mod, 'workload')
        xla_flops = get_flop_count(mod, workload_fn, inputs, skip_jit)
        result['xla_flops'] = xla_flops

        # Benchmark
        bench = benchmark_fn(
            workload_fn, inputs,
            num_warmup=num_warmup,
            num_iters=num_iters,
            skip_jit=skip_jit,
            label=f'{name}_{variant}',
        )

        out = bench.pop('output')

        # Compute TFLOPS and utilization
        tflops = 0.0
        utilization_pct = 0.0
        median_ms = bench['median_ms']
        if xla_flops > 0 and median_ms > 0:
            tflops = xla_flops / (median_ms / 1000) / 1e12
            utilization_pct = (tflops / peak_tflops) * 100

        result.update({
            'status': 'success',
            'timing_method': bench['timing_method'],
            'median_ms': bench['median_ms'],
            'mean_ms': bench['mean_ms'],
            'std_ms': bench['std_ms'],
            'min_ms': bench['min_ms'],
            'wall_clock_median_ms': bench['wall_clock_median_ms'],
            'xla_flops': xla_flops,
            'tflops': round(tflops, 2),
            'utilization_pct': round(utilization_pct, 1),
            'output_shape': list(out.shape) if hasattr(out, 'shape') else [],
            'num_iters': num_iters,
        })

    except Exception as e:
        result['error'] = str(e)[:300]
        result['traceback'] = traceback.format_exc()[-500:]

    return result


def run_all(tpu='auto', num_warmup=5, num_iters=50, output_dir=None):
    """Run all workloads (baseline + optimized if present).

    Args:
        tpu: TPU target ('v5e', 'v6e', or 'auto')
        num_warmup: Number of warmup iterations
        num_iters: Number of benchmark iterations
        output_dir: Directory for results.json and results.csv (default: benchmark/)

    Returns:
        dict with metadata and results list.
    """
    if output_dir is None:
        output_dir = BENCHMARK_DIR

    tpu_name, tpu_spec = get_tpu_spec(tpu)

    print(f"JAX {jax.__version__} | Devices: {jax.devices()}")
    print(f"TPU: {tpu_name} | Peak: {tpu_spec['peak_tflops_bf16']} TFLOPS (bf16)")
    print(f"Timing: jax.profiler device-side (jit_*() events)")
    print()

    workload_names = list_workloads()
    print(f"Found {len(workload_names)} workloads")
    print()

    results = []
    header = (f"{'#':>3} {'Workload':<42} {'Var':<10} {'Median(ms)':>10} "
              f"{'TFLOPS':>8} {'Util%':>7} {'Method':<18}")
    print(header)
    print("-" * len(header))

    for i, name in enumerate(workload_names, 1):
        for variant in ['baseline', 'optimized']:
            vpath = os.path.join(get_workload_dir(name), f'{variant}.py')
            if not os.path.exists(vpath):
                continue

            jax.clear_caches()

            r = run_workload(name, variant, tpu=tpu_name,
                             num_warmup=num_warmup, num_iters=num_iters)
            if r is None:
                continue

            results.append(r)

            if r['status'] == 'success':
                tflops_str = f"{r['tflops']:.1f}" if r['tflops'] > 0 else "—"
                util_str = f"{r['utilization_pct']:.1f}%" if r['utilization_pct'] > 0 else "—"
                method = r.get('timing_method', '?')
                print(f"{i:>3} {name:<42} {variant:<10} "
                      f"{r['median_ms']:>10.3f} {tflops_str:>8} "
                      f"{util_str:>7} {method:<18}")
            else:
                err = r.get('error', '?')[:40]
                print(f"{i:>3} {name:<42} {variant:<10} "
                      f"{'ERROR':>10} {'':>8} {'':>7} {err}")

    # Summary
    print()
    successes = [r for r in results if r['status'] == 'success']
    baselines = [r for r in successes if r['variant'] == 'baseline']
    optimized = [r for r in successes if r['variant'] == 'optimized']

    print(f"Total: {len(results)} runs ({len(successes)} success, "
          f"{len(results) - len(successes)} error)")
    print(f"Baselines: {len(baselines)}/{len(workload_names)} succeeded")
    if optimized:
        print(f"Optimized: {len(optimized)} variants")

    profiler_count = sum(1 for r in successes
                         if r.get('timing_method') == 'device_profiler')
    wall_count = sum(1 for r in successes
                     if 'wall_clock' in r.get('timing_method', ''))
    print(f"\nTiming: {profiler_count} device_profiler, {wall_count} wall_clock")

    utils = [r['utilization_pct'] for r in baselines if r['utilization_pct'] > 0]
    if utils:
        print(f"\nTPU Utilization (baselines):")
        print(f"  Median: {np.median(utils):.1f}%")
        print(f"  Mean:   {np.mean(utils):.1f}%")
        print(f"  Max:    {np.max(utils):.1f}%")
        print(f"  Min:    {np.min(utils):.1f}%")

    if optimized:
        print(f"\nOptimized vs Baseline speedups:")
        for opt_r in optimized:
            base_r = next((r for r in baselines if r['name'] == opt_r['name']), None)
            if base_r:
                sp = (base_r['median_ms'] / opt_r['median_ms']
                      if opt_r['median_ms'] > 0 else 0)
                print(f"  {opt_r['name']}: {base_r['median_ms']:.3f}ms -> "
                      f"{opt_r['median_ms']:.3f}ms ({sp:.2f}x)")

    # Save results
    output = {
        'metadata': {
            'jax_version': jax.__version__,
            'devices': str(jax.devices()),
            'tpu': tpu_name,
            'tpu_peak_tflops_bf16': tpu_spec['peak_tflops_bf16'],
            'timing_method': 'jax.profiler device-side (jit_*() events from Perfetto trace)',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'num_workloads': len(workload_names),
            'num_succeeded': len(successes),
            'num_warmup': num_warmup,
            'num_iters': num_iters,
        },
        'results': results,
    }

    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")

    csv_path = os.path.join(output_dir, 'results.csv')
    _save_csv(results, csv_path)
    print(f"CSV saved to {csv_path}")

    return output


def _save_csv(results, path):
    """Save results as CSV, sorted by workload number."""
    from JAXBench.benchmark import _sort_key

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'workload', 'variant', 'status', 'timing_method',
            'median_ms', 'mean_ms', 'std_ms', 'min_ms',
            'xla_flops', 'tflops', 'utilization_pct',
            'output_shape', 'error',
        ])

        for r in sorted(results, key=lambda r: _sort_key(r['name'])):
            writer.writerow([
                r['name'], r['variant'], r['status'],
                r.get('timing_method', ''),
                r.get('median_ms', ''), r.get('mean_ms', ''),
                r.get('std_ms', ''), r.get('min_ms', ''),
                r.get('xla_flops', ''), r.get('tflops', ''),
                r.get('utilization_pct', ''),
                str(r.get('output_shape', '')), r.get('error', ''),
            ])
