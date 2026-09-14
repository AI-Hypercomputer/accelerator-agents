#!/usr/bin/env python3
"""Profile a single workload using jax.profiler for device-side timing.

Uses jax.profiler.trace() to capture XProf traces, then parses the
Perfetto JSON for accurate device-side kernel execution times.

Usage (on TPU VM):
    PJRT_DEVICE=TPU python3 benchmark/profile_workload.py <workload_dir> [variant]

Example:
    PJRT_DEVICE=TPU python3 benchmark/profile_workload.py benchmark/8p_GEMM baseline
    PJRT_DEVICE=TPU python3 benchmark/profile_workload.py benchmark/8p_GEMM optimized
"""

import importlib.util
import gzip
import json
import os
import sys
import time
import glob
import shutil

import jax
import jax.numpy as jnp
import numpy as np

TPU_PEAK_TFLOPS_BF16 = 918.0


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def extract_device_times_from_trace(trace_dir):
    """Parse Perfetto JSON trace and extract per-iteration device kernel times.

    Matches on jit_<name>(...) events which wrap the complete device execution
    for each JIT-compiled call. These include all sub-kernels (fusions for XLA,
    named kernels for Pallas) and give true total device time per iteration.
    """
    perfetto_files = glob.glob(f"{trace_dir}/**/perfetto_trace.json.gz", recursive=True)
    if not perfetto_files:
        return None

    with gzip.open(perfetto_files[0], 'rt') as f:
        data = json.load(f)

    events = data.get('traceEvents', data) if isinstance(data, dict) else data

    # jit_workload(...) or jit_<name>(...) events wrap entire device execution
    kernel_times = []
    for e in events:
        if not isinstance(e, dict) or e.get('dur', 0) <= 0:
            continue
        name = e.get('name', '')
        if name.startswith('jit_') and '(' in name:
            kernel_times.append(e['dur'])  # microseconds

    return kernel_times if kernel_times else None


def profile_workload(workload_dir, variant='baseline', num_warmup=5, num_profile_iters=50):
    """Profile a workload variant. Returns result dict with device-side timing."""
    name = os.path.basename(workload_dir)
    module_path = os.path.join(workload_dir, f'{variant}.py')

    if not os.path.exists(module_path):
        return {'name': name, 'variant': variant, 'status': 'error', 'error': f'{variant}.py not found'}

    result = {'name': name, 'variant': variant, 'status': 'error'}

    try:
        mod = load_module(module_path, f'{name}.{variant}')

        skip_jit = getattr(mod, '_skip_jit', False)
        workload_fn = getattr(mod, 'workload')
        create_fn = getattr(mod, 'create_inputs')

        if 'dtype' in create_fn.__code__.co_varnames:
            inputs = create_fn(dtype=jnp.bfloat16)
        else:
            inputs = create_fn()
        if not isinstance(inputs, (list, tuple)):
            inputs = (inputs,)

        # Get FLOP count
        get_flops_fn = getattr(mod, 'get_flops', None)
        if get_flops_fn:
            flops = int(get_flops_fn())
        elif not skip_jit:
            try:
                compiled = jax.jit(workload_fn).lower(*inputs).compile()
                cost = compiled.cost_analysis()
                if isinstance(cost, list):
                    cost = cost[0] if cost else {}
                flops = int(cost.get('flops', 0))
            except:
                flops = 0
        else:
            flops = 0

        if skip_jit:
            # Eager workloads can't use profiler effectively, fall back to wall clock
            for _ in range(num_warmup):
                out = workload_fn(*inputs)
                if hasattr(out, 'block_until_ready'):
                    out.block_until_ready()
            times = []
            for _ in range(num_profile_iters):
                t0 = time.perf_counter()
                out = workload_fn(*inputs)
                if hasattr(out, 'block_until_ready'):
                    out.block_until_ready()
                times.append((time.perf_counter() - t0) * 1000)
            device_times_ms = times
            timing_method = 'wall_clock'
        else:
            jitted = jax.jit(workload_fn)

            # Warmup
            for _ in range(num_warmup):
                out = jitted(*inputs)
                out.block_until_ready()

            # Also measure wall-clock for comparison
            wall_times = []
            for _ in range(num_profile_iters):
                t0 = time.perf_counter()
                out = jitted(*inputs)
                out.block_until_ready()
                wall_times.append((time.perf_counter() - t0) * 1000)

            # Profile with jax.profiler for device-side timing
            # Wrap each call in jax.named_scope so we get a consistent event name
            trace_dir = f'/tmp/jax_profile_{name}_{variant}'
            if os.path.exists(trace_dir):
                shutil.rmtree(trace_dir)
            os.makedirs(trace_dir, exist_ok=True)

            with jax.profiler.trace(trace_dir, create_perfetto_link=False, create_perfetto_trace=True):
                for _ in range(num_profile_iters):
                    with jax.named_scope('bench_kernel'):
                        out = jitted(*inputs)
                    out.block_until_ready()

            # Extract device-side times
            kernel_times_us = extract_device_times_from_trace(trace_dir)

            if kernel_times_us and len(kernel_times_us) >= num_profile_iters:
                device_times_ms = [t / 1000.0 for t in kernel_times_us[:num_profile_iters]]
                timing_method = 'device_profiler'
            else:
                device_times_ms = wall_times
                timing_method = 'wall_clock_fallback'

            # Clean up trace
            shutil.rmtree(trace_dir, ignore_errors=True)

            result['wall_clock_median_ms'] = round(float(np.median(wall_times)), 4)

        # Compute stats
        times_arr = np.array(device_times_ms)
        median_ms = float(np.median(times_arr))
        mean_ms = float(np.mean(times_arr))
        std_ms = float(np.std(times_arr))
        min_ms = float(np.min(times_arr))

        tflops = 0.0
        utilization_pct = 0.0
        if flops > 0 and median_ms > 0:
            tflops = flops / (median_ms / 1000) / 1e12
            utilization_pct = (tflops / TPU_PEAK_TFLOPS_BF16) * 100

        result.update({
            'status': 'success',
            'timing_method': timing_method,
            'median_ms': round(median_ms, 4),
            'mean_ms': round(mean_ms, 4),
            'std_ms': round(std_ms, 4),
            'min_ms': round(min_ms, 4),
            'flops': flops,
            'tflops': round(tflops, 2),
            'utilization_pct': round(utilization_pct, 1),
            'output_shape': list(out.shape) if hasattr(out, 'shape') else [],
            'num_iters': len(device_times_ms),
        })

    except Exception as e:
        import traceback
        result['error'] = str(e)[:300]
        result['traceback'] = traceback.format_exc()[-500:]

    return result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 benchmark/profile_workload.py <workload_dir> [variant]")
        sys.exit(1)

    workload_dir = sys.argv[1]
    variant = sys.argv[2] if len(sys.argv) > 2 else 'baseline'

    print(f"Profiling {os.path.basename(workload_dir)} ({variant})...")
    result = profile_workload(workload_dir, variant)

    print(json.dumps(result, indent=2))
