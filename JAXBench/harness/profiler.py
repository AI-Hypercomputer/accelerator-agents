"""Device-side profiling via jax.profiler.trace() and Perfetto trace parsing.

Captures TPU kernel execution times by parsing jit_*() events from Perfetto
JSON traces, measuring only actual device execution (excluding host dispatch).
"""

import glob
import gzip
import json
import os
import shutil
import time

import jax
import jax.numpy as jnp
import numpy as np


def extract_device_times(trace_dir, num_iters, is_eager=False):
    """Parse Perfetto JSON trace and extract per-iteration device kernel times.

    For JIT workloads: matches jit_workload(...) events which wrap the complete
    device execution per iteration (including all sub-fusions/Pallas kernels).

    For eager (_skip_jit) workloads: collects all jit_*() device events and
    groups them into per-iteration batches.

    Returns list of durations in milliseconds, or None if no events found.
    """
    perfetto_files = glob.glob(
        f"{trace_dir}/**/perfetto_trace.json.gz", recursive=True
    )
    if not perfetto_files:
        return None

    with gzip.open(perfetto_files[0], 'rt') as f:
        data = json.load(f)

    events = data.get('traceEvents', data) if isinstance(data, dict) else data

    if not is_eager:
        kernel_times_ms = []
        for e in events:
            if not isinstance(e, dict) or e.get('dur', 0) <= 0:
                continue
            name = e.get('name', '')
            if name.startswith('jit_') and '(' in name:
                kernel_times_ms.append(e['dur'] / 1000.0)
        return kernel_times_ms if kernel_times_ms else None
    else:
        all_jit_times = []
        for e in events:
            if not isinstance(e, dict) or e.get('dur', 0) <= 0:
                continue
            name = e.get('name', '')
            if name.startswith('jit_') and '(' in name:
                all_jit_times.append(e['dur'] / 1000.0)

        if not all_jit_times:
            return None

        ops_per_iter = len(all_jit_times) // num_iters if num_iters > 0 else 0
        if ops_per_iter <= 0:
            return None

        iter_times = []
        for i in range(num_iters):
            batch = all_jit_times[i * ops_per_iter:(i + 1) * ops_per_iter]
            iter_times.append(sum(batch))
        return iter_times


def benchmark_fn(fn, inputs, num_warmup=5, num_iters=50, skip_jit=False,
                 label='workload'):
    """Run a function with device-side profiling.

    Args:
        fn: The function to benchmark (raw, not yet JIT'd unless skip_jit=True).
        inputs: Tuple of input tensors.
        num_warmup: Number of warmup iterations.
        num_iters: Number of timed iterations.
        skip_jit: If True, run eagerly (don't wrap in jax.jit).
        label: Label for trace directory naming.

    Returns:
        dict with keys:
            timing_method: 'device_profiler' or 'wall_clock_fallback'
            median_ms, mean_ms, std_ms, min_ms: float
            wall_clock_median_ms: float
            raw_times_ms: list[float]
    """
    run_fn = fn if skip_jit else jax.jit(fn)
    bench_iters = 10 if skip_jit else num_iters

    # Warmup
    for _ in range(num_warmup):
        out = run_fn(*inputs)
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()

    # Wall-clock timing
    wall_times = []
    for _ in range(bench_iters):
        t0 = time.perf_counter()
        out = run_fn(*inputs)
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
        wall_times.append((time.perf_counter() - t0) * 1000)

    # Device-side profiler timing
    trace_dir = f'/tmp/jax_profile_{label}_{os.getpid()}'
    if os.path.exists(trace_dir):
        shutil.rmtree(trace_dir)
    os.makedirs(trace_dir, exist_ok=True)

    with jax.profiler.trace(trace_dir, create_perfetto_link=False,
                            create_perfetto_trace=True):
        for _ in range(bench_iters):
            with jax.named_scope('bench_kernel'):
                out = run_fn(*inputs)
            if hasattr(out, 'block_until_ready'):
                out.block_until_ready()

    device_times = extract_device_times(trace_dir, bench_iters, is_eager=skip_jit)
    shutil.rmtree(trace_dir, ignore_errors=True)

    if device_times and len(device_times) >= bench_iters:
        times = device_times[:bench_iters]
        timing_method = 'device_profiler'
    else:
        times = wall_times
        timing_method = 'wall_clock_fallback'

    times_arr = np.array(times)
    return {
        'timing_method': timing_method,
        'median_ms': round(float(np.median(times_arr)), 4),
        'mean_ms': round(float(np.mean(times_arr)), 4),
        'std_ms': round(float(np.std(times_arr)), 4),
        'min_ms': round(float(np.min(times_arr)), 4),
        'wall_clock_median_ms': round(float(np.median(wall_times)), 4),
        'raw_times_ms': [round(t, 4) for t in times],
        'output': out,
    }
