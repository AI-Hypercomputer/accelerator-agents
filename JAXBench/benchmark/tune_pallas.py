#!/usr/bin/env python3
"""Grid search tuning for Pallas kernel block sizes.

Runs a grid search over block size configurations for each Pallas-optimized
priority kernel, measuring device-side timing via jax.profiler.

Usage (on TPU VM):
    PJRT_DEVICE=TPU python3 benchmark/tune_pallas.py [workload_name]

Without arguments, tunes all priority kernels with optimized.py.
"""

import importlib.util
import gzip
import json
import os
import sys
import time
import glob
import shutil
import copy

import jax
import jax.numpy as jnp
import numpy as np

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BENCHMARK_DIR))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def extract_device_times(trace_dir):
    """Parse Perfetto trace for device kernel times.

    Matches jit_*() events for total device time per iteration.
    """
    perfetto_files = glob.glob(f"{trace_dir}/**/perfetto_trace.json.gz", recursive=True)
    if not perfetto_files:
        return None
    with gzip.open(perfetto_files[0], 'rt') as f:
        data = json.load(f)
    events = data.get('traceEvents', data) if isinstance(data, dict) else data

    kernel_times = []
    for e in events:
        if not isinstance(e, dict) or e.get('dur', 0) <= 0:
            continue
        name = e.get('name', '')
        if name.startswith('jit_') and '(' in name:
            kernel_times.append(e['dur'] / 1000.0)  # us -> ms

    return kernel_times if kernel_times else None


def benchmark_config(workload_dir, config_override, num_warmup=3, num_iters=20):
    """Benchmark a workload with a specific TUNED_PARAMS config. Returns median_ms or None on error."""
    name = os.path.basename(workload_dir)
    module_path = os.path.join(workload_dir, 'optimized.py')

    try:
        # Reload module fresh each time
        if f'{name}.optimized_tune' in sys.modules:
            del sys.modules[f'{name}.optimized_tune']

        mod = load_module(module_path, f'{name}.optimized_tune')

        # Override TUNED_PARAMS
        mod.TUNED_PARAMS = config_override

        create_fn = mod.create_inputs
        if 'dtype' in create_fn.__code__.co_varnames:
            inputs = create_fn(dtype=jnp.bfloat16)
        else:
            inputs = create_fn()
        if not isinstance(inputs, (list, tuple)):
            inputs = (inputs,)

        jax.clear_caches()

        jitted = jax.jit(mod.workload)

        # Warmup
        for _ in range(num_warmup):
            out = jitted(*inputs)
            out.block_until_ready()

        # Profile
        trace_dir = f'/tmp/jax_tune_{name}'
        if os.path.exists(trace_dir):
            shutil.rmtree(trace_dir)
        os.makedirs(trace_dir, exist_ok=True)

        with jax.profiler.trace(trace_dir, create_perfetto_link=False, create_perfetto_trace=True):
            for _ in range(num_iters):
                with jax.named_scope('bench_kernel'):
                    out = jitted(*inputs)
                out.block_until_ready()

        times = extract_device_times(trace_dir)
        shutil.rmtree(trace_dir, ignore_errors=True)

        if times and len(times) >= num_iters:
            return float(np.median(times[:num_iters]))

        # Fallback to wall clock
        wall_times = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            out = jitted(*inputs)
            out.block_until_ready()
            wall_times.append((time.perf_counter() - t0) * 1000)
        return float(np.median(wall_times))

    except Exception as e:
        print(f"    Error with config {config_override}: {str(e)[:100]}")
        return None


# --- Search spaces for each kernel ---

SEARCH_SPACES = {
    '8p_GEMM': {
        'param_name': ['block_shape', 'block_k'],
        'configs': [
            {'block_shape': [bm, bn], 'block_k': bk}
            for bm in [256, 512, 1024]
            for bn in [512, 1024, 2048]
            for bk in [256, 512, 1024]
            if bm * bk <= 1024 * 1024  # VMEM constraint
            and bk * bn <= 2048 * 1024
        ],
    },
    '11p_Megablox_GMM': {
        'param_name': ['tiling'],
        'configs': [
            {'tiling': [tm, tk, tn]}
            for tm in [64, 128, 256]
            for tk in [256, 512, 1024]
            for tn in [256, 512, 1024]
        ],
    },
    '6p_Paged_Attention': {
        'param_name': ['pages_per_compute_block'],
        'configs': [
            {'pages_per_compute_block': p}
            for p in [16, 32, 64, 128]
        ],
    },
    '7p_Ragged_Paged_Attention': {
        'param_name': ['num_kv_pages_per_block', 'num_queries_per_block', 'vmem_limit_bytes'],
        'configs': [
            {'num_kv_pages_per_block': kv, 'num_queries_per_block': q, 'vmem_limit_bytes': 33554432}
            for kv in [16, 32, 64]
            for q in [32, 64, 128]
        ],
    },
    # Attention kernels — search the most impactful forward-pass params
    '1p_Flash_Attention': {
        'param_name': ['block_q', 'block_k_major', 'block_k'],
        'configs': [
            {
                'block_q': bq, 'block_k_major': bkm, 'block_k': bk,
                'block_b': 1,
                'block_q_major_dkv': 128, 'block_k_major_dkv': 128,
                'block_k_dkv': 128, 'block_q_dkv': 128,
                'block_k_major_dq': 128, 'block_k_dq': 128, 'block_q_dq': 128,
            }
            for bq in [512, 1024, 2048, 4096]
            for bkm in [512, 1024, 2048, 4096]
            for bk in [256, 512, 1024]
            if bkm >= bk  # block_k_major >= block_k
            and bq <= 4096 and bkm <= 4096  # seq_len constraint
        ],
    },
    '2p_GQA_Attention': {
        'param_name': ['block_q', 'block_kv', 'block_kv_compute'],
        'configs': [
            {
                'block_q': bq, 'block_kv': bkv, 'block_kv_compute': bkvc,
                'q_layout': 1, 'k_layout': 1, 'v_layout': 1,
                'head_shards': 1, 'q_seq_shards': 1,
                'block_q_dkv': None, 'block_kv_dkv': None, 'block_kv_dkv_compute': None,
                'block_q_dq': None, 'block_kv_dq': None,
            }
            for bq in [512, 1024, 2048]
            for bkv in [512, 1024, 2048]
            for bkvc in [256, 512, 1024]
            if bkv >= bkvc
        ],
    },
    '3p_MLA_Attention': {
        'param_name': ['block_q', 'block_k_major', 'block_k'],
        'configs': [
            {
                'block_q': bq, 'block_k_major': bkm, 'block_k': bk,
                'block_b': 1,
                'block_q_major_dkv': 128, 'block_k_major_dkv': 128,
                'block_k_dkv': 128, 'block_q_dkv': 128,
                'block_k_major_dq': 128, 'block_k_dq': 128, 'block_q_dq': 128,
            }
            for bq in [512, 1024, 2048]
            for bkm in [512, 1024, 2048]
            for bk in [256, 512, 1024]
            if bkm >= bk
        ],
    },
    '4p_Sparse_Attention': {
        'param_name': ['block_q', 'block_kv', 'block_kv_compute'],
        'configs': [
            {
                'block_q': bq, 'block_kv': bkv, 'block_kv_compute': bkvc,
                'q_layout': 1, 'k_layout': 1, 'v_layout': 1,
                'head_shards': 1, 'q_seq_shards': 1,
                'block_q_dkv': None, 'block_kv_dkv': None, 'block_kv_dkv_compute': None,
                'block_q_dq': None, 'block_kv_dq': None,
            }
            for bq in [512, 1024, 2048, 4096]
            for bkv in [512, 1024, 2048, 4096]
            for bkvc in [256, 512, 1024]
            if bkv >= bkvc
        ],
    },
}


def tune_workload(workload_dir):
    """Run grid search tuning for a single workload."""
    name = os.path.basename(workload_dir)
    optimized_path = os.path.join(workload_dir, 'optimized.py')

    if not os.path.exists(optimized_path):
        print(f"  No optimized.py found for {name}, skipping")
        return None

    if name not in SEARCH_SPACES:
        print(f"  No search space defined for {name}, skipping")
        return None

    space = SEARCH_SPACES[name]
    configs = space['configs']
    print(f"\n{'='*60}")
    print(f"Tuning {name}: {len(configs)} configurations")
    print(f"{'='*60}")

    # Get current best
    mod = load_module(optimized_path, f'{name}.optimized_check')
    current_params = copy.deepcopy(mod.TUNED_PARAMS)
    print(f"Current params: {current_params}")

    # Benchmark current
    current_time = benchmark_config(workload_dir, current_params)
    print(f"Current time: {current_time:.3f}ms" if current_time else "Current: FAILED")

    best_time = current_time or float('inf')
    best_config = current_params
    results = []

    for i, config in enumerate(configs):
        print(f"  [{i+1}/{len(configs)}] {config} ... ", end='', flush=True)
        t = benchmark_config(workload_dir, config)
        if t is not None:
            print(f"{t:.3f}ms {'*BEST*' if t < best_time else ''}")
            results.append({'config': config, 'median_ms': t})
            if t < best_time:
                best_time = t
                best_config = config
        else:
            print("FAILED")
            results.append({'config': config, 'median_ms': None})

    print(f"\nBest config: {best_config}")
    print(f"Best time: {best_time:.3f}ms")
    if current_time:
        speedup = current_time / best_time
        print(f"Speedup vs current: {speedup:.2f}x")

    return {
        'workload': name,
        'current_params': current_params,
        'current_time_ms': current_time,
        'best_params': best_config,
        'best_time_ms': best_time,
        'speedup': current_time / best_time if current_time else None,
        'all_results': results,
    }


def main():
    print(f"JAX {jax.__version__} | Devices: {jax.devices()}")

    if len(sys.argv) > 1:
        # Tune specific workload
        workload_name = sys.argv[1]
        workload_dir = os.path.join(BENCHMARK_DIR, workload_name)
        result = tune_workload(workload_dir)
        if result:
            print(json.dumps(result, indent=2, default=str))
    else:
        # Tune all priority kernels with optimized.py
        results = {}
        for name in sorted(SEARCH_SPACES.keys()):
            workload_dir = os.path.join(BENCHMARK_DIR, name)
            if os.path.exists(os.path.join(workload_dir, 'optimized.py')):
                result = tune_workload(workload_dir)
                if result:
                    results[name] = result

        # Save results
        output_path = os.path.join(BENCHMARK_DIR, 'tuning_results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nTuning results saved to {output_path}")

        # Summary
        print(f"\n{'='*60}")
        print("TUNING SUMMARY")
        print(f"{'='*60}")
        for name, r in results.items():
            sp = r.get('speedup', 0) or 0
            print(f"  {name}: {r['current_time_ms']:.3f}ms -> {r['best_time_ms']:.3f}ms ({sp:.2f}x)")


if __name__ == '__main__':
    main()
