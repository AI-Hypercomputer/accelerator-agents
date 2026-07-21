"""Rigorous test harness template for Pallas kernels."""

TEST_TEMPLATE = """
import time
import json
import jax
import jax.numpy as jnp
import traceback
import sys

# Safe import for base kernel
try:
    import base_kernel as base_mod
except ImportError:
    base_mod = None

# Safe import for optimized kernel
try:
    import optimized_kernel as optimized_mod
except ImportError:
    optimized_mod = base_mod

{input_gen_code}

def benchmark(func, args, static_argnums, num_runs=20, num_warmups=5):
    dynamic_args = tuple(arg for i, arg in enumerate(args) if i not in static_argnums)

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
        else:
            return jax.block_until_ready(res)

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

        if hasattr(compiled_func, 'input_formats'):
            arg_formats, _ = compiled_func.input_formats
            
            @jax.jit
            def enforce_layout(*xs):
                return xs
                
            enforce_layout_compiled = jax.jit(
                enforce_layout,
                out_shardings=arg_formats
            ).lower(*dynamic_args).compile()
            
            dynamic_args = enforce_layout_compiled(*dynamic_args)
    except Exception as e:
        compiled_func = jax.jit(benchmark_func, static_argnums=static_argnums).lower(*args).compile()

    for _ in range(num_warmups):
        res = compiled_func(*dynamic_args)
    jax.block_until_ready(res)

    start = time.perf_counter()
    for _ in range(num_runs):
        res = compiled_func(*dynamic_args)
    jax.block_until_ready(res)
    end = time.perf_counter()
    return (end - start) / num_runs

def main():
    try:
        raw_inputs = get_inputs()
        
        # Check if the input is a list of tuples or a single tuple
        if isinstance(raw_inputs, list):
            inputs_list = raw_inputs
        elif isinstance(raw_inputs, tuple) and len(raw_inputs) == 2:
            inputs_list = [raw_inputs]
        else:
            raise ValueError("get_inputs() must return a list of tuples or "
            "a single (dynamic_args, static_args) tuple.")

        if not base_mod or not hasattr(base_mod, '{kernel_name}'):
            raise RuntimeError("base_kernel.{kernel_name} not found.")
            
        if not optimized_mod or not hasattr(optimized_mod, 'computation'):
            raise RuntimeError("optimized_kernel.computation not found.")

        all_correct = True
        times_base = []
        times_optimized = []

        for idx, inputs in enumerate(inputs_list):
            if not isinstance(inputs, tuple) or len(inputs) != 2:
                raise ValueError(f"Each input config must return exactly 2 "
                                 f"elements: (dynamic_args, static_args). "
                                 f"Got: {{type(inputs)}}")

            dynamic_args, static_args = inputs
            args = tuple(dynamic_args) + tuple(static_args)
            static_argnums = tuple(range(len(dynamic_args), len(args)))

            jit_base = jax.jit(getattr(base_mod, '{kernel_name}'), static_argnums=static_argnums)
            out_base = jax.block_until_ready(jit_base(*args))
            out_base_cpu = jax.device_get(out_base)
            del out_base

            for leaf in jax.tree_util.tree_leaves(out_base_cpu):
                if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
                    if jnp.issubdtype(leaf.dtype, jnp.floating) or jnp.issubdtype(leaf.dtype, jnp.complexfloating):
                        val = jnp.nan
                    elif jnp.issubdtype(leaf.dtype, jnp.bool_):
                        val = True
                    else:
                        val = 123
                    dummy = jnp.full(leaf.shape, val, dtype=leaf.dtype)
                    dummy.block_until_ready()
                    del dummy

            jit_optimized = jax.jit(getattr(optimized_mod, 'computation'), static_argnums=static_argnums)
            out_optimized = jax.block_until_ready(jit_optimized(*args))
            out_optimized_cpu = jax.device_get(out_optimized)
            del out_optimized

            out_base_flat = jax.tree_util.tree_leaves(out_base_cpu)
            out_optimized_flat = jax.tree_util.tree_leaves(out_optimized_cpu)

            is_correct = True
            if len(out_base_flat) != len(out_optimized_flat):
                is_correct = False
                print(f"Output count mismatch for input config {{idx}}: Expected {{len(out_base_flat)}}, Got {{len(out_optimized_flat)}}")
            else:
                for i, (b, o) in enumerate(zip(out_base_flat, out_optimized_flat)):
                    if b.shape != o.shape:
                        is_correct = False
                        print(f"Mismatch in output tensor {{i}} for input config {{idx}}:")
                        print(f"  Expected shape: {{b.shape}}, Got shape: {{o.shape}}")
                        continue
                    
                    match = bool(jnp.allclose(b, o, atol={atol}, rtol={rtol}))
                    if not match:
                        is_correct = False
                        print(f"Mismatch in output tensor {{i}} for input config {{idx}}:")
                        max_diff = jnp.max(jnp.abs(b - o))
                        print(f"  Max absolute difference: {{max_diff}}")
                        diff_mask = jnp.abs(b - o) > {atol} + {rtol} * jnp.abs(b)
                        print(f"  Mismatched elements: {{jnp.sum(diff_mask)}} / {{b.size}} ({{(jnp.sum(diff_mask)/b.size)*100:.2f}}%)")

            if not is_correct:
                all_correct = False
                continue

            time_base = benchmark(getattr(base_mod, '{kernel_name}'), args, static_argnums)
            time_optimized = benchmark(getattr(optimized_mod, 'computation'), args, static_argnums)
            
            times_base.append(time_base)
            times_optimized.append(time_optimized)
            speedup = (time_base / time_optimized) if time_optimized > 0 else 0
            print(f"SPEEDUP_CASE_{{idx}}: {{speedup:.2f}}")

        print(f"CORRECTNESS: {{all_correct}}")

        if not all_correct:
            sys.exit(1)

        import math
        valid_opt = [t for t in times_optimized if t > 0]
        valid_base = [t for t in times_base if t > 0]
        
        if valid_opt and valid_base:
            geo_mean_time_opt = math.exp(sum(math.log(t) for t in valid_opt) / len(valid_opt))
            geo_mean_time_base = math.exp(sum(math.log(t) for t in valid_base) / len(valid_base))
            geo_mean_speedup = (geo_mean_time_base / geo_mean_time_opt) if geo_mean_time_opt > 0 else 0

            print(f"PERF_METRICS: {{geo_mean_time_opt * 1000:.6f}}")
            print(f"RESULT_TIME: {{geo_mean_time_opt * 1000:.6f}} ms")
            print(f"SPEEDUP: {{geo_mean_speedup:.2f}}")
    except Exception as e:
        print(f"ERROR: {{e}}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
