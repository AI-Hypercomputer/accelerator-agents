import os
import sys
import importlib.util
import argparse
import jax
import jax.numpy as jnp
import numpy as np

sys.dont_write_bytecode = True

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def check_match(baseline_out, reference_out, problem_name):
    if isinstance(baseline_out, (list, tuple)):
        if (not isinstance(reference_out, (list, tuple))
                or len(baseline_out) != len(reference_out)):
            print(f"  [FAIL] Output structures differ for {problem_name}")
            return False
        
        all_match = True
        for i, (b, r) in enumerate(zip(baseline_out, reference_out)):
            arr_b = np.array(b)
            arr_r = np.array(r)
            if not np.array_equal(arr_b, arr_r, equal_nan=True):
                try:
                    diff = np.max(np.abs(arr_b - arr_r))
                except Exception:
                    diff = "N/A"
                print(
                    f"  [FAIL] Element {i} does not match exactly for "
                    f"{problem_name} (Max diff: {diff})"
                )
                all_match = False
        
        if all_match:
            print(f"  [PASS] {problem_name}")
        return all_match
    else:
        arr_b = np.array(baseline_out)
        arr_r = np.array(reference_out)
        if not np.array_equal(arr_b, arr_r, equal_nan=True):
            try:
                diff = np.max(np.abs(arr_b - arr_r))
            except Exception:
                diff = "N/A"
            print(
                f"  [FAIL] Outputs do not match exactly for "
                f"{problem_name} (Max diff: {diff})"
            )
            return False
        
        print(f"  [PASS] {problem_name}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Verify outputs")
    parser.add_argument(
        "--level", "-l", type=str, default="level2",
        help="Level to test (e.g. level1 or level2)"
    )
    parser.add_argument(
        "--problem", "-p", type=str, default=None,
        help="Name of a specific problem to verify (e.g. 1p_Flash_Attention)"
    )
    args_parsed = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    jaxbench_dir = os.path.join(
        script_dir, f"../../../JAXBench/benchmark/{args_parsed.level}"
    )
    adapted_dir = os.path.join(script_dir, f"./{args_parsed.level}")

    if not os.path.exists(adapted_dir):
        print(f"Directory not found: {adapted_dir}")
        return
        
    problems = [
        d for d in os.listdir(adapted_dir)
        if os.path.isdir(os.path.join(adapted_dir, d))
    ]
    problems.sort()
    
    if args_parsed.problem:
        if args_parsed.problem not in problems:
            print(f"Problem '{args_parsed.problem}' not found in {adapted_dir}")
            return
        problems = [args_parsed.problem]
    
    for problem in problems:
        print(f"Verifying {problem}...")
        try:
            # Load JaxBench baseline
            baseline_path = os.path.join(jaxbench_dir, problem, "baseline.py")
            if not os.path.exists(baseline_path):
                print(
                    f"  [SKIP] No baseline.py found for {problem} in JaxBench"
                )
                continue
            baseline_mod = load_module(f"{problem}_baseline", baseline_path)
            
            # Load Adapted reference
            reference_path = os.path.join(adapted_dir, problem, "reference.py")
            if not os.path.exists(reference_path):
                print(
                    f"  [SKIP] No reference.py found for {problem} in "
                    f"adapted dataset"
                )
                continue
            reference_mod = load_module(f"{problem}_reference", reference_path)
            
            # 1. Run JaxBench version
            if hasattr(baseline_mod, "create_inputs"):
                baseline_inputs = baseline_mod.create_inputs()
            elif hasattr(baseline_mod, "create_input"):
                baseline_inputs = baseline_mod.create_input()
            else:
                print(
                    f"  [ERROR] No create_inputs/create_input found in "
                    f"baseline for {problem}"
                )
                continue
                
            if not isinstance(baseline_inputs, (list, tuple)):
                baseline_inputs = (baseline_inputs,)
                
            baseline_out = baseline_mod.workload(*baseline_inputs)
            if hasattr(baseline_out, "block_until_ready"):
                baseline_out.block_until_ready()
            elif isinstance(baseline_out, (list, tuple)):
                for o in baseline_out:
                    if hasattr(o, "block_until_ready"):
                        o.block_until_ready()
            
            # 2. Run Adapted version
            get_inputs_ret = reference_mod.get_inputs()
            if len(get_inputs_ret) == 2:
                dynamic_args, static_args = get_inputs_ret
            else:
                print(
                    f"  [ERROR] get_inputs() returned {len(get_inputs_ret)} "
                    f"elements instead of 2 for {problem}"
                )
                continue
                
            args = list(dynamic_args)
            if static_args:
                args += list(static_args)
                
            reference_out = reference_mod.computation(*args)
            if hasattr(reference_out, "block_until_ready"):
                reference_out.block_until_ready()
            elif isinstance(reference_out, (list, tuple)):
                for o in reference_out:
                    if hasattr(o, "block_until_ready"):
                        o.block_until_ready()
                
            # 3. Compare outputs
            check_match(baseline_out, reference_out, problem)
            
        except Exception as e:
            print(f"  [ERROR] {problem}: {e}")

if __name__ == '__main__':
    main()
