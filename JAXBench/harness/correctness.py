"""Correctness checking for kernel outputs.

Uses the same tolerance as KernelBench: atol=1e-2, rtol=1e-2.
"""

import jax
import numpy as np

ATOL = 1e-2
RTOL = 1e-2


def check_correctness(ref_output, test_output, atol=ATOL, rtol=RTOL):
    """Compare reference and test outputs after converting to float32.

    Handles single tensors, tuples, and pytree outputs.

    Returns:
        dict with keys:
            correct (bool): Whether outputs match within tolerance
            max_diff (float): Maximum absolute difference
            reason (str): 'ok', 'output count mismatch', 'shape mismatch: ...', 'values differ'
    """
    ref_flat = jax.tree.leaves(ref_output) if isinstance(ref_output, (tuple, list)) else [ref_output]
    test_flat = jax.tree.leaves(test_output) if isinstance(test_output, (tuple, list)) else [test_output]

    if len(ref_flat) != len(test_flat):
        return {
            'correct': False,
            'max_diff': float('inf'),
            'reason': f'output count mismatch: {len(ref_flat)} vs {len(test_flat)}',
        }

    max_diff = 0.0
    for r, t in zip(ref_flat, test_flat):
        r_np = np.array(r, dtype=np.float32)
        t_np = np.array(t, dtype=np.float32)

        if r_np.shape != t_np.shape:
            return {
                'correct': False,
                'max_diff': float('inf'),
                'reason': f'shape mismatch: {r_np.shape} vs {t_np.shape}',
            }

        diff = float(np.max(np.abs(r_np - t_np)))
        max_diff = max(max_diff, diff)

        if not np.allclose(r_np, t_np, atol=atol, rtol=rtol):
            return {
                'correct': False,
                'max_diff': max_diff,
                'reason': 'values differ',
            }

    return {
        'correct': True,
        'max_diff': max_diff,
        'reason': 'ok',
    }
