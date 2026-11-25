"""Correctness checking for kernel outputs.

Integer outputs must match exactly. Floating outputs use the same tolerance
as KernelBench: atol=1e-2, rtol=1e-2.
"""

import jax
import numpy as np

ATOL = 1e-2
RTOL = 1e-2


def check_correctness(ref_output, test_output, atol=ATOL, rtol=RTOL):
    """Compare integers exactly and floating outputs with float32 tolerances.

    Handles single tensors, tuples, and pytree outputs.
    Integer widths may differ, but integer and non-integer outputs cannot mix.

    Returns:
        dict with keys:
            correct (bool): Whether outputs match within tolerance
            max_diff (float): Maximum absolute difference
            reason (str): 'ok', 'output count mismatch', 'shape mismatch: ...',
                'dtype mismatch: ...', 'values differ'
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
        r_np = np.asarray(r)
        t_np = np.asarray(t)

        if r_np.shape != t_np.shape:
            return {
                'correct': False,
                'max_diff': float('inf'),
                'reason': f'shape mismatch: {r_np.shape} vs {t_np.shape}',
            }

        # JAX also recognizes low-bit integer dtypes such as int4/uint4.
        r_integer = jax.dtypes.issubdtype(r_np.dtype, np.integer)
        t_integer = jax.dtypes.issubdtype(t_np.dtype, np.integer)
        if r_integer or t_integer:
            if not (r_integer and t_integer):
                return {
                    'correct': False,
                    'max_diff': float('inf'),
                    'reason': f'dtype mismatch: {r_np.dtype} vs {t_np.dtype}',
                }

            r_signed = jax.dtypes.issubdtype(r_np.dtype, np.signedinteger)
            t_signed = jax.dtypes.issubdtype(t_np.dtype, np.signedinteger)
            if r_signed != t_signed:
                # NumPy can promote int64/uint64 comparisons to float64.
                # Reject negatives before comparing both arrays as uint64.
                signed = r_np if r_signed else t_np
                equal = np.all(signed >= 0) and np.array_equal(
                    r_np.astype(np.uint64, copy=False), t_np.astype(np.uint64, copy=False)
                )
            else:
                equal = np.array_equal(r_np, t_np)
            if equal:
                continue

            # Python integers avoid overflow and preserve unit differences at
            # int64/uint64 limits. Only mismatches need this slower conversion.
            diff = float(np.max(np.abs(r_np.astype(object) - t_np.astype(object))))
            return {
                'correct': False,
                'max_diff': max(max_diff, diff),
                'reason': 'values differ',
            }

        r_np = np.asarray(r_np, dtype=np.float32)
        t_np = np.asarray(t_np, dtype=np.float32)
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
