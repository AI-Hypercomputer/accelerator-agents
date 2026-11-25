"""Regression tests for exact integer output validation."""

import unittest
from pathlib import Path
import tempfile
from unittest import mock

import jax.numpy as jnp
import numpy as np

from JAXBench.harness.correctness import check_correctness
from JAXBench.harness import evaluator


class IntegerCorrectnessTest(unittest.TestCase):
    def test_integer_mismatch_is_not_hidden_by_float_tolerance(self):
        for dtype in (
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ):
            with self.subTest(dtype=dtype):
                result = check_correctness(
                    np.array([100], dtype=dtype), np.array([101], dtype=dtype)
                )
                self.assertFalse(result["correct"])
                self.assertEqual(result["max_diff"], 1.0)

    def test_integer_mismatch_ignores_even_explicitly_loose_tolerances(self):
        result = check_correctness(
            np.array([100], dtype=np.int32),
            np.array([101], dtype=np.int32),
            atol=100,
            rtol=100,
        )
        self.assertFalse(result["correct"])

    def test_adjacent_large_integers_remain_distinguishable(self):
        for dtype, first in (
            (np.int32, 2**24),
            (np.int64, 2**53),
            (np.uint64, 2**64 - 2),
        ):
            with self.subTest(dtype=dtype):
                result = check_correctness(
                    np.array([first], dtype=dtype),
                    np.array([first + 1], dtype=dtype),
                    atol=0,
                    rtol=0,
                )
                self.assertFalse(result["correct"])
                self.assertEqual(result["max_diff"], 1.0)

    def test_equal_values_can_have_different_integer_widths_or_signedness(self):
        for reference_dtype, candidate_dtype, value in (
            (np.int8, np.int64, -1),
            (np.uint8, np.uint64, 100),
            (np.int32, np.uint64, 100),
            (np.uint64, np.int64, 2**63 - 1),
            (jnp.int4, np.int8, -1),
            (jnp.uint4, np.uint64, 7),
        ):
            with self.subTest(
                reference_dtype=reference_dtype, candidate_dtype=candidate_dtype
            ):
                result = check_correctness(
                    np.array([value], dtype=reference_dtype),
                    np.array([value], dtype=candidate_dtype),
                )
                self.assertTrue(result["correct"])
                self.assertEqual(result["max_diff"], 0.0)

    def test_signed_unsigned_comparison_does_not_round_or_wrap(self):
        for signed, unsigned, difference in (
            (2**63 - 1, 2**63, 1),
            (-1, 2**64 - 1, 2**64),
        ):
            for reverse in (False, True):
                with self.subTest(signed=signed, unsigned=unsigned, reverse=reverse):
                    pair = [
                        np.array([signed], dtype=np.int64),
                        np.array([unsigned], dtype=np.uint64),
                    ]
                    if reverse:
                        pair.reverse()
                    result = check_correctness(*pair)
                    self.assertFalse(result["correct"])
                    self.assertEqual(result["max_diff"], float(difference))

    def test_error_metric_does_not_overflow_integer_storage(self):
        for dtype in (np.int8, np.int16, np.int32, np.int64):
            with self.subTest(dtype=dtype):
                bounds = np.iinfo(dtype)
                result = check_correctness(
                    np.array([bounds.min], dtype=dtype),
                    np.array([bounds.max], dtype=dtype),
                )
                self.assertFalse(result["correct"])
                self.assertEqual(
                    result["max_diff"], float(int(bounds.max) - int(bounds.min))
                )

    def test_integer_and_float_outputs_are_not_interchangeable(self):
        pair = [np.array([100], dtype=np.int32), np.array([100.0], dtype=np.float32)]
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                ordered_pair = pair[::-1] if reverse else pair
                result = check_correctness(*ordered_pair)
                self.assertFalse(result["correct"])
                self.assertIn("dtype", result["reason"])

    def test_low_bit_integer_dtypes_are_compared_exactly(self):
        for dtype in (jnp.int4, jnp.uint4):
            with self.subTest(dtype=dtype):
                result = check_correctness(
                    np.array([6], dtype=dtype),
                    np.array([7], dtype=dtype),
                    atol=100,
                    rtol=100,
                )
                self.assertFalse(result["correct"])
                self.assertEqual(result["max_diff"], 1.0)

    def test_jax_integer_arrays_are_compared_exactly(self):
        result = check_correctness(
            jnp.array([100], dtype=jnp.int32), jnp.array([101], dtype=jnp.int32)
        )
        self.assertFalse(result["correct"])

    def test_integer_leaf_is_exact_inside_nested_output(self):
        reference = (np.array([1.0]), [np.array([100], dtype=np.int32)])
        candidate = (np.array([1.005]), [np.array([101], dtype=np.int32)])
        result = check_correctness(reference, candidate)
        self.assertFalse(result["correct"])
        self.assertEqual(result["max_diff"], 1.0)

    def test_floating_point_tolerance_is_preserved(self):
        reference = np.array([1.0], dtype=np.float32)
        self.assertTrue(check_correctness(reference, reference + 0.005)["correct"])
        self.assertFalse(check_correctness(reference, reference + 0.1)["correct"])

    def test_bfloat16_tolerance_is_preserved(self):
        result = check_correctness(
            np.array([1.0], dtype=jnp.bfloat16),
            np.array([1.0078125], dtype=jnp.bfloat16),
        )
        self.assertTrue(result["correct"])

    def test_integer_shape_mismatch_is_rejected(self):
        result = check_correctness(
            np.array([1, 2], dtype=np.int32),
            np.array([[1, 2]], dtype=np.int32),
        )
        self.assertFalse(result["correct"])
        self.assertIn("shape mismatch", result["reason"])


class IntegerEvaluationTest(unittest.TestCase):
    """Exercise the real CPU JAX correctness gate, with discovery and timing stubbed."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.directory = Path(self.temp_dir.name)
        (self.directory / "baseline.py").write_text(
            "import jax.numpy as jnp\n"
            "def create_inputs():\n"
            "    return (jnp.array([100], dtype=jnp.int32),)\n"
            "def workload(x):\n"
            "    return x\n"
        )

    def evaluate(self, expression):
        candidate = self.directory / "candidate.py"
        candidate.write_text(f"def workload(x):\n    return {expression}\n")

        def fake_benchmark(fn, inputs, **kwargs):
            return {
                "output": fn(*inputs),
                "median_ms": 1.0,
                "mean_ms": 1.0,
                "std_ms": 0.0,
                "min_ms": 1.0,
                "timing_method": "test_stub",
            }

        with (
            mock.patch.object(
                evaluator, "get_workload_dir", return_value=str(self.directory)
            ),
            mock.patch.object(
                evaluator,
                "get_tpu_spec",
                return_value=("v6e", {"peak_tflops_bf16": 918.0}),
            ),
            mock.patch.object(evaluator, "has_optimized", return_value=False),
            mock.patch.object(
                evaluator, "benchmark_fn", side_effect=fake_benchmark
            ) as benchmark,
        ):
            result = evaluator.evaluate_kernel("synthetic", str(candidate), tpu="v6e")
        return result, benchmark

    def test_wrong_integer_candidate_is_rejected_before_benchmarking(self):
        result, benchmark = self.evaluate("x + 1")
        self.assertEqual(result["status"], "incorrect")
        self.assertEqual(result["correctness"]["max_diff"], 1.0)
        benchmark.assert_not_called()

    def test_float_candidate_cannot_replace_integer_output(self):
        result, benchmark = self.evaluate('x.astype("float32")')
        self.assertEqual(result["status"], "incorrect")
        self.assertIn("dtype", result["correctness"]["reason"])
        benchmark.assert_not_called()

    def test_matching_integer_candidate_reaches_benchmarking(self):
        result, benchmark = self.evaluate("x")
        self.assertEqual(result["status"], "correct")
        self.assertTrue(result["correctness"]["correct"])
        self.assertEqual(benchmark.call_count, 2)


if __name__ == "__main__":
    unittest.main()
