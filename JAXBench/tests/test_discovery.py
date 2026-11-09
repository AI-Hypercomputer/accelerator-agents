"""CPU-only regression tests for level-aware workload discovery and CLI routing."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from JAXBench import benchmark
from JAXBench import __main__ as cli


class WorkloadDiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        for level in ("level1", "level2"):
            (self.root / level).mkdir()
        patcher = mock.patch.object(benchmark, "BENCHMARK_DIR", str(self.root))
        patcher.start()
        self.addCleanup(patcher.stop)

    def add_workload(self, level, name, optimized=False):
        directory = self.root / level / name
        directory.mkdir()
        # Discovery must not import workload modules or require JAX/TPU access.
        (directory / "baseline.py").write_text("raise RuntimeError('not imported')\n")
        if optimized:
            (directory / "optimized.py").touch()
        return str(directory)

    def test_default_level_discovers_only_valid_workloads_in_numeric_order(self):
        self.add_workload("level1", "10p_Tenth")
        self.add_workload("level1", "2p_Second")
        self.add_workload("level1", "_private")
        self.add_workload("level1", ".hidden")
        self.add_workload("level2", "1p_Other")
        (self.root / "level1" / "3p_NoBaseline").mkdir()
        (self.root / "level1" / "4p_NotADirectory").touch()

        self.assertEqual(benchmark.list_workloads(), ["2p_Second", "10p_Tenth"])

    def test_same_name_resolves_to_selected_level(self):
        level1_path = self.add_workload("level1", "8p_GEMM", optimized=True)
        level2_path = self.add_workload("level2", "8p_GEMM")

        self.assertEqual(benchmark.get_workload_dir("8p_GEMM"), level1_path)
        self.assertEqual(
            benchmark.get_workload_dir("8p_GEMM", level="level2"), level2_path
        )
        self.assertEqual(benchmark.list_workloads(level="level2"), ["8p_GEMM"])
        self.assertTrue(benchmark.has_optimized("8p_GEMM"))
        self.assertFalse(benchmark.has_optimized("8p_GEMM", level="level2"))

    def test_missing_workload_does_not_fall_back_to_another_level(self):
        self.add_workload("level1", "8p_GEMM", optimized=True)

        with self.assertRaisesRegex(ValueError, "level2"):
            benchmark.get_workload_dir("8p_GEMM", level="level2")
        self.assertFalse(benchmark.has_optimized("8p_GEMM", level="level2"))

    def test_invalid_level_is_rejected_by_each_helper(self):
        for function, args in (
            (benchmark.list_workloads, ()),
            (benchmark.get_workload_dir, ("8p_GEMM",)),
            (benchmark.has_optimized, ("8p_GEMM",)),
        ):
            with self.subTest(function=function.__name__):
                with self.assertRaises(ValueError):
                    function(*args, level="../level1")


class WorkloadCLITest(unittest.TestCase):
    def test_list_cli_matches_each_level_without_loading_jax(self):
        repo_root = Path(__file__).resolve().parents[2]
        for level in ("level1", "level2"):
            with self.subTest(level=level):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "JAXBench",
                        "list",
                        "--level",
                        level,
                        "--json",
                    ],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                items = json.loads(result.stdout)
                expected = sorted(
                    directory.name
                    for directory in (
                        repo_root / "JAXBench/benchmark" / level
                    ).iterdir()
                    if directory.is_dir() and (directory / "baseline.py").is_file()
                )
                self.assertTrue(expected)
                self.assertEqual(sorted(item["name"] for item in items), expected)
                for item in items:
                    self.assertEqual(
                        item["has_optimized"],
                        (
                            repo_root
                            / "JAXBench/benchmark"
                            / level
                            / item["name"]
                            / "optimized.py"
                        ).is_file(),
                    )

    def test_list_cli_defaults_to_level1(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with mock.patch.object(sys, "argv", ["JAXBench", "list", "--json"]):
                self.assertEqual(cli.main(), 0)
        items = json.loads(output.getvalue())
        self.assertEqual(
            [item["name"] for item in items], benchmark.list_workloads(level="level1")
        )

    def test_run_cli_forwards_level_to_both_variants_and_batch_runner(self):
        runner = types.ModuleType("JAXBench.harness.runner")
        runner.run_workload = mock.Mock(return_value=None)
        runner.run_all = mock.Mock(return_value={})
        modules = {
            "JAXBench.harness": types.ModuleType("JAXBench.harness"),
            "JAXBench.harness.runner": runner,
        }
        options = [
            "--level",
            "level2",
            "--tpu",
            "v6e",
            "--num-warmup",
            "1",
            "--num-iters",
            "2",
        ]
        with mock.patch.dict(sys.modules, modules):
            with mock.patch.object(
                sys, "argv", ["JAXBench", "run", "--workload", "8p_GEMM", *options]
            ):
                self.assertEqual(cli.main(), 0)
            self.assertEqual(
                runner.run_workload.call_args_list,
                [
                    mock.call(
                        name="8p_GEMM",
                        variant=variant,
                        level="level2",
                        tpu="v6e",
                        num_warmup=1,
                        num_iters=2,
                    )
                    for variant in ("baseline", "optimized")
                ],
            )
            with mock.patch.object(sys, "argv", ["JAXBench", "run", "--all", *options]):
                self.assertEqual(cli.main(), 0)
            runner.run_all.assert_called_once_with(
                level="level2", tpu="v6e", num_warmup=1, num_iters=2
            )

    def test_evaluate_cli_forwards_level(self):
        evaluator = types.ModuleType("JAXBench.harness.evaluator")
        evaluator.evaluate_kernel = mock.Mock(return_value={"status": "correct"})
        evaluator.format_eval_result = mock.Mock(return_value="correct")
        modules = {
            "JAXBench.harness": types.ModuleType("JAXBench.harness"),
            "JAXBench.harness.evaluator": evaluator,
        }
        argv = [
            "JAXBench",
            "evaluate",
            "--workload",
            "8p_GEMM",
            "--kernel",
            "candidate.py",
            "--level",
            "level2",
            "--tpu",
            "v6e",
            "--num-warmup",
            "1",
            "--num-iters",
            "2",
        ]
        with (
            mock.patch.dict(sys.modules, modules),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(cli.main(), 0)
        evaluator.evaluate_kernel.assert_called_once_with(
            workload_name="8p_GEMM",
            kernel_path="candidate.py",
            level="level2",
            tpu="v6e",
            num_warmup=1,
            num_iters=2,
        )


if __name__ == "__main__":
    unittest.main()
