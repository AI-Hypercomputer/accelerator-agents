---
title: 'Module: dlpack/collect_env.py'
type: catalog
provenance: extracted
module: dlpack/collect_env.py
status: fresh
symbol_base: scip-python python learning-machine 0.0.0 `dlpack.collect_env`/
symbols:
  get_env_info: get_env_info().
  pretty_str: pretty_str().
  get_os: get_os().
  get_conda_packages: get_conda_packages().
  get_platform: get_platform().
  get_pip_packages: get_pip_packages().
  TORCH_AVAILABLE: TORCH_AVAILABLE.
  run_and_parse_first_match: run_and_parse_first_match().
  get_nvidia_driver_version: get_nvidia_driver_version().
  get_gpu_info: get_gpu_info().
  get_pretty_env_info: get_pretty_env_info().
  main: main().
  get_nvidia_smi: get_nvidia_smi().
  run: run().
  get_gcc_version: get_gcc_version().
  get_clang_version: get_clang_version().
  get_cmake_version: get_cmake_version().
  get_running_cuda_version: get_running_cuda_version().
  get_cudnn_version: get_cudnn_version().
  get_cpu_info: get_cpu_info().
  get_mac_version: get_mac_version().
  get_windows_version: get_windows_version().
  get_lsb_version: get_lsb_version().
  check_release_file: check_release_file().
  get_libc_version: get_libc_version().
  get_cuda_module_loading_config: get_cuda_module_loading_config().
  is_xnnpack_available: is_xnnpack_available().
  run_and_read_all: run_and_read_all().
  COMMON_PATTERNS: COMMON_PATTERNS.
  NVIDIA_PATTERNS: NVIDIA_PATTERNS.
  get_env_info.get_version_or_na: get_env_info().get_version_or_na().
  pretty_str.prepend: pretty_str().prepend().
  pretty_str.replace_if_empty: pretty_str().replace_if_empty().
  SystemEnv: SystemEnv.
  CONDA_PATTERNS: CONDA_PATTERNS.
  PIP_PATTERNS: PIP_PATTERNS.
  get_python_platform: get_python_platform().
  get_cachingallocator_config: get_cachingallocator_config().
  env_info_fmt: env_info_fmt.
  pretty_str.replace_nones: pretty_str().replace_nones().
  pretty_str.replace_bools: pretty_str().replace_bools().
  pretty_str.maybe_start_on_next_line: pretty_str().maybe_start_on_next_line().
  run_and_return_first_line: run_and_return_first_line().
---
# Module: [`dlpack/collect_env.py`](../../../../../raw/code/learning-machine/dlpack/collect_env.py)

## Functions
- `check_release_file(run_lambda)` — [`L390`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L390) — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `get_cachingallocator_config()` — [`L460`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L460)
- `get_clang_version(run_lambda)` — [`L150`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L150)
- `get_cmake_version(run_lambda)` — [`L154`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L154)
- `get_conda_packages(run_lambda, patterns=None)` — [`L132`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L132) — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `get_cpu_info(run_lambda)` — [`L321`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L321) — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `get_cuda_module_loading_config()` — [`L467`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L467) — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `get_cudnn_version(run_lambda)` — [`L193`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L193) — Return a list of libcudnn.so; it's hard to tell which one is being used. — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `get_env_info()` — [`L483`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L483) — Collects environment information to aid in debugging. — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `get_gcc_version(run_lambda)` — [`L147`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L147)
- `get_gpu_info(run_lambda)` — [`L167`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L167) — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `get_libc_version()` — [`L430`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L430)
- `get_lsb_version(run_lambda)` — [`L386`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L386) — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `get_mac_version(run_lambda)` — [`L368`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L368) — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `get_nvidia_driver_version(run_lambda)` — [`L158`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L158) — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `get_nvidia_smi()` — [`L230`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L230)
- `get_os(run_lambda)` — [`L395`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L395) — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `get_pip_packages(run_lambda, patterns=None)` — [`L437`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L437) — Return `pip list` output. Note: will also find conda-installed pytorch and numpy packages. — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `get_platform()` — [`L355`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L355) — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `get_pretty_env_info()` — [`L663`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L663) — Returns a pretty string of environment information. — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `get_python_platform()` — [`L425`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L425)
- `get_running_cuda_version(run_lambda)` — [`L189`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L189)
- `get_version_or_na(cfg, prefix)` — [`L510`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L510)
- `get_windows_version(run_lambda)` — [`L372`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L372) — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `is_xnnpack_available()` — [`L476`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L476)
- `main()` — [`L678`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L678)
- `maybe_start_on_next_line(string)` — [`L614`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L614)
- `prepend(text, tag='[prepend]')` — [`L604`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L604)
- `pretty_str(envinfo)` — [`L588`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L588) — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `replace_bools(dct, true='Yes', false='No')` — [`L596`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L596)
- `replace_if_empty(text, replacement='No relevant packages')` — [`L609`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L609)
- `replace_nones(dct, replacement='Could not collect')` — [`L589`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L589)
- `run(command)` — [`L90`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L90) — Return (return-code, stdout, stderr). — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `run_and_parse_first_match(run_lambda, command, regex)` — [`L114`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L114) — Run command using run_lambda, returns the first regex match if it exists. — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `run_and_read_all(run_lambda, command)` — [`L106`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L106) — Run command using run_lambda; reads and returns entire output if rc is 0. — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `run_and_return_first_line(run_lambda, command)` — [`L124`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L124) — Run command using run_lambda and returns first line if output is not empty.

## Module values
- `COMMON_PATTERNS` — [`L51`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L51) — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `CONDA_PATTERNS` — [`L76`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L76) — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `NVIDIA_PATTERNS` — [`L58`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L58) — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `PIP_PATTERNS` — [`L83`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L83) — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `SystemEnv` — [`L23`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L23) — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `TORCH_AVAILABLE` — [`L18`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L18) — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)
- `env_info_fmt` — [`L555`](../../../../../raw/code/learning-machine/dlpack/collect_env.py#L555) — documented in [dlpack-collect_env](../../concepts/dlpack-collect_env.md)

