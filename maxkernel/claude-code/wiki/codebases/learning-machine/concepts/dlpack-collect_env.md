---
title: dlpack/collect_env — vendored PyTorch environment-diagnostics script
type: concept
provenance: mixed
concept: dlpack-collect_env
updated: 2026-07-03
status: fresh
---
# dlpack/collect_env — vendored PyTorch environment-diagnostics script
A near-verbatim copy of PyTorch's own `torch.utils.collect_env` bug-report utility — collects torch/CUDA/ROCm/OS/compiler/pip/conda version information into one pretty-printed report. It carries no TPU-performance-relevant logic itself; it sits in the `dlpack/` directory alongside `dlpack_repro.py` as an environment-capture aid for reproducing whatever DLPack interop issue that sibling script investigates.

## Entry points
- [`get_pretty_env_info`](../catalog/dlpack/collect_env.md#get_pretty_env_info) — the public one-call API (`"""...used in python collect_env.py that should be executed when reporting a bug."""`, quoting the function's own docstring); calls [`get_env_info`](../catalog/dlpack/collect_env.md#get_env_info) then [`pretty_str`](../catalog/dlpack/collect_env.md#pretty_str).
- The script's `if __name__ == '__main__'` block (calling [`get_pretty_env_info`](../catalog/dlpack/collect_env.md#get_pretty_env_info) and printing the result, then checking for PyTorch minidumps) is the direct-execution entry point: `python collect_env.py`.

## Mechanism (step-by-step)
1. [`get_env_info`](../catalog/dlpack/collect_env.md#get_env_info) (its own docstring: "Collects environment information to aid in debugging... Returns: SystemEnv (namedtuple)") gathers every field of the [`SystemEnv`](../catalog/dlpack/collect_env.md#SystemEnv) namedtuple by calling a battery of small probes — [`get_gpu_info`](../catalog/dlpack/collect_env.md#get_gpu_info), [`get_nvidia_driver_version`](../catalog/dlpack/collect_env.md#get_nvidia_driver_version), [`get_cudnn_version`](../catalog/dlpack/collect_env.md#get_cudnn_version), [`get_cpu_info`](../catalog/dlpack/collect_env.md#get_cpu_info), [`get_conda_packages`](../catalog/dlpack/collect_env.md#get_conda_packages), [`get_pip_packages`](../catalog/dlpack/collect_env.md#get_pip_packages), [`get_os`](../catalog/dlpack/collect_env.md#get_os) — each guarded by [`TORCH_AVAILABLE`](../catalog/dlpack/collect_env.md#TORCH_AVAILABLE) where the probe needs an actual `torch` import.
2. Every probe that shells out (nvidia-smi, gcc, cmake, lscpu, etc.) goes through [`run`](../catalog/dlpack/collect_env.md#run) — a thin `subprocess.Popen` wrapper — via helpers [`run_and_read_all`](../catalog/dlpack/collect_env.md#run_and_read_all) (whole stdout if exit code 0) or [`run_and_parse_first_match`](../catalog/dlpack/collect_env.md#run_and_parse_first_match) (first regex group) — a uniform "run, check rc, extract" pattern reused across every OS/toolchain probe.
3. [`get_os`](../catalog/dlpack/collect_env.md#get_os) and several probes branch on [`get_platform`](../catalog/dlpack/collect_env.md#get_platform) (`linux`/`win32`/`cygwin`/`darwin`), calling platform-specific helpers ([`get_windows_version`](../catalog/dlpack/collect_env.md#get_windows_version), [`get_mac_version`](../catalog/dlpack/collect_env.md#get_mac_version), [`get_lsb_version`](../catalog/dlpack/collect_env.md#get_lsb_version)/[`check_release_file`](../catalog/dlpack/collect_env.md#check_release_file) for Linux distro detection).
4. [`pretty_str`](../catalog/dlpack/collect_env.md#pretty_str) post-processes the raw `SystemEnv` into display form — replacing `None`/`True`/`False` with human strings, tagging pip/conda package lists with `[pip]`/`[conda]` prefixes, and special-casing "no CUDA" when `torch.cuda.is_available()` is false and every dynamic CUDA field came back empty — before formatting into [`env_info_fmt`](../catalog/dlpack/collect_env.md#env_info_fmt), a fixed multi-line template string.

## Key data structures
- [`SystemEnv`](../catalog/dlpack/collect_env.md#SystemEnv) — the `namedtuple` schema for every collected field (torch version, CUDA/ROCm details, compiler versions, OS, pip/conda package lists, CPU info); it is the single contract between the collection phase ([`get_env_info`](../catalog/dlpack/collect_env.md#get_env_info)) and the formatting phase ([`pretty_str`](../catalog/dlpack/collect_env.md#pretty_str)).
- [`COMMON_PATTERNS`](../catalog/dlpack/collect_env.md#COMMON_PATTERNS) / [`NVIDIA_PATTERNS`](../catalog/dlpack/collect_env.md#NVIDIA_PATTERNS) / [`CONDA_PATTERNS`](../catalog/dlpack/collect_env.md#CONDA_PATTERNS) / [`PIP_PATTERNS`](../catalog/dlpack/collect_env.md#PIP_PATTERNS) — substring allowlists used to filter `pip list`/`conda list` output down to torch-ecosystem-relevant packages only.

## Dynamics (design intent)
> [!inferred] This file's header comment ("Unlike the rest of the PyTorch this file must be python2 compliant") and its byte-for-byte structural match to PyTorch's upstream `torch/utils/collect_env.py` indicate it was copied in wholesale (likely to support a DLPack interop bug report, given its location alongside `dlpack_repro.py`) rather than authored for this repo — it carries no repo-specific logic.

## Edge cases
- [`get_cuda_module_loading_config`](../catalog/dlpack/collect_env.md#get_cuda_module_loading_config) calls `torch.cuda.init()` as a side effect merely to read an environment variable back — a real (if minor) cost/side-effect for a "just report the environment" script.

## Open questions
> [!inferred] Nothing in this packet's cited subgraph ties this file to TPU execution at all — every probe is CUDA/ROCm/CPU-oriented; it is included in this repo purely as a diagnostic utility, not as evidence of any TPU-specific investigation.

## See also
- No sibling concept page in this repo covers `dlpack/dlpack_repro.py` (not part of this ingest's selected packets) — that file is the likely reason this diagnostics script was copied in.
