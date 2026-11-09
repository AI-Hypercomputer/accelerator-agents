# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""What chip and what software version the claims in `reference/` apply to.

Every other probe's answer is a fact about *this* row of output. A Pallas
symbol that exists on one JAX and not the next is the single most common way
for a knowledge base to go stale, so the version stamp is not preamble.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform

import harness

_GROUP = "environment"


@harness.probe(
  _GROUP,
  "software_versions",
  "Which jax, jaxlib and libtpu is this image on?",
)
def _software_versions() -> str:
  versions = {}
  for package in ("jax", "jaxlib", "libtpu", "torch", "torch_xla", "tokamax"):
    try:
      versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
      versions[package] = "absent"
  versions["python"] = platform.python_version()
  return " ".join(f"{name}={value}" for name, value in sorted(versions.items()))


@harness.probe(
  _GROUP,
  "device_inventory",
  "What devices does jax see, and what does it call this chip?",
)
def _device_inventory() -> str:
  import jax

  devices = jax.devices()
  kinds = sorted({device.device_kind for device in devices})
  first = devices[0]
  attributes = []
  for name in ("num_cores", "core_on_chip", "coords", "id", "process_index"):
    if hasattr(first, name):
      attributes.append(f"{name}={getattr(first, name)}")
  return (
    f"backend={jax.default_backend()} local_device_count={jax.local_device_count()} "
    f"device_count={jax.device_count()} kinds={kinds} "
    f"devices[0]: {' '.join(attributes)}"
  )


@harness.probe(
  _GROUP,
  "device_memory_stats",
  "How much HBM does one device report, and how much is already in use?",
)
def _device_memory_stats() -> str:
  import jax

  stats = jax.devices()[0].memory_stats()
  if stats is None:
    raise harness.Unsupported("device.memory_stats() returned None")
  interesting = (
    "bytes_limit",
    "bytes_in_use",
    "peak_bytes_in_use",
    "bytes_reservable_limit",
    "largest_free_block_bytes",
  )
  parts = []
  for key in interesting:
    if key in stats:
      parts.append(f"{key}={stats[key]} ({stats[key] / 2**30:.2f} GiB)")
  other = {k: v for k, v in stats.items() if k not in interesting}
  return f"{' '.join(parts)} | other_keys={sorted(other)}"


@harness.probe(
  _GROUP,
  "pallas_hello_world",
  "Does the simplest possible pallas_call compile and produce the right answer?",
)
def _pallas_hello_world() -> str:
  """The gate on every other probe: if this fails, nothing else means anything."""
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  def add_one_kernel(source_ref, output_ref) -> None:
    output_ref[...] = source_ref[...] + 1.0

  source = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
  result = pl.pallas_call(
    add_one_kernel,
    out_shape=jax.ShapeDtypeStruct(source.shape, source.dtype),
  )(source)

  expected = source + 1.0
  max_error = float(jnp.max(jnp.abs(result - expected)))
  if max_error != 0.0:
    return (
      harness.Verdict.FAIL,
      f"exact match expected, max_abs_error={max_error}",
    )
  return f"8x128 f32 add-one exact; result[0,:3]={result[0, :3].tolist()}"


@harness.probe(
  _GROUP,
  "pltpu_symbol_surface",
  "Which pltpu / pl symbols that the reference files name actually exist here?",
)
def _pltpu_symbol_surface() -> str:
  """A missing symbol is a version fact, and it is the cheapest one to check."""
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  wanted = {
    "pl": (
      pl,
      [
        "pallas_call",
        "BlockSpec",
        "GridSpec",
        "Slice",
        "ds",
        "dslice",
        "load",
        "store",
        "swap",
        "when",
        "program_id",
        "num_programs",
        "multiple_of",
        "atomic_add",
        "reciprocal",
        "Buffered",
        "CostEstimate",
        "run_scoped",
        "core_map",
        "debug_print",
        "BoundedSlice",
        "Indirect",
        "Element",
        "Squeezed",
        "Blocked",
        "no_block_spec",
      ],
    ),
    "pltpu": (
      pltpu,
      [
        "ANY",
        "VMEM",
        "SMEM",
        "CMEM",
        "HBM",
        "SemaphoreType",
        "PrefetchScalarGridSpec",
        "emit_pipeline",
        "make_async_copy",
        "make_async_remote_copy",
        "async_copy",
        "repeat",
        "roll",
        "bitcast",
        "TPUCompilerParams",
        "CompilerParams",
        "delay",
        "device_id",
        "get_barrier_semaphore",
        "semaphore_signal",
        "semaphore_wait",
        "BufferedRef",
        "with_memory_space_constraint",
        "sync_copy",
        "assume_multiple",
      ],
    ),
  }

  lines = []
  for namespace, (module, names) in wanted.items():
    present = [name for name in names if hasattr(module, name)]
    missing = [name for name in names if not hasattr(module, name)]
    lines.append(f"{namespace}.PRESENT={present}")
    lines.append(f"{namespace}.MISSING={missing}")
  return " || ".join(lines)


@harness.probe(
  _GROUP,
  "tpu_info_reported_constants",
  "What per-core hardware constants does pltpu.get_tpu_info() report for this chip?",
)
def _tpu_info_reported_constants() -> str:
  """JAX's own hardware table, read on the device it describes.

  These are *reported* constants, not measured ones — they are what Mosaic
  itself budgets against, which makes them the right thing for a kernel to
  plan with and the wrong thing to quote as an achieved rate. Every rate here
  should be paired with the measured fraction of it from `compute_units`.
  """
  from jax.experimental.pallas import tpu as pltpu

  if not hasattr(pltpu, "get_tpu_info"):
    raise harness.Unsupported("pltpu.get_tpu_info does not exist on this jax")

  info = pltpu.get_tpu_info()

  scalar_fields = (
    "chip_version",
    "generation",
    "num_cores",
    "num_lanes",
    "num_sublanes",
    "mxu_column_size",
    "num_mxus",
    "num_accumulators",
    "vmem_capacity_bytes",
    "cmem_capacity_bytes",
    "smem_capacity_bytes",
    "hbm_capacity_bytes",
    "mem_bw_bytes_per_second",
    "bf16_ops_per_second",
    "int8_ops_per_second",
    "fp8_ops_per_second",
    "int4_ops_per_second",
  )
  parts = []
  for name in scalar_fields:
    if not hasattr(info, name):
      parts.append(f"{name}=field-absent")
      continue
    value = getattr(info, name)
    if name.endswith("_capacity_bytes"):
      parts.append(f"{name}={value} ({value / 2**20:.2f} MiB)")
    elif name == "mem_bw_bytes_per_second":
      parts.append(f"{name}={value} ({value / 1e12:.3f} TB/s)")
    elif name.endswith("_ops_per_second"):
      parts.append(f"{name}={value} ({value / 1e12:.1f} Tops/s)")
    else:
      parts.append(f"{name}={value}")

  for name in ("is_lite", "is_split_chip", "is_megacore"):
    if hasattr(info, name):
      parts.append(f"{name}={getattr(info, name)}")

  sparse_core = getattr(info, "sparse_core", None)
  parts.append(f"sparse_core={sparse_core!r}")

  import jax.numpy as jnp

  if hasattr(info, "get_sublane_tiling"):
    tilings = {
      name: info.get_sublane_tiling(getattr(jnp, name))
      for name in ("float32", "bfloat16", "int8", "int4")
      if hasattr(jnp, name)
    }
    parts.append(f"get_sublane_tiling={tilings}")

  if hasattr(info, "is_matmul_supported"):
    native = []
    candidates = (
      "float32",
      "bfloat16",
      "int8",
      "int4",
      "float8_e4m3fn",
      "float8_e5m2",
    )
    for lhs_name in candidates:
      for rhs_name in candidates:
        lhs = getattr(jnp, lhs_name, None)
        rhs = getattr(jnp, rhs_name, None)
        if lhs is None or rhs is None:
          continue
        if info.is_matmul_supported(lhs, rhs):
          native.append(f"{lhs_name}x{rhs_name}")
    parts.append(f"natively_supported_matmul_dtypes={native}")

  return " ".join(parts)


@harness.probe(
  _GROUP,
  "pltpu_full_surface",
  "What is the complete public symbol list of jax.experimental.pallas.tpu here?",
)
def _pltpu_full_surface() -> str:
  """A named-symbol check goes stale; the full dump does not."""
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  return (
    f"dir(pltpu)={sorted(n for n in dir(pltpu) if not n.startswith('_'))} "
    f"|| dir(pl)={sorted(n for n in dir(pl) if not n.startswith('_'))}"
  )


@harness.probe(
  _GROUP,
  "compiler_params_spelling",
  "Is the compiler-params argument spelled TPUCompilerParams or CompilerParams here?",
)
def _compiler_params_spelling() -> str:
  """Two spellings coexisted across versions; guessing costs a compile."""
  import inspect

  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  findings = []
  for name in ("TPUCompilerParams", "CompilerParams"):
    cls = getattr(pltpu, name, None)
    if cls is None:
      findings.append(f"{name}=absent")
      continue
    try:
      fields = list(inspect.signature(cls).parameters)
    except (TypeError, ValueError):
      fields = ["<uninspectable>"]
    findings.append(f"{name}(fields={fields})")

  call_params = list(inspect.signature(pl.pallas_call).parameters)
  return f"{' | '.join(findings)} || pallas_call params={call_params}"
