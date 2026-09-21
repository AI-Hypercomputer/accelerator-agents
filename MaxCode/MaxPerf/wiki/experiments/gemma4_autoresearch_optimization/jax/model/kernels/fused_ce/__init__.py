# pylint: skip-file
"""Vendored import shim for levanter's fused_cross_entropy_loss Pallas TPU kernel.

Exposes ``fused_cross_entropy_loss_and_logsumexp_penalty`` — the only public
TPU Pallas CE kernel with native ``logit_soft_cap`` support (Gemma 4's
``final_logit_softcapping=30.0`` is applied inline on each VMEM logits tile
before the streaming softmax, so ``[B, S, V]`` logits never materialize in
HBM).

The kernel lives under the marin submodule at::

    raw/code/marin/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/

Rather than copying ~4.3k lines of kernel + autotune helper code into this
tree, we pre-populate ``sys.modules`` with minimal stubs for levanter's
top-level package (which otherwise triggers ``equinox`` / ``draccus`` imports)
and for the one ``rigging.filesystem`` call site inside
``autotune_cache_utils``. Autotune runtime is never triggered on the hot
path because we pass ``implementation="pallas_tpu"`` with block sizes that
have a tuned match (see ``tuned_block_sizes.py``).
"""
from __future__ import annotations

from pathlib import Path
import sys
import types


_LEVANTER_SRC = (
    Path(__file__).resolve().parents[7]
    / "raw"
    / "code"
    / "marin"
    / "lib"
    / "levanter"
    / "src"
)


def _install_shims() -> None:
  # 1. Vendored levanter path.
  src = str(_LEVANTER_SRC)
  if src not in sys.path:
    sys.path.insert(0, src)

  # 2. Skip levanter/__init__.py's import of analysis/trainer/data/... by
  #    installing an empty package stub BEFORE first import. Submodules
  #    (kernels.pallas.*) still resolve via the file-system loader.
  if "levanter" not in sys.modules:
    pkg = types.ModuleType("levanter")
    pkg.__path__ = [str(_LEVANTER_SRC / "levanter")]
    pkg.__version__ = "shim"
    sys.modules["levanter"] = pkg

  # 3. The kernel's autotune_cache_utils.py does ``from rigging.filesystem
  #    import url_to_fs`` at module top. rigging isn't installed; provide
  #    a minimal stub. url_to_fs is only called from load_json/write_json
  #    which run during autotune-on-miss; we sidestep that path by using
  #    implementation="pallas_tpu" with a tuned-match block size.
  if "rigging" not in sys.modules:
    rigging = types.ModuleType("rigging")
    sys.modules["rigging"] = rigging
  if "rigging.filesystem" not in sys.modules:
    fs_mod = types.ModuleType("rigging.filesystem")

    def _unavailable(*_args, **_kwargs):
      raise RuntimeError(
          "rigging.filesystem.url_to_fs is not available in this wiki; "
          "autotune cache I/O is disabled. Set "
          "LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0 or pass a tuned-match "
          "block size to avoid this path."
      )

    fs_mod.url_to_fs = _unavailable
    sys.modules["rigging.filesystem"] = fs_mod
    sys.modules["rigging"].filesystem = fs_mod  # type: ignore[attr-defined]

  # 4. levanter.utils.fsspec_utils.join_path — pure-Python helper, but its
  #    module imports the wider levanter.utils package which pulls in heavy
  #    deps. Provide a minimal stub with just join_path.
  if "levanter.utils" not in sys.modules:
    utils_mod = types.ModuleType("levanter.utils")
    utils_mod.__path__ = [str(_LEVANTER_SRC / "levanter" / "utils")]
    sys.modules["levanter.utils"] = utils_mod
    sys.modules["levanter"].utils = utils_mod  # type: ignore[attr-defined]
  if "levanter.utils.fsspec_utils" not in sys.modules:
    fsspec_stub = types.ModuleType("levanter.utils.fsspec_utils")

    def _join_path(lhs: str, rhs: str) -> str:
      if not lhs:
        return rhs
      if lhs.endswith("/"):
        return lhs + rhs
      return lhs + "/" + rhs

    fsspec_stub.join_path = _join_path
    sys.modules["levanter.utils.fsspec_utils"] = fsspec_stub
    sys.modules["levanter.utils"].fsspec_utils = fsspec_stub  # type: ignore[attr-defined]


def load_kernel():
  """Install shims and import the kernel entry point.

  Returns:
      The ``fused_cross_entropy_loss_and_logsumexp_penalty`` callable.
  """
  _install_shims()
  from levanter.kernels.pallas.fused_cross_entropy_loss.api import (
      fused_cross_entropy_loss_and_logsumexp_penalty,
  )

  return fused_cross_entropy_loss_and_logsumexp_penalty
