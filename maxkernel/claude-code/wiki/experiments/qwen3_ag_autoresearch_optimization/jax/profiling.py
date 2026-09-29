"""JAX-profiler capture + GCS publish, wired for xprof-mcp.

xprof-mcp (and the xprof browser UI) read profiles from a GCS logdir, e.g. the
server is launched as:

    xprof --logdir=gs://<bucket>/autoresearch --port=8791

A profile becomes queryable when a *run directory* containing the
`plugins/profile/<timestamp>/<host>.xplane.pb` subtree appears under that
logdir:

    gs://<bucket>/autoresearch/<run-name>/plugins/profile/<ts>/<host>.xplane.pb
                              ^^^^^^^^^^^^  <- this is the <run-name> xprof
                              shows

`jax.profiler.start_trace(dir)` writes exactly that `plugins/profile/...`
subtree
under `dir`. So there are two equivalent ways to land a profile where xprof-mcp
can see it:

  (A) Direct  — point `start_trace` straight at the GCS run dir. The profiler
      streams the trace to GCS itself (works on Cloud TPU where the TSL GCS
      filesystem is available; this is what MaxText does).
  (B) Capture-then-upload — `start_trace` to a local dir, then `gsutil -m rsync`
      the local run dir into the GCS run dir after the run. Use this when the
      box can't stream straight to GCS.

This module supports both and prints the run name + the direct xprof URL so the
experiment page's `## Profile` section can link straight through.

Typical use in a training loop:

    prof = TraceController(
        local_dir=profile_dir,          # local OR gs:// path; None disables
        gcs_dir=profile_gcs_dir,        # gs://.../<run-name> to upload to
        (optional)
        start_step=profile_start_step,
        num_steps=profile_steps,
        xprof_url_base=xprof_url_base,
    )
    for i, batch in enumerate(loader):
        prof.maybe_start(i)
        ... run step i ...
        prof.maybe_stop(i)
    prof.finalize()
"""

from __future__ import annotations

import os
import subprocess

import jax


def _is_gcs(path: str | None) -> bool:
  return bool(path) and path.startswith("gs://")


def _run_name_of(path: str) -> str:
  """The xprof run name is the trailing component of the run dir."""
  return os.path.basename(path.rstrip("/"))


class TraceController:
  """Captures a contiguous range of training steps and publishes to GCS.

  `local_dir` is where `jax.profiler.start_trace` writes. It may itself be a
  `gs://` path (mode A, direct stream). `gcs_dir` is an optional `gs://` run
  directory to `rsync` a *local* capture into after the run (mode B). If both
  are given and `local_dir` is local, the capture goes local then uploads.
  """

  def __init__(
      self,
      *,
      local_dir: str | None,
      gcs_dir: str | None = None,
      start_step: int = 8,
      num_steps: int = 3,
      xprof_url_base: str = "http://localhost:8791",
  ):
    self.trace_dir = local_dir  # where start_trace writes (local or gs://)
    self.gcs_dir = gcs_dir  # optional upload destination (gs://)
    self.start_step = start_step
    self.num_steps = max(1, num_steps)
    self.last_step = start_step + self.num_steps - 1
    self.xprof_url_base = xprof_url_base.rstrip("/")
    self.enabled = bool(local_dir or gcs_dir)
    self._active = False
    self._captured = False

    # If the only destination given is a GCS dir, stream straight to it.
    if self.enabled and not self.trace_dir and _is_gcs(self.gcs_dir):
      self.trace_dir = self.gcs_dir

  # -- loop hooks -----------------------------------------------------------

  def maybe_start(self, step: int) -> None:
    if not self.enabled or self._active or self._captured:
      return
    if step != self.start_step:
      return
    if not _is_gcs(self.trace_dir):
      os.makedirs(self.trace_dir, exist_ok=True)
    dest = "GCS (direct)" if _is_gcs(self.trace_dir) else "local"
    print(
        f"[profile] start_trace -> {self.trace_dir} ({dest}); "
        f"capturing steps {self.start_step}..{self.last_step}",
        flush=True,
    )
    jax.profiler.start_trace(self.trace_dir)
    self._active = True

  def maybe_stop(self, step: int) -> None:
    if not self._active or step < self.last_step:
      return
    jax.profiler.stop_trace()
    self._active = False
    self._captured = True
    print(
        f"[profile] stop_trace (captured steps {self.start_step}.."
        f"{self.last_step})",
        flush=True,
    )

  # -- end of run -----------------------------------------------------------

  def finalize(self) -> None:
    """Stop if still active, upload local capture to GCS if requested, and

    print the xprof run name + direct URL.
    """
    if self._active:
      jax.profiler.stop_trace()
      self._active = False
      self._captured = True

    if not self._captured:
      if self.enabled:
        print(
            "[profile] WARNING: no steps captured — check "
            "profile_start_step vs train_steps.",
            flush=True,
        )
      return

    published_dir = self.trace_dir
    # Mode B: local capture + explicit GCS destination → rsync it up.
    if self.gcs_dir and not _is_gcs(self.trace_dir):
      self._rsync_to_gcs(self.trace_dir, self.gcs_dir)
      published_dir = self.gcs_dir

    run_name = _run_name_of(published_dir)
    print("\n---------------- profile ----------------", flush=True)
    print(f"profile dir   : {published_dir}", flush=True)
    if _is_gcs(published_dir):
      print(f"xprof run     : {run_name}", flush=True)
      print(
          f"xprof URL     : {self.xprof_url_base}/?run={run_name}", flush=True
      )
      print(
          "(visible in xprof-mcp once the run dir is under the server "
          "--logdir root)",
          flush=True,
      )
    else:
      print(
          "profile is LOCAL only — pass --profile_gcs_dir "
          "gs://<bucket>/autoresearch/<run-name> to publish for xprof-mcp.",
          flush=True,
      )
    print("-----------------------------------------", flush=True)

  # -- internals ------------------------------------------------------------

  @staticmethod
  def _rsync_to_gcs(local_dir: str, gcs_dir: str) -> None:
    cmd = ["gsutil", "-m", "rsync", "-r", local_dir, gcs_dir]
    print(f"[profile] uploading: {' '.join(cmd)}", flush=True)
    try:
      subprocess.run(cmd, check=True)
      print(f"[profile] upload complete -> {gcs_dir}", flush=True)
    except FileNotFoundError:
      print(
          "[profile] ERROR: gsutil not found on PATH; cannot upload. "
          f"Local capture remains at {local_dir}.",
          flush=True,
      )
    except subprocess.CalledProcessError as e:
      print(
          f"[profile] ERROR: gsutil rsync failed ({e}); local capture "
          f"remains at {local_dir}.",
          flush=True,
      )
