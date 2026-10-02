---
title: "XProf docs: Capturing profiles"
type: source
tags: [docs, profiler, profile-capture, xprof, tpu, gpu]
created: 2026-04-22
updated: 2026-04-22
---

Top-level XProf guide that enumerates the two mechanisms for capturing a profile — **programmatic** (code-instrumented `start_trace`/`stop_trace` or context manager) and **on-demand / manual** (XProf UI's "CAPTURE PROFILE" button talks to a gRPC profiler server started inside the ML workload). It also introduces **continuous profiling snapshots** (retroactive capture at an instant), explains how a single run can hold multiple sessions, recommends `cloud-diagnostics-xprof` for GCP, and lists common troubleshooting cases. This is the "start here" doc — framework specifics live in the JAX/PyTorch-XLA/TensorFlow sibling pages.

## Overview

Three capture modes, all writing `.xplane.pb` files under `<logdir>/plugins/profile/<session>/`:

1. **Programmatic capture.** Code is annotated with trace start/stop; useful for deterministic profiling of specific steps.
2. **On-demand / manual capture.** An in-process gRPC profiler server listens; the XProf UI (or a `trace` client call) triggers an N-ms capture on demand. Used when something goes wrong mid-run and you want a profile *now*.
3. **Continuous profiling snapshots.** Retroactively capture a profile ending *at* a chosen instant (contrast with on-demand, which captures forward from the trigger). Intended for long-running jobs where the problem is diagnosed post hoc.

Multiple captures in one run become separate **sessions** under the same run (date-stamped subdirectories).

## Key claims

- Both manual and continuous-snapshot modes **require the profiler server to be started inside the workload process** (e.g. `jax.profiler.start_server(9999)`). Without it, the XProf UI has nothing to connect to.
- On GCP, the recommended wrapper is `cloud-diagnostics-xprof` (stores profiles in GCS, hosts TensorBoard/XProf on GCE/GKE, creates shareable links, supports on-demand profiling on GKE/GCE).
- Local profiles captured on a workload VM are **ephemeral** — they are deleted when the researcher's run finishes. Use GCS for retention.
- For remote profiling, SSH local port forwarding on port **8791** (XProf UI default) is the documented pattern:
  ```
  ssh -L 8791:localhost:8791 <remote>
  gcloud compute ssh <machine-name> -- -L 8791:localhost:8791
  ```
- GPU profiling failure modes: missing `libcupti.so` (fix with `LD_LIBRARY_PATH`) and `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` (requires `NVreg_RestrictProfilingToAdminUsers=0` and reboot, or `docker run --privileged=true`).

## Key data points

| Mode | Trigger | Captures | When to use |
|---|---|---|---|
| Programmatic | Inline code (`start_trace`/`stop_trace`, context manager) | Fixed code region | Deterministic profiling of known hot path / steps |
| On-demand (manual) | XProf UI "CAPTURE PROFILE" button → gRPC | Forward N ms from trigger | Ad-hoc diagnosis mid-run |
| Continuous snapshot | `get_snapshot(...)` API | Ending *at* trigger time | Long-running jobs; profile the instant a problem is detected |

| Default port | Purpose |
|---|---|
| 8791 | XProf UI / TensorBoard profile plugin |
| 9999 | Common example port for in-process profiler gRPC server (JAX) |
| 50051 | XProf worker gRPC service (aggregator/worker mode) |

| Troubleshooting symptom | Root cause | Fix |
|---|---|---|
| `Could not load dynamic library 'libcupti.so.10.1'` | CUPTI library not on loader path | `export LD_LIBRARY_PATH=/usr/local/cuda-<ver>/extras/CUPTI/lib64/:$LD_LIBRARY_PATH` |
| `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` | Kernel restricts non-admin perf counters | `options nvidia "NVreg_RestrictProfilingToAdminUsers=0"` in `/etc/modprobe.d/nvidia-kernel-common.conf`, `update-initramfs -u`, reboot |
| CUPTI priv error in Docker | Same, inside container | `docker run --privileged=true` |
| `ValueError: Duplicate plugins for name projector` | Multiple TensorBoard/TF/xprof installs | Fully uninstall `tensorflow tf-nightly tensorboard tb-nightly xprof xprof-nightly tensorboard-plugin-profile tbp-nightly`, reinstall `tensorboard xprof` |
| Only host traces visible on GPU run | CUPTI not loaded | Check logs; fix `LD_LIBRARY_PATH` (but the warning may be spurious — confirm by inspecting trace viewer) |

## Techniques referenced

- **Programmatic capture** (`jax.profiler.start_trace` / `stop_trace` / `trace` context manager; PyTorch/XLA `xp.start_trace`; TF `tf.profiler.experimental.start` / context manager / Keras TensorBoard callback).
- **gRPC profiler server** inside the workload — required for on-demand and snapshot modes; implemented per framework (e.g. `jax.profiler.start_server(port)`).
- **Continuous profiling snapshots** via `xprof.api.continuous_profiling_snapshot.{start_continuous_profiling, get_snapshot, stop_continuous_profiling}`.
- **SSH port forwarding** for accessing XProf UI on a remote workload VM.
- **GCS-backed logdir** (via `cloud-diagnostics-xprof`) for profile persistence beyond the workload lifetime.

## Gaps & caveats

- Doc does not specify TPU-specific capture caveats (e.g. `LIBTPU_INIT_ARGS`, sparse-core tracing, multi-host coordination) — those live in framework-specific docs and XProf tool docs.
- CUPTI-related troubleshooting is GPU-only; none of it applies to TPU runs but it occupies the bulk of the troubleshooting section.
- Port `9999` for the in-process profiler server is an *example*, not a default; frameworks pick their own (PyTorch/XLA examples use `9012`; TF uses `6009`). Consistency is on the user.
- "Multiple sessions per run" is documented but no guidance on how to keep session labels meaningful — they are date-stamped, not named.
- `cloud-diagnostics-xprof` is recommended but not required; nothing in the core XProf CLI depends on it. Experiments in this wiki may ingest from either local `raw/profiles/` or GCS.
- Hide-capture-button mode (`--hide_capture_profile_button`) is mentioned in sibling deployment docs but not here — relevant because aggregator/worker deployments disable the UI button.

## Connections

- `profile-capture` — umbrella concept (programmatic vs on-demand vs continuous snapshot).
- `xprof-server` — the in-process gRPC server (`jax.profiler.start_server`, `xp.start_server`, `tf.profiler.experimental.server.start`).
- `continuous-profiling-snapshot` — retroactive capture at an instant.
- `xprof-sessions` — multi-session runs and session date-stamping.
- `cloud-diagnostics-xprof` — GCP wrapper for XProf/TensorBoard + GCS logdir.
- `cupti-privileges` — GPU-only; flagged for completeness.

## See also

- [xprof](../codebases/xprof.md)
- [XProf docs: JAX profiling](2026-xprof-jax-profiling.md)
- [XProf docs: PyTorch/XLA profiling](2026-xprof-pytorch-xla-profiling.md)
- [XProf docs: TensorFlow profiling](2026-xprof-tensorflow-profiling.md)
- [XProf docs: Docker deployment](2026-xprof-docker-deployment.md)
- [XProf docs: Kubernetes deployment](2026-xprof-kubernetes-deployment.md)

## Sources

- `raw/code/xprof/docs/capturing_profiles.md`
