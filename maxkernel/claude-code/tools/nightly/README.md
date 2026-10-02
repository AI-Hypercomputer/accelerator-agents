# Nightly MaxKernel sweeps

Runs each problem in its **own headless Claude Code conversation**
(`claude -p`), unattended.

```
run_one.sh <problem>              one problem under problems/ -> one run_dir
run_batch.sh [problems]           pick the N least-recently-run, run them
run_jaxbench_level2.sh [ops]      the jaxbench_level2/ ops, on the GKE cluster
gke_tpu_forward.sh {ensure|...}   bind 127.0.0.1:8000 to the cluster's TPUs
jaxbench_status.py [--watch N]    render JAXBENCH_LEVEL2_STATUS.md
```

> `run_one.sh`, `run_batch.sh`, `status.py` and `status_watch.sh` are the older
> `problems/` rotation, carried in untracked from `pytorch_cuda_latest`. They
> are not part of this branch and the `job.json` branch inside `run_one.sh` is
> dead here — this branch has no job-file mechanism at all.

Everything the sweeps write lives under `_nightly/` at the repo root
(`workspace/_nightly` is a symlink to it, so the older scripts' paths still
resolve to the same place):

| path | what |
|---|---|
| `results.csv` | one row per finished problem (speedup, iterations, exit code) |
| `last_run.tsv` | rotation state: problem -> epoch of last attempt |
| `logs/<date>/<problem>.<run_id>.log` | full stream-json transcript |
| `cron.log` | the sweep's own stdout |
| `.lock` | flock; stops two sweeps from overlapping |
| `JAXBENCH_LEVEL2_STATUS.md` | live status of the jaxbench_level2 sweep |
| `jaxbench_level2_results.csv` | one row per finished jaxbench op |
| `jaxbench_level2_runs/<op>.json` | per-op marker: run_id, run_dir, pid, status |
| `logs/jaxbench_level2/<date>/<op>.<run_id>.log` | jaxbench transcripts |
| `gke_forward.8000.log` | port-forward supervisor log |

## Try it by hand first

```bash
cd /home/cathygao_google_com/maxkernel
MK_TIMEOUT=2h tools/nightly/run_one.sh 63p_GDN       # one problem, foreground
tools/nightly/run_batch.sh 63p_GDN rpav3             # two, MK_JOBS at a time
```

## Install the recurring job

```bash
crontab -e
```

Add (the `PATH` line matters — cron's default PATH does not include
`~/.local/bin`, where `claude` lives):

```cron
PATH=/home/cathygao_google_com/.local/bin:/usr/local/bin:/usr/bin:/bin
MK=/home/cathygao_google_com/maxkernel

# 20:00 every day; the flock means a sweep still running from yesterday
# simply skips today's trigger.
0 20 * * * cd $MK && MK_BATCH=4 MK_JOBS=2 $MK/tools/nightly/run_batch.sh >> $MK/workspace/_nightly/cron.log 2>&1
```

Every 48h instead: `0 20 */2 * *`.

Verify with `crontab -l`, and after the first fire:

```bash
tail -f  /home/cathygao_google_com/maxkernel/workspace/_nightly/cron.log
column -s, -t /home/cathygao_google_com/maxkernel/workspace/_nightly/results.csv
```

## Sizing

A 5-iteration run has historically taken **6-24h of wall clock**, so a single
night cannot cover all 14 problems. That is why `run_batch.sh` rotates: each
trigger takes the `MK_BATCH` problems that went longest without an attempt, so
the full set cycles over roughly a week.

`MK_JOBS` is capped low on purpose. There is one local TPU v6e and one job
queue (`server/tpu_server.py`), so concurrent runs serialize their TPU work,
and a job that waits more than `MAX_QUEUE_WAIT_TIME` (2h) in that queue is
auto-cancelled. Most wall clock is model time rather than TPU time, so 2-3
concurrent runs is a real speedup; 8 would mostly manufacture queue timeouts.

## The jaxbench_level2 sweep

Separate from the `problems/` rotation above, and separate on purpose.

```bash
cd /home/cathygao_google_com/maxkernel

# one op, foreground, to sanity-check the wiring
MK_TIMEOUT=2h tools/nightly/run_jaxbench_level2.sh --one 65p_Quantized_Splash_Attention

# all 15, detached -- the sweep dies with the shell otherwise
setsid env MK_JOBS=4 MK_TIMEOUT=10h MK_DEADLINE=120h \
  tools/nightly/run_jaxbench_level2.sh \
  >> _nightly/jaxbench_level2.log 2>&1 < /dev/null &

# confirm it reparented, then watch
ps -o pid,ppid,sid -p $(pgrep -d, -f run_jaxbench_level2.sh)   # PPID must be 1
watch -n30 cat _nightly/JAXBENCH_LEVEL2_STATUS.md
```

What makes it different from `run_one.sh`:

| | `run_one.sh` | `run_jaxbench_level2.sh` |
|---|---|---|
| source | guessed from `problems/<p>/` | always `jaxbench_level2/<op>/reference.py` |
| inputs | agent authors `get_inputs()` | pre-seeded from `kernel_task.yaml`, **all** configs |
| TPU | whatever `tpu_config.json` names | always the GKE cluster |
| results | `results.csv` | `jaxbench_level2_results.csv` |
| status | `STATUS.md` | `JAXBENCH_LEVEL2_STATUS.md`, refreshed every 60s |

### The TPU is the GKE cluster

`gke_tpu_forward.sh` keeps `127.0.0.1:8000` port-forwarded to
`svc/maxkernel-tpu-service` in `maxkernel-v6e-cluster` (us-east5-b,
`tpu-prod-env-multipod`), where four `maxkernel-tpu-server` pods pull from a
central queue.

No new `tpu_client.py` mode was needed: `start_server_for_tpu()` returns
immediately once `127.0.0.1:<port>/health` reports healthy, and the cluster's
openresty gateway reports exactly that. So the forward alone redirects every
submit/profile/autotune to cluster hardware, and no local server is booted.

Each op gets a `<run_dir>/tpu_config.json` naming that endpoint;
`get_tpu_config()` prefers it over the repo-level file whenever `--run_dir` is
passed, so the local TPU's config is never edited.

**A cluster job sees one v6e chip, not eight.** The server pods set
`TPU_VISIBLE_CHIPS=2` and `TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1`, so
`jax.device_count()` is 1 inside a job even though each pod holds a 4-chip
`v6e-4` host. That is why the per-run config says `device_count: 1` while the
repo-level `tpu_config.json` says 8 — numbers from the two backends are not
comparable, and grids must be sized for a single device.

`MK_JOBS` defaults to 4 here (one per server pod) rather than the local
sweep's 2, since the four pods consume the queue in parallel.

### All configs, always

Each `kernel_task.yaml` carries an `input_gen_code` whose `get_inputs()`
returns a list of `(dynamic_args, static_args)` — one entry per config, 2 to 8
depending on the op. `tools/test_harness_template.py` already loops over that
list, checking correctness per config and aggregating the timings.

The runner writes that snippet verbatim to `<run_dir>/get_inputs.py` before
launching, and the prompt forbids rewriting it. That is what stops an op from
quietly being optimized against one easy shape.

### Cluster image gaps

The image ships jax 0.11.0, numpy 2.5.2 and pydantic, but not `jaxtyping`,
which `64p_Linear_Softmax_Cross_Entropy` imports — that op died in
`exec(base.py)` with `ModuleNotFoundError` before compiling anything.
`gke_tpu_forward.sh deps` installs the gap into all four server pods and the
runner calls it per op, because a pod that restarts comes back without it.
Add packages with `MK_GKE_EXTRA_PIP="jaxtyping foo"`.

Five of the fifteen also tripped a harness check: `52p`, `56p`, `57p`, `59p`
and `64p` bind their entry point as `computation = workload` rather than
`def computation`, and `assemble_test_harness.py` matched only the literal
`def computation`. The runtime binding (`_base_ns['computation']`) always
handled both, so the guard now accepts either form.

### The cancel hazard is sharper here

`MK_JOBS` defaults to 4, but note that on the local sweep each concurrent run
talks to a *different* port, whereas here all four share port 8000 — the one
gateway. The skill's Emergency Stop calls `tpu_client.py --cancel_job` with no
job id, which cancels every active job on the port, so on GKE one op giving up
can cancel all three siblings' in-flight jobs rather than just its own. The
loop retries the lost iteration, so it costs time rather than correctness; drop
to `MK_JOBS=1` if you want that risk gone entirely.

### Read the speedups carefully

These baselines are **already Pallas kernels**, not idiomatic JAX, so numbers
here are not comparable to the KernelBench sweep's 1.0-7.9x. A result near
1.00x is honest.

And because the source is already JAX/Pallas, `base.py` is just a copy of it:
there is no golden capture and no port verification to disagree with. The
configs in `kernel_task.yaml` are the **only** correctness gate, which is the
other reason the runner pins `get_inputs.py` rather than letting the agent
write it.

### No job files, no iteration budget

This branch has no job-file mechanism — `validate_job.py`, `docs/job-spec.md`
and `examples/jobs/` do not exist here, and nothing reads a `job.json`. So the
runner puts what a job file would have declared straight into the prompt: the
entry point (`computation`, which five ops bind by assignment), the source
role, and the `atol`/`rtol` read out of `kernel_task.yaml`.

`num_trials` and `max_iterations` are deliberately **not** set. The loop ends
on the measured criteria in `general_stopping.md` — `state.stopping
.recommend_stop` for the iteration that just finished — and the prompt says so
explicitly, in both directions: do not run to a fixed count, and do not stop
early while the criteria still say continue.

## Known hazard with MK_JOBS > 1

The skill's Emergency Stop runs `tpu_client.py --cancel_job` with no job id,
and `cancel_jobs()` with `job_id=None` cancels **every active job on the
port**. One run giving up therefore kills its siblings' in-flight TPU jobs.
Until that call passes a specific job id, `MK_JOBS=1` is the only strictly
safe setting; at 2-3 you trade a small chance of a spurious failed iteration
(which the loop retries) for much better throughput.
