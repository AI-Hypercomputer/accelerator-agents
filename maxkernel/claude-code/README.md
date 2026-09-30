# MaxKernel

An agent that writes and optimizes [Pallas](https://docs.jax.dev/en/latest/pallas/index.html)
TPU kernels, packaged for [Claude Code](https://claude.com/claude-code).

You point it at a kernel — JAX, CUDA or PyTorch — and a TPU; it runs a closed
self-refinement loop —
**plan → implement → compile → test → autotune → profile** — for five iterations,
keeping the fastest numerically-correct kernel it found. Every iteration is
grounded in real hardware: kernels compile on an actual TPU VM, correctness is
checked against the baseline, and the next plan is written from an XProf trace of
the last one rather than from a guess.

This repository contains everything except your personal Claude Code config.
`install.py` renders the prompts against your paths and installs them.

---

## Getting started

Read the **[User Guide](user_guide.md)** — a short walkthrough of how to use MaxKernel on Claude Code

---

## What's in here

| Path | What it is |
| --- | --- |
| `skill/SKILL.md` | The `maxkernel` skill — the orchestrator. It only reads state and dispatches the worker; it never writes kernel code itself. |
| `skill/general_rules.md` | The invariants every subagent loads before doing anything. |
| `agents/maxkernel-*.md` | 19 subagent prompts: one worker plus eighteen single-phase specialists. |
| `tools/` | The agent's CLI surface: TPU client, input classification, golden capture and port verification, CUDA fact extraction, the ideas ledger, test-harness assembly, autotune config selection, XProf trace analysis, HLO dumps, RAG retrieval. |
| `server/tpu_server.py` | The job-queue daemon that runs on (or beside) the TPU VM. |
| `hooks/` | Optional `PreToolUse` guard — see [Workspace guard](#workspace-guard-optional). |
| `reference/` | The TPU knowledge base the planner and implementer read by explicit path: `tpu_memory_overlapping.md`, `tpu_mxu_and_register_optimization.md`, `tpu_profiling_and_diagnostics.md`, `pallas_xla_interaction.md`. |
| `requirements.txt` | Everything `maxkernel_venv` needs — see [Install](#install). |
| `docs/tpu-setup.md` | Pointing MaxKernel at a TPU (local / remote / multi-TPU) and driving the TPU client. |
| `docs/torchax-oracle-pipeline.md` | The torch → JAX → jaxpr → Pallas path: what each gate checks, with measured numbers, and the torch/torchax version pin. |
| `evaluation/` | Standalone harness: times the user's real PyTorch against the generated kernel on a TPU, and checks that a kernel actually ran. |
| `install.py` | Renders and installs the skill, subagents and hooks into `~/.claude`. |

### How the loop is wired

```
you ──▶ maxkernel skill (orchestrator)
             │  step 1: tools/classify_inputs.py ─▶ primary + references
             │  step 2: reference_mode = torchax | llm_port
             │  then one Agent call per iteration; re-reads state.json between them
             ▼
        maxkernel-worker
             │
             │  ── torch ──▶ JAX (pure) ──▶ jaxpr ──▶ plan ──▶ Pallas ──
             │
             │ 0.2  torchax_oracle.py        MECHANICAL conversion + step-0 GATE
             │        extract_jax ─▶ jax_fn (pure JAX)   ◀── ORACLE, hidden
             │        step 0: jax_fn(x) vs eager model(x)
             │      ‖ llm_port mode: capture_torch_golden.py ─▶ golden values
             │
             │ 0.3  jaxpr_export.py ─▶ ref.jaxpr.txt   what was asked for
             │                      ─▶ ref.hlo.txt     what XLA actually did
             │
             │ 0.4  analyze-torch-source / analyze-source   (llm_port mode only)
             │ 0.5  cuda_static_facts.py + analyze-cuda-reference   (if a ref)
             │ 0.6  reconcile-reference ─▶ ideas_ledger.json, then ref/ SEALED
             │
             │ 0.7  write-jnp-reference ─▶ base.py     (torchax mode, optional)
             │        GATE jaxpr_compare.py --strict
             │             value     vs the hidden oracle
             │             structure vs the oracle's jaxpr
             │ 0.8  verify_port.py ─ GATE               (llm_port mode)
             │        base.py vs golden, on CPU
             │        fails ─▶ fix-port ×3 ─▶ STOP the run
             │
             ├──────────▶ plan-kernel ──▶ implement-kernel
             │              ▲   ▲              │   (no read access to ref/)
             │              │   └ roofline first, then jaxpr + HLO, then ledger
             │              └ reads the primary's context brief
             │            fix-kernel-compilation ─┘
             │            compilation-summary
             │            generate-test-file / fix-test-script
             │            test-script-validation-summary
             │            summarize-test-results
             │            autotune-planner / autotune-summary
             └──────────  generate-profile-script / summarize-profile
                              └ adjudicates each adopted ledger idea
                                against the trace: confirmed | refuted

        evaluation/compare_kernel.py   (separate, needs a TPU)
             └ times the user's REAL PyTorch against the generated kernel,
               via torch_xla + make_kernel_from_pallas. The internal loop
               measures against base.py; this measures against PyTorch, and
               the gap between the two is the diagnostic.
```

State lives in `workspace/<run_id>/state.json`, not in anyone's context. The
orchestrator branches on what it just read from disk, so a run survives a
subagent dying mid-iteration.

### What you can hand it: a primary, and optionally a reference

Inputs come in two kinds, and they carry different authority.

The **primary** is the thing being converted. It defines the semantics, the
baseline `base.py`, and every number the run reports. A **reference** is an
independent implementation of roughly the same computation — typically a CUDA
kernel someone already tuned — supplied so the planner can mine it for design
ideas. A reference is never executed, never measured, and never compared
against.

`tools/classify_inputs.py` sorts every file you hand over by deterministic
marker matching, not a model's opinion. When two files could each be the
primary, it refuses to guess and the orchestrator asks you — picking wrong
would silently benchmark the wrong computation for five iterations.

| primary language | What happens |
| --- | --- |
| `jax` | Your file becomes `base.py` unchanged. This is the original path. |
| `pytorch` | A `torch` module. It is executed on CPU to capture golden values, ported to a JAX `base.py`, and that port is then **verified against the golden values** before the loop starts. |
| `cuda` | `.cu`/`.cuh`, or CUDA carried in a Python file for `load_inline`. Ported to `base.py` via `cuda_context.md`. No golden capture — a TPU host cannot execute CUDA. |

A KernelBench-style file (a torch `Model` plus a `cuda_sources` string) fills
both slots at once: the torch module is the primary, the embedded CUDA is a
reference.

**Verifying the port.** For a PyTorch primary, `capture_torch_golden.py` runs
the user's own module on the CPU and records the inputs, the parameters and
the outputs; `verify_port.py` then checks `base.py` against them. Without this,
a non-JAX run has no way to detect a mistranslated port — harness validation
binds `base.py` as both sides of its own comparison, so it proves the harness
runs and nothing about the port. If the check fails three times, the run
**stops**: five iterations optimizing against a wrong denominator is worse
than no run, because it ends in a confident, wrong number.

**Borrowing from a reference.** `maxkernel-reconcile-reference` first asks
whether the reference actually computes what the primary computes, and records
a verdict — `aligned`, `partial`, `divergent` or `rejected`. Then it distils
the reference into `ideas_ledger.json`, one entry per borrowable idea, each
triaged into one of three classes:

- **ALGORITHMIC** — survives the hardware change. An online softmax recurrence,
  the fusion boundary the author chose, which operand stays resident. These
  are statements about the problem, not about the GPU, and they are the reason
  to read a CUDA kernel at all.
- **STRUCTURAL** — a real decision whose value is hardware-specific. `BLOCK_M=128`
  is evidence about the working set; it says nothing about what fits in 16 MB
  of VMEM. The planner re-derives its own tiles and may cite the reference only
  as corroboration.
- **NON_PORTABLE** — warp shuffles, `__syncthreads`, bank-conflict padding.
  Recorded explicitly so a plan can show the mechanism was considered and
  discarded — a stronger defence against transliteration than silence.

The planner must write its roofline analysis *before* it opens the ledger, and
tags every hypothesis with its provenance (`LEDGER-003`, `ROOFLINE`,
`PROFILE-iter2`). After each iteration the profile summarizer checks each
adopted idea's falsifiable claim against the XProf trace and marks it
`confirmed` or `refuted`, so the final report can say which borrowed ideas
actually produced speedups instead of asserting that the reference helped.

Once reconciliation is done, `ref/` is sealed and the guard hook denies further
reads: the implementer works from the plan, never from the raw CUDA.

One thing to keep straight when reading the results: a TPU cannot run CUDA, so
the baseline every speedup is measured against is the **JAX port**, not the
original kernel on a GPU.

---

## Install

Setup is a one-time job you do **before** starting Claude Code. The agent loop
itself never installs anything — it checks that the environment is there and
stops if it is not.

### 1. Prerequisites

- A TPU you can reach — either the machine you're on *is* a TPU VM, or you have
  `gcloud` SSH access to one
- Claude Code
- Python 3.12

### 2. Clone

```bash
git clone https://github.com/AI-Hypercomputer/accelerator-agents.git
cd ~/maxkernel/claude-code
```

**This checkout is the project root.** There is no second directory to create:
`install.py` resolves the root to wherever it lives, renders it into every
prompt as `{{MAXKERNEL_ROOT}}`, and creates `workspace/` inside it for run
directories. Clone it somewhere else and everything follows — the name
`maxkernel` above is only a suggestion.

### 3. Create `maxkernel_venv` from `requirements.txt`

The agents run every tool through `~/maxkernel_venv/bin/python`, so every
requirement goes into that venv:

```bash
python3.12 -m venv ~/maxkernel_venv
~/maxkernel_venv/bin/pip install --upgrade pip setuptools wheel
~/maxkernel_venv/bin/pip install -r requirements.txt
```

Check it:

```bash
~/maxkernel_venv/bin/python -c "import jax; print(jax.__version__, jax.devices())"
# 0.11.0 [TpuDevice(id=0, ...), ...]      (on a TPU VM)
```

This pins `jax==0.11.0` with `libtpu==0.0.47`. In remote mode the TPU VM builds its own venv at
`~/maxkernel_venv` on first contact, so this is only the machine you run Claude
Code on. When `requirements.txt` changes after a `git pull`, re-run the
`pip install -r` line.

> **Optional — smaller install.** The default `torch==2.9.0` wheel pulls ~3.5 GB
> of CUDA libraries a TPU VM never uses. To skip them, install the CPU wheel
> *before* `requirements.txt`; its `torch==2.9.0` line is then already satisfied:
>
> ```bash
> ~/maxkernel_venv/bin/pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
> ~/maxkernel_venv/bin/pip install -r requirements.txt
> ```

### 4. Install into Claude Code

```bash
python3 install.py --with-guard
```

`install.py` renders every `{{PLACEHOLDER}}` in `agents/` and `skill/` against
your actual paths and writes:

```
~/.claude/agents/maxkernel-*.md          14 subagents
~/.claude/skills/maxkernel/SKILL.md      the orchestrator skill
~/.claude/skills/maxkernel/general_rules.md
```

Nothing else in your config is touched. Useful flags:

| Flag | Effect |
| --- | --- |
| `--dry-run` | Print every change without making it. |
| `--venv PATH` | Interpreter the agents run tools with (default `~/maxkernel_venv`). |
| `--claude-dir PATH` | Config directory (default `$CLAUDE_CONFIG_DIR`, else `~/.claude`). |
| `--with-guard` | Also install the workspace guard — see below. |
| `--uninstall` | Remove everything the installer wrote. |

Re-run it after every `git pull`, or after editing a template.

### 5. Point it at a TPU

`install.py` seeds `tpu_config.json` from `tpu_config.example.json`, which
assumes you are running **on** a TPU VM. For a remote TPU, register it through the CLI rather than editing the
file: it takes a lock, de-duplicates, assigns a local port, and caches the
hardware spec.

```bash
~/maxkernel_venv/bin/python tools/tpu_client.py \
  --add_tpu '{"tpu_name": "<name>", "zone": "<zone>", "project": "<gcp-project>"}'
```

Add several to run against a pool. `docs/tpu-setup.md` Part 1 covers all three
modes (local, remote, multi-TPU); Part 2 covers the job queue and the rest of
the `tpu_client.py` CLI.

### 6. Run it

Start Claude Code from the project root:

```bash
cd ~/maxkernel && claude
```

Then ask for a kernel:

> Optimize this attention forward pass into a Pallas TPU kernel: `path/to/model.py`

The skill also triggers on any request to write, autotune, or profile a
Pallas/JAX TPU kernel, and you can invoke it explicitly with `/maxkernel`.

Results land in `workspace/<run_id>/`: a plan, an implementation, a profile
summary and an autotune summary per iteration, with the winner at
`state.best_code_path`.

---

## Workspace guard (optional)

`hooks/workspace-guard.py` is a `PreToolUse` hook that does two things:

1. Confines the agent's filesystem access to the project root — the directory
   you cloned into, whatever you named it.
2. Blocks every route to Pallas/Mosaic **reference implementations** —
   `jax.experimental.pallas` source, `jax/_src/pallas`, `pallas/ops`, and web
   lookups of the same.

The second is the interesting one. Without it, an agent asked for a flash
attention kernel will find and adapt the reference implementation, and you learn
nothing about whether the loop can actually derive a kernel. With it, kernels get
written from first principles.

It deliberately does *not* block the word "pallas" in your own files or in
`from jax.experimental import pallas as pl` imports — only reads of the reference
sources.

Three directories under the config dir stay reachable so the setup remains
maintainable in place — `skills/`, `agents/` and `hooks/`, i.e. the skill, the
subagent prompts and the guard itself. Everything else under it stays sealed:
transcripts, tool-result dumps, sessions, credentials, `settings.json`. Note
this is read *and* write: an agent that can edit `hooks/workspace-guard.py` can
weaken its own guard, so it is a convenience boundary there, not a security one.
Drop the entries from `ALLOWED_EXCEPTIONS` if you would rather it were sealed.

```bash
python3 install.py --with-guard
```

This writes the hook to `~/.claude/hooks/`, backs up `settings.json`, and adds
the `PreToolUse` entry plus matching `permissions.deny` rules. It is strict: the
config directory is sealed, and `/tmp` is not reachable.

> **Start Claude Code from the project root**, or the guard refuses *every*
> shell command — `echo hello` included:
>
> ```bash
> cd ~/maxkernel && claude
> ```
>
> A shell resolves a bare relative path like `cat ../secrets` against its working
> directory, and the hook never sees that directory change. Requiring it to sit
> inside the workspace is what makes relative paths safe without parsing shell
> syntax. The block message tells you this, but it is easier to read here than to
> discover after your first command fails.

To disable it for one session without uninstalling:

```bash
MAXKERNEL_GUARD_OFF=1 claude
```

The guard has a test suite — run it with the guard disabled, since the fixtures
contain the very paths it blocks:

```bash
MAXKERNEL_GUARD_OFF=1 python3 ~/.claude/hooks/guard-tests.py
```

### Manual setup, and the OS-level sandbox

`restriction.json` is the same configuration as a standalone file, plus a
`sandbox` block that `install.py` does **not** write. Use it if you would rather
edit `settings.json` yourself, or if you want the sandbox.

Merge its contents into `~/.claude/settings.json` (keep any keys you already
have) and replace three placeholders:

| Placeholder | Replace with | Find it with |
|---|---|---|
| `{{MAXKERNEL_ROOT}}` | your clone of this repo | `pwd` |
| `{{VENV_NAME}}` | your virtualenv directory name | — |
| `{{PYTHON_VERSION}}` | e.g. `python3.12` | `ls <venv>/lib/` |

Getting `{{PYTHON_VERSION}}` wrong is the failure worth guarding against: the
path simply matches nothing, and the reference kernels stay readable with no
error to tell you.

### No web access

The same file closes the two ways an agent could read the open internet:

- `permissions.deny` lists `WebSearch` and `WebFetch` as bare tool names, which
  denies every call to them. This is the only lever that reaches those two —
  they run in-process, so the sandbox's network rules never see them.
- `sandbox.network` pins shell egress to an allowlist, with `strictAllowlist`
  making a non-matching host a hard deny rather than a prompt. Anything a
  command could reach out with — `curl`, `wget`, `urllib`, a stray `pip
  install` — is confined to it.

The allowlist is not empty, because MaxKernel needs egress to work:
`*.googleapis.com` covers `gcloud` reaching the remote TPU and
`tools/retrieval.py` querying the Vertex AI RAG corpus, `accounts.google.com`
covers the auth redirect, and `allowLocalBinding` keeps the `127.0.0.1:8000`
SSH tunnel to `server/tpu_server.py` open. No search engine or documentation
host is on it. Trim it further if you run against a local TPU only — dropping
both domains leaves the tunnel working and nothing else.

`strictAllowlist` is read from user, managed or `--settings` sources only; a
project-level `.claude/settings.json` cannot set it. Merging `restriction.json`
into `~/.claude/settings.json`, as above, is a source that counts.

One gap this does not cover: an MCP server that fetches on the model's behalf
is neither a web tool nor a shell command. Name it in `deniedMcpServers`, or
list only the servers you want in `allowedMcpServers`, if you run any.

**Why the sandbox matters.** The hook inspects tool inputs as *text*. That is a
strong guardrail, but a determined agent can obfuscate a shell command, and bare
relative paths are resolved by the shell against a working directory the hook
never sees change. `sandbox.filesystem.denyRead` is enforced by the kernel
instead — a `cat` inside a sandboxed shell simply cannot open the file.

It denies the whole home directory and re-allows only the workspace, the
virtualenv and the shell rc files. Note the venv must be allowed wholesale: the
sandbox constrains the *process*, so an unreadable `site-packages` means Python
cannot import JAX at all. Only `pallas/ops` — the reference kernels, never
imported in normal use — is denied inside it.

Requires bubblewrap:

```bash
sudo apt install bubblewrap
```

`autoAllowBashIfSandboxed` then gives you prompt-free operation without
`--dangerously-skip-permissions`: the sandbox provides the guarantee that would
otherwise need your judgement on each prompt. `allowUnsandboxedCommands: false`
stops a tool call from opting out of the sandbox per-command.

Verify it is actually on — a config that silently protects nothing looks exactly
like one that works:

```bash
cat ~/.bash_history                                   # should fail
ls <venv>/lib/<python>/site-packages/jax/experimental/pallas/ops   # should fail
<venv>/bin/python -c "from jax.experimental import pallas as pl"   # should succeed
```

The third matters most: if it fails, the sandbox has broken imports and the
workspace is unusable.

---

## Editing the prompts

`agents/` and `skill/` are the source of truth. Edit the template, then re-run
`python3 install.py`. Placeholders
available in any template:

| Token | Resolves to |
| --- | --- |
| `{{MAXKERNEL_ROOT}}` | this checkout |
| `{{CLAUDE_DIR}}` | the Claude Code config directory |
| `{{VENV}}` / `{{VENV_PYTHON}}` / `{{VENV_NAME}}` | the agents' virtualenv |
| `{{HOME}}` | `$HOME` |

An unknown `{{TOKEN}}` is a hard error, so a typo fails the install instead of
shipping a broken prompt.

Two invariants worth preserving if you rewrite prompts:

- **The orchestrator never writes code.** It reads `state.json`, dispatches
  `maxkernel-worker`, and re-reads `state.json`. That is what keeps a five
  iteration run inside one context window.
- **Verify on disk, not on the report.** After every subagent returns, the caller
  checks that the phase's artifact exists and is non-empty. A subagent's prose
  summary is not evidence.

---

## Notes

- `tools/retrieval.py` queries the local 3-tiered LLMWiki (`wiki/`) for TPU
  architecture specifications, hardware envelopes, and Pallas patterns.
  It supports two modes:
  - **`suppressed` (Default)**: Benchmark evaluation mode. Excludes reference
    implementation directories (`tokamax/`, `classes/`, `briefs/`, `distilled/`)
    so the agent is forced to design kernels from first principles.
  - **`full`**: Unrestricted mode for development and production. If you want to
    run in full mode, simply export:
    ```bash
    export WIKI_MODE=full
    ```
    (or pass `--mode full` via CLI).
  - Fast retrieval uses `ripgrep` (`rg`) as Tier 1. If `ripgrep` is not
    installed, it automatically falls back to Tier 2 (pure-Python lexical search)
    with zero extra configuration.
- The remote TPU VM bootstraps its own venv at `~/maxkernel_venv` on that
  machine. That path is fixed by `tools/tpu_client.py` and is unrelated to
  `--venv` on the host.
- `workspace/` and `tpu_config.json` are gitignored — runs are large and the TPU
  inventory is per-machine.
