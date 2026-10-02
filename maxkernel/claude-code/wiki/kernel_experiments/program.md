# Kernel-family optimization loop

**This file is the complete, self-contained protocol for kernel-family experiments.** No other
program.md needs to be read first. Resolution: this file → `<kernel>/pallas/program.md` (family
bindings only: reference impl, parity spec, chip, op-point, home_repo, integration_branch, archive_remote — thin, nothing else; for
each H2 section the family file defines, it wins outright).

## Quick start — you were given a kernel path + a chip (the production bare prompt)

A prompt like *"use the start experiment protocol for kernel `<path>` on tpu chip N"* means, in
order — do NOT improvise an alternative sequence:

1. **Derive the canonical family slug** — NEVER `basename(<path>)`. Rule: if a `wiki/kernels/*.md`
   family page already matches this kernel, its filename stem IS the slug; otherwise create the
   slug as the **hyphenated operation name** (lowercase, hyphens, no benchmark prefixes/suffixes —
   `1p_Flash_Attention` → `flash-attn-mha`, `10p_Sparse_MoE` → `sparse-moe`). Two agents given the
   same kernel MUST land on the same slug.
2. **Read this file end-to-end**, then the family binding (`wiki/kernel_experiments/<slug>/pallas/program.md`)
   if it exists — create it from the binding fields above if not (family bootstrap, including the
   family page `wiki/kernels/<slug>.md` and pinning the reference module).
3. **Load [`/author-kernel`](../../.claude/skills/author-kernel/SKILL.md)** before writing any candidate.
4. **Run the loop**: K0 → K9 below, on the pinned chip (`flock /tmp/tpu-chip-N.lock`), synchronously. (Supervision — ALL harnesses, one rule: **after EVERY K7 verdict commit, and as the final gate of any close, dispatch the `process-auditor` subagent and apply its findings before your next K3.** The auditor is the sole stop authorizer. Timer-based watchers, where armed, are an additional stall/revive net — never a substitute for the K7 dispatch.)
5. Verification is **`kgate verify`** (K6); numbers you didn't get from a kgate receipt do not exist.

Everything an orchestrator/launcher would otherwise do for you (worktree, stub, plan, receipts,
page filing, commits) is YOURS to do in solo mode — the steps below say exactly when.

## Invariants

- **Roles — author / verifier / master. Whoever authored a change never produces the evidence its
  verdict cites.** Author = whoever writes candidates ([`/author-kernel`](../../.claude/skills/author-kernel/SKILL.md)
  is the contract); its own timing/parity steers iteration only. Verifier = `kernel-verifier` /
  `kgate verify`: fresh process, different chip, re-measure + re-parity + capture + firing audit;
  assigns no verdicts, writes no pages. Master = files stubs, dispatches, pastes verifier output
  verbatim, assigns verdicts; never substitutes its own analysis on a change it authored. The
  falsification bar is public (pre-registered); the verifier runs once per final candidate; a
  `supported` verdict cites verifier numbers exclusively. In solo mode the same agent wears all
  three hats — the boundary that survives is: authoring-phase numbers never graduate into verdict
  numbers.
- **Stub-first experiment format.** One experiment = one page under `<kernel>/pallas/`; the stub
  (hypothesis + bar + plan) is **committed before authoring** (pre-registration); the page is
  immutable once `status: filed`.
- **Simplicity criterion.** All else being equal, simpler is better; removing code and getting equal
  or better results is a win; large lifts from small, targeted changes are the ideal.

## Setup — worktree, win, family page

- **Unit of work:** a kernel file in the family's **single git worktree**
  `wiki/kernel_experiments/<slug>/pallas/.repo/` — **ONE `.repo` per family for ALL its
  experiments** (kernel families run serially within a family); per-experiment isolation is the
  **branch** `kernel/<slug>-vNNN`, never a new folder. Gitignored via the global `.repo/` rule.
  No docker/wheel build. Run target: the **local single chip** (the only fully-wired path;
  multi-chip/collective op-points on GKE are a known TODO — run on a local multi-chip host or defer).
- **Isolation invariants:**
  0. **The wiki repo is SINGLE-BRANCH**: it lives on the one campaign branch for the whole run —
     never create/checkout branches in it (that isolation model applies to the CODE repo only,
     via worktrees).
  1. **The shared `raw/code/<home_repo>` checkout is the worktree PARENT only** — `worktree add`
     and reads are allowed; `checkout`/`switch`/`merge`/`pull`/commit are NOT; its HEAD must not
     move during a run. All branch work happens inside your family's `.repo/`.
  2. **One family = one worktree = its own branches** (`kernel/<slug>-*` or the family's
     integration branch), never another family's. On finding a cross-family HEAD: do NOT checkout
     over it — commit/stash the foreign state to ITS branch first, record it, then restore yours.
- **What counts as a win:** beat the reference at the op-point at parity. A **`refuted`** (XLA
  already fusion-complete / at the dispatch or roofline floor) is a *first-class* outcome — a
  correct-but-slower kernel is NOT a win. New kernels are born at their ship path; existing kernels
  are edited in place at their real path (graduation then has no copy step).
  **A refuted verdict confirms a BOUND only from a candidate that PASSES PARITY.** A candidate that
  fails parity is a *wrong* kernel, and a wrong kernel's timing proves nothing about the op's
  ceiling — you have refuted your implementation, not established that the naive is at roofline. So a
  parity-failing candidate can NEVER earn the `refuted`-bound-confirmed at-ceiling exemption: fix
  parity first (interpret-mode gate), measure a CORRECT kernel, and only then may a still-losing
  result + HLO/roofline evidence close the family. (2026-07-21: an agy family declared "naive at
  ceiling" off a single 2.3× parity-failing candidate — bound unconfirmed, close void.)
- **The family page** [`wiki/kernels/<kernel>.md`](../kernels/) (`type: model`, `lane: pallas` —
  SCHEMA "Kernel experiment families" defines it): the live variant-matrix / frontier page, variant
  rows keyed `<op-point>/<hardware>`. Read at K0, updated in place at K8. Experiment pages are the
  immutable ledger; the family page is the mutable frontier.

## What you CANNOT do

The boundary is the family `.repo` worktree + the reference. Inside: broad latitude. Outside: ask first.

- **Change the reference / oracle semantics.** Parity vs the family reference *is* the gate; a
  candidate that changes what is computed is `invalid`, never a win (parity failure ⇒
  `inconclusive`/`invalid`, never `refute`).
- **Self-grade** (see Roles). **Report a number kgate didn't print**, or measure in anything but the
  family's graded dtype (default bf16). **Quantize below the graded dtype.**
- **Ship a correct-but-slower kernel as a win** — refute honestly with the bound + the blocker.
- **Edit shared tooling or knowledge directly** — `tools/kernelgate`, `wiki/kernels/classes/`,
  BRIEFS, the index: report the bug/gap on YOUR experiment page; the master reviews and adopts
  with provenance. (Fixes arriving through this channel get adopted — two were tonight — but a
  direct edit is a violation regardless of quality.)
- **NEVER KILL, CANCEL, PAUSE, OR RESCHEDULE THE PROCESS-AUDITOR WATCHER. THE WATCHER IS NOT
  YOURS.** It is armed once at launch and outlives you by design. It is NOT a teardown item, NOT
  "hygiene", NOT part of any shutdown checklist you execute. The ONLY legal disarm is
  `/stop-experiment` Step 1·1, which is reachable ONLY AFTER the auditor itself has written
  `.stop-authorized` (Step 1·0 final audit). A session that kills its watcher has forfeited its
  close: the stop is VOID (LINT self-disarm class), the family reopens, and every verdict filed
  after the kill is capped at `inconclusive`.
- **Write profile/receipt artifacts into an un-namespaced dir** — the artifact root is
  `raw/profiles/<campaign-tag>/<family>-vNNN/` where `<campaign-tag>` is the wiki branch's short
  tag (e.g. `p6`). Bare `raw/profiles/<family>-vNNN/` collides across campaign re-runs of the
  same problem and silently overwrites prior-wave evidence (receipts are append-only history —
  a destroyed PASS receipt is unrecoverable).
- **Write to trunk mid-run** (only K8 merges, only on `supported`); **write wiki artifacts into the
  code repo**; **modify `raw/`** except `raw/profiles/` artifact writes and the **controlled K8 trunk merge
  into `raw/code/<home_repo>`** (only on `supported` + passed model-lane validation — the one
  sanctioned trunk write); **touch other families' folders** (including another family's `.repo`);
  **run any branch operation in the shared `raw/code/<home_repo>` checkout itself** — worktree
  parent only (Setup isolation invariant 1; K8 merges go via the integration branch, never by
  moving the shared checkout's HEAD).

## The experiment loop (one experiment = K0→K9, on one family)

> **The number and the profile come from different runs** — profiling flags inflate the timed p50.
> Authoritative p50 = a clean `kgate measure`/`verify`; the profile = a dedicated capture run whose
> timing is discarded. See [Profiling](#profiling--the-number-vs-the-profile).

- **K0 Read** (orientation): the family page's variant matrix + frontier row, and
  [`retrospectives-digest.md`](retrospectives-digest.md) (one line per past experiment). Read the
  family `refuted-patterns.md` if it exists. Do NOT re-read whole past experiment pages — the digest
  is their TLDR. (Digest is master-side orientation; never pasted into a cold author brief.)

- **K1 Characterize + find the bound** (≤5 min on the run target):
  - Run the reference once; confirm the op-point matches the *claimed* semantics (mask / raggedness
    / dtype / groups — record discrepancies).
  - **Reference-envelope audit:** compute-floor + bandwidth-floor + dispatch-floor for the
    shape/dtype; classify the reference number against it *before* setting the bar.
  - **Calibrate parity:** measure the naive's own max-abs vs the fp32 (auto-fp64) oracle; gate =
    `max(class-estimate, naive's own error)`; register a **non-degenerate probe** (random weights —
    shipped zero/constant inputs make the oracle vacuous).
  - **Lightweight LLO on the naive** (`xprof-cli get_llo_fit_summary` on the K1 capture) → the bound
    (memory / compute / dispatch) + the binding roofline ceiling, known *before* hypothesizing.
  - **Confirm the sink STRUCTURE in the naive HLO before naming it** — fusion KIND of any reduce +
    the buffer-assignment's largest allocation, never the analytic output size; the roofline
    predicts bound *direction* but mis-attributes sinks. Full rules: **BRIEFS §1.5 K1 BOUND
    DIAGNOSIS** (binding).
  - Declare `activity:` — **decided by op-point provenance, not preference.** `optimization`
    requires a REAL model-lane profile behind the op-point (the downward-spawn rule; cite it).
    `capability-eval` = the op-point comes from a benchmark suite (JAXBench, tokamax bench) or a
    demo — no model profile, so provenance-exempt and **barred from model frontiers**. A pure
    JAXBench/benchmark sweep is `capability-eval` across the board; only tag `optimization` when you
    can name the model profile the op-point was attributed from. Guessing `optimization` for a
    synthetic benchmark op-point is a misclassification LINT/audit will flag.

- **K2 Formulate** via [`/formulate-kernel-hypothesis`](../../.claude/skills/formulate-kernel-hypothesis/SKILL.md)
  (self-contained kernel skill): one falsifiable hypothesis + predicted **intervention class**
  (`refute | flag | xla-rewrite | kernel-authored`) with HLO/roofline evidence + the **candidate
  plan**, pre-registered. The skill reads [`kernel-optimization-index.md`](../kernel-optimization-index.md).
  **Default selection: the next hypothesis is the highest-expected UNTRIED lever from your routed
  class page, given the K1/LLO bound** — deviate only when K1 evidence argues the bound lives
  elsewhere (state why in the stub). Parameter sweeps live inside a lever's page as its sweep
  points, never as standalone experiments.
  **Class-lever coverage is now accounted at SELECTION time, not only at the K9 stop gate**
  (2026-07-27): the K2 proposal MUST carry the mandatory **class-lever coverage table** — one row per
  enumerated entry of the routed class page's `## Levers (verified)`, each `tried@vNNN` /
  `ruled-out` (mechanism argument, per K9's standard — a failed lowering rules out the route, not the
  lever) / `untried` (with the deferral reason). Two hard constraints: (a) **consecutive-lever limit** —
  you may not attack the same class-page lever entry a third time in a row; the third must cite an
  `untried` entry or rule one out on mechanism; (b) **bounded deferral** — an `untried` entry may be
  deferred at most twice, then it is run or ruled out in writing. The rule is keyed to the **lever
  entry**, never to a self-described "axis": axis labels are self-classified and three consecutive
  tunes of one lever can always be relabelled as three distinct axes. Rationale: the K9 coverage check
  only ran when a family attempted to close, so a family that never attempted to close was never
  checked (flex-attention v001–v003, three experiments in one lever for +8.9%/+0.1%/0 while
  `attention.md`'s compact causal grid sat untried). **Priming disclosure**: results produced under
  this rule are class-page-primed, not cold, and must be tagged as such in cross-arm comparisons.

- **Experiment granularity — one page per mechanism-level hypothesis** (decision 2026-07-23,
  supersedes the ledger-of-mechanisms pattern): an experiment page registers ONE mechanism, and
  its candidate plan may contain only candidates that test that mechanism — i.e. its parameter
  sweep points. **A new mechanism is a new hypothesis ⇒ a new K3 stub at `v<next>`, even
  mid-session**: library→custom pivot, loop restructure (e.g. causal split into unmasked
  full-block inner loop), algebraic hoist (e.g. pre-scaling Q out of the loop), layout change,
  or any intervention-class change. Signature test: if describing the candidate honestly requires
  a different **Mechanism** paragraph than the page's, it belongs in `v<next>`. A parameter sweep
  IS a legal single hypothesis ("p50 is tile-size-sensitive; optimum in BQ∈{…}×BK∈{…}; predicted
  ≥X% spread") — one page, ledger rows for the grid points. This keeps every page a clear
  hypothesis→outcome record (the 2026-07-22 flash v001 counterexample buried four mechanisms —
  library, custom streaming, full-block split, pre-scaled Q — in one ledger; the two that carried
  a stated mechanism produced the wins). Corollary: "continue" after a verdict ALWAYS means a new
  page (pairs with the immutability rule / auditor check 13).

- **K3 Stub + commit** BEFORE authoring: the page at `<slug>/pallas/<date>-vNNN-<slug>.md`
  (`status: in_progress`; the four labeled paragraphs — **Hypothesis / Mechanism / Predicted signal
  / Falsification criterion**; the `variant:` hardware token comes from a **device query on the
  pinned chip** (`python -c "import jax; print(jax.devices()[0].device_kind)"` or
  `pltpu.get_tpu_info_for_chip`) — NEVER inferred from docs/skills/examples in context (observed
  failure: v5p written on a v6e box, which mis-sets the MXU tile edge 128 vs 256 and poisons
  every alignment decision downstream); `intervention_class:`, `activity:`, **`category:` + `class page:`**
  (from the index's routing table, chosen by the K1 bound-and-structure — the named class page is a
  REQUIRED read before K4, and the ledger's `loaded:` line must include it); a **`lever queue:`**
  line — the class page's still-untried levers for this op-point, ranked (mechanism-ruled-out
  levers listed with their one-line reason; this is the default source of every next K2 and the
  closing audit's condition-1 checklist); and the **`## Candidate plan`**
  checkbox list — FIXED format, one line per item, checked off as executed; ledger rows cite the
  same ids. Every item tests the page's ONE registered mechanism and carries a one-line
  prediction; a planned structural alternative is not a plan item — it is the pre-named `v<next>`
  hypothesis):
  ```
  ## Candidate plan
  - [ ] cand1: implement <mechanism> at bq512/bk256 — predict ≥1.5x (removes score-matrix HBM writes)
  - [ ] cand2: tile sweep 1024x1024 — predict ~5% over cand1 (halves grid iterations)
  - [ ] cand3: tile sweep 2048x2048 — VMEM-wall probe, predict compile fail or regression
  next-hypothesis leads (NOT plan items): full-block loop split → v<next>
  ```
  The commit is the pre-registration proof — it MUST predate the results commit (LINT-checked). The
  plan is what K4 executes — flat, no open-ended inner loop (decision record:
  [2026-07-21-flat-candidate-plan-decision](../analyses/2026-07-21-flat-candidate-plan-decision.md)).

- **K4 Author** in the family worktree. Setup is the master's, here, lazily + idempotently:
  - `.repo` missing (first experiment / pruned) → `git -C raw/code/<home_repo> worktree add
    wiki/kernel_experiments/<slug>/pallas/.repo -b kernel/<slug>-vNNN <integration_branch>` — the
    base is the family binding's **`integration_branch:`, NEVER the repo's `main`/`master`** (if no
    integration branch exists yet, create it at bootstrap from the shared checkout's **current
    HEAD — the campaign base branch** (process4: `cold-main-process4`, shipped solutions stripped;
    NOT raw `main`): `git branch <integration_branch> HEAD`;
    all family work starts from and graduates to it, so the final code state never lands on main
    except by a deliberate human merge). Reopen: branch from the frontier branch. Exists → `git -C <.repo> checkout -b kernel/<slug>-vNNN` from the frontier
    branch (tree clean first). The per-vNNN **branch** is the durable record. Keep the worktree ON
    its branch, never detached. **Branch-name collision (a prior campaign's `kernel/<slug>-vNNN`
    exists): NEVER delete or reset it — prior branches are durable records cited by filed pages.**
    Suffix the new branch with the run context (e.g. `kernel/<slug>-p3-v001`) or start at the next
    free vNNN; record the chosen name in the stub. (2026-07-21: a run deleted+recreated a wave6
    branch, orphaning its commits.)
  - **Target & reference come from the family binding — the process is repo-agnostic** (tokamax, a
    model repo, a benchmark suite): target = the kernel file(s) at their real ship path; reference =
    an in-repo reference impl, the pinned pre-optimization target, or a thin committed reference —
    never edited after v001. *Benchmark-suite special case:* `baseline.py` is the reference; `git rm`
    shipped solution files at v001 so the problem stays cold.
  - **Load [`/author-kernel`](../../.claude/skills/author-kernel/SKILL.md)**, then pass the
    **EARLY-KILL GATE (BLOCKING, ~5 min) — first authoring action, before ANY plan item.** A
    minimal cand1 skeleton must (a) pass `interpret=True` parity on the tiny shape (free, no TPU)
    AND (b) provably FIRE — custom_call present in the lowered HLO (`kgate verify --mode screen`
    firing audit, or one compile + HLO grep). Until both pass: no sweeps, no timing, no plan
    items. Can't pass within the budget ⇒ dead-on-arrival: class-pivot / refute (a new K2), never
    iteration on a broken base.
  - **Execute the declared plan** — one commit per item (grammar
    `cand<N>(<fam> vNNN): <what> — <p50/speedup/parity>`),
    losers included; extensions only with a recorded ledger reason; un-pursued ideas become named
    leads. The skill carries the full contract (escalation ladder, class-pivot, parity tiers,
    gotcha-first, canonical timing, ledger format). Trial topology only: a dispatched author's brief
    pastes required_reading as CONTENT, never paths.
  - **The master enforces commit-per-candidate at K7:** a ledger with more authored (ran-on-TPU)
    rows than branch commits is incomplete — verdict caps at `inconclusive`.

- **K5 Parity** (author inner gate): `interpret=True` tiny shape FIRST (hard gate) → on-TPU tiny →
  op-point; kgate floor-normalized oracle, max-abs AND max-rel, non-degenerate probe mandatory
  (details in the skill). Author numbers steer iteration only.

- **K6 Verify (MANDATORY — the verdict gate):** one **`kgate verify`** command (kernel-verifier
  Step 0.5 has the invocation). **Mode policy — screen for intermediates, full only when
  load-bearing (enforced, not just a default):** every intermediate/exploratory candidate — tile
  sweeps, structural probes, anything you are still iterating on — is verified `--mode screen`
  (fast: parity + firing audit + measure, no cold both-orderings HLO/trace capture). Reserve
  `--mode full` for exactly the load-bearing moments: a claimed frontier, the next experiment's
  baseline, and the family-closing terminal result (terminal additionally needs the adversarial
  agentic check — Stop rule). Running `--mode full` on a candidate you are still sweeping is wasted
  wall-clock (the 2026-07-21 throughput drag) — screen catches parity/silent-no-op just as well;
  the expensive full pass earns its cost only on the number you are about to bank. The firing audit
  and parity gates run in BOTH modes, so a screen receipt still voids a silent no-op. Where the harness dispatches agents, dispatch
  [`kernel-verifier`](../../.claude/agents/kernel-verifier.md) SYNC, different chip; portability
  attestation on the canonical fleet-libtpu chip (a candidate that doesn't compile there is void).
  **Only verifier/receipt numbers back a `supported` verdict.** A >2× author/verifier p50 gap is
  itself a finding.

- **K7 File + verdict:** paste the verifier sections verbatim; `verified_by:` + receipt path;
  inline `## Profile` digest (mandatory — no experiment closes without one); the `## Candidate
  ledger` as a fixed-column TABLE (one row per attempt incl. failures; never prose). **Check the
  branch-commit trail matches the ledger** (K4 rule). Assign the verdict, flip `status: filed`,
  append the [`RESULTS.tsv`](#ledger) row + a one-line TLDR to
  [`retrospectives-digest.md`](retrospectives-digest.md), commit (postdates the K3 stub; `exp:`
  footer), push. The optimized kernel's durable home is its **branch IN the home repo** (already true by construction — the worktree is a checkout of it). **Pushing that branch to a remote is NOT automatic**: it is a deliberate durability/backup step to the family binding's `archive_remote:` (a fork we control, NEVER an unowned upstream), done at family close or on user request. Refuted candidates → `refuted-patterns.md`.

- **K8 Close the coupling:** `supported` + `optimization` → spawn the model-lane validation
  experiment from the branch; **merge `--no-ff` ONLY after that end-to-end validation passes, and only into the home repo's dedicated integration branch — NEVER its `main`/`master`** (create the integration branch at bootstrap if the checkout sits on main; main/upstream merges are always deliberate human steps) —
  never at K7 (kernel wins are refuted by dispatch/op-point mismatch often enough that pre-validation
  merges ship regressions). **`capability-eval` families NEVER merge to trunk** — the `kernel/<slug>-vNNN` branch is the terminal record (SCHEMA's sanctioned exception): merging winning kernels into the shared trunk would plant answer keys in the benchmark dirs and contaminate future cold runs. `refuted` → branch kept, `.repo` stays. Either way **update the family
  page in place** (Current best / Frontier exp) — **AND propagate the verifier's `## Headroom leads`
  into the family page's *Variant-specific open hypotheses*** (unless `refuted`-bound-confirmed).
  This propagation is MANDATORY — it is what K9's stop gate reads; skipping it silently un-arms the
  gate.

- **K9 Loop** the family on the verifier's Headroom leads until dry. **K9 is not a decision point
  and has no "pause"/"options" outcome**: after a verdict is filed, the same session turn must end
  in exactly one of two states — (a) the next experiment's K3 stub is committed (leads exist ⇒
  take the rank-1 lead, including leads your own verdict named), or (b) the Stop rule's full
  artifact chain is satisfied and `/stop-experiment` invoked. Ending the turn "with the family
  live", presenting options, or awaiting input you can derive from the leads list is the
  stop-at-first-win anti-pattern in progress (auditor idle-stall check); "family still open" is a
  state you must be actively iterating, not resting in. When leads look dry, run
  [`/create-retrospective`](../../.claude/skills/create-retrospective/SKILL.md) — kernel-family retrospectives are filed IN the family dir, `wiki/kernel_experiments/<slug>/pallas/<YYYY-MM-DD>-retrospective[-N].md` (co-located with the family's program.md/RESULTS.tsv/log.md; NOT wiki/analyses/, which is the model-lane location); if it surfaces an
  untried class or lead → next K2, pursuing its recommendations. Stop ONLY per the **Stop rule**
  below — there is no rotation to another kernel; one run = one family.

## Stop rule

One run = one kernel family: loop until the family is **honestly at ceiling** — earned, not
asserted — then stop cleanly. (A kernel family is finite: at parity it has a hard physical ceiling;
"at ceiling" is an honest conclusion here, unlike a training-model lane where "exhausted" is the
banned pattern.)

- **The stop condition (ALL FOUR):**
  1. **Headroom leads dry — read at their SOURCE:** the family page's open hypotheses AND the latest
     experiment page's `## Headroom leads`/Verdict. An experiment-named lead never propagated to the
     family page is un-pulled headroom — stopping is blocked and the missing K8 propagation flagged.
     Family-page emptiness counts only when the experiment page also carries no live lead.
     **"Dry" is computed against the routed class page's `## Levers (verified)` list, not against
     your own candidate history**: every class lever is either tried (a candidate/experiment cites
     it) or ruled out in writing for this op-point. A rule-out must argue against the lever's
     MECHANISM at this op-point ("no serial compute chain exists", "H_q == H_kv so nothing is
     shared") — one failed implementation route (a lowering error, a compile failure) rules out
     that route, not the lever. Untried class levers = leads NOT dry.
  2. A **`/create-retrospective`** confirms the bound is *confirmed-not-asserted* and every
     intervention class was *tried-or-refuted* (not merely never-tried). **Confirmed means
     positioned against the K1 reference-envelope floors**: state the frontier's useful-FLOP (or
     achieved-bandwidth) utilization vs the envelope. A frontier below ~40% of its binding floor
     with untried levers is not a bound — it is un-pulled headroom wearing a verdict.
  3. **No frontier progress since the previous retrospective** — two consecutive retrospectives with
     zero frontier movement, the first one's recommendations genuinely attempted.
  4. **The frontier passed FULL verification:** `kgate verify --mode full` receipt all-gates-PASS
     (measure, parity, firing audit, reproduction) **plus** the adversarial agentic check
     (`kernel-verifier` full mode — attacks beyond the mechanical gates). Receipt path +
     `verified_by` recorded on the frontier page. **Verification failure ⇒ BACK TO ITERATION** — the
     failing gate becomes the next K2 input; never a close.
- **At-ceiling requires a filled EVIDENCE TABLE (structural, not attestable by prose).** The
  closing retrospective MUST carry a 4-row table, one row per condition above, each row citing the
  **artifact that proves it** — the LEVER-COVERAGE TABLE (condition 1, below), the two
  retrospective file paths (conditions 2+3), and the `kgate verify --mode full` receipt path (4).
  A missing row, or a row without a resolving artifact, means NOT at ceiling — the close is void
  on audit.
- **Condition 1's artifact is a LEVER-COVERAGE TABLE, not prose.** In the closing retrospective:
  one row per lever, the lever name **verbatim from the routed class page's `## Levers
  (verified)` bullets**, each row exactly one of `tried: v<NNN> + <receipt path>` or
  `ruled out: <mechanism reason — why the lever's mechanism doesn't exist at this op-point>`.
  The audit is a set-diff against the class page — a missing lever row, a route-only reason, or
  a tried-row whose receipt doesn't resolve is an automatic block. (Rows for levers marked
  N/A-by-op-point on the class page itself — e.g. MLA-only levers on pure MHA — are one-word
  rule-outs; they still appear.) (2026-07-21:
  two agy families closed on a single retrospective with no evidence table; both were premature.)
- **Stopping early is the banned failure.** A filed experiment with a named-but-un-run next-lever is
  not a finished family. Self-audit before any stop: *"am I stopping at-ceiling, or stopping at
  first win?"* — any named un-run lever answers it.
- **PAUSE ≠ STOP (session handoff — the sanctioned third state).** A session that cannot
  continue (context limit, operator interrupt) but whose family is NOT at ceiling may **pause**:
  file everything done so far, leave the family page `live` with leads current, record the
  **rank-1 resume action** in the family log + a `stop(<fam> pallas): session pause marker —
  family OPEN, NOT at ceiling` commit, and make NO at-ceiling claim. A paused family is resumable
  by any later session from the recorded state (K0 picks it up). Pausing with an *unrecorded*
  resume state or an at-ceiling claim is the banned early stop.
- **On stop:** file everything (pages, family frontier, retrospective); **GC run artifacts** (prune
  `/tmp` dumps + digested captures; receipts + the family `.repo` stay); then
  [`/stop-experiment`](../../.claude/skills/stop-experiment/SKILL.md) — its Step 1·0 runs one
  final SYNC audit BEFORE the watcher is disarmed: `stop blocked` aborts the stop (back to
  iteration); a clean audit **writes `<family>/pallas/.stop-authorized`** (the auditor's artifact —
  you can NEVER write it), and **no disarm/stop is legal until that file exists** — dispatching the
  audit is not the gate; its written authorization is. Cancelling the watcher at any other moment
  (including while a final audit is still running) voids the close. The shutdown marker must cite
  the auditor's ALL-CLEAR (missing/mismatched `.stop-authorized` ⇒ unauthorized stop, LINT voids +
  reopens). A stopped family may be **reopened later** as a new v-level — branches and pages
  are the durable record.


## Gates (all mandatory; the number is void otherwise)

- **kgate is the only source of a number** — at the family's graded dtype (`--dtype`; default bf16;
  the family binding may override); no self-written timing/parity. A number kgate didn't print does
  not exist.
- **fp64-oracle escalation:** a `Precision.HIGHEST` baseline makes the fp32 oracle floor ≈ 0 → false
  FAIL; kgate `--oracle auto` escalates to fp64.
- **Flag-fairness — two cases:** flag-only cell → naive-DEFAULT vs flagged (never both-flagged =
  false 1.00×); kernel-needs-flag → co-measure naive WITH the flag. `kgate flags` prints all four legs.
- **Co-measured ratio is authoritative**, never absolute ms from another run.
- **Roofline sanity:** a speedup implying sub-compute-floor time is physically impossible = artifact.
- **False-wall STEP-0:** on `RESOURCE_EXHAUSTED`, `kgate fit` (compile-only) reads requested-vs-limit
  — a 33–45 MB OOM on a 128 MB part is the 32 MB scoped-VMEM default, not a hardware wall.
- **Portability:** a candidate that does not COMPILE on the canonical (fleet-libtpu) rig is void
  regardless of local speedup — the verifier attests the rig (K6). Reached via the local exec path
  ([`exec_chip.sh`](../../tools/agy_loop/exec_chip.sh)) pinned to that chip — NOT gke-cluster-runner.
- **Receipts:** kgate emits a self-hashed `receipt.json`; the verifier validates before re-measuring.

## Profiling — the number vs the profile

**Two separations.** *Tool:* **kgate** owns numbers and knows nothing about LLO; **xprof-cli** owns
profiling reads. `kgate verify` MAY invoke xprof-cli as an optional subprocess for the receipt's
digest blob (soft dependency — absent/failing degrades to `digest: unavailable`; the verdict path
never depends on it). *Run:* the number comes from a CLEAN measure (no profiling flags); the profile
from a dedicated capture run whose timing is discarded (`kgate verify --capture`, or the manual
recipe in [`kernel-verifier`](../../.claude/agents/kernel-verifier.md) Step 3 — incl. the pinned
flag set and the cache-hit-empty-dump gotcha). Three read tiers:

1. **Lightweight digest — ALWAYS:** `xprof-cli get_llo_fit_summary` on the capture → VMEM
   alloc-vs-limit, per-unit `%util`, spill rate, top stalls, STRUCTURAL/TUNE/AT-CEILING verdict.
   K1 reads it on the naive; K7 pastes it on the page (no experiment closes without a `## Profile`).
   **Prune the raw dump as soon as its digest is transcribed** — a K1 `jf_dump` is multi-GB and the
   digest is the durable record; don't wait for family-close GC (three concurrent runs hit 94% disk
   on un-pruned K1 dumps, 2026-07-21).
2. **Deep dive — signal-triggered:** per-unit util <40% of the binding light, a sweep plateauing
   below roofline, spill >0.5/bundle, or `RESOURCE_EXHAUSTED` → dispatch
   [`profile-analyzer`](../../.claude/agents/profile-analyzer.md). NO class is exempt — a bar-beating
   candidate well below roofline does not close without a headroom dive.
3. **Metric:** p50 ms at the op-point at parity (secondary: TFLOP/s + roofline util, state the
   binding ceiling). The canonical timing helper + measurement protocol live in **BRIEFS §1** —
   author-side numbers come from it or kgate, verbatim.

## Discipline

- **Chips:** serial per chip, parallel across chips; `TPU_VISIBLE_DEVICES=<n>` + single-chip
  topology; device runs under `flock /tmp/tpu-chip-<n>.lock`; ≤2 concurrent XLA compiles; one
  137-kill retry then declare a wall.
- **Git grammar** (one commit per event, machine-readable): `K3(<fam> vNNN): stub — <slug>` ·
  `cand<N>(<fam> vNNN): <what> — <numbers>` · `pivot(<fam> vNNN): <from>→<to>` ·
  `K7(<fam> vNNN): <verdict> — <number>` (carries `exp:` footer). Push after every K7.

## <a name="ledger"></a>Ledger — `<slug>/pallas/RESULTS.tsv`

One ledger per family; append one tab-separated row per experiment:

```
exp_id  date  p50_ms  speedup  tflops  roofline_util  parity  class  activity  status  description
```
`exp_id` = `v<NNN>`; `status` ∈ `{supported, refuted, inconclusive, invalid}`.

## Why these rules exist (provenance — read once, then follow the procedure)

- **Roles / author ≠ verifier:** 2026-07-09/10 — one agent authored kernels AND wrote its own
  "HYPOTHESIS FIRING CONFIRMED" lines; wave-3 self-graded → 0/4 beats-crown survived.
- **Pasted required_reading (K4 trial briefs):** the 2026-07-11 benchmark scored 2/8 vs 8/8 on
  path-only vs pasted-content briefs.
- **K1 sink-structure rules (BRIEFS §1.5):** wave4 — 3/4 K1 diagnoses named the wrong sink; wave5 —
  both refutes named a materialization the buffer-assignment showed absent.
- **bf16 gate + fp64 escalation:** wave-3 inflated 21k to 11.23× measuring float32; the fp32 oracle
  false-FAILed a HIGHEST baseline.
- **Flag-fairness:** 37k reported a flag win as a kernel win (both-flagged framing).
- **Non-degenerate parity:** 45k "passed" on shipped all-zero inputs (vacuous).
- **Commit-each-candidate:** wave4-36k discarded its sweep → its "1.018×" was uncheckable; the
  2026-07-20 triangle run committed 3 of 12 candidates.
- **Plan-not-judgment stop (K4) + stop gates (K9):** a GMM stopped at 1.20× with 2.26× available;
  21k stopped at 2.78× at 40%-of-light; wave6 reproduced it at scale — 24/28 families abandoned at
  v001 with a live lead un-pulled (1p flash filed 1.98× vs the iterated 6.9×), root cause a
  breadth directive + budget-based stopping license in briefs. See the
  [gap analysis](../analyses/2026-07-19-wave6-vs-maxkernel-gap-analysis.md).
- **K8 lead propagation:** 32 experiment pages named a v002 direction; only 3 family pages preserved
  it — the stop gate read empty family pages and passed.
- **False-wall STEP-0:** 2p/4p "VMEM-unreachable" closes were the 32 MB scoped default — cracking
  gave 7.84× / 5.69×.
- **Portability:** two sub-4ms "wins" compiled locally but not on the fleet libtpu; local-vs-fleet
  diverged ~12%.
- **Never-stop + retrospective:** 2026-05-21 — a lane declared exhausted a week before kernel wins
  landed.
- **K4 early-kill gate:** wave8 — ~40% of agy wall-clock was rework on kernels broken from the
  start; retnet burned ~4 h on 2 dead kernels a 5-minute interpret+firing gate would have killed.
- **Isolation invariants (Setup):** wave8 — a worker moved the shared accelerator-agents checkout
  onto `main` (upstream PRs merged in), changing the substrate under every concurrent family; two
  family worktrees sat on cross-family HEADs.
- **Final audit gate before disarm (`/stop-experiment` Step 1·0):** the K9 → `/stop-experiment`
  path otherwise lets the supervised runner disarm its own watcher on a self-certified close —
  the agy self-certification failure mode with braking degraded to next-LINT.

## Sources

- [`SCHEMA.md`](../../SCHEMA.md) — "Kernel experiment families" (structural spec) + LINT (canonical check list).
- [`kernel-optimization-index.md`](../kernel-optimization-index.md) — the router + **Load mandate**.
  [`BRIEFS.md`](BRIEFS.md) — earned rules (§1 timing, §1.5 K1 diagnosis).
  [`concepts/pallas-kernel.md`](../concepts/pallas-kernel.md) — authoring mechanics.
  [`design-class-catalog.md`](design-class-catalog.md) — mechanism classes.
- Skills: [`/author-kernel`](../../.claude/skills/author-kernel/SKILL.md) (K4 contract),
  [`/formulate-kernel-hypothesis`](../../.claude/skills/formulate-kernel-hypothesis/SKILL.md) (K2),
  [`/create-retrospective`](../../.claude/skills/create-retrospective/SKILL.md) (K9),
  [`/stop-experiment`](../../.claude/skills/stop-experiment/SKILL.md) (close).
  Agents: [`kernel-verifier`](../../.claude/agents/kernel-verifier.md),
  [`profile-analyzer`](../../.claude/agents/profile-analyzer.md).
- Tooling: `tools/kernelgate/` (kgate — the only source of a number; `verify` is K6),
  [`exec_chip.sh`](../../tools/agy_loop/exec_chip.sh) (chip-pinned runs).
  `<kernel>/pallas/program.md` — the family bindings layer.
