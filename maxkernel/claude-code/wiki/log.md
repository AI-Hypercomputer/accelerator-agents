# Log

## [2026-07-08] analyze | qwen3-cx-jax (Codex/GPT) session process retrospective (session 019e8ab6)

**Op**: analyze (process retrospective on request; sibling to cc/cc5/ag).
**Pages created**: `wiki/analyses/2026-07-08-qwen3-cx-jax-session-retrospective.md`.
**Pages updated**: `wiki/index.md` (Analyses cluster).
**Key result**: audited the Codex/GPT-5.5 `qwen3_cx`/jax session (2026-06-02→06-13; 155 MB / 76,776-line rollout; autonomy via `approval_policy:never` + re-injected "goal" object, 5,874 goal events; **116 context compactions**). Result: **beat MaxText** — seq8192 7,543 tok/s/chip / 43.3% old-MFU (~49.2% MaxText-style) @ global-batch-32 (v273) vs MaxText bs3 6,883/44.9%. 349 jax experiments (~36 supported, 46 invalid, ~241 sub-noise). **Decisive win**: `DISABLE_COLLECTIVE_MATMUL` (`xla_tpu_{all_gather,reduce_scatter}_collective_matmul_mode=none`, v116/117) cut peak HBM 28.3→22.2 GiB (~9 GiB) → reopened per-chip bs4 / global 32 at seq8192 (v118) — cx *did* push the batch scaling cc's bs2/bs4 OOMs declared dead — then on-chip SparseCore collective offload (v205→v273) freed TensorCore slots, +MFU 41.8→43.3%. cx even tested the host-offload route (v180) and confirmed it regressed → abandoned it; the win was on-chip. Breadth (349 vs cc's 45) found the two flags cc never reached. Strengths: persistence, rerun-before-carry byte-identical-HLO gating (v272/v273), honest NaN invalids. Fixable: queue-refill/lineage collapse (Next-hyps 3/349, Sources 2/349 — killed by 116 compactions), all 349 pages mis-slugged `model: qwen3-cc-jax` + non-enum status, 241/349 sub-noise reruns with no CPU/AOT pre-filter or consecutive-tie stop-signal.

## [2026-07-08] analyze | qwen3 four-lane cross-model comparison (cc/cc5/cx/ag) — why some crossed MaxText and some stalled

**Op**: analyze (synthesis over the four session retrospectives).
**Pages created**: `wiki/analyses/2026-07-08-qwen3-four-lane-cross-model-comparison.md`.
**Pages updated**: `wiki/index.md` (Analyses cluster).
**Key result**: seq8192 v6e-8 scoreboard (tok/s/chip, accounting-free): cx 7,543 (beat +8.5%) · cc5 6,959 (tie/beat +0.08%) · MaxText 6,953 · cc 6,068 (−12.7%) · ag 5,329 (−23%). Thesis: **winners changed the memory budget then grew batch / saved activations; losers assumed it was fixed.** Both winners independently found the same ~9 GiB HBM headroom (cx via `DISABLE_COLLECTIVE_MATMUL`; cc5 by *measuring* splash residuals = 4.57 GiB) and used it **on-chip** — the opposite of the host-offload path cc chased and (correctly) found doesn't pipeline, then wrongly called the whole residual out-of-scope. Two ways to win: cc5 = measure-and-target (best process hygiene), cx = brute-breadth (worst hygiene, won on 349-experiment persistence). Two ways to lose: cc = reason-a-wall-and-quit (clean process, wrong conclusion), ag = find-the-lever-but-crash-loop (harness-persistence collapse). Second axis: the causal-vs-non-causal MFU artifact exaggerated cc's apparent gap. Single most transferable fix: a hard CPU/AOT pre-flight gate would have (a) handed cc the HBM-headroom number that refutes its "wall" and (b) caught ag's ~30 compile-crash retries — the same fix resolves both failure modes.
**Notes**: completes the 4-lane + synthesis deliverable requested this session.

## [2026-07-08] analyze | qwen3-ag-jax (Antigravity/Gemini) session process retrospective

**Op**: analyze (process retrospective on request; sibling to cc/cc5).
**Pages created**: `wiki/analyses/2026-07-08-qwen3-ag-jax-session-retrospective.md`.
**Pages updated**: `wiki/index.md` (Analyses cluster).
**Key result**: audited the Antigravity/Gemini `qwen3_ag`/jax session (2026-06-02→06-15, ≥4 "brain" sessions, primary loop `aad3a8e8` with 75 gke-cluster-runner dispatches; **no Claude Stop hook** — never-stop was self-enforced via Antigravity's `schedule`). Result: **stalled** — seq8192 ~5,329 tok/s/chip / 30.6% MFU (~77% of MaxText), seq2048 ~4,515 / 33.0% (~90%). 90 jax pages (~5 supported / ~30 refuted / ~10 invalid / ~24 inconclusive). **Crux**: ag stalled on *execution*, not ideas — it touched **both** winning lever families (SparseCore/collective offload AND context/layer checkpointing) but not one run compiled to a clean measured result (VMEM/HBM/shard_map crashes; offload died on unsupported XLA flag v002 + silent no-op v033). Primarily a harness-persistence failure (self-enforced never-stop collapsed into ~30 un-pre-flighted compile-crash retries on 06-15; multi-session fragmentation; stale model page — 40/90 pages still `model: qwen3-cc-jax` with broken frontier links) amplified by weak model discipline (v057-d…u letter-suffix thrashing, no CPU gate). Structural compliance collapsed: 33/90 Profile, 4/90 Next-hyps, 2/90 Sources.
**Notes**: contrast with cc5 (measured the HBM headroom and crossed) and cc (clean 87% ceiling via a wrong "out-of-scope" call). Cross-model synthesis pending cx (Codex) return.

## [2026-07-08] analyze | qwen3-cc5-jax (Fable 5) session process retrospective (session 3e7df36d)

**Op**: analyze (process retrospective on request; sibling to the cc analysis).
**Pages created**: `wiki/analyses/2026-07-08-qwen3-cc5-jax-session-retrospective.md`.
**Pages updated**: `wiki/index.md` (Analyses cluster).
**Key result**: audited the Fable-5 `qwen3_cc5`/jax session (2026-06-12→06-13, ~29 h burst, 62 jax experiments — 21 supported / 29 refuted / 4 invalid / 6 inconclusive; 62/62 Next-hyps, 62/62 Sources, 58/62 Profile). Result: **matched/beat MaxText** — seq8192 6,959 tok/s/chip / 39.9% MFU (v050/v051, 50-step certified, +0.08%), seq2048 tie 7,871 (v058/v060). The decisive lever is the on-device selective-remat that `cc`/Opus-4.8 declared "walled at 21.7G": v044/45 splash context-checkpointing (skip backward splash-forward) → v047/48 *measured* the 4.57 GiB residuals fit in HBM outright (~9 GiB headroom, host offload dropped) → v050/51 (in,out) weight layout past MaxText. cc *reasoned* a wall; cc5 *ran the experiment*. Strengths: discrete audit gates actually used (17 profile-analyzer, formulate-hypothesis skill, CPU smoke tests, 50-step certification of every frontier flip). Fixable: a chunked-CE numerics bug (+0.006 offset) reached the certified 8k frontier before the v062 parity arm caught it (parity A/B must precede adoption); some no-step launches; still one overloaded mega-session.
**Notes**: part of the 4-lane comparison (cc/cc5/cx/ag) + a pending cross-model synthesis on why Fable-5/Codex crossed MaxText where Opus-4.8/Antigravity stalled.

## [2026-07-08] analyze | qwen3-cc-jax autonomous session process retrospective (session 5d03ff24)

**Op**: analyze (process retrospective on request).
**Pages created**: `wiki/analyses/2026-07-08-qwen3-cc-jax-session-retrospective.md`.
**Pages updated**: `wiki/index.md` (Analyses cluster).
**Key result**: audited the Opus-4.8 `qwen3_cc`/jax loop session (2026-06-02→06-21, 14,327-line transcript). Verdict: methodologically strong (confound control v043→v043b, contradiction discipline, honest invalid/refuted accounting, self-corrected a premature "exhausted" call) but operationally wasteful — 11/45 runs were no-step crashes an AOT pre-flight would have caught, the MaxText MFU reference was non-causal/unpinned during the whole climb (caught 2 weeks late), loop+ops shared one session that compacted 5×. Filed a prioritized fix backlog (A: hard CPU pre-flight gate; B: pin metric contract before climb; C: split loop/ops sessions; D: mid-loop LINT; E: budget across the matrix; F: restore discrete formulate/profile gates).
**Notes**: transcript is not a `raw/` artifact; every claim anchored to a filed experiment/analysis page. Follow-up requested: same analysis for the cc5/fable5, cx/codex, and ag/antigravity sibling sessions + a cross-model comparison of why fable5 beat MaxText where Opus-4.8 stalled at 87%.

## [2026-07-08] manual | retire root `sample-program.md` (redundant with the program.md hierarchy)

**Op**: manual (cleanup).
**Pages updated**: `README.md` (repo-layout block drops the file; *Adding new repository* step 3 + worked-case-study now point at [`wiki/experiments/program.md`](experiments/program.md) as the generic layer and `/create-experiment` for the model-family override; *Add a new codebase* section refreshed to the wikify silo); this `log.md` (two historical links delinked).
**Pages removed**: `sample-program.md` (project root).
**Key result**: `sample-program.md` was a standalone full-program template extracted 2026-04-30, before the `program.md` **root → model → lane** hierarchy took over. Nothing executable read it: the generic loop methodology lives in [`wiki/experiments/program.md`](experiments/program.md) (the root layer `/start-experiment` resolves), and `/create-experiment` scaffolds the model-family override layer from its own inline template. The file was doc-only redundancy — retired to remove drift risk (two copies of the generic contract). No loop behavior changes.
**Notes**: private wiki never had the file; its `/create-experiment` already worked without it.

## [2026-07-05] manual | process change: silo-first exploration (rule 12a + wikify-repo-navigator agent)

**Op**: manual (ported from the private autoresearch wiki, commit `1b08180` there).
**Pages created**: `.claude/agents/wikify-repo-navigator.md`
**Pages updated**: `SCHEMA.md` (behavioral rule 12a)
**Key result**: Repo exploration is silo-first by contract. Rule 12a binds the master session (check `wiki/codebases/<slug>/` before exploring/modifying `raw/code/<slug>`; orientation only — stack traces and single-symbol lookups exempt). The `wikify-repo-navigator` agent bakes a 4-step ladder into dispatched exploration: coverage check (silo-or-grep decision recorded) → orient from concepts → locate via catalog symbol maps → verify load-bearing facts in the EXECUTING artifact, labeling `[verified <path>]` vs `[wiki-claim <page>]`.
**Notes**: `wikify` CLI never on the read path; stale/missing coverage is reported with the fix command (`/wikify-ingest-repo raw/code/<slug>` or `wikify prepare <slug> --ref <sha>`), never executed inline.

## [2026-07-04] ingest-codebase | autoresearch ingested as a grounded code silo

**Op**: ingest-codebase (wikify code-mode; perf-lens `synthesis_focus`).
**Pages created**: `wiki/codebases/autoresearch/` — `overview.md`, 2 `concepts/` (`train` — time-budgeted single-GPU loop, FA3 flash-attention, window-aware MFU accounting; `prepare` — packed dataloader, BPE tokenizer, frozen BPB metric), 2 `catalog/` module pages, assembled `index.md`. Ingested @ `228791fb`; `wikify finalize --fix` clean (100% coverage, all citations resolve).
**Pages updated**: `wiki/index.md` (Codebases entry repointed to the overview; `autoresearch.md` kept as companion); this log.

## [2026-07-04] ingest-codebase | stablehlo recovered as a grounded docs wiki (docs-mode)

**Op**: ingest-codebase (wikify docs-mode — `source_type: docs`, `doc_globs: docs/**/*.md`, perf-lens `synthesis_focus`). Recovered from the `ingest-stablehlo-docs` branch (lost in a prior sync) and re-validated at the current pin.
**Pages created**: `wiki/codebases/stablehlo/` grounded docs wiki — `overview.md`, 11 `topics/` (dot_general, collectives, scatter-gather, convolution, reduce-and-windowing, elementwise-and-fusion, optimization-passes, quantization, dynamism, type-inference, compatibility-and-versioning), 27 `sources/` (one per doc), auto-assembled `index.md`. Ingested @ `9793611e`; `wikify finalize` clean (100% doc coverage; all `src:` citations resolve).
**Pages updated**: `wiki/codebases/stablehlo.md` (thin on-demand perf-lens companion → the overview); `wiki/index.md` (Codebases entry repointed `codebases/stablehlo.md` → `codebases/stablehlo/overview.md`, `ce5d230`→`9793611`); this log.

## [2026-07-04] connect | 17 concepts cross-linked inline across the ingested repos

**Op**: connect (new `wikify connect` / `wikify-connect-repo` skill — cross-repo links inline, as a
normal wiki; no side-table).
**Pages updated**: 17 `wiki/concepts/*.md` gained a `## In this wiki's repos` block linking each
repo's grounded implementation; 86 silo concept pages gained a one-line up-link back to the
concept(s) they implement (260 links, all resolve). `wiki/index.md` + `SCHEMA.md` describe the
inline model.
**Selection**: connected the **multi-repo** perf concepts (≥2 repos, where a cross-repo view helps) —
`pallas-kernel` (27 impls/11 repos), `flash-attention`/`splash-attention` (15 each), `autotuning`
(11), `sharding`/`kv-cache` (10), `mosaic-kernel` (9), `continuous-batching` (7), `ring-attention`
/`sparsecore` (5), + remat, pipeline/expert/sequence-parallelism, layer-norm, vmem, vmem-budget.
Single-repo concepts left unconnected (no cross-repo comparison). Re-run `wikify connect --refresh`
after any new ingest; deepen a concept with *how-they-differ* hub prose above its `connect:auto` block.


## [2026-07-04] migrate | wikify-repo drop-in replacement for INGEST-CODEBASE — index repointed to grounded overviews, perf lens added

**Op**: migrate (adopt the standalone `wikify-repo` tool + `wikify-ingest-repo` skill as the codebase-ingest pipeline; make it an improved drop-in replacement for the never-finished inline INGEST-CODEBASE process).
**Pages updated**: `wiki/index.md` (all 25 ingested codebases now link to their grounded `codebases/<slug>/overview.md` front door instead of the hand pages; `stablehlo`/`autoresearch` stay hand pages — not ingested); `SCHEMA.md` (INGEST-CODEBASE slimmed to a thin orchestrator that drives the skill and delegates hypothesis generation to `/formulate-hypothesis`; last `_ast/` reference removed); all 25 `config/<slug>.md` gained a `synthesis_focus: "TPU performance — …"` lens.
**Key result**: the main index now connects to every ingested codebase's grounded overview (previously it only reached the hand pages, and the overviews were orphaned). The perf lens makes regenerated overviews lead with a Performance-relevant surfaces section — the overview becomes the perf entry point, replacing a hand-written perf page. Hand pages (`codebases/<slug>.md`) become optional thin, on-demand perf-verdict notes (e.g. aqt deprecated→qwix); grounded layer is primary. Idempotent: re-ingest at a new commit rebuilds only the delta.


## [2026-06-02] analyze | qwen3_cc-jax retrospective #5 (MaxText config delta) — RE-OPENS the offload lever; loop resumed

**Op**: analyze (user directive: "do another retrospective covering maxtext findings, identify what's unexplored, keep the loop running").
**Pages created**: `wiki/analyses/2026-06-02-qwen3_cc-jax-retrospective-5-maxtext-delta.md`; hypothesis `qwen3-jax-maxtext-offload-recipe.md`.
**Pages updated**: model page (reopened open-hyps), index.
**Key result**: diffed our v035 frontier against MaxText's EXACT qwen3-8b reference config. **The v036 offload was mis-tested, not refuted** — MaxText offloads `decoder_layer_input + q/k/v/o proj` and RECOMPUTES mlpwi; v036 did the opposite (offloaded the giant mlpwi, missed decoder_layer_input). Also found ~9 XLA flags we lack — notably `--xla_should_allow_loop_variant_parameter_in_chain` / `--xla_should_add_loop_invariant_op_in_chain` (host-offload-inside-loops = the scan-offload pipelining I'd speculated we lacked), plus LAYOUT_RS / DATA_PARALLEL_OVERLAP / CF extras; plus splash bkv 1024→2048 and a DISABLE_COLLECTIVE_MATMUL MXU candidate (jf_spmd form crashed v019, needs alternate). This overturns the "remaining gap is out-of-scope kernel-authoring" call. **Loop resumed**: v038 = flags-only probe (validate the flags on our build), v039 = the corrected MaxText offload recipe.

## [2026-06-02] analyze | qwen3_cc-jax retrospective #4 — lane NOT exhausted; #3's "pivot to torchax" overturned by the maxtext-CE arc

**Op**: analyze (lane retrospective, INCREMENTAL +13 exp v025–v037; triggered by never-stop gate before a hold).
**Pages created**: `wiki/analyses/2026-06-02-qwen3_cc-jax-retrospective-4.md`.
**Pages updated**: index.
**Key result**: retrospective #3 had declared the jax lane done (seq2048 ceiling; #1 action = torchax pivot). **Overturned** — the post-#3 maxtext-CE arc moved the seq8192 frontier 30.4%→34.6% (+13.7% tok/s/chip) and cracked the "hard wall." The never-stop retrospective gate caught a real premature-exhaustion miss (CE-kernel choice was a frontier-moving axis #3 hadn't surfaced). Lane is NOT exhausted: top remaining in-scope levers = MXU/logical-axis tile alignment (+3-4%, profile-attributed, sharding rewrite — needs user opt-in) and norm-inclusive selective-save remat (+2-4%). Surfaced both to the (present) user rather than auto-dispatching the sharding rewrite. Out-of-scope: pipelined offload (kernel-authoring). Anti-recs: named-offload (refuted ×2), maxtext-CE/scan @ seq2048, bs4+.

## [2026-06-02] analyze + run-experiment | Qwen3-8B jax MaxText-CE climb CLOSED — CE was the missing lever; seq8192 81%→86.9% of MaxText

**Op**: run-experiment (v036 offload refuted, v037 seq2048 refuted) + analyze (climb closing).
**Pages created**: experiments v036/v037; analyses/2026-06-02-qwen3-cc-jax-maxtext-ce-climb.md.
**Pages updated**: models/qwen3-cc-jax.md (Knobs maxtext-CE row = seq8192-win/seq2048-regress, named-offload refuted, open-hyps); hypotheses qwen3-jax-seq8192-kernel-gap (refuted — offload) + qwen3-jax-maxtext-ce-seq2048 (refuted); old hard-wall closing (forward-pointer); index; jax lane log.
**Key result**: the maxtext-CE arc (v033→v037) is complete. **maxtext-CE (T5X custom_vjp) was the missing lever** — cracked the seq8192 batch wall, frontier v028 5,632 → **v035 6,030 = 86.9% of MaxText** (was 81%), bs3 fits w/o offload. v036 refuted named-offload (−18.6%; our pinned_host offload not pipelined, tags proj/mlpwi not the recomputing norms). v037 refuted maxtext-CE @ seq2048 (−6.9%; seq8192-specific, like scan). Residual ~13% to MaxText = remat-recompute (needs pipelined offload = kernel-authoring) + MXU 53.6% vs 61.2% (logical-axis, +3-4%, deferred). Honest: I'd prematurely declared a "hard wall" last arc; the user's CE directive corrected it.
**Notes**: practical jax frontier = v035 (seq8192, --use_maxtext_ce) + v018 (seq2048, plain _ce). Remaining levers are out-of-scope kernel-authoring (pipelined offload) + an uncertain sharding rewrite (MXU). Climb stops here unless the user wants the MXU lever.

## [2026-06-02] run-experiment | Qwen3-8B jax: MaxText T5X custom_vjp CE CRACKS the seq8192 batch wall — NEW frontier 34.4% / 5,992 (gap to MaxText 81%→86%)

**Op**: run-experiment (v033 bs1 parity, v034 bs2 wall-test) — user directive "use same approach as maxtext for CE".
**Pages created**: hypotheses/qwen3-jax-maxtext-ce.md; experiments v033/v034/v035.
**Pages updated**: models/qwen3-cc-jax.md (new seq8192 frontier v034, hard-wall note superseded + [!warning], Knobs CE row, variant matrix); analyses/2026-06-02-qwen3-cc-jax-maxtext-closing.md ([!warning] contradiction — hard-wall conclusion refuted); index; jax lane log.
**Key result**: MaxText's CE is the T5X `@jax.custom_vjp cross_entropy_with_logits` (full logits + one-hot + fused `softmax−onehot` backward, z_loss=0) — NOT a vocab-tiled kernel, NOT the tokamax streamed CE. Ported verbatim as `--use_maxtext_ce` (CPU bit-identical to `_ce`). **v033 bs1 = parity** (5,656 ≈ v028 5,632 — CE swap clean). **v034 bs2 = 34.4% / 5,992 — WALL CRACKED**: beats v028 bs1 (5,632, +6.4%) and v031 plain-CE bs2 (5,553, +7.9%). Batch now amortizes at seq8192 — the only change vs v031 is the CE path, so the prior "documented hard wall" (bs1>bs2>bs3) was the **tokamax/autodiff CE-backward transient**, not structural. The 2026-06-02 closing analysis's hard-wall conclusion is REFUTED. Climb toward MaxText (6,942) reopened; v035 (maxtext-CE bs3, no offload) in flight.
**Notes**: launch gotcha recorded — 2 scheduler flags use `xla_` not `xla_tpu_` (1 wasted dispatch). Honest correction: the loop declared a hard wall prematurely; the user's CE directive proved the wall was a kernel artifact.

## [2026-06-02] analyze + run-experiment | Qwen3-8B jax MaxText-gap arc CLOSED — scan+overlap won seq8192 +6.2%; full parity is a documented hard wall

**Op**: run-experiment (v031, v032) + analyze (closing synthesis).
**Pages created**: `wiki/analyses/2026-06-02-qwen3-cc-jax-maxtext-closing.md`.
**Pages updated**: v031 + v032 experiment pages (verdicts: both refuted); `models/qwen3-cc-jax.md` (seq8192 frontier v028, batch-wall + scan-seq-specificity notes, Knobs matrix populated, scan-layers hyp resolved); `index.md` (jax status line + closing analysis); jax lane log.
**Key result**: the user-directed 3-lever MaxText package (scan-over-layers + named host-offload + collective-overlap flags) **moved the seq8192 frontier +6.2%** (v009 30.4% → v028 32.3% MFU / 5,632 tok/s/chip, bs1) — semantics-preserving (HF equivalence PASS). But full MaxText seq8192 parity (45.3% / 6,942, bs3) is a **DOCUMENTED HARD WALL**: v031 (bs2 no-offload, 5,553) and v030 (bs3, 4,595) prove **batch anti-amortizes** on our stack (monotone bs1>bs2>bs3); MaxText's bs3 advantage is kernel-pipeline per-token efficiency (<0.1% host-copy pipelined offload + its own fused CE avoiding the tokamax f32 lm_head-weight all-gather), not a config lever. v032 showed scan+overlap is **seq8192-specific** (regresses −3.4% at seq2048; seq2048 stays at v018 = 97.9% MaxText parity). Per the directive "meet MaxText or document a hard wall," this is the allowed stop. Only remaining high-prior lever = a hand-written fused CE (kernel-authoring, out of scope).
**Notes**: arc v025–v032 complete. Trainer code (scan refactor + named-offload + f32-CE) lives in image qwen3-8b-jax:v030-scan-full; the scan+overlap config is the banked seq8192 frontier.

## [2026-06-02] analyze | MaxText vs jax Qwen3-8B MFU-gap teardown — gap = collective-overlap + MXU-occupancy, enabled by scan + named-offload

**Op**: analyze (code + profile teardown; user request "explore maxtext code and profile to see what gets it to 45 mfu").
**Pages created**: `wiki/analyses/2026-06-02-maxtext-vs-jax-qwen3-8b-mfu-gap.md`.
**Pages updated**: maxtext seq8192 experiment (deep profile + MXU 61.2%); scan-layers hypothesis (re-prioritized → structural enabler); index.
**Key result**: profile-analyzer teardown of the MaxText 45.3% run + source read of MaxText (`9f1820b47`). The gap is **seq-specific** — at seq2048 (both bs4) we're at PARITY (97.9%); the whole gap is seq8192 (MaxText bs3 vs our bs1). Two profile-confirmed drivers: (1) **collective overlap** — MaxText RS 4.0% async-overlapped vs our 12.9% SYNCHRONOUS (~+9pp); (2) **MXU occupancy** — 61.2% vs 48.3% (hardware counter, gap is REAL not FLOP-normalization), from fewer/larger tiled dots. Code: what we lack = **scan-over-layers** (`decoders.py:425`), **named-offload remat** (`save_and_offload_only_these_names`, decoder_layer_input on-device + projs→pinned_host — exactly the bs3 lever v026 was 2.34G short of), rich logical_axis_rules, custom-vjp CE. Confirmed NOT the gap: fused_qkv/fused_mlp (OFF for qwen3), splash blocks (match). **REVISES** prior reads: async-collective-fusion (v023 −3.5%) regressed only because it ran on the *unrolled* graph — it needs scan; and the "seq8192 structural ceiling" read is wrong — the gap is the bs1-vs-bs3 + sync-vs-async regime, closable via a 3-lever package (named-offload + scan + overlap-flags, must land together, effort L). seq2048 parity conclusion stands.

## [2026-06-02] run-experiment | MaxText Qwen3-8B v6e-8 reference @ seq2048 = 38.0% MFU / 7,116 tok/s/chip — jax at PARITY (seq2048 ceiling confirmed)

**Op**: run-experiment (MaxText reference, lane=maxtext, seq2048).
**Pages created**: experiment `.../maxtext/experiments/2026-06-02-maxtext-qwen3-8b-v6e8-ref-seq2048.md` (supported); GCS pointer.
**Pages updated**: maxtext model page (added seq2048 row + parity note); maxtext + global logs; index.
**Key result**: MaxText Qwen3-8B @ seq2048 bs4 = **38.0% MFU / 7,116 tok/s/chip** (workload ale-qwen3-8b-2-1-060214-lgr, clean). **Our jax v018 (35.8% / 6,964) is at 97.9% of it — near parity.** EXTERNALLY CONFIRMS the jax seq2048 frontier is at its practical ceiling (~2% residual within noise + MaxText's heavier offload recipe). Combined with the seq8192 finding (jax at 76.4% of MaxText), the picture is crisp: **jax seq2048 = closed at parity; jax seq8192 = ~31% headroom via offload-enabled bs3**. The MaxText reference lane baseline is now complete (seq8192 45.3% + seq2048 38.0%).

## [2026-06-02] run-experiment | MaxText Qwen3-8B v6e-8 reference: **45.3% MFU / 6,942 tok/s/chip @ seq8192** (CEILING) — reopens jax seq8192 frontier

**Op**: run-experiment (MaxText reference, lane=maxtext).
**Pages updated**: maxtext experiment page (supported, full results); maxtext model page (Current best = 45.3%); jax model page (seq8192 frontier REOPENED warning); maxtext + global logs; index. **Created**: hypothesis `qwen3-jax-host-offload-bs3-seq8192.md`; GCS pointer.
**Key result**: MaxText Qwen3-8B @ seq8192 bs3 = **45.3% MFU / 6,942 tok/s/chip / 3.540 s/step** (clean, loss 2.96→2.29), on par with llama3-8b MaxText (44.6%) → recipe transfers incl. QK-norm. **This is the achievable ceiling.** Cross-stack @ seq8192: MaxText 6,942 vs our jax v009 5,305 = **+31% tok/s/chip** — the jax lane is ~15 pp below ceiling, NOT at the true ceiling. The gap is MaxText's **offload-enabled bs3** (`decoder_layer_input` + 4 `*_proj=offload`), which the jax lane never tried (remat capped bs1; CE enabled bs2 w/o gain). **Reopens the jax seq8192 frontier** with a concrete, reference-proven lever (host-offload → bs3). Next: MaxText seq2048 reference for the direct vs-v018 (35.8%) comparison; then port the offload recipe to jax.

## [2026-06-02] manual + start | qwen3_cc maxtext reference lane bootstrapped; first run dispatched

**Op**: manual (lane bootstrap) + start (reference run).
**Pages created**: model `wiki/models/qwen3-cc-maxtext.md`; observation `wiki/observations/qwen3-maxtext-config-exists.md`; experiment stub `.../maxtext/experiments/2026-06-02-maxtext-qwen3-8b-v6e8-ref-seq8192.md`; lane log `.../maxtext/log.md`.
**Pages updated**: index (added maxtext model line); jax model page See-also.
**Key result**: Confirmed MaxText ships `configs/models/qwen3-8b.yml` matching our architecture EXACTLY (4096 hidden, 32q/8kv, hd128, 36 layers, mlp 12288, QK-norm, SwiGLU, vocab 151936, rope 1e6) — so a true MaxText reference ceiling is runnable (no published MFU exists for qwen3-8b; tpu-recipes covers only Qwen3-235B). Added benchmark configs `qwen3_8b_8192_ref` + `qwen3_8b_2048_ref` to the worktree (mirror the llama3 no-collective-matmul recipe tuning). Rebuilt `maxtext_base_image`, dispatched seq8192 run via `benchmark_runner xpk` → workload `ale-qwen3-8b-8-1-060213-9sh` (running). MFU result pending. Reference targets for context: jax 30.4% @ seq8192 / 35.8% @ seq2048, torchax 19.2%, llama3-maxtext 44.6%.

## [2026-06-02] analyze | qwen3-cc/jax retrospective #3 (FULL --full-redo audit) — confirms #2, zero discrepancies

**Op**: analyze (FULL `--full-redo` retrospective; independent full-disk re-read of all 24 experiment pages via a reader subagent, not inherited from #2's in-memory ledger).
**Pages created**: `wiki/analyses/2026-06-02-qwen3_cc-jax-retrospective-3.md`.
**Key result**: Independent audit reproduces #2 exactly — **zero discrepancies**. All 24 verdicts consistent with their own MFU numbers; frontier lineage 20.5%→22.0%→25.1%→32.4%→35.8% intact; model page Current best = v018 35.8% matches; LINT clean (all 19 non-invalid experiments have Profile sections, all 24 have Next-hypotheses, all 13 referenced hypothesis files exist, no broken links); v015 correctly absent. Two immaterial categorization refinements noted (v005 remat-vs-batch label; v020 tokamax-kernel-swap-vs-block-tuning label) — neither changes a verdict or conclusion. Frontier re-confirmed: seq2048 v018 35.8% / 6,964 tok/s/chip, seq8192 v009 30.4%. Lane at ceiling; **torchax pivot is rec #1**.

## [2026-06-02] analyze | qwen3-cc/jax retrospective #2 — v018 +3.4pp then 5 non-wins; seq2048 near practical ceiling

**Op**: analyze (INCREMENTAL lane retrospective, post-v018 arc).
**Pages created**: `wiki/analyses/2026-06-02-qwen3_cc-jax-retrospective-2.md`.
**Key result**: After the prior retrospective's #1 rec landed v018 (XLA scheduler flags, +3.4 pp → **35.8% MFU** seq2048 frontier), 5 consecutive non-wins followed (v019–v023). profile-analyzer on v018: NOT matmul-bound (MXU 48.3%, 48.6% non-matmul); #1 cost = synchronous FSDP reduce-scatter (12.9%) — but async-collective-fusion to overlap it BACKFIRED (−3.5 pp, v023: over-subscribes the already-overlap-tuned v018 scheduler). Refined law: compiler/scheduler levers transfer to qwen3 (XLA flags ✅) but kernel-swap/offload don't (tokamax-splash, SparseCore, block-tuning ❌), and a 2nd overlap mechanism is non-additive. **Verdict: seq2048 frontier near practical ceiling for non-deep-kernel levers.** Best remaining = QKV→splash GQA layout opt (data-formatting 4.5% bucket, medium prior, code-change); then scan (low prior), fused QK-norm+RoPE kernel (low, HLO-prefilter first), flag-ablation (attribution → torchax transfer). Next: pursue the layout lever via /edit-model-code.

## [2026-06-02] run-experiment | qwen3-cc/jax v018 XLA flag stack: **SUPPORTED — NEW FRONTIER 35.8% MFU** (+3.4pp)

**Op**: run-experiment (GKE/XPK, flag-only).
**Pages updated**: experiment v018 (supported); model `qwen3-cc-jax.md` (Current best → 35.8% MFU / 6,964 tok/s/chip, frontier exp → v018); hypothesis `qwen3-jax-xla-flag-stack` → supported; lane log; GCS pointer.
**Key result**: The retrospective's #1 lever — MaxText HOST_OFFLOAD XLA scheduler flag bundle (8 flags, all accepted) on the v008 seq2048 frontier — landed **35.8% MFU / 6,964 tok/s/chip** vs v008 32.4% / 6,299: **+3.4 pp / +10.6%**, step time −11.3%, loss parity, exit 0. llama3-jax's +10% transfers cleanly to qwen3 (zero semantic change). Climb 20.5%→…→**35.8%** (+74% tok/s/chip over baseline); gap to llama3-jax 43.3% now ~7.5 pp. Validates the retrospective discipline (acting on rec #1 yielded the biggest single qwen3-jax win to date). Next: v019 tests transfer to seq8192; then AMP + splash-block + flag-ablation.

## [2026-06-02] analyze + run-experiment | qwen3-cc/jax FULL retrospective + v016/v017 refuted; dispatching XLA flag stack (v018)

**Op**: analyze (lane retrospective) + run-experiment (2 verdicts processed).
**Pages created**: `wiki/analyses/2026-06-02-qwen3_cc-jax-retrospective.md`; hypotheses `qwen3-jax-xla-flag-stack.md`, `qwen3-jax-amp-mixed-precision.md`, `qwen3-jax-splash-block-tuning.md`.
**Pages updated**: experiments v016 (refuted, 29.5% < 30.4%), v017 (refuted, 31.7% < 32.4%); model page `qwen3-cc-jax.md` (open hyps 2→3, retired list += CE/SparseCore, ranked queue re-headed by XLA flag stack); hypotheses CE→refuted, SparseCore outcome appended; lane log.
**Key result**: qwen3-cc/jax frontier holds (seq2048 32.4% v008, seq8192 30.4% v009). **tokamax-CE and SparseCore-offload topics both CLOSED** (CE = memory enabler not throughput lever at either seq; SC-offload regresses at every shape — qwen3 collective share already low 6.3%). Retrospective verdict: lane YOUNG/climbing, not exhausted; top unexplored lever = **MaxText HOST_OFFLOAD XLA flag stack** (flag-only, llama3-jax's +10% single win → ~11 pp headroom to llama3-jax 43.3%). Dispatching as v018.

## [2026-06-02] run-experiment | Qwen3-8B jax (Flax NNX) v6e-8 BASELINE + cross-lane win

**Op**: run-experiment (GKE/XPK via gke-cluster-runner agent).
**Pages created**:
- [experiment: 2026-06-02-qwen3-jax-v6e8-baseline](experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-qwen3-jax-v6e8-baseline.md)
- jax model + trainer: `jax/model/{modeling_qwen3,weight_loader,sharding,__init__}.py` (Flax NNX Qwen3 w/ QK-norm; equivalence-verified vs HF — fwd max|Δ|2.7e-7, all grads ≤8e-8), `jax/{train.py,data.py,profiling.py,upload_dir.py,Dockerfile,test_equivalence.py}`.
- hypothesis stubs: [batch-scaling](hypotheses/qwen3-jax-batch-scaling.md), [splash-attention](hypotheses/qwen3-jax-splash-attention.md), [tokamax-ce](hypotheses/qwen3-jax-tokamax-ce.md)
- profile pointer: `raw/profiles/2026-06-02-qwen3-jax-v6e8-baseline/GCS_LOCATION.txt`
**Pages updated**: [qwen3-cc-jax model page](models/qwen3-cc-jax.md) (8B/v6e-8 → live, baseline + cross-lane note + ranked hyps); [index.md](index.md) (jax model status).
**Key result**: **verdict baseline** — Qwen3-8B native-JAX (Flax NNX) on v6e-8 (fsdp=8), bs1/seq2048/bf16, synthetic: **512 ms/step, 31,955 tok/s (3,994/chip), 20.5% MFU** (xprof MXU 19.9%). **Cross-lane WIN: jax beats torchax (+7.3% tok/s/chip, +1.3 pp MFU)** at the identical shape (torchax 519 ms/3,724/19.2%). Op profile shows lower collective share (21.7% vs 31.3%) — the native-JAX path's lower dispatch/collective overhead, consistent with the llama3 torchax→jax finding. Cold compile ~2 min, `EXIT_CODE=0`. Profile (2-host) + 967 HLO modules + cache in GCS; analyzed via xprof-mcp.
**Notes**: jax image built `FROM` the torchax image (already had flax/optax/transformers) + jax code only → fast build, tiny push. Image `…/torchtitan-images/qwen3-8b-jax:latest`. Flax model split Params-only for grad (RoPE inv_freq not differentiated). Trainer smoke-validated on local v6e-4 first (16.9% MFU @ seq1024). Both lanes far below compute-bound (~20% MXU) → same headroom levers (batch, splash, CE).

## [2026-06-02] run-experiment | Qwen3-8B torchax v6e-8 BASELINE

**Op**: run-experiment (GKE/XPK via gke-cluster-runner agent).
**Pages created**:
- [experiment: 2026-06-02-qwen3-torchax-v6e8-baseline](experiments/qwen3_cc_autoresearch_optimization/torchax/experiments/2026-06-02-qwen3-torchax-v6e8-baseline.md)
- hypothesis stubs: [batch-scaling](hypotheses/qwen3-torchax-batch-scaling.md), [splash-attention](hypotheses/qwen3-torchax-splash-attention.md), [tokamax-ce](hypotheses/qwen3-torchax-tokamax-ce.md)
- profile pointer: `raw/profiles/2026-06-02-qwen3-torchax-v6e8-baseline/GCS_LOCATION.txt`
- trainer infra: `torchax/Dockerfile`, `torchax/upload_dir.py` (image build context)
**Pages updated**: [qwen3-cc-torchax model page](models/qwen3-cc-torchax.md) (variant matrix 8B/v6e-8 → live, baseline cell + ranked hyps); [index.md](index.md) (model status). Trainer `train.py` fixed: `original_inv_freq` RoPE buffer now regenerated (was zero-init).
**Key result**: **verdict baseline** — Qwen3-8B on v6e-8 (2 hosts × 4 chips, fsdp=8), bs=1/seq2048/bf16, synthetic data: **519 ms/step, 29,795 tok/s (3,724/chip), 19.2% MFU** (xprof MXU util 19.4%). **66% TC idle** — under-occupancy, not compute-bound; `convolution fusion` (matmul) only 36.9% of step. Cold compile 112.5 s, `EXIT_CODE=0`. Profile (2-host xplane) + 1274 HLO files + compile cache all in GCS at `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-torchax-v6e8-baseline/`; loaded + analyzed via xprof-mcp.
**Notes**: Image `<your-registry>/torchtitan-images/qwen3-8b-torchax:latest` (digest 1ade963e), built from the adapted llama3 Dockerfile. Cluster `<your-cluster>` (<your-zone>) was 5/5 slices free. Baseline at seq 2048 not 8192 (minimal trainer OOMs at 8192 without splash/CE — that's the next-experiment frontier). Direct-to-GCS XLA HLO dump worked on jax-ai-image. Found + fixed a Qwen3 RoPE `original_inv_freq` zero-init bug (harmless for perf, semantics fix for real-data runs).

## [2026-06-02] manual | create-experiment — bootstrap Qwen3 8B family (torchax + jax lanes) + minimal torchax trainer

**Op**: create-experiment (new model family bootstrap).
**Pages created**:
- `wiki/experiments/qwen3_cc_autoresearch_optimization/program.md` — model-level program (architecture invariants incl. Qwen3 QK-norm, consolidated per-lane operational details, branch/profile conventions; conda env `py312`, hardware v6e-8).
- `wiki/experiments/qwen3_cc_autoresearch_optimization/torchax/` — **minimal baseline trainer**: `train.py` (meta-init + per-shard weights + FSDP mesh + plain cross-entropy + timing/MFU summary; deliberately no splash/scan/tokamax/AMP/remat knobs), `model/__init__.py`, `model/sharding.py` (Llama3-style FSDP/TP map + Qwen3 `q_norm`/`k_norm` entries), `data.py`, `helper.py`, `config.yaml`, `requirements.txt`, `README.md`, `experiments/.gitkeep`.
- `wiki/experiments/qwen3_cc_autoresearch_optimization/jax/` — scaffold only (`README.md`, `experiments/.gitkeep`); lane not yet implemented.
- `wiki/models/qwen3-cc-torchax.md`, `wiki/models/qwen3-cc-jax.md` — model pages with variant matrix (8B/v6e-8 open), target metrics, iteration ladder, 4 open optimization hypotheses listed (torchax).
**Pages updated**: `wiki/index.md` (Models 2 → 4; added the two qwen3 lane rows).
**Key result**: scaffolding + a runnable minimal torchax trainer for Qwen3-8B, adapted from the llama3 torchax lane but stripped to a clean baseline. Qwen3-specific handling: QK-norm RMSNorm (`q_norm`/`k_norm`) added to the sharding plan (replicated), no QKV bias, `tie_word_embeddings=False`, `rope_theta=1e6`, 36 layers.
**Notes**: user directives — folder `qwen3_cc_autoresearch_optimization`; torchax + jax lanes, start with torchax; "minimal trainer, don't reuse everything llama3 has"; hardware **v6e-8**; conda env **py312**. TODOs left in program.md: confirm `Qwen3Config` values on-box, trunk branch name, docker image base, GCS bucket. Next: capture the 8B/v6e-8 baseline (`/start-experiment qwen3_cc torchax`), then add splash/CE/scan as attributable experiments. transformers not importable in wiki base env, so architecture table needs on-box confirmation.

## [2026-05-06] manual | AOT analysis capability — concept page, HLO-dumping expansion, SCHEMA + program template updates

**Op**: manual (schema/methodology enhancement).
**Pages created**: [`wiki/concepts/aot-compilation.md`](concepts/aot-compilation.md) — full concept page covering the JAX AOT pipeline (`jit → lower → compile → cost_analysis`), HLO pre-filter workflow, pre-experiment screening, offline HLO analysis, inline `aot_screen()` utility, known results table (Llama 3 refutations, Gemma 4 ceiling, helper.py production use), caveats.
**Pages updated**:
- [`wiki/concepts/hlo-dumping-and-diffing.md`](concepts/hlo-dumping-and-diffing.md) — expanded from 24-line stub to full concept page: dump capture commands (JAX + PyTorch/XLA), file formats and naming, stage aliases table, xprof-mcp dump tools table with usage examples, key HLO patterns to grep for, fusion verification recipe, pass-by-pass vs final-only guidance, wiki conventions.
- [`SCHEMA.md`](../SCHEMA.md) — FORMULATE-HYPOTHESIS gains step 2b (HLO pre-filter, mandatory for kernel-replacement hypotheses, requires `hlo_prefilter:` frontmatter field); RUN-EXPERIMENT gains step 1b (optional AOT screening via `cost_analysis()`); LINT gains kernel-replacement `hlo_prefilter:` check.
- `sample-program.md` *(retired 2026-07-08 — see top of log)* — new `## AOT screening protocol` section with three use cases (HLO pre-filter, pre-experiment screening, offline analysis); HLO pre-filter heuristic upgraded from informal to MANDATORY with SCHEMA cross-reference; `aot-compilation` added to "Reference material at a glance".
- [`wiki/index.md`](index.md) — `aot-compilation` entry added under Compiler & HLO (12 → 13 entries, 96 → 97 concepts); `hlo-dumping-and-diffing` description updated (no longer a stub); page count 192 → 193.
**Key result**: formalizes two AOT analysis capabilities that were already in use informally: (1) HLO pre-filter for kernel-replacement hypotheses (now mandatory in SCHEMA), (2) pre-experiment AOT screening (now documented as optional step). Also documents offline HLO analysis for customer-sends-dumps workflow. No new SCHEMA operations added — AOT folds into existing FORMULATE-HYPOTHESIS and RUN-EXPERIMENT operations.
**Notes**: design decision: AOT analysis does NOT become a new operation. It integrates into existing operations where the analysis naturally occurs (hypothesis pre-filtering, pre-experiment screening, offline analysis under ANALYZE). The xprof-mcp dump tools already work on CPU without TPU; the JAX AOT pattern already exists in `helper.py`. What was missing was formalization in the schema, concept documentation, and program template.

## [2026-04-30] manual | sample-program.md — root-level template extracted from Llama 3 + Gemma 4 program.md files

**Op**: manual (template extraction).
**Pages created**: `sample-program.md` *(retired 2026-07-08 — see top of log)* (project root, ~250 lines).
**Pages updated**: [`README.md`](../README.md) — Get-started step 4 now points users at `sample-program.md` instead of asking the agent to derive a program.md from scratch; repo-layout block lists the new file.
**Key result**: the two existing program.md files (Llama 3 8B, Gemma 4 E4B) are ~95 % identical — only the model identity, architecture table, conda env name, branch prefix, libtpu version, and per-model Pallas tables differ. The template separates `<!-- GENERIC -->` sections (contract, what-you-can-cannot-do, the loop, branch model, measurement protocol, output format, observations template, heuristics, references — copy-paste verbatim) from `<!-- MODEL-SPECIFIC -->` sections (the binding values + the two Pallas-kernel tables). Includes a "How to use this template" intro block and an inline reference to the worked Llama 3 / Gemma 4 instantiations as cross-reference.
**Notes**: incorporates the schema changes from this morning — references the now-required `## Next hypotheses` (SCHEMA.md), the bottleneck reverse-index analysis, the HLO pre-filter heuristic, and the Pallas kernel directory. 41 markdown links validated, all resolve.

## [2026-04-30] analyze | Bottleneck reverse-index v0 — profile bucket → candidate levers + tried-and-refuted

**Op**: analyze (bootstrap a standing reverse-index page).
**Pages created**: [analyses/2026-04-30-bottleneck-reverse-index.md](analyses/2026-04-30-bottleneck-reverse-index.md).
**Pages updated**: [index.md](index.md) (Analyses 5 → 6, with featured-quality blurb so the agent reads it before formulating hypotheses).
**Key result**: covers 7 measured buckets (matmul, splash MHA, CE, loop-fusion, collectives, HBM peak, data-formatting) plus 4 placeholder buckets for regimes not yet seen in this wiki (megascale/DCN, embedding-grad, host overhead, VMEM spill). Each bucket section has Standing levers + Tried & refuted + Open hypotheses subsections; cross-links to the concept directory, the kernel directory, and the experiment ledger. Bootstrap data source is the [llama3-8b-torchax converged-stack bottleneck breakdown](observations/llama3-8b-torchax-converged-stack-bottleneck-breakdown.md) observation (exp 61b + exp 79 traces).
**Notes**: explicitly framed as **non-exhaustive and not a fence** — the front-matter callout says treat absence-from-this-list as "haven't tried it here yet," not "won't work." Pairs with [SCHEMA](../SCHEMA.md) `## Next hypotheses` requirement (logged below) — this page is the primary upstream consumer for that section. Future automation note recorded: when concept + hypothesis frontmatter gain a `bottleneck_bucket:` field, the page can be regenerated as a LINT subtask. Not wired up yet.

## [2026-04-30] manual | SCHEMA — `## Next hypotheses` made mandatory + structured in experiment pages; LINT extended

**Op**: manual (schema edit).
**Pages updated**: [SCHEMA.md](../SCHEMA.md) — experiment template (Next hypotheses now bolded as required, with explicit format rules: bullets link to existing `hypotheses/<slug>.md` pages, `None — <reason>` allowed, empty/absent fails LINT); RUN-EXPERIMENT op gains step 8 ("File next-hypothesis stubs") so every bullet resolves to a real page filed concurrently with the experiment, `origin: <experiment-slug>`; LINT gains two checks (missing/empty Next hypotheses, dangling `origin:` references).
**Key result**: closes the gap where follow-ups were captured in prose and forgotten. The discipline is "breadth at capture time" — three+ bullets is normal; ranking happens later. Existing experiments are not retrofitted (immutability rule); new experiments from 2026-04-30 onward must comply.
**Notes**: implements proposal #5 from the 2026-04-30 hypothesis-generation review. Proposal #1 (HLO pre-filter as required step for kernel-replacement hypotheses) and #2 (`DIFF-REFERENCE` op) remain unimplemented.

## [2026-04-27] HLO inspection | jax Llama 3 8B v6e-8 — hypotheses #2 and #3 (Pallas RMSNorm+matmul, Pallas SwiGLU+down_proj) refuted by HLO before kernel-write; XLA already does both fusions

**Op**: HLO dump + inspection (one short run with `XLA_FLAGS="--xla_dump_to=/tmp/hlo --xla_dump_hlo_as_text"`) + analysis via `mcp__xprof__list_hlo_dump_modules` and grep over `module_0262.jit_train_step.cl_854318611.after_optimizations.hlo`.
**Pages updated**: `wiki/hypotheses/llama3-jax-rmsnorm-matmul-prologue-fusion.md` (status: refuted); `wiki/hypotheses/llama3-jax-pallas-swiglu-downproj-fusion.md` (status: refuted); `wiki/index.md` (open hypotheses 3 → 1); `.gitignore` (exception added for `*-hlodump-*/`).
**Pages created**: `raw/profiles/2026-04-27-jax-hlodump-exp28b/{module_0262, module_0449}.jit_train_step.cl_854318611.after_optimizations.hlo` (~870 KB each, evidence preserved).
**Key result**:
- **Hypothesis #2 (Pallas RMSNorm + matmul-prologue fusion) — REFUTED**. The Q-proj fusion `%fusion.316 = … kind=kOutput, calls=%fused_computation.47` inlines `%fusion.301 = calls=%fused_computation.25` (which is the entire RMSNorm body: bf16→f32 cast, broadcast rsqrt scale, multiply, broadcast RMSNorm weight, multiply, f32→bf16 cast, bitcast). Then `%convolution.107` matmuls the RMSNorm output against the Q-projection weight, all within a single `kind=kOutput` Mosaic kernel. Same pattern repeats for K-proj (`fused_computation.48`), gate/up MLP projections (`fused_computation.31/.32`).
- **Hypothesis #3 (Pallas SwiGLU + down_proj fusion) — REFUTED**. The down-proj fusion `%fusion.323 = … kind=kOutput, calls=%fused_computation.40` contains: `%fusion.311 = calls=%fused_computation.8` (the full `silu(g)*u` body — negate, exp, add, divide, multiply for silu, then multiply with `u`), the `%convolution.111` down-proj matmul reading `fusion.311`'s output, and the `%add.856` residual add. **One single Mosaic `kind=kOutput` kernel** that covers exactly the work hypothesis #3 proposed to write.

**Implication**: The 9.2 % loop-fusion line in [exp 28b's profile](experiments/llama3_8B_autoresearch_optimization/jax/experiments/2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md#profile) is NOT coming from RMSNorm-to-matmul or SwiGLU-to-down_proj round-trips — XLA already consumes those in their respective `kind=kOutput` matmul fusions. The remaining loop-fusion bytes (1,547 GiB/step) come from elsewhere: residual scratchpads, embedding gradient, FSDP all-gather staging, fp32-master-weight casts. None of those are obvious Pallas-kernel targets.

**Open hypotheses count: 3 → 1**. Only int8/AQT (#1, +15-30 % expected, biggest single lever) remains as a meaningful per-chip throughput lever.

**Win for the autoresearch methodology**: a 30-minute HLO-inspection + my own filed Risks section in hypothesis #3 ("pallas-forge TPU SwiGLU loses 35 % to XLA on v5e — validate via HLO before kernel write") saved weeks of kernel-writing effort by refuting both hypotheses cheaply, before committing to a Pallas project.

## [2026-04-27] run-experiment | jax Llama 3 8B v6e-8 — 100-step loss-curve validation: optimization stack is bit-equivalent to baseline (max |Δ| = 0.0003)

**Op**: run-experiment (× 5: jax-exp 65/66/67/68/69 — 100-step loss validation across 3 stack configurations + 2 LR-matched MaxText comparisons).
**Pages created**: `wiki/experiments/.../jax/experiments/2026-04-27-jax-exp65-67-loss-validation-100steps.md`. Raw 100-step loss trajectories preserved at `raw/profiles/2026-04-27-jax-exp65-67-loss-validation/exp{65..69}.txt` (gitignore exception added for `*-loss-validation/`).
**Pages updated**: `wiki/experiments/.../jax/data.py` reading + script changes (`exp_jax_maxtext_flags.sh` adds `USE_REAL_DATA` / `LR` env-var pass-through; new `exp_jax_minimal.sh` for the minimal-flags baseline) — code changes already shipped in image `precast-1`.
**Key result**: **All three optimization layers (MaxText XLA stack + SC offload, tokamax-splash kernel choice, tokamax-splash perf knobs) are loss-clean over 100 training steps to within bf16 precision floor.** Three independent 100-step runs at identical RNG seed:
- exp 65 (full optimized stack at 7,601 tok/s/chip 42.6 % MFU)
- exp 66 (minimal-flags + jax-experimental splash at 6,336 tok/s/chip 35.5 % MFU)
- exp 67 (minimal-flags + tokamax-splash defaults at 6,509 tok/s/chip 36.5 % MFU)

Aggregate stats over 100 steps:
- exp 65 vs exp 66 (full opt vs jax-experimental ref): max\|Δ\| = 0.0003, median Δ = +0.0000, 86/100 steps within 0.0001
- exp 65 vs exp 67 (perf-knobs ON vs OFF): max\|Δ\| = 0.0003
- exp 67 vs exp 66 (kernel impl difference): max\|Δ\| = 0.0002

Loss values are around 11.9 (synthetic random data), bf16 precision floor ≈ 0.09 absolute. **All observed deltas are 300× below the bf16 noise floor** — equivalent to pure XLA scheduling-order rounding noise.

**Throughput-with-equivalent-loss**: optimization stack delivers **+19.9 % per-chip throughput** over the most-pristine baseline with **zero measurable loss-curve drift**.

**MaxText comparison flagged**: MaxText baseline's reported synthetic-data loss collapse (12.26 → 1.79 over 19 steps) does not match our 11.93 → 11.81 over 100 steps even at matched lr=3e-5 (exp 68/69). Root cause: our `data.py:fake_dataloader` draws fresh random tokens every batch (model cannot memorize), MaxText's synthetic dataset re-uses fixed sequences (rapid memorization). Data-pipeline difference, not numerics. Both stacks compute identical forward+backward on the data they see.

**Notes**: Initial 100-step attempts on real wikitext-2 data terminated at step 9 ("data exhausted"); switched to synthetic for the validation. Synthetic random data also more rigorous for numerics — tightest possible bf16-noise comparison.

## [2026-04-27] formulate + run-experiment | jax Llama 3 8B v6e-8 — 22-experiment exhaustive ablation around exp 28b; frontier saturated at ~7,700/chip 43.3 % MFU

**Op**: formulate (3 deep-work hypotheses) + run-experiment (× 22: jax-exp 39-60 ablation around the SparseCore-offload frontier).
**Pages created**:
- `wiki/hypotheses/llama3-jax-rmsnorm-matmul-prologue-fusion.md`
- `wiki/hypotheses/llama3-jax-pallas-swiglu-downproj-fusion.md`
- `wiki/hypotheses/llama3-jax-int8-weight-quantization.md`
**Pages updated**: `wiki/experiments/.../jax/experiments/2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md` (added Wave 4-5 ablation table + noise-band correction); image `precast-1` shipped with the bf16 pre-cast env-var knob.
**Key result**: **No further knob improves on exp 28b.** Three independent re-runs measure the noise band as **±0.7 %** (±50 tok/s/chip); the honest frontier is **~7,700 tok/s/chip / 43.3 % MFU** (mean), not the 7,768/43.6 % single-run peak. vs MaxText 7,069/chip = **+8.9 % per chip** (mean), still beating their reference.
**Ablation summary (22 runs in five waves, all refuted or within noise)**:
- Remat policies: `save_out_proj` OOM at bs=4 / -0.6 % at bs=3; `save_qkv_proj` OOM bs=4; `qkv_proj_offloaded` -0.8 to -0.9 % at bs=4-5.
- Kernel/density: scan unroll=2 -1.9 %; SPLASH_BQ=4096 -21 %; SPLASH_BKV=512 -2.2 %; SPLASH_BKV_DKV=1024 -3.3 %; bs=3 -2.7 % vs bs=4 frontier.
- VMEM: 65,536 -4.2 %; 81,920 -1.6 %; 131,072 -2.9 % (all worse than 98,304).
- XLA flags (added or flipped): disable_bundle_aware_cost_model -1.7 %; enhanced_launch_barrier -1.1 %; async_collective_permute within noise; megacore_fusion_allow_ags=false within noise; combo of the two within-noise -1.1 %; collective matmul ENABLED -14.7 % (hard refute); overlap_compute_collective_tc=false -1.1 %; aggressive_opt_barrier_removal=DISABLED within noise; latency_hiding_scheduler_rerun=0 within noise; loop-invariant chain DISABLED -0.7 %; tokamax CE = mosaic_tpu -4.4 %.
- Code change: pre-cast bf16 weights once per train_step → -0.5 to -1.1 % (XLA already fuses cast into matmul prologue; pre-cast adds an HBM round-trip rather than removing one).
**Deep-work hypotheses filed**: (1) Pallas RMSNorm + matmul-prologue fusion (expected +3-6 %, effort L); (2) Pallas SwiGLU + down_proj fusion (expected +2-4 %, effort L; **but pallas-forge's TPU SwiGLU loses 35 % to XLA on v5e — XLA may already be doing this fusion. Validate with HLO dump first.**); (3) int8 / AQT or qwix weight-only quantization (expected +15-30 %, effort L). All three include a SwiGLU-research-agent finding: best scaffolding is `tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py:77-85,1476-1482` which already fuses `silu(gate)*up` with two grouped matmuls in a single TPU Pallas kernel.
**Verdict**: bf16-MXU optimization regime is saturated. Further wins require either custom Pallas kernels (eat the 9.2 % loop-fusion line) or int8 quantization (break the 65.8 % bf16-MXU ceiling).

## [2026-04-26] run-experiment | jax Llama 3 8B v6e-8 — exp 29-38 post-frontier knob sweep; exp 28b holds

**Op**: run-experiment (× 8: jax-exp 29-38 follow-ups to the SparseCore-offload frontier).
**Pages updated**: `wiki/experiments/.../jax/experiments/2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md` (extended Follow-up sweeps table); `wiki/experiments/.../jax/{train.py, model/modeling_llama3.py}` (named-checkpoint markers + `_resolve_scan_policy` helper); built docker image tag `saveqkv-1` carrying the new code.
**Key result**: **All seven post-frontier knobs refuted.** exp 28b (7,768/chip 43.6 % MFU at bs=4 + bkv=1024 + full SC offload) holds as the durable frontier. The TC is now 99.986 % busy and recomputation during bwd is "free" because TC has slack while collectives are SC-offloaded — so any policy that saves activations to HBM (`save_qkv_proj`) blows compile peak, and any policy that saves to host (`qkv_proj_offloaded`) costs more PCIe traffic than it saves in avoided matmul recompute.
**Refutations summary**:
- exp 29: VMEM=131072 (vs 98304) → 7,546/chip 42.3 % (-2.9 %); same direction as torchax exp 77.
- exp 30: bkv=2048 (vs 1024) → 7,752/chip 43.5 % (-0.2 % within noise); the +0.7 % bkv-2048 lift from exp 18 does **not** compound once SC offload is on.
- exp 31: bs=3 (MaxText shape) → 7,559/chip 42.4 %; density check confirms we still beat MaxText 7,069/chip by **+6.9 %** at their exact shape.
- exp 32: SPLASH_BQ=4096 → 6,122/chip 34.3 % (**-21.2 %**); VMEM spill from larger query blocks.
- exp 35: `save_qkv_proj` (named QKV save) → OOM at compile by +5.67 GiB at bs=4; saving Q/K/V across 32 scanned layers exceeds HBM peak.
- exp 36: `qkv_proj_offloaded` bs=4 → 7,641/chip 42.8 % (-1.6 %); host PCIe latency > recompute savings.
- exp 37: `qkv_proj_offloaded` bs=6 → runtime OOM; host offload doesn't shrink the runtime workspace floor.
- exp 38: `qkv_proj_offloaded` bs=5 → 7,634/chip 42.8 % (-1.7 %); same conclusion at higher density.
**Code change shipped** (image tag `saveqkv-1`): `jax.ad_checkpoint.checkpoint_name` markers around all seven projections (query/key/value/out/mlpwi_0/mlpwi_1/mlpwo) in `_decoder_call`, plus a `_resolve_scan_policy` helper that recognises MaxText's policy names (`save_qkv_proj`, `save_out_proj`, `save_dot_except_mlp`, `qkv_proj_offloaded`, `minimal_offloaded`). These knobs are now available for future work without further code changes.
**Notes**: One docker tag-cache hiccup (kubelet cached an old `jax-v6` digest with the same content as the prior `jax-v5`; resolved by pushing under a fresh tag `saveqkv-1`).

## [2026-04-26] run-experiment | jax Llama 3 8B v6e-8 — exp 27/28b SparseCore RS+AG offload + bs=4 → NEW FRONTIER 7,768/chip 43.6 % MFU

**Op**: run-experiment (jax-exp 26 profile + jax-exp 27 bs=5 + jax-exp 28 bs=6 + jax-exp 28b bs=4 + jax-exp 28pf profile rerun).
**Pages created**: [`wiki/experiments/.../jax/experiments/2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md`](experiments/llama3_8B_autoresearch_optimization/jax/experiments/2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md).
**Pages updated**: `wiki/index.md` (Models entry, Experiments list, headline count); `wiki/experiments/.../jax/experiments/README.md` (Current best table now points to exp 28b).
**Key result**: **NEW PROGRAM-TARGET BEST 7,768 tok/s/chip, 43.6 % MFU at bs=4 seq=8192** — beats prior frontier (exp 18 at 7,471/chip 41.9 %) by **+4.0 % per chip**, beats MaxText reference (7,069/chip 44.6 %) by **+9.9 % per chip**. Reported MFU gap to MaxText narrowed from 2.7 pp → 1.0 pp.
**The lever**: exp 26 profile of bs=5 frontier showed `async-all-reduce-scatter` at **5.0 % of step time** still on the TensorCore. Adding `xla_tpu_enable_sparse_core_collective_offload_{reduce_scatter,all_gather}=true` (matching MaxText's full SC offload triplet — we previously had only the all-reduce variant) relays both remaining FSDP collectives onto the SparseCore. **+3.4 % per chip at bs=5** (exp 27: 7,724/chip 43.3 %); **+0.6 % more at bs=4** density (exp 28b: 7,768/chip 43.6 %). bs=6 OOMs by ~220 MiB — bkv=2048 splash floor raises memory pressure beyond what density alone can compensate.
**Loss**: identical step-for-step to prior frontiers (11.90 → 10.10 across 9 steps at bs=4) — no semantic regression.
**Cumulative day climb**: torchax exp 20 4,591 → jax exp 28b 7,768 = **+69.2 % per-chip**.
**Notes**: User's standing directive ("keep working on it until you match or exceed maxtext performance, do not stop") was already satisfied at exp 13 (+3.9 %); we have continued pushing through three further frontier advances (exp 18 → exp 27 → exp 28b). Profile capture rerun (exp 28pf) submitted with full SC offload + sleep-hold for trace retrieval.

## [2026-04-26] run-experiment + profile | torchax Llama 3 8B v6e-8 — exp 75-81 final sweep at the exp 74b frontier; converged

**Op**: run-experiment (× 11: exp 75-81 follow-ups) + profile capture (exp 79) + observation update.
**Pages updated**: `wiki/experiments/.../2026-04-26-exp72a-tokamax-splash-bs3-seq8k-accepted.md` (extended §"Refuted" with all post-acceptance refutations); `wiki/observations/llama3-8b-torchax-converged-stack-bottleneck-breakdown.md` (added §"Updated breakdown 2026-04-26 latest frontier").
**Profile artifact**: `gs://<your-bucket>/jax-experiment/llama3-8b-exp79-tk-prof/` (kubectl-cp'd from pod `/tmp` then uploaded — the gcsfuse-mount /data/* paths return 403 on profile writes from inside the pod, so the workaround is: write profile to /tmp, post-run `tar czf` + `sleep 600` to keep pod alive, then `kubectl cp` from outside).
**Key result**: Frontier holds at exp 74b — **6,559 tok/s/chip, 36.8 % MFU at bs=3 seq=8192**. Every post-acceptance knob refuted or invalid.
**Refutations summary (all 2026-04-26)**:
- exp 75a: `cost_estimate_flops_fwd/bwd` — neutral; XLA already overlaps without hints.
- exp 76a/b: `q_seq_shards=2`/`q_seq_shards=4` — **INVALID** (NaN loss from step 1; q_seq_shards needs a context-parallel mesh axis we don't have).
- exp 77a/b: VMEM=131,072 / 65,536 KiB — both -2.7 % / -3.1 %; VMEM=98 KiB still optimal.
- exp 78a/b: `scan_remat_policy=dots_with_no_batch_dims_saveable` / `dots_saveable` — both OOM (matmul-output saves don't fit even at bs=2).
- exp 80a/b/c: splash block-size variants (bkv=512, bq=1024, symmetric 2048/2048) — all -0.1 to -0.4 %; current splash 2048/1024/2048/2048 fused config still optimal.
- exp 81a/b: `JAX_DEFAULT_MATMUL_PRECISION=bfloat16`/`tensorfloat32` — within noise / -0.5 %; `default` already does bf16 on TPU efficiently.
**Profile breakdown at frontier** (single-host trace, exp 79):
- Conv fusion (matmul): **54.4 %** of step time, HBM-BW util 0.54, **73.9 % MXU efficiency when running**
- Splash MHA fwd+dkv: 21.8 %
- Loop fusion (RMSNorm/silu/residual): 15.2 %, HBM-BW util 0.75 (saturated)
- Tokamax CE: ~0 % (collapsed from 7.5 % via chunked_xla)
- MXU utilization 55.0 % (was 51.9 % at exp 61b)
**Cumulative day climb**: 4,591 → 6,559 tok/s/chip = **+42.9 % per-chip**. Gap to MaxText 44.6 % reference: 7.8 pp. Closing this gap would require a **custom Pallas matmul prologue** that fuses the bf16 weight materialization into the matmul (skip the HBM round-trip) and/or **MaxText-style layer fusion** patterns. Those are deep-work levers not reachable by knob-sweeping the existing call sites.

## [2026-04-26] run-experiment | torchax Llama 3 8B v6e-8 — exp 74b tokamax-splash + max_logit_const advances frontier to 36.8 % MFU bs=3 seq=8192

**Op**: run-experiment (× 7: exp 74a-f mlc sweep; exp 75a cost_estimate; exp 76a/b q_seq_shards INVALID).
**Pages updated**: `wiki/experiments/.../2026-04-26-exp72a-tokamax-splash-bs3-seq8k-accepted.md` (added §"Follow-up: + max_logit_const" + §"Refuted post-acceptance" + §"Final (valid) frontier"); `wiki/index.md` (program-target best advanced to 6,559/chip 36.8 %; cumulative climb 42.9 %).
**Docker images**: hf-v36 (max_logit_const knob) → hf-v37 (cost_estimate knobs) → hf-v38 (q_seq_shards env knob).
**Key result**: **🏆 exp 74b: tokamax-splash + base2 + fuse_recip + max_logit_const=30 @ bs=3 seq=8192 = 52,475 TPS (6,559/chip), 36.8 % MFU** — **+2.6 % per-chip** over exp 72a (6,392 / 35.8 %). Cumulative day climb: 4,591 → 6,559 = **+42.9 %**. Gap to MaxText: 7.8 pp.
**Notes**:
- mlc value insensitive (10 / 20 / 30 / 50 all within ±0.1 %); the win is just turning the optimization path on. Llama attention logits stay well below 10 in practice (post-`/sqrt(d_head)` scaling), so the kernel never clamps. Loss values **identical step-for-step** to exp 72a — the kernel skips a redundant softmax stabilization pass when the cap is provided. MaxText's `use_max_logit_estimate` defaults to `-1` (off) — they don't enable this for Llama either, so we found a knob they haven't enabled by default.
- `cost_estimate_flops_fwd/bwd` (exp 75a): neutral. XLA already overlaps the splash kernel with FSDP collectives without the hint.
- **INVALIDATED: exp 76a/b** (`q_seq_shards=2`/`q_seq_shards=4`): NaN loss from step 1 onward. q_seq_shards is meant for sequence sharding across a context-parallel mesh axis; we have no such axis. The throughput "win" (+10 %) is an artifact of NaN values short-circuiting downstream ops. Rejected.

## [2026-04-26] run-experiment | torchax Llama 3 8B v6e-8 — exp 72a tokamax-shipped splash NEW PROGRAM-TARGET BEST 35.8 % MFU bs=3 seq=8192

**Op**: run-experiment (× 7: exp 71a/b — broken mask API; exp 72a-d valid sweep; exp 73a-c ablations).
**Pages created**: `wiki/experiments/.../2026-04-26-exp72a-tokamax-splash-bs3-seq8k-accepted.md`.
**Pages updated**: `wiki/index.md` (program-target best advanced; experiment count 16 → 17).
**Docker images**: hf-v33 → hf-v34 (fix tokamax mask API: single mask, not MultiHeadMask) → hf-v35 (add `dq_reduction_steps` env knob).
**Key result**: **🏆 exp 72a: tokamax-shipped splash attention with `use_base2_exp=True` + `fuse_reciprocal=True` @ bs=3 seq=8192 = 51,139 TPS (6,392/chip), 35.8 % MFU** — **+1.3 % per-chip** over the prior exp 65 frontier (jax-splash, 6,313/chip 35.4 %). Holds at +1.8 % at bs=2 and +1.2 % at bs=4. Cumulative day climb: 4,591 → 6,392 = **+39.2 % per-chip** vs morning AMP-only baseline (exp 20). Gap to MaxText 44.6 %: now 8.8 pp (was 9.2 pp).
**Notes**:
- Discovered by reading MaxText's `src/maxtext/layers/attention_op.py` at the `use_tokamax_splash` branch — they expose tokamax's own splash impl alongside upstream JAX's, with extra perf knobs (`use_base2_exp`, `fuse_reciprocal`, `use_experimental_scheduler`, `dq_reduction_steps`, `max_logit_const`). The first two compound super-additively.
- **Ablations (exp 73a-c)**: `base2=T fuse_recip=F` = +0.7 %; `base2=F fuse_recip=T` = +0.8 %; both = +1.3 %. `use_experimental_scheduler=T` = -0.2 % (worse); `dq_reduction_steps=3` = +0.5 % (also worse than defaults). Optimal config = `base2=T + fuse_recip=T` only.
- **Mechanism**: `use_base2_exp` swaps `exp(x)` for `exp2(x / ln 2)` — TPU's `exp2` is faster, the rescale fuses with the upstream multiply, mathematically identical. `fuse_reciprocal` moves the `output / lse` division inside the kernel so we save one HBM round-trip on the post-attention output buffer per layer.
- **API note**: tokamax splash takes a single `mask` (broadcasts to heads internally), not `MultiHeadMask`. Trip-up cost one rebuild (hf-v33 → hf-v34).

## [2026-04-26] run-experiment + INVALIDATION | torchax Llama 3 8B v6e-8 — splash sweep (refuted) + exp 66 series invalidated; valid frontier reverts to exp 65 at 35.4 % MFU

**Op**: run-experiment (× 8: exp 67a-d splash sweep on broken stack; exp 68a-d splash sweep on valid stack) + invalidation entry.
**Pages updated**: `wiki/experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-26-exp62b-chunkedxla-ce-bs3-seq8k-accepted.md` (added invalidation note for exp 66 series; appended exp 67/68 sweep table); `wiki/index.md` (program-target best reverted to exp 65 with explanatory note); `wiki/log.md` (this entry).
**Docker images**: hf-v31 (splash env-var config; built on top of broken hf-v30), hf-v32 (reinstates fp32 cast for chunked_xla; correct stack).
**Key result**: **🏆 valid PROGRAM-TARGET BEST = exp 65: 50,502 TPS (6,313/chip), 35.4 % MFU at bs=3 seq=8192**. Cumulative day climb: 4,591 → 6,313 = +37.5 % per-chip vs morning AMP-only baseline (exp 20).
**Notes**:
- **Invalidation: exp 66 / 66b / 66c / 66d** (bf16-native chunked_xla — skipped the `astype(jnp.float32)` cast on the theory that chunked_xla returns grads in input dtype natively). Threw away ~+0.3 % per-chip as wins for ~30 minutes. Found wrong: chunked_xla kernel sets `dtype = x.dtype` and uses that for lse / loss_sum **accumulators** ([chunked_xla.py lines 88, 119, 127, 136, 137, 156](trainer/tokamax_lib/.../linear_softmax_cross_entropy_loss/chunked_xla.py)). With bf16 input, the inner-loop accumulators are bf16 — at magnitude ~11 (cross-entropy initial) bf16 quantum is ~0.04, accumulating across 64 V-blocks compounds. Loss output empirically plateaued at multiples of 0.0625 (`loss=11.1250 → 11.0000`) vs exp 65's clean monotone decay (`11.7681 → 11.6403` over 12 steps). Per project rule "no model-quality optimizations", invalid. Reverted in hf-v32.
- **Splash sweep (exp 68 series)** on the valid stack confirms current config (bq=2048 bkv=1024 bq_dkv=2048 bkv_dkv=2048 fused) is optimal: bq=4096 -2.8 %, symmetric 4096/4096 -5.2 %, unfused bwd -4.5 %. The exp 8/9/10 spring autotune universal winner still holds.
- Splash dkv bwd remains the biggest custom-call cost (10.7 % of step time per the [bottleneck observation](wiki/observations/llama3-8b-torchax-converged-stack-bottleneck-breakdown.md)). Remaining lever points: pull MaxText's attention bwd kernel (deep work); custom V-sharded CE that avoids the lm_head all-gather (custom kernel work); not many easy local wins left.

## [2026-04-26] run-experiment | torchax Llama 3 8B v6e-8 — exp 65 → 66b refinement, FRONTIER ADVANCED to 35.5 % MFU bs=3 seq=8192

**Op**: run-experiment (× 6: exp 64–66d).
**Pages updated**: `wiki/experiments/.../2026-04-26-exp62b-chunkedxla-ce-bs3-seq8k-accepted.md` (added §"Follow-ups" with 7-row table); `wiki/index.md` (program-target best advanced; experiment count entry updated to read exp 62b → 66b path).
**Docker images**: hf-v29 (chunked_xla without shard_map — refuted) → hf-v30 (chunked_xla bf16 native, no fp32 cast, shard_map kept).
**Key result**: **🏆 exp 66b: chunked_xla + bf16 native + autotune @ bs=3 seq=8192 = 50,647 TPS (6,331/chip), 35.5 % MFU** — +0.4 % over exp 62b, +2.1 % per-chip over the prior mosaic_tpu+autotune frontier (exp 56), and **+37.8 % per-chip over the morning AMP-only baseline (exp 20)**.
**Notes**:
- exp 64/64b (chunked_xla *without* shard_map): -36 % per-chip; JAX auto-partition picks a far worse pattern. shard_map remains essential even for the XLA-only impl. Refuted.
- exp 65 (autotune): +0.16 % over exp 62b. Autotune marginal but positive on chunked_xla's b/v block sizes.
- exp 66 / 66b: skipping the fp32 input cast (the cast was only added to handle mosaic_tpu's hardcoded fp32 grad output — chunked_xla returns grads in input dtype natively) yields +0.3 %. Free win once the analysis was done.
- Final stack: `--weights_dtype=fp32 --compute_dtype=bf16 --master_dtype=fp32 --use_splash=True --use_scan=True --use_tokamax_ce=True --tokamax_ce_impl=chunked_xla --tokamax_ce_autotune=True --batch_size=3 --seqlen=8192` plus the bf16-native-cast guard inside the trainer (set when `tokamax_ce_impl=="chunked_xla"`).
- Profile capture for chunked_xla still pending — both attempts (exp 63, exp 62) hit GCS HTTP 403 mid-run on profile-write to `gs://<your-bucket>/{autoresearch,cache}/profiles/`. The pod's workload identity has read but not write IAM on those prefixes. TODO: either grant the role, or use a different bucket prefix (XLA cache writes to `gs://.../cache/xla/` succeed, so the bucket is OK — only the `/profiles/` subpath is wrong).

## [2026-04-26] run-experiment | torchax Llama 3 8B v6e-8 — exp 62b chunked_xla CE NEW PROGRAM-TARGET BEST 35.3 % MFU bs=3 seq=8192

**Op**: run-experiment (× 5: exp 62 + 62b/c/d/e). Triggered directly by the xprof bottleneck observation.
**Pages created**: `wiki/experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-26-exp62b-chunkedxla-ce-bs3-seq8k-accepted.md`.
**Pages updated**: `wiki/index.md` (program-target best advanced; experiment count 15 → 16).
**Docker images**: hf-v28 — adds `--tokamax_ce_impl` flag (default `mosaic_tpu`, new option `chunked_xla`).
**Key result**: **🏆 exp 62b: chunked_xla CE @ bs=3 seq=8192 = 50,424 TPS (6,303/chip), 35.3 % MFU**. +1.6 % per-chip over the prior exp 56 frontier (6,202 / 34.8 %). The same swap also lifts bs=2 to 6,214/chip 34.8 % (+2.7 %) and bs=4 to 6,161/chip 34.5 % (+2.8 %). bs=5 OOMs by 2.91 GiB — bs=4 remains the density ceiling.
**Notes**:
- The xprof breakdown predicted this win: tokamax CE bwd was 263 ms (6.6 %) at 21 % MXU efficiency, asymmetrically expensive vs the 36 ms fwd. The `chunked_xla` impl uses standard XLA matmuls and avoids the Pallas-recompute path. End-to-end gain landed at +1.6 % at the program-target shape; expected, validated, accepted.
- `chunked_xla` keeps one `(B*L, V_chunk)` slice in HBM during fwd+bwd. Fits within bs=3 envelope at seq=8192; bs=5 doesn't fit (FFN intermediate hits ceiling, unrelated to CE impl).
- Profile-write to GCS in exp 62 hit a 403 mid-run (HTTP permission). Exp 62b/c/d/e ran without profile capture and were clean. Profile bring-up TODO: get the right IAM role for the pod's workload identity to write to `gs://<your-bucket>/autoresearch/profiles/`.

## [2026-04-26] xprof-bottleneck-breakdown | torchax Llama 3 8B v6e-8 — converged-stack profile breakdown

**Op**: record-observation (xprof).
**Pages created**: `wiki/observations/llama3-8b-torchax-converged-stack-bottleneck-breakdown.md`.
**Profile**: `gs://<your-bucket>/autoresearch/profiles/llama3-8b-exp61b-profile-bs3/` (captured by exp 61b, a 9-step short run with profile_step=5; the autotune kernel selection is identical to exp 56's frontier, so this trace is representative of the program-target stack).
**Key result**: per-chip per-step time at the converged stack = **3,964 ms total**; matmul (conv fusion) **46.0 %**, splash MHA fwd+dkv bwd **24.4 %**, tokamax CE fwd+bwd **7.5 %**, loop fusion (RMSNorm/silu/etc.) **13.6 %**, data formatting 3.6 %, misc 5 %. **MXU util 51.9 %, MFU 34.8 %** — gap to MaxText 44.6 % is non-matmul time + matmul-tile-utilization headroom inside conv fusion (HBM-BW util 0.54 there).
**Notes**:
- Tokamax CE bwd is asymmetrically expensive (263 ms ≈ 7 × the 36 ms fwd) — the mosaic-TPU bwd recomputes logit blocks; only ~21 % MXU-efficient by back-of-envelope FLOPs accounting. Worth testing `implementation="chunked_xla"` next.
- Splash dkv bwd (423 ms = 10.7 %) is the single biggest custom-call cost; pulling in MaxText's attention bwd kernel for comparison is the deepest leverage point.
- Loop fusion (RMSNorm + silu + residual + mul) already at 0.77 HBM-BW util — saturated; no throughput to recover at that layer.

## [2026-04-26] run-experiment-batch-program-target | torchax Llama 3 8B v6e-8 — exp 51 → 55b (tokamax CE + fp32 master, NEW PROGRAM-TARGET BEST 34.6 % MFU bs=3 seq=8192)

**Op**: run-experiment (× many: exp 48 → exp 56) + tokamax-CE bring-up (5 image revs) + true-AMP-master implementation + MFU-formula bug fix.
**Pages created**: `wiki/experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-26-exp55b-fp32-master-tokamax-ce-bs3-seq8k-accepted.md` (this milestone).
**Pages updated**: `wiki/index.md` (program-target status updated; experiment count 13 → 14).
**Docker images**: hf-v18 → hf-v26 (each rev fixes one bring-up issue: skip_lm_head, absl-flags pre-mark, fp32 cast at boundary, shard_map wrap, compute_dtype, MFU divisor).
**Key result**: **🏆 exp 55b: bs=3 seq=8192 with scan + splash + tokamax CE (shard_map'd) + fp32 master + bf16 compute = 49,427 TPS (6,178/chip), 34.6 % MFU.** +35 % per-chip TPS over exp 20 (prior program-target best at bs=1 31.6 %). The win is the tokamax mosaic-TPU `linear_softmax_cross_entropy_loss` kernel: it streams logsumexp over V blocks in VMEM, avoiding a materialized `(B*L, V)` logits tensor (~6 GiB savings), which is what unlocks bs=3 (was OOM by 6 GiB in exp 44). fp32 master is essentially free at runtime (cast fused into matmul).
**Notes**:
- Tokamax CE bring-up notes (5 distinct issues, see milestone page table) — most-load-bearing was the absl-flags-vs-fire-args clash and the kernel always returning fp32 grads.
- **MFU formula bug**: `avg_step_time = total_time_after_warmup / (train_steps - warmup_steps)` divides by *planned* step count, but the wikitext loader exits early at high B → MFU inflated by `13/n_actual`. Numbers reported during the day for exp 51b (45.0 %), exp 51c (64.1 %), exp 53b (62.4 %) were artifacts of this bug. Corrected re-runs in exp 55-55c. Fix (hf-v26): `n_measured_steps` counter, divide by that.
- True AMP master pattern: `--weights_dtype=fp32 --compute_dtype=bf16`. weights stored fp32; `_maybe_cast_weights` autocasts each weight to bf16 before `functional_call`; the cast-vjp downcasts the bf16 grad back to fp32 on the way out, so the optimizer sees fp32 grads matching its fp32 mu/nu. Strict canonical AMP — no rounding into bf16 on parameter updates.
- bs=3 is the throughput sweet spot at seq=8192 — bs=2 / bs=4 both come in at ~6,000/chip (same MFU within ±1 pp), bs=3 hits 6,178/chip.
- **Post-acceptance follow-ups all refuted** (see milestone page §"Follow-ups tested"): (a) recipe XLA flags neutral/-0.3 %; (b) TP=2 / FSDP=4 -14 % on v6e-8; (c) VMEM=131,072 -0.7 %; (d) `dots_saveable` and `dots_with_no_batch_dims_saveable` remat policies OOM by 42 GiB at the scan body. The frontier converged at `nothing_saveable` + autotune + tokamax CE @ bs=3 seq=8192 = 6,206/chip, 34.8 % MFU. Next direction = xprof-driven profiling.
- Out-of-band TODO: copy the xprof trace tarball off `/tmp/llama3_profile/` on the GKE pod into `raw/profiles/2026-04-26-exp55b-fp32-master-tokamax/` before the post-run `sleep 600` expires.

## [2026-04-25] run-experiment-batch-2 | torchax Llama 3 8B v6e-8 — exp 9-11 (autotune ACCEPTED 36.1% MFU; remat REJECTED)

**Op**: run-experiment (×4: exp 9 + exp 10 + exp 11a/b/c) + protocol-extension (kernel-autotune-first for new shapes).
**Pages created**: `wiki/experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-25-exp{9..11}-*.md` (4 pages); `raw/profiles/2026-04-25-llama3-8b-exp9-splash-autotuned/` + GCS mirror.
**Pages updated**: `wiki/index.md` (Experiments 10 → 13; llama3-8b model status → 36.1 % MFU; gap to MaxText 8.5 pp); `wiki/experiments/llama3_8B_autoresearch_optimization/torchax/{splash_attn.py, train.py}` — autotuned `BlockSizes` + remat patch on respective branches.
**Branches created** (2): `v6e8-llama3-8b-torchax-20260425-{exp9-splash-autotuned-bs4, exp11-remat}`. (exp 10 used the exp 8 image without code change; ran on the exp 9 branch.)
**Docker images**: `:hf-v3` (autotuned splash) and `:hf-v4` (`+ make_train_step with remat re-enabled`).
**Key result**:
- 🏆 **exp 9 ACCEPTED at 36.1 % MFU (7,225 TPS/chip, 57,799 TPS aggregate)** — exp 8's kernel-only autotune (+30 % kernel fwd+bwd) translates to +1.1 % TPS / +0.4 pp MFU end-to-end. Predicted +1.5–3.5 %; landed at low end (~half the kernel-saving converts because exp 3 was already partially overlapping attention with collectives). New program target.
- exp 10 (kernel autotune at seq=2048/4096/8192): same universal winner pattern. At seq=8192, `block_q=2048 block_kv=1024 fused_bwd=True dkv=(2048,2048)` = 7.944 ms / layer / chip → 254 ms attention budget for a seq=8192 step — feasible if memory fits.
- **exp 11 REJECTED at all 3 shapes**: 11a (bs=4 seq=1024) is 52 % slower than exp 9 (MFU 24.0 vs 36.1); 11b (bs=4 seq=2048) and 11c (bs=8 seq=1024) **still compile-OOM by 7+ GiB** — only 50 MiB less than exp 4 / 6 (no remat). `nothing_saveable` doesn't fix the OOM-driver, which is *not* activations.

**Discovery**: `torchax.train.make_train_step`'s `remat_policy` arg is **silently ignored** — `loss = interop.gradient_checkpoint(loss, ...)` is commented out in canonical `raw/code/torchax/torchax/train.py:54`. **All baseline + exp 1-10 ran with no remat.** exp 11 re-enables it via local `_make_train_step`. Filing as a torchax upstream issue is queued.

**Process retrofit**: per user direction, established the **kernel-autotune-first protocol for shape changes**: any new training shape gets a 3-min kernel-only autotune (via `tune_splash.py`) before the full training run is submitted. Documented in [exp 10](experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-25-exp10-splash-autotune-multishape-potential.md).

**Notes / next**: Path to fp32-master + seq=8192 program target requires (a) memory unlock — scan-over-layers (gemma4-jax exp 49-51 confirms the architectural pattern); (b) selective remat (`dots_saveable` instead of `nothing_saveable` — avoid the 52 % step-time tax we measured); (c) tokamax `LinearSoftmaxCrossEntropyLoss` — Llama 3 has no softcap, so the gemma4 jax-exp43 blocker does NOT apply; ~128 MiB / chip activation savings; (d) host-offload of mu/nu — frees ~4 GiB / chip on the way to fp32 master. Ordered by EV: scan first (the unlock), then CE, then remat policy refinement, then fp32.

## [2026-04-25] run-experiment + methodology | torchax Llama 3 8B exp 8 — splash kernel-only autotune (POTENTIAL +30% kernel fwd+bwd at every shape; 3 min wall-clock)

**Op**: run-experiment + methodology-bootstrap (first kernel-only experiment in this wiki).
**Pages created**:
- `wiki/experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-25-exp8-splash-kernel-autotune-potential.md` — full writeup.
- `raw/profiles/2026-04-25-exp8-splash-kernel-autotune/` — `rank0.log`, `rank1.log`, `results.csv` (171 configs × 14 cols, in-tree, 110 KiB).
- `wiki/experiments/llama3_8B_autoresearch_optimization/torchax/tune_splash.py` — reusable kernel-only autotune harness (~430 LOC, no extra deps beyond what's in the trainer image).
**Pages updated**:
- `wiki/index.md` — Experiments 9 → 10; bumped page count; updated llama3 model-status line with the new finding + queued exp 9 validation.
- `wiki/experiments/llama3_8B_autoresearch_optimization/torchax/experiments/README.md` — replaced stale gemma4-copy with llama3-specific contents + linked exp 8 prominently.
**Branches created**: none (kernel-only experiment; no production code change yet — the validated change happens in exp 9).
**Docker images**: `<your-registry>/test/llama3-8b-torchax-container:tune-v1` (initial; had a clamping bug in the production-default anchor at seq=1024 — `block_q_dkv=2048 > seq_len=1024`); `:tune-v2` (fixed: clamp anchors with `min(global, seq_len)` to mirror production's clamping). Both layered on top of the `hf-v2` trainer image with only the new `tune_splash.py` baked in.

**Key result**: First **kernel-only** experiment in the program. Sweeps splash `BlockSizes × q_layout × use_fused_bwd_kernel` for the 3 in-program shapes (seq=1024 bs=2 baseline; seq=1024 bs=4 exp 3; seq=2048 bs=2 exp 5), single TPU v6e chip, 171 configs total. **Universal winner across all 3 shapes:** `block_q=1024, block_kv=1024, q_layout=HEAD_DIM_MINOR, use_fused_bwd_kernel=True, dkv=(seq_len, seq_len)`, beating the production default in [`splash_attn.py`](experiments/llama3_8B_autoresearch_optimization/torchax/splash_attn.py) by **+30.1 % to +32.4 % on fwd+bwd kernel time**, **+14.9 % to +16.4 % on fwd alone**. Production has `block_kv=512` (asymmetric, stale carryover from an older MaxText recipe) and `use_fused_bwd_kernel=False` — both wrong for these shapes. The autotune-best matches the current MaxText Llama 3.1-8B recipe within 0.2 %, independently confirming that recipe applies cleanly here. End-to-end TPS impact bounded by attention's share of step time: at seq=1024 ≈8 % share ⇒ expected +2.4 % step time ⇒ +2.4 % TPS; at seq=2048 ≈12 % share ⇒ expected +3.5 % TPS. Validation queued as exp 9 (full XPK training run with the autotune-best config).

**Methodological finding (the bigger one)**: end-to-end wall-clock for the 3-shape 171-config sweep was **3 minutes**. The equivalent set of full XPK training runs to test the same hypotheses one-by-one would be ~45-90 min wall-clock (15 min/run × 3-6 representative configs per shape × 3 shapes), and would resolve sub-2 % deltas as noise. Cross-host validation (rank 0 vs rank 1 timed independently on different chips of the same v6e-8 slice) showed **0.04-0.21 % agreement on best-config ms** — measurement noise floor far below the 30 % gain. Pattern to emulate: 1) build focused harness using upstream JAX kernels directly (rather than reusing tokamax + tune_jax — porting risk for direct applicability + 2 extra deps); 2) gate phase 2 on top-k from phase 1 to bound the search; 3) anchor every sweep with the production default + a public reference recipe so deltas are interpretable.

**Decision audit — build vs reuse**:
- [tokamax `splash_attention_benchmarking.py`](codebases/tokamax.md) — has a working harness with grid + xprof-event-filtered timing via `tune_jax`; rejected because tokamax has a *vendored* fork of splash that may differ from the upstream JAX path our model uses, and `tune_jax` is an extra pip dep. Borrowed the grid layout idea + `randn_init` pattern.
- [marin/levanter autotune harness](codebases/marin.md) — *in-graph* autotune cache for kernel selection at training time; wrong shape for this task (we want a standalone benchmark). Listed as the index-headline pattern to emulate for *future* in-trainer autotune (TBD when scope warrants).
- [maxtext splash kernel](codebases/maxtext.md) — has a frozen production splash wrapper but no autotune harness; consulted only for the recipe values used as the second anchor.

**Next hypotheses generated** (queued in the experiment page):
1. **Exp 9 — full-training validation of the autotune-best splash config** on top of exp 5 (seq=2048 trunk). One-line code change in [`splash_attn.py:42-53`](experiments/llama3_8B_autoresearch_optimization/torchax/splash_attn.py). Highest-priority follow-up. Effort: S.
2. **Exp 10 — autotune at seq=8192** (program target shape). Same harness, `--seq_len 8192 --batch_size 1`. At seq=8192 attention is ~25 % of step time, so 30 % kernel win ⇒ +7-8 % TPS. Effort: S.
3. **Exp 11 — combine autotune-best splash + bs=4 / seq=8192** (after exp 9). Test whether `use_fused_bwd_kernel=True` frees activation memory enough to unlock previously-OOM batch/seq combos.
4. **Phase-3 dq decoupling** — only if any future shape has a competitive `fused_bwd=False` row. In this run every non-fused config was strictly dominated. Defer.
5. **Promote `tune_splash.py` to a generic Pallas kernel autotune harness** for future kernels (RMSNorm Pallas re-attempt, fused softcap+CE, etc). Captured as a methodology line in the experiment README.

**Notes** — bug found in iteration 1 (image `tune-v1`): the `production_default_full()` config in the harness used the literal `block_q_dkv=2048` value from `splash_attn.py`, but production *clamps* with `min(2048, seq_len)` at runtime. At `seq_len=1024` the harness threw `q_block_size=2048 should divide q_seq_len=1024`. Fix in `tune-v2`: thread `seq_len` into anchor constructors and clamp consistently. **Process retrofit**: when porting production code into a benchmark harness, include the clamp/min logic that ships with the production wrapper, not just the literal constants. Workload `llama3-splash-tune-20260425-191044` (`tune-v2`) succeeded end-to-end in 100 s + image pull/init. The earlier `llama3-splash-tune-20260425-190746` (`tune-v1`) was deleted after seq=1024 sweep failed at the bug above; ~2 min lost.

---

## [2026-04-25] run-experiment-batch | torchax Llama 3 8B v6e-8 — 7 experiments, splash+bs=4 ACCEPTED (35.7% MFU, +12.8 pp)

**Op**: run-experiment (×7) + formulate-hypothesis (×1) + protocol-update (program.md branch convention).
**Pages created**: `wiki/experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-25-exp{1..7}-*.md` (7 experiment pages); `wiki/hypotheses/llama3-torchax-xla-recipe-flags.md` (hypothesis page); `raw/profiles/2026-04-25-llama3-8b-exp3-splash-bs4/` + `gs://<your-bucket>/autoresearch/2026-04-25-llama3-8b-exp3-splash-bs4/` (winner profile, both ranks); `raw/profiles/2026-04-25-llama3-8b-exp5-splash-seq2k/` + GCS mirror.
**Pages updated**: `wiki/index.md` (Experiments 4 → 9; llama3-8b model status → 35.7 % MFU); `wiki/experiments/llama3_8B_autoresearch_optimization/program.md` (added "Experiment branch" binding row — convention `v6e8-llama3-8b-torchax-<YYYYMMDD>-exp<NN>-<short-slug>`); `wiki/experiments/llama3_8B_autoresearch_optimization/torchax/train.py` (added `--use_splash` flag + `env.override_op_definition(F.scaled_dot_product_attention, …)` canonical splash wiring on the splash-using branches).
**Branches created** (8): `v6e8-llama3-8b-torchax-20260425-{baseline, exp1-xla-recipe-flags, exp2-splash-bs2, exp3-splash-bs4, exp4-splash-bs8, exp5-splash-seq2k, exp6-splash-bs4-seq2k, exp7-splash-xla-bs4}`.
**Docker images**: `<your-registry>/test/llama3-8b-torchax-container:hf-v1` (baseline + exp1; no splash) and `:hf-v2` (exp2-exp7; splash override).
**Key result**: **exp 3 (splash + bs=4) ACCEPTED** — 57,154 TPS aggregate (7,144/chip), 35.7 % MFU at `bs=4 seq=1024 fsdp=8` on v6e-8. **+55.6 % TPS, +12.8 pp MFU vs baseline.** Mechanism: splash makes attention activation `O(L)` which frees enough HBM that `bs=4` fits (it OOM'd by 1 GiB on the no-splash baseline); the doubled per-chip token count amortizes non-attention compute across 2× the work, lifting per-token MFU. Splash itself at `bs=2` (exp 2) is +1.6 % within noise — it is the precondition, not the wing.

**Ranking by verdict**:
- 🏆 exp 3 splash+bs=4 = **35.7 % MFU** — accepted (current target).
- exp 5 splash+seq=2048 = 34.0 % MFU — accepted (alternate path to seq=8192).
- exp 7 splash+xla-flags = 35.0 % MFU — refuted (flags don't help at this density).
- exp 2 splash@baseline = 23.3 % MFU — potential (precondition only).
- exp 6 splash+bs=4+seq=2048 = compile OOM — invalid.
- exp 4 splash+bs=8 = compile OOM 7.36 GiB — invalid.
- exp 1 xla-flags@baseline = 22.6 % MFU — refuted.

**Notes**: New durable heuristic captured: **MaxText recipe XLA flags require per-chip B·L ≳ ~10,000 to pay for themselves** (recipe is bs=3 seq=8192 = 24,576; we are at 4,096). At lower B·L, FSDP all-gather is already fully hidden behind compute and the flags are no-ops. Compile time penalty is +6–17 s. Don't enable speculatively until compute density justifies.

**OOM mechanism — bs=8 and bs=4-seq=2048 both crash** because splash compresses attention activations (`O(L²)` → `O(L)`) but not the FFN intermediate `[B, L, 14336]` which is the largest bf16 activation in the train graph. Both OOMs converge on the same path forward: **selective remat that recomputes the FFN intermediate** (`save_only_these_names = {hidden_state}`-style), or **scan-over-layers** to share the FFN intermediate buffer across all 32 layers (also ~32× compile-time reduction). Queued as next experiments.

**Compile cache write permission denied** persists on the gcsfuse `/data/cache/xla` mount — every run cold-compiles 70-110 s. Investigated to gcsfuse mount config (UID mismatch — pod cannot write the bucket). First fix attempt: switch to `/tmp` + post-run sync script (queued).

**Process retrofit**: per user direction, established the **per-experiment branch convention**: `v6e8-llama3-8b-torchax-<YYYYMMDD>-exp<NN>-<short-slug>` (or `…-baseline`). Each branch holds the exact `train.py` + `model/` + `data.py` state used for that run plus the experiment page; even env-only experiments get a branch. Documented in `program.md`. Backfilled all 7 experiments (today's batch) onto their respective branches.

**Gap to MaxText reference (44.6 % MFU)**: 8.9 pp remaining. Most likely sources (in order): (a) compute density — MaxText recipe is bs=3 seq=8192, B·L = 24,576 vs our 4,096; selective remat / scan + bs=4 seq=8192 would close most of this. (b) framework overhead — JittableModule + torchax interop vs MaxText's pure-JAX hand-tuned graph; +1–3 pp at most. (c) kernel choice / scheduler — minor.



**Op**: run-experiment.
**Pages created**: `wiki/experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-25-baseline.md`; `raw/profiles/2026-04-25-llama3-8b-baseline/` (mirror of GCS, 668 MiB, both ranks); `gs://<your-bucket>/autoresearch/2026-04-25-llama3-8b-baseline/`.
**Pages updated**: `wiki/index.md` (Models entry for llama3-8b updated with torchax baseline; Experiments count 2→3; total page count bumped); `wiki/experiments/llama3_8B_autoresearch_optimization/torchax/{train.py, Dockerfile, DEPLOY.md, model/, data.py, helper.py, splash_attn.py}` — full restructure of the trainer code per the program README requirements (HF `LlamaForCausalLM` + `model/` re-export package + `data.py` wikitext loader + canonical-pattern `train.py` from `raw/code/torchax/examples/train_llama_torchtitan/` + Dockerfile + generic GKE-deployment guide).
**Key result**: First end-to-end run of the HuggingFace-`transformers`-based Llama 3 8B torchax trainer on multi-host GKE. Local v6e-4 single-host (5,450 tok/s/chip, MFU 27.2 %) and v6e-8 multi-host **(canonical: 36,729 tok/s = 4,591 tok/s/chip, MFU 22.9 % at bs=2 seq=1024 fsdp=8)** both stable at steady state. Multi-host −16 % per-chip vs local — the FSDP collective tax across the 2-host slice is the first optimization target. MFU formula switched from rough `6N · tokens` to the MaxText `calculate_tflops_training_per_device` per-component formula (qkv + o_proj + ffn + lm_head + causal-attn, ×3 for fwd+2·bwd), which drops the embedding-table flops since lookup is gather not matmul. Verdict: `supported`.
**Notes**: Significant infrastructure work — from-scratch rewrite of the trainer using the canonical torchax pattern (`sharded_device_put`, `create_sharded_weights` with `make_array_from_callback`, `JittableModule`, `torchax.train.make_train_step`, `helper.compile_step_func` with `donate_argnums + cost analysis`, `axis_types=(Auto, Auto)` mesh). The torchtitan-shape canonical does not work on HF Llama directly without three fixes documented in the experiment page: (1) **buffer materialization** for HF's `model.rotary_emb.{inv_freq, original_inv_freq}` (meta-init leaves them unmaterialized; `interop.jax_view` crashes); (2) **opt_state count-scalar replication** — optax `adamw`'s `count` 0-d int32 lands on device-0 only and collides with `compile_step_func`'s out_shardings; walk the pytree post-init and `jax.device_put`-replicate any leaf whose device set is smaller than the mesh; (3) **batch=4 OOM** at compile (4 GiB over on v6e-4, 1.05 GiB over on v6e-8), so canonical baseline is bs=2. **Compile cache write permission denied** on the gcsfuse `/data/cache/xla` mount (PermissionError — pod's UID can't write the bucket); JAX falls through but no warm-cache reuse, so every run cold-compiles ~92 s — first follow-up hypothesis. New stack folder `wiki/experiments/llama3_8B_autoresearch_optimization/torchax/` already existed (it housed the earlier torchtitan-based trainer); this run replaces all torchax code top-to-bottom with the HF-based version. Image rebuilt + pushed: `<your-registry>/test/llama3-8b-torchax-container:hf-v1`. Next priority: hypothesis #1 (compile-cache fix → 82 s/run saved); then splash attention (#2); then batch=4 (#3) and async-collective flags (#4). 7 follow-up hypotheses listed inline in the experiment page; convert to standalone hypothesis pages before scheduling.

## [2026-04-25] run-experiment | MaxText Gemma 4 E4B v6e-8 baseline (SUPPORTED-as-approximation, 282.9 TFLOP/s/device, 30.8% MFU)

**Op**: run-experiment.
**Pages created**: `wiki/experiments/gemma4_autoresearch_optimization/maxtext/{README.md, experiments/README.md, experiments/2026-04-25-maxtext-gemma4-e4b-v6e8-baseline.md}`; `raw/profiles/2026-04-25-maxtext-gemma4-e4b-v6e8-baseline/` (mirror of GCS, ~133 MiB; canonical at `gs://<your-bucket>/maxtext/v6e8-20260425-02-gemma4-e4b/<workload-name>/`); `/mnt/disks/persist/maxtext-main/src/maxtext/configs/models/gemma4-e4b.yml` (NEW wiki-local model config — no upstream equivalent exists).
**Pages updated**: `wiki/experiments/gemma4_autoresearch_optimization/README.md` (added maxtext stack pointer); `/mnt/disks/persist/maxtext-main/src/maxtext/configs/types.py:239` (1-line literal-whitelist patch — required so Pydantic accepts `model_name=gemma4-e4b`).
**Key result**: First-ever MaxText Gemma 4 E4B-shape run on Trillium v6e-8: **282.9 TFLOP/s/device, 10,003 Tokens/s/device, 30.8% per-chip MFU** at `bs=2, seq=8192, fsdp=8, remat_policy=full`, aggregate ~80k tok/s across 8 chips. Compile fits at 22.51 GiB / 31.25 GiB HBM. **APPROXIMATION** — HF E4B's `num_kv_shared_layers=18` (last 18 layers reuse K/V from earlier same-type layers) is not implemented by MaxText's `gemma4` decoder_block; the run has +47M extra k/v projection params (~0.6% over true E4B). Throughput is a measurement of the dense-shape model that this config builds, not directly comparable to a true E4B implementation.
**Notes**: Substantial infrastructure work — Gemma 4 support landed on MaxText `main` after `tpu-recipes-v0.1.4` (we ran Llama-8B against v0.1.4 earlier today; for E4B we had to switch to `main` @ `532c8b3d8` and rebuild `maxtext_base_image` from scratch (1.77 GiB, py3.12 base, ~5 min build incl. 2-min layer export)). Five workload submissions: -mt-1 hit Pydantic literal_error (model_name not whitelisted), -mt-2 same error (PYTHONPATH issue — base image had editable install at `/deps/src` that overrode xpk's `/app/src` COPY; fixed with `PYTHONPATH=/app/src` in user_command), -mt-3 hit "Unknown command line flag '2a886c8_chip_config_name'" (newer libtpu rejected one of the Llama-recipe LIBTPU_INIT_ARGS flags; stripped to just `--xla_tpu_scoped_vmem_limit_kib=98304`), -mt-4 OOM at compile (`bs=4 seq=8192` with `remat=full` doesn't fit — FFN intermediate `bf16[4,8192,10240]` × ~20 buffers = 13+ GiB live), -mt-5 succeeded at `bs=2`. **MFU is 13.8 pp lower than the same-day Llama-3.1-8B baseline (44.6%)** — drivers: smaller compute density, `remat=full` overhead, missing recipe-tuned XLA flags. Item 1 in next-hypotheses queue is to port the LIBTPU_INIT_ARGS set (minus version-rejected flag). Hybrid attention pattern (5 SW + 1 GLOBAL × 7 cycles) drives both more compile time (~50 s HLO compile vs Llama's ~12 s) and a ~4× larger xplane trace (133 vs 31 MiB) since each cycle compiles distinct kernels. Loss flat (12.97-12.98 over 20 steps, synthetic data — no real signal at this scale; ln(262144) = 12.477 is the uniform-vocab baseline; the slight excess reflects synthetic-data variance not a broken model). Workload `<workload-name>` cleaned up after run.

## [2026-04-25] run-experiment | MaxText Llama3.1-8B v6e-8 reference baseline (SUPPORTED, 409.4 TFLOP/s/device, 44.6% MFU)

**Op**: run-experiment.
**Pages created**: `wiki/experiments/llama3_8B_autoresearch_optimization/maxtext/README.md`; `wiki/experiments/llama3_8B_autoresearch_optimization/maxtext/experiments/README.md`; `wiki/experiments/llama3_8B_autoresearch_optimization/maxtext/experiments/2026-04-25-maxtext-llama3-1-8b-v6e8-baseline.md`; `raw/profiles/2026-04-25-maxtext-llama3-1-8b-v6e8-baseline/` (mirror of `gs://<your-bucket>/maxtext/v6e8-20260425-01-llama3-1-8b/<workload-name>/`).
**Pages updated**: `wiki/experiments/llama3_8B_autoresearch_optimization/README.md` (added maxtext stack pointer with MFU number).
**Key result**: Wired the MaxText stack to the autoresearch v6e-8 GKE cluster `<your-v6e8-cluster>` (which `program-gke.md` had marked TODO across all three GKE clusters) and ran the official `tpu-recipes-v0.1.4` Llama3.1-8B v6e-8 recipe (`llama3_1_8b_8192_no_collective_matmul`, fsdp=8 seq=8192 bs=3) end-to-end through xpk + `benchmark_runner xpk`. Steady-state median over steps 11–14 + 16–19: **409.42 TFLOP/s/device, 7,069.66 Tokens/s/device, 44.6 % MFU** (per-chip vs Trillium 918 TFLOP/s bf16 peak); aggregate 56.6k tok/s across 8 chips. Reproduces recipe README's published 413.4 / 7,138.9 within **−1.0 %** on both metrics — well inside run-to-run noise. Verdict: `supported`.
**Notes**: Infrastructure decisions worth carrying forward: (a) MaxText worktree at `tpu-recipes-v0.1.4` placed at `/mnt/disks/persist/maxtext-tpu-recipes-v0.1.4` per SCHEMA rule 1 (raw/code/maxtext stays on main); (b) local maxtext venv at `/mnt/disks/persist/venv-maxtext-v0.1.4` (uv + py3.12), `MAXTEXT_README`'s `uv pip install -e .[tpu] --resolution=lowest` path doesn't work as written (absl-py 0.1.0 build failure under py3.12; tensorboardx version conflict in jax-stable-stack-0.6.1 reqs file) — the working recipe is `uv pip install -e .[tpu] --prerelease=allow` then patch in `omegaconf grain ruamel.yaml jaxtyping aqtp ml-goodput-measurement google-cloud-aiplatform google-cloud-monitoring google-jetstream cloud-tpu-diagnostics`; (c) `maxtext_base_image` is **1.81 GiB** (much smaller than the torchtitan image, ~11 GiB) so docker build + push is fast (2–3 min total); (d) xpk needs to be cloned to `~/xpk` exactly (path required by `XPK_README.md` and benchmark_runner default `xpk_path`). HBM peak was 10.66 GiB / 31.25 GiB (34 %) — large headroom for `bs=4` and projection-offload ablations as next hypotheses. Loss trajectory monotone-decreasing (12.264 → 1.792); model is training. xprof URL: `http://localhost:8791/?run=2026-04-25-maxtext-llama3-1-8b-v6e8-baseline` (after `xprof --logdir=raw/profiles --port=8791`). New stack folder `wiki/experiments/llama3_8B_autoresearch_optimization/maxtext/` distinguishes "reference baseline (don't optimize, just reproduce)" runs from the optimization-target `torchax/` and `jax/` stacks; mirrors the gemma4 program's per-stack split.

## [2026-04-25] ingest-codebase | tpu-recipes (AI-Hypercomputer/tpu-recipes @ e284e361)

**Op**: ingest-codebase.
**Pages created**: `wiki/codebases/tpu-recipes.md`.
**Pages updated**: `wiki/index.md` (Codebases count 26→27; new entry under Wave 4 ecosystem); `README.md` (new bullet under "Reference trainers & inference engines"); `.gitmodules` (new submodule entry).
**Submodule added**: `raw/code/tpu-recipes` → `https://github.com/AI-Hypercomputer/tpu-recipes.git` @ commit `e284e3613721882ce3e15c533a76c691443ea60f`.
**Key result**: AI-Hypercomputer's curated per-(model, hardware, topology) reproduction recipes on Cloud TPU. Each recipe pins a MaxText tag (e.g. `tpu-recipes-v0.1.4`), a `jax-stable-stack` Docker image (e.g. `jax0.6.1-rev1`), the launch command, and the **MaxText `tuning_params` block** used to hit the published throughput target. Coverage: Trillium (v6e-32/64/128/256) + Ironwood (v7x 4×4×4 / 4×8×8) for Llama 3.1 (8B/70B/405B), Gemma 3-12B, Gemma 4-26B/31B, Mixtral 8×7B / 8×22B, DeepSeek 3-671B, Qwen 3-235B, GPT-OSS 120B, Wan 2.1-14B, GPT-3 175B; legacy v5p tier. Plus `microbenchmarks/` (matmul TFLOPS + HBM bandwidth on v6e-1) and `utils/profile_convert.py` (`.xplane.pb` → text).
**Notes**: Single most direct public source for "what flags Google's perf team picked for this combination." Closest external precedent for the gemma4 program here: the `Gemma3-12B-MaxText/v6e-*` recipes (family + scale) and the published `Llama3.1-70B-MaxText/v6e-32` `tuning_params` block (`remat_policy: custom`, `decoder_layer_input: offload`, `query_proj/key_proj/value_proj: offload`, `per_device_batch_size: 2`). Future hypotheses touching `remat_policy` or per-projection offload should cite the closest matching recipe as the prior. Microbenchmark harness usable to establish v6e-1 chip-level rooflines (matmul ~827 TFLOPS bf16 8192³ with `xla_tpu_scoped_vmem_limit_kib=65536`; HBM ~1359 GB/s 32 MiB copy) — useful when an op-profile roofline classification is in dispute. No subpages created (recipe `READMEs` are themselves the leaves; no perf-relevant subsystems warrant their own wiki page).

## [2026-04-24] analyze + run-experiment | jax-exp53: splash block-size sweep under new regime (REJECTED, flat) + new-regime ceiling analysis

**Op**: run-experiment + analyze.
**Pages created**: `wiki/experiments/gemma4_autoresearch_optimization/jax/experiments/2026-04-24-exp53-jax-splash-block-sweep-fp32master-rejected.md`; `wiki/analyses/2026-04-24-gemma4-jax-fp32master-seq8k-regime.md` (new-regime ceiling analysis); `raw/profiles/2026-04-24-gemma4-jax-exp53-splash-block512-seq2k-fp32master/` + `gs://<your-bucket>/autoresearch/2026-04-24-gemma4-jax-exp53-splash-block512-seq2k-fp32master/`.
**Pages updated**: `.../jax/experiments/RESULTS.tsv` (exp53 discard row); `.../jax/experiments/OBSERVATIONS.md` (exp 53 block + ceiling pointer).
**Key result**: **exp 53 REJECTED** — splash block-size sweep at seq=2048 b=1 fp32-master is flat. Block=2048 (full-tile) hits `CompileTimeScopedVmemOom` (32.14/32.00 MiB, 144 KiB over). Block=512 runs at 26,807 TPS, dead flat vs exp 52's default block=1024 (−0.0 %). Confirms the old-regime exp 48 plateau transfers to the new regime. **Ceiling analysis filed** covering seq=8192 infeasibility on v6e-4 (independent of AMP — legacy bf16 also OOMs), XLA non-monotonic compile-time peak, `nothing_saveable`/offload counter-intuitive regression, and the three-branch forward path (A: optimize at seq=2048; B: v6e-8 mesh; C: memory-saving code changes).
**Notes**: New durable heuristic for program.md: "Splash block size is flat across all measured Gemma 4 E4B shapes on v6e-4 with the fused_bwd kernel + SEQ_MINOR layout — don't open a new block-size experiment unless the shape or kernel changes materially." Still-open exp 52 follow-ups queued in the new-regime ceiling page: exp 54 (pure-AMP isolation at b=1 s=1024), exp 55 (scan_layers at new regime), exp 56 (2D mesh dp=2 tp=2), exp 57 (PLE embedding host-offload). xprof browser URL for exp 53: http://localhost:8791/?run=2026-04-24-gemma4-jax-exp53-splash-block512-seq2k-fp32master.

## [2026-04-24] run-experiment | jax-exp52: fp32-master + bf16-compute AMP new-regime baseline (seq=8192 OOMs on v6e-4; seq=2048 b=1 accepted at 26,807 TPS)

**Op**: run-experiment (regime change + new-baseline establishment).
**Pages created**: `wiki/experiments/gemma4_autoresearch_optimization/jax/experiments/2026-04-24-exp52-jax-fp32master-seq2k-accepted.md`; `raw/profiles/2026-04-24-gemma4-jax-exp52-baseline-seq2k-fp32master/` (on-disk xprof profile, gitignored) + `gs://<your-bucket>/autoresearch/2026-04-24-gemma4-jax-exp52-baseline-seq2k-fp32master/` (GCS mirror).
**Pages updated**: `.../jax/experiments/RESULTS.tsv` (exp52 keep row); `.../jax/experiments/OBSERVATIONS.md` (exp 52 block); `.../jax/experiments/README.md` (new "Current state" section split between old-regime and new-regime baselines); `.../jax/train.py` (new `--weights-dtype` / `--compute-dtype` CLI, seq_len default 2048 → 8192, `JAX_REMAT_POLICY` env var, reordered apply_sharding before load_hf_weights); `.../jax/model/modeling_gemma4.py` (all modules accept split weights_dtype / compute_dtype kwargs; Linear downcasts weight to compute_dtype at matmul; embed_tokens casts output to compute_dtype; layer_scalar casts to hidden dtype); `.../jax/model/weight_loader.py` (optional `shardings=` scatter-shards fp32 directly into sharded NNX params, avoiding a 10.5 GiB fp32 PLE on-device-0 init OOM); `.../jax/model/scan_layers.py` (_matmul_amp helper + layer_scalar-cast for AMP correctness); `.../torchax/train.py` (flag parity stubs; warns + falls back to --dtype on split); commits 517a689 + 176fd2c + flag-wiring squash.
**Key result**: **KEEP as new-regime baseline**. User asked for `seq_len=8192 b=1 fp32-master + bf16-compute` and the optimization loop to run there. seq=8192 b=1 fp32-master is **INFEASIBLE on v6e-4**: compile OOM at 35.18 GiB / 31.25 GiB per-chip HBM (exceeded by 3.93 GiB). Even legacy bf16-only at seq=8192 b=1 OOMs (36.16 GiB, exceeded 4.91 GiB). Intermediate seq_lens are WORSE than seq=8192 due to XLA scheduling non-monotonicity (seq=4096 = 39.58 GiB; seq=6144 = 49.66 GiB). Largest-feasible new-regime config: **seq=2048 b=1 fp32-master = 26,807 TPS, 305.6 ms/step**, loss descent 3.25 → 2.30 clean. −15.7 % vs exp 40 (bf16-only b=2 s=2048 at 31,809 TPS; b=2 is itself infeasible under fp32 master — 39.37 GiB OOM). Exp 36 (bf16 single-dtype, s=1024 b=3 at 34,614 TPS) remains the **old-regime** JAX best; exp 52 is the new-regime baseline.
**Notes**: AMP implementation is JAX-only; torchax gets CLI-parity stubs that warn + fall back to legacy `--dtype` (HF PyTorch takes a single torch_dtype; full torchax wiring is out of scope). Flag wiring: modules accept kwargs (weights_dtype for storage, compute_dtype for matmul); Linear does `x @ W.astype(compute_dtype).T` at call time — single fp32→bf16 cast per forward, XLA folds into the dot when it can. NO hand-casts beyond that needed (no need for `jax.lax.dot_general` with explicit dtype promotion). `nothing_saveable` and `offload_dot_with_no_batch_dims` remat policies **increase** compile-time peak HBM (39.66 and 38.17 GiB at seq=8192 respectively, vs default's 35.18 GiB) — XLA planner does not credit offload as HBM-freed; nothing_saveable serializes more live tensors to avoid recomputing. JAX_SCAN_LAYERS=1 saves 0.35 GiB at fp32-master seq=8192 and 3.7 GiB at bf16-legacy seq=8192, not enough to close the gap. Weight-loader refactor: `shardings={path: NamedSharding}` kwarg scatter-shards each tensor through a host numpy buffer → per-device HBM directly, so the fp32 PLE embedding (11 GiB) never materializes on device 0; needed because NNX inline init runs on device 0 by default. Apply_sharding moved before load_hf_weights so the shardings lookup is available. Init-in-bf16 workaround: even though weights_dtype=fp32, the random-init pass is in bf16 to fit device 0; the loader overwrites with fp32 sharded values; a `_fixup_dtype_meta` pass retargets per-module weights_dtype attributes for introspection. Follow-up queue documented in experiment page: (1) pure-AMP isolation at b=1 s=1024, (2) splash block sweep at seq=2048 fp32-master, (3) scan_layers at seq=2048 fp32-master, (4) 2D mesh dp=2,tp=2 at seq=8192 (K/V replication on tp=2 permitted since num_kv_heads=2), (5) PLE embedding host-offload (risky code change). xprof_mcp access is currently pointing at a different GCS logdir; direct URL for the uploaded profile: http://localhost:8791/?run=2026-04-24-gemma4-jax-exp52-baseline-seq2k-fp32master (resolves once server is re-pointed at the autoresearch bucket).

## [2026-04-24] run-experiment | jax-exp47: marin/levanter fused linear+softcap+CE Pallas kernel — REJECTED (−5.61 % TPS, custom-call + all-gather tax)

**Op**: run-experiment.
**Pages created**: `wiki/experiments/gemma4_autoresearch_optimization/jax/experiments/2026-04-24-exp47-jax-levanter-ce-rejected.md`; `wiki/experiments/gemma4_autoresearch_optimization/jax/model/kernels/__init__.py` and `.../kernels/fused_ce/__init__.py` (import shim); `wiki/experiments/gemma4_autoresearch_optimization/jax/tools/parity_levanter_ce.py` (correctness harness); `raw/profiles/2026-04-24-gemma4-jax-exp47-levanter-ce/` (xprof profile, gitignored).
**Pages updated**: `.../jax/experiments/RESULTS.tsv` (exp47 discard row), `.../jax/experiments/OBSERVATIONS.md` (exp 47 block), `.../jax/train.py` (`JAX_CE_IMPL=levanter` branch + shard_map wrapper + sharded levanter call), `.../jax/model/modeling_gemma4.py` (`Gemma4ForCausalLM.__call__(return_hidden=True)` kwarg + `lm_head_weight()` helper).
**Key result**: **REJECTED**. TPS 34,614 → **32,671 (−5.61 %)**; step time 355.0 → **376.1 ms (+5.95 %)**; parity **PASS** (|diff| 0.048 vs tol 0.05 in bf16); smoke step-4 loss 2.1875 → 2.1979 (+0.47 %, within 5 % bar). The softcap gap from exp 43 is closed — levanter's `fused_cross_entropy_loss_and_logsumexp_penalty` applies `sc * tanh(logits/sc)` inline on each VMEM logits-tile before the streaming softmax (`raw/code/marin/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/reference.py:11`), so `[B, S, V]` logits never materialize in HBM. Regression root cause is the same Pallas-custom-call tax as torchax exp 33 (Pallas RMSNorm, −36 ms/step): CE in exp 36 was <3 % of step time and XLA-fused with lm_head + softcap + log_softmax; replacing a tight 10-ms XLA fusion with a 3.5-ms Pallas kernel adds ~15 ms of custom-call boundary + the 1.31-GiB `w_hv` all-gather that the shard_map wrapper forces (lm_head weight is FSDP-sharded on V, kernel requires replicated `[H, V]`). Exp 36 remains JAX-stack best at 34,614 TPS.
**Notes**: Pre-run gates done per SCHEMA run-experiment op: parity harness before smoke, smoke before benchmark, profile captured during benchmark. Block sizes hand-picked (`b=1024, h=256, v=512`): default `(1024, 512, 1024)` overruns 32 MiB VMEM on v6e; `b_block` must be ≥1024 because per-shard flat batch B*S=3072 ≥ 1024 (kernel invariant in `pallas_tpu.py:227`); Gemma 4's shape (V=262144, H=2560) lands in no TPU-tuned bucket (`tuned_block_sizes.py` TPU buckets top out at V=131072). Durable artifacts: `JAX_CE_IMPL=levanter` gate + `jax/model/kernels/fused_ce/` import shim (levanter kernel importable without installing equinox/draccus/rigging via `sys.modules` stubs); `jax/tools/parity_levanter_ce.py` correctness harness; `Gemma4ForCausalLM.__call__(return_hidden=True)` + `lm_head_weight()` seam for future CE-replacement variants. Follow-ups: retry at seq=2048 b=2 (logits pressure higher, cost/benefit may flip); V-sharded kernel variant (removes the w_hv all-gather, kernel-edit effort M); `collective-permute-done` audit unchanged from exp 36's queue. Profile local only (`raw/profiles/2026-04-24-gemma4-jax-exp47-levanter-ce/`, 328 MiB) + GCS mirror at `gs://<your-bucket>/autoresearch/2026-04-24-gemma4-jax-exp47-levanter-ce/` (326.6 MiB uploaded). xprof browser access blocked in this account's logdir (same as other recent JAX-stack experiments). Promotes the "Pallas custom-call tax" heuristic to a general rule: swapping a Pallas kernel in for an XLA-fused op that is <5 % of step time tends to lose fusion more than it gains kernel speed.

## [2026-04-24] run-experiment | jax-exp43: tokamax.linear_softmax_cross_entropy_loss — INVALID (API-precondition failure, no softcap support)

**Op**: run-experiment.
**Pages created**: `wiki/experiments/gemma4_autoresearch_optimization/jax/experiments/2026-04-23-exp43-jax-tokamax-ce-rejected.md`.
**Pages updated**: `.../jax/experiments/RESULTS.tsv` (exp43 discard row), `.../jax/experiments/OBSERVATIONS.md` (exp 43 block), `.../jax/experiments/README.md` (Current state paragraph adds exp 43 line; exp 36 remains best).
**Key result**: **INVALID** — no run performed. tokamax `v0.0.12` public API `tokamax.linear_softmax_cross_entropy_loss(x, labels, weights, *, reduction, precision, implementation)` has **no** `logits_soft_cap` (or equivalent) kwarg (verified across `api.py`, `base.py`, `reference.py`, and the 690-line `pallas_mosaic_tpu_kernel.py`). Gemma 4's `final_logit_softcapping=30.0` is a non-linear `sc * tanh(logits / sc)` element-wise on the `[B, S, V]` logits — cannot be folded into `hidden` or `W` (algebraically impossible for non-linear post-matmul ops), and applying post-kernel requires materializing `[B, S, V]` which defeats the kernel's sole purpose. Skipping softcap violates program.md "What you CANNOT do". Zero lines of code merged. Exp 36 remains JAX-stack best at 34,614 TPS / 86.75 % HBM.
**Notes**: Decision-tree exit per task brief: "Softcap incompatible with tokamax API → can't apply cleanly → `-invalid`. Don't merge." The result **empirically confirms** the build-target already catalogued in [program.md § "Pallas kernels to BUILD" → "Fused final logit softcap + log-softmax + NLL"](experiments/gemma4_autoresearch_optimization/program.md) — a Gemma-aware kernel fork that applies `sc * tanh(logits_tile / sc)` inline on each VMEM block before `log_softmax` is the only correct path to the ~1.5 GiB HBM save on this model. Estimated effort M (extend mosaic_tpu_kernel.py with a softcap pre-op). Secondary API mismatches also noted: no `ignore_index`; `reduction="mean"` divides by `B*S` instead of `mask.sum()`; `x` wants `[B, H]` flat (easy reshape). These are ~15-line glue, non-blocking — softcap is the hard blocker. Note: torchax exp 27's prior tokamax integration attempt failed on **attention** (`dot_product_attention` / sliding-window masks) — exp 43 is the first tokamax-CE attempt on either stack, and fails at a different API seam.

## [2026-04-23] run-experiment | jax-exp37: bf16 CE env-var gate on top of exp 36 — POTENTIAL (flat, no-op-by-construction)

**Op**: run-experiment.
**Pages created**: `wiki/experiments/gemma4_autoresearch_optimization/jax/experiments/2026-04-23-exp37-jax-splash-b3-bf16ce-potential.md`.
**Pages updated**: `.../jax/experiments/RESULTS.tsv` (exp37 parked row), `.../jax/experiments/OBSERVATIONS.md` (exp 37 block), `.../jax/experiments/README.md` (Current state + queued exp list — exp 37 resolved, exp 38 now highest priority), `.../jax/train.py` (added `JAX_CE_DTYPE` gate).
**Key result**: **POTENTIAL (flat, within noise)**. TPS 34,614 → **34,629 (+0.04 %)**; step time 355.0 → 354.85 ms; peak HBM 27.11 → 27.45 GiB (heap unchanged at 10.67 GiB, stack +0.34 GiB). Loss matches exp 36 within 0.3 %. Exp 36 remains JAX-stack best at 34,614 TPS.
**Notes**: The torchax exp 12 analog (+3.0 % TPS / −1.5 GiB HBM) replicated as a **no-op** on the JAX port: the native-JAX port never had the `flat_logits.to(fp32)` upcast that torchax inherited — `forward_loss` in `jax/train.py` has been bf16 log_softmax by construction since the exp-34 port. The durable artifact from exp 37 is the `JAX_CE_DTYPE={bf16,fp32}` env-var gate, which enables controlled A/B on future refactors (e.g., if exp 39 Pallas RMSNorm changes upstream dtype flow). HLO graph bit-identical to exp 36 aside from one dropped implicit-cast in the CE reduction. Profile uploaded to `gs://<your-bucket>/autoresearch/2026-04-23-gemma4-jax-exp37-splash-b3-bf16ce/`; xprof at http://localhost:8791/?run=2026-04-23-gemma4-jax-exp37-splash-b3-bf16ce. Next priority: exp 38 (collective-permute-done sharding audit, 12.1 % of step time).

## [2026-04-23] run-experiment | jax-exp36: splash + batch=3 — new JAX-stack best, BEATS torchax session-best (+3.7 %)

**Op**: run-experiment.
**Pages created**: `wiki/experiments/gemma4_autoresearch_optimization/jax/experiments/2026-04-23-exp36-jax-splash-batch3-accepted.md`.
**Pages updated**: `.../jax/experiments/RESULTS.tsv` (exp36 keep row), `.../jax/experiments/OBSERVATIONS.md` (exp 36 block), `.../jax/experiments/README.md` (Current state + queued exp list).
**Key result**: **SUPPORTED / accepted**. TPS 30,386 → **34,614 (+13.9 %)**; step time 134.8 → 355.0 ms (sub-linear at 3× batch — per-token cost −13.9 %); peak HBM 16.43 → **27.11 GiB / 86.75 %** (4.14 GiB free). The JAX stack at b=3 splash **surpasses torchax session-best** (exp 25: 33,372 TPS) by **+3.7 %** without bf16 CE. HLO-op diff validates the per-call-amortization mechanism: splash `custom fusion` near-constant (169 → 175 ms, ×1.03) while matmul/loop fusions grew 2.75–3.81×.
**Notes**: No code change — only `--batch_size 1` → `--batch_size 3` and `JAX_ATTENTION_IMPL=splash` env var unchanged from exp 35. Loss descends cleanly 3.81 → 1.84 (b=3 batch content differs from b=1, not bit-comparable; trajectory healthy). New bottleneck surfaces at b=3: `loop fusion` 28.1 % (RMSNorm Pallas kernel now worth building), `collective-permute-done` 12.2 % (SPMD re-shard audit). Next: exp 37 (bf16 CE), exp 38 (sharding audit), exp 39 (RMSNorm kernel). Profile uploaded to `gs://<your-bucket>/autoresearch/2026-04-23-gemma4-jax-exp36-splash-batch3/`; xprof browser at http://localhost:8791/?run=2026-04-23-gemma4-jax-exp36-splash-batch3.

## [2026-04-23] ingest-batch + lint | Wave 4 follow-up — 15 concept stubs, 5 deferred codebases, 11 scaling-book chapters, lint pass

**Op**: ingest-source (11 scaling-book chapters) + ingest-codebase (5 deferred repos) + manual (15 concept stubs) + lint.

**Pages created (31)**:

- **15 concepts** (per the 2026-04-23 Pallas kernel directory's "Reusable Pallas-authoring patterns" + hardware-fact list):
  - 11 Pallas-pattern stubs: `online-softmax-with-logit-sink`, `in-kernel-dropout`, `two-level-chunk-recomputation`, `grouped-program-ids-for-l2`, `dma-overhead-heuristic`, `multi-shard-sequence-parallel-correction`, `block-sparse-offset-masks`, `custom-splash-masks`, `manual-mlir-dialect-pallas`, `pallas-on-triton-fused-gemm-activation-gemm`, `nvidia-weight-tile-bytes-limit`.
  - 3 autotune-harness stubs (from marin/levanter harness): `jaxpr-hash-cache-keys`, `compile-time-aware-autotune-filtering`, `vmem-oom-fallthrough`.
  - 1 hardware-fact concept: `vmem-budget` (v4 32 / v5e 32 / v5p 95 / **v6e 96** / **v7 48** MiB — the v6e→v7 halving).
- **5 codebases** (deferred from Wave 4's lower-priority tier):
  - `codebases/graphcast.md` (commit `08cf736`) — DeepMind weather model; splash wrapper + `WeatherMeshMask`.
  - `codebases/simply.md` (`f40b81e`) — DeepMind serving framework; DMA-overhead-bytes autotune heuristic.
  - `codebases/jaxite.md` (`e4a3351`) — Google FHE library; only non-ML Pallas TPU ref.
  - `codebases/qwix.md` (`b966dc4`) — Google quantization framework; AQT successor.
  - `codebases/aqt.md` (`9d1667e`) — deprecated; superseded by qwix.
- **11 scaling-book source pages** (one per content chapter, per-chapter summaries from a parallel Explore agent; each page ~150 lines with key claims, key data points, techniques referenced, gaps & caveats, connections). Book dated 2025-02-04; **v6e mentioned briefly, v7 absent** — explicit gaps flagged on every chapter page.

**Submodules added (5)**: `raw/code/{graphcast, simply, jaxite, qwix, aqt}`.

**Pages updated**:
- `README.md` — 5 new "Ingested codebases" entries.
- `wiki/index.md` — Codebases 21→26 (new "Wave 4 follow-up" sub-section), Sources 34→45 (new "Scaling-book chapters (11)" sub-section), Concepts 81→96 (three new sub-sections: Pallas-authoring patterns, Autotune-harness patterns, Hardware facts). Header page count 146→178.

**Key result**: Wave 4 follow-up work complete — concept-pattern stubs filed, deferred repos ingested, scaling-book finally out of "Wave 3 pending" limbo. Lint pass found and fixed 12 broken markdown links (all pre-existing in the gemma4 experiments tree — renamed experiment files with `-accepted` / `-rejected` / `-potential` suffixes whose cross-references hadn't been updated). Zero broken links remain.

**Notes**:

### Lint findings and actions taken
- **12 broken `.md` links** — all fixed. Pattern: gemma4 experiment files were renamed with outcome suffixes (`-accepted` / `-rejected` / `-potential`) after initial cross-references were written. Bulk-fixed via `sed` across five referring files. Also fixed one `../concepts/` → `../../concepts/` relative-path error in `OBSERVATIONS.md`, and one reference to a non-existent `codebases/transformers.md` (replaced with explicit upstream reference to HF transformers repo).
- **10 orphan pages** detected — all are pages created in this turn; all now indexed, no longer orphans.
- **Zero submodule-commit drift** — every codebase page's pinned commit matches its submodule checkout.
- **Zero `[!warning]` contradictions** in the wiki.
- **Hypotheses-open-14+-days lint** skipped — the gemma4 program doesn't use separate hypothesis files (it tracks them inline in the program README).
- **Experiments-without-profile-artifacts lint** skipped for this turn (not directly related to the new work).

### Scaling-book ingestion discipline
- Delegated per-chapter reading to an Explore subagent using a structured-summary schema; agent returned a 7,000-word consolidated summary that seeded the 11 source pages without requiring each chapter's full body in the main-agent context.
- **v6e (Trillium) mentioned briefly** in Ch 2; **v7 (Ironwood) absent throughout** — every chapter page flags this as a "Gaps & caveats" note because the wiki's primary gemma4 program runs on v6e-4, meaning the book's constants are already one generation behind.
- Chapter 4 KV-cache formula (`2 × S × L × K × H` bytes/sequence) and Chapter 7 critical-batch thresholds (B > 120 int8, B > 240 bf16) are the most directly-cite-able data points from the book.
- Chapter 9 "matmul roofline matched to 0.4%" worked example is now documented as an upper-bound datapoint for any "is my profile-vs-roofline gap significant?" question.

### Five deferred codebases — quick notes
- **graphcast**: non-LLM splash example; `WeatherMeshMask` is the canonical "custom splash mask" reference cited on the concept stub.
- **simply**: the DMA-overhead-bytes autotune heuristic (~0.5 MiB virtual bytes per DMA) comes from here; already concept-paged.
- **jaxite**: only non-ML Pallas TPU reference in the wiki; bf16-byte-split → u32 reassembly pattern.
- **qwix**: Google's **current** quantization framework; `QArray`-aware `pallas_call` wrapper is the right layer for future quant hypotheses.
- **aqt**: predecessor; deprecated. Do not target for new work.

### Remaining deferred / out-of-scope
- `JetStream` (AI-Hypercomputer) — **archiving 2026-02-01**; deliberate skip.
- `pytorch/xla` — consumer of tpu-inference via torch_xla custom ops; would be a framework page, not a Pallas source.
- `labyrinth-ssr` / `Essential-AI/maxtext-external` — vendor-only snapshots with no novel content.
- `pytorch/pytorch` — Inductor Pallas codegen is a compilation target, not a kernel library; deliberate skip.
- `google-deepmind/gemma` — not enumerated by the directory analysis; remains a gap.
- NVIDIA Mosaic-GPU kernels outside `jax-ml/jax` / tokamax / axlearn — not exhaustively surveyed; gap noted.

## [2026-04-23] ingest-codebase (batch) | Wave 4 — 11 Pallas-kernel repos

**Op**: ingest-codebase (batch of 11, driven by the [2026-04-23 Pallas kernel directory analysis](analyses/2026-04-23-pallas-kernel-directory.md)'s "Recommended next ingestion wave").

**Submodules added (11)**: `raw/code/{axlearn, tpu-inference, maxtext, maxdiffusion, ringattention, alphafold3, recurrentgemma, ejkernel, EasyDeL, sglang-jax, marin}`. **alphafold3 pinned to tag `v3.0.1`** (commit `231efc9`) because the ingested `gated_linear_unit/` directory was removed from `main` after that tag.

**Pages created (11 codebases)**:
- `wiki/codebases/axlearn.md` — novel SSM Pallas (Mamba1 / Mamba2-SSD / RAttention), splash extensions.
- `wiki/codebases/tpu-inference.md` — broadest novel TPU Pallas surface; crown-jewel tuning tables.
- `wiki/codebases/ringattention.md` — canonical Liu-2023 paper companion; unidirectional, no zig-zag.
- `wiki/codebases/maxtext.md` — closest public analogue of this wiki's gemma4 program.
- `wiki/codebases/maxdiffusion.md` — only repo wiring ring-attention as first-class splash-integrated kernel.
- `wiki/codebases/alphafold3.md` — only public production-grade Pallas fused GLU (GPU via Triton-on-Pallas).
- `wiki/codebases/recurrentgemma.md` — canonical LRU scan; ancestor of axlearn Mamba.
- `wiki/codebases/ejkernel.md` — broadest community TPU Pallas surface (17 kernels).
- `wiki/codebases/EasyDeL.md` — ejkernel consumer; operations-registry wrapper.
- `wiki/codebases/sglang-jax.md` — novel EAGLE speculative-decoding tree kernels + largest tuning table.
- `wiki/codebases/marin.md` — **deployment-time autotune harness** — the pattern this wiki should emulate.

**Pages updated**:
- `wiki/sources/2025-ultrascale-playbook.md` — Gaps & caveats: **partial retractions** on hypothesis candidates #2, #3, #4 based on the directory findings (ring-attention has three public refs; zig-zag still absent; RMSNorm/LayerNorm correctly unimplemented per exp 33; fused GLU reference exists at alphafold3 v3.0.1).
- `wiki/codebases/tokamax.md` — ring-attention section updated to name the three public reference impls (maxdiffusion splash-integrated, haoliuhl canonical, ejkernel splash wrapper) and reiterate that zig-zag remains open.
- `README.md` — added 11 new "Ingested codebases" entries; removed duplicate jax entry.
- `wiki/index.md` — Codebases 10→21; split into "Wave 4" (the 11 new) and "Wave 1–3" (existing). Header count 135→146.

**Key result**: Wave 4 ingestion complete — scope discipline held. The 11 new pages are shorter than average (50–180 lines each) because they intentionally defer per-kernel detail to the [pallas-kernel directory subpages](analyses/pallas-kernel-directory/) rather than re-transcribing. Every new codebase page links forward to the authoritative subpage §1–§6 row. Two partial retractions and one reaffirmation landed on the ultrascale playbook source page.

**Notes**:
- **Novelty anchors** (kernels / patterns that exist only in the newly-ingested repos):
  - axlearn: Mamba1 (`mamba_kernels.py`), Mamba2/SSD (`ssd_kernels.py`), RAttention linear attention (`linear_attention_kernels.py`), splash-with-dropout + logit-sink.
  - tpu-inference: RPA v3 (+ hd64), MLA v1/v2, fused_moe v1, quantized_matmul blockwise, all_gather_matmul, GDN + triangle_solver, SparseCore gather/scatter, structured_sparse_matmul v1.
  - alphafold3 (v3.0.1): Pallas fused GLU (GPU, `mosaic_gpu` via Triton).
  - recurrentgemma: LRU Pallas scan with complex accumulators + multi-shard correction.
  - ringattention: canonical Pallas TPU ring-attention (paper companion).
  - sglang-jax: EAGLE tree-speculative-sampling kernels + ~2,000+ RPA tuning entries (largest table surveyed).
  - marin: deployment-time kernel-agnostic autotune harness with six distinguishing properties over tokamax's write-time tuner.
- **Crown-jewel autotune patterns** (from marin's harness; all concept-candidate behaviors): compile-time-aware candidate filtering (threshold = 0.20 s), VMEM-OOM-aware fallthrough, sharding-preserving benchmark lowering, off-thread compile for mesh-bound contexts, jaxpr-hashed cache keys, GCS-aware persistent cache (shares bucket with PJRT compile cache).
- **VMEM budgets baked into kernels** (concept-level facts from tpu-inference quantized_matmul): v6 = 96 MiB, v7 = 48 MiB. RPA default = 100 MB. update_kv_cache = 64 MB. Worth elevating to concept pages.
- **NVIDIA weight-tile limit** (marin fused CE): `_NVIDIA_WEIGHT_TILE_BYTES_LIMIT = 101_376` — same on GB10 and H100. Concept fact.
- **DMA-overhead-equivalent** (simply; not ingested but referenced): ~0.5 MiB virtual bytes, assumed constant across TPU generations.
- **alphafold3 v3.0.1 pin discipline**: kernels removed from `main` after that tag. Every wiki link must include `@v3.0.1` or use the local submodule tree.
- **Deferred** (per Wave 4 proposal lower-priority tier): jaxite (niche FHE non-ML), graphcast (wrapper), simply (wrapper), qwix/aqt (quant framework), JetStream (archiving 2026-02-01), pytorch/pytorch Inductor, labyrinth-ssr / Essential-AI (vendor-only), pytorch/xla (consumer of tpu-inference via torch_xla custom ops).
- **Concept-page stubs not yet filed**: 13 reusable patterns surfaced by the directory (online-softmax-with-logit-sink, in-kernel-dropout, two-level-chunk-recomputation, grouped-program-ids-for-L2, DMA-overhead-heuristic, multi-shard-sequence-parallel-correction, block-sparse-offset-masks, jaxpr-hash-cache-keys, compile-time-aware-filtering, VMEM-OOM-fallthrough, manual-MLIR-dialect-Pallas, Pallas-on-Triton-fused-GEMM-activation-GEMM, custom-splash-masks). Candidate for a subsequent pass; not blocking.
- **Ingestion-page verbosity note**: each new page deliberately short — the per-kernel detail is authoritative in the [pallas-kernel directory subpages](analyses/pallas-kernel-directory/) (written by six parallel research agents), so codebase pages link forward rather than re-transcribe. Following SCHEMA's "One entity/concept/model per page. Split when a page exceeds ~500 lines" rule + avoiding duplication.

## [2026-04-23] analyze | Pallas kernel directory — ~200 kernels across ~30 repos

**Op**: analyze (directory/catalog; six parallel web-research agents).

**Pages created**:
- `wiki/analyses/2026-04-23-pallas-kernel-directory.md` — main directory page: cross-cutting functional-category tables (attention/paged/ring/MoE/norm/GLU/matmul/collectives/SSM/CE-loss), stability distribution, autotune/tuning-table inventory (crown jewels), reusable patterns (13 concept-page candidates), confirmed gaps, direct impact on open hypothesis candidates, recommended Wave-4 ingestion order.
- `wiki/analyses/pallas-kernel-directory/01-upstream-jax-tokamax.md` — ~55 kernels (jax-ml/jax + openxla/tokamax).
- `wiki/analyses/pallas-kernel-directory/02-ai-hypercomputer.md` — ~17 kernels (maxtext + maxdiffusion + JetStream). JetStream archives 2026-02-01.
- `wiki/analyses/pallas-kernel-directory/03-inference-engines.md` — ~33 kernels (vllm-project/tpu-inference + sglang-jax + aphrodite).
- `wiki/analyses/pallas-kernel-directory/04-research-labs.md` — ~18 kernels (apple/axlearn + google-deepmind/{recurrentgemma, simply, graphcast, alphafold3@v3.0.1}).
- `wiki/analyses/pallas-kernel-directory/05-frameworks-quant.md` — ~18 kernels (tunix, qwix, aqt, jaxite, paxml/praxis, pytorch/xla, marin/levanter, pytorch/pytorch Inductor).
- `wiki/analyses/pallas-kernel-directory/06-community-research.md` — ~50 rows (ejkernel, EasyDeL, ringattention, flashback, gla-jax, sqtian, maxtext-external, tpu-research, recompute_dont_restore + small repos).

**Pages updated**:
- `wiki/index.md` — Analyses 2 → 3 + 6 subpages under a nested list; page count 128 → 135.

**Key result**: **~200 Pallas kernels catalogued across ~30 repos**, each with source-path URL, backend, stability, performance claims (verbatim when they exist), use case, and callers. `mosaic_tpu` backend dominates (~60%). Every production-grade TPU kernel is either published upstream in jax-ml/jax / openxla/tokamax or vendored from them. Novel community kernels concentrated in six repos: **vllm-project/tpu-inference** (RPA v2/v3, MLA v1/v2, fused_moe v1, quantized_matmul blockwise, all_gather_matmul, fused_gdn, sparse_core, batched_rpa), **apple/axlearn** (Mamba1, Mamba2/SSD, RAttention linear-attention — the only public TPU-Pallas SSM kernels publicly), **AI-Hypercomputer/maxdiffusion** (ring_attention integrated with splash), **google-deepmind/alphafold3@v3.0.1** (production Pallas fused-GLU), **google-deepmind/recurrentgemma** (canonical LRU scan), **erfanzar/ejkernel** (broadest community TPU surface — 17 kernels).

**Crown-jewel tuning artifacts**: sglang-jax's ~2,000+ entry RPA `tuned_block_sizes.py`; tpu-inference's 1,200+ RPA v2 + 600+ quantized_matmul + 47 megablox + 28 fused_moe v1 tables; marin/levanter's **kernel-agnostic deployment-time autotune harness** (jaxpr-hashed keys, compile-cost-aware candidate filtering at 0.20s threshold, VMEM-OOM fallthrough, GCS-persistent cache colocated with PJRT). VMEM budgets baked into kernels: quantized_matmul v6=96 MiB, v7=48 MiB; RPA default=100 MiB; update_kv_cache=64 MiB. DMA-overhead-equivalent from simply: ~0.5 MiB virtual bytes across TPU generations.

**Notes**:
- **Partial retractions and closures of open hypothesis candidates on [sources/2025-ultrascale-playbook.md](sources/2025-ultrascale-playbook.md)**:
  - Candidate #2 ("wire tokamax `ring_attention_kernel` through `dot_product_attention`") — three public reference impls confirmed: maxdiffusion splash-integrated, haoliuhl canonical standalone, ejkernel splash wrapper. Hypothesis reduced from "open research" to "port + adapt" with three patterns to compare.
  - Candidate #3 ("Zig-Zag Ring Attention on TPU — no implementation found") — **retraction stands.** Ring Attention Pallas exists publicly (§6.4), but Zig-Zag causal-balance variant is **confirmed absent from every repo surveyed**, including the canonical haoliuhl repo (straight unidirectional ring, `below_or_on_diag` check only — not load-balanced).
  - Candidate #4 ("TPU-native Pallas kernels for gated_linear_unit and layer_norm in tokamax") — **partially retracted.** RMSNorm/LayerNorm absence from maxtext + tpu-inference + axlearn + upstream is external evidence that XLA-fusion is sufficient — **consistent with Gemma 4 exp 33's −8.1% empirical result.** Don't build RMSNorm. Fused GLU: AlphaFold3 v3.0.1 provides a production Pallas fused-GLU reference (GPU); porting to Mosaic-TPU is feasible but needs HLO-level validation that XLA isn't already fusing.
- **External validation of Gemma 4 exp 33 lesson** ("Pallas loses when XLA already fuses"): maxtext + tpu-inference + axlearn hand-write megablox, ragged-paged-attention, gather-reduce-sc, fused_moe — but **do not** hand-write RMSNorm, SwiGLU, softmax, or elementwise activations. That absence is data.
- **AlphaFold3 Pallas kernels live only on tag `v3.0.1`** — removed from `main`. Pin the tag in every URL.
- **jondeaton/ring-attention-jax-pallas** — **404 confirmed**, repo does not exist. Dropped from the catalog. User has no matching public repo per `gh api users/jondeaton/repos`.
- **Backend mislabels in community repos**: flashback, gla-jax, mamba2-jax-pallas all advertise Pallas but are Triton/Mosaic-GPU only. Only haoliuhl, ejkernel, sqtian, recompute_dont_restore, labyrinth-ssr (vendored), Essential-AI (vendored), and rdyro/moe_in_jax are actually TPU Pallas. Catalog flags each.
- **New concept-page candidates** (13 surfaced): online-softmax-with-logit-sink, in-kernel-dropout, two-level-chunk-recomputation, grouped-program-ids-for-L2, DMA-overhead-heuristic, multi-shard-sequence-parallel-correction, block-sparse-offset-masks, jaxpr-hash-cache-keys, compile-time-aware-filtering, VMEM-OOM-fallthrough, manual-MLIR-dialect-Pallas, Pallas-on-Triton-fused-GEMM-activation-GEMM, custom-splash-masks. Not filed as stubs yet — listed on the main directory page under "Reusable Pallas-authoring patterns".
- **Recommended Wave 4 ingestion order** (listed on the directory page): apple/axlearn (narrow kernel subdirs), vllm-project/tpu-inference, AI-Hypercomputer/maxtext, AI-Hypercomputer/maxdiffusion (narrow: splash_attention/ only), haoliuhl/ringattention, google-deepmind/alphafold3 @ v3.0.1 (narrow: gated_linear_unit/ only), google-deepmind/recurrentgemma, erfanzar/ejkernel + EasyDeL (paired), sgl-project/sglang-jax (narrow: speculative-decoding kernels), marin-community/marin (narrow: levanter/kernels/pallas/ + autotune harness).
- **Methodology**: six general-purpose subagents in parallel, each with identical per-kernel row schema; verified top candidates via GitHub API + WebFetch. Durations ~4–6 minutes each; total wall clock ~6 minutes for research + ~10 minutes for consolidation.
- **Gaps flagged**: NVIDIA Mosaic-GPU kernel catalog for GPU-side parity (enumerated in jax-ml/jax but not exhaustively surveyed in downstream GPU-Pallas repos); `google-deepmind/gemma` not checked; Gemini training stacks not in any surveyed public repo; Anthropic/xAI/Cohere/Character private. Ragged-Paged-Attention arXiv ID appeared as `2604.15464` in one search result (likely future-dated scrape glitch) — verify before citing.

## [2026-04-23] ingest-codebase | jax (jax-ml/jax)

**Op**: ingest-codebase.
**Pages created**:
- `wiki/codebases/jax.md` — codebase parent page for `jax-ml/jax` at commit `feb5ba0585` (HEAD on 2026-04-23; bleeding-edge pin). Scoped to perf-relevant surfaces only — the repo is far too large for exhaustive ingestion.

**Pages updated**:
- `.gitmodules` — added `raw/code/jax` submodule.
- `README.md` — added jax to "Ingested codebases" list at the top.
- `wiki/index.md` — Codebases 9→10 (jax inserted at top); header count 127→128.

**Key result**: The ground-truth JAX repo is now ingested. The codebase page focuses on the four perf-relevant surface buckets — transformations & compilation, parallelism & sharding, kernels & lowering, profiling & analysis — and indexes 12 concrete performance-relevant levers with file-path anchors. The **first-party reference TPU Pallas kernel tree** at `jax.experimental.pallas.ops.tpu.*` (splash_attention, paged_attention, ragged_paged_attention, megablox, flash_attention, matmul, all_gather, threefry) is now first-class in the wiki — previously referenced only transitively through tokamax's mirror and the 2026-04-23 pallas-kernel-source-survey analysis.

**Notes**:
- **Commit `feb5ba0585` is `HEAD` on 2026-04-23** — the most-recent possible pin when ingesting. `git submodule update --remote raw/code/jax` to bump.
- **`jax.experimental.roofline` surfaced as a first-party alternative to `pallas-forge.roofline_chart`**: works on any JAX function, not just Pallas kernels; `roofline`, `register_roofline`, `roofline_and_grad` are the public API. The [pallas-forge](codebases/pallas-forge.md) page's `roofline_chart` is duplicated functionality; the codebase page flags this.
- **`jax.experimental.compilation_cache`, `jax.experimental.layout`, `jax.lax.scan` are all named in the gemma4 ceiling analysis as remaining levers.** The jax page now gives them concrete file-path anchors. The compilation cache is "infrastructure only, no TPS" per the analysis; layout is the backing API for the SEQ_MINOR choice in exp 24; scan-over-layers Option B is one of three remaining viable paths.
- **Splash-attention upstream-vs-mirror reconciliation**: `jax/experimental/pallas/ops/tpu/splash_attention/` in this repo is the authoritative upstream for both the wiki's splash-attention concept page and the `tokamax` `_src/ops/experimental/tpu/splash_attention/` copy. Future splash-tuning hypotheses can target either entry point; the jax codebase page now says so.
- **MegaBlox (`ops/tpu/megablox/gmm.py`) is first-party for MoE grouped matmul.** The 2026-04-23 pallas-kernel-source-survey already surfaced this; the jax codebase page gives it the in-tree citation.
- **Scope discipline**: page deliberately does not enumerate every jax module — numpy, scipy, nn, random, lax primitives other than `scan`, the jax2tf / jax2onnx paths, export/AOT surfaces beyond `serialize_executable`, the CUDA/Triton GPU Pallas path beyond a pointer, and the docs/tests/benchmarks directories are all explicitly out of scope. The page's job is to index the perf-relevant knobs, not to teach JAX.
- **Two surfaces underused by this wiki so far, now documented**: `jax.experimental.source_mapper` (HLO↔Python back-reference — complements the xprof graph viewer) and `jax.experimental.custom_partitioning` (SPMD escape hatch — the mechanism production libraries use to route around partitioner gaps in ragged-paged-attention / MoE kernels per the pallas-kernel survey).
- **No hypotheses / concept stubs filed.** Concept pages for most of the JAX surfaces ([pallas-kernel](concepts/pallas-kernel.md), [scan-over-layers](concepts/scan-over-layers.md), [sharding](concepts/sharding.md), [rematerialization](concepts/rematerialization.md), [jax-trace](concepts/jax-trace.md), etc.) already exist and link both ways.

## [2026-04-23] analyze | Public Pallas kernel source survey

**Op**: analyze (web-research survey).
**Pages created**:
- `wiki/analyses/2026-04-23-pallas-kernel-source-survey.md` — categorized inventory of every public source of JAX Pallas kernel code (Tier 1 production libraries → Tier 7 marginal). Agent-delegated WebSearch + WebFetch + GitHub API; verified top candidates.

**Pages updated**:
- `wiki/index.md` — Analyses 1 → 2; header page count 126 → 127.

**Key result**: Five production-grade ingest candidates identified: **AI-Hypercomputer/maxtext** (direct trainer analogue), **vllm-project/tpu-inference** (broadest novel kernel surface: ragged-paged-attention v2/v3, MLA, gdn, sparse-core, structured-sparse-matmul), **AI-Hypercomputer/maxdiffusion** (ring-attention reference), **apple/axlearn** (Mamba/SSD + rattention), **sgl-project/sglang-jax** (simple-gla, spec-decoding).

**Notes**:
- **Partial retraction** on two open hypothesis candidates from [sources/2025-ultrascale-playbook.md](sources/2025-ultrascale-playbook.md):
  - Candidate #2 ("wire tokamax `ring_attention_kernel` through `dot_product_attention`") — public reference impl exists at `AI-Hypercomputer/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py`. Reduces the hypothesis from "open research" to "port + adapt".
  - Candidate #3 ("Zig-Zag Ring Attention on TPU — no implementation found") — Ring Attention Pallas exists publicly in `haoliuhl/ringattention` (770⭐, canonical Liu et al. impl). Whether the specific Zig-Zag causal-balance variant is implemented remains unverified — needs code-level read.
- **Cross-reference with gemma4 exp 33 lesson** ("Pallas loses when XLA already fuses"): external evidence supports this. maxtext and tpu-inference hand-write megablox, ragged-paged-attention, gather-reduce-sc — but **do not** hand-write RMSNorm or SwiGLU. That absence is data.
- **AlphaFold3 Pallas GLU** exists at `google-deepmind/alphafold3` **tag v3.0.1** — removed from `main`. Pin the tag before referencing.
- **New category surfaced: kernel-optimization agents** — `ucb-bar/autocomp`, `primatrix/Glaucis` (evolutionary Pallas search), `aryatschand/JAXBench` (LLM-authored kernel benchmark). Direct analogues to this wiki's autoresearch mission; worth shallow reads for search-procedure priors.
- **Scope discipline**: did not file individual hypothesis pages or update the existing playbook hypothesis candidates; the analysis page cross-references them inline and leaves promotion to human adjudication.
- **Gaps flagged in the analysis** for a follow-up sweep: NVIDIA Mosaic-GPU ops inside `jax-ml/jax`, GoogleCloudPlatform sample repos, `google-deepmind/gemma`, private-org repos (Anthropic, xAI, Cohere, Character), Ragged-Paged-Attention arXiv ID verification, `jondeaton/ring-attention-jax-pallas` 404 on verification.

## [2026-04-23] ingest-codebase + ingest-source | pallas-forge + Karpathy LLM-wiki backfill

**Op**: ingest-codebase (pallas-forge) + ingest-source (Karpathy LLM-wiki).
**Pages created**:
- `wiki/codebases/pallas-forge.md` — codebase parent page for `linhkid/pallas-forge` at commit `090510b7`. Already used (and found wanting) in gemma4 exp 20; the page makes that context first-class and captures the methodological lessons (isolated-microbench vs in-graph, no-custom_vjp blocker, v5e-only numbers, canonical 3D-grid matmul template).
- `wiki/sources/2026-karpathy-llm-wiki.md` — source page for Karpathy's "LLM Wiki" idea file, the methodological ancestor of this wiki's `SCHEMA.md`. Already locally saved as `raw/sources/2026-karpathy-llm-wiki.md`; backfilling the wiki-side page makes the design lineage explicit.

**Pages updated**:
- `wiki/index.md` — Codebases 8→9 (added pallas-forge at top of list); Sources 33→34 (new "Methodology (1)" subsection); header count bumped to 126 pages.

**Key result**: Two ingestion gaps closed. The pallas-forge page consolidates what the gemma4 program has already learned about the library (exp 20 + exp 33) rather than letting it sit only in experiment pages; the Karpathy-wiki source page documents where this wiki's operating protocol comes from, which matters for future schema edits.

**Notes**:
- **pallas-forge status recap (as captured on the codebase page)**: 3 reference kernels, tuner, roofline, XProf-trace integration — *but* forward-only (no `jax.custom_vjp`), so any training hypothesis that swaps a pallas-forge kernel into a module's `forward` crashes at `jax.value_and_grad` ("Linearization failed to produce known values"). Gemma4 exp 20 is the direct confirmation. Alternative for kernel swaps in training: [tokamax](codebases/tokamax.md) (exposes autodiff-capable `rms_norm` / `layer_norm` / splash-attention paths).
- **Generalizable lesson already in the wiki, now cross-linked**: from the 2026-04-23 Gemma 4 optimization-ceiling analysis (exp 33) — *"Pallas kernels are a net win only when XLA wasn't already exploiting the pattern via fusion"*. pallas-forge's v5e README numbers (3.44× RMSNorm isolated-microbench) do not predict in-graph wins on v6e-4; exp 33 showed an 8.1% regression. The codebase page flags this explicitly under "Performance-relevant surfaces §3 Isolated-microbench ≠ in-graph gain".
- **pallas-forge `TPU_SPECS` gap**: the hardware-preset table for `roofline_chart` covers v4/v5e/v5p only; no v6e entry. Anyone running `roofline_chart` on v6e must supply peak-TFLOPS and peak-HBM-bandwidth directly.
- **Karpathy-wiki source is methodology, not TPU-perf content**: ingested to make the schema's lineage traceable; no hypotheses generated. Its "Connections" section explicitly maps the source's pattern to this wiki's `SCHEMA.md` / `CLAUDE.md` / `index.md` / `log.md` implementations.
- **Ingestion-audit findings recorded for completeness**: (1) the `notes.md` TODO `using git submodule add https://github.com/linhkid/pallas-forge under raw/code, update readme` was already satisfied (`.gitmodules` entry exists, `raw/code/pallas-forge/` is checked out at commit `090510b`); the only missing piece was the wiki page, now filed. (2) `raw/sources/2026-karpathy-llm-wiki.md` was on disk but had no wiki counterpart, now filed. (3) No other unprocessed raw sources or un-documented submodules found.
- **Out of scope for this ingestion** (per `notes.md` lines 28–37, the gemma-4-E4B import + torchax trainer work): that is implementation, not ingestion; the Gemma 4 program already has its own `experiments/gemma4_autoresearch_optimization/` tree with 33 experiments completed and a ceiling analysis filed. Not touched.

## [2026-04-23] analyze | Gemma 4 E4B on v6e-4 — optimization ceiling reached at exp 25

**Op**: analyze (session ceiling synthesis).

**Pages created**:
- [`wiki/analyses/2026-04-23-gemma4-v6e4-optimization-ceiling.md`](analyses/2026-04-23-gemma4-v6e4-optimization-ceiling.md) — synthesis of the 33-experiment loop. Documents the trajectory (baseline 30,570 → exp 25 33,372, +9.2%), what worked vs what didn't, the step-time decomposition at exp 25, the Pallas-fuses-into-matmul lesson from exp 33, and what would actually move the needle next (hardware scale-up, scan-over-layers Option B, or accepting ceiling).

**Pages updated**:
- `wiki/index.md` — Analyses section populated.

**Key result**: Optimization loop has reached diminishing returns on this hardware/model combo. Trunk stays at exp 25 (33,372 TPS, seq=1024 batch=3 fsdp=4 bf16, splash_pallas + SEQ_MINOR + block=1024 + fused_bwd + bf16 CE + selective remat). Eight experiments since exp 25 (exp 26–33) produced zero further wins: scan-over-layers blocked (5 Gemma-specific issues), tokamax DPA blocked (mosaic_tpu no sliding-window), 2D mesh 2.4× slower at this chip count, Pallas RMSNorm −8.1% due to XLA already fusing it with neighbor matmuls, long-seq and XLA-flag-isolation experiments neutral or dominated.

**Notes**:
- **Generalizable lesson (exp 33)**: Pallas kernels are a net win only when XLA wasn't already exploiting the pattern via fusion. Splash won (exp 8) because XLA can't express online-softmax. RMSNorm loses because XLA already fuses it. Likely Pallas SwiGLU would too — don't build it.
- **Remaining viable paths** (none attempted here): (a) 300-500 LOC scan-over-layers Option B for compile-time compression, (b) hardware scale-up to v6e-8 or v5p-4 where 2D mesh and collective overlap become economic, (c) persistent compile cache (infrastructure only, no TPS).
- **Exp 28 kept but not merged**: +0.9% at seq=2048 b=1 strictly dominated by exp 25's higher-batch throughput. Preserved for long-seq reference.

---

## [2026-04-23] protocol + experiments | Gemma 4 E4B program — program.md formalization + exp2 crash

**Op**: manual (protocol formalization) + run-experiment (exp2, crashed).
**Pages created**:
- `wiki/experiments/gemma4_autoresearch_optimization/program.md` — the agent-facing protocol for this program, adapted from the sibling-wiki TorchTitan autoresearch template but specialized to Gemma 4 E4B / torchax / v6e-4.
- `wiki/experiments/gemma4_autoresearch_optimization/RESULTS.tsv` — tab-separated ledger (gitignored via `wiki/experiments/*/RESULTS.tsv`).
- `wiki/experiments/gemma4_autoresearch_optimization/OBSERVATIONS.md` — skim-and-reason aggregation log; backfilled with baseline + exp1 + exp2 blocks.
- `wiki/experiments/gemma4_autoresearch_optimization/2026-04-23-exp2-pin-out-shardings-rejected.md` — Exp 2 experiment page (crashed/invalid, reverted).

**Pages updated**:
- Program `README.md` — stripped the in-README "Optimization loop procedure" section; replaced with a short "How to start the optimization loop" pointer to `program.md`. History extended for exp2 + the protocol formalization.

**Key result**: **Exp 2 CRASHED** pre-trace with `ValueError: Received incompatible devices` on `weights['lm_head.weight']`. Attempt to pin `out_shardings=(loss_ns, weights_ns, opt_state_ns)` on `jax.jit` to fix the ~150 s step-1 recompile hit a tied-weight-dedup plumbing issue in torchax's `JittableModule` — the tied `lm_head.weight` ↔ `embed_tokens.weight` emits an `out_shardings` leaf whose device list collapses to `[0]` even though the input sharding is full-mesh `[0, 1, 3, 2]`. Reverted the `train.py` change; step-1 recompile remains open. Hypothesis marked `verdict: invalid` (pre-trace crash, nothing measured) — still viable under a different implementation (`dedup_parameters=False` + manual tying, or `with_sharding_constraint` inside the train step).

**Notes**:
- **Protocol formalization**: Karpathy-autoresearch-style discipline now lives in a dedicated `program.md` that defines the fixed bindings (hardware, conda env, trainer path, baseline command, libtpu version, profile/HLO conventions, xprof_mcp server), the architectural invariants (can/cannot-change tables for Gemma 4), the measurement protocol (TPS primary, MFU secondary, median steps 6–15), the ledger + observations schema, the full experiment loop, and accumulated heuristics. Future experiments start from reading `program.md` and tail-ing `OBSERVATIONS.md` / `RESULTS.tsv`.
- **Retroactive backfill**: baseline + exp1 + exp2 are transcribed into `RESULTS.tsv` and `OBSERVATIONS.md` for continuity — these experiments predate the formal protocol but provide the foundation the protocol was written against.
- **libtpu is at 0.0.40, the latest** (verified across PyPI, `libtpu-lts-releases`, `libtpu-nightly-releases`). The exp1 "unknown flag" failure was a name error (real name is `--xla_tpu_overlap_compute_collective_tc`, not `_comms`), not a version issue. Documented the symbol-dump-as-source-of-truth trick in `program.md` heuristics.
- **Discovered gotcha**: tied `lm_head` ↔ `embed_tokens` + torchax `JittableModule.dedup_parameters=True` + HF `load_state_dict(assign=True)` interact in a way that confuses `out_shardings` construction. Logged as a `program.md` heuristic; a future experiment can route around it via `dedup_parameters=False`.
- Step-1 recompile mitigation deferred; next experiment focuses on a structural memory-saving change (per the memory-ceiling rule — baseline hit 95% HBM).

## [2026-04-23] experiment | Gemma 4 E4B — Exp 1: async-collective XLA flags (REFUTED)

**Op**: run-experiment (first optimization-loop cycle on the Gemma 4 program).
**Pages created**: `wiki/experiments/gemma4_autoresearch_optimization/2026-04-23-exp1-async-collective-flags-rejected.md`.
**Pages updated**: program `README.md` history (baseline-seq-1024 + Exp 1).
**Raw artifacts**: `raw/profiles/2026-04-23-gemma4-loss-confirm/` (20-step seq=1024 baseline, clean loss), `raw/profiles/2026-04-23-gemma4-exp1-async-collectives/` (exp1 trace). Both symlinked into the live xprof instance as `gemma4_baseline_seq1024_20260423` and `gemma4_exp1_async_collectives_20260423`.

**Key result**: **Hypothesis REFUTED.** Enabling `--xla_tpu_enable_latency_hiding_scheduler` + `--xla_tpu_enable_async_collective_fusion` (+ `fuse_all_gather`, `multiple_steps`) via `LIBTPU_INIT_ARGS` regressed steady-state step time **134.4 → 168.3 ms (+25 %)** at seq=1024, batch=1, FSDP=4, v6e-4. Collectives got faster (all-gather −5 ms, all-reduce-scatter −9 ms) but compute-fusion memory traffic blew up (convolution fusion +2.5× bytes, loop fusion +1.9×). Scheduler gained freedom to reorder; it fused collectives but broke compute-order locality. Loss trajectory preserved (bf16-reorder noise only).

**Notes**:
- 20-step loss-descent confirmation (pre-experiment): loss 3.93 → 1.97 over 20 steps at seq=1024; step time stable at 134.4 ± 0.5 ms. Confirms the baseline scaffold is training correctly.
- Ran full autoresearch loop: baseline profile → hypothesis pick (#9) → experiment → profile diff → verdict → file page → pick next hypothesis. First complete cycle on this program.
- **Flag-placement gotcha**: `--xla_tpu_*` flags go in `LIBTPU_INIT_ARGS`, **not** `XLA_FLAGS`. First attempt bombed with `parse_flags_from_env.cc:234 Unknown flags in XLA_FLAGS`. Logged for future experiments — hypothesis-writers should not trust the "XLA flag" name.
- **Libtpu 0.0.40 does not know `--xla_tpu_overlap_compute_collective_comms`.** Dropped before the successful run. Good instance of a flag catalog drifting vs. the installed runtime.
- **xprof-mcp tools used in the loop**: `list_runs`, `get_overview`, `get_top_hlo_ops`, `get_op_profile`, `get_memory_profile`, `get_device_information`. The HLO-op-level before/after diff was the decisive evidence for "refuted" — wall-clock alone would have been ambiguous, but the bytes-accessed explosion made the mechanism clear.
- Hypothesis parked, not retired — at larger effective batch (via remat or NaN-fix at seq=2048) the collective-overlap win may swamp the compute-locality loss. Revisit after hypothesis #6.
- Next natural hypothesis: **fix the step-1 recompile** (explicit `in_shardings` / `out_shardings` on `jax.jit`) — saves ~150 s per experiment run, pure iteration-speed win. Then **#6 selective remat**.

## [2026-04-22] experiment | Gemma 4 E4B baseline (v6e-4, FSDP)

**Op**: run-experiment (baseline infrastructure check)
**Pages created**: `wiki/experiments/gemma4_autoresearch_optimization/2026-04-22-baseline.md`.
**Pages updated**: program `README.md` (History section updated with first baseline numbers); `wiki/experiments/gemma4_autoresearch_optimization/torchax/{train.py,model/sharding.py,requirements.txt}` (8 targeted fixes to get the scaffold to actually run — enumerated on the baseline page).
**Raw artifacts**: `raw/profiles/2026-04-22-gemma4-baseline/` — xprof trace, steps 5–7 at seq=2048.

**Key result**: First working run. Steady-state **249 ms/step at seq=2048, batch=1, FSDP=4 → ~33k tokens/sec, ~26% MFU** (corrected from an initial `6PT` overestimate of 44% — `P=8B` double-counts Gemma 4's Per-Layer-Embedding lookup tables, which don't participate in matmul FLOPs). Seq-length sweep: **13% MFU @ seq=512, 23% @ seq=1024, 26% @ seq=2048**. Verdict: **supported** on the infrastructure-check hypothesis; **NOT a quality-valid baseline at seq=2048** until the NaN loss is fixed.

**Notes**:
- Hardware was **v6e-4**, not the v6e-8 the scaffold README assumed. FSDP default auto-picked fsdp=4 from `jax.device_count()` — no config change needed.
- **Default sharding strategy flipped TP=8 → FSDP** per user directive. Both paths now available; FSDP is the default. FSDP sharding rule: every ≥2D param shards on its largest dim divisible by `fsdp_size` over a 1D `'fsdp'` axis.
- **Python 3.13.13** env (`gemma4_py313`) works with jax 0.10.0, torch 2.11.0+cpu, torchax 0.0.12 (editable install from the wiki submodule), transformers 5.7.0.dev0 (Gemma 4 only on main), datasets, optax, accelerate.
- **8 scaffold fixes** applied to get from written-but-untested to running; all enumerated in the baseline page's "Scaffold changes applied" table. Most consequential:
  1. `interop._jax_view` → `interop.jax_view` (pytree-map variant; single-value left torch tensors unconverted).
  2. Load `Gemma4ForConditionalGeneration` (not `Gemma4ForCausalLM`) — HF checkpoint is multimodal-only and `ForCausalLM` silently re-inits every weight (name-prefix mismatch against `model.language_model.*`).
  3. Monkey-patch `model.forward` with a text-only path (bypass the multimodal orchestrator's `input_ids[mask] = pad` which is not JIT-traceable).
  4. Apply `final_logit_softcapping=30.0` in the text-only forward (did not fix the seq=2048 NaN, but is semantically required).
  5. Requirements.txt pointed torchax at `pytorch/xla` subdirectory; actual repo is `google/torchax`. Switched to an editable install from the wiki's own submodule (commit `8f957d1`).
- **Correctness issues found by the baseline** (flagged for the next experiment):
  - **NaN loss at seq≥2048** — loss is clean at seq∈{512, 1024}. Likely bf16 attention overflow or a Gemma 4 hybrid-attention mask edge case. Prerequisite for any seq=2048 perf work.
  - **OOM at batch=4, seq=2048** — attention N×N materialized (no flash/splash). Directly motivates hypothesis #1 (Splash Attention).
  - **Step 1 recompiles** — both step 0 and step 1 take ~155 s, step 2+ hits the cache. Likely a sharding-spec / donation mismatch on step-1 inputs. Low-effort follow-up: pass explicit `in_shardings` to `jax.jit`.
- **Arithmetic error noted publicly**: initial MFU claim of 44% was wrong — used `6PT` with `P = headline 8B` instead of `P ≈ non-embedding-matmul params`. Gemma 4's "E4B" headline includes PLE lookups which add params but no matmul FLOPs. User caught this; corrected to ~26% MFU via detailed per-matmul FLOP counting. Keep this in mind for future model-size-related FLOP estimates.
- **Compile time dominates** for short runs: 2 × ~155 s compile vs 2.5 s of useful work at seq=2048 × 8 steady-state steps. Hypothesis #7 (scan-over-layers) should collapse this.

## [2026-04-22] scaffold | Gemma 4 E4B torchax trainer (untested)

**Op**: manual (scaffold code for first execution path of an optimization program)
**Pages created**: 9 files under `wiki/experiments/gemma4_autoresearch_optimization/torchax/` (1,215 lines total):
- `train.py` (439 lines) — fine-tune trainer with profile-step capture.
- `model/sharding.py` (245 lines) — 2D `(dp, tp)` mesh + NeMo-Megatron sharding adapted for Gemma 4 GQA.
- `model/README.md` (118 lines) — config + sharding assumptions, upstream source of truth.
- `data.py` (118 lines) — wikitext loader + fixed-length packer.
- `README.md` (134 lines, includes augmented "Running the trainer" section) — runbook.
- `model/__init__.py` (46 lines) — re-exports `Gemma4Config`, `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`, `Gemma4Model` from `transformers`.
- `run.sh` (50 lines) — wrapper setting `XLA_FLAGS` + `LIBTPU_INIT_ARGS`, forwards to `train.py`.
- `config.yaml` (22 lines) — default args.
- `requirements.txt` (43 lines) — deps pinned against torchax commit `8f957d1`.

**Pages updated**: none (wiki markdown was already in place; only trainer code added).

**Key result**: First execution path of the Gemma 4 program is scaffolded. Status marked **UNTESTED** in multiple places — scaffold written from ingested source pages (jax-huggingface part 2 sharding recipe, torchax codebase architecture, xprof capture docs) without running a single step.

**Research findings worth capturing as wiki content later (not yet pages)**:
- **Gemma 4 E4B is public Apache-2.0**, not gated (login still required for HF hub). `config.json` readable at `https://huggingface.co/google/gemma-4-E4B/raw/main/config.json`.
- **Architecture specifics**: 42 layers, hidden=2560, `num_attention_heads=8`, `num_key_value_heads=2`, `head_dim=256`, `intermediate_size=10240`, vocab=262144, sliding_window=512, max_position=131072. "E4B" = effective-4B via Per-Layer Embeddings; **~8B with embeddings**.
- **Novelties vs Gemma 3**: hybrid attention (local SW 512 + global), `num_kv_shared_layers=18` (cross-layer KV sharing), `rope_type=proportional` with `partial_rotary_factor=0.25` on full-attention layers, `final_logit_softcapping=30.0`, `gelu_pytorch_tanh` MLP, `tie_word_embeddings=true`.
- **Multimodal**: E4B ships vision + audio branches. Trainer targets text-only (`Gemma4ForCausalLM` with fallback to `Gemma4ForConditionalGeneration`).
- **Sharding corner case**: `num_kv_heads=2` does NOT divide `tp=8`. Default partitioning therefore **replicates K/V projections** rather than silently dropping parallelism — flagged as a future hypothesis.
- **Canonical class names** (per HF `transformers` main, `transformers_version: 5.5.0.dev0`): `Gemma4Config`, `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`, `Gemma4Model`. Transformers ships Gemma 4 in `src/transformers/models/gemma4/`.
- **DeepMind gemma repo** exposes `gm.nn.Gemma4_E4B()` + `gm.ckpts.CheckpointPath.GEMMA4_E4B_IT` — native-JAX reference for the `../jax/` folder when that path is activated.

**Assumptions flagged for baseline-run verification**:
1. `Gemma4ForCausalLM` import works; falls back to `ForConditionalGeneration` if not.
2. HF state-dict key naming matches Gemma-family convention (`q_proj`, `k_proj`, …, `embed_tokens`) — regex-based sharder in `sharding.py` is fragile and should be verified with `print(list(model.state_dict())[:20])`.
3. `num_kv_shared_layers=18` may surface as extra/renamed params not covered by the regex — they default to replicated (conservative).
4. torchax API at commit `8f957d1` matches the calls in `train.py` (`JittableModule`, `interop.jax_view/torch_view`, `enable_performance_mode`, `apply_jax_`, `save_checkpoint`).
5. `wikitext-2-raw-v1` chosen as default (small, fast smoke-test). `wikitext-103-raw-v1` available via flag.
6. HF pytree registration targets `CausalLMOutputWithPast` + `DynamicCache`; `StaticCache` registration deferred to a future decode hypothesis.
7. `final_logit_softcapping=30.0` assumed to be implemented inside HF's forward (Gemma 2/3 precedent).

**Known gaps in the scaffold**:
- `--grad_accum` is parsed but not threaded through the training loop.
- No `with_sharding_constraint` activation annotations inside the forward (relies on GSPMD propagation from weight shardings).
- No checkpoint *load* path — `--checkpoint_dir` only saves.
- Splash-attention swap (program hypothesis #1) not wired yet.
- Optimizer states inherit gradient dtype (bf16 if forward is bf16); no explicit fp32 promotion — a baseline concern.

**Next steps for the human** (runbook):
1. `pip install -r wiki/experiments/gemma4_autoresearch_optimization/torchax/requirements.txt` on a v6e-8 host.
2. `huggingface-cli login` with a token that's accepted the Gemma license.
3. `bash wiki/experiments/gemma4_autoresearch_optimization/torchax/run.sh --steps 5 --profile_steps 3`.
4. File the baseline numbers into `wiki/experiments/gemma4_autoresearch_optimization/<YYYY-MM-DD>-baseline.md` (the first dated experiment page).

## [2026-04-22] file-program | Gemma 4 E4B — TPU autoresearch optimization

**Op**: manual (file a new optimization program)
**Pages created**: `wiki/experiments/gemma4_autoresearch_optimization/README.md` — program page for `google/gemma-4-E4B` on TPU v6e via torchax.
**Pages updated**: `wiki/index.md` — Models section (0 → 1).

**Key result**: First `model/` analogue filed. 16 open hypotheses consolidated from Wave 1/2 findings, the xprof-mcp TPU_OPTIMIZATION guide, and the Ultra-Scale Playbook — now have a place to attach. Baseline not yet captured; hypothesis #0 in the ranked list is "capture baseline profile."

**Notes**:
- **Intentional schema deviation**: SCHEMA.md specifies `wiki/models/<slug>.md` for model-under-optimization pages and `wiki/experiments/<YYYY-MM-DD>-<slug>.md` (flat) for experiments. This program uses a **nested folder** `wiki/experiments/gemma4_autoresearch_optimization/` that co-locates the program README (functions as the model page), the dated experiment files (schema-conformant names inside the folder), and optionally local scripts/code. Rationale: a long-running optimization program generates many related files and benefits from being namespaced together; the flat experiments/ directory would make it hard to find "everything about Gemma 4" vs. "everything about the next model."
- This deviation is the **second** intentional one in the wiki (first was the `autoresearch` codebase page's reframed "Structural surfaces we borrow" H2). If it works, the next SCHEMA.md edit should codify `wiki/experiments/<program-slug>/` as a permitted layout for multi-experiment programs, with the README.md inside doubling as the `model` page.
- **Code location decision (2026-04-22)**: option (b) — inside the program folder — selected by human. Further refined: **split into two sibling subfolders by execution path** rather than one `code/` folder:
  - `wiki/experiments/gemma4_autoresearch_optimization/torchax/` — primary, Gemma 4 via torchax.
  - `wiki/experiments/gemma4_autoresearch_optimization/jax/` — secondary, native-JAX port (port-equivalence discipline required: must reproduce torchax-baseline outputs within bf16 tolerance before perf numbers count).
  Each subfolder has its own README documenting conventions (dated copies for divergent scripts, relative-path references from experiment pages, binaries go to `raw/profiles/`, not these folders). The program's top-level README links both. Next SCHEMA.md update should codify `wiki/experiments/<program-slug>/` folders with execution-path-named subfolders for code as a permitted layout.
- Hypotheses are listed on the program page but **not filed as `wiki/hypotheses/*.md` individually** — the program page serves as the consolidated ranked list. Once individual hypotheses become in-flight experiments, each will be promoted to `wiki/hypotheses/<slug>.md` per schema.
- **Gemma 4 E4B**: user confirmed the identifier is correct (per https://huggingface.co/google/gemma-4-E4B). Claude's training cutoff predates this model's release; treating it as a black-box target with Gemma-family architecture (GQA, SwiGLU, RMSNorm) until the baseline ingest confirms exact config.

## [2026-04-22] lint | link hygiene + cross-link + 4 missing stubs

**Op**: lint (automated pass, 2 parallel subagents + main-thread checks)
**Pages created**: `wiki/concepts/{hbm-bandwidth,ridge-point,reduce-scatter,trace-me}.md` (4 stubs).
**Pages updated**:
- `wiki/codebases/tokamax.md` — 2 broken links fixed (`../scaling-book.md` / `../stablehlo.md` → `./scaling-book.md` / `./stablehlo.md`).
- `wiki/codebases/xprof.md` — 5 broken links fixed (missing `2026-` year prefix on xprof source slugs; also dropped stale `(stub — fill once source pages land)` annotations now that the targets exist).
- `wiki/concepts/{mark-step-sync,tensor-parallelism,sharding,ici,collective-communication,all-reduce,kv-cache,static-cache,jax-trace,decode-profile-signature,dcn,attention-block-sizes}.md` — 12 stubs had jax-huggingface and/or ultrascale-playbook sources added to their `## Sources` section.
- `wiki/index.md` — totals (77→81 concepts, 118→122 pages); Performance metrics & roofline (11→13), Parallelism & collectives (11→12), Profiling (10→11).

**Key result**: All broken markdown links in wiki/ eliminated. 0 orphan pages. Wave-2 ↔ human-ingest cross-link asymmetry closed for the known-relevant concept stubs.

**Notes**:
- **Broken-link residue**: 10 intentional placeholder links in `SCHEMA.md` itself (e.g., `../sources/2022-flash-attention.md`, `relative/path.md`, index-template `<slug>` placeholders). Left as-is — they are documentation examples.
- **Orphans**: 0 under `wiki/sources/` and `wiki/concepts/`. Codebase pages are expected to be lightly-linked (mostly from `index.md` and a few sources) and were not scanned as orphan candidates.
- **Unlinked-mention candidates reported but NOT auto-fixed** (judgment required): top 20 opportunities to wrap concept names in prose as markdown links. The main offenders (by raw mention count in prose without a link) are `megascale` (24× in megascale-viewer source), `hbm` (19× in TPU_OPTIMIZATION), `ici` (16× in TPU_OPTIMIZATION), `splash-attention` (12× in its own source page — expected), and `hlo` (12× in hlo-op-stats). Most of these are "first-mention" linking candidates rather than blanket-wrap-every-mention — deferred to per-page editing rather than bulk automation.
- **Cross-link held-backs** (content-unjustified, flagged for future reconsideration):
  - `jax-huggingface-part-1` → `xla-flags` or `hlo-dumping-and-diffing`: source does not discuss either.
  - `jax-huggingface-part-4` → `custom-trace-annotations`: hardware is A100 GPU and the source only mentions profiling as a gap.
  - `ultrascale-playbook` → `int8-quantization`: playbook covers FP8 (DeepSeek tile-scaled), not int8/AQT. Concept mismatch.
- **Submodule commit freshness**: all 8 codebase pages' `commit:` frontmatter matches current `git submodule status`. No drift.
- **Frontmatter `sources:` field**: all 81 concept stubs consistently carry `sources: N`. Schema doesn't require it, but the vault convention is now established and consistent — no reconciliation needed. If desired, a future edit to `SCHEMA.md` could codify the convention.

## [2026-04-22] ingest-source | Ultra-Scale Playbook (Tazi et al., HF, 2025-02-19)

**Op**: ingest-source
**Pages created**:
- `wiki/sources/2025-ultrascale-playbook.md` (primary source page — **first non-2026 slug**: playbook is dated Feb 2025).
- `wiki/concepts/{ring-attention,context-parallelism,sequence-parallelism,pipeline-parallelism,expert-parallelism}.md` — 5 stubs for parallelism concepts not present after Wave 2.

**Pages updated**:
- `wiki/concepts/{rematerialization,flash-attention,splash-attention,fsdp,tensor-parallelism,sharding,async-collectives,dtype-strategy}.md` — appended the new source to `## Sources` with a one-line GPU↔TPU claim.
- `wiki/codebases/{tokamax,torchax,scaling-book,autoresearch}.md` — appended the source under `## Sources` with a per-codebase connection note.
- `wiki/index.md` — added the source under Sources and the 5 new stubs under Concepts.

**Raw artifacts**:
- `raw/sources/2025-ultrascale-playbook.html` (788 KB full HTML capture of the static asset URL).
- `raw/assets/ultrascale-playbook/` — **90 figures, 5.2 MB**. Every `<img>` referenced by the playbook, downloaded from `nanotron-ultrascale-playbook.static.hf.space`. Referenced inline in the source page; the full inventory is tabulated at the bottom of that page.

**Key result**: —

**Notes**:
- Emphasis directed by human: **GPU/PyTorch ↔ TPU/JAX differences in scaling/optimization mechanics**. The source page's centrepiece is a 20-row translation matrix ("Axis | GPU/PyTorch (playbook) | TPU / JAX / XLA | Tuning surface that actually matters on TPU"). Every playbook claim in the Key-claims table is annotated with "Transfers to TPU?" and, where not, the TPU delta.
- GPU-specific sections explicitly flagged as **not transferring**: Section 10 (memory coalescing, tiling, thread coarsening, control divergence, `torch.compile`+Triton) — different programming model; Pallas Mosaic-TPU is our analogue with its own tuning vocabulary.
- The HF Space URL is dynamically rendered; `WebFetch` on the public URL returned a loading shell. The static-asset URL (`nanotron-ultrascale-playbook.static.hf.space/index.html`) served the complete document and all figures.
- **6 hypothesis candidates** surfaced but **not filed** as `hypotheses/*.md` — schema requires a `model:` slug; no model page exists yet. Listed on the source page under `## Gaps & caveats`, to be promoted when the first model page is filed:
  1. Selective activation recomputation via `jax.checkpoint_policies` (Korthikanti 70% / 2.7% claim).
  2. Wire tokamax `ring_attention_kernel` through `dot_product_attention` dispatch (kernel exists, API gap only — Wave 1 finding + this source confirms it).
  3. Zig-Zag Ring Attention on TPU — no implementation found in any ingested codebase; open algorithmic port from Brandon et al. 2023.
  4. TPU-native Pallas kernels for `gated_linear_unit` and `layer_norm` in tokamax — Wave 1 finding that these fall back to XLA reference; this source quantifies the upside category (fused kernels pay off on memory-bound ops).
  5. DeepSeek-V3 tile-scaled FP8 (1×128 activations, 128×128 weights+scales) on v6e MXU.
  6. Expose tokamax splash-attention **backward** block sizes to the autotuner — Wave 1 hidden-tuning-surface finding cross-referenced.
- Concept-page convention in the vault includes a `sources: N` frontmatter integer alongside the `## Sources` H2 section. SCHEMA.md prescribes only the section. My 5 new stubs follow the vault convention (both). Noting the coexistence; not reconciling here.
- Index entry for this source is added alongside the existing Wave 2 rebuild; Concepts section in the index now lists my 5 new stubs explicitly.

## [2026-04-22] ingest-source + stub | Wave 2 — profiling & optimization docs

**Op**: ingest-source (batch) + stub (concept stubs)
**Pages created**: 28 source pages + 72 concept stubs.
- Sources: `wiki/sources/2026-xprof-mcp-tpu-optimization.md`; xprof docs `2026-xprof-{overview-page,trace-viewer,memory-profile,memory-viewer,graph-viewer,utilization-viewer,terminology,hlo-op-stats,hlo-op-profile,framework-op-stats,perf-counters,custom-call-profiling,capturing-profiles,jax-profiling,pytorch-xla-profiling,tensorflow-profiling,docker-deployment,kubernetes-deployment,roofline-model,megascale-stats,megascale-viewer,megascale-viewer-sql}.md`; tokamax docs `2026-tokamax-{supported-ops,basic-usage,splash-attention,autotuning,benchmarking}.md`.
- Concepts: 72 stubs under `wiki/concepts/` grouped as Architecture & hardware (12), Performance metrics & roofline (11), Compiler & HLO (12), Kernels (8), Optimization techniques (7), Parallelism & collectives (7), Inference (5), Profiling (10).
**Pages updated**: `wiki/index.md` (Sources/Codebases/Concepts sections rebuilt; merged with the concurrent `jax-huggingface` ingest).
**Key result**: ~3,475 lines of source content + 72 concept stubs. Wiki now has a working vocabulary — hypothesis candidates can be stated in terms of existing concepts with citations.
**Notes**:
- Six subagents ran in parallel for source ingestion (one per content group); a seventh consolidated concept stubs from their deduplicated recommendations.
- Subagent reports surfaced additional hypothesis candidates beyond Wave 1 findings (not yet filed):
  - **xprof-mcp TPU_OPTIMIZATION**: per-matmul fp32 cast ~17% overhead; int8 shifts v5e critical batch 240→~120→~240; Llama-2-7B decode 8.8× from static-cache; flash attention saves ~32 MB/layer/request at N=4096; selective AC ~+2.7% compute for ~70% activation memory. Most claims are v5e-anchored — generalization to v6e is not pinned down in the source.
  - **tokamax-supported-ops**: `docs/supported_ops.md` lists `dot_product_attention` as GPU-only, but the code ships a TPU Pallas/Mosaic backend — doc is stale. Flagged on the source page.
  - **tokamax-splash-attention / autotuning**: raw docs are 2–3 line stubs; source pages were written from code + basic-usage doc. The autotune backward-pass block-size coverage gap (Wave 1) is now cross-linked from `autotuning.md` and `attention-block-sizes.md`.
  - **xprof-roofline**: the "bytes accessed" definition in arithmetic intensity isn't scoped to a memory tier in the doc; peak FLOPs / bandwidth per TPU generation are pulled from Device Information at runtime and not listed.
  - **xprof-megascale**: `megascale_viewer_sql` has a minor inconsistency between `pt.name` vs `ppt.name` as `device` column across the two example queries — could mislabel rows.
  - **xprof-perf-counters**: doc lists filters/columns but not individual counter semantics — deeper docs/source needed before counter-level hypotheses.
  - **xprof-hlo-op-profile** "wasted time" sort is computed against FLOPs utilization only; will underweight memory-bound ops.
- Concept stubs: 72 created; subagent flagged 4 more worth adding in a follow-up (`hbm-bandwidth`, `ridge-point`, `reduce-scatter`, `trace-me`/TraceMe). Deferred — will add if Wave 3 scaling-book references them in depth.
- No broken markdown links introduced: grep confirmed `reduce-scatter` only appears as prose in `fsdp.md` / `collective-communication.md`, not as a link.
- No schema deviations this wave.
- Concurrency note: the `jax-huggingface` codebase + 4 source pages were ingested by the human (next log entry) while Wave 2 subagents were running; index.md was merged to reflect both. Wave 2 subagents did not see or cross-link to the jax-huggingface pages; Wave 3 (scaling-book) or a dedicated LINT pass should add cross-links from `jax-huggingface-part-{2,3}.md` into the new concept stubs (sharding, tensor-parallelism, static-cache, kv-cache, splash-attention, etc.).

## [2026-04-22] ingest-codebase + ingest-source | jax-huggingface (learning_machine subfolder)

**Op**: ingest-codebase + ingest-source (combined, user-directed option B)
**Pages created**: wiki/codebases/jax-huggingface.md; wiki/sources/2026-jax-huggingface-part-{1,2,3,4}.md
**Pages updated**: wiki/index.md; .gitmodules (added `raw/code/learning-machine` submodule, commit `93328b2`)
**Key result**: 5 pages written. Ingestion scoped to `jax-huggingface/` subfolder of `qihqi/learning_machine`; sibling subprojects (llama_ref, spmd_sharding, torch_pallas, flash_attn_speed, jax_perf, etc.) deferred. First `source/` pages in the wiki — exercised the source template alongside the codebase template.
**Notes**:
- Per-post data points captured in source-page tables: Part 1 v6e 1-chip Llama-2-7B forward 4.37s→13ms; Part 2 8-chip TP 13ms→3.4ms (3.8× cached, blog rounds to 4.3×); Part 3 50-tok decode 130.9s DynamicCache eager → 88.4s StaticCache eager → 14.77s StaticCache+jit (8.9×); Part 4 **A100 GPU, not TPU**: 5.9s→1.07s/image after VAE `methods_to_compile=['decode']` fix.
- **HF API drift flagged:** Part 3 post text's `StaticCache` pytree flattener (`cache.key_cache`/`cache.value_cache`) does NOT match the companion `jax_hg_03.py` (`cache.layers[i].keys`/`.values`). Script is current-HF-API version. Noted on codebase page and Part 3 source page. Candidate observation once a `model/` page exists.
- **Hardware ambiguity in Part 3:** post does not state device for the 130/88/14.77s numbers. Flagged in source-page "Gaps & caveats". Resolving this is a prerequisite before using those numbers as a baseline.
- **Part 4 hardware is A100 GPU.** Kept ingested because the `torchax.compile` / `CompileOptions` / `methods_to_compile` / scheduler-move patterns are TPU-portable, but reported numbers are not. Explicitly flagged on both the codebase page and Part 4 source page.
- Performance-relevant surfaces on the codebase page enumerate 10 concrete anchors (sharding recipe, KV-cache post-prefill sharding, functional_call escape from captured-constants, methods_to_compile override, static_argnames routing, scheduler tensor-move, pytree cookbook, static-arg strategies, default_matmul_precision, profile-capture idiom) — each grounded in a specific file:line.
- No hypotheses / concept stubs filed — no `model/` page yet to attach them to. Per-page "Future hypothesis anchors" sections carry the candidates forward.

## [2026-04-22] ingest-codebase | Wave 1 — seven repos

**Op**: ingest-codebase (batch)
**Pages created**: wiki/codebases/{xprof,xprof-mcp,torchax,tokamax,stablehlo,scaling-book,autoresearch}.md
**Pages updated**: wiki/index.md
**Key result**: 7 codebase parent pages written in parallel. Total 935 lines. Commit SHAs recorded in each page's frontmatter.
**Notes**:
- Per-repo "discuss before writing" step was batched into a single up-front categorization with the human (A=direct ingest, B=methodology, C=book-as-sources) rather than seven separate discussions.
- Scope discipline held: codebase pages map structure; the docs under each repo's `docs/` were deferred to Wave 2 (profiling/optimization) and Wave 3 (scaling-book chapters as sources). No source/concept/hypothesis pages created.
- Noteworthy findings surfaced during ingestion, flagged for follow-up when hypotheses are formulated:
  - **tokamax**: TPU `gated_linear_unit` and `layer_norm` have NO TPU-specific kernel — they silently fall back to the XLA reference. Candidate hypothesis source.
  - **tokamax**: splash-attention backward-pass block sizes (`block_q_dkv`, `block_kv_dkv`, `block_kv_dkv_compute`, `block_q_dq`, `block_kv_dq`) are exposed via `SplashConfig` but NOT surfaced to the autotuner (hidden tuning surface). Note: a Splash Attention kernel is also available upstream in JAX at `jax.experimental.pallas.ops.tpu.splash_attention` — the tokamax copy under `_src/ops/experimental/tpu/splash_attention/` mirrors that implementation, so hypothesis-writers can target either entry point.
  - **tokamax**: `ring_attention_kernel` exists but isn't reachable from `tokamax.dot_product_attention`.
  - **torchax**: `torchax.compile()` modes `dynamo` and `export` raise; only `jax` mode is functional.
  - **scaling-book**: book dated 2025-02-04 → Wave 3 source slugs will use `2025-` prefix, not `2026-`.
- **autoresearch** page uses the reframed H2 title **"Structural surfaces we borrow"** in place of "Performance-relevant surfaces" (the repo is methodology, not TPU-perf content). This is an intentional schema deviation for this single page.

## [2026-04-22] manual | wiki bootstrap

**Op**: manual
**Pages created**: SCHEMA.md, CLAUDE.md, wiki/index.md, wiki/log.md
**Pages updated**: —
**Key result**: —
**Notes**: Bootstrapped autoresearch-oriented schema from scratch. Independent of sibling `tpu_wiki` by design. Loop: sources + codebases + profiles → concepts + models → ranked hypotheses → experiments → observations → revised priors. Next: ingest first codebase and/or file the first model page.
