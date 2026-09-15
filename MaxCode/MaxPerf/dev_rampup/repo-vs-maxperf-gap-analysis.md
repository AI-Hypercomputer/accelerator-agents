<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Gap Analysis: Broader Repo vs max_perf_g

What capabilities exist in `tpu_performance_autoresearch_wiki/` (the repo root) that do NOT exist in `max_perf_g/`.

---

## 1. Knowledge Base (wiki/) — Completely Absent from max_perf_g

The broader repo has a **340-page wiki** with structured page types. max_perf_g has none of this.

| Wiki Layer | Broader Repo | max_perf_g |
|-----------|-------------|------------|
| **Sources** (papers, docs) | `wiki/sources/` — indexed, peer-reviewed knowledge | None. Agents derive from first principles or hallucinate. |
| **Codebases** (indexed repos) | `wiki/codebases/` — 27 repos with performance-relevant file:line references | `raw/code/` exists but no index. Agents explore manually. |
| **Concepts** (living glossary) | `wiki/concepts/` — every technique has definition, mechanism, known results | `dev_rampup/concepts/` teaches the system but doesn't document the ecosystem. |
| **Models** (per-model state) | `wiki/models/` — baseline, current best, open/retired hypotheses, history | All state crammed into one `program.md`. Not composable. |
| **Observations** (reusable findings) | `wiki/observations/` — extracted findings inform multiple future hypotheses | Findings stay inside experiment pages. No extraction. |
| **Analyses** (syntheses) | `wiki/analyses/` — retrospectives, bottleneck indexes, kernel directories | None. The curriculum concepts serve a different purpose (teaching). |

---

## 2. Reference Infrastructure — Underutilized

The repo contains 27 ingested codebases under `raw/code/` (JAX, tokamax, maxtext, tpu-inference, sglang-jax, etc.) and hundreds of reference assets (ultrascale playbook SVGs, profiling diagrams).

max_perf_g does NOT:
- Provide **indexed entry points** to these codebases
- Reference existing **Pallas kernel implementations** as a lookup table (the broader repo has `wiki/analyses/2026-04-23-pallas-kernel-directory.md` cataloging 200+ kernels)
- Link curriculum concepts to upstream docs or source files

---

## 3. Operational Continuity — No Index or Log

| Feature | Broader Repo | max_perf_g |
|---------|-------------|------------|
| **Cross-section index** | `wiki/index.md` — all pages, model status, hypothesis counts, updated on every write | Nothing. |
| **Operational log** | `wiki/log.md` — append-only, newest-first, one entry per operation | Nothing. |
| **Audit trail** | "What experiments ran? What's open? What did we learn?" answered in seconds | Must manually scan RESULTS.tsv (raw numbers, no narrative). |

---

## 4. Tooling — No Automation Scripts

The broader repo provides (documented in `CLAUDE_CODE_INSTRUCTIONS.md`):

| Tool | Purpose | max_perf_g equivalent |
|------|---------|----------------------|
| `verify_layout.sh` | Validates repo structure, detects drift | None |
| `new_experiment.sh` | Scaffolds experiment directory + stub pages + git branch | None |
| `append_result.py` | Validates and appends RESULTS.tsv rows | None |
| Pre-commit hooks | Shellcheck, file-size checks, layout verification | None |
| Bash runner scripts | Env var validation, logging, placeholder detection | Empty `runs/` directory |

---

## 5. Multi-Campaign Capability — Single Model Only

| Aspect | Broader Repo | max_perf_g |
|--------|-------------|------------|
| Models supported | Multiple (Llama 3 8B, Gemma 4 E4B in current wiki) | Single (Qwen3-Coder-480B) |
| Hypothesis queues | Per-model | One queue in program.md |
| Experiment trees | Per-model folders | Single `experiments/` directory |
| Cross-model learning | Observations/analyses reusable across models | No mechanism for reuse |

---

## 6. Knowledge Discovery — No Visual UI

The broader repo is also an **Obsidian vault** (`.obsidian/` config exists). Same markdown files render as a visual graph with backlinks, full-text search, and navigable links. max_perf_g has no equivalent discovery interface beyond reading files.

---

## 7. Capability Impact Summary

| Capability | Broader Repo Can Do It | max_perf_g Cannot |
|-----------|----------------------|-------------------|
| Rapid literature lookup | grep `wiki/sources/`, read related pages | Agents rediscover or hallucinate sources |
| Codebase surface reference | `wiki/codebases/` has indexed file:line refs | Agents explore `raw/code/` manually — slow |
| Find existing Pallas kernel | Kernel directory catalogs 200+ by perf/stability | No directory; agents don't know what exists |
| Trace hypothesis to reference | observation → experiment → concept → source paper | Each experiment is isolated |
| Multi-experiment synthesis | `wiki/analyses/` compares N experiments, extracts patterns | RESULTS.tsv has raw numbers; no narrative |
| Track known bugs per model | Model page lists "NaN at seq ≥2048", checkpoint quirks | No per-model issue tracking |
| Audit last 10 decisions | grep `wiki/log.md` | Cannot trace decisions |
| Scale beyond 10 experiments | Tooling prevents structural drift | Manual file creation; error-prone |

---

## 8. Design Philosophy

This gap is **intentional**:

- **max_perf_g** is a **tactical optimization system** — lean, focused, one model, one campaign. Clear roles (7 agents), structured control flow (12-step loop), measurable output (RESULTS.tsv).

- **The broader repo** is a **strategic learning platform** — ingests external knowledge, builds a composable knowledge base, scales across models, maintains operational continuity.

The tradeoff: max_perf_g agents **cannot reference the broader repo's knowledge** without manual context-passing, and the broader repo's tools (Obsidian, LINT, cross-linking) cannot audit max_perf_g's state.

---

## 9. What Would Close the Gap

If max_perf_g wanted to inherit these capabilities:

1. **Link to wiki/** — add `wiki_root: ../../wiki/` reference so agents can search sources, codebases, observations
2. **Add index.md + log.md** — lightweight operational continuity within max_perf_g
3. **Add observation extraction** — after each experiment, extract reusable findings into a local `observations/` directory
4. **Add tooling scripts** — `new_experiment.sh`, `verify_layout.sh`, `append_result.py` in a `tools/` directory
5. **Import the kernel directory** — link or copy the Pallas kernel directory into max_perf_g for MaxKernel reference
6. **Add codebase index stubs** — at minimum, index the 3-4 codebases MaxKernel/AutoRefactor actually modify
