---
title: "LLM Wiki — a pattern for building personal knowledge bases with LLMs"
type: source
tags: [article, methodology, knowledge-base, llm-wiki, schema, obsidian]
author: Andrej Karpathy
upstream: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
created: 2026-04-23
updated: 2026-04-23
---

Karpathy's "idea file" describing the **LLM-wiki pattern**: an agent incrementally builds and maintains a persistent, interlinked markdown wiki that sits between the human's raw sources and the human's queries — *compiling* knowledge once and keeping it current rather than re-deriving it on every RAG query. This wiki's [SCHEMA.md](../../SCHEMA.md) is a domain-specialized instantiation of this pattern; ingesting the article makes the source-of-design explicit and ties the methodology back to its origin.

## Overview

The document is deliberately abstract — it describes the *pattern*, not a specific implementation — and explicitly invites the reader to hand it to an LLM agent and co-design the specifics. The core claim is that RAG-style "retrieve relevant chunks, re-derive answer" systems do not accumulate, while a wiki-style "compile once, maintain continuously" system does. The human curates sources and asks questions; the LLM does **everything else** — ingestion, cross-referencing, summarizing, filing, log-keeping, lint, synthesis.

The pattern decomposes cleanly into:

- **Three layers.** (1) Raw sources (immutable, human-curated); (2) the wiki (LLM-generated markdown, entirely LLM-owned); (3) the schema (a `CLAUDE.md` / `AGENTS.md` file encoding conventions and workflows, co-evolved).
- **Four operations.** Ingest, query, lint, (optionally) tooling (e.g. `qmd`-style local search once the wiki outgrows an index file).
- **Two navigation files.** `index.md` (content-oriented catalog) and `log.md` (append-only chronological record).
- **A UI affordance.** Obsidian as the read-only reader beside the LLM's write-only CLI — one side writes, the other reads, in real time.

Karpathy argues the pattern fits personal notes, research, books-as-you-read, team/business wikis, competitive analysis, hobby deep-dives — *"anything where you're accumulating knowledge over time and want it organized rather than scattered"*. The thing LLMs contribute that humans can't sustain is the **maintenance burden** (updating cross-refs across dozens of pages on every ingest), which is why human-maintained wikis traditionally decay.

## Key claims

1. **RAG retrieves and re-derives; wikis compile and compound.** On every query, RAG starts from scratch; a wiki is already cross-referenced, already synthesized, already contradiction-flagged. The difference compounds over weeks/months of sourcing.
2. **The LLM owns the wiki layer entirely.** The human never (or rarely) writes wiki pages. The human curates sources and directs questions.
3. **The schema file is the key configuration.** A `CLAUDE.md`/`AGENTS.md` that encodes structure + conventions + workflows is what turns a generic chatbot into a disciplined wiki maintainer. Co-evolves with the human over time.
4. **Ingestion touches many pages.** A single source can update 10–15 wiki pages (summary, entity pages, concept pages, index, log, cross-refs).
5. **Good query answers should be filed back as pages.** Comparisons, analyses, discovered connections — file them as new wiki pages so exploration compounds just like ingestion does.
6. **Lint is part of the loop.** Periodic health-checks: contradictions, stale claims superseded by newer sources, orphan pages, missing concept pages, broken cross-refs, data gaps suggesting new sources.
7. **Two special files suffice at moderate scale.** `index.md` (catalog) + `log.md` (timeline). "~100 sources, ~hundreds of pages" works without embedding-based RAG infrastructure.
8. **Chronological prefixes make logs greppable.** `## [YYYY-MM-DD] op | subject` prefix → `grep "^## \[" log.md | tail -5` gives recent events. Explicitly modeled in this wiki's `log.md`.
9. **Obsidian is the viewing UI; the LLM is the editor.** The human reads the wiki through Obsidian's graph/link views while the LLM writes in the CLI.
10. **Maintenance cost near zero is what makes it work.** Humans abandon wikis because maintenance grows faster than value; LLMs don't. Vannevar Bush's Memex (1945) couldn't solve "who does the maintenance"; LLMs do.

## Key data points

This is an idea file, not a measurement document — there are no numbers. The closest operational commitments:

| Item | Value |
|---|---|
| Working scale for plain index+log (no search infra) | ~100 sources, ~hundreds of pages |
| Pages updated per ingest (typical) | 10–15 |
| Required file structure | `raw/` + wiki dir + schema file (e.g. `CLAUDE.md`) |
| Navigation files | `index.md` (content) + `log.md` (chronological) |
| Log entry prefix (enables grep) | `## [YYYY-MM-DD] <op> \| <subject>` |
| Optional tooling mentioned | `qmd` (hybrid BM25/vector local search, CLI + MCP server) |
| Inspirational reference | [Tolkien Gateway](https://tolkiengateway.net/wiki/Main_Page) (community-built fan wiki) |
| Conceptual ancestor | Vannevar Bush's Memex (1945) |

## Techniques referenced

- **Three-layer separation (raw / wiki / schema).** Raw is immutable; wiki is LLM-owned; schema is co-evolved. This wiki's layout mirrors this exactly: [raw/](../../raw/) immutable, [wiki/](../) LLM-written, [SCHEMA.md](../../SCHEMA.md) + [CLAUDE.md](../../CLAUDE.md) co-evolved.
- **Ingest / query / lint** as the three workflows.
- **Contradiction flagging** as a first-class mechanic (rather than silent overwrite) — this wiki's `[!warning] Contradicted by ...` block is the concrete implementation.
- **Query-outputs-as-filed-pages** — this wiki's [analyses/](../analyses/) directory is the concrete landing zone for that pattern.
- **Greppable log prefixes** — this wiki's [log.md](../log.md) follows `## [YYYY-MM-DD] <op> | <subject>` verbatim.
- **Obsidian side-by-side** — the 2026-04-22 log notes "Connect Obsidian" was part of the wiki's initial setup.
- **Local markdown search (`qmd`)** — mentioned as optional upgrade-path; not yet adopted in this wiki.
- **Image handling via Obsidian Web Clipper + hotkey downloader** — mentioned as optional; not used here directly, but the wiki does save images under `raw/assets/` (e.g. `raw/assets/ultrascale-playbook/`).

## Gaps & caveats

- **Abstract, not prescriptive.** Karpathy explicitly says "The document's only job is to communicate the pattern. Your LLM can figure out the rest." No specific directory schema, page format, or workflow is mandated — this wiki's [SCHEMA.md](../../SCHEMA.md) is one instantiation of many possible.
- **No measured evidence of superiority over RAG.** The claim that wikis compound better than RAG is argued, not measured. No retrieval-quality, maintenance-cost, or synthesis-depth benchmarks. For this wiki's **autoresearch** use case, this doesn't matter much — the wiki is the agent's working memory, not a query-answering system — but it's worth flagging that "wiki > RAG" is a stated belief, not a tested one.
- **Scale limits are informal.** "~100 sources, ~hundreds of pages" before index+log breaks down is a rough prior, not a measurement. This wiki is currently at 33 sources + 81 concepts + 8 codebases + 1 model-program + 1 analysis (124 pages per the [index header](../index.md)) — still inside the claimed regime, but no leading indicator is given for when to adopt search tooling.
- **Specific to text-heavy knowledge.** The pattern is least compelling when sources are primarily numeric, tabular, or visual. The wiki for which this ingestion exists is ~70% text (documentation + papers) and ~30% tabular/numeric (experiment ledgers, profile numbers); the pattern still applies, but the concept-vs-table balance is a per-domain decision the source does not address.
- **Does not address adversarial drift.** If a sequence of ingested sources contradict each other, the agent adjudicates via `[!warning]` blocks awaiting human arbitration. The source does not discuss what happens when no human arbitrates for extended periods — in this wiki's autoresearch loop, where 33 experiments landed in one day, that's a live risk.
- **No TPU / performance content.** This is a methodology source, out of this wiki's primary scope (TPU perf). Ingested because [SCHEMA.md](../../SCHEMA.md) is a direct descendant of this document — understanding the parent pattern is useful when editing the schema.

## Connections

This source **directly informs** the wiki's operating protocol:
- [SCHEMA.md](../../SCHEMA.md) — specializes this pattern for TPU-perf autoresearch. The three-layer separation (raw/wiki/schema), the ingest/query/lint operations, the index+log navigation pair, the contradiction-flag convention, and the log-entry prefix format are all carried over verbatim. The autoresearch-specific additions — hypothesis / experiment / observation / model-program page types, "every hypothesis is falsifiable", "no model-quality optimizations", profile-mandatory rule — are the TPU-perf specialization.
- [CLAUDE.md](../../CLAUDE.md) — the `@SCHEMA.md` pointer file Karpathy describes ("e.g. CLAUDE.md for Claude Code or AGENTS.md for Codex").
- [wiki/index.md](../index.md) — the "content-oriented catalog" file from the source's "Indexing and logging" section.
- [wiki/log.md](../log.md) — the "append-only chronological record" with the prefix-pattern Karpathy recommends.

Related ingested material:
- [autoresearch (codebase)](../codebases/autoresearch.md) — Karpathy's parallel reference implementation of the **autoresearch** loop (experiment-ranking, verdicts, priors). This wiki's autoresearch mechanics borrow from that codebase; its maintenance mechanics borrow from the LLM-wiki article. The two documents are complementary: one says "how to run experiments", the other says "how to keep the knowledge base sane while doing so".

## See also

- [autoresearch (codebase)](../codebases/autoresearch.md) — the companion methodology source.
- [SCHEMA.md](../../SCHEMA.md) — the direct implementation of this pattern for this wiki.

## Sources

- `raw/sources/2026-karpathy-llm-wiki.md` (local copy, 75 lines)
- Upstream: <https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>
