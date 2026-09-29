---
title: levanter.inference.jit_scheduler — the fully-jitted continuous-batching decode scheduler
type: concept
provenance: mixed
concept: lib-levanter-src-levanter-inference-jit_scheduler
updated: 2026-07-03
status: fresh
---
# levanter.inference.jit_scheduler — the fully-jitted continuous-batching decode scheduler

<!-- connect:up:begin -->
> **Cross-repo concept:** part of [continuous-batching](../../../concepts/continuous-batching.md) across this wiki's repos.
<!-- connect:up:end -->
## Overview

[`DecodeState`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#DecodeState) is the
"hot set" of sequences currently being decoded — its own docstring: "State of sequences during
decoding. This manages a 'hot set' of sequences that are currently being decoded." It composes a
[`SequenceTable`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#SequenceTable)
("Compact view over per-sequence metadata"), a
[`PageTable`](../catalog/lib/levanter/src/levanter/inference/page_table.md#PageTable) ("Global KV
page allocator tracking only per-page reference counts"), and a `TokenQueue`. Every mutating operation
— assigning a new sequence to a slot, allocating pages, freeing pages, appending decoded tokens — is
`@eqx.filter_jit`-compiled, meaning the entire scheduling logic (not just the model forward pass) runs
inside JAX's traced/compiled world.

## Diagram

```mermaid
flowchart TD
  DS["DecodeState\n(sequences: SequenceTable, tqueue: TokenQueue)"]
  DS -->|assign_seq| ASSIGN["SequenceTable.assign_slot\n(bind seq_len/kv_pages/page_indices to a local slot)"]
  DS -->|allocate_for_seq| ALLOC["SequenceTable.allocate_for_seq(PageTable)\n-> (SequenceTable, PageTable, PageBatchInfo)"]
  ALLOC --> PT["PageTable\n(per-page ref counts)"]
  DS -->|update_tokens| UPD["TokenQueue.enqueue_tokens\n+ SequenceTable.seq_lens update"]
  DS -->|free_pages| FREE["SequenceTable.free_pages(PageTable)\n(donate=\"all\")"]
  FREE --> PT
```

## Design rationale (why it's built this way)

**`INVALID` is a single reserved sentinel integer used uniformly across slot ids, page indices, and
token ids, rather than `None`/`Optional`, because these values live inside JAX arrays that must have a
concrete dtype under `jit`.**
[`is_valid`](../catalog/lib/levanter/src/levanter/inference/utils.md#is_valid)'s doc — "Returns a
boolean array indicating whether each token in the input is valid" — and
[`purge`](../catalog/lib/levanter/src/levanter/inference/utils.md#purge)'s doc — "Set elements of the
array to `invalid` where the `mask` is True and slides the rest to the front" — both operate purely on
this sentinel rather than on Python-level `None` checks, since a JAX array can't hold a mix of
`int`/`None`.

**`free_pages` is JIT-compiled with `donate="all"`, meaning its input buffers are consumed and may be
reused for the output — an explicit performance choice for an operation that runs on every sequence
completion.** [`SequenceTable.free_pages`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#SequenceTable.free_pages)'s
`@eqx.filter_jit(donate="all")` decorator tells XLA it may alias input and output buffers rather than
allocating fresh ones — appropriate because the caller doesn't need the pre-free `SequenceTable`/`PageTable`
state once pages are freed.

**Sequence indices in [`PageBatchInfo`](../catalog/lib/levanter/src/levanter/inference/page_table.md#PageBatchInfo)
are batch-local, not `DecodeState`-global — recovering the global mapping requires `slot_ids`.** This
is exactly the same page-batch abstraction attention's `paged_decode` consumes (see
[root](root.md)); the scheduler here is what constructs it, via
[`SequenceTable._create_batch_info`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#SequenceTable._create_batch_info).

> [!inferred] `purge`'s "slides the rest to the front" behavior (compacting a queue after removing
> invalid entries) suggests `TokenQueue`/`SequenceTable` maintain a dense, front-packed layout as an
> invariant — freed slots don't leave permanent holes that later allocation logic would need to
> special-case.

## Entry points

- [`DecodeState.assign_seq`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#DecodeState.assign_seq) —
  binds a brand-new sequence (its initial tokens, optional pre-existing KV pages for a cloned
  sequence, and [`SeqDecodingParams`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#SeqDecodingParams))
  to a free local slot; called once per newly admitted request.
- [`SequenceTable.allocate_for_seq`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#SequenceTable.allocate_for_seq) /
  [`DecodeState.allocate_for_seq`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#DecodeState.allocate_for_seq) —
  allocates KV pages from the shared `PageTable` for a batch of new tokens and produces the
  `PageBatchInfo` the model's `paged_decode` consumes; called once per generation step.
- [`DecodeState.update_tokens`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#DecodeState.update_tokens) —
  appends newly sampled tokens (and their log-probs) into the token queue and advances `seq_lens`;
  called once per generation step after the model forward pass.
- [`SequenceTable.free_pages`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#SequenceTable.free_pages) —
  releases a finished sequence's pages back to the `PageTable`.

## Mechanism (step-by-step)

1. **A new sequence is bound to a slot via `assign_seq`, which delegates to
   [`SequenceTable.assign_slot`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#SequenceTable.assign_slot).**
   `assign_slot` records `seq_len`, `kv_pages`, `page_indices` for the slot, and — if
   `clone_source != INVALID` — the source slot a sequence is being cloned from (e.g. for
   multi-sample-per-prompt decoding sharing a prefix's KV cache).
2. **Each generation step, `allocate_for_seq` walks a per-page loop
   ([`SequenceTable.body`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#SequenceTable.body))
   to grow each active sequence's page allocation as needed**, producing an updated
   `SequenceTable`/`PageTable` plus a
   [`PageBatchInfo`](../catalog/lib/levanter/src/levanter/inference/page_table.md#PageBatchInfo) via
   [`_create_batch_info`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#SequenceTable._create_batch_info).
3. **The model runs its forward pass against that
   [`PageBatchInfo`](../catalog/lib/levanter/src/levanter/inference/page_table.md#PageBatchInfo)**
   (outside this packet's own
   subgraph — see [root](root.md)'s `Attention.paged_decode`), producing new tokens.
4. **`update_tokens` enqueues the new tokens into the `TokenQueue`
   ([`TokenQueue.enqueue_tokens`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#TokenQueue.enqueue_tokens))
   and updates each affected sequence's length**, purging any sequence that has hit
   [`stop_tokens`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#DecodeState.stop_tokens)
   or otherwise finished (`finished`, tracked alongside `sequences`).
5. **A finished sequence's pages are released via
   [`SequenceTable.free_pages`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#SequenceTable.free_pages)**,
   `donate="all"`-compiled so the
   freed buffers can be reused immediately by the next allocation.

## Key data structures

- **[`DecodeState`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#DecodeState)** —
  [`sequences`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#DecodeState.sequences)
  (a `SequenceTable`), `stop_tokens`, plus the token queue.
- **[`SequenceTable`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#SequenceTable)** —
  [`seq_lens`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#SequenceTable.seq_lens),
  [`page_indices`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#SequenceTable.page_indices),
  a `used_mask`, `page_ref_counts`, `clone_sources`; a fixed
  [`max_seqs`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#SequenceTable.max_seqs)
  slot capacity.
- **[`PageTable`](../catalog/lib/levanter/src/levanter/inference/page_table.md#PageTable)** — global
  per-page reference counts; "tracking only per-page reference counts" per its own docstring, i.e. it
  does not itself know which sequence owns a page (that's `SequenceTable.page_indices`'s job).
- **[`SeqDecodingParams`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#SeqDecodingParams)** —
  per-sequence decode-time parameters (temperature, top-p, stop tokens, PRNG key) attached at
  `assign_seq` time.

## Dynamics (design intent)

Every scheduler-mutating method is `@eqx.filter_jit`, meaning slot assignment, page allocation, token
enqueueing, and page freeing are all part of the same traced/compiled program the model forward pass
runs in — there is no host-side Python loop mutating scheduler state between JAX calls; the whole
continuous-batching step is one (or a small, fixed number of) compiled call(s).

## Edge cases

- [`SequenceTable.free_pages`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#SequenceTable.free_pages)
  donates all its input buffers — callers must not hold onto a reference to the pre-call
  `SequenceTable`/`PageTable` and expect it to remain valid/unaliased afterward.

## Open questions

- The exact interaction between `clone_source`-based slot cloning and page ref-counting (whether
  cloned sequences share physical pages copy-on-write, or copy immediately) isn't fully resolved by
  the symbols in this packet's subgraph alone.

## See also
- [root](root.md) — `Attention.paged_decode`, the model-side consumer of the `PageBatchInfo` this
  scheduler produces.
- [lib-levanter-src-levanter-inference-engine](lib-levanter-src-levanter-inference-engine.md) — the
  generation loop (`_prefill_kernel`, `_run_generation_loop`) driving this scheduler.
