---
title: levanter.inference.engine — the prefill-then-decode continuous-batching generation loop
type: concept
provenance: mixed
concept: lib-levanter-src-levanter-inference-engine
updated: 2026-07-03
status: fresh
---
# levanter.inference.engine — the prefill-then-decode continuous-batching generation loop

<!-- connect:up:begin -->
> **Cross-repo concept:** part of [continuous-batching](../../../concepts/continuous-batching.md) across this wiki's repos.
<!-- connect:up:end -->
## Overview

[`InferenceEngine.generate`](../catalog/lib/levanter/src/levanter/inference/engine.md#InferenceEngine.generate)
("Generate tokens for a batch of Requests") is the top-level serving loop: it admits requests from a
queue into free slots via prefill
([`InferenceEngine._prefill_batch`](../catalog/lib/levanter/src/levanter/inference/engine.md#InferenceEngine._prefill_batch)
→
[`_prefill_kernel`](../catalog/lib/levanter/src/levanter/inference/engine.md#_prefill_kernel)), then
runs a compiled autoregressive decode loop
([`_run_generation_loop`](../catalog/lib/levanter/src/levanter/inference/engine.md#_run_generation_loop))
until every admitted sequence finishes or a round budget is exhausted. State is carried in one
[`GenState`](../catalog/lib/levanter/src/levanter/inference/engine.md#GenState) container, and both
prefill and decode ultimately bottom out in the same
[`DecodeState.update_tokens`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#DecodeState.update_tokens)
call (see [lib-levanter-src-levanter-inference-jit_scheduler](lib-levanter-src-levanter-inference-jit_scheduler.md)).

## Diagram

```mermaid
sequenceDiagram
  participant Caller
  participant Engine as InferenceEngine
  participant Prefill as _prefill_kernel
  participant Loop as _run_generation_loop
  Caller->>Engine: generate(requests)
  Engine->>Engine: _prefill_batch (admit from queue into free slots)
  Engine->>Prefill: _prefill_kernel(gen_state, model, sampler, queue)
  Prefill->>Prefill: allocate_for_seq -> model fwd -> update_tokens
  Engine->>Loop: _run_generation_loop(gen_state, model, sampler, max_tokens_per_round, max_rounds)
  loop until all finished or max_rounds
    Loop->>Loop: body: allocate_for_seq -> model fwd -> update_tokens
  end
  Loop-->>Engine: (GenState, _DecodeOutputs)
  Engine->>Engine: _extract_outputs
  Engine-->>Caller: GenerationResult
```

## Design rationale (why it's built this way)

**Prefill runs against a *fresh, local* token queue, separate from the main decode queue, precisely
so newly prefilled sequences' first tokens flow through the same `update_tokens` path as decode-time
tokens.** [`_prefill_kernel`](../catalog/lib/levanter/src/levanter/inference/engine.md#_prefill_kernel)'s
doc states this directly: "Run prefill using a fresh, local token queue. Newly sampled tokens are
enqueued to the main decode queue via update_tokens" — this means prefill and decode don't need two
separate "commit a sampled token" implementations.

**`_run_generation_loop` is `jax.jit`-compiled with `donate_argnames=("gen_state",)` and
`static_argnums` on the round/token-budget integers — the whole multi-round decode loop is one
compiled program, not a Python loop issuing one call per step.**
[`_run_generation_loop`](../catalog/lib/levanter/src/levanter/inference/engine.md#_run_generation_loop)'s
doc — "Run autoregressive generation until all sequences finish or `max_rounds` reached" — combined
with the `donate_argnames` decoration means `gen_state`'s buffers are reused in place across the
internal `cond`/loop iterations rather than being copied each round.

**Multi-sample-per-prompt decoding ("clones") is handled as a distinct post-sampling step, not
folded into the main per-token sampling path.**
[`_handle_clones`](../catalog/lib/levanter/src/levanter/inference/engine.md#_handle_clones)'s doc —
"Sample alternative tokens for the given logits, slot_ids, pos_ids, and clone_targets" — runs after
the primary sample, calling `copy_pages_for_updated_seq` to give each clone its own physical KV pages
diverging from the shared prefix, and `find_src` to trace a clone back to its origin sequence.

> [!inferred] [`_prefill_batch`](../catalog/lib/levanter/src/levanter/inference/engine.md#InferenceEngine._prefill_batch)'s
> doc — "Admit a batch from the head of the queue that fits in free slots/pages" — implies admission
> is capacity-checked against *both* free decode slots and free KV pages before a request is pulled
> off the queue, not admitted speculatively and rolled back on OOM.

## Entry points

- [`InferenceEngine.generate`](../catalog/lib/levanter/src/levanter/inference/engine.md#InferenceEngine.generate) —
  the sole public entry point; takes a batch of `Request`s and an optional `step_callback`, returns a
  `GenerationResult`.
- [`_prefill_kernel`](../catalog/lib/levanter/src/levanter/inference/engine.md#_prefill_kernel) —
  called once per admitted prefill batch.
- [`_run_generation_loop`](../catalog/lib/levanter/src/levanter/inference/engine.md#_run_generation_loop) —
  called once per `generate()` call, running until completion or `max_rounds`.

## Mechanism (step-by-step)

1. **`generate` admits as many queued requests as fit into free slots/pages via `_prefill_batch`**,
   which calls
   [`_prefill_prompts`](../catalog/lib/levanter/src/levanter/inference/engine.md#InferenceEngine._prefill_prompts)
   ("Pack prompt work into a single `PrefillWork` structure for downstream device execution") to
   batch the admitted prompts, then
   [`_prefill_kernel`](../catalog/lib/levanter/src/levanter/inference/engine.md#_prefill_kernel) to
   actually run them.
2. **`_prefill_kernel` allocates pages
   ([`allocate_for_seq`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#SequenceTable.allocate_for_seq)),
   runs the model forward pass, and commits sampled tokens via
   [`update_tokens`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#DecodeState.update_tokens)**,
   handling clone sequences via `_handle_clones` along the way.
3. **`_run_generation_loop`'s `body` repeats the same allocate→forward→`update_tokens` cycle each
   round**, tracked in [`GenState`](../catalog/lib/levanter/src/levanter/inference/engine.md#GenState)'s
   [`decode_state`](../catalog/lib/levanter/src/levanter/inference/engine.md#GenState.decode_state)
   field, until a `cond` check finds every sequence finished or `max_rounds` is hit.
4. **Outputs accumulate in a `_DecodeOutputs` via
   [`append`](../catalog/lib/levanter/src/levanter/inference/jit_scheduler.md#_DecodeOutputs.append)
   each round**, tracking a `finished_snapshot` alongside the new tokens/logprobs.
5. **After the loop exits,
   [`_extract_outputs`](../catalog/lib/levanter/src/levanter/inference/engine.md#InferenceEngine._extract_outputs)
   appends newly available tokens into the caller-facing
   result, keyed by `(request_id, child_id)`** — the `child_id` component is what distinguishes a
   clone's output stream from its source sequence's.

## Key data structures

- **[`GenState`](../catalog/lib/levanter/src/levanter/inference/engine.md#GenState)** — "Container for
  generation state used during decoding"; holds
  [`decode_state`](../catalog/lib/levanter/src/levanter/inference/engine.md#GenState.decode_state) (a
  `DecodeState`) plus the page table and per-round bookkeeping.
- **`_DecodeOutputs`** — the accumulator `append`ed to each round; carries new tokens, slot ids,
  logprobs, and a finished-flags snapshot.
- **[`LmHeadModel`](../catalog/lib/levanter/src/levanter/models/lm_model.md#LmHeadModel)** — the model
  interface `_prefill_kernel`/`_run_generation_loop` call forward through; any concrete architecture
  (Llama, Olmo2, GPT-2, Mistral) satisfies this interface.

## Dynamics (design intent)

`_run_generation_loop`'s `donate_argnames=("gen_state",)` means the caller's `gen_state` reference is
invalidated after the call — consistent with the whole decode loop being one compiled, buffer-reusing
program rather than a sequence of independent, buffer-copying steps.

## Edge cases

- [`_prefill_kernel`](../catalog/lib/levanter/src/levanter/inference/engine.md#_prefill_kernel) is
  bounded by `max_seqs_in_prefill` — a prefill batch larger than this cap does not get admitted in one
  shot (per `_prefill_batch`'s capacity-fitting logic).

## Open questions

- The exact criterion `_prefill_batch` uses to decide a request "fits" (free slots vs. free pages vs.
  a combined budget) isn't fully resolved by the symbols in this packet's subgraph.

## See also
- [lib-levanter-src-levanter-inference-jit_scheduler](lib-levanter-src-levanter-inference-jit_scheduler.md) —
  `DecodeState`/`SequenceTable`/`PageTable`, the state this engine drives.
- [root](root.md) — the model-side `Attention.paged_decode` this engine's forward passes call into.
