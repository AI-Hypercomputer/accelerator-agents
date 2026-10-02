---
title: 'Module: examples/_grad_of_attention.py'
type: catalog
provenance: extracted
module: examples/_grad_of_attention.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `examples._grad_of_attention`/
symbols:
  weights: weights.
  env: env.
  crossent: crossent().
  attn: attn().
  k: k.
  m: m.
  mjit: mjit.
  f: f().
  shape: shape.
  M.forward: M#forward().
  graded: graded.
  q: q.
  v: v.
  M: M#
  M.a: M#a.
  M.__init__: M#__init__().
---
# Module: [`examples/_grad_of_attention.py`](../../../../../raw/code/torchax/examples/_grad_of_attention.py)

## Classes
### `M`  ·  implements/extends Module
- def: [`examples/_grad_of_attention.py:45`](../../../../../raw/code/torchax/examples/_grad_of_attention.py#L45)
- signature: `class M(torch.nn.Module):`
- members:
  - `forward(self, x)` — [`L50`](../../../../../raw/code/torchax/examples/_grad_of_attention.py#L50)
  - `a` — [`L48`](../../../../../raw/code/torchax/examples/_grad_of_attention.py#L48)
- protocol/private: `__init__`[`L46`](../../../../../raw/code/torchax/examples/_grad_of_attention.py#L46)
- used by: (1 test-only callers)

## Functions
- `attn(q, k, v)` — [`L34`](../../../../../raw/code/torchax/examples/_grad_of_attention.py#L34)
- `crossent(x, y)` — [`L67`](../../../../../raw/code/torchax/examples/_grad_of_attention.py#L67)
- `f(weights, x)` — [`L62`](../../../../../raw/code/torchax/examples/_grad_of_attention.py#L62)

## Module values
- `env` — [`L23`](../../../../../raw/code/torchax/examples/_grad_of_attention.py#L23)
- `graded` — [`L73`](../../../../../raw/code/torchax/examples/_grad_of_attention.py#L73)
- `k` — [`L78`](../../../../../raw/code/torchax/examples/_grad_of_attention.py#L78)
- `m` — [`L54`](../../../../../raw/code/torchax/examples/_grad_of_attention.py#L54)
- `mjit` — [`L57`](../../../../../raw/code/torchax/examples/_grad_of_attention.py#L57)
- `q` — [`L76`](../../../../../raw/code/torchax/examples/_grad_of_attention.py#L76)
- `shape` — [`L75`](../../../../../raw/code/torchax/examples/_grad_of_attention.py#L75)
- `v` — [`L77`](../../../../../raw/code/torchax/examples/_grad_of_attention.py#L77)
- `weights` — [`L81`](../../../../../raw/code/torchax/examples/_grad_of_attention.py#L81) — documented in [torchax-interop](../../concepts/torchax-interop.md)

