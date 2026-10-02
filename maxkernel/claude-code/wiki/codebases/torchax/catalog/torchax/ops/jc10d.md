---
title: 'Module: torchax/ops/jc10d.py'
type: catalog
provenance: extracted
module: torchax/ops/jc10d.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `torchax.ops.jc10d`/
symbols:
  op: op().
  op.inner: op().inner().
  _c10d_all_gather: _c10d_all_gather().
  _c10d_all_reduce: _c10d_all_reduce().
  _c10d_broadcast: _c10d_broadcast().
  _c10d_wait_tensor: _c10d_wait_tensor().
---
# Module: [`torchax/ops/jc10d.py`](../../../../../../raw/code/torchax/torchax/ops/jc10d.py)

## Functions
- `_c10d_all_gather(input, group_size: int, group_name: str)` — [`L32`](../../../../../../raw/code/torchax/torchax/ops/jc10d.py#L32)
- `_c10d_all_reduce(self, reduceOp: str, group_name: str)` — [`L37`](../../../../../../raw/code/torchax/torchax/ops/jc10d.py#L37)
- `_c10d_broadcast(self, src: int, group_name: str)` — [`L52`](../../../../../../raw/code/torchax/torchax/ops/jc10d.py#L52)
- `_c10d_wait_tensor(tensor)` — [`L62`](../../../../../../raw/code/torchax/torchax/ops/jc10d.py#L62)
- `inner(func)` — [`L23`](../../../../../../raw/code/torchax/torchax/ops/jc10d.py#L23)
- `op(*aten, **kwargs)` — [`L22`](../../../../../../raw/code/torchax/torchax/ops/jc10d.py#L22)

