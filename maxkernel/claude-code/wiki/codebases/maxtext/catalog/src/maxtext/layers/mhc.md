---
title: 'Module: src/maxtext/layers/mhc.py'
type: catalog
provenance: extracted
module: src/maxtext/layers/mhc.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.layers.mhc`/
symbols:
  ManifoldConstrainedHyperConnections.res_mapping: ManifoldConstrainedHyperConnections#res_mapping().
  ManifoldConstrainedHyperConnections.__call__: ManifoldConstrainedHyperConnections#__call__().
  ManifoldConstrainedHyperConnections.k: ManifoldConstrainedHyperConnections#k.
  ManifoldConstrainedHyperConnections.mhc_norm: ManifoldConstrainedHyperConnections#mhc_norm.
  ManifoldConstrainedHyperConnections.weight_dtype: ManifoldConstrainedHyperConnections#weight_dtype.
  ManifoldConstrainedHyperConnections.post_alpha_scale: ManifoldConstrainedHyperConnections#post_alpha_scale.
  ManifoldConstrainedHyperConnections.res_alpha: ManifoldConstrainedHyperConnections#res_alpha.
  ManifoldConstrainedHyperConnections.pre_alpha: ManifoldConstrainedHyperConnections#pre_alpha.
  ManifoldConstrainedHyperConnections.post_alpha: ManifoldConstrainedHyperConnections#post_alpha.
  ManifoldConstrainedHyperConnections.dtype: ManifoldConstrainedHyperConnections#dtype.
  ManifoldConstrainedHyperConnections.rngs: ManifoldConstrainedHyperConnections#rngs.
  ManifoldConstrainedHyperConnections.matmul_precision: ManifoldConstrainedHyperConnections#matmul_precision.
  ManifoldConstrainedHyperConnections.pre_beta: ManifoldConstrainedHyperConnections#pre_beta.
  ManifoldConstrainedHyperConnections.post_beta: ManifoldConstrainedHyperConnections#post_beta.
  ManifoldConstrainedHyperConnections.config: ManifoldConstrainedHyperConnections#config.
  ManifoldConstrainedHyperConnections.mapping: ManifoldConstrainedHyperConnections#mapping().
  ManifoldConstrainedHyperConnections.res_alpha_scale: ManifoldConstrainedHyperConnections#res_alpha_scale.
  ManifoldConstrainedHyperConnections.pre_alpha_scale: ManifoldConstrainedHyperConnections#pre_alpha_scale.
  ManifoldConstrainedHyperConnections.permutation_matrices: ManifoldConstrainedHyperConnections#permutation_matrices.
  ManifoldConstrainedHyperConnections.res_beta: ManifoldConstrainedHyperConnections#res_beta.
  get_functions: get_functions().
  ManifoldConstrainedHyperConnections.dim: ManifoldConstrainedHyperConnections#dim.
  sinkhorn: sinkhorn().
  get_permutation_matrices: get_permutation_matrices().
  get_functions.expand: get_functions().expand().
  get_functions.reduce: get_functions().reduce().
  sinkhorn.body_fun: sinkhorn().body_fun().
  ManifoldConstrainedHyperConnections.sinkhorn_iterations: ManifoldConstrainedHyperConnections#sinkhorn_iterations.
  ManifoldConstrainedHyperConnections: ManifoldConstrainedHyperConnections#
  ManifoldConstrainedHyperConnections.__init__: ManifoldConstrainedHyperConnections#__init__().
  ManifoldConstrainedHyperConnections.mesh: ManifoldConstrainedHyperConnections#mesh.
---
# Module: [`src/maxtext/layers/mhc.py`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py)

## Classes
### `ManifoldConstrainedHyperConnections`  ·  implements/extends Module
- def: [`src/maxtext/layers/mhc.py:82`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L82)
- doc: Implements Manifold-Constrained Hyper-Connections (mHC).
- signature: `class ManifoldConstrainedHyperConnections(nnx.Module):`
- members:
  - `__call__(self, norm_fn: Callable, branch_fn: Callable, x: Array, mhc_type: HyperConnectionType, **kwargs)` — [`L239`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L239) — Applying manifold-constrained hyper connection based on callable function.
  - `mapping(self, x: Array, alpha_scale: Array, alpha: Array, beta: Array, scale: int)` — [`L227`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L227) — Helper function for both pre and post mappings.
  - `res_mapping(self, x: Array)` — [`L196`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L196) — Helper function for residual mapping.
  - `config` — [`L101`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L101)
  - `dim` — [`L104`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L104)
  - `dtype` — [`L107`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L107)
  - `k` — [`L103`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L103)
  - `matmul_precision` — [`L109`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L109)
  - `mesh` — [`L106`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L106)
  - `mhc_norm` — [`L112`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L112)
  - `permutation_matrices` — [`L140`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L140)
  - `post_alpha` — [`L171`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L171)
  - `post_alpha_scale` — [`L130`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L130)
  - `post_beta` — [`L191`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L191)
  - `pre_alpha` — [`L161`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L161)
  - `pre_alpha_scale` — [`L126`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L126)
  - `pre_beta` — [`L187`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L187)
  - `res_alpha` — [`L151`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L151)
  - `res_alpha_scale` — [`L122`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L122)
  - `res_beta` — [`L183`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L183)
  - `rngs` — [`L105`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L105)
  - `sinkhorn_iterations` — [`L102`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L102)
  - `weight_dtype` — [`L108`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L108)
- protocol/private: `__init__`[`L94`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L94)
- uses (calls/refs, reference-scoped): [`sinkhorn`](mhc.md#sinkhorn), [`get_permutation_matrices`](mhc.md#get_permutation_matrices)

## Functions
- `body_fun(i, val)` — [`L69`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L69)
- `expand(x: Array)` — [`L48`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L48)
- `get_functions(expansion_rate: int)` — [`L42`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L42) — Creates functions to broadcast a single feature stream into multiple
- `get_permutation_matrices(k: int)` — [`L31`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L31) — Generates all permutation matrices of size k.
- `reduce(x: Array)` — [`L52`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L52)
- `sinkhorn(t, iters=20)` — [`L59`](../../../../../../../raw/code/maxtext/src/maxtext/layers/mhc.py#L59) — Computes the Sinkhorn normalization of a matrix (rows and columns sum to 1).

