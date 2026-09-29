---
title: 'Module: jax/_src/pallas/fuser/custom_fusion_lib.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/fuser/custom_fusion_lib.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.fuser.custom_fusion_lib`/
symbols:
  custom_fusion.__call__: custom_fusion#__call__().
  _custom_fusion_eval_rule: _custom_fusion_eval_rule().
  _custom_fusion_mosaic_lowering_rule: _custom_fusion_mosaic_lowering_rule().
  _custom_fusion_usage_rule: _custom_fusion_usage_rule().
  custom_fusion_p: custom_fusion_p.
  _custom_fusion_impl: _custom_fusion_impl().
  _custom_fusion_effectful_abstract_eval: _custom_fusion_effectful_abstract_eval().
  _custom_fusion_pull_block_spec_rule: _custom_fusion_pull_block_spec_rule().
  _custom_fusion_push_block_spec_rule: _custom_fusion_push_block_spec_rule().
  CustomPullBlockSpecRuleFn: CustomPullBlockSpecRuleFn.
  CustomPushBlockSpecRuleFn: CustomPushBlockSpecRuleFn.
  custom_fusion.eval_rule: custom_fusion#eval_rule.
  custom_fusion.pull_block_spec_rule: custom_fusion#pull_block_spec_rule.
  custom_fusion.def_pull_block_spec: custom_fusion#def_pull_block_spec().
  custom_fusion.def_push_block_spec: custom_fusion#def_push_block_spec().
  custom_fusion.def_eval_rule: custom_fusion#def_eval_rule().
  custom_fusion.push_block_spec_rule: custom_fusion#push_block_spec_rule.
  custom_fusion.fun: custom_fusion#fun.
  custom_fusion.pallas_impl: custom_fusion#pallas_impl.
  CustomEvalContext.out_block_specs: CustomEvalContext#out_block_specs.
  CustomEvalRuleFn: CustomEvalRuleFn#
  CustomEvalRuleFn.__call__: CustomEvalRuleFn#__call__().
  custom_fusion.__init__: custom_fusion#__init__().
  custom_fusion.def_pallas_impl: custom_fusion#def_pallas_impl().
  CustomEvalContext: CustomEvalContext#
  custom_fusion: custom_fusion#
  CustomEvalContext.out_block_indices: CustomEvalContext#out_block_indices.
---
# Module: [`jax/_src/pallas/fuser/custom_fusion_lib.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py)

## Classes
### `CustomEvalContext`
- def: [`jax/_src/pallas/fuser/custom_fusion_lib.py:46`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L46)
- signature: `class CustomEvalContext:`
- members:
  - `out_block_indices` — [`L48`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L48)
  - `out_block_specs` — [`L47`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L47)
- uses (calls/refs, reference-scoped): [`BlockSpec`](../core.md#BlockSpec)
- used by: [`_custom_fusion_eval_rule`](custom_fusion_lib.md#_custom_fusion_eval_rule), [`__call__`](custom_fusion_lib.md#CustomEvalRuleFn.__call__)

### `CustomEvalRuleFn`  ·  implements/extends Protocol
- def: [`jax/_src/pallas/fuser/custom_fusion_lib.py:50`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L50)
- signature: `class CustomEvalRuleFn(Protocol):`
- protocol/private: `__call__`[`L52`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L52)
- uses (calls/refs, reference-scoped): [`CustomEvalContext`](custom_fusion_lib.md#CustomEvalContext)
- used by: [`_custom_fusion_eval_rule`](custom_fusion_lib.md#_custom_fusion_eval_rule), [`eval_rule`](custom_fusion_lib.md#custom_fusion.eval_rule), [`def_eval_rule`](custom_fusion_lib.md#custom_fusion.def_eval_rule)

### `custom_fusion`
- def: [`jax/_src/pallas/fuser/custom_fusion_lib.py:61`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L61)
- signature: `class custom_fusion:`
- members:
  - `def_eval_rule(self, eval_rule: CustomEvalRuleFn)` — [`L93`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L93)
  - `def_pallas_impl(self, pallas_impl)` — [`L79`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L79)
  - `def_pull_block_spec(self, pull_block_spec_rule: CustomPullBlockSpecRuleFn)` — [`L83`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L83)
  - `def_push_block_spec(self, push_block_spec_rule: CustomPushBlockSpecRuleFn)` — [`L88`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L88)
  - `eval_rule` — [`L64`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L64)
  - `fun` — [`L62`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L62)
  - `pallas_impl` — [`L73`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L73)
  - `pull_block_spec_rule` — [`L66`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L66)
  - `push_block_spec_rule` — [`L69`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L69)
- protocol/private: `__call__`[`L99`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L99), `__init__`[`L75`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L75)
- uses (calls/refs, reference-scoped): [`bind`](../../core.md#Primitive.bind), [`typeof`](../../core.md#typeof), [`tree_unflatten`](../../tree_util.md#tree_unflatten), [`tree_flatten`](../../tree_util.md#tree_flatten), [`debug_info`](../../api_util.md#debug_info), [`api_boundary`](../../traceback_util.md#api_boundary), [`flatten_fun_nokwargs`](../../api_util.md#flatten_fun_nokwargs), [`custom_fusion_p`](custom_fusion_lib.md#custom_fusion_p), [`resolve_kwargs`](../../api_util.md#resolve_kwargs), [`CustomPullBlockSpecRuleFn`](custom_fusion_lib.md#CustomPullBlockSpecRuleFn), [`CustomPushBlockSpecRuleFn`](custom_fusion_lib.md#CustomPushBlockSpecRuleFn), [`CustomEvalRuleFn`](custom_fusion_lib.md#CustomEvalRuleFn)

## Functions
- `_custom_fusion_effectful_abstract_eval(*args, jaxpr: core.Jaxpr, pallas_jaxpr: core.Jaxpr | None, **_)` — [`L173`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L173)
- `_custom_fusion_eval_rule(ctx: block_spec_lib.KernelEvalContext, *args, eval_rule: CustomEvalRuleFn, num_consts: int, pallas_num_consts: int, **_)` — [`L194`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L194)
- `_custom_fusion_impl(*args, jaxpr: core.Jaxpr, num_consts: int, pallas_num_consts: int, **_)` — [`L159`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L159)
- `_custom_fusion_mosaic_lowering_rule(ctx: mosaic_lowering.LoweringRuleContext, *args, jaxpr: core.Jaxpr, num_consts: int, pallas_jaxpr: core.Jaxpr | None, pallas_num_consts: int, **_)` — [`L211`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L211)
- `_custom_fusion_pull_block_spec_rule(ctx: block_spec_lib.PullRuleContext, out_block_transforms: tuple[block_spec_lib.BlockIndexTransform, ...], *, pull_block_spec_rule: CustomPullBlockSpecRuleFn, **_)` — [`L230`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L230)
- `_custom_fusion_push_block_spec_rule(ctx: block_spec_lib.PushRuleContext, *block_specs: pallas_core.BlockSpec, push_block_spec_rule: CustomPushBlockSpecRuleFn, **_)` — [`L242`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L242)
- `_custom_fusion_usage_rule(ctx: block_spec_lib.UsageRuleContext, used_out: Sequence[set[block_spec_lib.Usage]], *, jaxpr: core.Jaxpr, **_)` — [`L254`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L254)

## Module values
- `CustomPullBlockSpecRuleFn` — [`L39`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L39)
- `CustomPushBlockSpecRuleFn` — [`L42`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L42)
- `custom_fusion_p` — [`L36`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/custom_fusion_lib.py#L36)

