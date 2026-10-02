---
title: 'Module: tpu_inference/layers/jax/constants.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/constants.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.constants`/
symbols:
  KVCacheType.KVCacheType: KVCacheType.KVCacheType.
  RouterType: RouterType#
  RouterType.TOP_K: RouterType#TOP_K.
  OPERATION_MODE: OPERATION_MODE#
  OPERATION_MODE.PREFILL: OPERATION_MODE#PREFILL.
  OPERATION_MODE.DECODE: OPERATION_MODE#DECODE.
  HuggingFaceArgNames: HuggingFaceArgNames#
  HuggingFaceArgNames.HIDDEN_ACT: HuggingFaceArgNames#HIDDEN_ACT.
  HuggingFaceArgNames.HIDDEN_SIZE: HuggingFaceArgNames#HIDDEN_SIZE.
  HuggingFaceArgNames.NUM_HIDDEN_LAYERS: HuggingFaceArgNames#NUM_HIDDEN_LAYERS.
  HuggingFaceArgNames.RMS_NORM_EPS: HuggingFaceArgNames#RMS_NORM_EPS.
  HuggingFaceArgNames.ROPE_SCALING: HuggingFaceArgNames#ROPE_SCALING.
  HuggingFaceArgNames.ROPE_THETA: HuggingFaceArgNames#ROPE_THETA.
  HuggingFaceArgNames.VOCAB_SIZE: HuggingFaceArgNames#VOCAB_SIZE.
  HuggingFaceArgNames.SHARED_EXPERTS: HuggingFaceArgNames#SHARED_EXPERTS.
  HuggingFaceArgNames.INTERMEDIATE_SIZE: HuggingFaceArgNames#INTERMEDIATE_SIZE.
  HuggingFaceArgNames.HEAD_DIM: HuggingFaceArgNames#HEAD_DIM.
  HuggingFaceArgNames.NUM_ATTENTION_HEADS: HuggingFaceArgNames#NUM_ATTENTION_HEADS.
  HuggingFaceArgNames.NUM_KEY_VALUE_HEADS: HuggingFaceArgNames#NUM_KEY_VALUE_HEADS.
  HuggingFaceArgNames.ATTENTION_DROPOUT: HuggingFaceArgNames#ATTENTION_DROPOUT.
  HuggingFaceArgNames.ATTENTION_BIAS: HuggingFaceArgNames#ATTENTION_BIAS.
  HuggingFaceArgNames.ATTENTION_CHUNK_SIZE: HuggingFaceArgNames#ATTENTION_CHUNK_SIZE.
  HuggingFaceArgNames.USE_QK_NORM: HuggingFaceArgNames#USE_QK_NORM.
  HuggingFaceArgNames.TEMPERATURE_TUNING: HuggingFaceArgNames#TEMPERATURE_TUNING.
  HuggingFaceArgNames.TEMPERATURE_TUNING_SCALE: HuggingFaceArgNames#TEMPERATURE_TUNING_SCALE.
  HuggingFaceArgNames.TEMPERATURE_TUNING_FLOOR_SCALE: HuggingFaceArgNames#TEMPERATURE_TUNING_FLOOR_SCALE.
  HuggingFaceArgNames.KV_LORA_RANK: HuggingFaceArgNames#KV_LORA_RANK.
  HuggingFaceArgNames.Q_LORA_RANK: HuggingFaceArgNames#Q_LORA_RANK.
  HuggingFaceArgNames.QK_NOPE_HEAD_DIM: HuggingFaceArgNames#QK_NOPE_HEAD_DIM.
  HuggingFaceArgNames.QK_ROPE_HEAD_DIM: HuggingFaceArgNames#QK_ROPE_HEAD_DIM.
  HuggingFaceArgNames.V_HEAD_DIM: HuggingFaceArgNames#V_HEAD_DIM.
  HuggingFaceArgNames.INTERMEDIATE_SIZE_MOE: HuggingFaceArgNames#INTERMEDIATE_SIZE_MOE.
  HuggingFaceArgNames.NUM_LOCAL_EXPERTS: HuggingFaceArgNames#NUM_LOCAL_EXPERTS.
  HuggingFaceArgNames.NUM_EXPERTS_PER_TOKEN: HuggingFaceArgNames#NUM_EXPERTS_PER_TOKEN.
  HuggingFaceArgNames.NUM_ROUTED_EXPERTS: HuggingFaceArgNames#NUM_ROUTED_EXPERTS.
  HuggingFaceArgNames.NUM_SHARED_ROUTED_EXPERTS: HuggingFaceArgNames#NUM_SHARED_ROUTED_EXPERTS.
  HuggingFaceArgNames.NUM_GROUPS: HuggingFaceArgNames#NUM_GROUPS.
  HuggingFaceArgNames.ROUTED_SCALING_FACTOR: HuggingFaceArgNames#ROUTED_SCALING_FACTOR.
  HuggingFaceArgNames.TOPK_GROUP: HuggingFaceArgNames#TOPK_GROUP.
  HuggingFaceArgNames.NORM_TOPK_PROB: HuggingFaceArgNames#NORM_TOPK_PROB.
  HuggingFaceArgNames.SCORING_FUNCTION: HuggingFaceArgNames#SCORING_FUNCTION.
  HuggingFaceArgNames.BOS_TOKEN_ID: HuggingFaceArgNames#BOS_TOKEN_ID.
  HuggingFaceArgNames.EOS_TOKEN_ID: HuggingFaceArgNames#EOS_TOKEN_ID.
---
# Module: [`tpu_inference/layers/jax/constants.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py)

## Classes
### `HuggingFaceArgNames`  ·  implements/extends Enum
- def: [`tpu_inference/layers/jax/constants.py:50`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L50)
- signature: `class HuggingFaceArgNames(enum.Enum):`
- members:
  - `ATTENTION_BIAS` — [`L71`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L71)
  - `ATTENTION_CHUNK_SIZE` — [`L72`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L72)
  - `ATTENTION_DROPOUT` — [`L70`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L70)
  - `BOS_TOKEN_ID` — [`L100`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L100)
  - `EOS_TOKEN_ID` — [`L101`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L101)
  - `HEAD_DIM` — [`L67`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L67)
  - `HIDDEN_ACT` — [`L52`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L52)
  - `HIDDEN_SIZE` — [`L53`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L53)
  - `INTERMEDIATE_SIZE` — [`L64`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L64)
  - `INTERMEDIATE_SIZE_MOE` — [`L88`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L88)
  - `KV_LORA_RANK` — [`L81`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L81)
  - `NORM_TOPK_PROB` — [`L96`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L96)
  - `NUM_ATTENTION_HEADS` — [`L68`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L68)
  - `NUM_EXPERTS_PER_TOKEN` — [`L90`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L90)
  - `NUM_GROUPS` — [`L93`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L93)
  - `NUM_HIDDEN_LAYERS` — [`L54`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L54)
  - `NUM_KEY_VALUE_HEADS` — [`L69`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L69)
  - `NUM_LOCAL_EXPERTS` — [`L89`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L89)
  - `NUM_ROUTED_EXPERTS` — [`L91`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L91)
  - `NUM_SHARED_ROUTED_EXPERTS` — [`L92`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L92)
  - `QK_NOPE_HEAD_DIM` — [`L83`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L83)
  - `QK_ROPE_HEAD_DIM` — [`L84`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L84)
  - `Q_LORA_RANK` — [`L82`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L82)
  - `RMS_NORM_EPS` — [`L55`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L55)
  - `ROPE_SCALING` — [`L56`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L56)
  - `ROPE_THETA` — [`L57`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L57)
  - `ROUTED_SCALING_FACTOR` — [`L94`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L94)
  - `SCORING_FUNCTION` — [`L97`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L97)
  - `SHARED_EXPERTS` — [`L61`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L61)
  - `TEMPERATURE_TUNING` — [`L76`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L76)
  - `TEMPERATURE_TUNING_FLOOR_SCALE` — [`L78`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L78)
  - `TEMPERATURE_TUNING_SCALE` — [`L77`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L77)
  - `TOPK_GROUP` — [`L95`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L95)
  - `USE_QK_NORM` — [`L75`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L75)
  - `VOCAB_SIZE` — [`L58`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L58)
  - `V_HEAD_DIM` — [`L85`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L85)

### `OPERATION_MODE`  ·  implements/extends Enum
- def: [`tpu_inference/layers/jax/constants.py:45`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L45)
- signature: `class OPERATION_MODE(enum.Enum):`
- members:
  - `DECODE` — [`L47`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L47)
  - `PREFILL` — [`L46`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L46)

### `RouterType`  ·  implements/extends Enum
- def: [`tpu_inference/layers/jax/constants.py:40`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L40)
- doc: Enum for router types.
- signature: `class RouterType(enum.Enum):`
- members:
  - `TOP_K` — [`L42`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L42)

## Module values
- `KVCacheType` — [`L37`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/constants.py#L37)

