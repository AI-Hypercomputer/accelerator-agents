---
title: 'Module: src/maxtext/common/common_types.py'
type: catalog
provenance: extracted
module: src/maxtext/common/common_types.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.common.common_types`/
symbols:
  Array: Array.
  MultimodalInput.image_embeddings: MultimodalInput#image_embeddings.
  MultimodalInput.image_masks: MultimodalInput#image_masks.
  MultimodalInput.video_embeddings: MultimodalInput#video_embeddings.
  MultimodalInput.video_masks: MultimodalInput#video_masks.
  MultimodalInput.audio_embeddings: MultimodalInput#audio_embeddings.
  MultimodalInput.audio_masks: MultimodalInput#audio_masks.
  MultimodalInput.bidirectional_mask: MultimodalInput#bidirectional_mask.
  MultimodalInput.bidirectional_mask_video: MultimodalInput#bidirectional_mask_video.
  Config: Config.
  PRNGKey: PRNGKey.
  DType: DType.
  Shape: Shape.
  AxisNames: AxisNames.
  AxisIdxes: AxisIdxes.
  BATCH: BATCH.
  BATCH_ATTN: BATCH_ATTN.
  ATTN_LENGTH: ATTN_LENGTH.
  LENGTH: LENGTH.
  PREFILL_LENGTH: PREFILL_LENGTH.
  Q_LENGTH: Q_LENGTH.
  Q_LORA_UP_PROJ: Q_LORA_UP_PROJ.
  KV_LENGTH: KV_LENGTH.
  KV_LORA_UP_PROJ: KV_LORA_UP_PROJ.
  ATTN_EMBED: ATTN_EMBED.
  EMBED: EMBED.
  HEAD: HEAD.
  PREFILL_KV_BATCH: PREFILL_KV_BATCH.
  KV_BATCH: KV_BATCH.
  KV_HEAD: KV_HEAD.
  KV_HEAD_DIM: KV_HEAD_DIM.
  D_KV: D_KV.
  DECODE_BATCH: DECODE_BATCH.
  DECODE_LENGTH: DECODE_LENGTH.
  CACHE_BATCH_PREFILL: CACHE_BATCH_PREFILL.
  CACHE_BATCH: CACHE_BATCH.
  CACHE_SEQUENCE: CACHE_SEQUENCE.
  CACHE_HEADS: CACHE_HEADS.
  CACHE_HEADS_NONE: CACHE_HEADS_NONE.
  CACHE_KV: CACHE_KV.
  CACHE_SCALE_BATCH: CACHE_SCALE_BATCH.
  CACHE_SCALE_SEQUENCE: CACHE_SCALE_SEQUENCE.
  CACHE_SCALE_HEADS: CACHE_SCALE_HEADS.
  CACHE_SCALE_KV: CACHE_SCALE_KV.
  MODEL_MODE_AUTOREGRESSIVE: MODEL_MODE_AUTOREGRESSIVE.
  MODEL_MODE_PREFILL: MODEL_MODE_PREFILL.
  MODEL_MODE_TRAIN: MODEL_MODE_TRAIN.
  DECODING_ACTIVE_SEQUENCE_INDICATOR: DECODING_ACTIVE_SEQUENCE_INDICATOR.
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  MultimodalInput: MultimodalInput#
  DecoderBlockType: DecoderBlockType#
  DecoderBlockType.DEFAULT: DecoderBlockType#DEFAULT.
  DecoderBlockType.LLAMA2: DecoderBlockType#LLAMA2.
  DecoderBlockType.MISTRAL: DecoderBlockType#MISTRAL.
  DecoderBlockType.MIXTRAL: DecoderBlockType#MIXTRAL.
  DecoderBlockType.DEEPSEEK: DecoderBlockType#DEEPSEEK.
  DecoderBlockType.GEMMA: DecoderBlockType#GEMMA.
  DecoderBlockType.GEMMA2: DecoderBlockType#GEMMA2.
  DecoderBlockType.GEMMA3: DecoderBlockType#GEMMA3.
  DecoderBlockType.GEMMA4: DecoderBlockType#GEMMA4.
  DecoderBlockType.GEMMA4_SMALL: DecoderBlockType#GEMMA4_SMALL.
  DecoderBlockType.QWEN2: DecoderBlockType#QWEN2.
  DecoderBlockType.QWEN3: DecoderBlockType#QWEN3.
  DecoderBlockType.QWEN3_MOE: DecoderBlockType#QWEN3_MOE.
  DecoderBlockType.QWEN3_CUSTOM_MOE: DecoderBlockType#QWEN3_CUSTOM_MOE.
  DecoderBlockType.QWEN3_NEXT: DecoderBlockType#QWEN3_NEXT.
  DecoderBlockType.QWEN3_5: DecoderBlockType#QWEN3_5.
  DecoderBlockType.GPT3: DecoderBlockType#GPT3.
  DecoderBlockType.GPT_OSS: DecoderBlockType#GPT_OSS.
  DecoderBlockType.SIMPLE: DecoderBlockType#SIMPLE.
  DecoderBlockType.SIMPLE_MLP: DecoderBlockType#SIMPLE_MLP.
  DecoderBlockType.LLAMA4: DecoderBlockType#LLAMA4.
  DecoderBlockType.OLMO3: DecoderBlockType#OLMO3.
  DecoderBlockType.DEEPSEEK4: DecoderBlockType#DEEPSEEK4.
  AttentionType: AttentionType#
  AttentionType.GLOBAL: AttentionType#GLOBAL.
  AttentionType.LOCAL_SLIDING: AttentionType#LOCAL_SLIDING.
  AttentionType.CHUNK: AttentionType#CHUNK.
  AttentionType.MLA: AttentionType#MLA.
  AttentionType.COMPRESSED: AttentionType#COMPRESSED.
  AttentionType.FULL: AttentionType#FULL.
  ShardMode: ShardMode#
  ShardMode.AUTO: ShardMode#AUTO.
  ShardMode.EXPLICIT: ShardMode#EXPLICIT.
  ReorderStrategy: ReorderStrategy#
  ReorderStrategy.AUTO: ReorderStrategy#AUTO.
  ReorderStrategy.DUAL_CHUNK_SWAP: ReorderStrategy#DUAL_CHUNK_SWAP.
  ReorderStrategy.STRIPED: ReorderStrategy#STRIPED.
  HyperConnectionType: HyperConnectionType#
  HyperConnectionType.ATTENTION: HyperConnectionType#ATTENTION.
  HyperConnectionType.MLP_MOE: HyperConnectionType#MLP_MOE.
  HyperConnectionType.MLP_DENSE: HyperConnectionType#MLP_DENSE.
  CustomRule: CustomRule#
  CustomRule.DEFAULT: CustomRule#DEFAULT.
  CustomRule.PURE_FSDP: CustomRule#PURE_FSDP.
  CustomRule.CP_AS_EP: CustomRule#CP_AS_EP.
  CustomRule.EP_AS_CP: CustomRule#EP_AS_CP.
  CustomRule.PIPELINE_LARGE_MOE: CustomRule#PIPELINE_LARGE_MOE.
  CustomRule.FSDP_2D: CustomRule#FSDP_2D.
  CustomRule.EP_AS_DP: CustomRule#EP_AS_DP.
---
# Module: [`src/maxtext/common/common_types.py`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py)

## Classes
### `AttentionType`  ·  implements/extends Enum
- def: [`src/maxtext/common/common_types.py:119`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L119)
- signature: `class AttentionType(enum.Enum):`
- members:
  - `CHUNK` — [`L122`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L122)
  - `COMPRESSED` — [`L124`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L124)
  - `FULL` — [`L125`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L125)
  - `GLOBAL` — [`L120`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L120)
  - `LOCAL_SLIDING` — [`L121`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L121)
  - `MLA` — [`L123`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L123)

### `CustomRule`  ·  implements/extends Enum
- def: [`src/maxtext/common/common_types.py:149`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L149)
- signature: `class CustomRule(enum.Enum):`
- members:
  - `CP_AS_EP` — [`L152`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L152)
  - `DEFAULT` — [`L150`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L150)
  - `EP_AS_CP` — [`L153`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L153)
  - `EP_AS_DP` — [`L156`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L156)
  - `FSDP_2D` — [`L155`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L155)
  - `PIPELINE_LARGE_MOE` — [`L154`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L154)
  - `PURE_FSDP` — [`L151`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L151)

### `DecoderBlockType`  ·  implements/extends Enum
- def: [`src/maxtext/common/common_types.py:91`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L91)
- doc: Decoder block types.
- signature: `class DecoderBlockType(enum.Enum):`
- members:
  - `DEEPSEEK` — [`L98`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L98)
  - `DEEPSEEK4` — [`L116`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L116)
  - `DEFAULT` — [`L94`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L94)
  - `GEMMA` — [`L99`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L99)
  - `GEMMA2` — [`L100`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L100)
  - `GEMMA3` — [`L101`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L101)
  - `GEMMA4` — [`L102`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L102)
  - `GEMMA4_SMALL` — [`L103`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L103)
  - `GPT3` — [`L110`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L110)
  - `GPT_OSS` — [`L111`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L111)
  - `LLAMA2` — [`L95`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L95)
  - `LLAMA4` — [`L114`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L114)
  - `MISTRAL` — [`L96`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L96)
  - `MIXTRAL` — [`L97`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L97)
  - `OLMO3` — [`L115`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L115)
  - `QWEN2` — [`L104`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L104)
  - `QWEN3` — [`L105`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L105)
  - `QWEN3_5` — [`L109`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L109)
  - `QWEN3_CUSTOM_MOE` — [`L107`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L107)
  - `QWEN3_MOE` — [`L106`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L106)
  - `QWEN3_NEXT` — [`L108`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L108)
  - `SIMPLE` — [`L112`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L112)
  - `SIMPLE_MLP` — [`L113`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L113)

### `HyperConnectionType`  ·  implements/extends Enum
- def: [`src/maxtext/common/common_types.py:143`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L143)
- signature: `class HyperConnectionType(enum.Enum):`
- members:
  - `ATTENTION` — [`L144`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L144)
  - `MLP_DENSE` — [`L146`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L146)
  - `MLP_MOE` — [`L145`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L145)

### `MultimodalInput`
- def: [`src/maxtext/common/common_types.py:78`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L78)
- doc: Multimodal inputs for encoder processing.
- signature: `class MultimodalInput:`
- members:
  - `audio_embeddings` — [`L85`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L85)
  - `audio_masks` — [`L86`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L86)
  - `bidirectional_mask` — [`L87`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L87)
  - `bidirectional_mask_video` — [`L88`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L88)
  - `image_embeddings` — [`L81`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L81)
  - `image_masks` — [`L82`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L82)
  - `video_embeddings` — [`L83`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L83)
  - `video_masks` — [`L84`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L84)
- uses (calls/refs, reference-scoped): [`Array`](common_types.md#Array)

### `ReorderStrategy`  ·  implements/extends Enum
- def: [`src/maxtext/common/common_types.py:133`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L133)
- doc: Reorder strategies for load-balanced context parallelism.
- signature: `class ReorderStrategy(enum.Enum):`
- members:
  - `AUTO` — [`L138`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L138)
  - `DUAL_CHUNK_SWAP` — [`L139`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L139)
  - `STRIPED` — [`L140`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L140)

### `ShardMode`  ·  implements/extends Enum
- def: [`src/maxtext/common/common_types.py:128`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L128)
- signature: `class ShardMode(enum.Enum):`
- members:
  - `AUTO` — [`L129`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L129)
  - `EXPLICIT` — [`L130`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L130)

## Module values
- `ATTN_EMBED` — [`L45`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L45)
- `ATTN_LENGTH` — [`L37`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L37)
- `Array` — [`L26`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L26)
- `AxisIdxes` — [`L32`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L32)
- `AxisNames` — [`L31`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L31)
- `BATCH` — [`L34`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L34)
- `BATCH_ATTN` — [`L35`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L35)
- `CACHE_BATCH` — [`L56`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L56)
- `CACHE_BATCH_PREFILL` — [`L55`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L55)
- `CACHE_HEADS` — [`L58`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L58)
- `CACHE_HEADS_NONE` — [`L59`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L59)
- `CACHE_KV` — [`L60`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L60)
- `CACHE_SCALE_BATCH` — [`L61`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L61)
- `CACHE_SCALE_HEADS` — [`L63`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L63)
- `CACHE_SCALE_KV` — [`L64`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L64)
- `CACHE_SCALE_SEQUENCE` — [`L62`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L62)
- `CACHE_SEQUENCE` — [`L57`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L57)
- `Config` — [`L24`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L24)
- `DECODE_BATCH` — [`L53`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L53)
- `DECODE_LENGTH` — [`L54`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L54)
- `DECODING_ACTIVE_SEQUENCE_INDICATOR` — [`L70`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L70)
- `DEFAULT_MASK_VALUE` — [`L74`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L74)
- `DType` — [`L28`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L28)
- `D_KV` — [`L52`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L52)
- `EMBED` — [`L46`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L46)
- `HEAD` — [`L47`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L47)
- `KV_BATCH` — [`L49`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L49)
- `KV_HEAD` — [`L50`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L50)
- `KV_HEAD_DIM` — [`L51`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L51)
- `KV_LENGTH` — [`L43`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L43)
- `KV_LORA_UP_PROJ` — [`L44`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L44)
- `LENGTH` — [`L39`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L39)
- `MODEL_MODE_AUTOREGRESSIVE` — [`L66`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L66)
- `MODEL_MODE_PREFILL` — [`L67`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L67)
- `MODEL_MODE_TRAIN` — [`L68`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L68)
- `PREFILL_KV_BATCH` — [`L48`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L48)
- `PREFILL_LENGTH` — [`L40`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L40)
- `PRNGKey` — [`L27`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L27)
- `Q_LENGTH` — [`L41`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L41)
- `Q_LORA_UP_PROJ` — [`L42`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L42)
- `Shape` — [`L29`](../../../../../../../raw/code/maxtext/src/maxtext/common/common_types.py#L29)

