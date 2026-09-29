---
title: 'Module: python/sgl_jax/srt/multimodal/common/ServerArgs.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/multimodal/common/ServerArgs.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.multimodal.common.ServerArgs`/MultimodalServerArgs#
symbols:
  MultimodalServerArgs.add_cli_args: add_cli_args().
  MultimodalServerArgs: ''
  MultimodalServerArgs.__post_init__: __post_init__().
  MultimodalServerArgs.precompile_frame_paddings: precompile_frame_paddings.
  MultimodalServerArgs.precompile_width_heights: precompile_width_heights.
  MultimodalServerArgs.embedded_cfg_scale: embedded_cfg_scale.
  MultimodalServerArgs.flow_shift: flow_shift.
  MultimodalServerArgs.dit_precision: dit_precision.
  MultimodalServerArgs.vae_precision: vae_precision.
  MultimodalServerArgs.vae_tiling: vae_tiling.
  MultimodalServerArgs.DEFAULT_TEXT_ENCODER_PRECISIONS: DEFAULT_TEXT_ENCODER_PRECISIONS.
  MultimodalServerArgs.image_encoder_precision: image_encoder_precision.
  MultimodalServerArgs.from_cli_args: from_cli_args().
  MultimodalServerArgs.vae_sp: vae_sp.
  MultimodalServerArgs.text_encoder_precisions: text_encoder_precisions.
---
# Module: [`python/sgl_jax/srt/multimodal/common/ServerArgs.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/ServerArgs.py)

## Classes
### `MultimodalServerArgs`  ·  implements/extends ServerArgs
- def: [`python/sgl_jax/srt/multimodal/common/ServerArgs.py:9`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/ServerArgs.py#L9)
- signature: `class MultimodalServerArgs(ServerArgs):`
- members:
  - `add_cli_args(parser: argparse.ArgumentParser)` — [`L26`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/ServerArgs.py#L26)
  - `from_cli_args(cls, args: argparse.Namespace)` — [`L131`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/ServerArgs.py#L131)
  - `DEFAULT_TEXT_ENCODER_PRECISIONS` — [`L18`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/ServerArgs.py#L18)
  - `dit_precision` — [`L13`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/ServerArgs.py#L13)
  - `embedded_cfg_scale` — [`L10`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/ServerArgs.py#L10)
  - `flow_shift` — [`L11`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/ServerArgs.py#L11)
  - `image_encoder_precision` — [`L20`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/ServerArgs.py#L20)
  - `precompile_frame_paddings` — [`L23`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/ServerArgs.py#L23)
  - `precompile_width_heights` — [`L22`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/ServerArgs.py#L22)
  - `text_encoder_precisions` — [`L19`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/ServerArgs.py#L19)
  - `vae_precision` — [`L15`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/ServerArgs.py#L15)
  - `vae_sp` — [`L17`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/ServerArgs.py#L17)
  - `vae_tiling` — [`L16`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/ServerArgs.py#L16)
- protocol/private: `__post_init__`[`L111`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/ServerArgs.py#L111)
- uses (calls/refs, reference-scoped): [`ServerArgs`](../../server_args.md#ServerArgs), [`__post_init__`](../../server_args.md#ServerArgs.__post_init__)
- used by: [`add_cli_args`](../../server_args.md#ServerArgs.add_cli_args), [`ServerArgs`](../../server_args.md#ServerArgs), [`from_cli`](../../server_args.md#ServerArgs.from_cli), [`from_cli_args`](../../server_args.md#ServerArgs.from_cli_args)

