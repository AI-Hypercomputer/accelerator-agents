"""Static context and hydrated meta-prompts for MaxText model bring-up.

This file contains the complete, raw source code of core MaxText reference files
from clean commit 313890777 (before any DeepSeek-V4 PRs) hydrated directly
into python string constants for use with Google ADK's `static_instruction`
parameter on `LlmAgent`.
"""

# pylint: disable=line-too-long

# File: src/maxtext/common/common_types.py (commit 313890777)
COMMON_TYPES_RAW = """\n# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

\"\"\"Common types.\"\"\"
import enum
from typing import Any, Sequence

import numpy as np

import jax.numpy as jnp
from flax import struct

Config = Any

Array = jnp.ndarray
PRNGKey = jnp.ndarray
DType = jnp.dtype
Shape = Sequence[int]

AxisNames = tuple[str, ...]
AxisIdxes = tuple[int, ...]

BATCH = "activation_batch"
BATCH_ATTN = "activation_batch_attn"

ATTN_LENGTH = "activation_length_attn"

LENGTH = "activation_length"
PREFILL_LENGTH = "prefill_activation_length"
Q_LENGTH = "activation_q_length"
Q_LORA_UP_PROJ = "q_lora_up_proj"
KV_LENGTH = "activation_kv_length"
KV_LORA_UP_PROJ = "kv_lora_up_proj"
ATTN_EMBED = "activation_embed_attn"
EMBED = "activation_embed"
HEAD = "activation_heads"
PREFILL_KV_BATCH = "activation_prefill_kv_batch"
KV_BATCH = "activation_kv_batch"
KV_HEAD = "activation_kv_heads"
KV_HEAD_DIM = "activation_kv_head_dim"
D_KV = "activation_kv"
DECODE_BATCH = "decode_batch"
DECODE_LENGTH = "decode_length"
CACHE_BATCH_PREFILL = "cache_batch_prefill"
CACHE_BATCH = "cache_batch"
CACHE_SEQUENCE = "cache_sequence"
CACHE_HEADS = "cache_heads"
CACHE_HEADS_NONE = "cache_heads_none"
CACHE_KV = "cache_kv"
CACHE_SCALE_BATCH = "cache_scale_batch"
CACHE_SCALE_SEQUENCE = "cache_scale_sequence"
CACHE_SCALE_HEADS = "cache_scale_heads"
CACHE_SCALE_KV = "cache_scale_kv"

MODEL_MODE_AUTOREGRESSIVE = "autoregressive"
MODEL_MODE_PREFILL = "prefill"
MODEL_MODE_TRAIN = "train"

DECODING_ACTIVE_SEQUENCE_INDICATOR = 1

# A large negative mask value is used for masking to ensure that the
# softmax function assigns an extremely low probability to the masked positions.
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


@struct.dataclass
class MultimodalInput:
  \"\"\"Multimodal inputs for encoder processing.\"\"\"

  image_embeddings: Array | None = None
  image_masks: Array | None = None
  audio_embeddings: Array | None = None
  audio_masks: Array | None = None
  bidirectional_mask: Array | None = None


class DecoderBlockType(enum.Enum):
  \"\"\"Decoder block types.\"\"\"

  DEFAULT = "default"
  LLAMA2 = "llama2"
  MISTRAL = "mistral"
  MIXTRAL = "mixtral"
  DEEPSEEK = "deepseek"
  GEMMA = "gemma"
  GEMMA2 = "gemma2"
  GEMMA3 = "gemma3"
  GEMMA4 = "gemma4"
  QWEN2 = "qwen2"
  QWEN3 = "qwen3"
  QWEN3_MOE = "qwen3_moe"
  QWEN3_CUSTOM_MOE = "qwen3_custom_moe"
  QWEN3_NEXT = "qwen3_next"
  GPT3 = "gpt3"
  GPT_OSS = "gpt_oss"
  SIMPLE = "simple"
  SIMPLE_MLP = "simple_mlp"
  LLAMA4 = "llama4"
  OLMO3 = "olmo3"

  LLAMA2LTI = "llama2_learn_to_init"


class AttentionType(enum.Enum):
  GLOBAL = "global"  # default, with causality
  LOCAL_SLIDING = "local_sliding"
  CHUNK = "chunk"
  MLA = "mla"
  FULL = "full"


class ShardMode(enum.Enum):
  AUTO = "auto"  # default
  EXPLICIT = "explicit"


class ReorderStrategy(enum.Enum):
  \"\"\"Reorder strategies for load-balanced context parallelism.
  Maps to transformer_engine.jax.attention.ReorderStrategy at runtime.
  \"\"\"

  AUTO = "auto"
  DUAL_CHUNK_SWAP = "dual_chunk_swap"
  STRIPED = "striped"


class HyperConnectionType(enum.Enum):
  ATTENTION = "attention"
  MLP_MOE = "mlp_moe"
  MLP_DENSE = "mlp_dense"


class CustomRule(enum.Enum):
  DEFAULT = ""
  PURE_FSDP = "pure-fsdp"
  CP_AS_EP = "cp-as-ep"  # Support CP and EP together
  EP_AS_CP = "ep-as-cp"  # Support EP only
  PIPELINE_LARGE_MOE = "pipeline-large-moe"
\n"""


# File: src/maxtext/layers/normalizations.py (commit 313890777)
NORMALIZATIONS_RAW = """\n# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

\"\"\"Normalization Layers.\"\"\"

from typing import Any

from flax import linen as nn
from flax import nnx
from flax.linen import initializers as linen_initializers
import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from maxtext.common.common_types import Array, DType, ShardMode
from maxtext.layers import nnx_wrappers
from maxtext.layers.initializers import Initializer, variable_to_logically_partitioned
from maxtext.utils import max_logging
from maxtext.utils import max_utils


class RMSNorm(nnx.Module):
  \"\"\"RMS normalization.\"\"\"

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-6,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      shard_mode: ShardMode = ShardMode.AUTO,
      kernel_axes: tuple[None | str, ...] = (),
      scale_init: Initializer = nn.initializers.ones,
      parameter_memory_host_offload: bool = False,
      scale_offset: float = 0.0,
      with_scale: bool = True,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.shard_mode = shard_mode
    self.kernel_axes = kernel_axes
    self.scale_init = scale_init
    self.parameter_memory_host_offload = parameter_memory_host_offload
    self.scale_offset = scale_offset
    self.with_scale = with_scale
    if self.with_scale:
      self.scale = nnx.Param(
          scale_init(rngs.params(), (num_features,), weight_dtype),
          sharding=kernel_axes,
      )
    else:
      self.scale = None

  def __call__(self, x: jnp.ndarray, out_sharding: NamedSharding | None = None) -> jnp.ndarray:
    \"\"\"Applies layer normalization on the input.\"\"\"
    x = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)

    # out_sharding must be None in auto shard mode
    if self.shard_mode != ShardMode.EXPLICIT:
      out_sharding = None

    if not self.with_scale:
      if out_sharding is not None:
        y = jax.lax.with_sharding_constraint(y, out_sharding)
      return y

    scale = self.scale.get_value()
    # Move scale to device if parameter offloading is enabled
    if self.parameter_memory_host_offload:
      max_logging.log("normalizations.py: Moving scale parameter to device")
      scale = jax.device_put(scale, max_utils.device_space())

    scale = jnp.asarray(scale, self.dtype)
    effective_scale = scale + self.scale_offset
    return jnp.einsum("...k,k->...k", y, effective_scale, out_sharding=out_sharding)


class GlobalRMSNorm(RMSNorm):
  \"\"\"
  Applies RMSNorm over the last two dimensions (Heads * HeadDim).
  Used for Olmo3 which normalizes across all heads combined.
  \"\"\"

  def __call__(self, x: jnp.ndarray, out_sharding: NamedSharding | None = None) -> jnp.ndarray:
    # x shape: [..., Heads, HeadDim]
    input_shape = x.shape

    # Flatten the last two dimensions: [..., Heads * HeadDim]
    # We use -2 and -1 to ensure we capture the last two dims regardless of rank
    flattened_shape = input_shape[:-2] + (input_shape[-2] * input_shape[-1],)
    x_flat = x.reshape(flattened_shape)

    # Apply standard RMSNorm (which normalizes over the last axis)
    y_flat = super().__call__(x_flat, out_sharding)

    # Reshape back to [..., Heads, HeadDim]
    return y_flat.reshape(input_shape)


def Qwen3NextRMSNorm(num_features: int, eps: float, dtype: DType, weight_dtype: DType, *, rngs: nnx.Rngs):
  \"\"\"
  Used for input and post attention layernorms
  in Qwen3NextDecoderLayer.

  This normalization layer is specific to Qwen3-Next. Key characteristics:
  1.  The learnable scale parameter `scale` is initialized to ZEROS.
  2.  The scale is applied as `(1.0 + self.scale)`, making the initial scale effectively 1.0.
      This matches the PyTorch implementation of Qwen3NextRMSNorm.
  \"\"\"
  return nnx.data(
      RMSNorm(
          num_features=num_features,
          epsilon=eps,
          dtype=dtype,
          weight_dtype=weight_dtype,
          scale_init=linen_initializers.zeros,
          scale_offset=1.0,
          rngs=rngs,
      )
  )


class Qwen3NextRMSNormGated(nnx.Module):
  \"\"\"
  This applies RMS Normalization and then a gated activation function (SiLU).
  This is used within the Qwen3NextGatedDeltaNet.

  The normalization is performed by an internal `RMSNorm` instance (`self.rms_norm`),
  which has its own learnable `scale` parameter, initialized to ONES.

  Attributes:
    num_features: The number of features in the input.
    eps: A small epsilon value to prevent division by zero in RMSNorm.
    dtype: The datatype of the computation.
    weight_dtype: The datatype of the internal RMSNorm scale.
  \"\"\"

  def __init__(self, num_features: int, eps: float, dtype: DType, weight_dtype: DType, *, rngs: nnx.Rngs):
    self.num_features = num_features
    self.eps = eps
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.rms_norm = nnx.data(
        RMSNorm(
            num_features=num_features,
            epsilon=eps,
            dtype=dtype,
            weight_dtype=weight_dtype,
            scale_init=nnx.initializers.ones,
            rngs=rngs,
        )
    )

  def __call__(self, hidden_states: Array, gate: Array) -> Array:
    \"\"\"
    Applies RMSNorm and then a SiLU gate.

    Args:
      hidden_states: The input array to be normalized (o). Shape: (..., F)
      gate: The gating array for the activation (z). Shape: (..., F)
            where F is num_features.

    Returns:
      The normalized and gated output array. Shape: (..., F)
    \"\"\"
    normalized_states = self.rms_norm(hidden_states)

    # Gated Activation using SiLU (Sigmoid-weighted Linear Unit)
    gated_states = normalized_states * jax.nn.silu(gate.astype(jnp.float32))

    return gated_states.astype(self.dtype)


def rms_norm(
    num_features: int,
    epsilon: float = 1e-6,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    shard_mode: ShardMode = ShardMode.AUTO,
    kernel_axes: tuple[None | str, ...] = (),
    scale_init: Initializer = nn.initializers.ones,
    name: None | str = None,
    parameter_memory_host_offload: bool = False,
    with_scale: bool = True,
):
  \"\"\"Creates a RMSNorm module.\"\"\"
  module = nnx_wrappers.to_linen(
      RMSNorm,
      num_features=num_features,
      epsilon=epsilon,
      dtype=dtype,
      weight_dtype=weight_dtype,
      shard_mode=shard_mode,
      kernel_axes=kernel_axes,
      scale_init=scale_init,
      parameter_memory_host_offload=parameter_memory_host_offload,
      with_scale=with_scale,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
  )
  return module


def l2norm(x: Array, dim: int = -1, eps: float = 1e-6) -> Array:
  \"\"\"L2 normalization function. Normalizes a vector to have a length of 1.

  Args:
    x: Input array.
    dim: The axis or axes along which to normalize. Defaults to the last axis.
    eps: Small epsilon to prevent division by zero.

  Returns:
    L2 normalized array with the same shape as x.
  \"\"\"

  inv_norm = jax.lax.rsqrt((x * x).sum(axis=dim, keepdims=True) + jnp.array(eps, dtype=x.dtype))
  return x * inv_norm


Qwen3NextRMSNormLinen = nnx_wrappers.to_linen_class(
    RMSNorm,
    base_metadata_fn=variable_to_logically_partitioned,
    scale_init=linen_initializers.zeros,
    scale_offset=1.0,
)
\n"""


# File: src/maxtext/layers/embeddings.py (commit 313890777)
EMBEDDINGS_RAW = """\n# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

\"\"\"Embedding Layers.\"\"\"

import dataclasses
import math

import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding

from flax import nnx

from maxtext.common.common_types import ShardMode, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN, Array, Config, DType
from maxtext.layers import nnx_wrappers
from maxtext.layers.initializers import Initializer, default_embed_init, variable_to_logically_partitioned
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils.sharding import logical_to_mesh_axes, create_sharding

_MAX_WAVELENGTH = 10_000


def _maybe_move_embedding_to_device(embedding_table: Array, config: Config) -> Array:
  \"\"\"Moves embedding table to device if parameter offloading is enabled.\"\"\"
  if config.parameter_memory_host_offload:
    max_logging.log("embeddings.py: Moving embedding parameter to device")
    return jax.device_put(embedding_table, max_utils.device_space())
  return embedding_table


def embed_as_linen(
    *,
    num_embeddings: int,
    num_features: int,
    config: Config,
    mesh: Mesh,
    cast_input_dtype: None | DType = None,
    dtype: DType = jnp.float32,
    attend_dtype: None | DType = None,
    embedding_init: Initializer = default_embed_init,
    name: str | None = None,
):
  \"\"\"Initializes the Embed NNX module and returns it as a Linen module.

  This function serves as a bridge to use the NNX-based `Embed` module within
  a Linen model. It wraps the `Embed` module using `nnx.bridge.to_linen`,
  making it compatible with the Linen API.

  Args:
    num_embeddings: The number of embeddings.
    num_features: The number of feature dimensions for each embedding.
    config: The model configuration.
    cast_input_dtype: The dtype to cast the input to, if any.
    dtype: The dtype of the embedding vectors.
    attend_dtype: The dtype for the `attend` method.
    embedding_init: The initializer for the embedding matrix.
    name: The name of the Linen module.

  Returns:
    A Linen module that wraps the NNX `Embed` module.
  \"\"\"
  return nnx_wrappers.to_linen(
      Embed,
      num_embeddings=num_embeddings,
      num_features=num_features,
      config=config,
      mesh=mesh,
      cast_input_dtype=cast_input_dtype,
      dtype=dtype,
      attend_dtype=attend_dtype,
      embedding_init=embedding_init,
      metadata_fn=variable_to_logically_partitioned,
      name=name,
  )


class Embed(nnx.Module):
  \"\"\"A parameterized function from integers [0, n) to d-dimensional vectors.\"\"\"

  def __init__(
      self,
      num_embeddings: int,
      num_features: int,
      config: Config,
      mesh: Mesh,
      cast_input_dtype: None | DType = None,
      dtype: DType = jnp.float32,
      attend_dtype: None | DType = None,
      embedding_init: Initializer = default_embed_init,
      *,
      # Not used in Embed but passed in by nnx.bridge.to_linen.
      rngs: nnx.Rngs,
  ):
    \"\"\"Initializes the Embed module.

    Args:
      num_embeddings: The number of embeddings.
      num_features: The number of feature dimensions for each embedding.
      config: The model configuration.
      cast_input_dtype: The dtype to cast the input to, if any.
      dtype: The dtype of the embedding vectors.
      attend_dtype: The dtype for the `attend` method.
      embedding_init: The initializer for the embedding matrix.
      rngs: The random number generators for initialization.
    \"\"\"
    self.num_embeddings = num_embeddings
    self.num_features = num_features
    self.config = config
    self.mesh = mesh
    self.cast_input_dtype = cast_input_dtype
    self.dtype = dtype
    self.attend_dtype = attend_dtype

    self.embedding = nnx.Param(
        embedding_init(
            rngs.params(),
            (self.num_embeddings, self.num_features),
            self.config.weight_dtype,
        ),
        sharding=("vocab", "embed_vocab"),
    )

  def __call__(self, inputs: Array, model_mode: str = MODEL_MODE_TRAIN) -> Array:
    \"\"\"Embeds the inputs along the last dimension.

    Args:
      inputs: input data, all dimensions are considered batch dimensions.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional `num_features` dimension appended.
    \"\"\"
    cfg = self.config
    if self.cast_input_dtype:
      inputs = inputs.astype(self.cast_input_dtype)
    if not jnp.issubdtype(inputs.dtype, jnp.integer):
      raise ValueError("Input type must be an integer or unsigned integer.")

    embedding = jnp.asarray(
        _maybe_move_embedding_to_device(self.embedding.get_value(), self.config),
        self.dtype,
    )

    output_axis_names = (
        (
            "activation_embed_and_logits_batch",
            "prefill_activation_length",
            "activation_embed",
        )
        if model_mode == MODEL_MODE_PREFILL
        else (
            "activation_embed_and_logits_batch",
            "activation_length",
            "activation_embed",
        )
    )
    out_pspec = logical_to_mesh_axes(output_axis_names, self.mesh, rules=getattr(self.config, "logical_axis_rules", None))

    out_sharding = NamedSharding(self.mesh, out_pspec) if self.config.shard_mode == ShardMode.EXPLICIT else None

    if cfg.use_iota_embed:
      iota = lax.iota(jnp.int32, self.num_embeddings)
      one_hot = jnp.array(inputs[..., jnp.newaxis] == iota, dtype=self.dtype)
      output = jnp.dot(one_hot, embedding, out_sharding=out_sharding)
    else:
      output = embedding.at[inputs].get(out_sharding=out_sharding)

    return output

  def attend(self, query: Array, out_sharding: NamedSharding | None = None) -> Array:
    \"\"\"Attend over the embedding using a query array.

    Args:
      query: array with last dimension equal the feature depth `num_features` of the
        embedding.
      out_sharding: NamedSharding object indicating how the output gets sharded

    Returns:
      An array with final dim `num_embeddings` corresponding to the batched
      inner-product of the array of query vectors against each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.
    \"\"\"
    embedding = self.embedding.get_value()
    attend_dtype = self.attend_dtype if self.attend_dtype is not None else self.dtype
    return attend_on_embedding(query, embedding, attend_dtype, self.config, out_sharding)


def attend_on_embedding(
    query: Array,
    embedding_table: Array,
    attend_dtype: DType,
    config: Config,
    out_sharding: NamedSharding | None = None,
) -> Array:
  \"\"\"Attend over an embedding table using a query array.

  TODO: Remove this method when Embed bridge to Linen is no longer needed

  Args:
    query: An array with a last dimension equal to the feature depth of the embedding.
    embedding_table: The embedding table to attend over.
    attend_dtype: The data type for the attention computation.
    config: The model configuration, used to check for parameter offloading.
    out_sharding: NamedSharding object indicating the output sharding

  Returns:
    An array with a final dimension equal to `num_embeddings`, corresponding to the
    batched inner-product of the query vectors against each embedding.
  \"\"\"
  # out_sharding must be None under auto shard_mode
  if config.shard_mode != ShardMode.EXPLICIT:
    out_sharding = None
  embedding_table = _maybe_move_embedding_to_device(embedding_table, config)
  return jnp.dot(
      query,
      jnp.asarray(embedding_table, jnp.bfloat16).T,
      preferred_element_type=attend_dtype,
      out_sharding=out_sharding,
  )


def rotary_embedding_as_linen(
    *,
    min_timescale: int,
    max_timescale: int,
    embedding_dims: int = 0,
    cast_as_fprop_dtype: bool = True,
    fprop_dtype: DType = jnp.bfloat16,
    name: str | None = None,
):
  \"\"\"Initializes the RotaryEmbedding module and returns it as a Linen module.

  Args:
    min_timescale: Start of the geometric index. Determines the periodicity of
      the added signal.
    max_timescale: End of the geometric index. Determines the frequency of the
      added signal.
    embedding_dims: Dimension of the embedding to be generated.
    cast_as_fprop_dtype: Whether to cast the output to the fprop dtype.
    fprop_dtype: The dtype of the output.
    name: Name of the Linen module.
  \"\"\"
  return nnx_wrappers.to_linen(
      RotaryEmbedding,
      min_timescale=min_timescale,
      max_timescale=max_timescale,
      embedding_dims=embedding_dims,
      cast_as_fprop_dtype=cast_as_fprop_dtype,
      fprop_dtype=fprop_dtype,
      metadata_fn=variable_to_logically_partitioned,
      name=name,
  )


class RotaryEmbedding(nnx.Module):
  \"\"\"Rotary Position Embedding.\"\"\"

  def __init__(
      self,
      min_timescale: int,
      max_timescale: int,
      mesh: Mesh,
      embedding_dims: int = 0,
      cast_as_fprop_dtype: bool = True,
      fprop_dtype: DType = jnp.bfloat16,
      shard_mode: ShardMode = ShardMode.AUTO,
      # Not used in RotaryEmbedding but passed in by nnx.bridge.to_linen.
      rope_linear_scaling_factor: float = 1.0,
      rngs: nnx.Rngs = None,
  ):
    \"\"\"Initializes the RotaryEmbedding module.

    Args:
      min_timescale: Start of the geometric index. Determines the periodicity of
        the added signal.
      max_timescale: End of the geometric index. Determines the frequency of the
        added signal.
      embedding_dims: Dimension of the embedding to be generated.
      cast_as_fprop_dtype: Whether to cast the output to the fprop dtype.
      fprop_dtype: The dtype of the output.
      rngs: rng keys passed in by nnx.bridge.to_linen.
    \"\"\"
    self.min_timescale = min_timescale
    self.max_timescale = max_timescale
    self.mesh = mesh
    self.embedding_dims = embedding_dims
    self.cast_as_fprop_dtype = cast_as_fprop_dtype
    self.fprop_dtype = fprop_dtype
    self.shard_mode = shard_mode
    self.rope_linear_scaling_factor = rope_linear_scaling_factor

    if self.embedding_dims % 2:
      raise ValueError("Embedding dim for rotary position embedding must be a multiple of 2.")

  @property
  def timescale(self):
    \"\"\"Returns the timescale for the rotary embedding.\"\"\"
    half_embedding_dim = self.embedding_dims // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
    timescale = self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction
    if self.rope_linear_scaling_factor != 1.0:
      timescale = timescale * self.rope_linear_scaling_factor
    return timescale

  def _rotate_half(self, x: jax.Array) -> jax.Array:
    \"\"\"Rotates half the hidden dims of the input: (x1, x2) -> (-x2, x1).\"\"\"
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)

  def apply_rotary(self, inputs: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    \"\"\"Applies the rotary transformation logic.\"\"\"
    return (inputs * cos) + (self._rotate_half(inputs) * sin)

  def __call__(
      self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      inputs: jax.Array,
      position: None | jax.Array = None,
  ) -> jax.Array:
    \"\"\"Generates a jax.Array of sinusoids with different frequencies.

    Args:
      inputs: The input sequence on which to apply the Rotary position
        embedding. Since rotary position embeddings are applied to query and
        keys after projection, it is assumed of shape [B, S, N, H].
      position: Optional position jax.Array which denotes the position of each
        token in the sequence. This only needs to be supplied when the sequence
        is packed. It is of shape [B, S].

    Returns:
      a jax.Array of shape [B, S, N, H] which includes the inputs together with
      the rotary position embedding incorporated in it.
    \"\"\"
    assert position is not None
    if len(inputs.shape) != 4:
      raise ValueError("Input is assumed to be a rank 4 tensor of shape" "[batch, sequence, heads, dims].")
    if self.embedding_dims != inputs.shape[3]:
      raise ValueError(
          "The embedding dims of the rotary position embedding" "must match the hidden dimension of the inputs."
      )

    position = position[:, :, jnp.newaxis, jnp.newaxis]
    sinusoid_inp = position / self.timescale
    sin_half = jnp.sin(sinusoid_inp).astype(inputs.dtype)
    cos_half = jnp.cos(sinusoid_inp).astype(inputs.dtype)

    sin = jnp.concatenate([sin_half, sin_half], axis=-1)
    cos = jnp.concatenate([cos_half, cos_half], axis=-1)

    x_out = self.apply_rotary(inputs, cos, sin)

    if self.cast_as_fprop_dtype:
      x_out = x_out.astype(self.fprop_dtype)
    return x_out


def llama_rotary_embedding_as_linen(
    *,
    min_timescale: int,
    max_timescale: int,
    embedding_dims: int = 0,
    cast_as_fprop_dtype: bool = True,
    fprop_dtype: DType = jnp.bfloat16,
    use_scale: bool = True,
    name: str | None = None,
):
  \"\"\"Initializes the LLaMARotaryEmbedding module and returns it as a Linen module.

  Args:
    min_timescale: Start of the geometric index. Determines the periodicity of
      the added signal.
    max_timescale: End of the geometric index. Determines the frequency of the
      added signal.
    embedding_dims: Dimension of the embedding to be generated.
    cast_as_fprop_dtype: Whether to cast the output to the fprop dtype.
    fprop_dtype: The dtype of the output.
    use_scale: Whether to apply LLaMA3.1 scaling factor.
    name: Name of the Linen module.
  \"\"\"
  return nnx_wrappers.to_linen(
      LLaMARotaryEmbedding,
      min_timescale=min_timescale,
      max_timescale=max_timescale,
      embedding_dims=embedding_dims,
      cast_as_fprop_dtype=cast_as_fprop_dtype,
      fprop_dtype=fprop_dtype,
      use_scale=use_scale,
      metadata_fn=variable_to_logically_partitioned,
      name=name,
  )


def partial_rotary_embedding_as_linen(
    *,
    min_timescale: int,
    max_timescale: int,
    mesh: Mesh,
    embedding_dims: int = 0,
    partial_rotary_factor: float = 0.25,
    cast_as_fprop_dtype: bool = True,
    fprop_dtype: DType = jnp.bfloat16,
    shard_mode: ShardMode = ShardMode.AUTO,
    name: str | None = None,
):
  \"\"\"Initializes the PartialRotaryEmbedding module and returns it as a Linen module.

  Args:
    min_timescale: Start of the geometric index. Determines the periodicity of
      the added signal.
    max_timescale: End of the geometric index. Determines the frequency of the
      added signal.
    embedding_dims: Dimension of the embedding to be generated.
    partial_rotary_factor: Ratio of dimensions to apply ROPE to.
    cast_as_fprop_dtype: Whether to cast the output to the fprop dtype.
    fprop_dtype: The dtype of the output.
    name: Name of the Linen module.
  \"\"\"
  return nnx_wrappers.to_linen(
      PartialRotaryEmbedding,
      min_timescale=min_timescale,
      max_timescale=max_timescale,
      mesh=mesh,
      embedding_dims=embedding_dims,
      partial_rotary_factor=partial_rotary_factor,
      cast_as_fprop_dtype=cast_as_fprop_dtype,
      fprop_dtype=fprop_dtype,
      shard_mode=shard_mode,
      metadata_fn=variable_to_logically_partitioned,
      name=name,
  )


class PartialRotaryEmbedding(RotaryEmbedding):
  \"\"\"Rotary Position Embedding applied to a partial fraction of dimensions.\"\"\"

  def __init__(
      self,
      min_timescale: int,
      max_timescale: int,
      mesh: Mesh,
      embedding_dims: int = 0,
      cast_as_fprop_dtype: bool = True,
      fprop_dtype: DType = jnp.bfloat16,
      partial_rotary_factor: float = 0.25,
      shard_mode: ShardMode = ShardMode.AUTO,
      rngs: nnx.Rngs = None,
  ):
    \"\"\"Initializes the PartialRotaryEmbedding module.

    Args:
      min_timescale: Start of the geometric index. Determines the periodicity of
        the added signal.
      max_timescale: End of the geometric index. Determines the frequency of the
        added signal.
      embedding_dims: Dimension of the embedding to be generated.
      partial_rotary_factor: Ratio of dimensions to apply ROPE to
      rngs: rng keys passed in by nnx.bridge.to_linen.
    \"\"\"
    self.head_dim = embedding_dims
    self.partial_rotary_factor = partial_rotary_factor
    self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)

    # Initialize the base class with only the rotary_dim
    super().__init__(
        min_timescale=min_timescale,
        max_timescale=max_timescale,
        mesh=mesh,
        embedding_dims=self.rotary_dim,
        cast_as_fprop_dtype=cast_as_fprop_dtype,
        fprop_dtype=fprop_dtype,
        shard_mode=shard_mode,
        rngs=rngs,
    )

  def __call__(self, inputs: jax.Array, position: None | jax.Array = None) -> jax.Array:
    \"\"\"Applies Partial variant of rotary position embedding.

    Args:
      inputs: The input sequence on which to apply the Rotary position
        embedding. It is assumed of shape [B, S, H, D].
      position: Optional position array [B, S]. Only needed when the sequence
        is packed.

    Returns:
      A jax.Array of shape [B, S, H, D - rotary_dim] with rotary position embeddings applied.
    \"\"\"
    # Split, apply base RoPE to the first fraction, and concatenate
    inputs_rot, inputs_pass = jnp.split(inputs, [self.rotary_dim], axis=-1)
    inputs_rot = super().__call__(inputs_rot, position)
    inputs = jnp.concatenate([inputs_rot, inputs_pass], axis=-1)
    return inputs


class Gemma4PartialRotaryEmbedding(RotaryEmbedding):
  \"\"\"Gemma 4 Rotary Position Embedding applied to a partial fraction of dimensions.

  Unlike standard PartialRotaryEmbedding which physically splits and concatenates
  features (resulting in a [Rotated, Unrotated] layout), Gemma 4 computes frequencies
  using the full embedding dimension denominator and pads the unrotated timescales
  with infinity.

  Because x / inf = 0, applying RoPE mathematically acts as an identity function
  on those unrotated dimensions. Because the base Rotary class splits the full tensor
  in half, this creates an interleaved feature layout in memory:
  [Rotated Half 1, Unrotated Half 1, Rotated Half 2, Unrotated Half 2].
  \"\"\"

  def __init__(
      self,
      min_timescale: int,
      max_timescale: int,
      mesh: Mesh,
      embedding_dims: int = 0,
      cast_as_fprop_dtype: bool = True,
      fprop_dtype: DType = jnp.bfloat16,
      partial_rotary_factor: float = 0.25,
      shard_mode: ShardMode = ShardMode.AUTO,
      rngs: nnx.Rngs = None,
  ):
    \"\"\"Initializes the instance.\"\"\"
    self.head_dim = embedding_dims
    self.partial_rotary_factor = partial_rotary_factor
    self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)

    # Pass the full head_dim to the base class so it splits at head_dim / 2,
    # ensuring the unrotated dimensions get correctly mixed into the center.
    super().__init__(
        min_timescale=min_timescale,
        max_timescale=max_timescale,
        mesh=mesh,
        embedding_dims=self.head_dim,
        cast_as_fprop_dtype=cast_as_fprop_dtype,
        fprop_dtype=fprop_dtype,
        shard_mode=shard_mode,
        rngs=rngs,
    )

  @property
  def timescale(self) -> jax.Array:
    \"\"\"The inf-padded timescale for Gemma 4 rotary embedding.\"\"\"
    half_rotary_dim = self.rotary_dim // 2

    # Gemma 4 uniquely uses the full head_dim as the denominator
    fraction = 2 * jnp.arange(0, half_rotary_dim) / self.head_dim
    timescale = self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction

    if getattr(self, "rope_linear_scaling_factor", 1.0) != 1.0:
      timescale = timescale * self.rope_linear_scaling_factor

    # Pad the remaining angles with jnp.inf.
    # When position is divided by inf, the angle becomes 0.
    # sin(0)=0 and cos(0)=1, which acts as a passthrough for unrotated dims.
    nope_angles = (self.head_dim // 2) - half_rotary_dim

    return jnp.pad(
        timescale,
        pad_width=(0, nope_angles),
        mode="constant",
        constant_values=(0.0, jnp.inf),
    )

  # Note: No __call__ override is required. The base RotaryEmbedding.__call__
  # handles the rotation perfectly using the padded self.timescale.


class LLaMARotaryEmbedding(RotaryEmbedding):
  \"\"\"LLaMA variant of ROPE.\"\"\"

  def __init__(
      self,
      min_timescale: int,
      max_timescale: int,
      mesh: Mesh,
      embedding_dims: int = 0,
      cast_as_fprop_dtype: bool = True,
      fprop_dtype: DType = jnp.bfloat16,
      use_scale: bool = True,
      shard_mode: ShardMode = ShardMode.AUTO,
      # Not used in LLaMARotaryEmbedding but passed in by nnx.bridge.to_linen.
      rngs: nnx.Rngs = None,
  ):
    \"\"\"Initializes the LLaMARotaryEmbedding module.

    Args:
      min_timescale: Start of the geometric index. Determines the periodicity of
        the added signal.
      max_timescale: End of the geometric index. Determines the frequency of the
        added signal.
      embedding_dims: Dimension of the embedding to be generated.
      cast_as_fprop_dtype: Whether to cast the output to the fprop dtype.
      fprop_dtype: The dtype of the output.
      use_scale: Whether to apply LLaMA3.1 scaling factor.
      rngs: rng keys passed in by nnx.bridge.to_linen.
    \"\"\"
    super().__init__(
        min_timescale=min_timescale,
        max_timescale=max_timescale,
        mesh=mesh,
        embedding_dims=embedding_dims,
        cast_as_fprop_dtype=cast_as_fprop_dtype,
        fprop_dtype=fprop_dtype,
        shard_mode=shard_mode,
        rngs=rngs,
    )

    # LLaMA3.1 ROPE scaling, see the original pytorch implementation:
    # https://github.com/meta-llama/llama-models/blob/301ca3a2b3b10e94ddcd1fdd2c57e52f812e1cac/models/llama3/reference_impl/model.py#L45C5-L45C18
    self.use_scale = use_scale

  @property
  def timescale(self):
    half_embedding_dim = self.embedding_dims // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
    fraction = jnp.repeat(fraction, 2)
    timescale = self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction

    # Apply scaling factor if enabled
    if self.use_scale:
      timescale = 1.0 / jax.vmap(self._apply_scaling_factor)(1.0 / timescale)

    # Expand timescale dimensions for broadcasting
    return timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]

  def _apply_scaling_factor(self, freq):
    \"\"\"apply scaling factor to rotary position embedding.\"\"\"
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * jnp.pi / freq

    def lower_wavelen(freq):
      return freq

    def bigger_or_equal_wavelen(freq):
      def bigger_wavelen(freq):
        return freq / scale_factor

      def equal_wavelen(freq):
        smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        return (1 - smooth) * freq / scale_factor + smooth * freq

      bigger_wavelen_cond = wavelen > low_freq_wavelen
      return jax.lax.cond(bigger_wavelen_cond, bigger_wavelen, equal_wavelen, freq)

    lower_wavelen_cond = wavelen < high_freq_wavelen
    return jax.lax.cond(lower_wavelen_cond, lower_wavelen, bigger_or_equal_wavelen, freq)

  def __call__(self, inputs: jax.Array, position: None | jax.Array = None) -> jax.Array:
    \"\"\"Applies LLaMA variant of rotary position embedding.

    Args:
      inputs: The input sequence on which to apply the Rotary position
        embedding. It is assumed of shape [B, S, N, H].
      position: Optional position array [B, S]. Only needed when the sequence
        is packed.

    Returns:
      A jax.Array of shape [B, S, N, H] with rotary position embeddings applied.
    \"\"\"
    # Ensure input is 4D
    if len(inputs.shape) != 4:
      raise ValueError("Input is assumed to be a rank 4 tensor of shape [B, S, N, H].")
    if self.embedding_dims != inputs.shape[3]:
      raise ValueError(
          "The embedding dims of the rotary position embedding must match the hidden dimension of the inputs."
      )

    # Shift the inputs left and right as per LLaMA's specific behavior
    inputs_shifted_left = jnp.concatenate([inputs[..., 1:], inputs[..., :1]], axis=-1)
    inputs_shifted_right = jnp.concatenate([inputs[..., -1:], inputs[..., :-1]], axis=-1)
    inputs_shifted = jax.lax.select(
        jnp.tile(
            jnp.mod(jnp.arange(self.embedding_dims, dtype=jnp.int32), 2),
            inputs.shape[:-1] + (1,),
        ),
        inputs_shifted_right,
        inputs_shifted_left,
    )

    # Determine positions if not provided
    if position is None:
      seq_length = inputs.shape[1]
      position = jnp.arange(seq_length, dtype=jnp.float32)[jnp.newaxis, :]

    # Calculate sinusoidal input
    position = position[:, :, jnp.newaxis, jnp.newaxis]
    sinusoid_inp = position / self.timescale

    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    # Apply alternating sign
    sign = jnp.tile(jnp.array([-1, 1]), self.embedding_dims // 2)

    # Combine original inputs with sinusoidal information
    outputs = inputs * cos + inputs_shifted * sin * sign

    if self.cast_as_fprop_dtype:
      outputs = outputs.astype(self.fprop_dtype)

    return outputs


def yarn_rotary_embedding_as_linen(
    *,
    embedding_dims: int,
    mesh: Mesh,
    max_position_embeddings: int = 4096 * 4,
    original_max_position_embeddings: int = 4096,
    beta_fast: float = 32,
    beta_slow: float = 1,
    rope_theta: float = 10000.0,
    rope_factor: float = 40,
    cast_as_fprop_dtype: bool = True,
    fprop_dtype: DType = jnp.bfloat16,
    name: str | None = None,
    interleave: bool = True,
    truncate: bool = True,
    attention_scaling: bool = False,
    shard_mode: ShardMode = ShardMode.AUTO,
):
  \"\"\"Initializes the YarnRotaryEmbedding module and returns it as a Linen module.

  Args:
    embedding_dims: The dimension of the embeddings.
    max_position_embeddings: The maximum number of positions.
    original_max_position_embeddings: The original maximum number of positions.
    beta_fast: The fast beta parameter for YaRN.
    beta_slow: The slow beta parameter for YaRN.
    rope_theta: The base for the rotary frequencies.
    rope_factor: The scaling factor for RoPE.
    cast_as_fprop_dtype: Whether to cast the output to `fprop_dtype`.
    fprop_dtype: The forward pass dtype.
    name: The name of the module.
  \"\"\"
  return nnx_wrappers.to_linen(
      YarnRotaryEmbedding,
      embedding_dims=embedding_dims,
      max_position_embeddings=max_position_embeddings,
      mesh=mesh,
      original_max_position_embeddings=original_max_position_embeddings,
      beta_fast=beta_fast,
      beta_slow=beta_slow,
      rope_theta=rope_theta,
      rope_factor=rope_factor,
      cast_as_fprop_dtype=cast_as_fprop_dtype,
      fprop_dtype=fprop_dtype,
      metadata_fn=variable_to_logically_partitioned,
      name=name,
      interleave=interleave,
      truncate=truncate,
      attention_scaling=attention_scaling,
      shard_mode=shard_mode,
  )


class YarnRotaryEmbedding(nnx.Module):
  \"\"\"Yarn rotary embedding.

  Based on https://arxiv.org/abs/2309.00071
  This implementation uses DeepSeek-v3 PyTorch as reference
  https://github.com/deepseek-ai/DeepSeek-V3/blob/2f7b80eecebf3d1c84da5a0d465f6639ea175012/inference/model.py#L294

  Implementation Notes:
  - YaRN vs. Standard RoPE:
    1. Frequency Initialization: YaRN modifies how frequencies are computed.
    2. Attention Scaling: YaRN typically scales embeddings by `0.1 * ln(rope_factor) + 1.0`
       when `rope_factor > 1`. This scaling can be applied within this layer (if `attention_scaling=True`)
       or externally.
  - RoPE Implementation Details (General):
    - Arithmetic: Uses complex number arithmetic. Real number arithmetic is not implemented here,
      though the resulting embeddings would be equivalent.
    - Input Layout: Supports both interleaved (`interleave=True`, e.g., [real1, img1, real2, img2]) and
      concatenated (`interleave=False`, e.g., [real1, real2, img1, img2]) formats.
    - Output Layout: Always returns concatenated format ([real, imag]). Interleaved output is not
      implemented: While the embedding is different, attention scores are invariant, as long as we apply
      the same output layout for Q and K.

  Attributes:
    embedding_dims: Dimension of the embedding to be generated.
    max_position_embeddings: The maximum sequence length that will be encountered.
    original_max_position_embeddings: The sequence length for which the base frequencies were defined.
    beta_fast: Lower bound parameter for correction.
    beta_slow: Upper bound parameter for correction.
    rope_theta: The base theta value for the frequency computation.
    rope_factor: Factor applied to adjust the frequencies.
    cast_as_fprop_dtype: Whether to cast the output to `fprop_dtype`.
    fprop_dtype: The forward pass dtype.
    rope_interleave: Whether complex representation is interleaved or concatenated.
    rope_truncate: Whether or not to floor lower bound and ceil upper bound for correction range.
    rope_attention_scaling: Whether or not to scale the rotary embedding output.
    rngs: rng keys passed in by nnx.bridge.to_linen.
  \"\"\"

  def __init__(
      self,
      embedding_dims: int,
      mesh: Mesh,
      max_position_embeddings: int = 4096 * 4,
      original_max_position_embeddings: int = 4096,
      beta_fast: float = 32,
      beta_slow: float = 1,
      rope_theta: float = 10000.0,
      rope_factor: float = 40,
      cast_as_fprop_dtype: bool = True,
      fprop_dtype: DType = jnp.bfloat16,
      shard_mode: ShardMode = ShardMode.AUTO,
      interleave=True,
      truncate=True,
      attention_scaling=False,
      # Not used in YarnRotaryEmbedding but passed in by nnx.bridge.to_linen.
      rngs: nnx.Rngs = None,
  ):
    \"\"\"Initializes the YarnRotaryEmbedding module.\"\"\"
    self.embedding_dims = embedding_dims
    self.max_position_embeddings = max_position_embeddings
    self.original_max_position_embeddings = original_max_position_embeddings
    self.beta_fast = beta_fast
    self.beta_slow = beta_slow
    self.rope_theta = rope_theta
    self.rope_factor = rope_factor
    self.cast_as_fprop_dtype = cast_as_fprop_dtype
    self.fprop_dtype = fprop_dtype
    self.interleave = interleave
    self.truncate = truncate
    self.mesh = mesh
    self.shard_mode = shard_mode
    self.attention_scaling = attention_scaling

    self.freqs_sharding = (
        create_sharding(mesh, ("activation_batch", "activation_length", "q_heads"))
        if shard_mode == ShardMode.EXPLICIT
        else None
    )

    if self.embedding_dims % 2:
      raise ValueError("Embedding dim for rotary position embedding must be a multiple of 2.")

  @property
  def freqs_cis(self):
    \"\"\"Frequencies for rotary embedding.\"\"\"
    half_dim = self.embedding_dims // 2
    # Compute base frequencies for each (even-indexed) dimension.
    # (Note: We use jnp.arange with float32 for precision.)
    freqs = 1.0 / (self.rope_theta ** (2.0 * jnp.arange(0, half_dim, dtype=jnp.float32) / self.embedding_dims))

    low, high = self._find_correction_range(
        self.beta_fast,
        self.beta_slow,
        self.embedding_dims,
        self.rope_theta,
        self.original_max_position_embeddings,
        self.truncate,
    )
    smooth = 1 - self._linear_ramp_factor(low, high, half_dim)
    # The corrected frequency is a weighted mix of the scaled and base values.
    freqs = freqs / self.rope_factor * (1 - smooth) + freqs * smooth

    # Precompute frequencies for all positions by taking the outer product.
    t = jnp.arange(self.max_position_embeddings, dtype=jnp.float32)  # shape [max_position_embeddings]
    # This gives a [max_position_embeddings, half_dim] tensor with rows as time steps.
    freqs = jnp.outer(t, freqs)

    # Compute the complex “cis” values: exp(i * theta).
    return jnp.exp(1j * freqs)  # shape [max_position_embeddings, half_dim]

  def _find_correction_dim(self, num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
    \"\"\"Compute the correction dimension for a given number of rotations.\"\"\"
    return dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

  def _find_correction_range(
      self,
      low_rot: float,
      high_rot: float,
      dim: int,
      base: float,
      max_position_embeddings: int,
      truncate: bool,
  ):
    \"\"\"Computes the range of correction dimensions for rotary positional embeddings.

    Args:
        low_rot (float): Lower bound for the number of rotations.
        high_rot (float): Upper bound for the number of rotations.
        dim (int): Dimensionality of the embedding space.
        base (float): Base value for the exponential computation.
        max_position_embeddings (int): Maximum sequence length.
        truncate (bool): Whether to floor lower bound and ceil upper bound.

    Returns:
        tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
    \"\"\"
    low = self._find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = self._find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
      low = math.floor(low)
      high = math.ceil(high)
    low = max(low, 0)
    high = min(high, dim - 1)
    return low, high

  def _linear_ramp_factor(self, min_val: float, max_val: float, dim: int) -> Array:
    \"\"\"Computes a linear ramp over the dimension.

    Returns a jax.Array of shape (dim,) with values between 0 and 1.
    \"\"\"
    if min_val == max_val:
      max_val += 0.001  # Avoid division by zero.
    linear_func = (jnp.arange(dim, dtype=jnp.float32) - min_val) / (max_val - min_val)
    return jnp.clip(linear_func, 0, 1)

  def __call__(self, inputs: Array, position: None | Array = None) -> Array:
    \"\"\"Applies the rotary positional embedding using the precomputed complex frequencies.

    Args:
      inputs: jax.Array of shape [B, S, N, H]. (H must equal self.embedding_dims.)
      position: jax.Array of shape [B, S] with integer positions (indexes into precomputed freqs).

    Returns:
      jax.Array of shape [B, S, N, H] with the rotary embedding applied.
    \"\"\"
    if len(inputs.shape) != 4:
      raise ValueError("Input is assumed to be a rank 4 tensor of shape [batch, sequence, heads, dims].")
    if self.embedding_dims != inputs.shape[3]:
      raise ValueError(
          "The embedding dims of the rotary position embedding must match the hidden dimension of the inputs."
      )

    # Determine positions if not provided
    if position is None:
      seq_length = inputs.shape[1]
      position = jnp.arange(seq_length, dtype=jnp.int32)[jnp.newaxis, :]
    else:
      position = position.astype(jnp.int32)

    # Lookup the precomputed frequencies using the position indices.
    # self.freqs_cis has shape [max_position_embeddings, half_dim] so we use jnp.take along axis 0.
    # After indexing, shape becomes [B, S, half_dim]; we then add an axis for the heads.
    freqs = self.freqs_cis.at[position].get(out_sharding=self.freqs_sharding)  # shape: [B, S, half_dim]
    freqs = freqs[:, :, jnp.newaxis, :]  # shape: [B, S, 1, half_dim]

    if self.interleave:
      # Inputs with interleaved format [real1, img1, real2, img2, ...] at last dimension
      # Convert the last dimension into a complex representation.
      # First reshape so that each pair of numbers represents the real and imaginary parts.
      B, S, N, H = inputs.shape
      half_dim = H // 2
      inputs_reshaped = inputs.reshape(B, S, N, half_dim, 2)
      first_half, second_half = inputs_reshaped[..., 0], inputs_reshaped[..., 1]
    else:
      # Inputs with concatenated format [real1, real2, ..., img1, img2, ...] at last dimension
      first_half, second_half = jnp.split(inputs, 2, axis=-1)

    inputs_complex = first_half + 1j * second_half  # shape: [B, S, N, half_dim]
    # Apply the rotary transformation via complex multiplication.
    rotated_sharding = (
        create_sharding(self.mesh, ("activation_batch", "activation_length", None, None))
        if self.shard_mode == ShardMode.EXPLICIT
        else None
    )
    freqs = jnp.broadcast_to(freqs, inputs_complex.shape, out_sharding=rotated_sharding)
    rotated = jnp.multiply(inputs_complex, freqs)  # shape: [B, S, N, half_dim]

    # Convert the complex result back to a real tensor.
    # Split the complex number into its real and imaginary parts.
    # [real1, real2, ..., img1, img2, ...]
    output = jnp.concatenate([jnp.real(rotated), jnp.imag(rotated)], axis=-1)

    if self.attention_scaling:
      attention_scaling = 1.0 if self.rope_factor <= 1 else (0.1 * math.log(self.rope_factor) + 1.0)
      output = output * attention_scaling

    if self.cast_as_fprop_dtype:
      output = output.astype(self.fprop_dtype)
    return output


def positional_embedding_as_linen(
    *,
    embedding_dims: int,
    max_wavelength: int = _MAX_WAVELENGTH,
    cast_as_fprop_dtype: bool = False,
    fprop_dtype: DType = jnp.bfloat16,
):
  \"\"\"Initializes the PositionalEmbedding module and returns it as a Linen module.

  Args:
    embedding_dims: The dimension of the embeddings.
    max_wavelength: The maximum wavelength for the sinusoidal positional embeddings.
    cast_as_fprop_dtype: Whether to cast output to fprop_dtype.
    fprop_dtype: The dtype of the output when cast_as_fprop_dtype is True.
  \"\"\"
  return nnx_wrappers.to_linen(
      PositionalEmbedding,
      embedding_dims=embedding_dims,
      max_wavelength=max_wavelength,
      cast_as_fprop_dtype=cast_as_fprop_dtype,
      fprop_dtype=fprop_dtype,
      metadata_fn=variable_to_logically_partitioned,
  )


@dataclasses.dataclass(repr=False)
class PositionalEmbedding(nnx.Module):
  \"\"\"Sinusoidal positional embeddings supporting both uniform and per-batch positions.

  This module computes sinusoidal positional embeddings and supports two use cases:

  1. Uniform positions across batch: All batch elements share the same position sequence.
     Pass position as 1D array (seq_len,) or None for sequential [0,1,2,...].
     Returns (seq_len, embedding_dims), caller broadcasts to batch.
     Example: pos_emb = layer(seq_len)  # Sequential positions
              pos_emb = layer(seq_len, position_1d)  # Custom 1D positions

  2. Per-batch positions (packed sequences): Each batch element has different positions.
     Pass position as 2D array (batch, seq_len).
     Returns (batch, seq_len, embedding_dims).
     Example: pos_emb = layer(seq_len, position_2d)

  As a side effect, the uniform case is more efficient since sin/cos are computed once
  and broadcasted, rather than per batch element.
  \"\"\"

  #: The dimension of the embeddings.
  embedding_dims: int
  #: The maximum wavelength for the sinusoidal positional embeddings.
  max_wavelength: int = _MAX_WAVELENGTH
  #: Whether to cast output to fprop_dtype.
  cast_as_fprop_dtype: bool = False
  #: The dtype of the output when cast_as_fprop_dtype is True.
  fprop_dtype: DType = jnp.bfloat16
  #: RNG state passed in by nnx.bridge.to_linen, not used in this module.
  rngs: nnx.Rngs = None  # Not used in PositionalEmbedding but passed in by nnx.bridge.to_linen

  def _compute_embeddings(self, position: Array) -> Array:
    \"\"\"Compute sinusoidal embeddings for given positions.

    Args:
      position: Either (seq_len,) for efficient path or (batch, seq_len) for full path.

    Returns:
      Embeddings of shape (seq_len, embedding_dims) or (batch, seq_len, embedding_dims).
    \"\"\"
    num_timescales = self.embedding_dims // 2
    log_timescale_increment = jnp.log(float(self.max_wavelength)) / jnp.maximum(
        jnp.asarray(num_timescales, dtype=jnp.float32) - 1, 1
    )
    inv_timescales = jnp.exp(jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment)

    if position.ndim == 1:
      # use the same position for the whole batch when position is (seq_len,)
      scaled_time = position[:, jnp.newaxis] * inv_timescales[jnp.newaxis, :]
    else:
      # when position is (batch, seq_len)
      position = position[:, :, jnp.newaxis]
      inv_timescales = inv_timescales[jnp.newaxis, jnp.newaxis, :]
      scaled_time = position * inv_timescales

    signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=-1)

    if self.cast_as_fprop_dtype:
      return signal.astype(self.fprop_dtype)
    else:
      return signal.astype(jnp.float32)

  def __call__(
      self,
      seq_len: int,
      position: Array | None = None,
  ) -> Array:
    \"\"\"Compute positional embeddings.

    Args:
      seq_len: Sequence length for computing embeddings.
      position: Optional position array. If None, uses sequential [0,1,2,...].
        Shape can be (seq_len,) or (batch, seq_len) for packed sequences.

    Returns:
      Positional embeddings of shape (seq_len, embedding_dims) or
      (batch, seq_len, embedding_dims) if position has batch dimension.
    \"\"\"
    if position is None:
      position = jnp.arange(seq_len, dtype=jnp.float32)

    return self._compute_embeddings(position)


def llama_vision_rotary_embedding_as_linen(
    *,
    image_size: int,
    patch_size: int,
    hidden_size: int,
    num_attention_heads: int,
    rope_theta: float = 10000.0,
    cast_as_fprop_dtype: bool = True,
    fprop_dtype: DType = jnp.bfloat16,
    name: str | None = None,
):
  \"\"\"Initializes the LlamaVisionRotaryEmbedding module and returns it as a Linen module.

  Args:
    image_size: The size of the input image.
    patch_size: The size of the image patches.
    hidden_size: The size of the hidden dimension.
    num_attention_heads: The number of attention heads.
    rope_theta: The base theta value for the frequency computation.
    cast_as_fprop_dtype: Whether to cast the output to the fprop dtype.
    fprop_dtype: The dtype of the output.
    name: The name of the Linen module.
  \"\"\"
  return nnx_wrappers.to_linen(
      LlamaVisionRotaryEmbedding,
      image_size=image_size,
      patch_size=patch_size,
      hidden_size=hidden_size,
      num_attention_heads=num_attention_heads,
      rope_theta=rope_theta,
      cast_as_fprop_dtype=cast_as_fprop_dtype,
      fprop_dtype=fprop_dtype,
      metadata_fn=variable_to_logically_partitioned,
      name=name,
  )


@dataclasses.dataclass(repr=False)
class LlamaVisionRotaryEmbedding(nnx.Module):
  \"\"\"Rotary position embedding for Llama4 vision encoder.

  Based on Pytorch Reference
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py
  This implementation follows the Llama4 vision encoder's rotary embedding approach,
  which uses 2D coordinates (x, y) to generate rotary position embeddings.
  \"\"\"

  #: size of the input image
  image_size: int
  #: size of the image patches
  patch_size: int
  #: size of the hidden dimension
  hidden_size: int
  #: number of attention heads
  num_attention_heads: int
  #: base theta value for the frequency computation
  rope_theta: float = 10000.0
  #: whether to cast the output to the fprop dtype
  cast_as_fprop_dtype: bool = True
  #: the dtype of the output
  fprop_dtype: DType = jnp.bfloat16
  # Not used in LlamaVisionRotaryEmbedding but passed in by nnx.bridge.to_linen.
  #: RNG state passed in by nnx.bridge.to_linen, not used in this module
  rngs: nnx.Rngs = None

  @property
  def freqs_cis(self):
    \"\"\"Frequencies for rotary embedding.\"\"\"
    idx = self.image_size // self.patch_size
    img_idx = jnp.arange(idx**2, dtype=jnp.int32).reshape(idx**2, 1)
    img_idx = jnp.concatenate([img_idx, img_idx[:1]], axis=0)
    img_idx = img_idx.at[-1, -1].set(-2)  # ID_CLS_TOKEN

    # Get 2D coordinates
    frequencies_x = img_idx % idx  # x coordinates
    frequencies_y = img_idx // idx  # y coordinates

    # Compute frequency dimensions
    freq_dim = self.hidden_size // self.num_attention_heads // 2
    rope_freq = 1.0 / (self.rope_theta ** (jnp.arange(0, freq_dim, 2)[: (freq_dim // 2)].astype(jnp.float32) / freq_dim))

    # Compute frequencies for x and y coordinates
    freqs_x = (frequencies_x + 1)[..., None] * rope_freq[None, None, :]
    freqs_y = (frequencies_y + 1)[..., None] * rope_freq[None, None, :]

    # Interleave x and y frequencies
    freqs_x = jnp.repeat(freqs_x, 2, axis=-1)
    freqs_y = jnp.repeat(freqs_y, 2, axis=-1)

    # Combine frequencies
    freqs = jnp.concatenate([freqs_x, freqs_y], axis=-1).astype(jnp.float32)
    freqs = freqs[..., ::2]

    # Mask out invalid positions
    freqs = jnp.where(img_idx.reshape(-1, 1, 1) < 0, 0, freqs)
    # Convert to complex representation
    return jnp.exp(1j * freqs)

  def __call__(self, inputs: Array, position: None | Array = None) -> Array:
    \"\"\"Applies rotary embeddings to the input tensor for Llama4 vision encoder.

    Args:
      inputs: Input tensor of shape [batch_size_times_tiles, num_patches_incl_cls, num_heads, head_dim]

    Returns:
      Tensor with rotary embeddings applied, maintaining the same shape as input.
    \"\"\"
    if len(inputs.shape) != 4:
      raise ValueError(
          \"\"\"Input is assumed to be a rank 4 tensor of shape [batch_size_times_tiles, num_patches_incl_cls,
          num_heads, head_dim].\"\"\"
      )

    # Reshape inputs to complex representation
    B, S, N, H = inputs.shape
    half_dim = H // 2

    # Convert the last dimension into a complex representation.
    # First reshape so that each pair of numbers represents the real and imaginary parts.
    inputs_reshaped = inputs.reshape(B, S, N, half_dim, 2)
    inputs_complex = inputs_reshaped[..., 0] + 1j * inputs_reshaped[..., 1]

    # Reshape freqs_ci for broadcasting
    freqs_ci = self.freqs_cis[jnp.newaxis, :, :, :]

    # Apply rotary transformation
    rotated = inputs_complex * freqs_ci

    # Convert the complex result back to a real tensor.
    # Split the complex number into its real and imaginary parts.
    rotated_real = jnp.stack([jnp.real(rotated), jnp.imag(rotated)], axis=-1)
    output = rotated_real.reshape(B, S, N, H)

    if self.cast_as_fprop_dtype:
      output = output.astype(self.fprop_dtype)

    return output


class Qwen3OmniMoeVisionRotaryEmbedding(nnx.Module):
  \"\"\"Rotary position embedding for Qwen3OmniMoe vision encoder.

  Attributes:
    hidden_size: Hidden dimension size
    num_attention_heads: Number of attention heads
    spatial_merge_size: Spatial merge block size (e.g., 2 for 2x2 blocks)
    rope_theta: Base theta for frequency computation (default 10000.0)
    cast_as_fprop_dtype: Whether to cast to fprop dtype
    fprop_dtype: Output dtype
    rngs: RNG state passed in by nnx.bridge.to_linen, not used in this module
  \"\"\"

  def __init__(
      self,
      hidden_size: int,
      num_attention_heads: int,
      spatial_merge_size: int,
      rope_theta: float = 10000.0,
      cast_as_fprop_dtype: bool = True,
      fprop_dtype: DType = jnp.bfloat16,
      rngs: nnx.Rngs = None,
  ):
    \"\"\"Initializes the Qwen3OmniMoe vision rotary embedding.

    Args:
      hidden_size: Hidden dimension size
      num_attention_heads: Number of attention heads
      spatial_merge_size: Spatial merge block size (e.g., 2 for 2x2 blocks)
      rope_theta: Base theta for frequency computation (default 10000.0)
      cast_as_fprop_dtype: Whether to cast to fprop dtype
      fprop_dtype: Output dtype
      rngs: RNG state passed in by nnx.bridge.to_linen, not used in this module
    \"\"\"
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.spatial_merge_size = spatial_merge_size
    self.rope_theta = rope_theta
    self.cast_as_fprop_dtype = cast_as_fprop_dtype
    self.fprop_dtype = fprop_dtype
    self.rngs = rngs
    self.head_dim = self.hidden_size // self.num_attention_heads

  def _compute_freq_table(self, max_hw: int) -> Array:
    \"\"\"Precompute frequency table for positions up to max_hw.

    Args:
      max_hw: Maximum height or width dimension

    Returns:
      Array of shape [max_hw, head_dim//4] containing frequencies for each position
    \"\"\"

    inv_freq = 1.0 / (self.rope_theta ** (jnp.arange(0, self.head_dim // 2, 2, dtype=jnp.float32) / (self.head_dim // 2)))
    # Compute for all positions [0, max_hw)
    positions = jnp.arange(max_hw, dtype=jnp.float32)
    freqs = jnp.outer(positions, inv_freq)  # [max_hw, head_dim//4]
    return freqs

  def _generate_position_ids_single(self, num_frames: int, height: int, width: int) -> Array:
    \"\"\"Generate 2D position IDs for a single image or video.

    Args:
      num_frames: Number of temporal frames (1 for images, >1 for videos)
      height: Height in patches
      width: Width in patches

    Returns:
      Array of shape [num_frames * height * width, 2] with (row_id, col_id)
    \"\"\"
    merge_size = self.spatial_merge_size
    merged_h = height // merge_size
    merged_w = width // merge_size

    # Block indices
    block_rows = jnp.arange(merged_h)  # [merged_h]
    block_cols = jnp.arange(merged_w)  # [merged_w]

    # Intra-block offsets
    intra_row = jnp.arange(merge_size)  # [merge_size]
    intra_col = jnp.arange(merge_size)  # [merge_size]

    # Full resolution positions using broadcasting
    # Shape: [merged_h, 1, merge_size, 1]
    row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
    # Shape: [1, merged_w, 1, merge_size]
    col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

    # Expand to full grid and flatten
    row_idx = jnp.broadcast_to(row_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)
    col_idx = jnp.broadcast_to(col_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)

    coords = jnp.stack([row_idx, col_idx], axis=-1)  # [h*w, 2]

    # Repeat for video frames
    if num_frames > 1:
      coords = jnp.tile(coords, (num_frames, 1))

    return coords

  def compute_cos_sin(self, num_frames: int, height: int, width: int) -> tuple[Array, Array]:
    \"\"\"Compute cos and sin embeddings for given static grid dimensions.

    Args:
      num_frames: Number of temporal frames
      height: Height in patches
      width: Width in patches

    Returns:
      Tuple of (cos_emb, sin_emb) each of shape [num_frames * height * width, head_dim]
    \"\"\"
    max_hw = max(height, width)
    freq_table = self._compute_freq_table(max_hw)  # [max_hw, head_dim//4]
    coords = self._generate_position_ids_single(num_frames, height, width)  # [T*H*W, 2]

    row_freqs = freq_table[coords[:, 0]]  # [T*H*W, head_dim//4]
    col_freqs = freq_table[coords[:, 1]]  # [T*H*W, head_dim//4]

    # Concatenate row and column frequencies
    embeddings = jnp.concatenate([row_freqs, col_freqs], axis=-1)  # [T*H*W, head_dim//2]

    # Double the embeddings to match head_dim
    embeddings = jnp.concatenate([embeddings, embeddings], axis=-1)  # [T*H*W, head_dim]

    cos_emb = jnp.cos(embeddings)
    sin_emb = jnp.sin(embeddings)

    if self.cast_as_fprop_dtype:
      cos_emb = cos_emb.astype(self.fprop_dtype)
      sin_emb = sin_emb.astype(self.fprop_dtype)

    return cos_emb, sin_emb

  def _rotate_half(self, x: Array) -> Array:
    \"\"\"Rotates half the hidden dims of the input.

    Args:
      x: Input tensor of any shape with last dimension divisible by 2

    Returns:
      Rotated tensor where (x1, x2) -> (-x2, x1)
    \"\"\"
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)

  def __call__(self, inputs: Array, num_frames: int, height: int, width: int) -> Array:
    \"\"\"Apply rotary position embeddings directly to inputs (Q or K tensors).

    Args:
      inputs: Input tensor of shape [B, T*H*W, N, head_dim] (batch, sequence, heads, head_dim)
             where T=num_frames, H=height, W=width (all static)
      num_frames: Number of temporal frames (static)
      height: Height in patches (static)
      width: Width in patches (static)

    Returns:
      Rotated inputs with same shape [B, T*H*W, N, head_dim]
    \"\"\"
    cos_emb, sin_emb = self.compute_cos_sin(num_frames, height, width)

    if len(inputs.shape) == 4:
      cos_emb = cos_emb[None, :, None, :]  # [1, S, 1, H]
      sin_emb = sin_emb[None, :, None, :]
    elif len(inputs.shape) == 3:
      # For [S, N, H] case
      cos_emb = cos_emb[:, None, :]  # [S, 1, H]
      sin_emb = sin_emb[:, None, :]

    rotated = inputs * cos_emb + self._rotate_half(inputs) * sin_emb

    return rotated


def qwen3omnimoe_vision_pos_embed_interpolate_as_linen(
    *,
    num_position_embeddings: int,
    hidden_size: int,
    spatial_merge_size: int,
    dtype: DType = jnp.float32,
    cast_as_fprop_dtype: bool = True,
    fprop_dtype: DType = jnp.bfloat16,
    name: str | None = None,
):
  \"\"\"Initializes Qwen3OmniMoe bilinear position embedding interpolation as Linen module.

  This implements fast bilinear interpolation of learned 2D positional embeddings
  for dynamic input sizes. The embeddings are learned on a fixed grid and interpolated
  to match the actual image/video dimensions.

  Args:
    num_position_embeddings: Number of position embeddings in the fixed grid (e.g., 1024 for 32x32)
    hidden_size: Hidden dimension size
    spatial_merge_size: Size of spatial merging blocks
    dtype: Data type for embeddings
    cast_as_fprop_dtype: Whether to cast the output to the fprop dtype
    fprop_dtype: The dtype of the output
    name: Module name

  Returns:
    A Linen module that wraps the NNX Qwen3OmniMoeVisionPosEmbedInterpolate module.
  \"\"\"
  return nnx_wrappers.to_linen(
      Qwen3OmniMoeVisionPosEmbedInterpolate,
      num_position_embeddings=num_position_embeddings,
      hidden_size=hidden_size,
      spatial_merge_size=spatial_merge_size,
      dtype=dtype,
      cast_as_fprop_dtype=cast_as_fprop_dtype,
      fprop_dtype=fprop_dtype,
      metadata_fn=variable_to_logically_partitioned,
      name=name,
  )


class Qwen3OmniMoeVisionPosEmbedInterpolate(nnx.Module):
  \"\"\"Bilinear interpolation of learned 2D positional embeddings for Qwen3OmniMoe vision.

  This module maintains a fixed grid of learned positional embeddings and interpolates
  them to match dynamic input dimensions using bilinear interpolation. This allows
  the model to handle images/videos of varying sizes while using a fixed embedding table.

  Attributes:
    num_position_embeddings: Number of position embeddings in the fixed grid
    hidden_size: Hidden dimension size
    spatial_merge_size: Spatial merge block size
    dtype: Data type for embeddings
    cast_as_fprop_dtype: Whether to cast to fprop dtype
    fprop_dtype: Output dtype
    rngs: RNG state passed in by nnx.bridge.to_linen
  \"\"\"

  def __init__(
      self,
      num_position_embeddings: int,
      hidden_size: int,
      spatial_merge_size: int,
      dtype: DType = jnp.float32,
      cast_as_fprop_dtype: bool = True,
      fprop_dtype: DType = jnp.bfloat16,
      rngs: nnx.Rngs = None,
  ):
    \"\"\"Initializes the Qwen3OmniMoe vision position embedding interpolation module.

    Args:
      num_position_embeddings: Number of position embeddings in the fixed grid
      hidden_size: Hidden dimension size
      spatial_merge_size: Spatial merge block size
      dtype: Data type for embeddings
      cast_as_fprop_dtype: Whether to cast to fprop dtype
      fprop_dtype: Output dtype
      rngs: RNG state passed in by nnx.bridge.to_linen
    \"\"\"
    self.num_position_embeddings = num_position_embeddings
    self.hidden_size = hidden_size
    self.spatial_merge_size = spatial_merge_size
    self.dtype = dtype
    self.cast_as_fprop_dtype = cast_as_fprop_dtype
    self.fprop_dtype = fprop_dtype
    self.rngs = rngs

    # Initialize the learned position embedding table
    if self.rngs is not None:
      # Initialize with normal distribution scaled by hidden_size^(-0.5)
      init_fn = nnx.initializers.normal(stddev=self.hidden_size**-0.5)
      self.pos_embed = nnx.Param(
          init_fn(
              self.rngs.params(),
              (self.num_position_embeddings, self.hidden_size),
              self.dtype,
          ),
      )
    self.num_grid_per_side = int(self.num_position_embeddings**0.5)

  def _interpolate_single(self, t: int, h: int, w: int) -> tuple[Array, Array]:
    \"\"\"Compute bilinear interpolation indices and weights for a single image/video.

    Args:
      t: Number of temporal frames
      h: Target height in patches
      w: Target width in patches

    Returns:
      Tuple of (indices, weights) where:
        - indices: [4, h*w] indices into pos_embed for 4 corners
        - weights: [4, h*w] bilinear weights for 4 corners
    \"\"\"
    N = self.num_grid_per_side

    # Create interpolation coordinates
    h_idxs = jnp.linspace(0, N - 1, h)
    w_idxs = jnp.linspace(0, N - 1, w)

    # Floor and ceiling indices
    h_idxs_floor = jnp.floor(h_idxs).astype(jnp.int32)
    w_idxs_floor = jnp.floor(w_idxs).astype(jnp.int32)
    h_idxs_ceil = jnp.minimum(h_idxs_floor + 1, N - 1)
    w_idxs_ceil = jnp.minimum(w_idxs_floor + 1, N - 1)

    # Fractional parts for interpolation weights
    dh = h_idxs - h_idxs_floor
    dw = w_idxs - w_idxs_floor

    # Compute flat indices for 2D grid
    base_h = h_idxs_floor * N
    base_h_ceil = h_idxs_ceil * N

    # 4 corner indices: (floor_h, floor_w), (floor_h, ceil_w), (ceil_h, floor_w), (ceil_h, ceil_w)
    indices = jnp.stack(
        [
            (base_h[:, None] + w_idxs_floor[None, :]).reshape(-1),
            (base_h[:, None] + w_idxs_ceil[None, :]).reshape(-1),
            (base_h_ceil[:, None] + w_idxs_floor[None, :]).reshape(-1),
            (base_h_ceil[:, None] + w_idxs_ceil[None, :]).reshape(-1),
        ],
        axis=0,
    )  # [4, h*w]

    # Bilinear weights
    weights = jnp.stack(
        [
            ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1),
            ((1 - dh)[:, None] * dw[None, :]).reshape(-1),
            (dh[:, None] * (1 - dw)[None, :]).reshape(-1),
            (dh[:, None] * dw[None, :]).reshape(-1),
        ],
        axis=0,
    )  # [4, h*w]

    return indices, weights

  def __call__(self, num_frames: int, height: int, width: int) -> Array:
    \"\"\"Interpolate positional embeddings for given static grid dimensions.

    Args:
      num_frames: Number of temporal frames (static)
      height: Height in patches (static)
      width: Width in patches (static)

    Returns:
      Interpolated positional embeddings of shape [num_frames * height * width, hidden_size]
    \"\"\"
    # Get interpolation indices and weights
    indices, weights = self._interpolate_single(num_frames, height, width)  # [4, h*w], [4, h*w]

    # Lookup embeddings for all 4 corners
    corner_embeds = self.pos_embed.value[indices]  # [4, h*w, hidden_size]

    # Apply bilinear weights and sum
    weighted_embeds = corner_embeds * weights[:, :, None]  # [4, h*w, hidden_size]
    interpolated = jnp.sum(weighted_embeds, axis=0)  # [h*w, hidden_size]

    # Repeat for temporal frames
    if num_frames > 1:
      interpolated = jnp.tile(interpolated, (num_frames, 1))  # [t*h*w, hidden_size]

    # Apply spatial merge permutation
    # Reshape to [t, h, w, hidden_size] then permute for block-based processing
    merge_size = self.spatial_merge_size
    merged_h = height // merge_size
    merged_w = width // merge_size

    # Reshape: [t*h*w, hidden_size] -> [t, h, w, hidden_size]
    interpolated = interpolated.reshape(num_frames, height, width, self.hidden_size)

    # Permute for spatial merging: [t, merged_h, merge_size, merged_w, merge_size, hidden_size]
    interpolated = interpolated.reshape(num_frames, merged_h, merge_size, merged_w, merge_size, self.hidden_size)
    # -> [t, merged_h, merged_w, merge_size, merge_size, hidden_size]
    interpolated = jnp.transpose(interpolated, (0, 1, 3, 2, 4, 5))
    # Flatten back to [t*merged_h*merged_w*merge_size*merge_size, hidden_size]
    interpolated = interpolated.reshape(-1, self.hidden_size)

    if self.cast_as_fprop_dtype:
      interpolated = interpolated.astype(self.fprop_dtype)

    return interpolated


class Qwen3OmniMoeThinkerTextRotaryEmbedding(RotaryEmbedding):
  \"\"\"Multi-dimensional Rotary Position Embedding (MRoPE) for Qwen3-Omni Thinker.

  This implements MRoPE which extends standard RoPE to handle 3D position IDs
  (temporal, height, width) for multimodal sequences containing text and vision tokens.

  For text-only sequences, it uses standard 2D position IDs.
  For sequences with vision tokens, it uses 3D position IDs where:
    - Dimension 0: Temporal position
    - Dimension 1: Height position (spatial)
    - Dimension 2: Width position (spatial)

  The implementation uses an interleaved pattern that reorganizes frequency
  components from chunked [TTT...HHH...WWW] to interleaved [THTHWHTHW...].
  \"\"\"

  def __init__(
      self,
      min_timescale: int,
      max_timescale: int,
      embedding_dims: int = 0,
      cast_as_fprop_dtype: bool = True,
      fprop_dtype: DType = jnp.bfloat16,
      mrope_section: tuple[int, int, int] | None = None,
      attention_scaling: float = 1.0,
      rngs: nnx.Rngs = None,
  ):
    \"\"\"Initializes the Qwen3OmniMoeThinkerTextRotaryEmbedding module.

    Args:
      min_timescale: Start of the geometric index (typically 1).
      max_timescale: End of the geometric index (rope_theta, e.g., 1000000).
      embedding_dims: Dimension of the embedding (head_dim).
      cast_as_fprop_dtype: Whether to cast output to fprop dtype.
      fprop_dtype: The dtype of the output.
      mrope_section: Tuple of (temporal_dim, height_dim, width_dim) for MRoPE.
                     Defaults to [24, 20, 20] if None.
      attention_scaling: Scaling factor applied to cos/sin embeddings. Defaults to 1.0.
      rngs: rng keys passed in by nnx.bridge.to_linen.
    \"\"\"
    super().__init__(
        min_timescale=min_timescale,
        max_timescale=max_timescale,
        mesh=None,
        embedding_dims=embedding_dims,
        cast_as_fprop_dtype=cast_as_fprop_dtype,
        fprop_dtype=fprop_dtype,
        rngs=rngs,
    )
    self.mrope_section = mrope_section if mrope_section is not None else (24, 20, 20)
    self.attention_scaling = attention_scaling

    if self.embedding_dims % 2:
      raise ValueError("Embedding dim for rotary position embedding must be a multiple of 2.")

  def _apply_interleaved_mrope(self, freqs: jax.Array) -> jax.Array:
    \"\"\"Apply interleaved MRoPE pattern to 3D rotary embeddings.

    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THTHWHTHW...], preserving frequency continuity.

    Args:
      freqs: Shape (3, batch, seq_len, head_dim // 2)
        Dimension 0: temporal frequencies
        Dimension 1: height frequencies
        Dimension 2: width frequencies

    Returns:
      freqs_t: Shape (batch, seq_len, head_dim // 2) with interleaved pattern
    \"\"\"
    # Start with temporal frequencies (dimension 0)
    freqs_t = freqs[0]  # (batch, seq_len, head_dim // 2)

    # Create interleaved pattern
    # For each spatial dimension (H, W), place frequencies at positions:
    # offset=1 for H, offset=2 for W, with stride=3
    for dim_idx, offset in enumerate([1, 2], start=1):  # H=1, W=2
      section_size = self.mrope_section[dim_idx] * 3  # Total positions for this dimension
      # Select positions with stride 3, starting at offset
      # Use slice syntax to match PyTorch behavior
      idx = slice(offset, section_size, 3)
      # Replace those positions with the corresponding spatial frequencies
      freqs_t = freqs_t.at[..., idx].set(freqs[dim_idx, ..., idx])

    return freqs_t

  def __call__(
      self,
      inputs: jax.Array,
      position: jax.Array,
  ) -> jax.Array:
    \"\"\"Generates rotary position embeddings for multimodal sequences.

    Args:
      inputs: Input tensor of shape [batch, sequence, heads, head_dim].
      position: Position IDs with shape:
        - [batch, sequence] for text-only (2D)
        - [3, batch, sequence] for multimodal with vision (3D)
          where dim 0 = temporal, dim 1 = height, dim 2 = width

    Returns:
      Tensor of shape [batch, sequence, heads, head_dim] with RoPE applied.
    \"\"\"
    if len(inputs.shape) != 4:
      raise ValueError("Input is assumed to be a rank 4 tensor of shape [batch, sequence, heads, head_dim].")
    if self.embedding_dims != inputs.shape[3]:
      raise ValueError(
          "The embedding dims of the rotary position embedding must match the hidden dimension of the inputs."
      )

    # Handle both 2D (text-only) and 3D (multimodal) position IDs
    if position.ndim == 2:
      # Text-only: expand (batch, seq) -> (3, batch, seq) with same positions
      position = jnp.broadcast_to(position[jnp.newaxis, ...], (3,) + position.shape)
    elif position.ndim != 3 or position.shape[0] != 3:
      raise ValueError(f"Position IDs must be 2D (batch, seq) or 3D (3, batch, seq), got shape {position.shape}")

    # Compute frequencies: (3, batch, seq, 1) @ (head_dim // 2, 1) -> (3, batch, seq, head_dim // 2)
    inv_freq_expanded = (1.0 / self.timescale)[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]  # (1, 1, 1, head_dim//2)
    position_expanded = position[..., jnp.newaxis]  # (3, batch, seq, 1)
    freqs = position_expanded * inv_freq_expanded  # (3, batch, seq, head_dim//2)

    # Apply interleaved MRoPE pattern for 3D positions
    freqs = self._apply_interleaved_mrope(freqs)  # (batch, seq, head_dim//2)

    # Compute sin and cos
    # Concatenate to get full head_dim: (batch, seq, head_dim//2) -> (batch, seq, head_dim)
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # Duplicate for both halves
    cos_emb = jnp.cos(emb) * self.attention_scaling  # (batch, seq, head_dim)
    sin_emb = jnp.sin(emb) * self.attention_scaling  # (batch, seq, head_dim)

    # Expand for heads dimension: (batch, seq, head_dim) -> (batch, seq, 1, head_dim)
    cos_emb = cos_emb[:, :, jnp.newaxis, :]
    sin_emb = sin_emb[:, :, jnp.newaxis, :]

    x_out = self.apply_rotary(inputs, cos_emb, sin_emb)

    if self.cast_as_fprop_dtype:
      x_out = x_out.astype(self.fprop_dtype)

    return x_out


def qwen3_omni_mrope_embedding_as_linen(
    *,
    min_timescale: int,
    max_timescale: int,
    embedding_dims: int = 0,
    cast_as_fprop_dtype: bool = True,
    fprop_dtype: DType = jnp.bfloat16,
    mrope_section: tuple[int, int, int] | None = None,
    name: str | None = None,
):
  \"\"\"Initializes Qwen3OmniMoeThinkerTextRotaryEmbedding and returns it as a Linen module.

  Args:
    min_timescale: Start of the geometric index.
    max_timescale: End of the geometric index (rope_theta).
    embedding_dims: Dimension of the embedding (head_dim).
    cast_as_fprop_dtype: Whether to cast output to fprop dtype.
    fprop_dtype: The dtype of the output.
    mrope_section: Tuple of (temporal_dim, height_dim, width_dim) for MRoPE.
    name: Name of the Linen module.
  \"\"\"
  return nnx_wrappers.to_linen(
      Qwen3OmniMoeThinkerTextRotaryEmbedding,
      min_timescale=min_timescale,
      max_timescale=max_timescale,
      embedding_dims=embedding_dims,
      cast_as_fprop_dtype=cast_as_fprop_dtype,
      fprop_dtype=fprop_dtype,
      mrope_section=mrope_section,
      metadata_fn=variable_to_logically_partitioned,
      name=name,
  )
\n"""


# File: src/maxtext/layers/linears.py (commit 313890777)
LINEARS_RAW = """\n# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

\"\"\"Linear Layers.\"\"\"

import functools
import operator
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import jax
import jax.numpy as jnp

from jax import lax
from jax.sharding import NamedSharding, Mesh
from jax.ad_checkpoint import checkpoint_name

from flax import nnx
import flax.linen as nn

from maxtext.common.common_types import DecoderBlockType, ShardMode, DType, Array, Config
from maxtext.common.common_types import MODEL_MODE_PREFILL
from maxtext.layers import nnx_wrappers, quantizations
from maxtext.layers import normalizations
from maxtext.layers.initializers import NdInitializer, nd_dense_init, default_bias_init, variable_to_logically_partitioned
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils.sharding import maybe_shard_with_logical


def _convert_to_activation_function(fn_or_string: str | Callable[..., Any]) -> Callable[..., Any]:
  \"\"\"Convert a string to an activation function.\"\"\"
  if fn_or_string == "linear":
    return lambda x: x
  elif isinstance(fn_or_string, str):
    return getattr(nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError(
        f\"\"\"Don't know how to convert {fn_or_string}
                         to an activation function\"\"\"
    )


def normalize_axes(axes: Iterable[int], ndim: int) -> tuple[int, ...]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


def _compute_dot_general(inputs, kernel, kernel_axes, axis, contract_ind, matmul_precision, quant):
  \"\"\"Computes a dot_general operation that may be quantized.\"\"\"
  dot_general = lax.dot_general
  matmul_precision = lax.Precision(matmul_precision)
  if quant:
    dot_general_cls = quant.dot_general_cls(mesh_axes=kernel_axes)
    dot_general = dot_general_cls()
    return dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=None)
  return dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=matmul_precision)


def _compute_dot_general_nnx(
    inputs,
    kernel,
    axis,
    contract_ind,
    matmul_precision,
    quant_dot_general: nnx_wrappers.ToNNX | None,
    initializing: bool,
    out_sharding: NamedSharding | None = None,
):
  \"\"\"Computes a dot_general operation that may be quantized.\"\"\"
  dot_general = lax.dot_general
  matmul_precision = lax.Precision(matmul_precision)
  if quant_dot_general is not None:
    if initializing:
      quant_dot_general.lazy_init(inputs, kernel, ((axis, contract_ind), ((), ())), precision=None)
    return quant_dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=None, mutable=["aqt"])

  return dot_general(
      inputs, kernel, ((axis, contract_ind), ((), ())), precision=matmul_precision, out_sharding=out_sharding
  )


class DenseGeneral(nnx.Module):
  \"\"\"A linear transformation with flexible axes.\"\"\"

  def __init__(
      self,
      in_features_shape: Iterable[int] | int,
      out_features_shape: Iterable[int] | int,
      axis: Iterable[int] | int = -1,
      weight_dtype: DType = jnp.float32,
      dtype: DType = jnp.float32,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes: tuple[None | str, ...] = (),
      quant: None | Quant = None,
      use_bias: bool = False,
      shard_mode: ShardMode = ShardMode.AUTO,
      matmul_precision: str = "default",
      parameter_memory_host_offload: bool = False,
      *,  # Following arguments are keyword-only
      rngs: nnx.Rngs = None,
  ):
    \"\"\"Initializes the DenseGeneral module.

    Args:
      in_features_shape: tuple with numbers of input features for axes specified in
        'axis'.
      out_features_shape: tuple with numbers of output features.
      axis: tuple with axes to apply the transformation on.
      weight_dtype: the dtype of the weights (default: float32).
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
      kernel_axes: logical axes for partitioning the kernel.
      quant: quantization config, defaults to None implying no quantization.
      use_bias: whether to add bias in linear transformation.
      shard_mode: auto or explicit shard mode.
      matmul_precision: Precision for matrix multiplication.
      parameter_memory_host_offload: Determines whether to offload params to host
      rngs: RNG state for initialization in nnx.
    \"\"\"
    self.in_features_shape = canonicalize_tuple(in_features_shape)
    self.out_features_shape = canonicalize_tuple(out_features_shape)
    self.axis = canonicalize_tuple(axis)
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.quant = quant
    self.use_bias = use_bias
    self.shard_mode = shard_mode
    self.matmul_precision = matmul_precision
    self.parameter_memory_host_offload = parameter_memory_host_offload

    # Parameter initialization
    kernel_shape = self.in_features_shape + self.out_features_shape
    kernel_in_axis = np.arange(len(self.axis))
    kernel_out_axis = np.arange(len(self.axis), len(self.axis) + len(self.out_features_shape))

    if not quantizations.in_serve_mode(self.quant):
      self.kernel = nnx.Param(
          self.kernel_init(
              rngs.params(),
              kernel_shape,
              self.weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          sharding=self.kernel_axes,
      )

    if self.use_bias:
      bias_axes = self.kernel_axes[-len(self.out_features_shape) :]
      bias_shape = kernel_shape[-len(self.out_features_shape) :]
      self.bias = nnx.Param(
          default_bias_init(rngs.params(), bias_shape, self.weight_dtype),
          sharding=bias_axes,
      )
    else:
      self.bias = None

    if quant:
      dot_general_cls = quant.dot_general_cls(mesh_axes=kernel_axes)
      dot_general_linen = dot_general_cls()
      quant_dot_general = nnx_wrappers.ToNNX(dot_general_linen, rngs=rngs)
      self._quant_dot_general_name = f"{type(dot_general_linen).__name__}_0"
      setattr(self, self._quant_dot_general_name, quant_dot_general)
      block_size = getattr(quant, "get_block_size", lambda: 1)()  # needed for TE MXFP8
      dummy_inputs = jnp.zeros((block_size, *self.in_features_shape), dtype=self.dtype)
      self(dummy_inputs, _initializing=True)
    else:
      self._quant_dot_general_name = None

  @property
  def quant_dot_general(self) -> nnx_wrappers.ToNNX | None:
    if self._quant_dot_general_name is None:
      return None
    return getattr(self, self._quant_dot_general_name)

  def __call__(self, inputs: Array, _initializing: bool = False, out_sharding: NamedSharding | None = None) -> Array:
    \"\"\"Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    \"\"\"
    inputs = jnp.asarray(inputs, self.dtype)
    norm_axis = normalize_axes(self.axis, inputs.ndim)

    for i, ax in enumerate(norm_axis):
      if inputs.shape[ax] != self.in_features_shape[i]:
        raise ValueError(
            f"Input dimension {inputs.shape[ax]} at axis {ax} "
            f"does not match expected input feature size {self.in_features_shape[i]}"
        )

    if quantizations.in_serve_mode(self.quant):
      kernel_shape = self.in_features_shape + self.out_features_shape
      kernel = jnp.zeros(kernel_shape, dtype=self.dtype)
    else:
      kernel = self.kernel[...]
      # Move logit_dense kernel to device if parameter offloading is enabled
      if self.parameter_memory_host_offload:
        max_logging.log("linear.py: Moving parameter logits_dense kernel to device")
        kernel = jax.device_put(kernel, max_utils.device_space())
      kernel = jnp.asarray(kernel, self.dtype)

    # out_sharding should be None for auto mesh axis
    if self.shard_mode != ShardMode.EXPLICIT:
      out_sharding = None

    contract_ind = tuple(range(0, len(self.axis)))
    output = _compute_dot_general_nnx(
        inputs,
        kernel,
        norm_axis,
        contract_ind,
        self.matmul_precision,
        self.quant_dot_general,
        _initializing,
        out_sharding,
    )

    if self.bias is not None:
      bias = jnp.asarray(self.bias[...], self.dtype)
      output += bias
    return output


def dense_general(
    *,
    inputs_shape: tuple[int, ...] | None = None,
    in_features_shape: tuple[int, ...] | int | None = None,
    out_features_shape: Iterable[int] | int,
    axis: Iterable[int] | int = -1,
    weight_dtype: DType = jnp.float32,
    dtype: DType = jnp.float32,
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
    kernel_axes: tuple[None | str, ...] = (),
    quant: None | Quant = None,
    use_bias: bool = False,
    shard_mode: ShardMode = ShardMode.AUTO,
    matmul_precision: str = "default",
    parameter_memory_host_offload: bool = False,
    name: None | str = None,
):
  \"\"\"Creates a DenseGeneral Linen module using nnx.bridge.to_linen.

  Args:
    inputs_shape: tuple with the shape of the inputs
    in_features_shape: tuple with numbers of input features for axes specified in
      'axis'.
    out_features_shape: tuple with numbers of output features.
    axis: tuple with axes to apply the transformation on.
    weight_dtype: the dtype of the weights (default: float32).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
    kernel_axes: logical axes for partitioning the kernel.
    quant: quantization config, defaults to None implying no quantization.
    use_bias: whether to add bias in linear transformation.
    shard_mode: indicating the shard mode
    matmul_precision: Precision for matrix multiplication.
    parameter_memory_host_offload: Determines whether to offload params to host
    name: name passed to the ToLinen Module
  \"\"\"
  if not (inputs_shape is not None) ^ (in_features_shape is not None):
    raise ValueError("Exactly one of inputs_shape or in_features must be specified.")

  if inputs_shape is not None:
    axis = canonicalize_tuple(axis)
    in_features_shape = tuple(inputs_shape[ax] for ax in normalize_axes(axis, len(inputs_shape)))
  else:
    assert in_features_shape is not None
  module = nnx_wrappers.to_linen(
      DenseGeneral,
      in_features_shape=in_features_shape,
      out_features_shape=out_features_shape,
      axis=axis,
      weight_dtype=weight_dtype,
      dtype=dtype,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes,
      quant=quant,
      use_bias=use_bias,
      shard_mode=shard_mode,
      matmul_precision=matmul_precision,
      parameter_memory_host_offload=parameter_memory_host_offload,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )
  return module


class Dropout(nnx.Dropout):
  \"\"\"Forked nnx.Dropout that is easier to use with bridge\"\"\"

  def __init__(  # pylint: disable=super-init-not-called
      self,
      rate: float,
      *,
      broadcast_dims: Sequence[int] = (),
      deterministic: bool = False,
      rng_collection: str = "dropout",
      rngs: nnx.Rngs | None = None,
  ):
    self.rate = rate
    self.broadcast_dims = broadcast_dims
    self.deterministic = deterministic
    self.rng_collection = rng_collection

    if isinstance(rngs, nnx.Rngs):
      self.rngs = rngs.fork() if hasattr(type(rngs), "fork") else rngs
    else:
      raise TypeError(f"rngs must be a Rngs, RngStream or None, but got {type(rngs)}.")


class MlpBlock(nnx.Module):
  \"\"\"Transformer MLP / feed-forward block.\"\"\"

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      in_features: int,
      intermediate_dim: int = 2048,
      activations: Sequence[str | Callable[..., Any]] = ("relu",),
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
      intermediate_dropout_rate: float = 0.1,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      use_bias: bool = False,
      use_pre_norm: bool = False,
      quant: None | Quant = None,
      model_mode: None | str = None,
      *,
      rngs: nnx.Rngs,
  ) -> None:
    \"\"\"A MlpBlock module.

    Args:
      config: Config object containing model parameters.
      mesh: Mesh object of device and physical axes information
      in_features: Number of input features.
      intermediate_dim: Shared dimension of hidden layers.
      activations: Type of activations for each layer.  Each element is either
        'linear', a string function name in flax.linen, or a function.
      kernel_init: Kernel function, passed to the dense layers.
      deterministic: Whether the dropout layers should be deterministic.
      intermediate_dropout_rate: Dropout rate used after the intermediate layers.
      dtype: computation data type for the dense layer.
      weight_dtype: weight data type for the dense layer.
      use_bias: whether to add bias in all feedforward layers.
      use_pre_norm: whether to add pre layer norm in mlp layers.
      quant: Optional quantization config, no quantization if None.
      out_sharding: Named sharding of outputs
    \"\"\"
    self.config = config
    self.mesh = mesh
    self.in_features = in_features
    self.intermediate_dim = intermediate_dim
    self.activations = activations
    self.kernel_init = kernel_init
    self.intermediate_dropout_rate = intermediate_dropout_rate
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.use_bias = use_bias
    self.use_pre_norm = use_pre_norm
    self.quant = quant
    self.model_mode = model_mode

    if self.use_pre_norm:
      self.mlp_layer_norm = self.get_norm_layer(num_features=in_features)(
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          kernel_axes=("norm",),
          epsilon=config.normalization_layer_epsilon,
          rngs=rngs,
      )
    else:
      self.mlp_layer_norm = None

    if self.model_mode == MODEL_MODE_PREFILL:
      self.intermediate_logical = ("activation_batch", "prefill_activation_length", "activation_mlp")
    else:
      self.intermediate_logical = ("activation_batch", "activation_length", "activation_mlp")

    if config.fused_mlp:
      self.wi = DenseGeneral(
          in_features_shape=in_features,
          out_features_shape=(len(self.activations), self.intermediate_dim),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          kernel_init=self.kernel_init,
          kernel_axes=("embed", "num_activations", "mlp"),
          quant=self.quant,
          use_bias=self.use_bias,
          shard_mode=self.config.shard_mode,
          matmul_precision=self.config.matmul_precision,
          rngs=rngs,
      )
    else:
      for idx in range(len(self.activations)):
        dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
        module = DenseGeneral(
            in_features_shape=in_features,
            out_features_shape=self.intermediate_dim,
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            kernel_init=self.kernel_init,
            kernel_axes=("embed", "mlp"),
            quant=self.quant,
            use_bias=self.use_bias,
            shard_mode=self.config.shard_mode,
            matmul_precision=self.config.matmul_precision,
            rngs=rngs,
        )
        setattr(self, dense_name, module)
    self.dropout = Dropout(rate=self.intermediate_dropout_rate, broadcast_dims=(-2,), rngs=rngs)
    self.wo = DenseGeneral(
        in_features_shape=self.intermediate_dim,
        out_features_shape=in_features,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_init=self.kernel_init,
        kernel_axes=("mlp", "embed"),
        quant=self.quant,
        use_bias=self.use_bias,
        shard_mode=self.config.shard_mode,
        matmul_precision=self.config.matmul_precision,
        rngs=rngs,
    )

    self._maybe_shard_with_logical = functools.partial(
        maybe_shard_with_logical,
        mesh=mesh,
        shard_mode=config.shard_mode,
        debug_sharding=config.debug_sharding,
    )

  def get_norm_layer(self, num_features: int):
    \"\"\"get normalization layer.\"\"\"
    if self.config.decoder_block in (
        DecoderBlockType.DEFAULT,
        DecoderBlockType.LLAMA2,
        DecoderBlockType.MISTRAL,
        DecoderBlockType.MIXTRAL,
        DecoderBlockType.GEMMA,
        DecoderBlockType.GEMMA2,
        DecoderBlockType.GEMMA3,
        DecoderBlockType.QWEN3,
        DecoderBlockType.DEEPSEEK,
        DecoderBlockType.LLAMA4,
    ):
      return functools.partial(normalizations.RMSNorm, num_features=num_features)
    elif self.config.decoder_block == DecoderBlockType.GPT3:
      from maxtext.models import gpt3  # pylint: disable=import-outside-toplevel

      return functools.partial(
          gpt3.Gpt3LayerNorm, num_features=num_features, reductions_in_fp32=False, use_bias=self.use_bias
      )
    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block.value=}")

  def __call__(
      self,
      inputs,
      decode: bool = False,
      deterministic: bool = False,
      intermediate_sharding: NamedSharding | None = None,
      out_sharding: NamedSharding | None = None,
  ):
    \"\"\"Applies Transformer MlpBlock module.\"\"\"
    cfg = self.config

    if self.mlp_layer_norm is not None:
      inputs = self.mlp_layer_norm(inputs)

    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    activations = []
    if cfg.fused_mlp:
      x = self.wi(inputs, out_sharding=intermediate_sharding)
      x = checkpoint_name(x, "mlpwi")
      for idx, act_fn in enumerate(self.activations):
        y = _convert_to_activation_function(act_fn)(x[:, :, idx, ...])
        activations.append(y)
    else:
      for idx, act_fn in enumerate(self.activations):
        dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
        module = getattr(self, dense_name)
        x = module(inputs, out_sharding=intermediate_sharding)
        x = checkpoint_name(x, "mlp" + dense_name)
        if cfg.activations_in_float32:
          x = x.astype(jnp.float32)
        x = _convert_to_activation_function(act_fn)(x)
        activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations).astype(self.dtype)
    # Apply dropout and final dense output projection.
    x = self.dropout(x, deterministic=deterministic)  # Broadcast along length.
    x = self._maybe_shard_with_logical(x, self.intermediate_logical)
    output = self.wo(x, out_sharding=out_sharding)

    output = checkpoint_name(output, "mlpwo")
    return output


def mlp_block(
    *,
    config: Config,
    mesh: Mesh,
    in_features: int,
    intermediate_dim: int = 2048,
    activations: Sequence[str | Callable[..., Any]] = ("relu",),
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
    intermediate_dropout_rate: float = 0.1,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    use_bias: bool = False,
    use_pre_norm: bool = False,
    quant: None | Quant = None,
    model_mode: None | str = None,
    name: None | str = None,
):
  \"\"\"Creates a MlpBlock Linen module using nnx.bridge.to_linen.\"\"\"
  module = nnx_wrappers.to_linen(
      MlpBlock,
      config=config,
      mesh=mesh,
      in_features=in_features,
      intermediate_dim=intermediate_dim,
      activations=activations,
      kernel_init=kernel_init,
      intermediate_dropout_rate=intermediate_dropout_rate,
      dtype=dtype,
      weight_dtype=weight_dtype,
      use_bias=use_bias,
      use_pre_norm=use_pre_norm,
      quant=quant,
      model_mode=model_mode,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )
  return module
\n"""


# File: src/maxtext/layers/attention_mla.py (commit 313890777)
ATTENTION_MLA_RAW = """\n#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

\"\"\"MLA Attention Layer.\"\"\"

import math
from typing import Any, Optional, Tuple
import copy

import jax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import layout
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding

Layout = layout.Format
if jax.__version_info__ >= (0, 6, 3):
  DLL = layout.Layout
else:
  DLL = layout.DeviceLocalLayout  # type: ignore

from flax import nnx

from maxtext.common.common_types import (
    Array,
    AxisIdxes,
    AxisNames,
    BATCH_ATTN,
    CACHE_BATCH,
    CACHE_BATCH_PREFILL,
    CACHE_SEQUENCE,
    CACHE_HEADS_NONE,
    CACHE_KV,
    Config,
    DECODE_BATCH,
    DECODE_LENGTH,
    D_KV,
    DType,
    EMBED,
    HEAD,
    Q_LORA_UP_PROJ,
    KV_BATCH,
    KV_HEAD,
    KV_HEAD_DIM,
    KV_LORA_UP_PROJ,
    LENGTH,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    PREFILL_KV_BATCH,
    PREFILL_LENGTH,
    AttentionType,
    DEFAULT_MASK_VALUE,
)

from maxtext.layers import nnx_wrappers
from maxtext.layers.attentions import Attention
from maxtext.layers.initializers import nd_dense_init, NdInitializer, variable_to_logically_partitioned
from maxtext.layers.linears import DenseGeneral
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.inference import kvcache
from maxtext.inference import page_manager
from maxtext.inference import paged_attention
from maxtext.inference.kvcache import KVQuant
from maxtext.utils.sharding import create_sharding
from maxtext.utils.globals import EPS


PLACEHOLDER_SEQ_LEN = 1


class Indexer(nnx.Module):
  \"\"\"Indexer for DeepSeek Sparse Attention (DSA).

  This module implements the sparse attention indexer introduced in DeepSeek
  V3.2.
  It computes relevance scores to select the top-k most relevant tokens for
  attention.

  References:
    DeepSeek-AI, `DeepSeek-V3.2: Pushing the Frontier of Open Large Language
    Models
      <https://arxiv.org/pdf/2512.02556>`_, 2026
    Implementation:
    https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py
  \"\"\"

  def __init__(
      self,
      config: Any,
      rotary_embedding,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      quant: Optional[Quant] = None,
      model_mode: str = MODEL_MODE_TRAIN,
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.config = config
    self.rotary_embedding = rotary_embedding
    self.quant = quant
    self.kernel_init = kernel_init
    self.model_mode = model_mode
    self.rngs = rngs
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype
    self.max_target_length = config.max_target_length

    self.n_heads = config.indexer_n_heads
    self.head_dim = config.indexer_head_dim
    self.indexer_topk = config.indexer_topk
    self.emb_dim = config.emb_dim
    self.rope_head_dim = config.qk_rope_head_dim
    self.q_lora_rank = config.q_lora_rank
    # scale head weights for numerical stability
    self.softmax_scale = self.head_dim**-0.5

    # Query Projection: Latent Query -> Indexer Query
    self.wq_b = DenseGeneral(
        in_features_shape=self.q_lora_rank,
        out_features_shape=(self.n_heads, self.head_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("q_lora", "q_heads", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

    # Key Projection: Input -> Shared Indexer Key
    self.wk = DenseGeneral(
        in_features_shape=self.emb_dim,
        out_features_shape=self.head_dim,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

    # Key Normalization with Bias
    self.k_norm = nnx.LayerNorm(num_features=self.head_dim, use_bias=True, dtype=self.weight_dtype, rngs=rngs)

    # Projection: Input -> Importance Weights for Heads
    # deepseek3.2 enforces FP32 and does not quantize, for precision and stability.
    self.weights_proj = DenseGeneral(
        in_features_shape=self.emb_dim,
        out_features_shape=self.n_heads,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "q_heads"),
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        quant=None,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

  def update_indexer_cache(self, kv_cache, k, decoder_segment_ids, model_mode, previous_chunk):
    \"\"\"Updates Indexer buffers by processing KV cache results.\"\"\"
    k_expanded = k[:, :, jnp.newaxis, :]
    p_res, a_res = kv_cache(
        key=k_expanded,
        value=k_expanded,
        decoder_segment_ids=decoder_segment_ids,
        model_mode=model_mode,
        use_ragged_attention=self.config.use_ragged_attention,
        previous_chunk=previous_chunk,
    )

    # Filter out None values to handle PREFILL vs AR modes uniformly
    active_results = [res for res in [p_res, a_res] if res is not None]

    if not active_results:
      return None, None

    # Extract keys (index 0) and segment IDs (index 2)
    keys = jnp.concatenate([res[0] for res in active_results], axis=1)
    segs = jnp.concatenate([res[2] for res in active_results], axis=1)

    # squeeze(2) removes the jnp.newaxis added above
    return keys.squeeze(2), segs

  def apply_partial_rope(
      self,
      inputs: Array,
      inputs_positions: Optional[Array | None] = None,
  ):
    \"\"\"Applies partial RoPE to the indexer query or key

    The Indexer's RoPE implementation differs from MLA's in two key aspects:
    1. Split Order: Indexer splits the head dimension into [rope, nope], whereas MLA uses [nope, rope].
    2. Input Layout: Indexer uses concatenated layout (interleave=False), whereas MLA uses interleaved (interleave=True).

    Args:
      inputs: Input array of shape [batch, seqlen, indexer_n_heads, indexer_head_dim].
      positions: Position array of shape [batch, seqlen].

    Returns:
      Array with partial RoPE applied, with shape [batch, seqlen, indexer_n_heads, indexer_head_dim]
    \"\"\"
    # indexer_head_dim -> [rope_head_dim, indexer_head_dim - rope_head_dim]
    x_pe, x_nope = jnp.split(inputs, [self.rope_head_dim], axis=-1)
    # x_pe [B, S, H, rope_head_dim], positions [B, S]
    x_pe = self.rotary_embedding(x_pe, position=inputs_positions)
    x = jnp.concatenate([x_pe, x_nope], axis=-1)
    return x

  def generate_mask(self, topk_indices, s):
    \"\"\"
    Creates a mask for top-k indices.

    Args:
        topk_indices: [b, t, k] int - The indices to keep.
        s: int - The total size to select from.

    Returns:
        mask: [b, t, s] - `0.0` at topk_indices, `DEFAULT_MASK_VALUE` (large negative) elsewhere.
    \"\"\"
    # 1. Create a range [0, 1, ..., s-1]
    # 2. Broadcast compare against [b, t, k] to get [b, t, k, s]
    # 3. Use .any() to see if a s-index is present in any of the k slots
    is_topk = (jnp.arange(s) == topk_indices[..., None]).any(axis=-2)
    # 4. Use where to select between 0.0 and the mask value
    # cast values to dtype
    val_true = jnp.array(0.0, dtype=self.dtype)
    val_false = jnp.array(DEFAULT_MASK_VALUE, dtype=self.dtype)
    return jnp.where(is_topk, val_true, val_false)

  def __call__(
      self,
      inputs_q: Array,
      low_rank_q: Array,
      inputs_kv: Array,
      inputs_positions: Optional[Array | None] = None,
      attention_mask: Optional[Array | None] = None,
      decoder_segment_ids: Optional[Array | None] = None,
      previous_chunk: Any = None,
      kv_cache: Any = None,
      model_mode: str = MODEL_MODE_TRAIN,
  ):
    \"\"\"Computes the index score to determine the top-k relevant tokens.

    This uses a ReLU-based similarity for QK with MQA-style broadcasting (shared K).
    It uses weighted aggregation over heads to produce a single score per token pair.

    Steps:
      1. Q = RoPE(Wq @ q_lora)
      2. K = RoPE(Norm(Wk @ X))
      3. Logits = ReLU(Q @ K.T)                      # Pairwise similarity
      4. Head_Weights = (W_proj @ X) * scale         # Dynamic head importance, scale for stability
      5. Score = Logits @ Head_Weights               # Aggregate heads
      6. Indices = ArgTopk(Score)

    Args:
      inputs_q: Input of shape [b, t, embed_dim].
      low_rank_q: Low-rank latent query representations of shape [b, t, q_lora_rank].
      inputs_kv: Input of shape [b, s, embed_dim], same as inputs_q
      inputs_positions: Position indices of shape [b, s].
      attention_mask: Optional attention mask of shape [b, t, s].
        Positions with `0.0` allow attention, while positions with
        `DEFAULT_MASK_VALUE` (a large negative number) prevent it.
        Returns `None` if no masking is determined to be necessary based on
        the inputs and configuration.
      decoder_segment_ids: Segment IDs for decoder masking.
      previous_chunk: Previous chunk info for prefill.
      kv_cache: Key-value cache used when serving models.
      model_mode: "train", "prefill", or "autoregressive".

    Returns:
      indexer_mask: A sparse mask [b, t, s] with 0.0 for top-k selected tokens
        and large negative values otherwise.
      topk_indices: Indices of the top-k selected tokens [b, t, k].
      indexer_score: The computed relevance scores [b, t, s].

    Notation:
      b: Batch size
      t: Query Sequence Length (Target), note t = s here
      s: Key/Value Sequence Length (Source)
      h: Number of Indexer Heads (indexer_n_heads)
      d: Indexer Head Dimension (indexer_head_dim)
    \"\"\"
    bsz, seqlen, _ = inputs_q.shape  # s = t = seqlen
    # ==============================================================================
    # Gradient Isolation Strategy: Main Model vs. Indexer
    # ==============================================================================
    # This creates a barrier to train both components independently, and applies
    # for both Dense Warm-up and Sparse Training stages:
    #
    # Forward Pass:
    # - The Indexer receives a detached copy of the inputs (via `stop_gradient`)
    #   to independently calculate its scores and `indexer_loss`.
    #
    # Backward Pass (Main Model):
    # - The main model optimizes its weights based solely on the LM loss.
    # - The `indexer_mask` in the Attention layer prevents gradients from the main
    #   loss from flowing into the Indexer's weights.
    #
    # Backward Pass (Indexer):
    # - Gradients from the `indexer_loss` flow back to update the Indexer's weights.
    # - The `stop_gradient` applied to the inputs acts as a mathematical wall, dropping
    #   gradients to 0.0 and preventing the Indexer loss from altering the main model's
    #   earlier layers.
    inputs_q = jax.lax.stop_gradient(inputs_q)
    low_rank_q = jax.lax.stop_gradient(low_rank_q)
    inputs_kv = jax.lax.stop_gradient(inputs_kv)

    # Query Processing: Project from Latent low_rank_q
    q = self.wq_b(low_rank_q)  # [b, t, q_lora_rank] -> [b, t, h * d]
    q = q.reshape(bsz, seqlen, self.n_heads, self.head_dim)  # [b, t, h, d]
    q = self.apply_partial_rope(q, inputs_positions=inputs_positions)

    # Key Processing: Project from Input
    k = self.wk(inputs_kv)  # [b, s, embed_dim] -> [b, s, d]
    k = self.k_norm(k)
    k = k[:, :, None, :]  # [b, s, d] -> [b, s, 1, d]
    k = self.apply_partial_rope(k, inputs_positions=inputs_positions)
    k = k.squeeze(2)  # [b, s, 1, d] -> [b, s, d]

    # Update and retrieve from cache if not training
    cached_s = None
    if model_mode != MODEL_MODE_TRAIN:
      k_cached, cached_s = self.update_indexer_cache(kv_cache, k, decoder_segment_ids, model_mode, previous_chunk)
      k = k_cached if k_cached is not None else k

    # NOTE: If the total available sequence length <= topk, indexer always selects all tokens.
    if k.shape[1] <= self.indexer_topk:
      return None, None, None

    # Compute Index Scores
    # QK product: relu(q @ k.T), [b, t, s, h]
    # Similar to MQA, each key is shared by h query head
    logits = jnp.einsum("bthd, bsd -> btsh", q, k, precision=self.config.matmul_precision)
    logits = jax.nn.relu(logits)
    # Compute head weights: project from input, [b, t, embed_dim] -> [b, t, h]
    weights = self.weights_proj(inputs_q)
    # Weights scaling affect indexer_score, but does not affect topk_indices. Keep scaling for numerical stability.
    # https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/inference/model.py#L478-L480
    weights = weights * (self.n_heads**-0.5) * self.softmax_scale
    # Aggregate head-wise logits: logits @ weights
    indexer_score = jnp.einsum("btsh, bth -> bts", logits, weights, precision=self.config.matmul_precision)  # [b, t, s]

    internal_padding_mask = None
    if cached_s is not None:
      # cached_s marks valid tokens from the original prefill step and all subsequent AR steps
      internal_padding_mask = jnp.where(cached_s > 0, 0.0, DEFAULT_MASK_VALUE)
      indexer_score += internal_padding_mask[:, None, :]

    # Apply attention mask before TopK
    if attention_mask is not None:
      indexer_score += attention_mask

    # TopK selection based on index score
    _, topk_indices = jax.lax.top_k(indexer_score, k=self.indexer_topk)  # topk_indices [b, t, k]

    # Create Sparse Index Mask: 0 and large negatives
    indexer_mask = self.generate_mask(topk_indices, k.shape[1])  # [b, t, s]

    # Re-apply attention mask after TopK: in case number of unmasked tokens < TopK
    if attention_mask is not None:
      indexer_mask += attention_mask

    if internal_padding_mask is not None:
      indexer_mask += internal_padding_mask[:, None, :]

    return indexer_mask, topk_indices, indexer_score


def mla_as_linen(
    *,
    config: Config,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_target_length: int,
    mesh: Mesh,
    attention_kernel: str,
    inputs_q_shape: Tuple,
    inputs_kv_shape: Tuple,
    dtype: DType = jnp.float32,
    weight_dtype: DType = jnp.float32,
    max_prefill_predict_length: int = -1,
    dropout_rate: float = 0.0,
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
    float32_qk_product: bool = False,  # computes logits in float32 for stability.
    float32_logits: bool = False,  # cast logits in float32 for stability.
    quant: Optional[Quant] = None,
    kv_quant: Optional[KVQuant] = None,
    attention_type: AttentionType = AttentionType.MLA,  # Default to MLA attention
    attn_logits_soft_cap: float | None = None,
    sliding_window_size: int | None = None,
    use_ragged_attention: bool = False,
    ragged_block_size: int = 256,
    use_qk_norm: bool = False,
    query_pre_attn_scalar: float | None = None,
    use_bias_in_projections: bool = False,  # Set to True will enable bias in q, k, v, o projections
    # Temperature tuning parameters used for Llama4
    temperature_tuning: bool = False,
    temperature_tuning_scale: float = 0.1,
    temperature_tuning_floor_scale: float = 8192.0,
    # Shard the query activation as the same as the key and value.
    prefill_query_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
    prefill_key_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
    prefill_value_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
    query_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM),
    key_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM),
    value_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM),
    input_axis_names: AxisNames = (BATCH_ATTN, LENGTH, EMBED),
    out_axis_names: AxisNames = (BATCH_ATTN, LENGTH, HEAD, D_KV),
    prefill_input_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, EMBED),
    decode_input_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, EMBED),
    prefill_out_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, HEAD, D_KV),
    decode_out_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV),
    prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
    ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
    compute_axis_order: AxisIdxes = (0, 1, 2, 3),
    reshape_q: bool = False,
    is_nope_layer: bool = False,
    is_vision: bool = False,
    model_mode: str = MODEL_MODE_TRAIN,
    q_lora_rank: int = 0,
    kv_lora_rank: int = 512,
    qk_nope_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    v_head_dim: int = 128,
    max_position_embeddings: int = 4096 * 4,
    original_max_position_embeddings: int = 4096,
    mscale: float = 1.0,  # scaling factor for softmax
    rope_factor: float = 40.0,  # rotary embedding factor
    name: str | None = None,
):
  \"\"\"A factory function to create an MLA as a Linen module.

  This function serves as a bridge to use the NNX-based `MLA` within a
  Linen model.
  \"\"\"
  return nnx_wrappers.to_linen(
      MLA,
      config=config,
      num_query_heads=num_query_heads,
      num_kv_heads=num_kv_heads,
      head_dim=head_dim,
      max_target_length=max_target_length,
      mesh=mesh,
      attention_kernel=attention_kernel,
      inputs_q_shape=inputs_q_shape,
      inputs_kv_shape=inputs_kv_shape,
      dtype=dtype,
      weight_dtype=weight_dtype,
      max_prefill_predict_length=max_prefill_predict_length,
      dropout_rate=dropout_rate,
      kernel_init=kernel_init,
      float32_qk_product=float32_qk_product,
      float32_logits=float32_logits,
      quant=quant,
      kv_quant=kv_quant,
      attention_type=attention_type,
      attn_logits_soft_cap=attn_logits_soft_cap,
      sliding_window_size=sliding_window_size,
      use_ragged_attention=use_ragged_attention,
      ragged_block_size=ragged_block_size,
      use_qk_norm=use_qk_norm,
      query_pre_attn_scalar=query_pre_attn_scalar,
      use_bias_in_projections=use_bias_in_projections,
      temperature_tuning=temperature_tuning,
      temperature_tuning_scale=temperature_tuning_scale,
      temperature_tuning_floor_scale=temperature_tuning_floor_scale,
      prefill_query_axis_names=prefill_query_axis_names,
      prefill_key_axis_names=prefill_key_axis_names,
      prefill_value_axis_names=prefill_value_axis_names,
      query_axis_names=query_axis_names,
      key_axis_names=key_axis_names,
      value_axis_names=value_axis_names,
      input_axis_names=input_axis_names,
      out_axis_names=out_axis_names,
      prefill_input_axis_names=prefill_input_axis_names,
      decode_input_axis_names=decode_input_axis_names,
      prefill_out_axis_names=prefill_out_axis_names,
      decode_out_axis_names=decode_out_axis_names,
      prefill_cache_axis_order=prefill_cache_axis_order,
      ar_cache_axis_order=ar_cache_axis_order,
      compute_axis_order=compute_axis_order,
      reshape_q=reshape_q,
      is_nope_layer=is_nope_layer,
      is_vision=is_vision,
      model_mode=model_mode,
      q_lora_rank=q_lora_rank,
      kv_lora_rank=kv_lora_rank,
      qk_nope_head_dim=qk_nope_head_dim,
      qk_rope_head_dim=qk_rope_head_dim,
      v_head_dim=v_head_dim,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      mscale=mscale,
      rope_factor=rope_factor,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )


class MLA(Attention):
  \"\"\"Multi-Head Latent Attention (MLA) layer.\"\"\"

  def __init__(
      self,
      config: Config,
      num_query_heads: int,
      num_kv_heads: int,
      head_dim: int,
      max_target_length: int,
      mesh: Mesh,
      attention_kernel: str,
      inputs_q_shape: Tuple,
      inputs_kv_shape: Tuple,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      max_prefill_predict_length: int = -1,
      dropout_rate: float = 0.0,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      float32_qk_product: bool = False,  # computes logits in float32 for stability.
      float32_logits: bool = False,  # cast logits in float32 for stability.
      quant: Optional[Quant] = None,
      kv_quant: Optional[KVQuant] = None,
      attention_type: AttentionType = AttentionType.MLA,  # Default to MLA attention
      attn_logits_soft_cap: float | None = None,
      sliding_window_size: int | None = None,
      use_ragged_attention: bool = False,
      ragged_block_size: int = 256,
      use_qk_norm: bool = False,
      query_pre_attn_scalar: float | None = None,
      use_bias_in_projections: bool = False,  # Set to True will enable bias in q, k, v, o projections
      # Temperature tuning parameters used for Llama4
      temperature_tuning: bool = False,
      temperature_tuning_scale: float = 0.1,
      temperature_tuning_floor_scale: float = 8192.0,
      # Shard the query activation as the same as the key and value.
      prefill_query_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
      prefill_key_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
      prefill_value_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
      query_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM),
      key_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM),
      value_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM),
      input_axis_names: AxisNames = (BATCH_ATTN, LENGTH, EMBED),
      out_axis_names: AxisNames = (BATCH_ATTN, LENGTH, HEAD, D_KV),
      prefill_input_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, EMBED),
      decode_input_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, EMBED),
      prefill_out_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, HEAD, D_KV),
      decode_out_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV),
      prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
      ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
      compute_axis_order: AxisIdxes = (0, 1, 2, 3),
      reshape_q: bool = False,
      is_nope_layer: bool = False,
      is_vision: bool = False,
      model_mode: str = MODEL_MODE_TRAIN,
      q_lora_rank: int = 0,
      kv_lora_rank: int = 512,
      qk_nope_head_dim: int = 128,
      qk_rope_head_dim: int = 64,
      v_head_dim: int = 128,
      max_position_embeddings: int = 4096 * 4,
      original_max_position_embeddings: int = 4096,
      mscale: float = 1.0,  # scaling factor for softmax
      rope_factor: float = 40.0,  # rotary embedding factor
      name: str | None = None,
      rngs: Optional[nnx.Rngs] = None,
  ):
    \"\"\"Initializes the MLA module.

    Args:
      config: The model configuration.
      ... and other configuration parameters for MLA attention.
      rngs: The random number generators for initialization, passed by the nnx.to_linen wrapper.
    \"\"\"
    base_kv_cache = config.attention != "paged" and config.mla_naive_kvcache

    # Setting these before call to super because a field is used in super
    self.q_lora_rank = q_lora_rank
    self.kv_lora_rank = kv_lora_rank
    self.qk_nope_head_dim = qk_nope_head_dim
    self.qk_rope_head_dim = qk_rope_head_dim
    self.v_head_dim = v_head_dim
    self.max_position_embeddings = max_position_embeddings
    self.original_max_position_embeddings = original_max_position_embeddings
    self.mscale = mscale
    self.rope_factor = rope_factor

    self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

    super().__init__(
        config=config,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_target_length=max_target_length,
        mesh=mesh,
        attention_kernel=attention_kernel,
        inputs_q_shape=inputs_q_shape,
        inputs_kv_shape=inputs_kv_shape,
        dtype=dtype,
        weight_dtype=weight_dtype,
        max_prefill_predict_length=max_prefill_predict_length,
        dropout_rate=dropout_rate,
        kernel_init=kernel_init,
        float32_qk_product=float32_qk_product,
        float32_logits=float32_logits,
        quant=quant,
        kv_quant=kv_quant,
        attention_type=attention_type,
        attn_logits_soft_cap=attn_logits_soft_cap,
        sliding_window_size=sliding_window_size,
        use_ragged_attention=use_ragged_attention,
        ragged_block_size=ragged_block_size,
        use_qk_norm=use_qk_norm,
        query_pre_attn_scalar=query_pre_attn_scalar,
        use_bias_in_projections=use_bias_in_projections,
        temperature_tuning=temperature_tuning,
        temperature_tuning_scale=temperature_tuning_scale,
        temperature_tuning_floor_scale=temperature_tuning_floor_scale,
        prefill_query_axis_names=prefill_query_axis_names,
        prefill_key_axis_names=prefill_key_axis_names,
        prefill_value_axis_names=prefill_value_axis_names,
        query_axis_names=query_axis_names,
        key_axis_names=key_axis_names,
        value_axis_names=value_axis_names,
        input_axis_names=input_axis_names,
        out_axis_names=out_axis_names,
        prefill_input_axis_names=prefill_input_axis_names,
        decode_input_axis_names=decode_input_axis_names,
        prefill_out_axis_names=prefill_out_axis_names,
        decode_out_axis_names=decode_out_axis_names,
        prefill_cache_axis_order=prefill_cache_axis_order,
        ar_cache_axis_order=ar_cache_axis_order,
        compute_axis_order=compute_axis_order,
        reshape_q=reshape_q,
        is_nope_layer=is_nope_layer,
        is_vision=is_vision,
        model_mode=model_mode,
        base_kv_cache=base_kv_cache,
        rngs=rngs,
    )

    # Initialize Indexer
    self.use_indexer = config.use_indexer
    if self.use_indexer:
      # Need two versions of rope.
      # MLA applies yarn with interleave layout.
      # Indexer applies yarn with concatenate layout.
      indexer_rope = copy.copy(self.rotary_embedding)
      indexer_rope.interleave = False
      self.indexer = Indexer(
          config,
          rngs=rngs,
          rotary_embedding=indexer_rope,
          kernel_init=kernel_init,
          quant=quant,
          model_mode=model_mode,
      )
      self.IndexerKVCache_0 = self.init_indexer_cache(inputs_kv_shape) if model_mode != MODEL_MODE_TRAIN else None
    else:
      self.indexer = None
      self.IndexerKVCache_0 = None

    # Module attribute names must match names previously passed to Linen for checkpointing
    self.MlaKVCache_0 = self.init_mla_kv_caches(inputs_kv_shape) if model_mode != MODEL_MODE_TRAIN else None

  def init_indexer_cache(self, inputs_kv_shape: Tuple):
    \"\"\"Initializes Indexer Cache.\"\"\"
    batch_size, _, _ = inputs_kv_shape
    # Use standard KVCache to store keys. Values are unused but required by KVCache API.
    # KVCache expects key_heads and value_heads. Since k is shared (MQA-like for Indexer),
    # we use key_heads=1, value_heads=1.
    return kvcache.KVCache(
        max_prefill_length=self.max_prefill_predict_length,
        max_target_length=self.max_target_length,
        batch=batch_size,
        key_seq_len=PLACEHOLDER_SEQ_LEN,
        value_seq_len=PLACEHOLDER_SEQ_LEN,
        key_heads=1,
        value_heads=1,
        key_head_size=self.config.indexer_head_dim,
        value_head_size=self.config.indexer_head_dim,
        dtype=self.dtype,
        kv_quant=None,  # Quantization is not yet supported by the indexer.
        prefill_cache_logical_axis_names=(CACHE_BATCH_PREFILL, CACHE_SEQUENCE, CACHE_HEADS_NONE, CACHE_KV),
        cache_logical_axis_names=(CACHE_BATCH, CACHE_SEQUENCE, CACHE_HEADS_NONE, CACHE_KV),
        prefill_cache_axis_order=(1, 2, 0, 3),
        ar_cache_axis_order=(1, 2, 0, 3),
        use_chunked_prefill=self.config.use_chunked_prefill,
        model_mode=self.model_mode,
        rngs=self.rngs,
    )

  def _init_projections(self, inputs_q_shape: Tuple, inputs_kv_shape: Tuple) -> None:
    \"\"\"Initializes the MLA-specific projections.\"\"\"
    # Assert required configuration parameters for MLA attention.
    assert (
        self.config.attention_type == AttentionType.MLA.value
    ), f"MLA requires MLA attention type {AttentionType.MLA.value}"
    assert self.kv_lora_rank > 0, "KV LoRA rank must be > 0"
    assert self.qk_nope_head_dim > 0, "QK NoPe head dim must be > 0"
    assert self.qk_rope_head_dim > 0, "QK RoPE head dim must be > 0"
    assert self.v_head_dim > 0, "V head dim must be > 0"
    assert self.num_query_heads == self.num_kv_heads, "MLA requires equal number of query and kv heads"
    assert not self.config.fused_qkv, "Fused QKV is not supported for MLA"

    if self.q_lora_rank == 0:
      # Standard Q projection (without LoRA).
      self.query = DenseGeneral(
          in_features_shape=self.config.emb_dim,
          out_features_shape=(self.num_query_heads, self.qk_head_dim),
          axis=-1,
          kernel_init=self.kernel_init,
          kernel_axes=("embed", "q_heads", "kv"),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          quant=self.quant,
          matmul_precision=self.config.matmul_precision,
          shard_mode=self.config.shard_mode,
          rngs=self.rngs,
      )
    else:
      # LoRA path for Q.
      self.wq_a = DenseGeneral(
          in_features_shape=self.config.emb_dim,
          out_features_shape=self.q_lora_rank,
          axis=-1,
          kernel_init=self.kernel_init,
          kernel_axes=("embed", "q_lora_up_proj"),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          quant=self.quant,
          matmul_precision=self.config.matmul_precision,
          shard_mode=self.config.shard_mode,
          rngs=self.rngs,
      )
      self.q_norm = RMSNorm(
          num_features=self.q_lora_rank,
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          epsilon=self.config.normalization_layer_epsilon,
          kernel_axes=("norm",),
          rngs=self.rngs,
      )
      self.wq_b = DenseGeneral(
          in_features_shape=self.q_lora_rank,
          out_features_shape=(self.num_query_heads, self.qk_head_dim),
          axis=-1,
          kernel_init=self.kernel_init,
          kernel_axes=("q_lora", "q_heads", "kv"),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          quant=self.quant,
          matmul_precision=self.config.matmul_precision,
          shard_mode=self.config.shard_mode,
          rngs=self.rngs,
      )

    # KV LoRA path.
    self.wkv_a = DenseGeneral(
        in_features_shape=self.config.emb_dim,
        out_features_shape=self.kv_lora_rank + self.qk_rope_head_dim,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "kv_lora_up_proj"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )
    self.kv_norm = RMSNorm(
        num_features=self.kv_lora_rank,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        epsilon=self.config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.wkv_b = DenseGeneral(
        in_features_shape=self.kv_lora_rank,
        out_features_shape=(
            self.num_query_heads,
            (self.qk_nope_head_dim + self.v_head_dim),
        ),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("kv_lora", "kv_heads", "kv_head_dim"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

    # Set softmax scaling.
    self.softmax_scale = self.qk_head_dim**-0.5
    if self.max_position_embeddings > self.original_max_position_embeddings:
      mscale = 0.1 * self.mscale * math.log(self.rope_factor) + 1.0
      self.softmax_scale = self.softmax_scale * mscale * mscale

    self.out = self.init_out_w(output_dim=inputs_q_shape[-1])

    # Setup paged attention op
    if self.config.attention == "paged":
      # Set head_dim to the max of qk_head_dim and v_head_dim. The current paged
      # attention kernel requires the head_dim to be the same for q, k, v.
      head_dim = max(self.qk_head_dim, self.v_head_dim)
      # Align head_dim to the pagedattn_head_dim_alignment if specified.
      if self.config.pagedattn_head_dim_alignment > 0:
        alignment = self.config.pagedattn_head_dim_alignment
        head_dim = (head_dim + alignment - 1) // alignment * alignment
      self.ds_paged_attention_op = paged_attention.PagedAttentionOp(
          mesh=self.mesh,
          num_pages=self.config.pagedattn_num_pages,
          tokens_per_page=self.config.pagedattn_tokens_per_page,
          max_pages_per_slot=(self.config.max_target_length + self.config.pagedattn_tokens_per_page - 1)
          // self.config.pagedattn_tokens_per_page,
          max_pages_per_prefill=(self.config.max_prefill_predict_length + self.config.pagedattn_tokens_per_page - 1)
          // self.config.pagedattn_tokens_per_page,
          pages_per_compute_block=self.config.pagedattn_pages_per_compute_block,
          num_kv_heads=self.num_kv_heads,
          kv_head_dim_size=head_dim,
          dtype=self.dtype,
          attn_logits_soft_cap=self.attn_logits_soft_cap,
          rngs=self.rngs,
      )

  @property
  def out_head_dim(self) -> int:
    return self.v_head_dim

  def mla_query_projection(
      self, inputs_q: Array, inputs_positions: Array, model_mode
  ) -> tuple[jax.Array, Optional[jax.Array]]:
    \"\"\"Query projection for MLA, e.g. includes LoRA if q_lora_rank > 0.\"\"\"
    # specify query logical name
    if model_mode == MODEL_MODE_PREFILL:
      query_logical_name = self.prefill_query_axis_names
      wqa_logical_name = (PREFILL_KV_BATCH, PREFILL_LENGTH, Q_LORA_UP_PROJ)
    else:
      query_logical_name = self.query_axis_names
      wqa_logical_name = (KV_BATCH, LENGTH, Q_LORA_UP_PROJ)
    query_sharding = create_sharding(self.mesh, query_logical_name)
    wqa_out_sharding = create_sharding(self.mesh, wqa_logical_name)
    # Set softmax scaling.
    self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
    self.softmax_scale = self.qk_head_dim**-0.5
    if self.max_position_embeddings > self.original_max_position_embeddings:
      mscale = 0.1 * self.mscale * math.log(self.rope_factor) + 1.0
      self.softmax_scale = self.softmax_scale * mscale * mscale

    # Low-rank latent vector for queries. This is also accessed by indexer.
    low_rank_q = None

    if self.q_lora_rank == 0:
      q = self.query(inputs_q, out_sharding=query_sharding)
    else:
      # LoRA path
      low_rank_q = self.wq_a(inputs_q, out_sharding=wqa_out_sharding)  # [B, L, q_lora_rank]
      low_rank_q = checkpoint_name(low_rank_q, "query_wa_proj")
      low_rank_q = self.q_norm(low_rank_q)  # RMSNorm on low rank
      low_rank_q = checkpoint_name(low_rank_q, "mla_q")
      q = self.wq_b(low_rank_q, out_sharding=query_sharding)  # [B, L, n_heads, qk_head_dim]

    # Partial RoPE: Split into non-positional and rotary parts.
    # last dimension: qk_nope_head_dim, qk_rope_head_dim
    q_nope, q_pe = jnp.split(q, [self.qk_nope_head_dim], axis=-1)
    q_nope = self._maybe_shard_with_logical(q_nope, query_logical_name)
    q_pe = self.apply_rotary_embedding(q_pe, inputs_positions=inputs_positions)
    q_pe = self._maybe_shard_with_logical(q_pe, query_logical_name)
    # Query projection is scaled by self.softmax_scale to be consistent MaxText implementation.
    # DeepSeek v3 was doing it in attention score computation.
    query = jnp.concatenate([q_nope, q_pe], axis=-1) * self.softmax_scale
    query = self._maybe_shard_with_logical(query, query_logical_name)
    return query, low_rank_q

  def mla_get_key_value(self, low_rank_main, key_rope, model_mode):
    \"\"\"get (key,value) pair from mla\"\"\"
    if model_mode == MODEL_MODE_PREFILL:
      key_logical_name = self.prefill_key_axis_names
      value_logical_name = self.prefill_value_axis_names
    else:
      key_logical_name = self.key_axis_names
      value_logical_name = self.value_axis_names

    wkva_out_sharding = create_sharding(self.mesh, key_logical_name)
    kv_out = self.wkv_b(low_rank_main, out_sharding=wkva_out_sharding)

    # Split kv_out into key_nope and value parts.
    key_nope, value = jnp.split(kv_out, [self.qk_nope_head_dim], axis=-1)
    key_rope = jnp.broadcast_to(key_rope, (key_nope.shape[0], key_nope.shape[1], self.num_query_heads, key_rope.shape[3]))
    key_nope = self._maybe_shard_with_logical(key_nope, key_logical_name)
    key_rope = self._maybe_shard_with_logical(key_rope, key_logical_name)

    key = jnp.concatenate([key_nope, key_rope], axis=-1)

    key = self._maybe_shard_with_logical(key, key_logical_name)
    value = self._maybe_shard_with_logical(value, value_logical_name)
    return key, value

  def init_mla_kv_caches(self, inputs_kv_shape: Tuple):
    \"\"\"Initializes MlaKVCache.

    Args:
      inputs_kv_shape: Key/value inputs shape for initialization.

    Returns:
      An MlaKVCache module instance.

    Raises:
      ValueError: If the configuration is invalid.

    \"\"\"
    batch_size, _, _ = inputs_kv_shape
    # During initialization, seq_len of inputs_kv is max_target_length,
    # which is not always correct for some functions in MlaKVCache.
    # However, MlaKVCache internal cache shapes are based on max_prefill_length
    # and max_target_length, not the passed seq_len.
    # We can use a placeholder value. The correct fix might involve refactoring
    # MlaKVCache.

    return kvcache.MlaKVCache(
        max_prefill_length=self.max_prefill_predict_length,
        max_target_length=self.max_target_length,
        batch=batch_size,
        key_seq_len=PLACEHOLDER_SEQ_LEN,
        value_seq_len=PLACEHOLDER_SEQ_LEN,
        key_head_size=self.kv_lora_rank,
        value_head_size=self.qk_rope_head_dim,
        dtype=self.dtype,
        kv_quant=self.kv_quant,
        prefill_cache_axis_order=self.prefill_cache_axis_order,
        ar_cache_axis_order=self.ar_cache_axis_order,
        model_mode=self.model_mode,
        use_chunked_prefill=self.config.use_chunked_prefill,
        rngs=self.rngs,
    )

  def update_mla_kv_caches(self, low_rank_main, key_rope, decoder_segment_ids, model_mode, previous_chunk=None):
    \"\"\"Updates the MLA (Multi-Head Latent Attention) KV caches.

    This method is specific to the MLA attention mechanism. It calls the
    `mla_kv_cache_as_linen` module to update and retrieve the caches, which
    store latent representations (`low_rank_main`) and RoPE-applied keys
    (`key_rope`). It then reconstructs the full key and value tensors from
    the cached components.

    Args:
      low_rank_main: The main latent component of the key.
      key_rope: The RoPE-applied component of the key.
      decoder_segment_ids: Segment IDs for decoder masking.
      model_mode: The operational mode ('train', 'prefill', 'autoregressive').
      previous_chunk: Information about previously processed chunks, for
        chunked prefill.

    Returns:
      A list containing two elements:
      - The prefill key-value cache, reconstructed from the MLA cache, or None.
      - The autoregressive key-value cache, reconstructed from the MLA cache, or None.
    \"\"\"

    prefill_mla_cache, ar_mla_cache = self.MlaKVCache_0(
        key_latent=low_rank_main,
        key_rope=key_rope,
        decoder_segment_ids=decoder_segment_ids,
        model_mode=model_mode,
        use_ragged_attention=self.use_ragged_attention,
        previous_chunk=previous_chunk,
    )

    if prefill_mla_cache:
      low_rank_main, key_rope, decoder_segment_ids = prefill_mla_cache
      key, value = self.mla_get_key_value(low_rank_main, key_rope, model_mode)
      prefill_kv_cache = key, value, decoder_segment_ids
    else:
      prefill_kv_cache = None

    if ar_mla_cache:
      low_rank_main, key_rope, decoder_segment_ids, lengths = ar_mla_cache
      key, value = self.mla_get_key_value(low_rank_main, key_rope, model_mode)
      ar_kv_cache = key, value, decoder_segment_ids, lengths
    else:
      ar_kv_cache = None
    return [prefill_kv_cache, ar_kv_cache]

  def mla_kv_projection(self, inputs: Array, inputs_positions: Array, decoder_segment_ids, model_mode, previous_chunk):
    \"\"\"MLA key/value projection with integrated rotary embedding.\"\"\"
    if model_mode == MODEL_MODE_PREFILL:
      wka_logical_name = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_LORA_UP_PROJ)
    else:
      wka_logical_name = (KV_BATCH, LENGTH, KV_LORA_UP_PROJ)
    wkva_out_sharding = create_sharding(self.mesh, wka_logical_name)
    low_rank = self.wkv_a(inputs, out_sharding=wkva_out_sharding)
    low_rank = checkpoint_name(low_rank, "kv_wa_proj")
    low_rank_main, low_rank_rope = jnp.split(low_rank, [self.kv_lora_rank], axis=-1)
    low_rank_main = self.kv_norm(low_rank_main)
    low_rank_main = checkpoint_name(low_rank_main, "mla_kv")
    # Apply rotary embedding to key_rope.
    key_rope = jnp.expand_dims(low_rank_rope, axis=2)
    key_rope = self.apply_rotary_embedding(key_rope, inputs_positions=inputs_positions)

    key, value = self.mla_get_key_value(low_rank_main, key_rope, model_mode)
    cached_values = [None, None]
    if self.config.attention != "paged" and model_mode != MODEL_MODE_TRAIN:
      if self.config.mla_naive_kvcache:
        cached_values = self.update_kv_caches(key, value, decoder_segment_ids, model_mode, previous_chunk)
      else:
        cached_values = self.update_mla_kv_caches(
            low_rank_main, key_rope, decoder_segment_ids, model_mode, previous_chunk
        )

    return key, value, cached_values

  def calculate_indexer_loss(
      self,
      indexer_score: Array,
      query: Array,
      key: Array,
      attention_mask: Optional[Array | None],
      indexer_mask: Array,
      sparse_loss: bool,
      scaling_factor: float,
  ) -> Array:
    \"\"\"Calculates the indexer KL divergence loss.

    This loss trains the indexer to predict which tokens are important by matching
    the distribution of true attention scores from the main model.

    The target distribution is derived through the following steps:
    1. Compute raw attention scores via Q @ K^T.
    2. Aggregate scores by summing across all attention heads.
    3. Apply L1-normalization across the sequence dimension.

    target_distribution = L1_Normalize(Sum_h(Softmax(Q @ K^T)))

    Reference:
    DeepSeek-V3.2 - https://arxiv.org/pdf/2512.02556

    Args:
      indexer_score: Scores predicted by indexer [batch, q_len, kv_len].
      query: Query tensor from main model [batch, q_len, heads, dim].
      key: Key tensor from main model [batch, kv_len, heads, dim].
      attention_mask: Attention mask [batch, q_len, kv_len] or None.
      indexer_mask: Indexer mask [batch, q_len, kv_len].
      sparse_loss: Whether to use sparse loss.
      scaling_factor: The scaling factor for the loss.

    Returns:
      The computed KL divergence loss.
    \"\"\"
    # Detach main model components from the computational graph.
    # The indexer should match the main model, but the main model should not be influenced
    # by the indexer's learning progress via this loss in sparse training stage.
    # We also apply this during the Dense Warm-up stage to save compute and memory.
    query = jax.lax.stop_gradient(query)
    key = jax.lax.stop_gradient(key)

    # Compute attention scores: [b, t, h, d] @ [b, s, h, d] -> [b, h, t, s]
    attention_scores = jnp.einsum("bthd, bshd -> bhts", query, key, precision=self.config.matmul_precision)

    if sparse_loss:
      # indexer_mask is already pre-filtered with the attention_mask if any
      attention_scores = attention_scores + indexer_mask[:, None, :, :]
      indexer_score = indexer_score + indexer_mask
    elif attention_mask is not None:
      # indexer_score already applies attention_mask; updating attention_scores only
      attention_scores = attention_scores + attention_mask[:, None, :, :]

    # Use float32 for softmax numerical stability.
    attention_probs = jax.nn.softmax(attention_scores.astype(jnp.float32), axis=-1)
    indexer_probs = jax.nn.softmax(indexer_score.astype(jnp.float32), axis=-1)

    # Aggregate heads: [b, h, t, s] -> [b, t, s]
    attention_probs = jnp.sum(attention_probs, axis=1)
    # L1 normalize aggregated target distribution
    attention_probs = attention_probs / (jnp.sum(attention_probs, axis=-1, keepdims=True) + EPS)

    # KL Divergence: KL(attention || indexer)
    log_attention_probs = jnp.log(attention_probs + EPS)
    log_indexer_probs = jnp.log(indexer_probs + EPS)
    kl_per_token = attention_probs * (log_attention_probs - log_indexer_probs)
    indexer_loss = jnp.mean(jnp.sum(kl_per_token, axis=-1))

    return indexer_loss * scaling_factor

  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      inputs_positions: Array | None = None,
      decoder_segment_ids: Array | None = None,
      out_sharding: NamedSharding | None = None,
      *,
      model_mode: str = MODEL_MODE_TRAIN,
      deterministic: bool = False,
      previous_chunk: Any = None,
      slot: Optional[int] = None,
      page_state: Optional[page_manager.PageState] = None,
      bidirectional_mask: Optional[Any] = None,
      rope_kwargs: dict | None = None,
      kv_cache: Optional[Array] = None,
      attention_metadata: Optional[dict[str, Any]] = None,
  ) -> tuple[Array, Optional[Array]]:
    \"\"\"Forward pass for MLA, reusing `AttentionOp` for the actual attention.

    Args:
      inputs_q: Query input [batch, q_length, embed_dim].
      inputs_kv: KV input   [batch, kv_length, embed_dim].
      inputs_positions: Positions for rotary embeddings or similar.
      decoder_segment_ids: Segment IDs for masking, if any.
      model_mode: "train", "prefill", or "autoregressive".
      deterministic: Disables dropout if set to True.
      previous_chunk: Information about previously processed chunks for chunked prefill.
      slot: The batch slot index for paged attention.
      page_state: The current state of the paged attention manager.
      bidirectional_mask: A mask for bidirectional attention, used in multimodal models.
      kv_cache: Optional key-value cache used when serving models with vLLM.
      attention_metadata: Optional attention-related metadata used when serving models with vLLM.

    Returns:
      A tensor of shape [batch, length, embed_dim] containing the
      MLA-attended outputs.
    \"\"\"
    if model_mode == MODEL_MODE_PREFILL:
      inputs_q = self._maybe_shard_with_logical(inputs_q, self.prefill_input_axis_names)
      inputs_kv = self._maybe_shard_with_logical(inputs_kv, self.prefill_input_axis_names)
      out_logical_name = (PREFILL_KV_BATCH, PREFILL_LENGTH, HEAD, D_KV)
    else:
      inputs_q = self._maybe_shard_with_logical(inputs_q, self.input_axis_names)
      inputs_kv = self._maybe_shard_with_logical(inputs_kv, self.input_axis_names)
      out_logical_name = (BATCH_ATTN, LENGTH, HEAD, D_KV)

    if model_mode != MODEL_MODE_TRAIN and decoder_segment_ids is None:
      decoder_segment_ids = jnp.ones(inputs_q.shape[:2], dtype=jnp.int32)

    query, low_rank_q = self.mla_query_projection(inputs_q, inputs_positions, model_mode)
    if self.config.force_q_layout:
      query = layout.with_layout_constraint(query, DLL(major_to_minor=(0, 2, 3, 1)))
    key, value, cached_values = self.mla_kv_projection(
        inputs_kv, inputs_positions, decoder_segment_ids, model_mode, previous_chunk
    )
    query = checkpoint_name(query, "query_proj")
    key = checkpoint_name(key, "key_proj")
    value = checkpoint_name(value, "value_proj")

    # Indexer Logic
    indexer_mask = None
    if self.use_indexer:
      # generate mask: with 0 and large negative, [b, 1, 1, q_len, kv_len] -> [b, q_len, kv_len]
      attention_mask = self.attention_op.generate_attention_mask(
          query, key, decoder_segment_ids, model_mode, previous_chunk, bidirectional_mask
      )
      if attention_mask is not None:
        attention_mask = attention_mask.squeeze(axis=(1, 2))
      # apply indexer, indexer_mask [b, q_len, kv_len]
      indexer_mask, _, indexer_score = self.indexer(
          inputs_q=inputs_q,
          low_rank_q=low_rank_q,
          inputs_kv=inputs_kv,
          inputs_positions=inputs_positions,
          attention_mask=attention_mask,
          decoder_segment_ids=decoder_segment_ids,
          previous_chunk=previous_chunk,
          kv_cache=self.IndexerKVCache_0,
          model_mode=model_mode,
      )

      if indexer_mask is not None and self.config.indexer_loss_scaling_factor > 0.0:
        indexer_loss = self.calculate_indexer_loss(
            indexer_score=indexer_score,
            query=query,
            key=key,
            attention_mask=attention_mask,
            indexer_mask=indexer_mask,
            sparse_loss=self.config.indexer_sparse_training,
            scaling_factor=self.config.indexer_loss_scaling_factor,
        )
        self.indexer_loss = nnx.Intermediate(indexer_loss)

    # Check if we need QK Clip stats
    use_qk_clip = self.model_mode == MODEL_MODE_TRAIN and self.config.use_qk_clip

    if self.config.attention == "paged" and model_mode != MODEL_MODE_TRAIN:
      unnormalized_out, _, exp_sum = self.ds_paged_attention_op(
          query, key, value, decoder_segment_ids, model_mode, previous_chunk, slot=slot, page_state=page_state
      )
      unnormalized_out = unnormalized_out[..., : self.v_head_dim]
      out = unnormalized_out / (exp_sum + 1e-9) if exp_sum is not None else unnormalized_out
    else:
      out = self.attention_op(
          query,
          key,
          value,
          decoder_segment_ids,
          inputs_positions,
          model_mode,
          cached_values,
          indexer_mask=indexer_mask,
          record_max_logits=use_qk_clip,
      )

    out = self._maybe_shard_with_logical(out, self.out_axis_names)
    out = jax.ad_checkpoint.checkpoint_name(out, "attention_out")

    out_sharding = create_sharding(self.mesh, out_logical_name)
    out = self.out_projection(out, out_sharding=out_sharding)
    out = checkpoint_name(out, "out_proj")
    return out, kv_cache
\n"""


# File: src/maxtext/layers/moe.py (commit 313890777)
MOE_RAW = """\n# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


\"\"\"MoE related Layers.\"\"\"

import enum
import functools
import math
import random
from typing import Iterable, Optional, Tuple, Union

from aqt.jax.v2 import aqt_tensor as aqt
from flax import nnx
import jax
from jax import ad_checkpoint as adc
from jax.experimental import xla_metadata
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from maxtext.common import common_types as ctypes
from maxtext.common.common_types import ShardMode
from maxtext.kernels import megablox as mblx
from maxtext.layers import attentions, linears, nnx_wrappers, quantizations
from maxtext.layers.initializers import NdInitializer, default_bias_init, nd_dense_init, variable_to_logically_partitioned
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils.sharding import create_sharding, maybe_shard_with_logical, maybe_shard_with_pspec
from maxtext.utils.sharding import logical_to_mesh_axes
import numpy as np
import qwix
from qwix.contrib.sparsity import sparsity_module
import qwix.pallas as qpl
import tokamax

set_xla_metadata = xla_metadata.set_xla_metadata


DISPATCH = "dispatch"
COMBINE = "combine"


def _sort_activations(
    inputs: jax.Array,
    sort_indices: jax.Array,
    use_custom_vjp: bool,
) -> jax.Array:
  \"\"\"Sort activations by `sort_indices`.

  If `use_custom_vjp=True`, then we use a custom backward pass that
  reverses the sort order. Specifically, this unsort operation is simply a sort
  with `jnp.argsort(sort_indices)` as the sort indices. This is only needed in
  the case where the compiler generates a less efficient backward pass op.

  Note that `use_custom_vjp=True` assumes that `sort_indices` is a permutation
  of `jnp.arange(inputs.shape[0])`.

  Args:
    inputs: `(tokens, ...)`-shaped array of input activations to sort.
    sort_indices: `(tokens,)`-shaped array containing the sort order.
    use_custom_vjp: Whether to use the explicit backward pass.

  Returns:
    `(tokens, ...)`-shaped array of input activations sorted by `sort_indices`.
  \"\"\"
  assert inputs.shape[0] == sort_indices.shape[0]

  with jax.named_scope("sort_activations"):
    if use_custom_vjp:
      return _sort_activations_custom(inputs, sort_indices)
    return inputs[sort_indices, ...]


@jax.custom_vjp
def _sort_activations_custom(inputs: jax.Array, sort_indices: jax.Array) -> jax.Array:
  \"\"\"Sort functions with custom vjp.\"\"\"
  return inputs[sort_indices, ...]


def _sort_activations_custom_fwd(inputs: jax.Array, sort_indices: jax.Array) -> tuple[jax.Array, jax.Array]:
  \"\"\"Forward pass of the custom vjp for `_sort_activations()`.\"\"\"
  return _sort_activations_custom(inputs, sort_indices), sort_indices


def _sort_activations_custom_bwd(residuals: jax.Array, grads: jax.Array) -> tuple[jax.Array, None]:
  \"\"\"Backward pass of the custom vjp for `_sort_activations()`.\"\"\"
  sort_indices = residuals
  return _sort_activations_custom(grads, jnp.argsort(sort_indices)), None


_sort_activations_custom.defvjp(_sort_activations_custom_fwd, _sort_activations_custom_bwd)


def get_batchsplit_init_kernel_axes():
  return (
      ("embed_moe", None, "expert_only"),
      ("embed_moe", "expert_only", None),
  )


def random_routing(rng_key, gate_logits, num_experts_per_tok):
  \"\"\"Performs random routing of tokens to experts.

  Args:
    rng_key: A JAX PRNGKey for randomness.
    gate_logits: A JAX array of shape (batch_size, sequence_length, num_experts)
      representing the logits for each expert.
    num_experts_per_tok: The number of experts to select for each token.

  Returns:
    A tuple containing:
      - top_k_indices: JAX array of shape (batch_size, sequence_length,
      num_experts_per_tok)
                       representing the indices of the selected experts for each
                       token.
      - top_k_weights: JAX array of shape (batch_size, sequence_length,
      num_experts_per_tok)
                       representing the weights for the selected experts.
  \"\"\"
  bs, seq_len, num_experts = gate_logits.shape
  selected_num = bs * seq_len * num_experts_per_tok
  # Directly generate random integers in the range [0, num_experts)
  top_k_indices = jax.random.randint(
      rng_key,
      shape=(selected_num,),
      minval=0,
      maxval=num_experts,
      dtype=jnp.int32,
  )
  top_k_indices = top_k_indices.reshape(bs, seq_len, num_experts_per_tok)
  top_k_weights = jnp.take_along_axis(gate_logits, top_k_indices, axis=-1)
  return top_k_weights, top_k_indices


def calculate_load_balance_updates(top_k_indices, num_experts, rate):
  \"\"\"
  Computes a bias adjustment update based on expert load.
  Used in DeepSeek V3: https://arxiv.org/html/2412.19437v1.
  Implementation reference: https://arxiv.org/pdf/2408.15664.

  Args:
      top_k_indices: Shape (batch, sequence, top_k).
      num_experts: Total number of experts.
      rate: The update rate.

  Returns:
      update: The value to add to the expert bias. Shape (num_experts,).
  \"\"\"
  flat_indices = top_k_indices.ravel()
  expert_counts = jnp.bincount(flat_indices, length=num_experts)

  total_tokens = flat_indices.size
  average_load = total_tokens / num_experts
  direction = jnp.sign(average_load - expert_counts)
  output = direction * rate
  return output


class GateLogit(nnx.Module):
  \"\"\"A layer used to compute gate logits, allowing to return the pre bias values for DeepSeek routing.\"\"\"

  def __init__(
      self,
      in_features_shape: Union[Iterable[int], int],
      out_features_shape: Union[Iterable[int], int],
      model_name: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      axis: Union[Iterable[int], int] = -1,
      weight_dtype: ctypes.DType = jnp.float32,
      dtype: ctypes.DType = jnp.float32,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes: Tuple[Optional[str], ...] = (),
      use_bias: bool = False,
      score_func: str = "",
      quant: Optional[quantizations.AqtQuantization] = None,
      shard_mode: ShardMode = ShardMode.AUTO,
      matmul_precision: str = "default",
  ):
    \"\"\"Initializes the GateLogit module.

    Attributes:
      in_features_shape: The shape of the input features.
      out_features_shape: The shape of the output features, typically the number of experts.
      model_name: The name of the model.
      rngs: An `nnx.Rngs` object used for initializing parameters.
      axis: The axis or axes over transformation is applied.
      weight_dtype: The data type of the kernel weights.
      dtype: The data type for the computation.
      kernel_init: The initializer function for the kernel weight matrix.
      kernel_axes: A tuple of logical axis names for partitioning the kernel.
      use_bias: Whether to add learnable bias in gate logit scores. When enabled,
        this bias aids expert load balancing (like in DeepSeek V3), and is not
        part of the loss calculation.
      score_func: Scoring function for output normalization before applying bias.
      quant: The quantization configuration. If None, no quantization is applied.
      matmul_precision: The precision level for the matrix multiplication.
    \"\"\"
    self.in_features_shape = linears.canonicalize_tuple(in_features_shape)
    self.out_features_shape = linears.canonicalize_tuple(out_features_shape)
    self.model_name = model_name
    self.mesh = mesh
    self.axis = linears.canonicalize_tuple(axis)
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.use_bias = use_bias
    self.score_func = score_func
    self.quant = quant
    self.shard_mode = shard_mode
    self.matmul_precision = matmul_precision

    # Parameter initialization
    kernel_shape = self.in_features_shape + self.out_features_shape
    kernel_in_axis = np.arange(len(self.axis))
    kernel_out_axis = np.arange(len(self.axis), len(self.axis) + len(self.out_features_shape))

    if not quantizations.in_serve_mode(self.quant):
      self.kernel = nnx.Param(
          self.kernel_init(
              rngs.params(),
              kernel_shape,
              self.weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          out_sharding=self.kernel_axes,
      )

    if self.use_bias:
      bias_axes = self.kernel_axes[-len(self.out_features_shape) :]
      bias_shape = kernel_shape[-len(self.out_features_shape) :]
      self.bias = nnx.Param(
          default_bias_init(rngs.params(), bias_shape, self.weight_dtype),
          out_sharding=bias_axes,
      )
    else:
      self.bias = None

    if quant:
      dot_general_cls = quant.dot_general_cls(mesh_axes=kernel_axes)
      dot_general_linen = dot_general_cls()
      quant_dot_general = nnx_wrappers.ToNNX(dot_general_linen, rngs=rngs)
      self._quant_dot_general_name = f"{type(dot_general_linen).__name__}_0"
      setattr(self, self._quant_dot_general_name, quant_dot_general)
      dummy_inputs = jnp.zeros((1, *self.in_features_shape), dtype=self.dtype)
      self(dummy_inputs, _initializing=True)
    else:
      self._quant_dot_general_name = None

  @property
  def quant_dot_general(self) -> nnx_wrappers.ToNNX | None:
    if self._quant_dot_general_name is None:
      return None
    return getattr(self, self._quant_dot_general_name)

  def __call__(self, inputs: jax.Array, _initializing: bool = False) -> Tuple[jax.Array, Optional[jax.Array]]:
    inputs = jnp.asarray(inputs, self.dtype)
    norm_axis = linears.normalize_axes(self.axis, inputs.ndim)

    if quantizations.in_serve_mode(self.quant):
      kernel_shape = self.in_features_shape + self.out_features_shape
      kernel = jnp.zeros(kernel_shape, dtype=self.dtype)
    else:
      kernel = self.kernel[...]
    kernel = jnp.asarray(kernel, self.dtype)

    contract_ind = tuple(range(0, len(norm_axis)))
    output_sharding = (
        create_sharding(self.mesh, ("activation_batch", "activation_length", None))
        if self.shard_mode == ShardMode.EXPLICIT
        else None
    )
    output = linears._compute_dot_general_nnx(
        inputs,
        kernel,
        norm_axis,
        contract_ind,
        self.matmul_precision,
        self.quant_dot_general,
        _initializing,
        out_sharding=output_sharding,
    )
    pre_bias_logits = None

    if self.score_func:
      output = linears._convert_to_activation_function(self.score_func)(output)
      if self.model_name.startswith("deepseek3"):
        pre_bias_logits = output

    if self.use_bias:
      bias = jnp.asarray(self.bias[...], self.dtype)
      output += bias
    return output, pre_bias_logits


class RoutedMoE(nnx.Module):
  \"\"\"Implements a routed MoE block.\"\"\"

  def __init__(
      self,
      config: ctypes.Config,
      num_experts: int,
      num_experts_per_tok: int,
      mesh: jax.sharding.Mesh,
      kernel_init: attentions.NdInitializer,
      kernel_axes: Tuple[Optional[str], ...],
      rngs: nnx.Rngs,
      intermediate_dim: int = 2048,
      weight_dtype: ctypes.DType = jnp.float32,
      dtype: ctypes.DType = jnp.float32,
      quant: Optional[quantizations.AqtQuantization] = None,
  ):
    \"\"\"Initializes the RoutedMoE module.

    Attributes:
      config: The main config setting.
      num_experts: Number of experts.
      num_experts_per_tok: Number of experts for each token.
      mesh: Mesh, device mesh.
      kernel_init: The initializer function for the kernel weight matrix.
      kernel_axes: A tuple of logical axis names for partitioning the kernel.
      rngs: An `nnx.Rngs` object used for initializing parameters.
      intermediate_dim: Intermediate dimension of MoE.
      weight_dtype: The data type of the kernel weights.
      dtype: The data type for the computation.
      quant: The quantization configuration. If None, no quantization is applied.
    \"\"\"
    self.config = config
    self.num_experts = num_experts
    self.num_experts_per_tok = num_experts_per_tok
    self.mesh = mesh
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.intermediate_dim = intermediate_dim
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.quant = quant
    self.rngs = rngs

    self.moe_expert_input_dim = (
        self.config.emb_dim if self.config.moe_expert_input_dim <= 0 else self.config.moe_expert_input_dim
    )

    if self.config.shard_exp_on_fsdp:
      # special sharding for dsv3
      self.wi_kernel_axes = ("embed_moe", None, "mlp_moe")
      self.wo_kernel_axes = ("embed_moe", "mlp_moe", None)
    elif self.config.use_2d_fsdp_sharding:
      self.wi_kernel_axes = ("embed_moe", "mlp_moe", None)
      self.wo_kernel_axes = ("embed_moe", "mlp_moe", None)
    elif self.config.use_batch_split_schedule:
      self.wi_kernel_axes, self.wo_kernel_axes = get_batchsplit_init_kernel_axes()
    else:
      self.wi_kernel_axes = ("exp", "embed_moe", "mlp_moe")
      self.wo_kernel_axes = ("exp", "mlp_moe", "embed_moe")

    if self.config.attention == "vllm_rpa":
      # vLLM uses 'model' as the tensor parallelism axis name
      self._tensor_parallelism_name = ("model", "attn_dp")
    else:
      self._tensor_parallelism_name = "tensor"

    if self.config.attention == "vllm_rpa" and self.config.enable_dp_attention:
      self._expert_parallelism_name = "attn_dp_expert"
    elif self.config.custom_mesh_and_rule == ctypes.CustomRule.CP_AS_EP:
      # when custom mesh and rule is cp-as-ep, context axis is same with expert in MoE component
      self._expert_parallelism_name = ("context", "expert")
    else:
      self._expert_parallelism_name = "expert"

    self.gate = GateLogit(
        in_features_shape=self.moe_expert_input_dim,
        out_features_shape=self.num_experts,
        mesh=self.mesh,
        model_name=self.config.model_name,
        dtype=jnp.float32 if self.config.float32_gate_logits else self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        kernel_init=self.kernel_init,
        kernel_axes=self.kernel_axes,
        use_bias=self.config.routed_bias,
        # tpu-inference applies the score function in the fused_moe_gmm kernel,
        # so we don't apply it here to avoid redundant computation.
        # See https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/layers/common/fused_moe_gmm.py#L58.
        score_func="" if self.config.attention == "vllm_rpa" else self.config.routed_score_func,
        matmul_precision=self.config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=self.rngs,
    )
    rule = qpl.get_current_rule("gmm")
    sparsity_rule = None
    if rule is not None:
      if not isinstance(rule, qwix.QtRule):
        raise ValueError("Expect a QtRule for quantized training.")
      if rule.additional_qt_config and "sparsity_rule" in rule.additional_qt_config:
        q_s_rule = rule.additional_qt_config["sparsity_rule"]
        if q_s_rule and q_s_rule.weight_sparsity_n and q_s_rule.weight_sparsity_m:
          sparsity_rule = q_s_rule

    if sparsity_rule is not None:
      self.wi_0_sparsity_module = sparsity_module.SparsityModule(
          shape=(self.num_experts, self.config.emb_dim, self.intermediate_dim),
          sharding_axes=self.wi_kernel_axes,
          sparsity_rule=sparsity_rule,
      )
      self.wi_1_sparsity_module = sparsity_module.SparsityModule(
          shape=(self.num_experts, self.config.emb_dim, self.intermediate_dim),
          sharding_axes=self.wi_kernel_axes,
          sparsity_rule=sparsity_rule,
      )
      self.wo_sparsity_module = sparsity_module.SparsityModule(
          shape=(self.num_experts, self.intermediate_dim, self.config.emb_dim),
          sharding_axes=self.wo_kernel_axes,
          sparsity_rule=sparsity_rule,
      )
    else:
      self.wi_0_sparsity_module = None
      self.wi_1_sparsity_module = None
      self.wo_sparsity_module = None

    # pylint: disable=protected-access
    self.activation_fn = linears._convert_to_activation_function(self.config.mlp_activations[0])

    kernel_in_axis = np.arange(1)
    kernel_out_axis = np.arange(1, 2)

    if quantizations.in_serve_mode(self.quant):
      # During aqt convert state we delete kernel weight from params to save
      # memory. Instead they are retrieved from the tensors stored in the 'aqt'
      # collection.
      self.wi_0 = jnp.zeros((num_experts, self.moe_expert_input_dim, intermediate_dim))
      self.wi_1 = jnp.zeros((num_experts, self.moe_expert_input_dim, intermediate_dim))
      self.wo = jnp.zeros((num_experts, intermediate_dim, self.moe_expert_input_dim))
    elif self.config.prefuse_moe_weights and self.config.attention == "vllm_rpa":
      # Pad model dimension in Fused MoE weight kernels for GMM_v2 execution.
      moe_intermediate_dim = (
          self.config.padded_base_moe_mlp_dim
          if self.config.padded_base_moe_mlp_dim is not None
          else self.intermediate_dim
      )
      self.wi = nnx.Param(
          self.kernel_init(
              self.rngs.params(),
              (num_experts, self.moe_expert_input_dim, moe_intermediate_dim * 2),
              weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          out_sharding=self.wi_kernel_axes,
      )
      self.wo = nnx.Param(
          self.kernel_init(
              self.rngs.params(),
              (self.num_experts, self.intermediate_dim, self.moe_expert_input_dim),
              self.weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          out_sharding=self.wo_kernel_axes,
      )
    else:
      # Pad model dimension in Unfused MoE weight kernels for GMM_v2 execution.
      moe_intermediate_dim = (
          self.config.padded_base_moe_mlp_dim
          if self.config.padded_base_moe_mlp_dim is not None
          else self.intermediate_dim
      )
      self.wi_0 = nnx.Param(
          self.kernel_init(
              self.rngs.params(),
              (num_experts, self.moe_expert_input_dim, moe_intermediate_dim),
              weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          out_sharding=self.wi_kernel_axes,
      )
      self.wi_1 = nnx.Param(
          self.kernel_init(
              self.rngs.params(),
              (num_experts, self.moe_expert_input_dim, moe_intermediate_dim),
              weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          out_sharding=self.wi_kernel_axes,
      )
      self.wo = nnx.Param(
          self.kernel_init(
              self.rngs.params(),
              (self.num_experts, self.intermediate_dim, self.moe_expert_input_dim),
              self.weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          out_sharding=self.wo_kernel_axes,
      )

    if self.config.mlp_bias:
      wi_bias_axes = ("exp", "activation_mlp")
      wo_bias_axes = ("exp", "activation_embed")
      wi_bias_shape = (self.num_experts, self.intermediate_dim)
      wo_bias_shape = (self.num_experts, self.moe_expert_input_dim)
      self.wi_0_bias = nnx.Param(
          default_bias_init(self.rngs.params(), wi_bias_shape, self.weight_dtype),
          out_sharding=wi_bias_axes,
      )
      self.wi_1_bias = nnx.Param(
          default_bias_init(self.rngs.params(), wi_bias_shape, self.weight_dtype),
          out_sharding=wi_bias_axes,
      )
      self.wo_bias = nnx.Param(
          default_bias_init(self.rngs.params(), wo_bias_shape, self.weight_dtype),
          out_sharding=wo_bias_axes,
      )
    else:
      self.wi_0_bias = None
      self.wi_1_bias = None
      self.wo_bias = None

    if self.config.decoder_block == ctypes.DecoderBlockType.GEMMA4:
      self.per_expert_scale = nnx.Param(
          jnp.ones((self.num_experts,), dtype=self.weight_dtype),
          out_sharding=("exp",),
      )
    else:
      self.per_expert_scale = None

  def _maybe_shard_with_logical(self, inputs, logical_name):
    return maybe_shard_with_logical(
        inputs,
        logical_name,
        mesh=self.mesh,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
        extra_stack_level=1,
    )

  def _logical_to_mesh_axes(self, logical_name):
    logical_rules = None if self.config.using_pipeline_parallelism else self.config.logical_axis_rules
    return logical_to_mesh_axes(logical_name, mesh=self.mesh, rules=logical_rules)

  def _maybe_shard_with_pspec(self, inputs, pspec: jax.sharding.PartitionSpec | None):
    return maybe_shard_with_pspec(
        inputs,
        pspec,
        mesh=self.mesh,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
        extra_stack_level=1,
    )

  def get_expert_parallelism_size(self):
    # When expert parallelism has more than one physical axes, take product of their shapes
    if isinstance(self._expert_parallelism_name, tuple):
      return math.prod(self.mesh.shape.get(name, 1) for name in self._expert_parallelism_name)
    return self.mesh.shape.get(self._expert_parallelism_name, 1)

  def get_tensor_parallelism_size(self):
    if isinstance(self._tensor_parallelism_name, tuple):
      size = 1
      for axis in self._tensor_parallelism_name:
        size *= self.mesh.shape.get(axis, 1)
      return size
    return self.mesh.shape.get(self._tensor_parallelism_name, 1)

  def get_tensor_transpose_parallelism_size(self):
    return self.mesh.shape.get("tensor_transpose", 1)

  def get_context_autoregressive_parallelism_size(self):
    return self.mesh.shape.get("context_autoregressive", 1)

  def should_update_load_balance(self):
    \"\"\"Determines if loss-free load balancing updates should be applied.\"\"\"
    return self.config.routed_bias and self.config.routed_bias_update_rate > 0.0

  def get_topk(self, gate_logits, pre_bias_logits, rngs=None):
    \"\"\"get topk.\"\"\"
    # shape of top_k_weights & top_k_indices:
    # (batch, sequence, num_experts_per_tok).
    if self.config.use_random_routing:
      if rngs is None:
        raise ValueError("The random key cannot be None for random routing.")
      # Reuse the 'params' RNG stream to ensure random routing
      rng = rngs.params()
      top_k_weights, top_k_indices = random_routing(rng, gate_logits, self.num_experts_per_tok)
      return top_k_weights, top_k_indices

    if self.config.model_name.startswith("deepseek3"):
      top_k_weights, top_k_indices = self.deepseek_routing(gate_logits, pre_bias_logits)
    elif self.config.decoder_block == ctypes.DecoderBlockType.GEMMA4:
      router_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1)
      _, top_k_indices = jax.lax.top_k(gate_logits, self.num_experts_per_tok)
      top_k_weights = jnp.take_along_axis(router_probs, top_k_indices, axis=-1).astype(self.dtype)
    else:
      top_k_weights, top_k_indices = jax.lax.top_k(gate_logits, self.num_experts_per_tok)

    if self.config.decoder_block == ctypes.DecoderBlockType.DEEPSEEK:
      top_k_weights = self.deepseek_scale_weights(top_k_weights)
    elif self.config.decoder_block not in (ctypes.DecoderBlockType.LLAMA4, ctypes.DecoderBlockType.GEMMA4):
      top_k_weights = jax.nn.softmax(top_k_weights.astype(jnp.float32), axis=-1).astype(self.dtype)

    # Normalization of router weights (e.g. used by Qwen3, Gemma4).
    if self.config.norm_topk_prob:
      top_k_weights /= top_k_weights.sum(axis=-1, keepdims=True)

    return top_k_weights, top_k_indices

  def deepseek_scale_weights(self, weights):
    \"\"\"Scales weights according to DeepSeek's v3 reference implementation.\"\"\"
    # https://github.com/deepseek-ai/DeepSeek-V3/blob/2f7b80eecebf3d1c84da5a0d465f6639ea175012/inference/model.py#L592-L594.
    if self.config.routed_score_func == "sigmoid":
      weights /= weights.sum(-1, keepdims=True)
    weights *= self.config.routed_scaling_factor
    return weights

  def expert_group_mask(self, gate_logits: jax.Array) -> jax.Array:
    \"\"\"Returns a mask that selects only the top-k groups of experts.

    Groups of experts are selected based on the sum of the top-2 expert scores
    for each group.

    Args:
      gate_logits: Array of shape `(batch, seq, num_experts)`.

    Returns:
      Array of shape `(batch, seq, num_experts)` that is 1 for experts in the
      top-k groups and 0 elsewhere.
    \"\"\"
    # Find top groups based on each group's top-2 expert scores, where
    # `scores_grouped.shape =
    # (batch * seq, n_routing_groups, experts_per_group)`.
    scores_grouped = jnp.reshape(
        gate_logits,
        gate_logits.shape[:-1] + (self.config.n_routing_groups, -1),
    )
    top2_in_group_vals, _ = jax.lax.top_k(scores_grouped, k=2)
    group_scores = jnp.sum(jnp.astype(top2_in_group_vals, jnp.float32), axis=-1)
    _, group_idx = jax.lax.top_k(group_scores, k=self.config.topk_routing_group)

    # Mask selected groups so that only those experts are considered.
    group_mask = jax.nn.one_hot(group_idx, num_classes=self.config.n_routing_groups, dtype=jnp.float32)
    group_mask = jnp.sum(group_mask, axis=-2)

    # Apply masks and get top-k indices.
    score_mask_expanded = jnp.broadcast_to(
        group_mask[..., None],
        group_mask.shape + (self.num_experts // self.config.n_routing_groups,),
    )
    return jnp.reshape(
        score_mask_expanded,
        score_mask_expanded.shape[:-2] + (self.num_experts,),
    )

  def deepseek_routing(self, gate_logits: jax.Array, pre_bias_logits: jax.Array) -> tuple[jax.Array, jax.Array]:
    \"\"\"DeepSeek routing logit.

    If the configuration does not specify routing groups (`n_routing_groups` is
    -1), we use a standard top-k routing mechanism. Otherwise, we force all
    selected experts to be from the a subset of the highest rated expert groups.

    The selection process uses post_bias logits, while the return weights use
    pre_bias logits.

    Args:
      gate_logits: Array of shape `(batch, seq, num_experts)`.
      pre_bias_logits: Array of shape `(batch, seq,num_experts)`.

    Returns:
      - top_k_weights: `(batch, seq, num_experts_per_tok)` array of weight values for
        each selected expert.
      - top_k_indices: `(batch, seq, num_experts_per_tok)` array of indices
        identifying the selected experts for each token.
    \"\"\"
    expert_mask = 1 if self.config.n_routing_groups == -1 else self.expert_group_mask(gate_logits)
    _, top_k_indices = jax.lax.top_k(
        jnp.where(expert_mask > 0, gate_logits, -jnp.inf),
        k=self.num_experts_per_tok,
    )
    top_k_weights = jnp.take_along_axis(pre_bias_logits, top_k_indices, axis=-1)
    return top_k_weights, top_k_indices

  def apply_ffn_activation(self, layer_w0, layer_w1):
    \"\"\"Applies FFN activation function.\"\"\"
    with jax.named_scope("ffn_act"):
      if self.config.decoder_block == ctypes.DecoderBlockType.GPT_OSS:
        layer_w0 = jnp.clip(layer_w0, min=None, max=self.config.mlp_activations_limit)
        layer_w1 = jnp.clip(layer_w1, min=-self.config.mlp_activations_limit, max=self.config.mlp_activations_limit)
        layer_act = self.activation_fn(layer_w0 * 1.702)
        glu = jnp.multiply(layer_w0, layer_act)
        intermediate_layer = jnp.multiply(glu, (layer_w1 + 1))
      else:
        layer_act = self.activation_fn(layer_w0)
        intermediate_layer = jnp.multiply(layer_act, layer_w1)
      return intermediate_layer.astype(self.dtype)

  def permute(self, inputs, gate_logits, pre_bias_logits, use_custom_sort_vjp=True, rngs=None, roll_to_expert_id=None):
    \"\"\"Permute tokens to group by expert to fit gmm call.\"\"\"
    # reshape inputs (batch, sequence, emb) to (batch * sequence, emb)
    inputs_shape = inputs.shape
    bsz_times_seq_len = inputs_shape[0] * inputs_shape[1]
    inputs_2d = jnp.reshape(inputs, (bsz_times_seq_len, inputs_shape[2]))
    weights, selected_experts = self.get_topk(gate_logits, pre_bias_logits, rngs)

    lb_loss = None
    if self.config.load_balance_loss_weight > 0.0:
      softmax_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
      lb_loss = self.load_balance_loss(selected_experts, softmax_probs)

    if self.should_update_load_balance():
      bias_updates = calculate_load_balance_updates(
          selected_experts, self.config.num_experts, self.config.routed_bias_update_rate
      )
    else:
      bias_updates = None

    if self.config.decoder_block == ctypes.DecoderBlockType.LLAMA4:
      # weights will be of shape (batch_size, seq_len, num_experts_per_tok)
      router_scores = jax.nn.sigmoid(weights.astype(jnp.float32))  # weights are top_k_weights here
      # Squeeze router_scores to (batch_size * seq_len, num_experts_per_tok)
      inputs_2d = inputs_2d * router_scores.reshape(bsz_times_seq_len, -1)

    flatten_selected_experts = jnp.ravel(selected_experts)
    if roll_to_expert_id is not None:
      flatten_selected_experts = (flatten_selected_experts - roll_to_expert_id) % self.num_experts
    sorted_selected_experts = jnp.argsort(flatten_selected_experts)
    # sort inputs for number of selected experts
    replicated_inputs_2d = jnp.repeat(inputs_2d, self.num_experts_per_tok, axis=0)
    sorted_inputs = _sort_activations(replicated_inputs_2d, sorted_selected_experts, use_custom_sort_vjp).astype(
        self.dtype
    )
    group_size = jnp.bincount(flatten_selected_experts, length=self.num_experts)
    # Return the experts for each sorted input.
    expert_indices = jnp.arange(self.num_experts)
    sorted_experts = jnp.repeat(
        expert_indices,
        repeats=group_size,
        total_repeat_length=flatten_selected_experts.shape[0],
    )
    return (
        sorted_inputs,
        sorted_selected_experts,
        weights,
        group_size,
        sorted_experts,
        lb_loss,
        bias_updates,
    )

  def unpermute(
      self,
      intermediate,
      sorted_selected_experts,
      weights,
      batch_size,
      sequence_length,
      use_custom_sort_vjp=True,
  ):
    \"\"\"Unpermute tokens to original order and combine weights.\"\"\"

    unsort_intermediate = _sort_activations(
        intermediate,
        jnp.argsort(sorted_selected_experts),
        use_custom_sort_vjp,
    )
    reshaped_weights = jnp.reshape(weights, (-1, self.num_experts_per_tok))
    reshaped_intermediate = jnp.reshape(
        unsort_intermediate,
        (reshaped_weights.shape[0], self.num_experts_per_tok, -1),
    )
    with jax.named_scope("weight_sum"):
      matmul_precision = jax.lax.Precision(self.config.matmul_precision)
      if self.config.decoder_block == ctypes.DecoderBlockType.LLAMA4:
        # For Llama4, combine using weights of 1 for selected experts
        reshaped_weights = jnp.ones_like(reshaped_weights)
      if self.config.float32_weight_sum:
        reshaped_intermediate = reshaped_intermediate.astype(jnp.float32)
        reshaped_weights = reshaped_weights.astype(jnp.float32)
      output = jnp.einsum(
          "BKE,BK -> BE",
          reshaped_intermediate,
          reshaped_weights,
          precision=matmul_precision,
      )
    return output.reshape(batch_size, sequence_length, -1).astype(self.dtype)

  @staticmethod
  def local_permute(
      inputs,
      global_group_sizes,
      local_expert_size,
      shard_index,
      is_offset=False,
      global_sorted_experts=None,
      use_custom_sort_vjp=True,
  ):
    \"\"\"Permutes tokens locally within an expert shard.

    This function prepares the input tokens for processing by the experts
    located
    on the current shard. It groups the tokens by their assigned local expert
    index (0 to local_expert_size - 1).

    Args:
      inputs: The input data (tokens) assigned to the experts on this shard.
        Shape `[tokens, emb_dim]`.
      global_group_sizes: The count of tokens assignments for each global expert
        across all the batch shards. Shape `[num_batch_shards, num_experts].
      local_expert_size: The number of experts handled by the current shard.
      shard_index: The index of the current expert shard (0 to
        num_expert_parallelism - 1).
      is_offset: If True, assumes `inputs` are pre-sorted by global expert ID
        and selects the slice relevant to this shard's assigned experts. If
        False, assumes that `inputs` corresponding to the shard's experts start
        from the beginning of the tensor but need to be permuted by expert ID.
      global_sorted_experts: Global expert IDs for the `inputs` used when
        `is_offset` is True. Shape `[total_tokens_for_this_shard]`.

    Returns:
      A tuple containing:
        sorted_inputs: Input data permuted local expert ID.
        sorted_indices: Indices used to permute the inputs.
        local_group_size: Number of tokens assigned to each local expert on this
          shard.
        sorted_experts_ids: expert ID corresponding to each token of the permuted
        inputs.
    \"\"\"

    # Slice the count of local expert IDs in each batch shard.
    # all_shard_local_sizes.shape: [expert_shard, local_expert_size]
    all_shard_local_sizes = jax.lax.dynamic_slice_in_dim(
        global_group_sizes,
        shard_index * local_expert_size,
        local_expert_size,
        axis=1,
    )
    local_sizes = all_shard_local_sizes.reshape(-1)

    # Total count of the local expert IDs is the sum of the counts across all
    # batch shards, since all batch shards will send their contributions to the
    # current expert shard.
    local_group_size = jnp.sum(all_shard_local_sizes, axis=0)

    # In this case, the data that needs to be processed by the local shard
    # does not start from row 0 but actually starts at
    # (jnp.concatenate((jnp.array([0]),
    #  jnp.cumsum(local_group_sizes[:-1]))[shard_id]).
    # This happens if batches (`inputs`) are replicated across expert shards and
    # pre-sorted by global Expert ID (via permute()).
    if is_offset:
      divided_assignments = jnp.floor_divide(global_sorted_experts, local_expert_size)
      expert_indices = jnp.where(
          divided_assignments == shard_index,
          jnp.mod(global_sorted_experts, local_expert_size),
          local_expert_size,
      )

    # In this case the `input` data has been received from the batch shards and
    # needs to be reorganized in order of local Expert IDs.
    else:
      base_indices = jnp.mod(jnp.arange(local_sizes.shape[0]), local_expert_size)
      expert_indices = jnp.repeat(base_indices, local_sizes, total_repeat_length=inputs.shape[0])

    sorted_indices = jnp.argsort(expert_indices)
    sorted_inputs = _sort_activations(inputs, sorted_indices, use_custom_sort_vjp)
    sorted_experts_ids = expert_indices[sorted_indices]
    return (
        sorted_inputs,
        sorted_indices,
        local_group_size,
        sorted_experts_ids,
    )

  @staticmethod
  def get_all_to_all_params(
      all_shards_group_sizes,
      shard_id,
      num_expert_parallelism,
      is_batch_sharded=True,
  ):
    \"\"\"Generates input offsets, send sizes, output offsets, and receive sizes used for ragged_all_to_all.\"\"\"

    class TransformStrategy(enum.Enum):
      INPUT_OFFSET = enum.auto()
      SEND_SIZE = enum.auto()
      OUTPUT_OFFSET = enum.auto()
      RECV_SIZE = enum.auto()

    def transform_array(input_array, shard_id, strategy, is_batch_sharded):
      \"\"\"Transforms the input array based on the specified strategy.\"\"\"
      # Prepares it for the usage with `ragged_all_to_all` API. The
      # transformation determines how data is sent and received between shards.
      if is_batch_sharded:
        if strategy == TransformStrategy.INPUT_OFFSET:
          # Index of input array for the send
          local_array = input_array[shard_id]
          return jnp.concatenate((jnp.array([0]), jnp.cumsum(local_array)[:-1]))
        elif strategy == TransformStrategy.SEND_SIZE:
          # Size of input array for the send
          return input_array[shard_id]
        elif strategy == TransformStrategy.OUTPUT_OFFSET:
          # Received index in the target output
          zero_row = jnp.zeros((1,) + input_array.shape[1:], dtype=input_array.dtype)
          array_with_zeros = jnp.concatenate((zero_row, input_array), axis=0)
          cumulated_array = jnp.cumsum(array_with_zeros, axis=0, dtype=input_array.dtype)
          return cumulated_array[shard_id]
        elif strategy == TransformStrategy.RECV_SIZE:
          # Received size in the target output
          return input_array[:, shard_id]
        else:
          raise ValueError(f"Unknown transform array strategy: {strategy}")

      # If the batch is unsharded then we send the same data slice to all other
      # shards. We also assume each shard will have the local processed inputs
      # sorted to start from index 0. Finally, len(input_array.shape) == 1 since
      # there is only one batch shard.
      else:
        if strategy == TransformStrategy.INPUT_OFFSET:
          # The data on each shard always starts at 0.
          return jnp.zeros(num_expert_parallelism, dtype=input_array.dtype)
        elif strategy == TransformStrategy.SEND_SIZE:
          # The send amount is always the amount of data the current expert
          # shard needs to process.
          return jnp.repeat(input_array[shard_id], num_expert_parallelism)
        elif strategy == TransformStrategy.OUTPUT_OFFSET:
          # The offset in each shard will just be the start of the group which
          # that shard is responsible for.
          output_offset = jnp.concatenate((jnp.array([0]), jnp.cumsum(input_array[:-1])))[shard_id]
          return jnp.repeat(output_offset, num_expert_parallelism)
        # The amount that each shard receives from all other shards is
        # equivalent to the group sizes (aka input_array).
        elif strategy == TransformStrategy.RECV_SIZE:
          # Received size in the target output
          return input_array
        else:
          raise ValueError(f"Unknown transform array strategy: {strategy}")

    input_offsets = transform_array(
        all_shards_group_sizes,
        shard_id,
        TransformStrategy.INPUT_OFFSET,
        is_batch_sharded,
    )
    send_sizes = transform_array(
        all_shards_group_sizes,
        shard_id,
        TransformStrategy.SEND_SIZE,
        is_batch_sharded,
    )
    output_offsets = transform_array(
        all_shards_group_sizes,
        shard_id,
        TransformStrategy.OUTPUT_OFFSET,
        is_batch_sharded,
    )
    recv_sizes = transform_array(
        all_shards_group_sizes,
        shard_id,
        TransformStrategy.RECV_SIZE,
        is_batch_sharded,
    )
    return input_offsets, send_sizes, output_offsets, recv_sizes

  def transform_bias(self, experts_index, *biases):
    \"\"\"Selects bias values for a variable number of bias tensors based on chosen experts.\"\"\"
    return tuple(bias[experts_index] for bias in biases)

  @staticmethod
  def get_ragged_buffer_size(local_batch, ep_degree, global_experts, top_k, ragged_buffer_factor):
    \"\"\"Calculates the token batch size of the ragged buffer.
    When explicitly setting ragged_buffer_factor>0, this is balanced_size * ragged_buffer_factor, which can drop tokens.
    Otherwise this will be worst case size to ensure no dropping.

    Inputs:
      local_batch: local token batch (batch*seq blown up by top_k) shard on this device (e.g. inside shard_map)
      ep_degree: degree of expert parallelism, generally equal to ici_expert_parallelism
      global_experts: unsharded expert count, e.g. 256 for deepseek
      top_k: aka num_experts_per_tok, 8 for deepseek.
      ragged_buffer_factor: When set > 0, the buffer is balanced_size * ragged_buffer_factor.
        The value 1.0 will be dropless only in the perfectly balanced case, else tokens will be dropped.
    Outputs:
      The ragged buffer's token batch size.
    \"\"\"
    balanced_size = local_batch
    if ragged_buffer_factor > 0.0:
      # This will drop tokens if the true distribution exceeds this buffer.
      return int(balanced_size * ragged_buffer_factor)
    else:
      # Worst case
      # Either determined by degree of EP, or can be less when num_local_exp is smaller than top_k:
      # Example: If we have 4 EP shards, top_k=8, and experts=256 (deepseek), then worst case is
      # all tokens in our EP replica get routed to a single shard, e.g. rank 0 - thus is |EP|=4x larger than perfectly
      # balanced. However if we use EP=128, then there are only 256/128 = 2 local experts, and thus at most in an EP
      # replica group only the 2 experts of top_k=8 can be chosen, so at most 1/4 of all tokens goes to the most
      # popular shard. Thus the imbalance factor goes like |EP|/(top_k/local_exp) = 128/4 = 32.
      # In general for local_experts < top_k (e.g. |EP|>32), the balance will go as
      # EP * local_experts / top_k = EP * (global_exp/EP) / top_k = global_exp / top_k.
      # This is constant as a function of the model - e.g. for deepseek the imbalance is never worse than
      # 256 exp / 8 top_k = 32. In practice the imbalance should be much less and potentially can use
      # ragged_buffer_factor set to >1  e.g. 3.0, and likely have no dropping (not guaranteed)
      worst_case_factor = min(ep_degree, global_experts / top_k)
      return int(balanced_size * worst_case_factor)

  def sparse_matmul(
      self,
      inputs,
      gate_logits,
      pre_bias_logits,
      w0_kernel,
      w1_kernel,
      wo_kernel,
      w0_bias,
      w1_bias,
      wo_bias,
  ):
    \"\"\"Perform sparse matrix multiplication of inputs and Experts.\"\"\"

    def jax_ragged_dot_gmm(inputs, kernel, tiling, group_sizes, expert_assignments, padding_amount):
      \"\"\"Execute jax.lax.ragged_dot, with potential quantization\"\"\"
      m, k, n = inputs.shape[0], inputs.shape[1], kernel.shape[2]
      tiling = (
          min(tiling[0], m),
          min(tiling[1], k),
          min(tiling[2], n),
      )
      rhs_inputs = kernel
      if isinstance(kernel, aqt.QTensor):
        if kernel.bias or kernel.sparsity_mask or len(kernel.scale) > 1:
          raise ValueError("Unsupported usecase for ragged_dot with quantized kernel.")
        rhs_inputs = kernel.qvalue
      if self.config.use_qwix_quantization:
        # Use full contraction for QWIX quantization to allow quantization
        # fusion (max reduce over contracting dimension).
        tiling = (tiling[0], k, tiling[2])

      is_tpu = self.mesh.devices.flat[0] == "tpu"
      # TPU needs random mosaic_fusion_group; GPU/CPU needs deterministic ID for autotuner sync
      mosaic_group_id = f"{random.randint(0, 1000000000)}" if is_tpu else "0"
      with set_xla_metadata(
          ragged_dot_tiling=",".join([str(t) for t in tiling]),
          mosaic_fusion_group=mosaic_group_id,
      ):
        output = jax.lax.ragged_dot(
            lhs=inputs,
            rhs=rhs_inputs,
            group_sizes=group_sizes,
            preferred_element_type=self.dtype,
        )
      if isinstance(kernel, aqt.QTensor):
        # Multiply outputs by the kernely scale
        scales = jnp.take(kernel.scale[0].squeeze(), indices=expert_assignments, axis=0)
        if padding_amount > 0:
          scales = jax.lax.pad(
              scales,
              jnp.array(0.0, dtype=scales.dtype),
              [(0, padding_amount, 0), (0, 0, 0)],
          )
        output *= scales
      return output

    def get_tokamax_group_sizes(group_sizes, inputs, kernel):
      # TODO (b/491979205) pipeline fsdp ag per repeat fails tokamax gmm
      if self.config.use_qwix_quantization or (
          self.config.using_pipeline_parallelism and self.config.pipeline_fsdp_ag_per_repeat
      ):
        return group_sizes
      elif self.config.attention == "vllm_rpa":
        return group_sizes
      else:
        return tokamax.RaggedDotGroupSizes(
            group_sizes,
            (inputs.shape[0] // kernel.shape[0],) * kernel.shape[0],
        )

    def get_quantization_dtypes():
      lhs_quantize_dtype, rhs_quantize_dtype = None, None
      if self.quant is not None:
        quant_dg = self.quant.quant_dg
        lhs_quantize_dtype = quant_dg.fwd.dg_quantizer.lhs.numerics.get_dtype()
        rhs_quantize_dtype = quant_dg.fwd.dg_quantizer.rhs.numerics.get_dtype()
      return lhs_quantize_dtype, rhs_quantize_dtype

    def gmm(inputs, kernel, tiling, group_sizes, expert_assignments, weight_gather_axes):
      if inputs.shape[0] != expert_assignments.shape[0]:
        raise ValueError("The number of input tokens must match the number of expert assignments!")

      tokamax_group_sizes = get_tokamax_group_sizes(group_sizes, inputs, kernel)
      orig_inputs_shape = inputs.shape  # save shape of inputs before potentially padding.
      inputs, padding_amount = max_utils.maybe_pad(inputs, self.config.wi_tile_fwd_batch_seq)
      inputs = inputs.astype(self.dtype)
      kernel = kernel.astype(self.dtype)
      lhs_quantize_dtype, rhs_quantize_dtype = get_quantization_dtypes()

      # We support three implementations for gmm - tokamax, older forked kernel, or jax.lax.ragged_dot
      # For quantized tokamax we call a forked version that supports our quantization recipes.
      if self.config.use_tokamax_gmm:
        if self.config.quantization:  # tokamax (quantized)
          output = mblx.gmm(
              lhs=inputs,
              rhs=kernel,
              group_sizes=group_sizes,
              preferred_element_type=self.dtype,
              tiling=tiling,
              lhs_quantize_dtype=lhs_quantize_dtype,
              rhs_quantize_dtype=rhs_quantize_dtype,
              use_qwix_quantization=self.config.use_qwix_quantization,
              use_tokamax_backend=self.config.use_tokamax_gmm,
              weight_gather_axes=weight_gather_axes,
          )
        else:  # tokamax (unquantized)
          output = tokamax.ragged_dot(
              lhs=inputs,
              rhs=kernel,
              group_sizes=tokamax_group_sizes,
              precision=jax.lax.Precision.DEFAULT,
              preferred_element_type=self.dtype,
              implementation="mosaic",
          )
      elif self.config.megablox:  # Older forked megablox
        output = mblx.gmm(
            lhs=inputs,
            rhs=kernel,
            group_sizes=group_sizes,
            preferred_element_type=self.dtype,
            tiling=tiling,
            lhs_quantize_dtype=lhs_quantize_dtype,
            rhs_quantize_dtype=rhs_quantize_dtype,
            use_qwix_quantization=self.config.use_qwix_quantization,
            use_tokamax_backend=self.config.use_tokamax_gmm,
            weight_gather_axes=weight_gather_axes,
        )
      else:  # jax.lax.ragged_dot
        output = jax_ragged_dot_gmm(inputs, kernel, tiling, group_sizes, expert_assignments, padding_amount)
      if padding_amount > 0:
        output = output[: orig_inputs_shape[0]]
      return output

    def is_batch_sharded_by_ep(input_activation):
      # The batch is sharded by expert, except during inference decoding (where batch size == 1).
      # In the decoding case, the expert axis is instead replicated along the tensor's batch dimension.
      return input_activation.shape[0] > 1

    def explicitly_weight_ag(shard_exp_on_fsdp):
      if shard_exp_on_fsdp:
        quantization_rule = qpl.get_current_rule("gmm")
        if quantization_rule and quantization_rule.weight_calibration_method.startswith("fixed"):
          return True
      return False

    def maybe_aqt_partition(w0_kernel, w0_pspec, w1_kernel, w1_pspec, wo_kernel, wo_pspec):
      if isinstance(w0_kernel, aqt.QTensor):
        w0_pspec = aqt.partition_spec(w0_pspec, (1,), w0_kernel.dtype, use_bias=False)
      if isinstance(w1_kernel, aqt.QTensor):
        w1_pspec = aqt.partition_spec(w1_pspec, (1,), w1_kernel.dtype, use_bias=False)
      if isinstance(wo_kernel, aqt.QTensor):
        wo_pspec = aqt.partition_spec(wo_pspec, (1,), wo_kernel.dtype, use_bias=False)
      return w0_pspec, w1_pspec, wo_pspec

    def get_routed_moe_shardings(is_batch_sharded_by_expert):
      if is_batch_sharded_by_expert:
        batch_logical_axis = "activation_batch"
      else:
        batch_logical_axis = "decode_batch_moe"

      if self.get_tensor_transpose_parallelism_size() > 1:
        input_partition_pspec = self._logical_to_mesh_axes(
            (batch_logical_axis, "activation_norm_length", "activation_embed")
        )
        w0_bias_pspec = self._logical_to_mesh_axes(("exp", None))
        w1_bias_pspec = self._logical_to_mesh_axes(("exp", None))
        wo_bias_pspec = self._logical_to_mesh_axes(("exp", "activation_embed"))
      else:
        input_partition_pspec = self._logical_to_mesh_axes((batch_logical_axis, "activation_norm_length", None))
        w0_bias_pspec = self._logical_to_mesh_axes(("exp", "activation_mlp"))
        w1_bias_pspec = self._logical_to_mesh_axes(("exp", "activation_mlp"))
        wo_bias_pspec = self._logical_to_mesh_axes(("exp", "activation_embed"))

      gate_logits_pspec = self._logical_to_mesh_axes((batch_logical_axis, "activation_norm_length", None))
      if self.config.model_name.startswith("deepseek3"):
        pre_bias_logits_pspec = self._logical_to_mesh_axes((batch_logical_axis, "activation_norm_length", None))
      else:
        # pre_bias_logits is None for non-DeepSeek v3 models
        pre_bias_logits_pspec = None

      # w0, w1, wo needs to be un sharded on fsdp / fsdp_transpose axis, so use
      # mlp_no_fsdp axis
      if self.config.shard_exp_on_fsdp:
        quantization_rule = qpl.get_current_rule("gmm")
        if quantization_rule and quantization_rule.weight_calibration_method.startswith("fixed"):
          # special sharding when using static scaling for weights in quantization with shard_exp_on_fsdp
          w0_pspec = self._logical_to_mesh_axes(self.wi_kernel_axes)
          w1_pspec = self._logical_to_mesh_axes(self.wi_kernel_axes)
          wo_pspec = self._logical_to_mesh_axes(self.wo_kernel_axes)
        else:
          # special sharding for dsv3 to remove overhead between gmm/AG
          w0_pspec = self._logical_to_mesh_axes(("embed_tensor_transpose", None, "mlp_no_fsdp"))
          w1_pspec = self._logical_to_mesh_axes(("embed_tensor_transpose", None, "mlp_no_fsdp"))
          wo_pspec = self._logical_to_mesh_axes(("embed_tensor_transpose", "mlp_no_fsdp", None))
      elif self.config.use_2d_fsdp_sharding:
        w0_pspec = self._logical_to_mesh_axes(("embed_tensor_transpose", "mlp_no_fsdp", None))
        w1_pspec = self._logical_to_mesh_axes(("embed_tensor_transpose", "mlp_no_fsdp", None))
        wo_pspec = self._logical_to_mesh_axes(("embed_tensor_transpose", "mlp_no_fsdp", None))
      else:
        # These are the main shardings used by default - they use funky rules to AG over FSDP.
        w0_pspec = self._logical_to_mesh_axes(("exp", "embed_tensor_transpose", "mlp_no_fsdp"))
        w1_pspec = self._logical_to_mesh_axes(("exp", "embed_tensor_transpose", "mlp_no_fsdp"))
        wo_pspec = self._logical_to_mesh_axes(("exp", "mlp_no_fsdp", "embed_tensor_transpose"))
      return (
          batch_logical_axis,
          input_partition_pspec,
          gate_logits_pspec,
          pre_bias_logits_pspec,
          w0_pspec,
          w1_pspec,
          wo_pspec,
          w0_bias_pspec,
          w1_bias_pspec,
          wo_bias_pspec,
      )

    is_batch_sharded_by_expert = is_batch_sharded_by_ep(inputs)
    weight_gather = explicitly_weight_ag(self.config.shard_exp_on_fsdp)
    (
        batch_logical_axis,
        input_partition_pspec,
        gate_logits_pspec,
        pre_bias_logits_pspec,
        w0_pspec,
        w1_pspec,
        wo_pspec,
        w0_bias_pspec,
        w1_bias_pspec,
        wo_bias_pspec,
    ) = get_routed_moe_shardings(is_batch_sharded_by_expert)
    w0_pspec, w1_pspec, wo_pspec = maybe_aqt_partition(w0_kernel, w0_pspec, w1_kernel, w1_pspec, wo_kernel, wo_pspec)

    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=(
            input_partition_pspec,
            gate_logits_pspec,
            pre_bias_logits_pspec,
            w0_pspec,
            w1_pspec,
            wo_pspec,
            w0_bias_pspec,
            w1_bias_pspec,
            wo_bias_pspec,
            P(),  # Replicate the input key
        ),
        out_specs=(
            self._logical_to_mesh_axes((batch_logical_axis, "activation_norm_length", "activation_embed")),
            P(),  # Handle None or replicate the output
            P(),  # Handle None or replicate the output
        ),
        check_vma=False,
    )
    def wrapper(x, logits, pre_bias_logits, w0, w1, wo, w0_bias, w1_bias, wo_bias, rngs):
      batch_size, sequence_length, _ = x.shape
      num_expert_parallelism = self.get_expert_parallelism_size()
      if num_expert_parallelism > 1:
        expert_shard_id = jax.lax.axis_index(self._expert_parallelism_name)
      else:
        expert_shard_id = 0
      num_expert_parallelism = self.get_expert_parallelism_size()
      if self.config.use_ring_of_experts:
        # The ring-of-experts strategy first duplicates the inputs to all
        # expert shards, and then routes within each shard.

        # Duplicate inputs to all expert shards.
        x, logits, pre_bias_logits = tuple(
            jax.lax.all_gather(z, axis_name=self._expert_parallelism_name, tiled=True)
            for z in (x, logits, pre_bias_logits)
        )

        # "Route" tokens within each shard.
        num_experts_per_shard = self.config.num_experts // num_expert_parallelism
        x, sorted_selected_experts, weights, group_sizes, selected_experts, lb_loss, bias_updates = self.permute(
            x,
            logits,
            pre_bias_logits,
            self.config.use_custom_sort_vjp,
            roll_to_expert_id=num_experts_per_shard * expert_shard_id,
            rngs=rngs,
        )

        # Filter down to the group sizes that apply to only the experts in the
        # current shard.
        group_sizes = group_sizes[:num_experts_per_shard]
        mask = jnp.arange(x.shape[0]) < jnp.sum(group_sizes)
        x = jnp.where(mask[:, None], x, 0)
      else:
        x, sorted_selected_experts, weights, group_sizes, selected_experts, lb_loss, bias_updates = self.permute(
            x, logits, pre_bias_logits, self.config.use_custom_sort_vjp, rngs
        )

        if num_expert_parallelism > 1:
          batch_axis = self._expert_parallelism_name if is_batch_sharded_by_expert else "data"
          # get group sizes for all shards
          local_expert_size = self.config.num_experts // num_expert_parallelism
          reshaped_group_sizes = jnp.sum(group_sizes.reshape(-1, local_expert_size), axis=1)
          global_group_sizes = group_sizes

          if is_batch_sharded_by_expert:
            all_shards_group_sizes = jax.lax.all_gather(reshaped_group_sizes, axis_name=batch_axis)
            input_offsets, send_sizes, output_offsets, recv_sizes = RoutedMoE.get_all_to_all_params(
                all_shards_group_sizes,
                expert_shard_id,
                num_expert_parallelism,
            )

            buffer_size = self.get_ragged_buffer_size(
                jnp.shape(x)[0],
                num_expert_parallelism,
                self.config.num_experts,
                self.config.num_experts_per_tok,
                self.config.ragged_buffer_factor,
            )
            output_shape = jax.lax.empty((buffer_size, self.moe_expert_input_dim), dtype=x.dtype)

            x = jax.lax.ragged_all_to_all(
                x,
                output_shape,
                input_offsets,
                send_sizes,
                output_offsets,
                recv_sizes,
                axis_name=self._expert_parallelism_name,
            )
            global_group_sizes = jax.lax.all_gather(group_sizes, axis_name=self._expert_parallelism_name)
            x, local_sorted_indices, group_sizes, selected_experts = RoutedMoE.local_permute(
                x,
                global_group_sizes,
                local_expert_size,
                shard_index=expert_shard_id,
                use_custom_sort_vjp=self.config.use_custom_sort_vjp,
            )
          else:
            x, local_sorted_indices, group_sizes, selected_experts = RoutedMoE.local_permute(
                x,
                global_group_sizes[None, :],
                local_expert_size,
                shard_index=expert_shard_id,
                is_offset=True,
                global_sorted_experts=selected_experts,
                use_custom_sort_vjp=self.config.use_custom_sort_vjp,
            )

      if self.config.mlp_bias:
        w0_bias, w1_bias, wo_bias = self.transform_bias(selected_experts, w0_bias, w1_bias, wo_bias)

      def get_active_sharding_axes(pspec_dim_axes, tensor_dim_index):
        if pspec_dim_axes is None:
          return []
        axes = (pspec_dim_axes,) if isinstance(pspec_dim_axes, str) else pspec_dim_axes
        active = []
        for ax in axes:
          if ax and self.mesh.shape.get(ax, 1) > 1:
            active.append((ax, tensor_dim_index))
        return active

      wi_gather_axes = []
      wo_gather_axes = []

      if weight_gather:
        # wi [Experts, In, Hidden] -> Gather Exp(0) and Hidden(2)
        wi_gather_axes.extend(get_active_sharding_axes(w0_pspec[0], 0))
        wi_gather_axes.extend(get_active_sharding_axes(w0_pspec[2], 2))

        # wo [Experts, Hidden, Out] -> Gather Exp(0) and Hidden(1)
        wo_gather_axes.extend(get_active_sharding_axes(wo_pspec[0], 0))
        wo_gather_axes.extend(get_active_sharding_axes(wo_pspec[1], 1))
      gmm_fn = functools.partial(
          gmm,
          group_sizes=group_sizes,
          expert_assignments=selected_experts,
      )
      wi_tile_size = (
          self.config.wi_tile_fwd_batch_seq,  # m (LHS batch)
          self.config.wi_tile_fwd_embed_dim,  # k  (contracting)
          self.config.wi_tile_fwd_mlp_dim,  # n (RHS batch)
          self.config.wi_tile_dlhs_batch_seq,  # m (LHS batch)
          self.config.wi_tile_dlhs_mlp_dim,  # k (contracting)
          self.config.wi_tile_dlhs_embed_dim,  # n (RHS batch)
          self.config.wi_tile_drhs_batch_seq,  # Called m in megablox, but this is contracting
          self.config.wi_tile_drhs_embed_dim,  # Called k in megablox, but this is LHS batch dim
          self.config.wi_tile_drhs_mlp_dim,  # Called n in megablox, and indeed is RHS batch dim
      )
      wo_tile_size = (
          self.config.wo_tile_fwd_batch_seq,  # m (LHS batch)
          self.config.wo_tile_fwd_mlp_dim,  # k (contracting)
          self.config.wo_tile_fwd_embed_dim,  # n (RHS batch)
          self.config.wo_tile_dlhs_batch_seq,  # m (LHS batch)
          self.config.wo_tile_dlhs_embed_dim,  # k (contracting)
          self.config.wo_tile_dlhs_mlp_dim,  # n (RHS)
          self.config.wo_tile_drhs_batch_seq,  # Called m in megablox, but this is contracting
          self.config.wo_tile_drhs_mlp_dim,  # Called k in megablox, but this is LHS batch dim
          self.config.wo_tile_drhs_embed_dim,  # Called n in megablox, and indeed is the RHS batch dim
      )

      layer_w0 = gmm_fn(
          x,
          w0,
          tiling=wi_tile_size,
          weight_gather_axes=wi_gather_axes,
      )
      if self.get_tensor_transpose_parallelism_size() > 1:
        layer_w0 = jax.lax.psum(layer_w0, "tensor_transpose")
      if self.config.mlp_bias:
        layer_w0 = layer_w0 + w0_bias
      layer_w0 = adc.checkpoint_name(layer_w0, "moe_mlpwi_0")

      layer_w1 = gmm_fn(
          x,
          w1,
          tiling=wi_tile_size,
          weight_gather_axes=wi_gather_axes,
      )
      if self.get_tensor_transpose_parallelism_size() > 1:
        layer_w1 = jax.lax.psum(layer_w1, "tensor_transpose")
      if self.config.mlp_bias:
        layer_w1 = layer_w1 + w1_bias
      layer_w1 = adc.checkpoint_name(layer_w1, "moe_mlpwi_1")
      intermediate_layer = self.apply_ffn_activation(layer_w0, layer_w1)

      intermediate_output = gmm_fn(
          intermediate_layer,
          wo,
          tiling=wo_tile_size,
          weight_gather_axes=wo_gather_axes,
      )
      if self.get_tensor_parallelism_size() > 1:
        intermediate_output = jax.lax.psum_scatter(
            intermediate_output, self._tensor_parallelism_name, scatter_dimension=1, tiled=True
        )
      if self.config.mlp_bias:
        intermediate_output = intermediate_output + wo_bias
      intermediate_output = adc.checkpoint_name(intermediate_output, "moe_mlpwo")

      if self.config.use_ring_of_experts:
        # Set the outputs of tokens which were not processed to 0.
        mask = jnp.arange(intermediate_output.shape[0]) < jnp.sum(group_sizes)
        intermediate_output = jnp.where(mask[:, None], intermediate_output, 0)

        # Unsort and deduplicate the outputs locally.
        output = self.unpermute(
            intermediate_output,
            sorted_selected_experts,
            weights,
            batch_size=batch_size,
            sequence_length=sequence_length,
            use_custom_sort_vjp=self.config.use_custom_sort_vjp,
        )

        # Sum up the partial outputs across the expert shards.
        output = jnp.reshape(
            output, (-1, sequence_length, self.moe_expert_input_dim // self.get_tensor_parallelism_size())
        )
        output = jax.lax.psum_scatter(output, self._expert_parallelism_name, scatter_dimension=0, tiled=True)

      else:
        if num_expert_parallelism > 1:
          original_inputs_first_dim = batch_size * sequence_length * self.config.num_experts_per_tok
          if sorted_selected_experts.shape[0] != original_inputs_first_dim:
            raise ValueError("original_inputs_first_dim does not match the original tensor" " shape!")
          output_shape = jax.lax.empty(
              (
                  original_inputs_first_dim,
                  self.moe_expert_input_dim // self.get_tensor_parallelism_size(),
              ),
              dtype=intermediate_output.dtype,
          )

          if is_batch_sharded_by_expert:
            # locally unpermute back to the original order
            local_output = _sort_activations(
                intermediate_output,
                jnp.argsort(local_sorted_indices),  # pylint: disable=undefined-variable
                self.config.use_custom_sort_vjp,
            )
            input_offsets, send_sizes, output_offsets, recv_sizes = RoutedMoE.get_all_to_all_params(
                jnp.transpose(all_shards_group_sizes),  # pylint: disable=undefined-variable
                expert_shard_id,
                num_expert_parallelism,
            )
            intermediate_output = jax.lax.ragged_all_to_all(
                local_output,
                output_shape,
                input_offsets,
                send_sizes,
                output_offsets,
                recv_sizes,
                axis_name=self._expert_parallelism_name,
            )
          else:
            # If bach is replicated across EP shards then each shard should send
            # 0..local_shard_size data to the other shards and receive the
            # local_shard data from all of the other shards using
            # ragged_all_to_all.
            input_offsets, send_sizes, output_offsets, recv_sizes = RoutedMoE.get_all_to_all_params(
                reshaped_group_sizes,  # pylint: disable=undefined-variable
                expert_shard_id,
                num_expert_parallelism,
                is_batch_sharded=False,
            )
            intermediate_output = jax.lax.ragged_all_to_all(
                intermediate_output,
                output_shape,
                input_offsets,
                send_sizes,
                output_offsets,
                recv_sizes,
                axis_name=self._expert_parallelism_name,
            )

        output = self.unpermute(
            intermediate_output,
            sorted_selected_experts,
            weights,
            batch_size=batch_size,
            sequence_length=sequence_length,
            use_custom_sort_vjp=self.config.use_custom_sort_vjp,
        )

      return output, lb_loss, bias_updates

    if self.config.moe_fsdp_use_two_stage_all_gather:
      # Unshard on fsdp axis
      w0_kernel = self._maybe_shard_with_logical(w0_kernel, ("exp_with_fsdp", "embed_tensor_transpose", "mlp"))
      w1_kernel = self._maybe_shard_with_logical(w1_kernel, ("exp_with_fsdp", "embed_tensor_transpose", "mlp"))

      # Unshard on fsdp_transpose axis
      wo_kernel = self._maybe_shard_with_logical(wo_kernel, ("exp_with_fsdp", "mlp", "embed_tensor_transpose"))

      # Make sure XLA does not optimize by combining above All-Gather to unshard
      # on FSDP axis and the subsequent unshard on fsdp_transpose axis
      w0_kernel = jax.lax.optimization_barrier(w0_kernel)
      w1_kernel = jax.lax.optimization_barrier(w1_kernel)
      wo_kernel = jax.lax.optimization_barrier(wo_kernel)

      # Unshard on both fsdp and fsdp_transpose transpose
      w0_kernel = self._maybe_shard_with_logical(w0_kernel, ("exp_with_fsdp", "embed_tensor_transpose", "mlp_no_fsdp"))
      w1_kernel = self._maybe_shard_with_logical(w1_kernel, ("exp_with_fsdp", "embed_tensor_transpose", "mlp_no_fsdp"))
      wo_kernel = self._maybe_shard_with_logical(wo_kernel, ("exp_with_fsdp", "mlp_no_fsdp", "embed_tensor_transpose"))

    if self.get_tensor_transpose_parallelism_size() > 1:
      input_axes = (batch_logical_axis, "activation_norm_length", "activation_embed")
    else:
      input_axes = (batch_logical_axis, "activation_norm_length", None)

    gate_logits_axes = (batch_logical_axis, "activation_norm_length", None)
    if self.config.model_name.startswith("deepseek3"):
      pre_bias_logits_axes = (batch_logical_axis, "activation_norm_length", None)
    else:
      pre_bias_logits_axes = None

    inputs = self._maybe_shard_with_logical(inputs, input_axes)
    gate_logits = self._maybe_shard_with_logical(gate_logits, gate_logits_axes)
    pre_bias_logits = self._maybe_shard_with_logical(pre_bias_logits, pre_bias_logits_axes)

    w0_kernel = self._maybe_shard_with_pspec(w0_kernel, w0_pspec)
    w1_kernel = self._maybe_shard_with_pspec(w1_kernel, w1_pspec)
    wo_kernel = self._maybe_shard_with_pspec(wo_kernel, wo_pspec)
    if w0_bias is not None:
      w0_bias = self._maybe_shard_with_pspec(w0_bias, w0_bias_pspec)
    if w1_bias is not None:
      w1_bias = self._maybe_shard_with_pspec(w1_bias, w1_bias_pspec)
    if wo_bias is not None:
      wo_bias = self._maybe_shard_with_pspec(wo_bias, wo_bias_pspec)

    return wrapper(
        inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel, w0_bias, w1_bias, wo_bias, self.rngs
    )

  def reshape_and_update_weights(self, weights, indices):
    \"\"\"reshape and update weights.\"\"\"
    # input of weights and indices: (batch_size, seq_len, num_experts_per_tok)
    # output of updated weights: (batch_size, seq_len, num_experts)
    update_weights = jnp.zeros((weights.shape[0], weights.shape[1], self.num_experts), dtype=self.dtype)
    index_update = (
        self._maybe_shard_with_logical(jnp.arange(weights.shape[0])[:, None, None], ("activation_batch", None, None)),
        self._maybe_shard_with_logical(jnp.arange(weights.shape[1])[:, None], ("activation_length", None)),
        indices,
    )
    weight_sharding = (
        create_sharding(self.mesh, ("activation_batch", "activation_length", None))
        if self.config.shard_mode == ShardMode.EXPLICIT
        else None
    )
    update_weights = update_weights.at[index_update].set(weights, out_sharding=weight_sharding)
    return update_weights

  def get_context_partition_and_sub_seq(self, seq_len):
    cp = self.get_context_autoregressive_parallelism_size()
    if seq_len % cp != 0:
      cp = 1
    sub_seq = seq_len // cp
    return cp, sub_seq

  def generate_masks_subgroup(self, top_k_indices, softmax_probs):
    \"\"\"Subgroup mask generation for inference only.\"\"\"
    # calculate
    # expert_capacity = (tokens_per_batch / num_experts) * capacity_factor
    batch_size, seq_len, _ = top_k_indices.shape
    cp, sub_seq = self.get_context_partition_and_sub_seq(seq_len)

    # Break sequence into subsequences (groups) of tokens, and route only within
    # each group.
    top_k_indices = jnp.reshape(top_k_indices, (batch_size, cp, sub_seq, top_k_indices.shape[2]))

    tokens_per_batch = sub_seq * self.num_experts_per_tok
    # this is to avoid expert_capacity_per_batch = 0
    expert_capacity_per_batch = int(
        max(
            math.ceil(tokens_per_batch / self.num_experts) * self.config.capacity_factor,
            self.config.capacity_factor,
        )
    )
    max_logging.log("Applying potential token dropping with a batch expert_capacity of" f" {expert_capacity_per_batch}")

    # calculate expert mask and drop tokens if needed
    # shape of output expert mask: (batch, sequence, num_experts_per_tok)
    #
    # A small example:
    # give num_experts=4 & num_experts_per_tok=2, and two tokens are routed to
    # expert [0, 1] & [1, 3],
    # then expert_mask becomes
    # [[[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 1, 0, 0],[0, 0, 0, 1]]]],
    # after cumsum, expert_token_count becomes
    # [[[[1, 0, 0, 0],[1, 1, 0, 0]], [[1, 2, 0, 0],[1, 2, 0, 1]]]],
    # if we set expert_capacity=1,
    # trunc_expert_mask becomes
    # [[[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 0, 0, 0],[0, 0, 0, 1]]]],
    # so the 2nd token for expert #1 ([0, 1] & [1, 3]) is dropped, output of
    # updated_expert_mask is [[[1, 1],[0, 1]]].
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
    expert_mask_fused = jnp.reshape(
        expert_mask,
        (batch_size, cp, sub_seq * self.num_experts_per_tok, self.num_experts),
    )
    expert_mask_fused = self._maybe_shard_with_logical(expert_mask_fused, ("activation_batch", None, None, None))
    expert_token_count_fused = jnp.cumsum(expert_mask_fused, axis=2)
    expert_token_count = jnp.reshape(
        expert_token_count_fused,
        ((batch_size, cp, sub_seq, self.num_experts_per_tok, self.num_experts)),
    )
    expert_token_count = self._maybe_shard_with_logical(
        expert_token_count,
        ("activation_batch", "activation_norm_length", None, None, None),
    )
    trunc_expert_mask = expert_mask * jnp.less_equal(expert_token_count, expert_capacity_per_batch)
    combined_expert_mask = jnp.sum(trunc_expert_mask, axis=3)

    # reshape & update weights
    softmax_probs = jnp.reshape(
        softmax_probs,
        ((batch_size, cp, sub_seq, self.num_experts)),
    )
    softmax_probs *= combined_expert_mask

    # calculate token position in expert capacity dimension
    expert_token_position_fused = expert_mask_fused * expert_token_count_fused
    expert_token_position = jnp.reshape(
        expert_token_position_fused,
        (batch_size, cp, sub_seq, self.num_experts_per_tok, self.num_experts),
    )
    combined_expert_token_position = jnp.sum(expert_token_position, axis=3) * combined_expert_mask
    expert_token_position_in_capacity = jax.nn.one_hot(
        combined_expert_token_position,
        num_classes=expert_capacity_per_batch + 1,
        dtype=jnp.int32,
    )

    # shape of combine_mask is
    # (batch_size, seq_len, num_experts, expert_capacity_per_batch + 1),
    # and cut 0-dimension which is always 0
    combine_mask = softmax_probs[..., None] * expert_token_position_in_capacity
    combine_mask = combine_mask[..., 1:]
    dispatch_mask = combine_mask.astype(bool)

    # ici_context_parallelism
    dispatch_mask = jnp.reshape(
        dispatch_mask,
        (batch_size, cp, sub_seq, self.num_experts, expert_capacity_per_batch),
    )
    combine_mask = jnp.reshape(
        combine_mask,
        (batch_size, cp, sub_seq, self.num_experts, expert_capacity_per_batch),
    )

    return dispatch_mask, combine_mask

  def generate_masks(self, top_k_indices, softmax_probs):
    \"\"\"Generate masks.\"\"\"
    # calculate
    # expert_capacity = (tokens_per_batch / num_experts) * capacity_factor
    batch_size, seq_len, _ = top_k_indices.shape

    tokens_per_batch = seq_len * self.num_experts_per_tok
    # this is to avoid expert_capacity_per_batch = 0
    expert_capacity_per_batch = int(
        max(
            math.ceil(tokens_per_batch / self.num_experts) * self.config.capacity_factor,
            self.config.capacity_factor,
        )
    )
    max_logging.log("Applying potential token dropping with a batch expert_capacity of" f" {expert_capacity_per_batch}")

    # calculate expert mask and drop tokens if needed
    # shape of output expert mask: (batch, sequence, num_experts_per_tok)
    #
    # A small example:
    # give num_experts=4 & num_experts_per_tok=2, and two tokens are routed to
    # expert [0, 1] & [1, 3],
    # then expert_mask becomes
    # [[[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 1, 0, 0],[0, 0, 0, 1]]]],
    # after cumsum, expert_token_count becomes
    # [[[[1, 0, 0, 0],[1, 1, 0, 0]], [[1, 2, 0, 0],[1, 2, 0, 1]]]],
    # if we set expert_capacity=1,
    # trunc_expert_mask becomes
    # [[[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 0, 0, 0],[0, 0, 0, 1]]]],
    # so the 2nd token for expert #1 ([0, 1] & [1, 3]) is dropped, output of
    # updated_expert_mask is [[[1, 1],[0, 1]]].
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
    expert_mask_fused = jnp.reshape(
        expert_mask,
        (batch_size, seq_len * self.num_experts_per_tok, self.num_experts),
    )
    expert_mask_fused = self._maybe_shard_with_logical(expert_mask_fused, ("activation_batch_moe", None, None))
    expert_token_count_fused = jnp.cumsum(expert_mask_fused, axis=1)
    expert_token_count = jnp.reshape(
        expert_token_count_fused,
        ((batch_size, seq_len, self.num_experts_per_tok, self.num_experts)),
    )
    expert_token_count = self._maybe_shard_with_logical(
        expert_token_count,
        ("activation_batch", "activation_norm_length", None, None),
    )
    trunc_expert_mask = expert_mask * jnp.less_equal(expert_token_count, expert_capacity_per_batch)
    combined_expert_mask = jnp.sum(trunc_expert_mask, axis=2)

    softmax_probs *= combined_expert_mask

    # calculate token position in expert capacity dimension
    expert_token_position_fused = expert_mask_fused * expert_token_count_fused
    expert_token_position = jnp.reshape(
        expert_token_position_fused,
        (batch_size, seq_len, self.num_experts_per_tok, self.num_experts),
    )
    combined_expert_token_position = jnp.sum(expert_token_position, axis=2) * combined_expert_mask
    expert_token_position_in_capacity = jax.nn.one_hot(
        combined_expert_token_position,
        num_classes=expert_capacity_per_batch + 1,
        dtype=jnp.int32,
    )

    # shape of combine_mask is
    # (batch_size, seq_len, num_experts, expert_capacity_per_batch + 1),
    # and cut 0-dimension which is always 0
    combine_mask = softmax_probs[..., None] * expert_token_position_in_capacity
    combine_mask = combine_mask[..., 1:]
    dispatch_mask = combine_mask.astype(bool)

    return dispatch_mask, combine_mask

  # See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details.
  def load_balance_loss(self, top_k_indices, logits) -> jax.Array:
    \"\"\"Compute the load balance loss.\"\"\"
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
    summed_expert_mask = jnp.sum(expert_mask, axis=2)
    # Get fraction of tokens dispatched to each expert
    density = jnp.mean(summed_expert_mask, axis=1)
    # get fraction of probability allocated to each expert
    density_prob = jnp.mean(logits, axis=1)
    loss = jnp.mean(density * density_prob) * (self.num_experts**2) * self.config.load_balance_loss_weight
    return loss

  def get_einsum(
      self,
      rhs_mesh_axes: Tuple[Optional[str], ...] = (),
      einsum_name: str | None = None,
  ):
    \"\"\"Get the Einstein summation.\"\"\"

    # the check is to prevent aqteinsum as einsum op for dispatch and combine
    # einsums in ase when capacity_factor > 0
    # this is necessary to load pre-quantized weights in case of inference
    if self.config.model_call_mode == "inference" and einsum_name in (
        DISPATCH,
        COMBINE,
    ):
      return jnp.einsum

    if self.quant:

      def aqt_einsum(*args, **kwargs):  # pylint: disable=unused-argument
        # simply skip kwargs, since aqt einsum doesn't support any kwargs
        # like precision
        is_aqt = not isinstance(self.quant, quantizations.Fp8Quantization)
        kw = {"mesh_axes": rhs_mesh_axes} if is_aqt else {"dtype": self.dtype}
        return self.quant.einsum(**kw)(*args)  # pytype: disable=attribute-error

      einsum_op = aqt_einsum
    else:
      einsum_op = jnp.einsum
    return einsum_op

  def maybe_all_gather_kernel_weight_in_expert_parallelism(
      self, kernel: jax.Array, kernel_axes: Tuple[Optional[str], ...]
  ):
    \"\"\"All-gather kernel weight in expert parallelism if needed.\"\"\"
    if self.get_expert_parallelism_size() > 1:
      # This will trigger all-gather using weight_dtype
      # relax it unless really necessary in expert parallelism only
      # Otherwise compiler will handle communication automatically
      # esp. with int8 quantization, kernel will be all-gathered in int8 instead
      # of weight_dtype
      kernel = self._maybe_shard_with_logical(kernel, kernel_axes)
    return kernel

  def dense_matmul(
      self,
      inputs,
      gate_logits,
      pre_bias_logits,
      w0_kernel,
      w1_kernel,
      wo_kernel,
      w0_bias,
      w1_bias,
      wo_bias,
  ) -> tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]:
    \"\"\"Dense matrix multiplication.\"\"\"
    # gate_logits: batch, length, expert
    gate_logits = self._maybe_shard_with_logical(gate_logits, ("activation_batch_moe", "activation_length_moe", None))
    if self.config.model_name.startswith("deepseek3"):
      # pre_bias_logits is None for non-DeepSeek v3 models
      pre_bias_logits = self._maybe_shard_with_logical(
          pre_bias_logits, ("activation_batch_moe", "activation_length_moe", None)
      )
    top_k_weights, top_k_indices = self.get_topk(gate_logits, pre_bias_logits, self.rngs)
    is_llama4_decoder_layer = self.config.decoder_block == ctypes.DecoderBlockType.LLAMA4
    if is_llama4_decoder_layer:
      router_scores = jax.nn.sigmoid(top_k_weights.astype(jnp.float32)).astype(self.dtype)
      inputs = inputs * router_scores
    else:
      weights = self.reshape_and_update_weights(top_k_weights, top_k_indices)
    matmul_precision = jax.lax.Precision(self.config.matmul_precision)

    # Calculate load balance loss
    if self.config.model_call_mode != "inference":
      softmax_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
      lb_loss = (
          self.load_balance_loss(top_k_indices, softmax_probs) if self.config.load_balance_loss_weight > 0.0 else None
      )
    else:
      lb_loss = None

    # Calculate routed bias updates (loss-free)
    if self.should_update_load_balance():
      bias_updates = calculate_load_balance_updates(
          top_k_indices, self.config.num_experts, self.config.routed_bias_update_rate
      )
    else:
      bias_updates = None

    batch_size = inputs.shape[0]
    seq_len = inputs.shape[1]

    cp, sub_seq = self.get_context_partition_and_sub_seq(seq_len)

    if self.config.capacity_factor > 0:
      # token dropping if needed
      if self.config.model_call_mode != "inference":
        # TODO(b/425930949): remove this pylint by refactoring the logic here.
        dispatch_mask, combine_mask = self.generate_masks(
            top_k_indices, weights  # pylint: disable=undefined-variable,possibly-used-before-assignment
        )
        mask_axes = ("activation_batch_moe", "activation_norm_length_moe", None, None)
        dispatch_axis = (
            "activation_exp",
            "activation_batch_moe",
            None,
            "activation_embed_moe",
        )
        mlp_axis = (
            "activation_exp",
            "activation_batch_moe",
            None,
            "activation_mlp",
        )
        dispatch_eimsum = "BSM,BSEC -> EBCM"
        mlp_up_einsum = "EBCM,EMH -> EBCH"
        mlp_down_einsum = "EBCH,EHM -> EBCM"
        output_einsum = "EBCM,BSEC -> BSM"
      else:
        # TODO(b/425930507): Try replacing `softmax_probs` with padded weights
        # and verify with decode acc tests.
        softmax_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
        dispatch_mask, combine_mask = self.generate_masks_subgroup(top_k_indices, softmax_probs)
        if self.get_context_autoregressive_parallelism_size() > 0 and cp == 1:
          mask_axes = (
              "activation_norm_length_moe",
              "activation_batch_moe",
              None,
              None,
              None,
          )
          input_axis = (
              "activation_norm_length_moe",
              "activation_batch_moe",
              None,
              "activation_embed_moe",
          )
          dispatch_axis = (
              "activation_exp",
              "activation_batch_moe",
              None,
              None,
              "activation_embed_moe",
          )
          mlp_axis = (
              "activation_exp",
              "activation_batch_moe",
              None,
              None,
              "activation_mlp",
          )
        else:
          mask_axes = (
              "activation_batch_moe",
              "activation_norm_length_moe",
              None,
              None,
              None,
          )
          input_axis = (
              "activation_batch_moe",
              "activation_norm_length_moe",
              None,
              "activation_embed_moe",
          )
          dispatch_axis = (
              "activation_exp",
              "activation_batch_moe",
              None,
              None,
              "activation_embed_moe",
          )
          mlp_axis = (
              "activation_exp",
              "activation_batch_moe",
              None,
              None,
              "activation_mlp",
          )
        dispatch_eimsum = "BNSM,BNSEC -> EBNCM"
        mlp_up_einsum = "EBNCM,EMH -> EBNCH"
        mlp_down_einsum = "EBNCH,EHM -> EBNCM"
        output_einsum = "EBNCM,BNSEC -> BNSM"

        inputs = jnp.reshape(inputs, (batch_size, cp, sub_seq, inputs.shape[2]))
        inputs = self._maybe_shard_with_logical(inputs, input_axis)

      dispatch_mask = self._maybe_shard_with_logical(dispatch_mask, mask_axes)
      combine_mask = self._maybe_shard_with_logical(combine_mask, mask_axes)

      with jax.named_scope("dispatch"):
        # only cp during prefill
        dispatch = self.get_einsum(rhs_mesh_axes=mask_axes, einsum_name=DISPATCH)(
            dispatch_eimsum, inputs, dispatch_mask, precision=matmul_precision
        )
        if cp > 1:
          dispatch = self._maybe_shard_with_logical(
              dispatch,
              (
                  None,
                  "activation_batch_moe",
                  "activation_norm_length_moe",
                  None,
                  "activation_embed_moe",
              ),
          )
        dispatch = self._maybe_shard_with_logical(
            dispatch,
            dispatch_axis,
        )
      with jax.named_scope("wi_0"):
        w0_kernel_axes = ("exp", None, "mlp")
        w0_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(w0_kernel, w0_kernel_axes)
        layer_w0 = self.get_einsum(rhs_mesh_axes=w0_kernel_axes)(
            mlp_up_einsum, dispatch, w0_kernel, precision=matmul_precision
        )
        if self.config.mlp_bias:
          w0_bias = w0_bias[:, None, None, :]
          layer_w0 = layer_w0 + w0_bias

        if self.config.activations_in_float32:
          layer_w0 = layer_w0.astype(jnp.float32)
        layer_w0 = self._maybe_shard_with_logical(
            layer_w0,
            mlp_axis,
        )
        layer_w0 = adc.checkpoint_name(layer_w0, "moe_mlpwi_0")
      with jax.named_scope("wi_1"):
        w1_kernel_axes = ("exp", None, "mlp")
        w1_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(w1_kernel, w1_kernel_axes)
        layer_w1 = self.get_einsum(rhs_mesh_axes=w1_kernel_axes)(
            mlp_up_einsum, dispatch, w1_kernel, precision=matmul_precision
        )
        if self.config.mlp_bias:
          w1_bias = w1_bias[:, None, None, :]
          layer_w1 = layer_w1 + w1_bias
        if self.config.activations_in_float32:
          layer_w1 = layer_w1.astype(jnp.float32)
        layer_w1 = self._maybe_shard_with_logical(
            layer_w1,
            mlp_axis,
        )
        layer_w1 = adc.checkpoint_name(layer_w1, "moe_mlpwi_1")
      layer_multiply = self.apply_ffn_activation(layer_w0, layer_w1)
      with jax.named_scope("wo"):
        wo_kernel_axes = ("exp", "mlp", None)
        wo_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(wo_kernel, wo_kernel_axes)
        intermediate_layer = self.get_einsum(rhs_mesh_axes=wo_kernel_axes)(
            mlp_down_einsum,
            layer_multiply,
            wo_kernel,
            precision=matmul_precision,
        )
        if self.config.mlp_bias:
          wo_bias = wo_bias[:, None, None, :]
          intermediate_layer = intermediate_layer + wo_bias
        if self.config.activations_in_float32:
          intermediate_layer = intermediate_layer.astype(jnp.float32)
        if self.config.model_call_mode != "inference":
          intermediate_layer = self._maybe_shard_with_logical(
              intermediate_layer,
              (
                  "activation_exp",
                  "activation_batch_moe",
                  None,
                  "activation_embed_moe",
              ),
          )
        intermediate_layer = adc.checkpoint_name(intermediate_layer, "moe_mlpwo")
      with jax.named_scope("combine"):
        # Matmul & element wise operation
        output = self.get_einsum(rhs_mesh_axes=mask_axes, einsum_name=COMBINE)(
            output_einsum,
            intermediate_layer,
            combine_mask,
            precision=matmul_precision,
        )
        if output.ndim == 4:
          output = jnp.reshape(
              output,
              (
                  output.shape[0],
                  output.shape[1] * output.shape[2],
                  output.shape[3],
              ),
          )
      return output, lb_loss, bias_updates
    else:
      inputs = self._maybe_shard_with_logical(
          inputs, ("activation_batch_moe", "activation_norm_length_moe", "activation_embed_moe")
      )
      with jax.named_scope("wi_0"):
        layer_w0 = self.get_einsum(rhs_mesh_axes=self.wi_kernel_axes)(
            "BSM,EMH -> BSEH", inputs, w0_kernel, precision=matmul_precision
        )
        if self.config.mlp_bias:
          layer_w0 = layer_w0 + w0_bias[None, None, :, :]
        if self.config.activations_in_float32:
          layer_w0 = layer_w0.astype(jnp.float32)
        layer_w0 = adc.checkpoint_name(layer_w0, "moe_mlpwi_0")
      with jax.named_scope("wi_1"):
        layer_w1 = self.get_einsum(rhs_mesh_axes=self.wi_kernel_axes)(
            "BSM,EMH -> BSEH", inputs, w1_kernel, precision=matmul_precision
        )
        if self.config.mlp_bias:
          layer_w1 = layer_w1 + w1_bias[None, None, :, :]
        if self.config.activations_in_float32:
          layer_w1 = layer_w1.astype(jnp.float32)
        layer_w1 = adc.checkpoint_name(layer_w1, "moe_mlpwi_1")
      layer_multiply = self.apply_ffn_activation(layer_w0, layer_w1)

      with jax.named_scope("wo"):
        intermediate_layer = self.get_einsum(rhs_mesh_axes=self.wo_kernel_axes)(
            "BSEH,EHM -> BSEM",
            layer_multiply,
            wo_kernel,
            precision=matmul_precision,
        )
        if self.config.mlp_bias:
          intermediate_layer = intermediate_layer + wo_bias[None, None, :, :]
        if self.config.activations_in_float32:
          intermediate_layer = intermediate_layer.astype(jnp.float32)
        intermediate_layer = adc.checkpoint_name(intermediate_layer, "moe_mlpwo")
      with jax.named_scope("weight_sum"):
        if is_llama4_decoder_layer:
          weights = self.reshape_and_update_weights(jnp.ones_like(top_k_weights), top_k_indices)
        if self.config.float32_weight_sum:
          intermediate_layer = intermediate_layer.astype(jnp.float32)
          weights = weights.astype(jnp.float32)
        # cast to f32 for sum up in einsum op
        output = jnp.einsum(
            "BSEM,BSE -> BSM",
            intermediate_layer,
            weights,
            precision=matmul_precision,
        ).astype(self.dtype)
      return output, lb_loss, bias_updates

  def fused_moe_matmul(
      self,
      inputs,
      gate_logits,
      wo_kernel,
      w0_kernel=None,
      w1_kernel=None,
      fused_kernel=None,
  ) -> tuple[jax.Array, None, None]:
    \"\"\"Fused MoE via tpu_inference fused_moe_func (vllm_rpa path only).

    fused_moe_func handles routing, GMM, and weighted combination internally.
    It does not compute lb_loss or bias_updates (inference-only).
    \"\"\"
    try:
      # pylint: disable=import-outside-toplevel
      # pytype: disable=import-error
      from tpu_inference.layers.common.fused_moe_gmm import fused_moe_func
    except ImportError as e:
      raise ImportError("fused_moe_matmul requires the tpu-inference package.") from e

    # Reshape 3D [B, S, D] -> 2D [T, D] (fused_moe_func expects 2D input)
    batch_size, seq_len, emb_dim = inputs.shape
    hidden_states = jnp.reshape(inputs, (batch_size * seq_len, emb_dim))
    gating_output = jnp.reshape(gate_logits, (batch_size * seq_len, self.num_experts))

    # Concatenate gate and up projections: [E, D, H] + [E, D, H] -> [E, D, 2H]
    # fused_moe_func splits this internally: gate=w1[..., :H], up=w1[..., H:]
    if fused_kernel is None:
      fused_kernel = jnp.concatenate([w0_kernel, w1_kernel], axis=-1)

    # Use expert parallelism if the expert axis has size > 1
    use_ep = self.get_expert_parallelism_size() > 1

    # Map MaxText config fields to fused_moe_func args
    activation = self.config.mlp_activations[0]  # e.g. "silu"
    scoring_fn = self.config.routed_score_func if self.config.routed_score_func else "softmax"

    # Check if the model architecture intrinsically renormalizes weights
    renormalize = self.config.norm_topk_prob or (
        self.config.decoder_block not in (ctypes.DecoderBlockType.LLAMA4, ctypes.DecoderBlockType.GEMMA4)
    )

    output_2d = fused_moe_func(
        hidden_states=hidden_states,
        w1=fused_kernel,
        w2=wo_kernel,
        w1_scale=None,
        w2_scale=None,
        w1_bias=None,
        w2_bias=None,
        gating_output=gating_output,
        topk=self.num_experts_per_tok,
        renormalize=renormalize,
        mesh=self.mesh,
        use_ep=use_ep,
        activation=activation,
        scoring_fn=scoring_fn,
        sc_kernel_threshold=16777216,
        sc_kernel_col_chunk_size=1024,
    )

    # Reshape output 2D [T, D] -> 3D [B, S, D]
    output = jnp.reshape(output_2d, (batch_size, seq_len, emb_dim))
    return output, None, None

  def retrieve_quantized_weight(
      self,
      inputs,
      gate_logits,
      pre_bias_logits,
      w0_kernel,
      w1_kernel,
      wo_kernel,
      w0_bias,
      w1_bias,
      wo_bias,
  ) -> tuple[aqt.QTensor, aqt.QTensor, aqt.QTensor]:
    \"\"\"Retrieve quantized weights.\"\"\"
    # This is called only during tracing. This is to invoke creation of
    # quantized tensor inside AqtEinsum.  After jit, this will become no-op and
    # will not affect performance.
    _ = self.dense_matmul(
        inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel, w0_bias, w1_bias, wo_bias
    )

    w0_kernel = self.variables["aqt"]["AqtEinsum_0"]["AqtDotGeneral_0"]["qrhs"]["frozen"]
    w1_kernel = self.variables["aqt"]["AqtEinsum_1"]["AqtDotGeneral_0"]["qrhs"]["frozen"]
    wo_kernel = self.variables["aqt"]["AqtEinsum_2"]["AqtDotGeneral_0"]["qrhs"]["frozen"]

    w0_kernel = max_utils.unbox_logicallypartioned(w0_kernel)
    w1_kernel = max_utils.unbox_logicallypartioned(w1_kernel)
    wo_kernel = max_utils.unbox_logicallypartioned(wo_kernel)
    return w0_kernel, w1_kernel, wo_kernel

  def __call__(
      self, inputs: jax.Array, gate_inputs: jax.Array | None = None, out_sharding: NamedSharding | None = None
  ) -> tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]:
    cfg = self.config
    inputs = inputs.astype(cfg.dtype)
    gate_dtype = jnp.float32 if cfg.float32_gate_logits else cfg.dtype
    routing_inputs = inputs if gate_inputs is None else gate_inputs.astype(gate_dtype)
    gate_logits, pre_bias_logits = self.gate(routing_inputs)

    wo_kernel = jnp.asarray(self.wo[...], self.dtype)

    fused_kernel = None
    w0_kernel = None
    w1_kernel = None
    if cfg.prefuse_moe_weights and cfg.attention == "vllm_rpa":
      fused_kernel = jnp.asarray(self.wi[...], self.dtype)
    else:
      w0_kernel = jnp.asarray(self.wi_0[...], self.dtype)
      w1_kernel = jnp.asarray(self.wi_1[...], self.dtype)

    if self.per_expert_scale is not None:
      wo_kernel = wo_kernel * jnp.asarray(self.per_expert_scale[...], self.dtype)[:, None, None]

    if self.wi_0_sparsity_module is not None:
      _, w0_kernel = self.wi_0_sparsity_module(jnp.zeros_like(w0_kernel), w0_kernel)
      _, w1_kernel = self.wi_1_sparsity_module(jnp.zeros_like(w1_kernel), w1_kernel)
      _, wo_kernel = self.wo_sparsity_module(jnp.zeros_like(wo_kernel), wo_kernel)
    if cfg.mlp_bias:
      w0_bias = jnp.asarray(self.wi_0_bias[...], self.dtype)
      w1_bias = jnp.asarray(self.wi_1_bias[...], self.dtype)
      wo_bias = jnp.asarray(self.wo_bias[...], self.dtype)
    else:
      w0_bias, w1_bias, wo_bias = None, None, None

    # vllm_rpa codepath uses fused_moe_func from tpu_inference for optimized inference.
    if cfg.attention == "vllm_rpa":
      output, lb_loss, bias_updates = self.fused_moe_matmul(
          inputs, gate_logits, wo_kernel, w0_kernel=w0_kernel, w1_kernel=w1_kernel, fused_kernel=fused_kernel
      )
    elif cfg.sparse_matmul:
      if quantizations.in_serve_mode(self.quant):
        w0_kernel, w1_kernel, wo_kernel = self.retrieve_quantized_weight(
            inputs,
            gate_logits,
            pre_bias_logits,
            w0_kernel,
            w1_kernel,
            wo_kernel,
            w0_bias,
            w1_bias,
            wo_bias,
        )
      output, lb_loss, bias_updates = self.sparse_matmul(
          inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel, w0_bias, w1_bias, wo_bias
      )
    else:
      output, lb_loss, bias_updates = self.dense_matmul(
          inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel, w0_bias, w1_bias, wo_bias
      )
    return output, lb_loss, bias_updates


class RoutedAndSharedMoE(nnx.Module):
  \"\"\"Implements a block which combines shared and routed experts.\"\"\"

  def __init__(
      self,
      config: ctypes.Config,
      mesh: jax.sharding.Mesh,
      kernel_init: NdInitializer,
      kernel_axes: Tuple[Optional[str], ...],
      rngs: nnx.Rngs,
      weight_dtype: ctypes.DType = jnp.float32,
      dtype: ctypes.DType = jnp.float32,
      quant: Optional[quantizations.AqtQuantization] = None,
  ):
    \"\"\"Initializes the RoutedAndSharedMoE module.

    Attributes:
      config: The main config setting.
      mesh: Mesh, device mesh.
      kernel_init: The initializer function for the kernel weight matrix.
      kernel_axes: A tuple of logical axis names for partitioning the kernel.
      rngs: An `nnx.Rngs` object used for initializing parameters.
      weight_dtype: The data type of the kernel weights.
      dtype: The data type for the computation.
      quant: The quantization configuration. If None, no quantization is applied.
    \"\"\"
    self.config = config
    self.mesh = mesh
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.quant = quant
    self.rngs = rngs
    self.moe_expert_input_dim = (
        self.config.emb_dim if self.config.moe_expert_input_dim <= 0 else self.config.moe_expert_input_dim
    )

    # NOTE: the name MoeBlock_0 is to ensure reverse compatibility with
    # existing checkpoints for routed experts.
    self.MoeBlock_0 = RoutedMoE(
        config=self.config,
        num_experts=self.config.num_experts,
        num_experts_per_tok=self.config.num_experts_per_tok,
        mesh=self.mesh,
        kernel_init=self.kernel_init,
        kernel_axes=("embed_moe", None),
        intermediate_dim=self.config.moe_mlp_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        quant=self.quant,
        rngs=self.rngs,
    )

    shared_expert_mlp_dim = (
        self.config.mlp_dim if self.config.decoder_block == ctypes.DecoderBlockType.GEMMA4 else self.config.moe_mlp_dim
    )
    self.shared_experts = linears.MlpBlock(
        mesh=self.mesh,
        in_features=self.moe_expert_input_dim,
        intermediate_dim=self.config.shared_experts * shared_expert_mlp_dim,
        activations=self.config.mlp_activations,
        kernel_init=self.kernel_init,
        intermediate_dropout_rate=self.config.dropout_rate,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        config=self.config,
        quant=self.quant,
        rngs=self.rngs,
    )

  @property
  def routed_moe(self):
    return self.MoeBlock_0

  def __call__(
      self,
      inputs: jax.Array,
      original_inputs: jax.Array | None = None,
      gate_inputs: jax.Array | None = None,
      intermediate_sharding: NamedSharding | None = None,
      out_sharding: NamedSharding | None = None,
  ) -> tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]:
    routed_experts, load_balance_loss, moe_bias_updates = self.routed_moe(
        inputs, gate_inputs=gate_inputs, out_sharding=out_sharding
    )
    shared_experts = self.shared_experts(inputs, intermediate_sharding=intermediate_sharding, out_sharding=out_sharding)
    return routed_experts + shared_experts, load_balance_loss, moe_bias_updates


def get_gate_logit(
    inputs_shape: tuple[int, ...],
    out_features_shape: Union[Iterable[int], int],
    model_name: str,
    axis: Union[Iterable[int], int] = -1,
    weight_dtype: ctypes.DType = jnp.float32,
    dtype: ctypes.DType = jnp.float32,
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
    kernel_axes: Tuple[Optional[str], ...] = (),
    use_bias: bool = False,
    score_func: str = "",
    quant: Optional[quantizations.AqtQuantization] = None,
    matmul_precision: str = "default",
    name: Optional[str] = None,
):
  \"\"\"Creates a GateLogit Linen module.\"\"\"

  axis = linears.canonicalize_tuple(axis)
  in_features_shape = tuple(inputs_shape[ax] for ax in linears.normalize_axes(axis, len(inputs_shape)))

  module = nnx_wrappers.to_linen(
      GateLogit,
      in_features_shape=in_features_shape,
      out_features_shape=out_features_shape,
      model_name=model_name,
      axis=axis,
      weight_dtype=weight_dtype,
      dtype=dtype,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes,
      use_bias=use_bias,
      score_func=score_func,
      quant=quant,
      matmul_precision=matmul_precision,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )
  return module


def get_routed_moe(
    config: ctypes.Config,
    num_experts: int,
    num_experts_per_tok: int,
    mesh: jax.sharding.Mesh,
    kernel_init: NdInitializer,
    kernel_axes: Tuple[Optional[str], ...],
    intermediate_dim: int = 2048,
    weight_dtype: ctypes.DType = jnp.float32,
    dtype: ctypes.DType = jnp.float32,
    quant: Optional[quantizations.AqtQuantization] = None,
    name: Optional[str] = None,
):
  \"\"\"Creates a RoutedMoE Linen module.\"\"\"

  module = nnx_wrappers.to_linen(
      RoutedMoE,
      config=config,
      num_experts=num_experts,
      num_experts_per_tok=num_experts_per_tok,
      mesh=mesh,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes,
      intermediate_dim=intermediate_dim,
      weight_dtype=weight_dtype,
      dtype=dtype,
      quant=quant,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )
  return module


def get_routed_and_shared_moe(
    config: ctypes.Config,
    mesh: jax.sharding.Mesh,
    kernel_init: NdInitializer,
    kernel_axes: Tuple[Optional[str], ...],
    weight_dtype: ctypes.DType = jnp.float32,
    dtype: ctypes.DType = jnp.float32,
    quant: Optional[quantizations.AqtQuantization] = None,
    name: Optional[str] = None,
):
  \"\"\"Creates a RoutedAndSharedMoE Linen module.\"\"\"

  module = nnx_wrappers.to_linen(
      RoutedAndSharedMoE,
      config=config,
      mesh=mesh,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes,
      weight_dtype=weight_dtype,
      dtype=dtype,
      quant=quant,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )
  return module
\n"""


# File: src/maxtext/layers/decoders.py (commit 313890777)
DECODERS_STACK_RAW = """\n# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

\"\"\"Module for decoder layers\"\"\"
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

import functools
from typing import Any
import warnings

from flax import linen as nn
from flax import nnx
from flax.linen.partitioning import ScanIn
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import Config, DecoderBlockType, ShardMode
from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.inference import page_manager
from maxtext.layers import linears
from maxtext.layers import mhc
from maxtext.layers import normalizations
from maxtext.layers import pipeline
from maxtext.layers import quantizations
from maxtext.layers.attentions import attention_as_linen
from maxtext.layers.embeddings import attend_on_embedding, embed_as_linen, positional_embedding_as_linen
from maxtext.layers.normalizations import rms_norm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.models import (
    deepseek,
    deepseek_batchsplit,
    deepseek_batchsplit_fp8,
    gemma,
    gemma2,
    gemma3,
    gemma4,
    gpt3,
    gpt_oss,
    llama2,
    llama4,
    mistral,
    mixtral,
    olmo3,
    qwen2,
    qwen3,
    qwen3_custom,
    simple_layer,
)
from maxtext.multimodal import utils as mm_utils
from maxtext.utils.sharding import create_sharding
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding

# ------------------------------------------------------------------------------
# The network: Decoder Definitions
# ------------------------------------------------------------------------------


class DecoderLayer(nn.Module):
  \"\"\"
  Transformer decoder layer that attends to the encoder.
  This is the core, reusable building block for both the main model's
  decoder stack and the auxiliary MTP layers.
  \"\"\"

  config: Config
  mesh: Mesh
  model_mode: str
  quant: None | Quant = None

  @nn.compact
  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot: None | int = None,
      page_state: None | page_manager.PageState = None,
      kv_cache: jax.Array | None = None,
      attention_metadata: dict[str, Any] | None = None,
  ):
    cfg = self.config
    mesh = self.mesh
    _maybe_shard_with_logical = functools.partial(
        sharding.maybe_shard_with_logical,
        mesh=mesh,
        shard_mode=cfg.shard_mode,
        debug_sharding=cfg.debug_sharding,
    )

    if self.model_mode == MODEL_MODE_PREFILL:
      logical_axis_names = ("activation_batch", "prefill_activation_length", "activation_embed")
    else:
      logical_axis_names = ("activation_batch", "activation_length", "activation_embed")

    if model_mode == MODEL_MODE_PREFILL:
      inputs = _maybe_shard_with_logical(inputs, logical_axis_names)
    else:
      inputs = _maybe_shard_with_logical(inputs, logical_axis_names)

    inputs = checkpoint_name(inputs, "decoder_layer_input")
    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    lnx = rms_norm(
        num_features=inputs.shape[-1],
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="pre_self_attention_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )(inputs)
    if model_mode == MODEL_MODE_PREFILL:
      lnx = _maybe_shard_with_logical(lnx, logical_axis_names)
    else:
      lnx = _maybe_shard_with_logical(lnx, logical_axis_names)

    attention_layer = attention_as_linen(
        config=self.config,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention,
        inputs_q_shape=lnx.shape,
        inputs_kv_shape=lnx.shape,
        mesh=mesh,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        dropout_rate=cfg.dropout_rate,
        name="self_attention",
        float32_qk_product=cfg.float32_qk_product,
        float32_logits=cfg.float32_logits,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(cfg),
        prefill_cache_axis_order=tuple(map(int, cfg.prefill_cache_axis_order.split(","))),
        ar_cache_axis_order=tuple(map(int, cfg.ar_cache_axis_order.split(","))),
        compute_axis_order=tuple(map(int, cfg.compute_axis_order.split(","))),
        reshape_q=cfg.reshape_q,
        use_mrope=cfg.use_mrope,
        mrope_section=cfg.mrope_section,
        share_kv_projections=cfg.share_kv_projections,
        model_mode=model_mode,
    )

    attention_lnx, kv_cache = attention_layer(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )

    if model_mode == MODEL_MODE_PREFILL:
      attention_lnx = _maybe_shard_with_logical(attention_lnx, logical_axis_names)
    else:
      attention_lnx = _maybe_shard_with_logical(attention_lnx, logical_axis_names)

    # MLP block.
    mlp_lnx = linears.mlp_block(
        in_features=lnx.shape[-1],
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="mlp",
        model_mode=model_mode,
        config=cfg,
        quant=self.quant,
        mesh=self.mesh,
    )(lnx, deterministic=deterministic)
    if model_mode == MODEL_MODE_PREFILL:
      mlp_lnx = _maybe_shard_with_logical(mlp_lnx, logical_axis_names)
    else:
      mlp_lnx = _maybe_shard_with_logical(mlp_lnx, logical_axis_names)

    next_layer_addition = mlp_lnx + attention_lnx

    next_layer_addition_dropped_out = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(
        next_layer_addition, deterministic=deterministic
    )

    layer_output = next_layer_addition_dropped_out + inputs
    if model_mode == MODEL_MODE_PREFILL:
      layer_output = _maybe_shard_with_logical(
          layer_output,
          logical_axis_names,
      )
    else:
      layer_output = _maybe_shard_with_logical(
          layer_output,
          logical_axis_names,
      )

    if cfg.record_internal_nn_metrics:
      self.sow("intermediates", "activation_mean", jnp.mean(layer_output))
      self.sow("intermediates", "activation_stdev", jnp.std(layer_output))
      self.sow(
          "intermediates",
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output, kv_cache


class SequentialBlockDecoderLayers(nn.Module):
  \"\"\"Sequential unscanned series of decoder layers.\"\"\"

  decoder_layer: Any
  num_decoder_layers: int
  config: Config
  mesh: Mesh
  quant: Quant
  model_mode: str

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids,
      decoder_positions,
      deterministic: bool,
      model_mode,
      slot: None | int = None,
      page_state: None | page_manager.PageState = None,
  ) -> jnp.ndarray:
    for lyr in range(self.num_decoder_layers):
      inputs = self.decoder_layer(
          config=self.config, mesh=self.mesh, name=f"layers_{lyr}", quant=self.quant, model_mode=model_mode
      )(
          inputs,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          slot=slot,
          page_state=page_state,
      )
      if self.config.scan_layers:
        inputs = inputs[0]  #  When scan_layers is True the decoder layers return (outputs, None).
    if self.config.scan_layers:
      return inputs, None  # pytype: disable=bad-return-type
    else:
      return inputs


def deepstack_process(hidden_states, bidirectional_mask, visual_embeds):
  \"\"\"Process deepstack visual embeddings by adding them to hidden states at visual token positions.

  Args:
    hidden_states: [batch, seq_len, hidden_dim] decoder hidden states
    bidirectional_mask: [batch, seq_len] boolean mask marking visual token positions
    visual_embeds: [batch, num_visual_tokens, hidden_dim] visual features from encoder layer

  Returns:
    Updated hidden_states with visual features added at visual positions
  \"\"\"
  # Expand mask to [batch, seq_len, 1] for broadcasting
  mask_expanded = bidirectional_mask[:, :, jnp.newaxis]
  # Use cumsum to map each True position in mask to its index in visual_embeds
  visual_token_idx = jnp.cumsum(bidirectional_mask, axis=1) - 1  # [batch, seq_len], 0-indexed

  # Gather visual tokens: for each position, get the corresponding visual token
  batch_idx = jnp.arange(hidden_states.shape[0])[:, jnp.newaxis]  # [batch, 1]
  visual_embeds_scattered = visual_embeds[batch_idx, visual_token_idx, :]  # [batch, seq_len, hidden]

  # Only add where mask is True: hidden_states += visual_embeds * mask
  hidden_states = hidden_states + visual_embeds_scattered * mask_expanded
  return hidden_states


class Decoder(nn.Module):
  \"\"\"A stack of decoder layers as a part of an encoder-decoder architecture.\"\"\"

  config: Config
  mesh: Mesh
  quant: None | Quant = None
  model_mode: str = MODEL_MODE_TRAIN

  def setup(self):
    \"\"\"Initialize decoder layer.\"\"\"
    self.decoder_layer = self.get_decoder_layers()
    self.norm_layer = self.get_norm_layer(num_features=self.config.emb_dim)
    if self.config.using_pipeline_parallelism:
      pipeline_stage_module = self.get_pipeline_stage_module(self.decoder_layer)
      remat_policy = self.get_remat_policy()
      self.pipeline_module = pipeline.create_pipeline(
          config=self.config, mesh=self.mesh, layers=pipeline_stage_module, remat_policy=remat_policy
      )

  def minimal_policy(self, with_context=False, with_quantization=False):
    \"\"\"Helper for creating minimal checkpoint policies.\"\"\"
    names = [
        "query_proj",
        "value_proj",
        "key_proj",
        "qkv_proj",
        "out_proj",
        "mlpwi_0",
        "mlpwi_1",
        "mlpwi",
        "mlpwo",
    ]
    if with_context:
      names.append("context")
    if with_quantization:
      names.append("quantization")
    return jax.checkpoint_policies.save_only_these_names(*names)

  def get_remat_policy(self):
    \"\"\"Get remat policy\"\"\"
    policy = None
    cfg = self.config
    if cfg.remat_policy != "none":
      if cfg.remat_policy in ("minimal_with_context", "minimal_flash"):
        # save all
        if cfg.remat_policy == "minimal_flash":
          max_logging.log("WARNING: 'minimal_flash' will be deprecated soon, please use 'minimal_with_context' instead.")
          max_logging.log("WARNING: 'minimal_flash' will be deprecated soon, please use 'minimal_with_context' instead.")
        policy = self.minimal_policy(with_context=True)
      elif cfg.remat_policy == "minimal":
        # save all except context
        policy = self.minimal_policy()
      elif cfg.remat_policy == "minimal_with_quantization":
        if cfg.scan_layers:
          warnings.warn(
              "Scan layers can introduce overhead to checkpointed values that in some configurations is slower"
              "than not checkpointing at all. If you are using scan layers, benchmark with and without quantization "
              "checkpointing in your workflow to see which is faster. Without scan layers, checkpointing quantizations is "
              "beneficial for performance."
          )
        policy = self.minimal_policy(with_context=False, with_quantization=True)
      elif cfg.remat_policy == "minimal_with_context_and_quantization":
        if cfg.scan_layers:
          warnings.warn(
              "Scan layers can introduce overhead to checkpointed values that in some configurations is slower"
              "than not checkpointing at all. If you are using scan layers, benchmark with and without quantization "
              "checkpointing in your workflow to see which is faster. Without scan layers, checkpointing quantizations is "
              "beneficial for performance."
          )
        policy = self.minimal_policy(with_context=True, with_quantization=True)
      elif cfg.remat_policy == "save_dot_with_context_except_mlp":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj",
            "value_proj",
            "key_proj",
            "qkv_proj",
            "context",
            "out_proj",
        )
      elif cfg.remat_policy == "save_dot_except_mlpwi":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj",
            "value_proj",
            "key_proj",
            "qkv_proj",
            "out_proj",
            "mlpwo",
        )
      elif cfg.remat_policy == "save_dot_except_mlp":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj",
            "value_proj",
            "key_proj",
            "qkv_proj",
            "out_proj",
        )
      elif cfg.remat_policy == "save_qkv_proj":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj",
            "value_proj",
            "key_proj",
            "qkv_proj",
        )
      elif cfg.remat_policy == "qkv_proj_offloaded":
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=[],
            names_which_can_be_offloaded=["query_proj", "value_proj", "key_proj"],
            offload_src="device",
            offload_dst="pinned_host",
        )
      elif cfg.remat_policy == "minimal_offloaded":
        # offload all except context
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=[],
            names_which_can_be_offloaded=[
                "query_proj",
                "value_proj",
                "key_proj",
                "qkv_proj",
                "out_proj",
                "mlpwi_0",
                "mlpwi_1",
                "mlpwi",
                "mlpwo",
            ],
            offload_src="device",
            offload_dst="pinned_host",
        )
      elif cfg.remat_policy == "custom":
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=cfg.tensors_on_device,
            names_which_can_be_offloaded=cfg.tensors_to_offload,
            offload_src="device",
            offload_dst="pinned_host",
        )
      elif cfg.remat_policy == "save_out_proj":
        policy = jax.checkpoint_policies.save_only_these_names(
            "out_proj",
        )
      else:
        assert cfg.remat_policy == "full", "Remat policy needs to be on list of remat policies"
        policy = None
    return policy

  def get_decoder_layers(self):
    \"\"\"Retrieves a list of decoder layer classes based on the `decoder_block` config.

    Returns:
        A list containing one or more `nn.Module` classes for the decoder.
    \"\"\"
    match self.config.decoder_block:
      case DecoderBlockType.DEFAULT:
        return [DecoderLayer]
      case DecoderBlockType.LLAMA2:
        return [llama2.LlamaDecoderLayerToLinen]
      case DecoderBlockType.LLAMA2LTI:
        return [llama2.LlamaLTIDecoderLayerToLinen]
      case DecoderBlockType.MISTRAL:
        # TODO(ranran): update to Mistral with sliding window attention
        return [mistral.MistralDecoderLayerToLinen]
      case DecoderBlockType.MIXTRAL:
        return [mixtral.MixtralDecoderLayerToLinen]
      case DecoderBlockType.DEEPSEEK:
        return [
            deepseek.DeepSeekDenseLayerToLinen,
            deepseek.DeepSeekMoELayerToLinen,
        ]
      case DecoderBlockType.GEMMA:
        return [gemma.GemmaDecoderLayerToLinen]
      case DecoderBlockType.GEMMA2:
        return [gemma2.Gemma2DecoderLayerToLinen]
      case DecoderBlockType.GEMMA3:
        return [gemma3.Gemma3DecoderLayerToLinen]
      case DecoderBlockType.GEMMA4:
        return [gemma4.Gemma4ScannableBlockToLinen] if self.config.scan_layers else [gemma4.Gemma4DecoderLayerToLinen]
      case DecoderBlockType.GPT3:
        return [gpt3.Gpt3DecoderLayerToLinen]
      case DecoderBlockType.GPT_OSS:
        return [gpt_oss.GptOssScannableBlockToLinen] if self.config.scan_layers else [gpt_oss.GptOssDecoderLayerToLinen]
      case DecoderBlockType.QWEN2:
        return [qwen2.Qwen2DecoderLayerToLinen]
      case DecoderBlockType.QWEN3:
        return [qwen3.Qwen3DecoderLayerToLinen]
      case DecoderBlockType.QWEN3_MOE:
        return [qwen3.Qwen3MoeDecoderLayerToLinen]
      case DecoderBlockType.QWEN3_CUSTOM_MOE:
        return [qwen3_custom.Qwen3CustomMoeDecoderLayerToLinen]
      case DecoderBlockType.QWEN3_NEXT:
        return [qwen3.Qwen3NextScannableBlockToLinen] if self.config.scan_layers else [qwen3.Qwen3NextDecoderLayerToLinen]
      case DecoderBlockType.SIMPLE:
        return [simple_layer.SimpleDecoderLayerToLinen]
      case DecoderBlockType.SIMPLE_MLP:
        return [simple_layer.SimpleMlpDecoderLayerToLinen]
      case DecoderBlockType.LLAMA4:
        return [llama4.Llama4ScannableBlockToLinen] if self.config.scan_layers else [llama4.Llama4DecoderLayerToLinen]
      case DecoderBlockType.OLMO3:
        return [olmo3.Olmo3ScannableBlockToLinen] if self.config.scan_layers else [olmo3.Olmo3DecoderLayerToLinen]

      case _:
        # Default case to handle any unknown decoder block types.
        raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block.value=}")

  def set_remat_policy(self, block_layers, policy):
    \"\"\"Set remat policy\"\"\"
    RemattedBlockLayers = []
    for block_layer in block_layers:
      if self.config.parameter_memory_host_offload:
        # Define parameter movement with mesh-based sharding
        def move_to_device(variables):
          \"\"\"Move parameters to device with proper sharding.\"\"\"

          def map_fn(path, value):
            max_logging.log(f"models.py: Moving parameter {path} to device")
            return jax.device_put(value, max_utils.device_space())

          return jax.tree_util.tree_map_with_path(map_fn, variables)

        # Transform layer class before remat
        block_layer = nn.map_variables(block_layer, ["params"], move_to_device, mutable=True)

      # Apply remat policy to layer
      layer = nn.remat(
          block_layer,
          prevent_cse=maxtext_utils.should_prevent_cse_in_remat(self.config),
          policy=policy,
          static_argnums=(4, 5),  # Deterministic and model mode are static arguments.
      )
      RemattedBlockLayers.append(layer)
    return RemattedBlockLayers

  def get_norm_layer(self, num_features: int):
    \"\"\"get normalization layer (return type inherits from nn.Module)\"\"\"
    if self.config.decoder_block in (
        DecoderBlockType.DEFAULT,
        DecoderBlockType.LLAMA2,
        DecoderBlockType.MISTRAL,
        DecoderBlockType.MIXTRAL,
        DecoderBlockType.DEEPSEEK,
        DecoderBlockType.GEMMA,
        DecoderBlockType.GEMMA2,
        DecoderBlockType.GEMMA3,
        DecoderBlockType.GEMMA4,
        DecoderBlockType.QWEN2,
        DecoderBlockType.QWEN3,
        DecoderBlockType.QWEN3_MOE,
        DecoderBlockType.QWEN3_CUSTOM_MOE,
        DecoderBlockType.GPT_OSS,
        DecoderBlockType.SIMPLE,
        DecoderBlockType.SIMPLE_MLP,
        DecoderBlockType.LLAMA4,
        DecoderBlockType.OLMO3,
        DecoderBlockType.LLAMA2LTI,
    ):
      return functools.partial(rms_norm, num_features=num_features, shard_mode=self.config.shard_mode)
    elif self.config.decoder_block == DecoderBlockType.GPT3:
      return functools.partial(gpt3.gpt3_layer_norm, num_features=num_features, reductions_in_fp32=False, use_bias=True)
    elif self.config.decoder_block == DecoderBlockType.QWEN3_NEXT:
      return functools.partial(
          normalizations.Qwen3NextRMSNormLinen, num_features=num_features, shard_mode=self.config.shard_mode
      )
    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block.value=}")

  def scan_decoder_layers(self, cfg, decoder_layer, length, metadata_axis_name, mesh, in_axes_tuple, **kwargs):
    \"\"\"scan decoder layers, calls `flax.linen.transforms.scan`\"\"\"
    initializing = self.is_mutable_collection("params")
    params_spec = cfg.param_scan_axis if initializing else ScanIn(cfg.param_scan_axis)
    cache_spec = 0
    scan_fn = nn.scan(
        decoder_layer,
        variable_axes={
            "params": params_spec,
            "cache": cache_spec,
            "intermediates": 0,
            "aqt": 0,
            "batch_stats": 0,
            "_overwrite_with_gradient": 0,
        },
        split_rngs={
            "params": True,
            "dropout": cfg.enable_dropout,
        },
        in_axes=in_axes_tuple,
        length=length,
        metadata_params={nn.PARTITION_NAME: metadata_axis_name},
    )
    return scan_fn(
        config=cfg, mesh=mesh, name=metadata_axis_name, quant=self.quant, **kwargs  # pytype: disable=wrong-keyword-args
    )

  def get_pipeline_stage_module(self, decoder_blocks):
    \"\"\"get pipeline stage module\"\"\"

    def get_layer_to_pipeline(blocks, cfg):
      if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
        return blocks[1]  # return the sparse block
      else:
        return blocks[0]

    cfg = self.config
    base_stage = get_layer_to_pipeline(decoder_blocks, cfg)
    if cfg.set_remat_policy_on_layers_per_stage:
      policy = self.get_remat_policy()
      base_stage = self.set_remat_policy([base_stage], policy)[0]
    if cfg.num_layers_per_pipeline_stage == 1:
      stage_module = base_stage(config=cfg, mesh=self.mesh, quant=self.quant, model_mode=self.model_mode)
    elif cfg.scan_layers_per_stage:
      stage_module = self.scan_decoder_layers(
          cfg,
          base_stage,
          cfg.num_layers_per_pipeline_stage,
          "layers_per_stage",
          self.mesh,
          in_axes_tuple=(nn.broadcast,) * 4,
      )
    else:
      stage_module = SequentialBlockDecoderLayers(
          decoder_layer=base_stage,
          num_decoder_layers=cfg.num_layers_per_pipeline_stage,
          config=cfg,
          mesh=self.mesh,
          quant=self.quant,
          model_mode=self.model_mode,
      )
    return stage_module

  @nn.compact
  def _apply_embedding(
      self,
      shared_embedding: nn.Module | nnx.Module,
      decoder_input_tokens,
      decoder_positions,
      deterministic,
      model_mode,
      multimodal_input=None,
  ):
    \"\"\"Applies token and positional embeddings to the input tokens.\"\"\"
    cfg = self.config

    y = shared_embedding(decoder_input_tokens.astype("int32"), model_mode=model_mode)

    # Merge the image embeddings with the text embeddings for multimodal models
    if multimodal_input is not None:
      image_embeddings = multimodal_input.image_embeddings
      bidirectional_mask = multimodal_input.bidirectional_mask
      image_masks = multimodal_input.image_masks
      audio_embeddings = multimodal_input.audio_embeddings
      audio_masks = multimodal_input.audio_masks

      if image_embeddings is not None and cfg.use_multimodal:
        if cfg.model_name in [
            "gemma3-4b",
            "gemma3-12b",
            "gemma3-27b",
            "gemma4-26b",
            "gemma4-31b",
            "llama4-17b-16e",
            "llama4-17b-128e",
            "qwen3-omni-30b-a3b",
        ]:
          y = mm_utils.merge_mm_embeddings(
              text_embeddings=y,
              multimodal_embeddings=image_embeddings,
              mask=bidirectional_mask,
              token_masks=image_masks,
          )
        # TODO(hengtaoguo): Add support for other multimodal models such as Llama4, refactor if needed
        else:
          raise ValueError(f"Unsupported model_name for multimodal: {cfg.model_name}")

      if audio_embeddings is not None and cfg.use_audio:
        if cfg.model_name in ["qwen3-omni-30b-a3b"]:
          y = mm_utils.merge_mm_embeddings(
              text_embeddings=y,
              multimodal_embeddings=audio_embeddings,
              mask=audio_masks,
              token_masks=None,
          )
        else:
          raise ValueError(f"Unsupported model_name for audio: {cfg.model_name}")

    y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    if cfg.use_untrainable_positional_embedding:
      y += positional_embedding_as_linen(embedding_dims=cfg.base_emb_dim)(y.shape[1], decoder_positions)

    if cfg.trainable_position_size > 0:
      y += embed_as_linen(
          num_embeddings=cfg.trainable_position_size,
          num_features=cfg.emb_dim,
          dtype=cfg.dtype,
          embedding_init=nn.initializers.normal(stddev=1.0),
          name="position_embedder",
          config=cfg,
          mesh=self.mesh,
      )(decoder_positions.astype("int32"), model_mode=model_mode)
    return y

  @nn.compact
  def apply_output_head(self, shared_embedding: nn.Module | nnx.Module, y, deterministic, model_mode):
    \"\"\"Applies final normalization and projects hidden states to logits.\"\"\"

    cfg = self.config
    if cfg.shard_mode == ShardMode.EXPLICIT:
      norm_out_sharding = create_sharding(self.mesh, ("activation_batch", "activation_length", "activation_embed"))
    else:
      norm_out_sharding = None

    y = self.get_norm_layer(num_features=y.shape[-1])(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="decoder_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
        parameter_memory_host_offload=cfg.parameter_memory_host_offload,
    )(y, out_sharding=norm_out_sharding)
    y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(y, deterministic=deterministic)

    if model_mode in (MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE):
      out_sharding = create_sharding(self.mesh, (None, None, "activation_vocab"))
    else:
      out_sharding = create_sharding(
          self.mesh, ("activation_embed_and_logits_batch", "activation_length", "activation_vocab")
      )

    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      if isinstance(shared_embedding, nnx.Module):
        embedding_table = shared_embedding.embedding.value
      else:
        embedding_table = shared_embedding.variables["params"]["embedding"]
      if isinstance(embedding_table, nn.spmd.LogicallyPartitioned):
        embedding_table = embedding_table.unbox()
      attend_dtype = jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype
      logits = attend_on_embedding(y, embedding_table, attend_dtype, self.config, out_sharding)

      if self.config.normalize_embedding_logits:
        # Correctly normalize pre-softmax logits for this shared case.
        logits = logits / jnp.sqrt(y.shape[-1])
      if cfg.final_logits_soft_cap:
        logits = logits / cfg.final_logits_soft_cap
        logits = jnp.tanh(logits) * cfg.final_logits_soft_cap
    else:
      logits = linears.dense_general(
          inputs_shape=y.shape,
          out_features_shape=cfg.vocab_size,
          weight_dtype=cfg.weight_dtype,
          dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,  # for logit training stability
          kernel_axes=("embed_vocab", "vocab"),
          shard_mode=cfg.shard_mode,
          name="logits_dense",
          matmul_precision=self.config.matmul_precision,
          parameter_memory_host_offload=cfg.parameter_memory_host_offload,
      )(
          y,
          out_sharding=out_sharding,
      )  # We do not quantize the logits matmul.

    if self.config.cast_logits_to_fp32:
      logits = logits.astype(jnp.float32)

    return logits

  @nn.compact
  def __call__(
      self,
      shared_embedding: nn.Module | nnx.Module,
      decoder_input_tokens,
      decoder_positions,
      decoder_segment_ids=None,
      deterministic=False,
      model_mode=MODEL_MODE_TRAIN,
      previous_chunk=None,
      slot: None | int = None,
      page_state: None | page_manager.PageState = None,
      multimodal_input=None,
      kv_caches: list[jax.Array] | None = None,
      attention_metadata=None,
      deepstack_visual_embeds: None | list[jnp.ndarray] = None,
  ):
    cfg = self.config
    mesh = self.mesh
    assert decoder_input_tokens.ndim == 2  # [batch, len]

    # [batch, length] -> [batch, length, emb_dim]
    y = self._apply_embedding(
        shared_embedding,
        decoder_input_tokens,
        decoder_positions,
        deterministic,
        model_mode,
        multimodal_input=multimodal_input,
    )

    mhc_expand, mhc_reduce = mhc.get_functions(cfg.mhc_expansion_rate)
    if cfg.mhc_expansion_rate > 1:
      # (batch, length, emb_dim) --> (batch, length, mhc_expansion_rate, emb_dim)
      y = mhc_expand(y)

    policy = self.get_remat_policy()
    RemattedBlockLayers = self.set_remat_policy(self.decoder_layer, policy)
    # scan does not support kwargs in layer call, passing broadcast_args as positional arg
    broadcast_args = (
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
    )
    if cfg.using_pipeline_parallelism:
      logical_partition_spec = (
          self.pipeline_module.get_weight_sharding(y, decoder_segment_ids, decoder_positions, deterministic, model_mode)
          if cfg.pipeline_fsdp_ag_once or cfg.pipeline_fsdp_ag_per_repeat
          else None
      )
      if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
        assert len(RemattedBlockLayers) == 2, "Scanned layers must have a length of 2 using deepseek."
        dense_layer = RemattedBlockLayers[0]
        moe_layer = RemattedBlockLayers[1]
        num_moe_layers = cfg.num_decoder_layers - cfg.first_num_dense_layers
        num_moe_layers_outside_pp = num_moe_layers - self.config.pipeline_parallel_layers
        logical_axis_rules_pp_as_dp = sharding.logical_axis_rules_pp_act_as_dp(self.config.logical_axis_rules)
        # We chose not to pipeline the dense layers, only sparse for SPMD.
        with self.mesh, nn.partitioning.axis_rules(logical_axis_rules_pp_as_dp):
          y, _ = self.scan_decoder_layers(
              cfg,
              dense_layer,
              cfg.first_num_dense_layers,
              "dense_layers",
              mesh,
              in_axes_tuple=(nn.broadcast,) * len(broadcast_args),
              model_mode=model_mode,
          )(y, *broadcast_args)
          if num_moe_layers_outside_pp > 0:
            y, _ = self.scan_decoder_layers(
                cfg,
                moe_layer,
                num_moe_layers_outside_pp,
                "moe_layers",
                mesh,
                in_axes_tuple=(nn.broadcast,) * len(broadcast_args),
                model_mode=model_mode,
            )(y, *broadcast_args)
        y = self.pipeline_module(y, *broadcast_args, logical_partition_spec=logical_partition_spec)
      else:  # Not DeepSeek
        y = self.pipeline_module(y, *broadcast_args, logical_partition_spec=logical_partition_spec)
        remaining_layers = self.config.num_decoder_layers - self.config.pipeline_parallel_layers
        if remaining_layers > 0:
          logical_axis_rules_pp_as_dp = sharding.logical_axis_rules_pp_act_as_dp(self.config.logical_axis_rules)
          with self.mesh, nn.partitioning.axis_rules(logical_axis_rules_pp_as_dp):
            y, _ = self.scan_decoder_layers(
                cfg,
                RemattedBlockLayers[0],
                remaining_layers,
                "layers_outside_pipeline",
                mesh,
                in_axes_tuple=(nn.broadcast,) * len(broadcast_args),
                model_mode=model_mode,
            )(y, *broadcast_args)
    else:
      if cfg.scan_layers:
        if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
          assert len(RemattedBlockLayers) == 2, "Scanned layers must have a length of 2 using deepseek."
          layer_call_kwargs = {
              "page_state": page_state,
              "previous_chunk": previous_chunk,
              "slot": slot,
          }
          dense_layer = RemattedBlockLayers[0]
          moe_layer = RemattedBlockLayers[1]
          if cfg.engram_layers:
            original_dense_call = dense_layer.__call__
            original_moe_call = moe_layer.__call__
            dense_layer.__call__ = functools.partial(dense_layer.__call__, **layer_call_kwargs)
            moe_layer.__call__ = functools.partial(moe_layer.__call__, **layer_call_kwargs)

            common_kwargs = {
                "dense_layer": dense_layer,
                "moe_layer": moe_layer,
                "original_dense_call": original_dense_call,
                "original_moe_call": original_moe_call,
                "layer_call_kwargs": layer_call_kwargs,
                "decoder_segment_ids": decoder_segment_ids,
                "decoder_positions": decoder_positions,
                "deterministic": deterministic,
                "model_mode": model_mode,
                "decoder_input_tokens": decoder_input_tokens,
                "broadcast_args": broadcast_args,
            }

            # Apply Dense Layers
            y = self._apply_interleaved_scanned_layers(
                y,
                layer_type="dense",
                start_idx=0,
                end_idx=cfg.first_num_dense_layers,
                engram_indices=cfg.engram_layers,
                **common_kwargs,
            )

            # Apply MoE Layers
            y = self._apply_interleaved_scanned_layers(
                y,
                layer_type="moe",
                start_idx=cfg.first_num_dense_layers,
                end_idx=cfg.num_decoder_layers,
                engram_indices=cfg.engram_layers,
                **common_kwargs,
            )
          else:
            dense_layer.__call__ = functools.partial(dense_layer.__call__, **layer_call_kwargs)
            y, _ = self.scan_decoder_layers(
                cfg,
                dense_layer,
                cfg.first_num_dense_layers,
                "dense_layers",
                mesh,
                in_axes_tuple=(nn.broadcast,) * len(broadcast_args),
                model_mode=model_mode,
            )(y, *broadcast_args)
            moe_layer.__call__ = functools.partial(moe_layer.__call__, **layer_call_kwargs)
            num_moe_layers = cfg.num_decoder_layers - cfg.first_num_dense_layers

            # If batch-split schedule is used and initialization is complete,
            # as detected by immutable params, use deepseek_batchsplit custom
            # scan with initialized parameters.
            if cfg.use_batch_split_schedule and not self.is_mutable_collection("params"):
              # old version of batch-split that fully uses qwix quantization.
              if cfg.use_qwix_quantization and not cfg.use_manual_quantization:
                y = deepseek_batchsplit_fp8.scan_batch_split_layers(
                    y,
                    self.variables["params"]["moe_layers"],
                    decoder_positions,
                    decoder_segment_ids,
                    model_mode=model_mode,
                    mesh=mesh,
                    quant=self.quant,
                    cfg=cfg,
                    policy=policy,
                )
              else:
                # bf16 and fp8 code path for pure-JAX batch-split.
                # fp8 code path supports both manual quantization and qwix
                # quantization.
                y = deepseek_batchsplit.scan_batch_split_layers(
                    y,
                    self.variables["params"]["moe_layers"],
                    decoder_positions,
                    mesh=mesh,
                    cfg=cfg,
                    num_layers=num_moe_layers,
                )
            else:
              y, _ = self.scan_decoder_layers(
                  cfg,
                  moe_layer,
                  num_moe_layers,
                  "moe_layers",
                  mesh,
                  in_axes_tuple=(nn.broadcast,) * len(broadcast_args),
                  model_mode=model_mode,
              )(y, *broadcast_args)
        elif cfg.decoder_block == DecoderBlockType.GEMMA3:
          bidirectional_mask_value = multimodal_input.bidirectional_mask if multimodal_input is not None else None
          y = self._apply_gemma3_scanned_blocks(
              y,
              decoder_segment_ids,
              decoder_positions,
              deterministic,
              model_mode,
              bidirectional_mask_value,
              previous_chunk,
              page_state,
              slot,
          )
        elif cfg.decoder_block == DecoderBlockType.GEMMA4:
          bidirectional_mask_value = multimodal_input.bidirectional_mask if multimodal_input is not None else None
          y = self._apply_gemma4_scanned_blocks(
              y,
              decoder_segment_ids,
              decoder_positions,
              deterministic,
              model_mode,
              bidirectional_mask_value,
              previous_chunk,
              page_state,
              slot,
          )
        else:
          RemattedBlockLayer = RemattedBlockLayers[0]
          scan_length = int(cfg.num_decoder_layers / cfg.inhomogeneous_layer_cycle_interval)
          layer_kwargs = {}
          if cfg.decoder_block == DecoderBlockType.LLAMA4:
            layer_kwargs = {
                "nope_layer_interval": self.config.nope_layer_interval,
                "interleave_moe_layer_step": self.config.interleave_moe_layer_step,
            }
          # Update broadcast_args and in_axes_tuple for vLLM RPA
          in_axes_tuple = (nn.broadcast,) * len(broadcast_args)
          current_broadcast_args = list(broadcast_args)
          current_in_axes_tuple = list(in_axes_tuple)

          if kv_caches is not None:
            # Stack kv_caches for scan: [num_layers, ...]
            stacked_kv_cache = jnp.stack(kv_caches, axis=0)

            # We pass (y, stacked_kv_cache, 0) as the carry
            carry = (y, stacked_kv_cache, 0)

            # We don't pass kv_cache as a scanned argument anymore

            # Pass None for previous_chunk, slot, page_state, kv_cache to align with __call__ signature
            current_broadcast_args.extend([None, None, None, None, attention_metadata])
            current_in_axes_tuple.extend([nn.broadcast] * 5)

            max_logging.info(f"DEBUG: len(current_broadcast_args)={len(current_broadcast_args)}")
            max_logging.info(f"DEBUG: current_broadcast_args={[type(a) for a in current_broadcast_args]}")

            final_carry, _ = self.scan_decoder_layers(
                cfg,
                RemattedBlockLayer,
                scan_length,
                "layers",
                mesh,
                in_axes_tuple=tuple(current_in_axes_tuple),
                model_mode=model_mode,
                **layer_kwargs,
            )(carry, *current_broadcast_args)

            y, returned_kv_cache, _ = final_carry

            # Update the list of KV caches from the scanned results
            for i in range(cfg.num_decoder_layers):
              kv_caches[i] = returned_kv_cache[i]
          else:
            # Fallback to old behavior if kv_caches is None (not vLLM RPA)
            current_broadcast_args.append(None)
            current_in_axes_tuple.append(nn.broadcast)

            y, _ = self.scan_decoder_layers(
                cfg,
                RemattedBlockLayer,
                scan_length,
                "layers",
                mesh,
                in_axes_tuple=tuple(current_in_axes_tuple),
                model_mode=model_mode,
                **layer_kwargs,
            )(y, *current_broadcast_args)
      else:
        if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
          assert len(RemattedBlockLayers) == 2, "Unscanned layers must have a length of 2 using deepseek."
          dense_layer = RemattedBlockLayers[0]
          moe_layer = RemattedBlockLayers[1]

          layers = [dense_layer, moe_layer]
          layer_prefixes = ["dense_layers", "moe_layers"]
          num_moe_layers = cfg.num_decoder_layers - cfg.first_num_dense_layers
          num_layers_list = [cfg.first_num_dense_layers, num_moe_layers]
          # Iterate over the two layer groups (dense and MoE) and apply layer transformation
          global_layer_idx_offset = 0
          for layer, num_layers, layer_prefix in zip(layers, num_layers_list, layer_prefixes):
            for index in range(num_layers):
              global_layer_idx = global_layer_idx_offset + index
              kv_cache = kv_caches[index] if kv_caches is not None else None
              input_tokens = decoder_input_tokens if cfg.engram_layers else None
              y, kv_cache = layer(
                  config=cfg,
                  mesh=mesh,
                  name=f"{layer_prefix}_{index}",
                  quant=self.quant,
                  model_mode=self.model_mode,
                  layer_idx=global_layer_idx,
              )(
                  y,
                  decoder_segment_ids,
                  decoder_positions,
                  deterministic,
                  model_mode,
                  previous_chunk=previous_chunk,
                  page_state=page_state,
                  slot=slot,
                  kv_cache=kv_cache,
                  attention_metadata=attention_metadata,
                  decoder_input_tokens=input_tokens,
              )
              if kv_caches is not None and kv_cache is not None:
                kv_caches[index] = kv_cache
            global_layer_idx_offset += num_layers
        else:
          for lyr in range(cfg.num_decoder_layers):
            RemattedBlockLayer = RemattedBlockLayers[0]
            layer_kwargs = {}
            layer_call_kwargs = {}
            if cfg.decoder_block == DecoderBlockType.GEMMA3:
              # Gemma3 uses both global and sliding window attention depending on the layer index.
              bidirectional_mask_value = multimodal_input.bidirectional_mask if multimodal_input is not None else None
              layer_kwargs = {"attention_type": gemma3.get_attention_type(layer_id=lyr)}
              layer_call_kwargs = {"bidirectional_mask": bidirectional_mask_value}
            if cfg.decoder_block == DecoderBlockType.GEMMA4:
              # Gemma4 uses both global and sliding window attention depending on the layer index.
              bidirectional_mask_value = multimodal_input.bidirectional_mask if multimodal_input is not None else None
              layer_kwargs = {"attention_type": gemma4.get_attention_type(layer_id=lyr)}
              layer_call_kwargs = {"bidirectional_mask": bidirectional_mask_value}
            if cfg.decoder_block == DecoderBlockType.LLAMA4:
              layer_kwargs = {
                  "is_nope_layer": llama4.determine_is_nope_layer(lyr, self.config.nope_layer_interval),
                  "is_moe_layer": llama4.determine_is_moe_layer(lyr, self.config.interleave_moe_layer_step),
              }
            if cfg.decoder_block == DecoderBlockType.QWEN3_NEXT:
              layer_kwargs = {"layer_idx": lyr}
            kv_cache = None
            if kv_caches is not None and cfg.decoder_block != DecoderBlockType.QWEN3_NEXT:
              kv_cache = kv_caches[lyr]
            elif kv_caches is not None and cfg.decoder_block == DecoderBlockType.QWEN3_NEXT:
              # For Qwen3Next, kv_caches is a dictionary of lists of caches.
              if (lyr + 1) % cfg.inhomogeneous_layer_cycle_interval == 0:
                kv_cache = (kv_caches["key_cache"][lyr], kv_caches["value_cache"][lyr])

            if cfg.decoder_block == DecoderBlockType.GPT_OSS:
              layer_kwargs = {"attention_type": gpt_oss.get_attention_type(layer_id=lyr)}
            if cfg.decoder_block == DecoderBlockType.OLMO3:
              layer_kwargs = {"attention_type": olmo3.get_attention_type(layer_id=lyr)}
            layer = RemattedBlockLayer(
                config=cfg, mesh=mesh, name=f"layers_{lyr}", quant=self.quant, model_mode=self.model_mode, **layer_kwargs
            )
            y, returned_cache = layer(
                y,
                decoder_segment_ids,
                decoder_positions,
                deterministic,
                model_mode,
                previous_chunk=previous_chunk,
                page_state=page_state,
                slot=slot,
                kv_cache=kv_cache,
                attention_metadata=attention_metadata,
                **layer_call_kwargs,
            )
            if kv_caches is not None and returned_cache is not None:
              if cfg.decoder_block != DecoderBlockType.QWEN3_NEXT:
                kv_caches[lyr] = returned_cache
              elif (lyr + 1) % cfg.inhomogeneous_layer_cycle_interval == 0:
                kv_caches["key_cache"][lyr] = returned_cache[0]
                kv_caches["value_cache"][lyr] = returned_cache[1]

            if deepstack_visual_embeds is not None and lyr < len(deepstack_visual_embeds):
              visual_embeds = deepstack_visual_embeds[lyr]
              # Use bidirectional_mask to identify visual token positions
              bidirectional_mask_value = multimodal_input.bidirectional_mask if multimodal_input is not None else None
              if bidirectional_mask_value is not None and visual_embeds is not None:
                y = deepstack_process(y, bidirectional_mask_value, visual_embeds)

    assert isinstance(y, jax.Array)

    # After the final transformer layer, `y` holds the raw, un-normalized hidden state.
    if cfg.mhc_expansion_rate > 1:
      # (batch, length, mhc_expansion_rate, emb_dim) --> (batch, length, emb_dim)
      hidden_state = mhc_reduce(y)
    else:
      hidden_state = y

    # When initializing with vLLM RPA attention, we need to run the output head to
    # initialize any parameters associated with it.
    if self.is_initializing() and cfg.attention == "vllm_rpa":
      _ = self.apply_output_head(shared_embedding, hidden_state, deterministic, model_mode)

    # When invoking from vLLM with RPA attention, logit computation is deferred to a later stage.
    if cfg.attention == "vllm_rpa":
      logits = None
    # When in the Indexer Dense Warm-up stage, skip the expensive output head projection
    # for efficiency, as the main model is frozen and the LM loss is not needed.
    # TODO(b/501446870): Investigate model_mode as train at beginning for decoding stage
    elif (
        cfg.use_indexer and cfg.indexer_loss_scaling_factor > 0.0 and not cfg.indexer_sparse_training
    ) and model_mode == MODEL_MODE_TRAIN:
      logits = None
    # When vocab tiling is enabled in training mode, full logits won't generate to reduce memory
    # Instead, we keep track on the hidden states, which has smaller size compared to full logits
    elif cfg.num_vocab_tiling > 1 and model_mode == MODEL_MODE_TRAIN:
      logits = None
      self.sow("intermediates", "hidden_states", hidden_state)

    else:
      logits = self.apply_output_head(shared_embedding, hidden_state, deterministic, model_mode)

    # The API of the Decoder is now a tuple, providing both the main output
    # and the raw hidden state needed for auxiliary tasks.
    return logits, hidden_state, kv_caches

  def _apply_gemma3_scanned_blocks(
      self,
      y,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      bidirectional_mask,
      previous_chunk,
      page_state,
      slot,
  ):
    \"\"\"Applies Gemma3 scanned decoder blocks, handling main scan and remainders.\"\"\"

    cfg = self.config
    mesh = self.mesh

    # Define the repeating pattern length and calculate how many full blocks to scan
    attention_pattern_length = len(gemma3.GEMMA3_ATTENTION_PATTERN)
    scan_length = cfg.num_decoder_layers // attention_pattern_length

    policy = self.get_remat_policy()
    RemattedGemma3Block = self.set_remat_policy([gemma3.Gemma3ScannableBlockToLinen], policy)[0]

    layer_call_kwargs = {"bidirectional_mask": bidirectional_mask}
    layer_kwargs = {"num_of_layers": attention_pattern_length}

    # Apply the main scan over the full blocks
    if scan_length > 0:
      broadcast_args = (
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
      )
      y, _ = self.scan_decoder_layers(
          cfg,
          RemattedGemma3Block,
          scan_length,
          "layers",
          mesh,
          in_axes_tuple=(nn.broadcast,) * len(broadcast_args),
          model_mode=self.model_mode,
          **layer_kwargs,
      )(y, *broadcast_args, **layer_call_kwargs)

    # Apply any remaining layers that did not fit into a full scanned block
    num_remaining_layers = cfg.num_decoder_layers % attention_pattern_length
    if num_remaining_layers > 0:
      # We name the remainder block with a 'remainder' suffix to avoid parameter name collisions
      rem_layer_kwargs = {"num_of_layers": num_remaining_layers}
      layer = RemattedGemma3Block(
          config=cfg, mesh=mesh, quant=self.quant, model_mode=self.model_mode, name="layers_remainder", **rem_layer_kwargs
      )  # pytype: disable=wrong-keyword-args
      y, _ = layer(
          y,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk=previous_chunk,
          page_state=page_state,
          slot=slot,
          **layer_call_kwargs,
      )
    return y

  def _apply_gemma4_scanned_blocks(
      self,
      y,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      bidirectional_mask,
      previous_chunk,
      page_state,
      slot,
  ):
    \"\"\"Applies Gemma4 scanned decoder blocks, handling main scan and remainders.\"\"\"

    cfg = self.config
    mesh = self.mesh

    # Define the repeating pattern length and calculate how many full blocks to scan
    block_pattern_len = len(gemma4.GEMMA4_ATTENTION_PATTERN)
    num_full_blocks = cfg.num_decoder_layers // block_pattern_len
    remainder_layers = cfg.num_decoder_layers % block_pattern_len

    broadcast_args = (
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        slot,
        page_state,
        previous_chunk,
        bidirectional_mask,
    )

    if num_full_blocks > 0:
      ScannableBlockToLinen = gemma4.Gemma4ScannableBlockToLinen
      policy = self.get_remat_policy()
      RemattedGemma4Block = self.set_remat_policy([ScannableBlockToLinen], policy)[0]
      # For a fully scanned block, apply it inside a nn.scan over the calculated number of full blocks
      y, _ = nn.scan(
          RemattedGemma4Block,
          variable_axes={
              "params": cfg.param_scan_axis,
              "cache": 0,
              "intermediates": 0,
              "aqt": 0,
              "_overwrite_with_gradient": 0,
          },
          split_rngs={"params": True, "dropout": cfg.enable_dropout},
          in_axes=(nn.broadcast,) * len(broadcast_args),
          length=num_full_blocks,
          metadata_params={
              nn.PARTITION_NAME: "layers",
              "abstract_init": False,
          },
      )(
          config=cfg,
          mesh=mesh,
          quant=self.quant,
          model_mode=model_mode,
          num_of_layers=block_pattern_len,
          name="scanned_blocks",
      )(
          y, *broadcast_args
      )

    # Process any remaining layers that don't fit into a full scanned block
    for layer_id in range(cfg.num_decoder_layers - remainder_layers, cfg.num_decoder_layers):
      attention_type = gemma4.get_attention_type(layer_id)
      layer = gemma4.Gemma4DecoderLayerToLinen(
          config=cfg,
          mesh=mesh,
          model_mode=model_mode,
          quant=self.quant,
          attention_type=attention_type,
          layer_idx=layer_id,
      )
      y = layer(y, *broadcast_args)
      if cfg.scan_layers:
        y = y[0]

    return y

  # TODO(b/490118813): Relocate the following functions to their designated directories
  # once the plug-in strategy is implemented: _find_next_boundary(), _apply_single_engram_layer()
  # _apply_scanned_chunk() and _apply_interleaved_scanned_layers().
  def _find_next_boundary(self, current_idx, end_idx, engram_indices):
    \"\"\"Finds the next index boundary, either the next Engram layer index or the overall end index.\"\"\"
    next_engrams = [l for l in engram_indices if l > current_idx]
    if next_engrams:
      return min(end_idx, *next_engrams)
    return end_idx

  def _apply_single_engram_layer(self, y, current_idx, layer_type, **kwargs):
    \"\"\"Applies a single, unscanned Engram layer.\"\"\"
    layer = kwargs["dense_layer"] if layer_type == "dense" else kwargs["moe_layer"]
    layer_prefix = "dense_layers" if layer_type == "dense" else "moe_layers"
    original_call = kwargs["original_dense_call"] if layer_type == "dense" else kwargs["original_moe_call"]
    layer_call_kwargs = kwargs["layer_call_kwargs"]

    layer.__call__ = original_call
    y, _ = layer(
        config=self.config,
        mesh=self.mesh,
        name=f"{layer_prefix}_engram_{current_idx}",
        quant=self.quant,
        model_mode=self.model_mode,
        layer_idx=current_idx,
    )(
        y,
        kwargs["decoder_segment_ids"],
        kwargs["decoder_positions"],
        kwargs["deterministic"],
        kwargs["model_mode"],
        decoder_input_tokens=kwargs["decoder_input_tokens"],
        **layer_call_kwargs,
    )
    layer.__call__ = functools.partial(original_call, **layer_call_kwargs)
    return y

  def _apply_scanned_chunk(self, y, current_idx, next_boundary, layer_type, **kwargs):
    \"\"\"Applies a contiguous chunk of layers using the scan operation.\"\"\"
    layer = kwargs["dense_layer"] if layer_type == "dense" else kwargs["moe_layer"]
    layer_prefix = "dense_layers" if layer_type == "dense" else "moe_layers"
    broadcast_args = kwargs["broadcast_args"]
    scan_length = next_boundary - current_idx

    if scan_length > 0:
      y, _ = self.scan_decoder_layers(
          self.config,
          layer,
          scan_length,
          f"{layer_prefix}_{current_idx}_{next_boundary - 1}",
          self.mesh,
          in_axes_tuple=(nn.broadcast,) * len(broadcast_args),
          model_mode=kwargs["model_mode"],
      )(y, *broadcast_args)
    return y

  def _apply_interleaved_scanned_layers(self, y, layer_type, start_idx, end_idx, engram_indices, **kwargs):
    \"\"\"Applies a mix of scanned standard layers and unscanned Engram layers.\"\"\"
    current_idx = start_idx
    while current_idx < end_idx:
      if current_idx in engram_indices:
        # Handle individual unscanned Engram layer
        y = self._apply_single_engram_layer(y, current_idx, layer_type, **kwargs)
        current_idx += 1
      else:
        # Find next boundary and scan the chunk
        next_boundary = self._find_next_boundary(current_idx, end_idx, engram_indices)
        y = self._apply_scanned_chunk(y, current_idx, next_boundary, layer_type, **kwargs)
        current_idx = next_boundary
    return y
\n"""


# File: src/maxtext/layers/initializers.py (commit 313890777)
INITIALIZERS_RAW = """\n# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

\"\"\"Initializers.\"\"\"

from typing import Callable

import jax

from flax import linen as nn
from flax import nnx
from aqt.jax.v2 import aqt_tensor

from maxtext.common.common_types import Array, DType, Shape, PRNGKey

Initializer = Callable[[PRNGKey, Shape, DType], Array]
InitializerAxis = int | tuple[int, ...]
NdInitializer = Callable[[PRNGKey, Shape, DType, InitializerAxis, InitializerAxis], Array]

default_embed_init = nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)

default_bias_init = jax.nn.initializers.constant(0.0)
default_scalar_init = jax.nn.initializers.constant(0.01)


def nd_dense_init(scale, mode, distribution):
  \"\"\"Creates a variance-scaling initializer with dynamic in/out axes.

  This function is a factory that returns an initializer function. The returned
  function is a wrapper around `jax.nn.initializers.variance_scaling` that
  allows the `in_axis` and `out_axis` to be specified at call time, rather
  than at creation time.

  Args:
    scale: The scaling factor for the variance.
    mode: The mode for variance scaling ('fan_in', 'fan_out', 'fan_avg').
    distribution: The distribution to sample from ('normal', 'uniform', etc.).

  Returns:
    A function that takes a PRNG key, shape, dtype, in_axis, and out_axis,
    and returns an initialized array.
  \"\"\"

  def init_fn(key, shape, dtype, in_axis, out_axis):
    \"\"\"Initializes an array using variance scaling with specified axes.\"\"\"
    fn = jax.nn.initializers.variance_scaling(scale, mode, distribution, in_axis, out_axis)
    return fn(key, shape, dtype)

  return init_fn


def variable_to_logically_partitioned(variable: nnx.Variable):
  \"\"\"Wraps an NNX variable's value in `nn.LogicallyPartitioned`.

  This function inspects the metadata of an `nnx.Variable` object. If
  sharding information ('out_sharding', 'sharding' or 'sharding_names') is
  present, it wraps the variable's value in `nn.LogicallyPartitioned` to apply
  the specified sharding constraints.

  It handles special cases for `aqt_tensor.QTensor` and variables of type
  `_overwrite_with_gradient` by returning their values directly without
  wrapping.

  Args:
    variable: The `nnx.Variable` object to process.

  Returns:
    The variable's value, potentially wrapped in `nn.LogicallyPartitioned`.
  \"\"\"
  val = variable.get_value()
  if isinstance(val, aqt_tensor.QTensor):
    return val

  if variable.type.__name__ == "_overwrite_with_gradient":
    return val

  metadata = variable.get_metadata()
  out_sharding = None
  if "out_sharding" in metadata:
    out_sharding = metadata["out_sharding"]
  elif "sharding_names" in metadata:
    out_sharding = metadata["sharding_names"]
  elif "sharding" in metadata:
    out_sharding = metadata["sharding"]

  if out_sharding is not None:
    return nn.LogicallyPartitioned(  # type: ignore[wrong-keyword-args]
        val,
        out_sharding,  # type: ignore[arg-type]
        mesh=metadata.get("mesh"),
        rules=metadata.get("rules"),
    )
  else:
    return val
\n"""


# File: src/maxtext/models/deepseek.py (commit 313890777)
DEEPSEEK_MODEL_RAW = """\n# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

\"\"\"Transformer model definition.\"\"\"
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

import functools
from typing import Optional

from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import Config
from maxtext.common.common_types import HyperConnectionType, MODEL_MODE_PREFILL
from maxtext.inference import page_manager
from maxtext.layers import attention_mla
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import mhc
from maxtext.layers import moe
from maxtext.layers import nnx_wrappers
from maxtext.layers import quantizations
from maxtext.layers.linears import Dropout
from maxtext.layers.engram import Engram
from maxtext.layers.engram import NgramHashMapping
from maxtext.layers.normalizations import RMSNorm
from maxtext.models import deepseek_batchsplit
from maxtext.models import deepseek_batchsplit_fp8
from maxtext.utils import max_utils
from maxtext.utils.sharding import create_sharding
from maxtext.utils.sharding import maybe_shard_with_logical

import transformers

# -----------------------------------------
# The Decoder Layer for DeepSeek v3
# -----------------------------------------


class DeepSeekGenericLayer(nnx.Module):
  \"\"\"Generic DeepSeek layer with Multi-Head Latent Attention.

  This is to be used as a base class for DeepSeek layers with dense/sparse MLPs.
  This class follows a pattern of separating module creation from execution.
  \"\"\"

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      layer_idx: int = -1,
  ) -> None:
    self.config = config
    self.model_mode = model_mode
    self.mesh = mesh
    self.quant = quant
    self.rngs = rngs
    self.is_mhc_enabled = config.mhc_expansion_rate > 1
    self.layer_idx = layer_idx
    self.is_engram_enabled = config.engram_layers and layer_idx in config.engram_layers

    batch_size, sequence_length = max_utils.get_batch_seq_len_for_mode(self.config, self.model_mode)
    self.dummy_inputs_shape = (batch_size, sequence_length, self.config.emb_dim)

    self.out_sharding = create_sharding(self.mesh, self.logical_axis_names, rules=self.config.logical_axis_rules)
    self.mlp_intermediate_sharding = create_sharding(
        self.mesh, self.mlp_logical_axis_names, rules=self.config.logical_axis_rules
    )

    self.pre_self_attention_layer_norm = RMSNorm(
        num_features=self.dummy_inputs_shape[-1],
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=self.config.normalization_layer_epsilon,
        rngs=rngs,
    )

    self.post_self_attention_layer_norm = RMSNorm(
        num_features=self.dummy_inputs_shape[-1],
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=self.config.normalization_layer_epsilon,
        rngs=rngs,
    )

    if self.is_engram_enabled:
      self.engram_layer_norm = RMSNorm(
          num_features=self.dummy_inputs_shape[-1],
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          kernel_axes=("norm",),
          epsilon=self.config.normalization_layer_epsilon,
          rngs=rngs,
      )
      tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer_path, token=config.hf_access_token)
      # TODO(ranran): Refactor NgramHashMapping to initialize once globally or at the model level.
      # Moving this to decoders.py currently causes JAX initialization errors.
      self.ngram_hash_mapping = NgramHashMapping(
          engram_vocab_bases=config.engram_vocab_bases,
          max_ngram_size=config.engram_max_ngram_size,
          engram_num_heads=config.engram_num_heads,
          layer_ids=config.engram_layers,
          tokenizer=tokenizer,
          pad_id=tokenizer.pad_token_id,
          seed=config.engram_seed,
      )
      self.engram = Engram(
          config=config,
          mesh=mesh,
          vocab_sizes=self.ngram_hash_mapping.get_vocab_sizes(layer_idx),
          engram_num_heads=config.engram_num_heads,
          engram_head_dim=config.engram_head_dim,
          engram_max_ngram_size=config.engram_max_ngram_size,
          engram_kernel_size=config.engram_kernel_size,
          mhc_expansion_rate=config.mhc_expansion_rate,
          quant=quant,
          rngs=rngs,
      )
    else:
      self.engram_layer_norm = None
      self.engram = None

    self.self_attention = attention_mla.MLA(
        config=self.config,
        num_query_heads=self.config.num_query_heads,
        num_kv_heads=self.config.num_kv_heads,
        head_dim=self.config.head_dim,
        max_target_length=self.config.max_target_length,
        max_prefill_predict_length=self.config.max_prefill_predict_length,
        attention_kernel=self.config.attention,
        attention_type=self.config.attention_type,
        inputs_q_shape=self.dummy_inputs_shape,
        inputs_kv_shape=self.dummy_inputs_shape,
        mesh=mesh,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        dropout_rate=self.config.dropout_rate,
        name="self_attention",
        quant=quant,
        kv_quant=quantizations.configure_kv_quant(config),
        q_lora_rank=self.config.q_lora_rank,
        kv_lora_rank=self.config.kv_lora_rank,
        qk_nope_head_dim=self.config.qk_nope_head_dim,
        qk_rope_head_dim=self.config.qk_rope_head_dim,
        v_head_dim=self.config.v_head_dim,
        max_position_embeddings=self.config.max_position_embeddings,
        original_max_position_embeddings=self.config.original_max_position_embeddings,
        mscale=self.config.mscale,
        rope_factor=self.config.rope_factor,
        model_mode=model_mode,
        rngs=rngs,
        attn_logits_soft_cap=self.config.attn_logits_soft_cap,
    )

    self.dropout = Dropout(rate=self.config.dropout_rate, broadcast_dims=(-2,), rngs=self.rngs)
    if self.is_mhc_enabled:
      self.mhc_attention = mhc.ManifoldConstrainedHyperConnections(self.config, self.config.emb_dim, self.mesh, self.rngs)
      self.mhc_mlp = mhc.ManifoldConstrainedHyperConnections(self.config, self.config.emb_dim, self.mesh, self.rngs)

  def mlp_op(self, x, deterministic, *args, **kwargs):
    \"\"\"Executes the MLP operation. To be implemented by subclasses.\"\"\"
    raise NotImplementedError()

  def with_logical_constraint(self, x):
    return maybe_shard_with_logical(
        x,
        logical_axes=self.logical_axis_names,
        mesh=self.mesh,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
        extra_stack_level=1,
        rules=self.config.logical_axis_rules,
    )

  def dropout_op(self, x, deterministic):
    dropout = self.dropout(x, deterministic=deterministic)
    return self.with_logical_constraint(dropout)

  def pre_attention_norm_op(self, x):
    pre_attention_norm = self.pre_self_attention_layer_norm(x)
    return self.with_logical_constraint(pre_attention_norm)

  def post_attention_norm_op(self, x):
    post_attention_norm = self.post_self_attention_layer_norm(x)
    return self.with_logical_constraint(post_attention_norm)

  def attention_op(
      self,
      x,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
  ):
    \"\"\"Executes the attention layer.\"\"\"
    attention_result, _ = self.self_attention(
        x,
        x,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=self.model_mode,
        out_sharding=self.out_sharding,
        previous_chunk=previous_chunk,
        page_state=page_state,
        slot=slot,
    )
    return self.with_logical_constraint(attention_result)

  @property
  def logical_axis_names(self):
    \"\"\"Generate logical names for activations generally.\"\"\"
    length_name = "prefill_activation_norm_length" if self.model_mode == MODEL_MODE_PREFILL else "activation_norm_length"
    axis_names = ["activation_batch", length_name, "activation_embed"]
    return axis_names

  @property
  def mlp_logical_axis_names(self):
    \"\"\"Generate logical names for activations in MLP.\"\"\"
    length_name = "prefill_activation_norm_length" if self.model_mode == MODEL_MODE_PREFILL else "activation_norm_length"
    axis_names = ["activation_batch", length_name, "activation_mlp"]
    return axis_names

  def post_process(self, layer_output, load_balance_loss, moe_bias_updates, kv_cache=None):
    \"\"\"postprocessing.\"\"\"

    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)

    if self.config.routed_bias and self.config.routed_bias_update_rate > 0.0 and moe_bias_updates is not None:
      self.sow(nnx.Intermediate, "moe_bias_updates", moe_bias_updates)

    if self.config.record_internal_nn_metrics:
      self.sow(nnx.Intermediate, "activation_mean", jnp.mean(layer_output))
      self.sow(nnx.Intermediate, "activation_stdev", jnp.std(layer_output))
      self.sow(
          nnx.Intermediate,
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if self.config.scan_layers:
      return layer_output, None
    return layer_output, kv_cache

  def self_attention_with_norm_op(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
  ):
    \"\"\"self-attention with normalization\"\"\"
    if self.is_mhc_enabled:
      intermediate_inputs, _ = self.mhc_attention(
          self.pre_attention_norm_op,
          self.self_attention,
          x=inputs,
          mhc_type=HyperConnectionType.ATTENTION,
          decoder_segment_ids=decoder_segment_ids,
          inputs_positions=decoder_positions,
          deterministic=deterministic,
          model_mode=self.model_mode,
          out_sharding=self.out_sharding,
          previous_chunk=previous_chunk,
          page_state=page_state,
          slot=slot,
      )
    else:
      lnx = self.pre_attention_norm_op(inputs)
      attention_lnx = self.attention_op(
          lnx,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          previous_chunk,
          page_state,
          slot,
      )
      intermediate_inputs = inputs + attention_lnx
    # Normalization
    hidden_states = self.post_attention_norm_op(intermediate_inputs)
    return hidden_states, intermediate_inputs

  def engram_op(self, x, decoder_input_tokens):
    normed_x = self.engram_layer_norm(x)
    hash_ids = self.ngram_hash_mapping(decoder_input_tokens)[self.layer_idx]
    return self.engram(normed_x, hash_ids)


class DeepSeekDenseLayer(DeepSeekGenericLayer):
  \"\"\"DeepSeek-style dense layer with Multi-Head Latent Attention.\"\"\"

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      layer_idx: int = -1,
  ) -> None:
    super().__init__(config, model_mode, mesh, rngs, quant, layer_idx)
    self.mlp = linears.MlpBlock(
        in_features=self.dummy_inputs_shape[-1],
        intermediate_dim=self.config.mlp_dim,
        activations=self.config.mlp_activations,
        intermediate_dropout_rate=self.config.dropout_rate,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        config=self.config,
        quant=quant,
        model_mode=model_mode,
        mesh=mesh,
        rngs=self.rngs,
    )

  def mlp_op(self, x, deterministic):
    mlp = self.mlp(x, deterministic, intermediate_sharding=self.mlp_intermediate_sharding, out_sharding=self.out_sharding)
    return self.with_logical_constraint(mlp)

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
      kv_cache=None,
      attention_metadata=None,
      decoder_input_tokens=None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    x = self.with_logical_constraint(inputs)
    x = checkpoint_name(x, "decoder_layer_input")

    if self.is_engram_enabled:
      engram_output = self.engram_op(x, decoder_input_tokens)
      x = x + engram_output

    hidden_states, intermediate_inputs = self.self_attention_with_norm_op(
        x,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        previous_chunk,
        page_state,
        slot,
    )

    if self.is_mhc_enabled:
      layer_output, _ = self.mhc_mlp(
          self.post_attention_norm_op,
          self.mlp,
          x=intermediate_inputs,
          mhc_type=HyperConnectionType.MLP_DENSE,
          deterministic=deterministic,
      )
    else:
      mlp_lnx = self.mlp_op(hidden_states, deterministic)
      layer_output = mlp_lnx + intermediate_inputs
    layer_output = self.dropout_op(layer_output, deterministic=deterministic)

    return self.post_process(layer_output, None, None, kv_cache)


DeepSeekDenseLayerToLinen = nnx_wrappers.to_linen_class(
    DeepSeekDenseLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


class DeepSeekMoELayer(DeepSeekGenericLayer):
  \"\"\"DeepSeek-style MoE layer with Multi-Head Latent Attention.

  Supports dropless and dropping base on configs. Uses a bias in routing instead
  of load balancing loss.
  \"\"\"

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      layer_idx: int = -1,
  ) -> None:
    super().__init__(config, model_mode, mesh, rngs, quant, layer_idx)
    self.DeepSeekMoeBlock_0 = moe.RoutedAndSharedMoE(
        config=self.config,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(self.config.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        quant=quant,
        rngs=self.rngs,
    )

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
      kv_cache=None,
      attention_metadata=None,
      decoder_input_tokens=None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]

    # This code should only be traced during initialization when using
    # batch-split schedule. It is never run during model execution, since
    # `Decoder` directly calls `batch_split_schedule` during execution.
    # That is also why we can split/merge activations here as well as
    # in `Decoder`, since they will never be executed together.
    if self.config.use_batch_split_schedule:
      # The older version of batch-split that fully uses qwix quantization.
      if self.config.use_qwix_quantization and not self.config.use_manual_quantization:
        activation_pspec = jax.sharding.PartitionSpec(
            ("data", "fsdp", "fsdp_transpose", "expert", "context"),
            None,
            None,
        )
        inputs = jax.shard_map(
            functools.partial(
                deepseek_batchsplit_fp8.split,
                split_factor=self.config.batch_split_factor,
            ),
            mesh=self.mesh,
            in_specs=activation_pspec,
            out_specs=[activation_pspec] * self.config.batch_split_factor,
        )(inputs)
        dpos = deepseek_batchsplit_fp8.split(decoder_positions, self.config.batch_split_factor)
        dseg = deepseek_batchsplit_fp8.split(decoder_segment_ids, self.config.batch_split_factor)
        weights = deepseek_batchsplit_fp8.fetch_weights(nnx.to_pure_dict(nnx.state(self, nnx.Param)), self.config.dtype)
        outputs = deepseek_batchsplit_fp8.batch_split_schedule(
            inputs,
            weights,
            dpos,
            dseg,
            model_mode=model_mode,
            mesh=self.mesh,
            quant=self.quant,
            cfg=self.config,
        )
        outputs = jax.shard_map(
            functools.partial(
                deepseek_batchsplit_fp8.merge,
                split_factor=self.config.batch_split_factor,
            ),
            mesh=self.mesh,
            in_specs=([activation_pspec] * self.config.batch_split_factor,),
            out_specs=activation_pspec,
        )(outputs)
        return outputs, None

      # bf16 and fp8 code path for pure-JAX batch-split.
      # fp8 code path supports both manual quantization and qwix
      # quantization.
      input_sharding = jax.typeof(inputs).sharding
      activation_pspec = jax.sharding.PartitionSpec(
          ("data", "fsdp", "expert"),
          None,
          None,
      )
      inputs = jax.reshard(inputs, jax.sharding.NamedSharding(self.mesh, activation_pspec))
      yarn_freqs = deepseek_batchsplit.initialize_yarn_freqs(
          decoder_positions,
          embedding_dims=self.config.qk_rope_head_dim,
          rope_theta=self.config.rope_max_timescale,
          max_position_embeddings=self.config.max_position_embeddings,
          original_max_position_embeddings=self.config.original_max_position_embeddings,
          beta_fast=self.config.beta_fast,
          beta_slow=self.config.beta_slow,
          rope_factor=self.config.rope_factor,
          mesh=self.mesh,
          activation_pspec=activation_pspec,
      )
      yarn_mask = deepseek_batchsplit.initialize_yarn_mask(self.config.qk_rope_head_dim)
      splash_kernel = deepseek_batchsplit.init_splash_kernel(self.config)
      inputs = jax.shard_map(
          functools.partial(
              deepseek_batchsplit.split,
              split_factor=self.config.batch_split_factor,
          ),
          mesh=self.mesh,
          in_specs=activation_pspec,
          out_specs=[activation_pspec] * self.config.batch_split_factor,
      )(inputs)
      yarn_freqs = deepseek_batchsplit.split(yarn_freqs, self.config.batch_split_factor)

      def extract_fn(x):
        if isinstance(x, nnx.variablelib.Variable):
          return maybe_shard_with_logical(
              x.value,
              x.sharding_names,
              self.mesh,
              shard_mode=self.config.shard_mode,
              rules=self.config.logical_axis_rules,
          )
        return x

      weights = deepseek_batchsplit.fetch_weights(
          nnx.to_pure_dict(nnx.state(self, nnx.Param), extract_fn), self.config.dtype
      )
      weights = deepseek_batchsplit.gather_weights(weights, self.mesh)
      outputs, _ = deepseek_batchsplit.batch_split_schedule(
          inputs,
          weights,
          yarn_freqs,
          mesh=self.mesh,
          cfg=self.config,
          splash_kernel=splash_kernel,
          activation_pspec=activation_pspec,
          pairwise_swap_and_negate_mask=yarn_mask,
      )
      moe_inputs, routed_expert_out, shared_expert_out, selected_experts = outputs[1]
      outputs[1], _ = deepseek_batchsplit.unroute_ubatch_shard_mapped(
          moe_inputs,
          routed_expert_out,
          shared_expert_out,
          selected_experts,
          expert_axis_name="expert",
          use_gather_mosaic_kernel=False,
          target_length=self.config.max_target_length,
          mesh=self.mesh,
          activation_pspec=activation_pspec,
      )
      outputs = jax.shard_map(
          functools.partial(
              deepseek_batchsplit.merge,
              split_factor=self.config.batch_split_factor,
          ),
          mesh=self.mesh,
          in_specs=([activation_pspec] * self.config.batch_split_factor,),
          out_specs=activation_pspec,
      )(outputs)
      outputs = jax.reshard(outputs, input_sharding)
      return outputs, None

    x = self.with_logical_constraint(inputs)
    x = checkpoint_name(x, "decoder_layer_input")

    if self.is_engram_enabled:
      engram_output = self.engram_op(x, decoder_input_tokens)
      x = x + engram_output

    hidden_states, intermediate_inputs = self.self_attention_with_norm_op(
        x,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        previous_chunk,
        page_state,
        slot,
    )

    if self.is_mhc_enabled:
      layer_output, metadata = self.mhc_mlp(
          self.post_attention_norm_op,
          self.DeepSeekMoeBlock_0,
          x=intermediate_inputs,
          mhc_type=HyperConnectionType.MLP_MOE,
      )
      load_balance_loss = metadata["load_balance_loss"]
      moe_bias_updates = metadata["moe_bias_updates"]
    else:
      mlp_lnx, load_balance_loss, moe_bias_updates = self.mlp_op(hidden_states, deterministic)
      layer_output = mlp_lnx + intermediate_inputs
    layer_output = self.dropout_op(layer_output, deterministic=deterministic)

    return self.post_process(layer_output, load_balance_loss, moe_bias_updates, kv_cache)

  def mlp_op(self, x, deterministic, *args, **kwargs):
    mlp_lnx, load_balance_loss, moe_bias_updates = self.DeepSeekMoeBlock_0(
        x, intermediate_sharding=self.mlp_intermediate_sharding, out_sharding=self.out_sharding
    )
    return self.with_logical_constraint(mlp_lnx), load_balance_loss, moe_bias_updates


DeepSeekMoELayerToLinen = nnx_wrappers.to_linen_class(
    DeepSeekMoELayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
\n"""


# File: src/maxtext/models/models.py (commit 313890777)
MODELS_RAW = """\n# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

\"\"\"Transformer models.\"\"\"
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from flax import linen as nn
from flax import nnx

from maxtext.common.common_types import Config, DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_TRAIN, MultimodalInput
from maxtext.inference import page_manager
from maxtext.layers.nnx_decoders import NNXDecoder
from maxtext.layers import initializers
from maxtext.layers import nnx_wrappers
from maxtext.layers.decoders import Decoder
from maxtext.layers.embeddings import Embed, embed_as_linen
from maxtext.layers.encoders import AudioEncoder, VisionEncoder, audio_encoder_as_linen, vision_encoder_as_linen
from maxtext.layers.multi_token_prediction import multi_token_prediction_block_as_linen
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.multimodal import processor as mm_processor
from maxtext.utils import max_utils

# ------------------------------------------------------------------------------
# The network: Transformer Definitions
# ------------------------------------------------------------------------------


class TransformerLinenPure(nn.Module):
  \"\"\"An autoregressive transformer model.\"\"\"

  # Make new attributes required, so that all Transformer dependencies (train, decode,
  # compile, etc) will error instead of silently use defaults.
  # pylint: disable=attribute-defined-outside-init
  config: Config
  mesh: Mesh
  quant: Quant
  # Possible model_mode values can be found in maxtext.common.common_types.
  # We generally use maxtext.common.common_types.MODEL_MODE_TRAIN or
  # maxtext.common.common_types.MODEL_MODE_PREFILL for initializations here.
  model_mode: str = MODEL_MODE_TRAIN  # May be different than the model_mode passed to __call__
  # pylint: enable=attribute-defined-outside-init

  def init(self, *args, model_mode: str = MODEL_MODE_TRAIN, **kwargs):
    \"\"\"Initializes the model.\"\"\"
    module = self.clone(model_mode=model_mode)
    kwargs["model_mode"] = model_mode
    return nn.Module.init(module, *args, **kwargs)

  def apply(self, *args, model_mode: str = MODEL_MODE_TRAIN, **kwargs):
    \"\"\"Applies the model.\"\"\"
    module = self.clone(model_mode=model_mode)
    kwargs["model_mode"] = model_mode
    return nn.Module.apply(module, *args, **kwargs)

  def setup(self):
    \"\"\"Initialize shared_embedding & decoder layers.\"\"\"

    cfg = self.config
    mesh = self.mesh
    self.shared_embedding = embed_as_linen(
        num_embeddings=cfg.vocab_size,
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        name="token_embedder",
        config=cfg,
        mesh=self.mesh,
    )
    self.vision_encoder = vision_encoder_as_linen(config=cfg, mesh=mesh) if cfg.use_multimodal else None
    self.audio_encoder = audio_encoder_as_linen(config=cfg, mesh=mesh) if cfg.use_audio else None
    self.decoder = Decoder(config=cfg, mesh=mesh, quant=self.quant, model_mode=self.model_mode)

    # If MTP is enabled via config, set up the MTP block.
    if self.config.mtp_num_layers > 0:
      # Get the list of layer blueprints for the current model.
      # For MTP, we use the DecoderLayer blueprint to ensure architectural consistency.
      # By convention, this is the last layer in the list.
      layer_types = self.decoder.get_decoder_layers()
      mtp_layer_linen = layer_types[-1]
      # UNWRAP: The MTP block is pure NNX. If the decoder returned a Linen wrapper,
      # extract the native NNX class to preserve parameter tracing/scoping.
      mtp_layer_nnx = getattr(mtp_layer_linen, "module_class", mtp_layer_linen)
      self.mtp_block = multi_token_prediction_block_as_linen(
          config=self.config,
          mesh=self.mesh,
          transformer_layer_module=mtp_layer_nnx,
          decoder=self.decoder,
          rngs=self.make_rng("mtp_block"),
      )

  def logits_from_hidden_states(self, hidden_states, deterministic, model_mode):
    \"\"\"
    Compute logits from hidden states (wrapping decoder.apply_output_head).
    This function is only used for vocabulary tiling.
    \"\"\"
    logits = self.decoder.apply_output_head(
        shared_embedding=self.shared_embedding,
        y=hidden_states,
        deterministic=deterministic,
        model_mode=model_mode,
    )
    return logits

  def __call__(
      self,
      decoder_input_tokens: jnp.ndarray,
      decoder_positions: jnp.ndarray,
      decoder_segment_ids=None,
      encoder_images: None | jnp.ndarray = None,
      encoder_image_masks: None | jnp.ndarray = None,
      encoder_audios: None | jnp.ndarray = None,
      enable_dropout=True,
      model_mode=MODEL_MODE_TRAIN,
      previous_chunk=None,
      true_length: None | int = None,
      slot: None | int = None,
      page_state: None | page_manager.PageState = None,
      decoder_target_tokens: None | jnp.ndarray = None,
      decoder_target_mask: None | jnp.ndarray = None,
      nnx_method=None,
      kv_caches: list[jax.Array] | None = None,
      attention_metadata: dict[str, Any] | None = None,
  ):
    \"\"\"Applies Transformer decoder-branch on encoded-input and target.

    Args:
      true_length: (Optional) Prompt length before padding
      slot: (Optional) An integer representing the decode batch index selected
        for this request.
    \"\"\"

    if decoder_segment_ids is not None and model_mode == MODEL_MODE_AUTOREGRESSIVE:
      raise ValueError(
          f"During autoregressive decoding we assume the tokens are in the active sequence"
          f" which is always {DECODING_ACTIVE_SEQUENCE_INDICATOR}."
      )

    bidirectional_mask = None
    image_embeddings = None
    audio_embeddings = None
    deepstack_visual_embeds = None

    if self.config.use_multimodal and encoder_images is not None:
      image_embeddings, deepstack_visual_embeds = self.vision_encoder(
          input_images=encoder_images, deterministic=not enable_dropout
      )

      bidirectional_mask = mm_processor.get_bidirectional_mask_vision(self.config, decoder_input_tokens)

    if self.config.use_multimodal and encoder_audios is not None and self.audio_encoder is not None:
      audio_embeddings = self.audio_encoder(input_audio=encoder_audios, deterministic=not enable_dropout)

    # Create audio mask for placeholder tokens (qwen3-omni models)
    audio_masks = None
    if audio_embeddings is not None:
      audio_masks = mm_processor.get_bidirectional_mask_audio(self.config, decoder_input_tokens)

    multimodal_input = None
    if image_embeddings is not None or audio_embeddings is not None:
      multimodal_input = MultimodalInput(
          image_embeddings=image_embeddings,
          image_masks=encoder_image_masks,
          audio_embeddings=audio_embeddings,
          audio_masks=audio_masks,
          bidirectional_mask=bidirectional_mask,
      )

    logits, hidden_state, kv_caches = self.decoder(
        shared_embedding=self.shared_embedding,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=not enable_dropout,
        model_mode=model_mode,
        previous_chunk=previous_chunk,
        slot=slot,
        page_state=page_state,
        multimodal_input=multimodal_input,
        kv_caches=kv_caches,
        attention_metadata=attention_metadata,
        deepstack_visual_embeds=deepstack_visual_embeds,
    )  # pytype: disable=wrong-keyword-args

    # If we are initializing the model AND MTP is enabled, we must create
    # dummy target tensors. This allows Flax to trace the MTPBlock and create
    # all its necessary parameters, without requiring the main training pipeline
    # to be aware of this initialization detail.
    if self.is_initializing() and self.config.mtp_num_layers > 0:
      if decoder_target_tokens is None:
        dummy_shape = decoder_input_tokens.shape
        decoder_target_tokens = jnp.ones(dummy_shape, dtype=jnp.int32)
        decoder_target_mask = jnp.ones(dummy_shape, dtype=jnp.int32)
        decoder_segment_ids = jnp.ones(dummy_shape, dtype=jnp.int32)

    # The Multi-Token Prediction (MTP) block functions as a "side-car" to the main
    # model, active only during training. It computes an auxiliary loss based on
    # predicting multiple future tokens, as described in the DeepSeek-V3 paper.
    # To ensure architectural consistency, it uses two key components from the parent Transformer:
    #   1. The same `DecoderLayer` blueprint for its internal transformer blocks.
    #   2. The `shared_embedding` for both embedding future tokens and for its final
    #      logit projection.
    # Its only effect is to "sow" these losses; it does not alter the primary logits output.
    if self.config.mtp_num_layers > 0:
      self.mtp_block(
          shared_embedding=self.shared_embedding,
          main_hidden_state=hidden_state,
          input_ids=decoder_input_tokens,
          target_ids=decoder_target_tokens,
          target_mask=decoder_target_mask,
          position_ids=decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          deterministic=not enable_dropout,
          model_mode=model_mode,
      )

    if self.config.attention == "vllm_rpa":
      # In vLLM, logits are computed separately after updating the KV cache.
      return hidden_state, kv_caches

    return logits


def transformer_as_linen(
    config: Config,
    mesh: Mesh,
    quant: Quant,
    model_mode: str = MODEL_MODE_TRAIN,
    *,
    name: str | None = None,
) -> nnx_wrappers.ToLinen | TransformerLinenPure:
  \"\"\"Constructs a Transformer model as a Linen or NNX module.

  This function returns an autoregressive Transformer model as either a Linen module
  or an NNX-wrapped module, depending on the `config.enable_nnx` flag. The returned module
  is suitable for training, evaluation, or decoding.

  If `config.enable_nnx` is True, returns a `TransformerLinen` that wraps the NNX-style
  Transformer for integration with NNX-specific APIs and workflows.
  Otherwise, returns a pure Flax Linen implementation (`TransformerLinenPure`).

  Args:
    config (Config): The configuration object specifying model hyperparameters and options.
    mesh (Mesh): The JAX sharding mesh for device partitioning.
    quant (Quant): The quantization module or configuration to use.
    model_mode (str, optional): The operational mode for the model, e.g.
      training, prefill, or autoregressive. Defaults to `MODEL_MODE_TRAIN`.
    name (str, optional): Optional module name for Linen/NNX construction.

  Returns:
    nnx_wrappers.ToLinen | TransformerLinenPure:
      A constructed Transformer model compatible with the specified framework (Linen or NNX).
  \"\"\"
  if config.enable_nnx:
    return TransformerLinen(
        Transformer,
        args=(),
        kwargs=nn.FrozenDict(
            {
                "mesh": mesh,
                "config": config,
                "quant": quant,
                "model_mode": model_mode,
            }
        ),
        metadata_fn=initializers.variable_to_logically_partitioned,
        name=name,
    )
  else:
    return TransformerLinenPure(config, mesh, quant, model_mode=model_mode, name=name)


class TransformerLinen(nnx_wrappers.ToLinen):
  \"\"\"Transformer model as a linen module.\"\"\"

  def init(self, *args, model_mode: str = MODEL_MODE_TRAIN, **kwargs):
    \"\"\"Initializes the model.\"\"\"
    model_kwargs = self.kwargs.copy({"model_mode": model_mode})  # type: ignore[wrong-arg-types]
    module = self.clone(kwargs=model_kwargs)
    kwargs["model_mode"] = model_mode
    return nnx_wrappers.ToLinen.init(module, *args, **kwargs)

  def apply(self, *args, model_mode: str = MODEL_MODE_TRAIN, **kwargs):
    \"\"\"Applies the model.\"\"\"
    model_kwargs = self.kwargs.copy({"model_mode": model_mode})  # type: ignore[wrong-arg-types]
    module = self.clone(kwargs=model_kwargs)
    kwargs["model_mode"] = model_mode
    return nnx_wrappers.ToLinen.apply(module, *args, **kwargs)


class Transformer(nnx.Module):
  \"\"\"An autoregressive transformer model.\"\"\"

  # Make new attributes required, so that all Transformer dependencies (train, decode,
  # compile, etc) will error instead of silently use defaults.
  # pylint: disable=attribute-defined-outside-init
  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: Quant,
      *,
      model_mode: str = MODEL_MODE_TRAIN,
      rngs: nnx.Rngs,
  ):
    \"\"\"Initialize shared_embedding & decoder layers.\"\"\"
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode

    cfg = self.config
    mesh = self.mesh
    self.token_embedder = Embed(
        mesh=self.mesh,
        num_embeddings=cfg.vocab_size,
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        config=cfg,
        rngs=rngs,
    )
    self.vision_encoder = VisionEncoder(config=cfg, mesh=mesh, rngs=rngs) if cfg.use_multimodal else None
    self.audio_encoder = AudioEncoder(config=cfg, mesh=mesh, rngs=rngs) if cfg.use_audio else None
    if cfg.pure_nnx_decoder:
      self.decoder = NNXDecoder(config=cfg, mesh=mesh, quant=self.quant, model_mode=self.model_mode, rngs=rngs)
    else:
      decoder_linen = Decoder(config=cfg, mesh=mesh, quant=self.quant, model_mode=self.model_mode)
      self.decoder = nnx_wrappers.ToNNX(decoder_linen, rngs=rngs)
    self.hidden_states = None

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config=cfg, model_mode=model_mode)
    dummy_decoder_input_tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    dummy_decoder_positions = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    if self.config.attention == "vllm_rpa":
      try:
        # pylint: disable=import-outside-toplevel
        from tpu_inference.layers.common.attention_metadata import AttentionMetadata  # pytype: disable=import-error
      except ImportError as e:
        raise ImportError(
            "vLLM RPA attention requires the vllm-tpu package. Please install it with `pip install vllm-tpu`."
        ) from e
      dummy_attention_metadata = AttentionMetadata(
          input_positions=jnp.ones((batch_size * seq_len,), dtype=jnp.int32),
          block_tables=jnp.ones((seq_len,), dtype=jnp.int32),
          seq_lens=jnp.ones((1), dtype=jnp.int32),
          query_start_loc=jnp.ones((2), dtype=jnp.int32),
          request_distribution=jnp.ones((3), dtype=jnp.int32),
      )
    else:
      dummy_attention_metadata = None

    if not cfg.pure_nnx_decoder:
      self.decoder.lazy_init(
          shared_embedding=self.token_embedder,
          decoder_input_tokens=dummy_decoder_input_tokens,
          decoder_positions=dummy_decoder_positions,
          attention_metadata=dummy_attention_metadata,
      )

    # If MTP is enabled via config, set up the MTP block.
    if self.config.mtp_num_layers > 0:
      # Get the list of layer blueprints for the current model.
      layer_types = self.decoder.get_decoder_layers()
      # For MTP, we use the DecoderLayer blueprint to ensure architectural consistency.
      # By convention, this is the last layer in the list.
      mtp_layer = layer_types[-1]
      mtp_block_linen = multi_token_prediction_block_as_linen(
          config=self.config,
          mesh=self.mesh,
          transformer_layer_module=mtp_layer,
          decoder=self.decoder,
          rngs=rngs,
          name="mtp_block",
      )
      self.mtp_block = nnx_wrappers.ToNNX(mtp_block_linen, rngs=rngs)

      self.mtp_block.lazy_init(
          shared_embedding=self.token_embedder,
          main_hidden_state=jnp.ones((1, 1, self.config.emb_dim), dtype=self.config.dtype),
          input_ids=jnp.ones((1, 1), dtype=jnp.int32),
          target_ids=jnp.ones((1, 1), dtype=jnp.int32),
          target_mask=jnp.ones((1, 1), dtype=jnp.int32),
          position_ids=jnp.ones((1, 1), dtype=jnp.int32),
          decoder_segment_ids=jnp.ones((1, 1), dtype=jnp.int32),
          deterministic=True,
      )

  def no_op(self, *args, **kwargs):
    \"\"\"A no-op method to allow the model to be used in a lazy context.\"\"\"
    return

  def init_cache(self, cache_size: int, batch_size: int, dtype=jnp.float32):
    \"\"\"Initializes the KV cache for the Transformer.

    Args:
      cache_size: The maximum size of the KV cache.
      batch_size: The batch size for which the cache is initialized.
      dtype: Data type for the cache. Defaults to `jnp.float32`.

    Returns:
      True if the cache is successfully initialized.
    \"\"\"
    return True

  def __call__(
      self,
      decoder_input_tokens: jnp.ndarray,
      decoder_positions: jnp.ndarray,
      decoder_segment_ids=None,
      cache=None,
      encoder_images: jax.Array | None = None,
      encoder_image_masks: jax.Array | None = None,
      encoder_audios: jax.Array | None = None,
      enable_dropout=True,
      model_mode=MODEL_MODE_TRAIN,
      previous_chunk=None,
      true_length: int | None = None,
      slot: int | None = None,
      page_state: page_manager.PageState | None = None,
      decoder_target_tokens: jax.Array | None = None,
      decoder_target_mask: jax.Array | None = None,
      kv_caches: list[jax.Array] | None = None,
      attention_metadata: dict[str, Any] | None = None,
  ):
    \"\"\"Applies the Zero-1 FSDP wrapped Transformer model.

    This method handles the all-gather operation for model weights before
    applying the underlying Transformer model, and then releases them.

    Args:
      decoder_input_tokens: Input tokens for the decoder.
      decoder_positions: Positional encodings for the decoder inputs.
      decoder_segment_ids: Segment IDs for the decoder inputs (optional).
      encoder_images: Encoder images for multimodal models (optional).
      enable_dropout: Whether to enable dropout. Defaults to True.
      previous_chunk: Previous chunk for incremental decoding (optional).
      true_length: True length of the prompt before padding (optional).
      slot: An integer representing the decode batch index selected for this request (optional).
      page_state: Page state for paged attention (optional).
      partition_spec: Partition specification for FSDP all-gather.
      decoder_target_tokens: Target tokens for the decoder (optional, used in MTP).
      decoder_target_mask: Target mask for the decoder (optional, used in MTP).
      nnx_method: Method to call on the NNX module (optional).
      kv_caches: List of KV caches for each attention layer, used when invoking from vLLM (optional).
      attention_metadata: Mapping to store attention metadata, used when invoking from vLLM (optional).

    Returns:
      Logits from the Transformer model. Logits, hidden_state, kv_caches if called by vLLM.
    \"\"\"
    if decoder_segment_ids is not None and model_mode == MODEL_MODE_AUTOREGRESSIVE:
      raise ValueError(
          f"During autoregressive decoding we assume the tokens are in the active sequence"
          f" which is always {DECODING_ACTIVE_SEQUENCE_INDICATOR}."
      )

    bidirectional_mask = None
    image_embeddings = None
    deepstack_visual_embeds = None
    if self.config.use_multimodal and encoder_images is not None:
      image_embeddings, deepstack_visual_embeds = self.vision_encoder(
          input_images=encoder_images, deterministic=not enable_dropout
      )
      bidirectional_mask = mm_processor.get_bidirectional_mask_vision(self.config, decoder_input_tokens)

    audio_embeddings = None
    if self.config.use_multimodal and encoder_audios is not None and self.audio_encoder is not None:
      audio_embeddings = self.audio_encoder(input_audio=encoder_audios, deterministic=not enable_dropout)

    # Create audio mask for placeholder tokens (qwen3-omni models)
    audio_masks = None
    if audio_embeddings is not None:
      audio_masks = mm_processor.get_bidirectional_mask_audio(self.config, decoder_input_tokens)

    multimodal_input = None
    if image_embeddings is not None or audio_embeddings is not None:
      multimodal_input = MultimodalInput(
          image_embeddings=image_embeddings,
          image_masks=encoder_image_masks,
          audio_embeddings=audio_embeddings,
          audio_masks=audio_masks,
          bidirectional_mask=bidirectional_mask,
      )

    mutable_collections = []
    if self.config.record_internal_nn_metrics:
      mutable_collections.append("intermediates")
    if self.config.distill_beta > 0.0 and "intermediates" not in mutable_collections:
      mutable_collections.append("intermediates")
    if self.config.load_balance_loss_weight > 0.0 and "intermediates" not in mutable_collections:
      mutable_collections.append("intermediates")

    if self.config.pure_nnx_decoder:
      logits, hidden_state, kv_caches = self.decoder(
          shared_embedding=self.token_embedder,
          decoder_input_tokens=decoder_input_tokens,
          decoder_positions=decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          deterministic=not enable_dropout,
          model_mode=model_mode,
          previous_chunk=previous_chunk,
          slot=slot,
          page_state=page_state,
          multimodal_input=multimodal_input,
          kv_caches=kv_caches,
          attention_metadata=attention_metadata,
          deepstack_visual_embeds=deepstack_visual_embeds,
      )  # pytype: disable=wrong-keyword-args
    else:
      logits, hidden_state, kv_caches = self.decoder(
          shared_embedding=self.token_embedder,
          decoder_input_tokens=decoder_input_tokens,
          decoder_positions=decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          deterministic=not enable_dropout,
          model_mode=model_mode,
          previous_chunk=previous_chunk,
          slot=slot,
          page_state=page_state,
          multimodal_input=multimodal_input,
          kv_caches=kv_caches,
          attention_metadata=attention_metadata,
          deepstack_visual_embeds=deepstack_visual_embeds,
          mutable=mutable_collections,
      )  # pytype: disable=wrong-keyword-args

    # Materialize hidden state when vocab tiling is enabled
    if self.config.num_vocab_tiling > 1:
      self.hidden_states = hidden_state

    # If we are initializing the model AND MTP is enabled, we must create
    # dummy target tensors. This allows Flax to trace the MTPBlock and create
    # all its necessary parameters, without requiring the main training pipeline
    # to be aware of this initialization detail.
    # if self.is_initializing() and self.config.mtp_num_layers > 0:
    #   if decoder_target_tokens is None:
    #     dummy_shape = decoder_input_tokens.shape
    #     decoder_target_tokens = jnp.ones(dummy_shape, dtype=jnp.int32)
    #     decoder_target_mask = jnp.ones(dummy_shape, dtype=jnp.int32)
    #     decoder_segment_ids = jnp.ones(dummy_shape, dtype=jnp.int32)

    # The Multi-Token Prediction (MTP) block functions as a "side-car" to the main
    # model, active only during training. It computes an auxiliary loss based on
    # predicting multiple future tokens, as described in the DeepSeek-V3 paper.
    # To ensure architectural consistency, it uses two key components from the parent Transformer:
    #   1. The same `DecoderLayer` blueprint for its internal transformer blocks.
    #   2. The `shared_embedding` for both embedding future tokens and for its final
    #      logit projection.
    # Its only effect is to "sow" these losses; it does not alter the primary logits output.
    if self.config.mtp_num_layers > 0:
      self.mtp_block(
          shared_embedding=self.token_embedder,
          main_hidden_state=hidden_state,
          input_ids=decoder_input_tokens,
          target_ids=decoder_target_tokens,
          target_mask=decoder_target_mask,
          position_ids=decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          deterministic=not enable_dropout,
          model_mode=model_mode,
      )

    if self.config.attention == "vllm_rpa":
      # In vLLM, logits are computed separately after updating the KV cache.
      return hidden_state, kv_caches

    return logits
\n"""


# File: src/maxtext/utils/sharding.py (commit 313890777)
SHARDING_RAW = """\n# Copyright 2025-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=line-too-long, disable=bare-except, consider-using-generator
\"\"\" Utils that are only interesting to MaxText and sharding related. \"\"\"

from flax import linen as nn

from collections.abc import Iterable

import jax
from jax.core import Tracer
from jax.sharding import PartitionSpec as P, NamedSharding, reshard

import optax

from maxtext.common.common_types import ShardMode
from maxtext.utils import max_logging
from maxtext.utils import max_utils

import inspect  # for debugging only
from pathlib import Path

_LOGGED_ACTIVATION_SHARDINGS = set()
_ACTIVATION_SHARDINGS_DUMP = []


def clear_input_shardings_dump():
  \"\"\"Clear the input shardings dump\"\"\"
  _LOGGED_ACTIVATION_SHARDINGS.clear()
  _ACTIVATION_SHARDINGS_DUMP.clear()


def get_input_data_sharding(config, mesh):
  \"\"\"Get the input data sharding for the model\"\"\"
  if config.enable_diloco:
    data_sharding = create_sharding(
        mesh, ["diloco"] + config.input_data_sharding_logical_axes, rules=config.logical_axis_rules
    )
  else:
    data_sharding = create_sharding(mesh, config.input_data_sharding_logical_axes, rules=config.logical_axis_rules)
  return data_sharding


def _get_sharding_desc(inputs, extra_stack_level):
  \"\"\"Get the inputs sharding description using inspect module\"\"\"
  frame = inspect.currentframe()
  # Traverse back extra_stack_level times:
  for _ in range(1 + extra_stack_level):
    if frame is not None:
      frame = frame.f_back
  if frame is not None:
    callers_local_vars = frame.f_locals.items()

    x = [var_name for var_name, var_val in callers_local_vars if var_val is inputs]
    if len(x) > 0:
      caller_path_full = inspect.stack()[1 + extra_stack_level].filename
      # Use pathlib.Path to easily extract just the filename from the full path.
      caller_filename = Path(caller_path_full).name
      return f"{caller_filename[:-3]}/{x[0]}"
  return "Unknown"


def maybe_shard_with_name(
    inputs, named_sharding, shard_mode, debug_sharding=False, extra_stack_level=0, sharding_desc="", logical_axes=None
):
  \"\"\"
  In auto shardmode, this function hints inputs follow given named_sharding.
  In explicit shardmode, this function enforces inputs following named_sharding.
  sharding_desc is description of inputs of upper layer(s) of caller (with the form of <filename>/<variable>).
   It is used as key in log/dump files when debug_sharding==true
  \"\"\"
  if inputs is None:
    return None
  if (
      debug_sharding and isinstance(inputs, Tracer) and isinstance(named_sharding, NamedSharding)
  ):  # only print pspec for JitTracer
    if not sharding_desc:
      sharding_desc = _get_sharding_desc(inputs, extra_stack_level + 1)

    if not logical_axes:
      logical_axes = "Unknown"
    elif isinstance(logical_axes, list):
      logical_axes = tuple(logical_axes)

    pspec = remove_size_one_mesh_axis(getattr(named_sharding, "spec"), getattr(named_sharding, "mesh"))
    log_key = (sharding_desc, str(jax.typeof(inputs)), tuple(pspec), extra_stack_level)
    if log_key not in _LOGGED_ACTIVATION_SHARDINGS:
      max_logging.info(f"{sharding_desc} Logical: {log_key[1]:.<60} {logical_axes}.", stacklevel=3 + extra_stack_level)
      max_logging.info(f"{sharding_desc} Physical: {log_key[1]:.<60} {log_key[2]}.", stacklevel=3 + extra_stack_level)
      _LOGGED_ACTIVATION_SHARDINGS.add(log_key)

      _ACTIVATION_SHARDINGS_DUMP.append(
          {
              f"{sharding_desc}: {log_key[1]}": {
                  "logic_axes": f"{logical_axes}",
                  "PartitionSpec": f"P{log_key[2]}",
              }
          }
      )
  if shard_mode == ShardMode.EXPLICIT:
    return reshard(inputs, named_sharding)
  else:
    return jax.lax.with_sharding_constraint(inputs, named_sharding)


def maybe_shard_with_pspec(
    inputs, pspec: jax.sharding.PartitionSpec | None, mesh, shard_mode, debug_sharding=False, extra_stack_level=0
):
  if pspec is None:
    return None
  sharding = NamedSharding(mesh, pspec)
  return maybe_shard_with_name(
      inputs,
      sharding,
      shard_mode=shard_mode,
      debug_sharding=debug_sharding,
      extra_stack_level=extra_stack_level + 1,
  )


def maybe_shard_with_logical(
    inputs, logical_axes, mesh, shard_mode, rules=None, debug_sharding=False, extra_stack_level=0, sharding_desc=""
):
  \"\"\"
  A wrapper of maybe_shard_with_name when logical axes are inputs
  sharding_desc is description of inputs of upper layer(s) of caller (with the form of <filename>/<variable>).
   It is used as key in log/dump files when debug_sharding==true
  \"\"\"
  if inputs is None:
    return None

  if debug_sharding and not sharding_desc:
    sharding_desc = _get_sharding_desc(inputs, extra_stack_level + 1)

  named_sharding = create_sharding(mesh, logical_axes, rules=rules)

  return maybe_shard_with_name(
      inputs,
      named_sharding,
      shard_mode,
      debug_sharding=debug_sharding,
      extra_stack_level=extra_stack_level + 1,
      sharding_desc=sharding_desc,
      logical_axes=logical_axes,
  )


def remove_size_one_mesh_axis(spec, mesh):
  \"\"\"
  Removes mesh axes from a PartitionSpec (P) where the axis size is 1.

  This is a common optimization to simplify sharding by excluding redundant axes.
  Function originally from jax._src.core:
  https://github.com/jax-ml/jax/blob/main/jax/_src/core.py
  \"\"\"
  if spec is None:
    return None
  new_spec = []  # type: ignore
  for s in spec:
    if s is None or s == P.UNCONSTRAINED:
      new_spec.append(s)  # type: ignore
    elif isinstance(s, tuple):
      new_spec.append(tuple(i for i in s if mesh.shape.get(i, 1) != 1))
    else:
      new_spec.append(None if mesh.shape.get(s, 1) == 1 else s)  # type: ignore
  return P(*new_spec, unreduced=spec.unreduced, reduced=spec.reduced)


def logical_to_mesh_axes(logical_names, mesh, rules=None):
  \"\"\"Remove size one mesh axes given logical names.\"\"\"
  tensor_spec = nn.logical_to_mesh_axes(logical_names, rules=rules)
  return remove_size_one_mesh_axis(tensor_spec, mesh)


def logical_to_mesh(tree, mesh, rules=None):
  \"\"\"Remove size one mesh axes given logical pspec pytree.\"\"\"
  if tree is None:
    return None
  return jax.tree.map(
      lambda x: logical_to_mesh_axes(x, mesh, rules=rules),
      tree,
      is_leaf=lambda x: isinstance(x, P),
  )


def logical_to_mesh_sharding(tree, mesh, rules=None):
  \"\"\"Return sharding pytree given logical specs pytree\"\"\"
  if tree is None:
    return None
  return jax.tree.map(
      lambda x: NamedSharding(mesh, x),
      logical_to_mesh(tree, mesh, rules=rules),
      is_leaf=lambda x: isinstance(x, P),
  )


def create_sharding(mesh, logical_names, rules=None):
  \"\"\"Create NamedSharding with given logical names.\"\"\"
  return NamedSharding(mesh, logical_to_mesh_axes(logical_names, mesh, rules=rules))


def get_mesh_axes_used_by_tensor_spec(tensor_sharding_spec):
  \"\"\"
  Extracts the set of mesh axis names that a tensor's PartitionSpec uses.

  This function inspects a tensor's sharding specification (PartitionSpec) and
  identifies which mesh axes are actively used for sharding. If a tensor is not
  sharded (i.e., fully replicated), the resulting set will be empty.

  Args:
    tensor_sharding_spec: The PartitionSpec of a tensor, which defines how it's partitioned across the mesh.
    It can be None or contain strings and iterables representing the mesh axes.
    all_mesh_axis_names: A collection of all available mesh axis names in the current device mesh.

  Returns:
    A set of strings, where each string is a mesh axis name used by the
    tensor's sharding spec. Returns an empty set for unsharded tensors.
  \"\"\"
  # Flatten the sharding spec, as it can contain nested iterables (e.g., ('data', 'mdl')).
  tensor_sharding_spec = sum(
      [
          [axis] if isinstance(axis, str) else list(axis) if isinstance(axis, Iterable) else []
          for axis in tensor_sharding_spec
      ],
      [],
  )
  return tensor_sharding_spec


def _get_nontrival_mesh_axes(mesh):
  \"\"\"
  Returns mesh axes from config that are valid and have more than one shard.

  This function identifies which of the predefined potential sharding axes are
  actually present in the current device mesh and are configured with a size
  greater than one (i.e., are actually sharded).

  Args:
    mesh: The device mesh object, which contains information about the mesh topology, including axis names and their sizes.

  Returns:
    A set of strings, where each string is a mesh axis name that is both
    pre-configured as a target for sharding and has more than one shard in the mesh.
  \"\"\"

  target_sharding_axes_config = [
      "fsdp",
      "fsdp_transpose",
      "sequence",
      "context",
      "context_autoregressive",
      "tensor",
      "tensor_transpose",
      "tensor_sequence",
      "stage",
      "expert",
  ]

  # Filter the target axes to find those that exist in the current mesh
  # and have a size greater than 1, meaning they are actually used for sharding.
  return {axis for axis in target_sharding_axes_config if axis in mesh.axis_names and mesh.shape[axis] > 1}


def _analyze_sharding(params, mesh, valid_target_mesh_axes):
  \"\"\"
  Analyzes parameters to find which are unsharded on any valid mesh axis.

  This function iterates through all parameters in a model, checking their
  sharding specifications. It identifies parameters that are not sharded along any
  of the provided valid target axes (i.e., they are fully replicated across these axes).

  Args:
    params: A PyTree of model parameters.
    mesh: The device mesh object.
    valid_target_mesh_axes: A set of mesh axis names that are considered valid targets for sharding.

  Returns:
    A tuple containing:
      - unsharded_params_total_size (int): The total size (number of elements) of all parameters found to be
        unsharded on the target axes.
      - problematic_tensors_details (list): A list of dictionaries, where each
        dictionary contains details about a tensor that is not sharded on any of the target axes.
  \"\"\"
  unsharded_params_total_size = 0  # Initialize a counter for the size of unsharded parameters.
  problematic_tensors_details = []  # Initialize a list to store details of problematic tensors.

  # Get a flattened list of all parameters (leaves) in the PyTree, along with their paths.
  all_params_leaves = jax.tree_util.tree_leaves_with_path(params)

  for path, p_leaf in all_params_leaves:  # Iterate over each parameter leaf
    param_name_str = jax.tree_util.keystr(path)  # Convert the tree path to a readable string

    # Check that sharding and spec exist and are valid
    sharding = getattr(p_leaf, "sharding", None)
    spec = getattr(sharding, "spec", None)
    assert sharding is not None and spec is not None and isinstance(spec, P), (
        f"Parameter '{param_name_str}' is missing a valid '.sharding.spec'."
        "Expected 'p_leaf.sharding.spec' to be a non-null 'partitionspec'."
    )

    current_sharding_spec = p_leaf.sharding.spec  # Extract the current tensor's sharding spec
    # Identify axes used for sharding
    mesh_axes_used = get_mesh_axes_used_by_tensor_spec(current_sharding_spec)
    # Check if the parameter is sharded on all the valid target axes.
    is_sharded_on_all_target_axis = all(axis in mesh_axes_used for axis in valid_target_mesh_axes)

    # If the parameter is not sharded on all of the target axes, it's considered "problematic."
    if not is_sharded_on_all_target_axis:
      unsharded_params_total_size += p_leaf.size  # Add to total unsharded parameter size
      unsharded_axes = set(valid_target_mesh_axes) - set(mesh_axes_used)
      # Add detailed info to list of problematic tensors
      problematic_tensors_details.append(
          {
              "name": param_name_str,  # Tensor name
              "size": p_leaf.size,  # tensor size
              "shape": p_leaf.shape,  # tensor shape
              "spec": str(current_sharding_spec),  # Tensor sharding spec as string
              "available_axes": sorted(list(valid_target_mesh_axes)),  # Axes that could be used for sharding
              "unsharded_axes": sorted(list(unsharded_axes)),  # Unsharded axes
          }
      )
  # Return the total size of unsharded parameters and the list of problematic tensors.
  return unsharded_params_total_size, problematic_tensors_details  # Return results


def _raise_if_unsharded_exceeds_tolerance(unsharded_size, total_size, tolerance, problematic_tensors_details):
  \"\"\"
  Raises an AssertionError if the percentage of unsharded parameters exceeds the given tolerance.

  This function calculates the proportion of model parameters that are unsharded
  and compares it against a specified tolerance. If the tolerance is exceeded,
  it constructs and raises a detailed error message.

  Args:
    unsharded_size: The total size of parameters not sharded on target axes.
    total_size: The total size of all parameters in the model.
    tolerance: A float (e.g., 0.05 for 5%) representing the maximum allowed percentage of unsharded parameters.
    problematic_tensors_details: A list of details about the unsharded tensors,
    used to generate an informative error message.

  Raises:
    AssertionError: If the percentage of unsharded parameters is greater than the tolerance.
  \"\"\"
  if total_size <= 0:
    raise ValueError("Total size must be greater than zero.")

  # Calculate the percentage of unsharded parameters.
  unsharded_param_perc = unsharded_size / total_size

  # If the percentage is over the tolerance, prepare and raise an error.
  if unsharded_param_perc > tolerance:
    # Sort the problematic tensors by size to show the largest ones first.
    problematic_tensors_details.sort(key=lambda x: x["size"], reverse=True)

    # Begin constructing the error message.
    error_msg_lines = [
        f"Unsharded parameter percentage ({unsharded_param_perc:.2%})" f"exceeds tolerance ({tolerance:.2%})."
    ]
    # Add a header explaining the issue.
    error_msg_lines.append(
        "The following large tensors are replicated (unsharded) but could be sharded on at "
        "least one of the available axes:"
    )
    # Add details for the top 5 largest problematic tensors.
    for detail in problematic_tensors_details[:5]:  # Show top 5 largest problematic tensors
      error_msg_lines.append(
          f" - Name: {detail['name']}(Size: {detail['size']}, Shape: {detail['spec']}, Spec: {detail['spec']}) "
          f" is unsharded on axis: {detail['unsharded_axes']}"
          f" could be sharded on: {detail['available_axes']}"
      )

    # Raise the assertion error with the combined, formatted message.
    raise AssertionError("\n".join(error_msg_lines))


def assert_params_sufficiently_sharded(params, mesh, tolerance):
  \"\"\"
  Asserts that the total size of replicated parameters is within a given tolerance.

  This is the main function that orchestrates the sharding analysis. It determines
  the total number of parameters, identifies valid sharding axes, analyzes the
  sharding of all parameters, and then raises an error if the amount of
  unsharded parameters exceeds the specified tolerance.

  Args:
    params: A PyTree of model parameters.
    mesh: The device mesh object.
    tolerance: A float representing the maximum allowed percentage of unsharded parameters.
  \"\"\"
  # Calculate the total size of all parameters in the model.
  total_num_params = max_utils.calculate_bytes_from_pytree(params)

  # Get the set of nontrival mesh axes that can be used for sharding.
  valid_target_mesh_axes = _get_nontrival_mesh_axes(mesh)
  # If there are no valid axes to shard along, there's nothing to check, so we can exit.
  if not valid_target_mesh_axes:
    return  # Exit early

  # Analyze the parameters to find the total size of unsharded parameters
  # and get details on which tensors are problematic.
  unsharded_params_total_size, problematic_tensors_details = _analyze_sharding(params, mesh, valid_target_mesh_axes)

  # Check if the amount of unsharded parameters is within the tolerance and
  # raise an exception if it is not.
  _raise_if_unsharded_exceeds_tolerance(
      unsharded_params_total_size, total_num_params, tolerance, problematic_tensors_details
  )


def add_data_to_sharding(mesh, path, aval, sharding):
  \"\"\"Adds 'data' dimension to sharding spec if compatible and not already present.

  This function attempts to add data parallelism to a sharding specification by finding
  a dimension that is divisible by the 'data' mesh axis size and doesn't conflict with
  existing partitioning (e.g., tensor parallelism).
  This function is mainly used to add data parallelism to the optimizer state for Zero-1 style sharding.

  Args:
    mesh: The device mesh
    path: JAX tree path to the value being sharded
    aval: Abstract value with shape information
    sharding: Current NamedSharding to potentially augment

  Returns:
    NamedSharding: Updated sharding with 'data' dimension added, or original if unchanged

  Raises:
    AssertionError: If sharding is not NamedSharding or shape cannot be sharded
  \"\"\"
  if not isinstance(sharding, jax.sharding.NamedSharding):
    raise AssertionError(f"Expected NamedSharding, found {sharding} of {type(sharding)=} at {jax.tree_util.keystr(path)}")
  try:
    sharded_shape = sharding.shard_shape(aval.shape)
  except Exception as e:
    raise AssertionError(f"Could not shard {jax.tree_util.keystr(path)} of shape={aval.shape} with {sharding=}") from e
  pspec = sharding.spec

  if "data" in jax.tree.leaves(pspec):
    return sharding

  for idx, (size, partition) in enumerate(zip(sharded_shape, pspec)):
    if partition is None:
      partition = ()

    if isinstance(partition, str):
      partition = (partition,)

    if size % mesh.shape["data"] == 0 and (partition is None or "tensor" not in partition):
      added_component = ("data",) + partition
      new_pspec = jax.sharding.PartitionSpec(*(pspec[:idx] + (added_component,) + pspec[idx + 1 :]))
      new_sharding = jax.sharding.NamedSharding(sharding.mesh, new_pspec)
      return new_sharding
  return sharding


def maybe_update_params_sharding_with_opt(config, state_mesh_shardings):
  \"\"\"Updates parameter sharding configuration when optimizer state sharding is enabled.

  When shard_optimizer_over_data is enabled (Zero-1 style sharding), this function
  extracts the optimizer state shardings from the Adam optimizer's first moment (mu)
  and merges them with the parameter shardings. This ensures parameter sharding is
  consistent with how the optimizer state is distributed across the compute mesh.

  Args:
    config: Configuration object with shard_optimizer_over_data flag
    state_mesh_shardings: Train state mesh shardings containing params and opt_state

  Returns:
    A tuple of (prev_params_shardings, updated_state_mesh_shardings):
      - prev_params_shardings: Original parameter shardings before the update
      - updated_state_mesh_shardings: State mesh shardings with updated params field
        (unchanged if shard_optimizer_over_data is False)
  \"\"\"
  prev_params_shardings = state_mesh_shardings.params
  if config.shard_optimizer_over_data:
    if isinstance(state_mesh_shardings.opt_state, optax.ScaleByAdamState):
      sharded_fp32_params = state_mesh_shardings.opt_state.mu
    elif isinstance(state_mesh_shardings.opt_state, tuple) and isinstance(
        state_mesh_shardings.opt_state[0], optax.ScaleByAdamState
    ):
      sharded_fp32_params = state_mesh_shardings.opt_state[0].mu
    else:
      raise NotImplementedError(f"Could not find optimizer state shardings from {type(state_mesh_shardings.opt_state)}")
    if "params" not in sharded_fp32_params.keys():
      # When quantization=fp8 is enabled the sharded_fp32_params
      # are not wrapped in `params`. Here we wrap them back.
      sharded_fp32_params = {"params": sharded_fp32_params}
    state_mesh_shardings = state_mesh_shardings.replace(params=dict(prev_params_shardings, **sharded_fp32_params))
  return prev_params_shardings, state_mesh_shardings


def logical_axis_rules_pp_act_as_dp(logical_rules):
  \"\"\"Add stage as a physical axes before data for each rule, so stage acts just like data instead of PP.
  This is used when we want to pipeline only a subset of layers, and leave the rest like DP.
  \"\"\"
  new_rules = []
  for key, physical_axes in logical_rules:
    if isinstance(physical_axes, str):
      physical_axes = (physical_axes,)
    else:
      physical_axes = tuple(physical_axes)
    new_physical_axes = tuple(axis for axis in physical_axes if axis != "stage")
    if "data" in new_physical_axes:
      data_idx = new_physical_axes.index("data")
      new_physical_axes = new_physical_axes[0:data_idx] + ("stage",) + new_physical_axes[data_idx:]
    new_rules.append((key, new_physical_axes))
  return tuple(new_rules)


def get_formatted_sharding_annotations(params, mesh=None):
  \"\"\"
  Generates a readable string report of sharding annotations for all parameters.

  This function iterates through a PyTree of model parameters and inspects the
  sharding information attached to each parameter (leaf). It creates a
  human-readable summary that is useful for debugging sharding configurations.

  Args:
    params: The PyTree of model parameters to inspect.
    mesh: (Optional) The device mesh. If provided, its axis names and shape
          are included in the report for additional context.

  Returns:
    A single string containing the formatted report of sharding annotations
    for every parameter, with each entry on a new line.
  \"\"\"
  # Initialize a list to hold the lines of the report, starting with a title.
  annotation_lines = ["Comprehensice Weight Sharding Annotations:"]

  # If a mesh object is provided, add its details to the report header.
  if mesh:
    annotation_lines.append(f"Mesh axes: {mesh.axis_names}, Mesh shape: {mesh.shape}")
    annotation_lines.append("-" * 30)

  # Get a flattened list of all parameters (leaves) and their corresponding paths in the PyTree.
  all_params_leaves = jax.tree_util.tree_leaves_with_path(params)

  # Loop through each parameter leaf in the flattened list.
  for path, p_leaf in all_params_leaves:
    # Convert the parameter's path (a sequence of keys) into a readable string name.
    param_name_str = jax.tree_util.keystr(path)
    # Get the shape of the parameter as a string.
    shape_str = str(p_leaf.shape)
    # Set a default description for sharding, in case none is found.
    sharding_desc = "N/A"

    # Check if the parameter leaf has a 'sharding' attribute.
    if hasattr(p_leaf, "sharding"):
      # Case 1: Standard JAX sharding with a PartitionSpec.
      if hasattr(p_leaf.sharding, "spec") and p_leaf.sharding.spec is not None:
        # The spec is a tuple (PartitionSpec), format it for readability.
        spec_parts = []
        for item in p_leaf.sharding.spec:
          # Represent None as "Replicated" to make it explicit.
          spec_parts.append(str(item) if item is not None else "Replicated")
        sharding_desc = f"PartitionSpec({', '.join(spec_parts)})"
      # Case 2: The parameter is explicitly marked as fully replicated.
      elif hasattr(p_leaf.sharding, "spec") and p_leaf.sharding.spec is None:
        sharding_desc = "Fully Replicated (spec is None)"
      # Case 3: A generic fallback if a sharding object exists but has no recognized spec attribute.
      else:
        # Print the string representation of the sharding object itself.
        sharding_desc = str(p_leaf.sharding)
    # Case 4: The parameter has no .sharding attribute at all.
    else:
      sharding_desc = "No .sharding attribute found"

    # Append the formatted details for the current parameter to our list of lines.
    annotation_lines.append(f" - Param: {param_name_str}\n" f"   Shape: {shape_str}\n" f"   Sharding: {sharding_desc}")
  # Join all the collected lines into a single string, separated by newlines.
  return "\n".join(annotation_lines)


def remove_fsdp_sharding(sharding_tree):
  \"\"\"Recursively traverses the sharding tree to remove fsdp axes.\"\"\"

  def _remove_fsdp_from_partition_spec(named_sharding):
    \"\"\"Removes 'fsdp' and 'fsdp_transpose' from a PartitionSpec.\"\"\"
    if isinstance(named_sharding, jax.sharding.NamedSharding):
      new_spec = []
      # Iterate through each axis in the original PartitionSpec.
      for axis in named_sharding.spec:
        if axis is None:
          new_spec.append(None)
        elif isinstance(axis, str):
          # If the axis is 'fsdp', replace it with None to signify replication.
          if axis not in ("fsdp", "fsdp_transpose"):
            new_spec.append(axis)
          else:
            new_spec.append(None)
        elif isinstance(axis, (list, tuple)):
          # If the axis is a collection, filter out 'fsdp'.
          new_axis = [a for a in axis if a not in ("fsdp", "fsdp_transpose")]
          new_spec.append(tuple(new_axis))
        else:
          raise ValueError(f"Unsupported_axis_type: {type(axis)}")
        # Return a new sharding object with the modified spec.
      return jax.sharding.NamedSharding(named_sharding.mesh, jax.sharding.PartitionSpec(*new_spec))
    return named_sharding

  return jax.tree.map(_remove_fsdp_from_partition_spec, sharding_tree)


def get_physical_spec_no_fsdp(full_logical, mesh, logical_axis_rules):
  \"\"\"
  Generates a physical sharding spec for fully replicated weights.

  This function computes a target sharding layout where model parameters are fully
  replicated across the 'fsdp' mesh axis. It starts with the original logical
  sharding and removes any rules that shard along the 'fsdp' or
  'fsdp_transpose' axes.

  Replacing a sharding axis with `None` in a PartitionSpec instructs JAX to
  replicate the array data along that physical mesh dimension. The resulting
  specification is used as a target layout for an all-gather operation.

  Args:
    full_logical: A PyTree of logical PartitionSpecs for the model parameters.
    mesh: The JAX device mesh.
    logical_axis_rules: Rules for converting logical axes to physical mesh axes.

  Returns:
    A PyTree of physical `jax.sharding.NamedSharding` objects that describe a
    layout where parameters are fully gathered (replicated) across the 'fsdp'
    mesh axis.
  \"\"\"

  # Convert the high-level logical spec to a physical one using default rules.
  physical = logical_to_mesh_sharding(full_logical, mesh=mesh, rules=logical_axis_rules)
  # Apply the function to remove the FSDP sharding, defining our target layout.
  physical_no_fsdp = remove_fsdp_sharding(physical)
  return physical_no_fsdp


def all_gather_over_fsdp(variables, sharding_info, mesh, logical_axis_rules, shard_mode):
  \"\"\"Performs an all-gather on FSDP-sharded variables via a sharding constraint.
  This function triggers an all-gather operation on the model's parameters.
  It does so by applying a sharding constraint that specifies a fully
  replicated layout.

  The JAX compiler satisfies this constraint by automatically inserting the
  necessary `all-gather` collective communication operations into the
  computation graph, effectively gathering the sharded weights.

  Args:
    variables: The PyTree of model parameters, currently sharded across devices.
    sharding_info: The logical partition spec of the currently sharded `variables`.
    mesh: The JAX device mesh.
    logical_axis_rules: Rules for converting logical axes to physical mesh axes.
    shard_mode: auto or explicit shard mode.

  Returns:
    The model's variables with the all-gather operation applied, resulting
    in the weights being fully replicated on all devices in the 'fsdp' mesh.
  \"\"\"
  # Get the target physical layout (weights fully replicated).
  physical_constraint_no_fsdp = get_physical_spec_no_fsdp(sharding_info, mesh, logical_axis_rules)
  # Apply the constraint to the model's current variables. This tells JAX to
  # gather the weights into this layout.
  return maybe_shard_with_name(variables, physical_constraint_no_fsdp, shard_mode=shard_mode)
\n"""


# File: src/maxtext/configs/base.yml (commit 313890777)
BASE_CONFIG_RAW = """\n# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This sentinel is a reminder to choose a real run name.
# If there is already a checkpoint under this run, that checkpoint will auto-resume.
#
# NOTE: Some unit/integration tests in MaxText do not always run this file directly.
# When running in decoupled mode (DECOUPLE_GCLOUD=TRUE), tests may use
# `decoupled_base_test.yml` instead of `base.yml` via `tests/utils/test_helpers.py`.
run_name: ""

model_name: "default" # override config settings to match a specific model. other than the override, nothing should use this!
override_model_config: False # When set to true allows overriding model parameters via CLI (or kwargs or env vars) for the purpose of debugging/testing.
override_logical_axis_rules: False # When set overrides logical axis rules instead of merging them.
debug:
  rl: False # RL-specific debugging

normalization_layer_epsilon: 1.e-05 # epsilon value for rmsnorm, layernorm.

################################## CHECKPOINTING ##################################
# Checkpointing makes the following choices in the following order, starting with (1):
#   (1) If there is already a checkpoint for this run_name, we load the latest entire checkpoint.
#     This ensures if we're resuming a run after preemption or hardware failure we lose minimum state.
#   (2) Same priority and mutually exclusive -- you can't set both!
#      * If load_parameters_path is set, we load a parameter only checkpoint from that path.
#      * If load_full_state_path is set, we load a full state checkpoint from that path.
#   (3) We don't load a checkpoint and initialize state instead!

# Loads a just parameters from a specific directory
# e.g. gs://my-base-output-directory/my-previous-run-name/checkpoints/items/NUMBER or NUMBER/items
load_parameters_path: ""

# LoRA adapter support configs
lora_input_adapters_path: ""    # Input GCS path for a parent directory which has all the LoRA adapters (lora_id as subdir)

# Loads a full checkpoint including optimizer state and step count from a specific directory
# e.g. gs://my-base-output-directory/my-previous-run-name/checkpoints/items/NUMBER or NUMBER/items
load_full_state_path: ""

# If enable_checkpointing is true, an asynchronous checkpointer will be used if
# async_checkpointing is true, else a synchronous one is used. If you have
# problems with the checkpointer we recommend trying the synchronous one.
enable_checkpointing: True
save_checkpoint_on_completion: True
async_checkpointing: True
checkpoint_period: 10_000
max_num_checkpoints_to_keep: None
enable_continuous_checkpointing: False
# enables one replica to read the ckpt then broadcast to the rest
enable_single_replica_ckpt_restoring: False
# Subdirectory to move checkpoints to before deletion. For example: ".todelete" (Ignored if directory is prefixed with gs://)
checkpoint_todelete_subdir: None
# Full path to move checkpoints to before deletion.
checkpoint_todelete_full_path: None

force_unroll: False # during generate_param_only_checkpoint should we unroll the loop?

# checkpointing using orbax has two important parameters: array driver
# and its underlying storage - the kvstore (preferably ocdbt)
# orbax supports setting a target file size, chunking a single
# large arrays into small physical files (<2GB) can speed up distributed and over
# the network loading enormously
checkpoint_storage_target_data_file_size_bytes: 2147483648
checkpoint_storage_use_ocdbt: True
checkpoint_storage_use_zarr3: True
# larger models requires higher concurrent GB for I/O
# default concurrent gb for PytreeCheckpointHandler is 96GB
checkpoint_storage_concurrent_gb: 96

# Bool flag for enabling Orbax v1.
enable_orbax_v1: False
# function for processing loaded checkpoint dict into a format maxtext can understand. (for other formats, i.e. safetensors)
checkpoint_conversion_fn: none
# optional checkpoint context to use for loading. options: "orbax", "safetensors"
source_checkpoint_layout: "orbax"

# Only applicable to Single Controller/Pathways on Cloud. Experimental feature, under testing
colocated_python_checkpointing: False

# enables autocheckpoint, which saves a checkpoint at the preemption step.
enable_autocheckpoint: False
############################### end checkpointing ##################################


reuse_example_batch: 0 # for testing tpu performance, this options repeated uses the same batch.


metrics_file: "" # for testing, local file that stores scalar metrics. if empty, no metrics are written.
# if true save metrics such as loss and tflops to gcs in {base_output_directory}/{run_name}/metrics/
gcs_metrics: false

# if true save config to gcs in {base_output_directory}/{run_name}/
save_config_to_gcs: false

# gradient dtype
grad_dtype: "float32"

# activation dtypes.
dtype: "bfloat16"
# used to configure quantization in the transformer layers, defaults to null implying bf16.
# possible alternative settings are as follows:
# 'int8' for dynamic range quantization using 8-bits
# 'intmp' for mixed precision quantization for inference as described here: src/maxtext/configs/quantization/readme.md
# 'fp8' for 8-bit floating-point gemms on nvidia gpus.
# 'nanoo_fp8' for 8-bit floating-point gemms on amd mi300/mi325 gpus.
# 'fp8_full' for fp8 quantization with static scaling.
quantization: ""
# used to configure constant_bound_config in aqt lib for static scaling, e.g. constant_bound_config='0.5, 0.5, 0.5, 0.5, 0.5, 0.5'
constant_bound_config: ""
# choose one of default, high, and highest.
# https://kolonist26-jax-kr.readthedocs.io/en/latest/jax.lax.html#jax.lax.precision
matmul_precision: "default"
activations_in_float32: false # sets activations to float32 before nonlinearity it true, else dtype
# used to replicate the quantization scale to avoid the inefficient xla fusion for 2d sharding.
replicate_quant_scale: false
# path to file with quantization config for intmp.
quant_cfg_path: ""
quantize_kvcache: false # set to true to quantize kv cache values, defaults to false
# valid kv_quant_axis values:
#   - "" is valid only when quantize_kvcache is false
#   - "dkv" indicates quantize kv cache over the cache_kv, i.e. kv dimension axis
#   - "heads_and_dkv" indicates quantize kv cache over cache_heads and cache_kv axes
# default to "heads_and_dkv" for faster compution, kv_quant_axis is not used when quantize_kvcache is false
#   - "dkv" is expected with better accuracy but degraded computation
kv_quant_axis: "heads_and_dkv"
kv_quant_dtype: "int8"
checkpoint_is_quantized: false # set to true if reading from a saved aqt quantized checkpoint
# saves params quantized on fly at following path
save_quantized_params_path: ""
#used to configure the mode in which model is called
# when left as is, corresponds to training
# accepted values are "inference"
model_call_mode: ""
use_qwix_quantization: false # whether to use qwix for quantization. if set to true, the model will be quantized using qwix.
use_manual_quantization: false # a flag if to use manual quantization for batch split. Only used if use_batch_split_schedule is True.
# quantization calibration method used for weights and activations. supported methods can be found in https://github.com/google/qwix/blob/dc2a0770351c740e5ab3cce7c0efe9f7beacce9e/qwix/qconfig.py#l70-l80
weight_quantization_calibration_method: "absmax"
act_quantization_calibration_method: "absmax"
bwd_quantization_calibration_method: "absmax"
# shard the range finding operation for quantization. by default this is set to number of slices.
quantization_local_shard_count: -1
# The 'N' in N:M sparsity, representing the maximum number of non-zero values in each block.
weight_sparsity_n: null
# The 'M' in N:M sparsity, representing the number of values in each block.
weight_sparsity_m: null
# The step size to update the sparsity masks.
weight_sparsity_update_step: 10
# The first number of steps before updating the sparsity masks.
weight_sparsity_start_step: 50

decoder_block: "llama2" # which style of decoderblock to use.
# global parameter scale needs to be a power of 2. if you want finer grained control of the model sizes
# then you should explicitly set base_embed_dim, base_num_query_heads, base_num_kv_heads,
# base_mlp_dim, base_num_decoder_layers and/or head_dim.
weight_dtype: "float32"
global_parameter_scale: 1
base_emb_dim: 2048
base_num_query_heads: 16
base_num_kv_heads: 16
base_mlp_dim: 7168
dense_init_scale: 1.0
base_num_decoder_layers: 16
head_dim: 128
attention_output_dim: -1
# Those parameters are only used with global attention for Gemma4.
global_head_dim: 0
global_num_kv_heads: 0
mlp_activations: ["silu", "linear"]
mlp_activations_limit: -1.0
dropout_rate: 0.0
logits_via_embedding: false
normalize_embedding_logits: true  # whether to normalize pre-softmax logits if logits_via_embedding is true
logits_dot_in_fp32: false  # whether to use fp32 in logits_dense or shared_embedding dot product for stability
cast_logits_to_fp32: true # whether to cast the logits to fp32. the higher precision is generally beneficial, but it can vary slightly.
float32_qk_product: false # in dot_product attention, whether to cast to fp32 the inputs to qk product
float32_logits: false # in dot_product attention, whether to cast to fp32 the inputs to softmax
float32_weight_sum: true # whether to use full fp32 precision to sum expert weights for numerical stability
float32_gate_logits: false # whether to cast inputs to fp32 to compute MoE gate logits for numerical stability

# multi-token prediction configs
# the number of auxiliary prediction layers to use for mtp.
# set to 0 to disable the feature.
mtp_num_layers: 0
# the scaling factor (lambda) for the mtp auxiliary loss. the final loss is:
# main_loss + mtp_loss_scaling_factor * avg_mtp_loss
mtp_loss_scaling_factor: 0.1
# specifies which mtp layer (1-indexed) is used to calculate metrics like the
# acceptance rate during evaluation. for example, a value of `1` targets the
# first auxiliary prediction head. set to 0 to disable this specific evaluation
mtp_eval_target_module: 0

# mixture of experts (moe)
num_experts: 1
num_experts_per_tok: 1
megablox: true
sparse_matmul: true
capacity_factor: -1.0 # a factor to decide expert capacity for token dropping, and no dropping by default
ragged_buffer_factor: -1.0 # a factor to determine the size of the ragged buffer for routed MoE activations.
# By default (-1), the routed buffer is worst case size to ensure no dropping.
# When set to 1.0 this buffer if set to the size assuming perfectly balanced. If the routing dictates
# a size larger than this then tokens are dropped.
# In general if ragged_buffer_factor > 0, the ragged_buffer_size is balanced_size * ragged_buffer_factor.
moe_expert_input_dim: -1 # feature dimension of the tokens entering the MoE expert blocks.
base_moe_mlp_dim: -1 # intermediate dimension at MoE layer. For a fully MoE model, base_mlp_dim must be equal to base_moe_mlp_dim.
load_balance_loss_weight: 0.0 # weight for the load balance loss
use_random_routing: false # whether to use random routing for debug/test purpose
use_custom_sort_vjp: true # whether to use a custom VJP sort for efficient backward pass processing in sparse matmul
use_ring_of_experts: false # whether to use ring of experts for sparse matmul expert parallelism
use_gather_mosaic_kernel: false # whether to use a custom mosaic kernel for token gather ops
# tunable tiling dimensions used for mlp gmm
# megablox/jax ragged dot - supports forward pass only (6 configs: `wi_tile_fwd...` and `wo_tile_fwd_...`)
# tokamax ragged dot - supports all 18 configs
wi_tile_fwd_batch_seq: 512
wi_tile_fwd_embed_dim: 1024
wi_tile_fwd_mlp_dim: 1024
wi_tile_dlhs_batch_seq: 512
wi_tile_dlhs_embed_dim: 1024
wi_tile_dlhs_mlp_dim: 1024
wi_tile_drhs_batch_seq: 512
wi_tile_drhs_embed_dim: 1024
wi_tile_drhs_mlp_dim: 1024

wo_tile_fwd_batch_seq: 512
wo_tile_fwd_embed_dim: 1024
wo_tile_fwd_mlp_dim: 1024
wo_tile_dlhs_batch_seq: 512
wo_tile_dlhs_embed_dim: 1024
wo_tile_dlhs_mlp_dim: 1024
wo_tile_drhs_batch_seq: 512
wo_tile_drhs_embed_dim: 1024
wo_tile_drhs_mlp_dim: 1024

merge_gating_gmm: False

norm_topk_prob: false # boolean to enable the top-k probability normalization. qwen3-specific normalization of router weights.

# when moe weight matrices are sharded on both fsdp and fsdp-transpose axes, use two separate all-gather calls
moe_fsdp_use_two_stage_all_gather: false
# Shard the expert dimension of the MLP weights on the FSDP axis.
# This configuration is recommended only when num_experts is a multiple of fsdp_parallelism
shard_exp_on_fsdp: False
# use fsdp and fsdp_transpose axes for sharding the moe weights
use_2d_fsdp_sharding: False

# deepseek moe
first_num_dense_layers: 0 # number of initial dense layers in the model
shared_experts: 0
routed_scaling_factor: 1.0 # scaling factor for routing scores
routed_score_func: "" # scoring function for routing
routed_bias: False # a flag if a learnable bias is added for routing
routed_bias_update_rate: 0.0 # a flag indicate the update rate applied to the router bias term
mlp_bias: False # a flag if a learnable bias is added for MLP matmul, and originally implemented to support the GPT-OSS model architecture.
n_routing_groups: -1 # number of groups for routing, disabled by default
topk_routing_group: -1 # number of top groups to route inputs. For EP,
# Splits the batch to allow for better scheduling when using expert parallelism by overlapping the
# all-to-all communication with compute. Currently only implemented with DeepSeek sparse layers.
use_batch_split_schedule: False # a flag if splitting batch into micro-batches to hide communications that yields performance benefits.
batch_split_factor: 1 # the factor by which to split the batch. Only used if use_batch_split_schedule is True.

# For complex architectures like llama4 there are repeated sets of
# inhomogeneous layers. E.g. maverick uses [dense+rope, moe+rope, dense+rope, moe+nope]
# which can only be scanned together in one large block of inhomogeneous_layer_cycle_interval=4 layers.
inhomogeneous_layer_cycle_interval: 1

# pipeline parallelism
# The number of decoder layers is equal to the product of num_stages, num_layers_per_pipeline_stage and num_pipeline_repeats.
# There is a tradeoff between the num_layers_per_pipeline_stage and num_pipeline_repeats: The more layers per stage the easier
# it is to hide the pipeline communication behind the compute since there is more compute per stage, however there will be a larger bubble
# since there are fewer repeats. Similarly, there is tradeoff for num_pipeline_microbatches - more microbatches leads to a smaller bubble,
# but a smaller size per microbatch which may hurt per-stage performance. Additionally, note when microbatches > num_stages we have the opportunity to
# perform the circular transfer (last stage to first) asynchronously.
# The bubble fraction is (num_stages - 1) / (num_pipeline_repeats * num_pipeline_microbatches + num_stages - 1)
num_layers_per_pipeline_stage: 1
# The number of repeats will be set to num_decoder_layers / (num_pipeline_stages * num_layers_per_pipeline_stage)
num_pipeline_repeats: -1
pipeline_parallel_layers: -1 # Pipeline only this number of layers - for the remaining layers the "stage" mesh axes will act like data parallelism.
# This option helps when the number of layers does not have friendly divisors since SPMD pipelining requires that the
# PP degree divides the number of layers.
# By default (when set to -1) we pipeline all of the decoder layers.

# num_pipeline_microbatches must be a multiple of the number of pipeline stages. By default it is set to the number of stages.
# Note the microbatch_size is given by global_batch_size / num_pipeline_microbatches, where global_batch_size = per_device_batch_size * num_devices
num_pipeline_microbatches: -1
pipeline_delay_activation_forwarding: False # This delays the activation forwarding one loop iteration simplifying XLA's task of overlapping since
# the communication and compute in each iteration are now independent. However this comes at the cost of doubling the pipeline bubble,
# and you must set the number of microbatches to at least 2 * num_stages (the minimum 2 * num_stages is set by default with this delay).

pipeline_fsdp_ag_once: False # If set to true then all gather all of the weights over FSDP before the first pipeline iteration.
# This is a memory/time tradeoff - we now have to store the FSDP gathered weights and gradients (typically in bf16), as opposed
# to only one stage's worth, however we only execute one all-gather and reduce across per repeat, as opposed
# to every microbatch. This is similar to zero-1 sharding, since we also don't need to all gather the FSDP weights in the backward pass.
# An alternative to setting this to true may be to replace any FSDP with DP and use optimizer offloading if necessary.
pipeline_fsdp_ag_per_repeat: False
# Pipeline weight prefetching per repeat is an advanced SPMD pipeline parallelism improvement technique
# When enabled, it prefetches necessary weight gathering ahead of microbatched computation, therefore reducing collectives

# There are two loops for PP:
#  1)  Outer loop over microbatches (pipeline iterations)
#  2)  Inner loop over layers (layers per stage)
# We have observed extra remat when a remat policy and scanning is performed on both, and recommend the default
# settings below of scanning and setting a remat policy only over the pipeline iterations.
# It may be useful to do the reverse when the layers_per_stage is very large.
# The below settings only have effect when using pipeline parallelism.
scan_pipeline_iterations: True
scan_pipeline_repeats: False
scan_layers_per_stage: False
set_remat_policy_on_pipeline_iterations: True
set_remat_policy_on_layers_per_stage: False


# Choose 'remat_policy' between 'minimal_with_context', 'minimal', 'save_dot_with_context_except_mlp', 'save_dot_except_mlpwi', 'save_dot_except_mlp',
# 'save_qkv_proj', 'qkv_proj_offloaded', 'custom', 'minimal_offloaded', 'save_out_proj' and 'full'.
# These options offer a trade-off between speed (fastest to slowest) and HBM usage (highest to lowest)
remat_policy: 'full'
# If "custom" remat_policy is chosen, you can select tensors from the following list to offload on host memory, rematerialize or save on device memory.
# Pick one of these options for following tensors: ['remat','device','offload']
decoder_layer_input: 'device' # this tensor cannot be rematerialized - it serves as periodic checkpoints that act as the remat start points
context: 'remat' # From https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/jax/attention.py#L581-L583
mlpwi: 'remat'
mlpwi_0: 'remat'
mlpwi_1: 'remat'
mlpwo: 'remat'
moe_mlpwi_0: 'remat'
moe_mlpwi_1: 'remat'
moe_mlpwo: 'remat'
query_proj: 'remat'
key_proj: 'remat'
value_proj: 'remat'
qkv_proj: 'remat'
out_proj: 'remat'
query_wa_proj: 'remat'
kv_wa_proj: 'remat'
mla_q: 'remat'
mla_kv: 'remat'
attention_out: 'remat'
engram: 'remat'

optimizer_memory_host_offload: False
parameter_memory_host_offload: False
scan_layers: True # We recommend setting this to false when using pipeline parallelism, instead scanning the PP iterations.
param_scan_axis: 1

# The attention parameter dictates the specific algorithm/methodology used to compute the attention scores
# The attention_type parameter determines the variants of attention, e.g. global or local_sliding
attention: 'autoselected' # Supported attention: autoselected, dot_product, flash, cudnn_flash_te
attention_type: 'global' # Supported attention_type: global, local_sliding, chunk, mla
share_kv_projections: False # Note: Not compatible with attention_type='mla'
attention_bias: False # If True, adds a learnable bias to the query, key, and value projections
attention_sink: False
sliding_window_size: 0
chunk_attn_window_size: 0
attn_logits_soft_cap: 0.0
final_logits_soft_cap: 0.0
z_loss_multiplier: 0.0
use_post_attn_norm: False
use_post_ffw_norm: False
v_norm_with_scale: True
qk_norm_with_scale: True
mla_naive_kvcache: True


# Adding Mixture of Block Attention Support (MoBA): https://github.com/MoonshotAI/MoBA/blob/master/MoBA_Tech_Report.pdf
moba: False
moba_chunk_size: 1024
moba_topk: 8

# DeepSeek Sparse Attention (DSA)
# deepseek3.2 introduces indexer in MLA
use_indexer: False
indexer_head_dim: 128
indexer_n_heads: 64
indexer_topk: 2048
# Determines the training strategy for the indexer:
# - False (Dense Warm-up): Computes indexer loss over all tokens. Used with `trainable_parameters_mask` to freeze other model parameters.
# - True (Sparse Training): Computes indexer loss over top-k tokens only and detaches the indexer input for independent optimization.
# Note: This is only active when `indexer_loss_scaling_factor` > 0.
indexer_sparse_training: False
# Multiplier for the indexer KL divergence loss
indexer_loss_scaling_factor: 0.0

# MLA parameters
q_lora_rank: 0
kv_lora_rank: 512
qk_nope_head_dim: 128
qk_rope_head_dim: 64
v_head_dim: 128

# QK-Clip (Muon Clip) Configuration
use_qk_clip: False  # Enable QK-Clip (supported in MLA with DotProduct or Tokamax Splash)
qk_clip_threshold: 100.0  # Threshold for clipping (tau in the paper)

# Combine matmuls for QKV and MLP
fused_qkv: False
fused_mlp: False

record_internal_nn_metrics: 0

# Output directory
# Create a GCS bucket, e.g. my-maxtext-outputs and set this to "gs://my-maxtext-outputs/"
base_output_directory: ""

# Multi-tier checkpointing is an experimental Orbax feature that: periodically saves to persistent storage(GCS bucket) dictated by `multi_tier_checkpointing_backup_interval_minutes` and,
# saves to a local directory for smaller checkpoint intervals(local_checkpoint_period).
# The local checkpoint directory must be specified when enabling multi-tier checkpointing.
# During restore, if a local copy is available in any slice, it will be broadcast to other slices without having to fetch from persistent storage.
# See more details on https://github.com/google/orbax/tree/main/checkpoint/orbax/checkpoint/experimental/emergency/multi_tier_checkpointing.
# Example for enabling multi-tier checkpointing
# enable_multi_tier_checkpointing=True local_checkpoint_directory="/local" local_checkpoint_period=20 multi_tier_checkpointing_backup_interval_minutes=20
enable_multi_tier_checkpointing: False

# The interval to backup local checkpoints to the persistent storage(GCS bucket) in minutes.
# It should be a positive number when enabling multi-tier checkpointing.
multi_tier_checkpointing_backup_interval_minutes: 0

# Number of identical pipelines in job, should be equal to ICI data parallelism * DCN data parallelism.
# It should be a positive number when enabling multi-tier checkpointing. If set to 0, it will be set to num of slices.
mtc_data_parallelism: 0


# Whether to enable emergency checkpoint. If True, `local_checkpoint_directory` and a non-zero `local_checkpoint_period` must also be specified.
# Emergency checkpoint is an experimental Orbax feature that: periodically saves to persistent storage and, with a larger invertal, saves to a local directory.
# During restore, if a local copy is available in any slice, it will be broadcast to other slices without having to fetch from persistent storage.
# See more details on https://github.com/google/orbax/tree/main/checkpoint/orbax/checkpoint/experimental/emergency.
enable_emergency_checkpoint: False

# It should be specified when and only when `enable_emergency_checkpoint` is True. Or when `enable_multi_tier_checkpointing` is True.
local_checkpoint_directory: ""

# It should be a positive number when and only when `enable_emergency_checkpoint` or `enable_multi_tier_checkpointing` is True.
local_checkpoint_period: 0

# Jax cache directory
jax_cache_dir: "~/jax_cache"

# Hardware
hardware: 'tpu' # Supported hardware types are 'tpu', 'gpu', 'gpu_multiprocess' and 'cpu'

# internal_compile allows bypassing open-source topology name mappings when using internal topologies directly via get_topology_desc.
internal_compile: False
internal_compile_num_devices: -1 # You must specify the number of devices when using internal_compile.
compile_xla_flags: "" # Compiler options e.g. compile_xla_flags="--xla_tpu_num_sparse_cores_for_gather_offloading=1 --xla_tpu_scoped_vmem_limit_kib=65536"

# Parallelism
shard_mode: "auto" # can be either auto or explicit
custom_mesh_and_rule: "" # replace default mesh and logical rule by specifying yml name under config/mesh_and_rule/.
mesh_axes: ['diloco', 'data', 'stage', 'fsdp', 'fsdp_transpose', 'context', 'context_autoregressive', 'tensor', 'tensor_transpose', 'tensor_sequence', 'expert', 'autoregressive']
logical_axis_rules: [
                      # ==========================================
                      # Vocabulary Embedding
                      # ==========================================
                      # Vocab Activations
                      ['activation_embed_and_logits_batch', ['data', 'stage', 'fsdp', 'fsdp_transpose', 'expert']],
                      ['activation_embed_and_logits_batch_sequence', ['data', 'stage', 'fsdp', 'fsdp_transpose', 'context', 'expert']],
                      ['activation_vocab', ['tensor', 'tensor_transpose', 'tensor_sequence']],
                      ['activation_vocab', ['tensor', 'tensor_transpose']],
                      ['activation_vocab', 'tensor_sequence'],
                      # Vocab Weights
                      ['vocab', ['tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive']],
                      ['embed_vocab', ['fsdp', 'fsdp_transpose', 'context', 'expert']],
                      # ==========================================
                      # Attention
                      # ==========================================
                      # Attention Activations
                      ['activation_batch_attn', ['data', 'fsdp', 'fsdp_transpose', 'expert']],
                      ['activation_heads', ['tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive']],
                      ['activation_kv_heads', ['tensor', 'tensor_transpose', 'tensor_sequence']],
                      ['activation_length_attn', ['context']],
                      ['activation_q_length', ['context']],
                      ['activation_kv_length', []],
                      ['activation_embed_attn', ['tensor', 'tensor_transpose']],
                      ['activation_kv', ['tensor', 'tensor_transpose', 'tensor_sequence']],
                      ['activation_kv_batch', ['data', 'fsdp', 'fsdp_transpose', 'expert']],
                      ['activation_kv_head_dim', ['tensor', 'tensor_transpose', 'tensor_sequence']],
                      # Attention Weights
                      ['heads', ['tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive']],
                      ['q_heads', ['tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive']],
                      ['kv_heads', ['tensor', 'tensor_transpose', 'tensor_sequence', 'autoregressive']],
                      ['qkv', []],
                      ['kv', []],
                      ['kv_head_dim', []],
                      ['q_lora', ['fsdp', 'fsdp_transpose', 'context', 'tensor_transpose', 'expert']],
                      ['q_lora', ['fsdp', 'context', 'tensor_transpose', 'expert']],
                      ['q_lora', ['fsdp', 'fsdp_transpose', 'context', 'expert']],
                      ['q_lora', ['fsdp', 'context', 'expert']],
                      ["q_lora_up_proj", []],
                      ['kv_lora', ['fsdp', 'fsdp_transpose', 'context', 'tensor_transpose', 'expert']],
                      ['kv_lora', ['fsdp', 'context', 'tensor_transpose', 'expert']],
                      ['kv_lora', ['fsdp', 'fsdp_transpose', 'context', 'expert']],
                      ['kv_lora', ['fsdp', 'context', 'expert']],
                      ["kv_lora_up_proj", []],
                      # ==========================================
                      # Mixture of Experts (MoE)
                      # ==========================================
                      # MoE Activations
                      ['activation_batch_moe', ['data', 'fsdp', 'fsdp_transpose']],
                      ['activation_length_moe', ['context']],
                      ['activation_norm_length_moe', ['tensor_sequence', 'context']],
                      ['activation_embed_moe', ['tensor', 'tensor_transpose']],
                      ['activation_mlp_moe', ['tensor', 'tensor_transpose', 'tensor_sequence']],
                      ['activation_exp', ['expert']],
                      # MoE Weights
                      ['exp', 'expert'],
                      ['mlp_moe', ['fsdp_transpose', 'tensor', 'tensor_sequence', 'autoregressive']],
                      ['embed_moe', ['fsdp', 'fsdp_transpose', 'tensor_transpose', 'context']],
                      ['embed_moe', ['fsdp', 'tensor_transpose', 'context']],
                      ['embed_moe', ['fsdp', 'fsdp_transpose', 'context']],
                      ['embed_moe', ['fsdp', 'context']],
                      # ==========================================
                      # Standard MLP / Dense Layers / Model Structure
                      # ==========================================
                      # Dense Activations
                      ['activation_mlp', ['tensor', 'tensor_transpose', 'tensor_sequence']],
                      # Note activation batch and length also get used in vocab
                      ['activation_batch', ['data', 'fsdp', 'fsdp_transpose', 'expert']],
                      ['activation_length', ['context']],
                      ['activation_norm_length', ['tensor_sequence', 'context']],
                      ['activation_embed', ['tensor', 'tensor_transpose']],
                      ['activation_stage', 'stage'],
                      # General Weights
                      ['mlp', ['fsdp_transpose', 'tensor', 'tensor_sequence', 'autoregressive']],
                      ['embed', ['fsdp', 'fsdp_transpose', 'tensor_transpose', 'context', 'expert']],
                      ['embed', ['fsdp', 'tensor_transpose', 'context', 'expert']],
                      ['embed', ['fsdp', 'fsdp_transpose', 'context', 'expert']],
                      ['embed', ['fsdp', 'context', 'expert']],
                      ['norm', ['tensor', 'tensor_transpose']],
                      ['layers', 'stage'],
                      ['diloco', 'diloco'],
                      ['engram_dim', ['tensor']],
                      ['dense_layers', []],
                      ['moe_layers', []],
                      ['mhc', []],
                      # ==========================================
                      # Inference(Prefill, Decode, Cache)
                      # ==========================================
                      ['prefill_activation_length', ['context']],
                      ['prefill_activation_norm_length', ['tensor_sequence', 'context']],
                      ['activation_prefill_kv_batch', ['data', 'fsdp', 'fsdp_transpose', 'expert']],
                      ['decode_batch', ['data', 'fsdp', 'fsdp_transpose', 'expert']],
                      ['decode_length', []],
                      ['cache_heads', ['autoregressive', 'tensor', 'tensor_transpose', 'tensor_sequence']],
                      ['cache_heads', ['autoregressive', 'tensor', 'tensor_sequence']],
                      ['paged_kv_heads', ['tensor']],
                      ['cache_batch_prefill', []],
                      ['cache_batch', []],
                      ['cache_heads_none', []],
                      ['cache_kv', []],
                      ['cache_sequence', []],
                      ['num_pages', []],
                      ['tokens_per_page', []],
                      ['paged_kv_head_dim_size', []],
                      # ==========================================
                      # Deprecated / Scheduled for Removal
                      # ==========================================
                      ['mlp_no_fsdp', ['tensor', 'tensor_sequence', 'autoregressive']],
                      ['embed_tensor_transpose', ['tensor_transpose']],
                      ['exp_with_fsdp', 'fsdp'],
                  ]
# Axes used for DCN must be earlier in this list than ICI, see (b/339009148) for details
data_sharding: [['data', 'stage', 'fsdp', 'fsdp_transpose', 'context', 'context_autoregressive', 'tensor', 'tensor_transpose', 'tensor_sequence', 'expert', 'autoregressive']]
input_data_sharding_logical_axes: ['activation_embed_and_logits_batch', 'activation_norm_length']
# Determines which physical axis plays the role of context parallelism for input data processing and load balancing
# only supports "context" or "expert" (when custom_mesh_and_rule=ep-as-cp)
context_sharding: "context"

# sharding tolerance: float between 0.0 and 1.0 representing the allowed percentage of non-sharded parameters.
sharding_tolerance: 0.02

# One axis for each parallelism type may hold a placeholder (-1)
# value to auto-shard based on available slices and devices.
# By default, product of the DCN axes should equal number of slices
# and product of the ICI axes should equal number of devices per slice.
dcn_diloco_parallelism: 1
dcn_data_parallelism: -1  # recommended DCN axis to be auto-sharded
dcn_fsdp_parallelism: 1
dcn_fsdp_transpose_parallelism: 1
dcn_sequence_parallelism: 1  # never recommended
dcn_context_parallelism: 1
dcn_context_autoregressive_parallelism: 1
dcn_tensor_parallelism: 1 # never recommended
dcn_tensor_transpose_parallelism: 1
dcn_tensor_sequence_parallelism: 1 # never recommended
dcn_pipeline_parallelism: 1
dcn_expert_parallelism: 1
dcn_autoregressive_parallelism: 1 # never recommended
ici_diloco_parallelism: 1
ici_data_parallelism: 1
ici_fsdp_parallelism: -1 # recommended ICI axis to be auto-sharded
ici_fsdp_transpose_parallelism: 1
ici_sequence_parallelism: 1
ici_context_parallelism: 1
ici_context_autoregressive_parallelism: 1
ici_tensor_parallelism: 1
ici_tensor_transpose_parallelism: 1
ici_tensor_sequence_parallelism: 1
ici_autoregressive_parallelism: 1
ici_pipeline_parallelism: 1
ici_expert_parallelism: 1

# Enable ZeRO-1 optimizer sharding over data axis
shard_optimizer_over_data: False

# Unless explicitly specified, the number of TPU slices is automatically determined. It should only be set for
# disaggregated reinforcement learning workloads using multiple slices. For ahead of time compilation,
# you should set compile_toplogy_num_slices, which will in turn set this value. For non-TPU environments this is set to 1.
num_slices: -1

# Vocab Tiling Configs
# Enables a memory-saving optimization by computing the cross-entropy loss in chunks.
# The logits are tiled into `num_vocab_tiling` parts along the batch-sequence axis,
# reducing peak memory usage. This is highly recommended for models with large
# vocabularies (e.g., Gemma). Set to a value greater than 1 to enable.
num_vocab_tiling: 1

# Tokenizer
vocab_size: 32_000 # powers of 2 for sharding
tokenizer_path: ""
# grain and tfds pipeline supports tokenizer_type: sentencepiece, huggingface, tiktoken
# hf pipeline only supports huggingface type, and will ignore tokenizer_type flag
tokenizer_type: "sentencepiece" # Currently supporting: "tiktoken", "sentencepiece", "huggingface"
use_chat_template: False
chat_template_path: "" # path to chat template json file
tokenize_train_data: True  # False if the dataset is pre-tokenized
tokenize_eval_data: True  # False if the dataset is pre-tokenized
add_bos: True
add_eos: True
# If False, use chunking for long sequences instead of truncation.
# Note: use_truncation=False is only available in grain's pretrain preprocessing pipeline.
# See the TokenizeAndTrim and TokenizeAndChunk classes in
# `src/maxtext/input_pipeline/_grain_tokenizer.py` for implementation details.
use_truncation: True

# Dataset
per_device_batch_size: 12.0
# When expansion_factor_real_data is set to > 1, total_hosts//expansion_factor_real_data will load data.
# Each data-loading host will load per_device_batch_size * expansion_factor_real_data.
# When set to between 0 and 1, it's for grain pipeline to use a smaller chip count to read checkpoint from a larger chip count job.
# Details in https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/data_input_pipeline/data_input_grain.md#using-grain
expansion_factor_real_data: -1.0
eval_per_device_batch_size: 0.0
max_corpus_chars: 10_000_000
train_data_columns: ['text'] # for DPO dataset containing "chosen" and "rejected"
train_image_column: 'image'
eval_data_columns: ['text'] # for DPO dataset containing "chosen" and "rejected"
eval_image_column: 'image'
packing: True
num_epoch: 1
generate_padding_batch_train: False
generate_padding_batch_eval: False
# Maximum number of segments that can be packed into a single sequence
# This needs to be passed to TransformerEngine's DotProductAttention layer for packing
# This also affects packing for grain, since TransformerEngine may crash or cause
# data corruption if there are more segments packed than specified
# Set this to something like 32 for GPUs when using TransformerEngine
max_segments_per_seq: -1
# Rampup batch size, similar to Megatron-LM, see
# https://github.com/NVIDIA/Megatron-LM/blob/2a01637aa54ccdaf7ea9afc1f1b80f58c53d7f3c/megatron/core/num_microbatches_calculator.py#L233-L237
# The ramp-up proceeds in stages from `per_device_batch_size_start` up to
# the final `per_device_batch_size`. For a clean ramp-up, the total range
# (`per_device_batch_size` - `per_device_batch_size_start`)
# should be evenly divisible by batch size increment.
enable_rampup_batch_size: False
per_device_batch_size_start: 4.0
per_device_batch_size_increment: 2.0
# The target number of training samples to process during the ramp-up phase.
# There is no strict rule for this value, it only needs to be positive.
global_rampup_samples: 500

# direct preference optimization (DPO)
use_dpo: False
dpo_label_smoothing: 0.0
dpo_beta: 0.1

# Supervised Fine-Tuning (SFT)
use_sft: False
# sft_train_on_completion_only=False trains on both prompt and completion tokens; trains only on completion tokens otherwise
sft_train_on_completion_only: False

# dataset_type must be synthetic, hf, grain, tfds
# details in: https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/data_input_pipeline.md
dataset_type: tfds
# for TFDS input pipeline (dataset_type=tfds)
dataset_path: "" # your path given as argument in download_dataset.sh, e.g. "gs://my-maxtext-dataset/"
dataset_name: 'c4/en:3.0.1'
eval_dataset_name: 'c4/en:3.0.1'
train_split: 'train'
eval_split: 'validation'
# for HuggingFace input pipeline (dataset_type=hf)
# Check definition at https://github.com/huggingface/datasets/blob/0feb65dd8733191dd2d1e74215b422fc5939a56a/src/datasets/load.py#L1338-L1408
hf_path: ''
hf_name: ''
hf_data_dir: ''
hf_train_files: ''
hf_eval_split: ''
hf_eval_files: ''
hf_access_token: ''
# for Grain input pipeline (dataset_type=grain)
# Path to grain data files. Can be a single pattern or multiple patterns with weights.
# For multiple patterns, use semicolon (;) to separate and comma (,) to specify weights.
# Example: "path/to/data1.array_record*,0.3;path/to/data2.array_record*,0.7"
# Note: When using multiple files (separated by ';'), only ArrayRecord format is supported.
# For more details, see https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/data_input_pipeline/data_input_grain.md
grain_train_files: ''
grain_eval_files: ''
grain_train_mixture_config_path: '' # Path to a JSON file specifying the mixture weights for Grain training data.
grain_file_type: 'arrayrecord' # arrayrecord or parquet
grain_packing_type: 'first_fit' # 'first_fit', 'best_fit' or 'concat_then_split'. See details of the corresponding module in https://google-grain.readthedocs.io/en/latest/grain.experimental.html
grain_worker_count: 1 # Set to -1 to enable auto-tuning: automatically determines optimal worker count. See https://google-grain.readthedocs.io/en/latest/_autosummary/grain.experimental.pick_performance_config.html
grain_per_worker_buffer_size: 1
# num_threads and prefetch_buffer_size are per-worker per-dataset.
# When using array_records, they are used in ReadOptions (https://google-grain.readthedocs.io/en/latest/tutorials/data_loader_tutorial.html#per-worker-readoptions)
# The default value matches that in the Grain package. If mixing multiple data sources, consider lowering these values to reduce memory usage.
# When using parquet, grain_num_threads is the number of files to read and interleave in parallel
grain_num_threads: 16
grain_prefetch_buffer_size: 500
grain_worker_count_eval: 1
grain_per_worker_buffer_size_eval: 1
grain_ram_budget_mb: 1024 # RAM budget (MB) for auto-tuning worker count. Only used when grain_worker_count is -1.
grain_num_threads_eval: 16
grain_prefetch_buffer_size_eval: 500
grain_data_source_max_workers: 16  # Max workers for ThreadPoolExecutor when mixing multiple Grain data sources.
grain_shuffle_buffer_size: 100 # shuffle buffer when using sequential access formats such as Parquet, TFRecord.
grain_use_elastic_iterator: False # For elastic training, set to this true and packing=False
# for using pathways
colocated_python_data_input: False  # experimental feature, under testing

# Training loop
steps: 150_001 # If set to -1 then will inherit value from learning_rate_schedule_steps
log_period: 100 # The frequency of Tensorboard flush, gcs metrics writing, and managed profiler metrics updating.

jax_distributed_initialization_timeout: 300 # This is the default timeout in https://github.com/jax-ml/jax/blob/main/jax/_src/distributed.py
# Note there are two separate initializations - the jax coordination service (aka jax.distributed.initialize) and the backend (e.g. PjRT), the timeout above refers
# only to the jax coordination service.
jax_debug_log_modules: "" # Set this to "jax" to enable jax verbose logging such as for the jax coordination service initialization.
skip_jax_distributed_system: False # If True we will not initialize the jax distributed system.
# Currently the jax distributed is needed on cloud TPUs for async checkpointing.
# However when run on google internal TPUs the coordination service is started automatically
# and we should set this to True so we won't try to initialize a second time manually.

# Learning rate schedule structure depends on lr_schedule_type:
#
# Cosine schedule (lr_schedule_type='cosine'):
# Inspired by Llama2's learning rate schedule, see https://arxiv.org/pdf/2307.09288.pdf section 2.2
# 1) Linear warmup from 0 to [learning_rate] over steps 0 to [learning_rate_schedule_steps * warmup_steps_fraction]
# 2) Cosine decay from [learning_rate] to [learning_rate * learning_rate_final_fraction] until learning_rate_schedule_steps
# 3) Constant learning rate of 0 from learning_rate_schedule_steps to steps (if steps > learning_rate_schedule_steps)
#
# WSD schedule (lr_schedule_type='wsd', Warmup-Stable-Decay):
# 1) Linear warmup from 0 to [learning_rate] over steps 0 to [learning_rate_schedule_steps * warmup_steps_fraction]
# 2) Stable phase at [learning_rate] for the majority of training
# 3) Decay from [learning_rate] to [learning_rate * learning_rate_final_fraction] over [learning_rate_schedule_steps * wsd_decay_steps_fraction] steps
#    The decay can be either linear or cosine based on wsd_decay_style
# 4) Constant learning rate of 0 from learning_rate_schedule_steps to steps (if steps > learning_rate_schedule_steps)
#
# The zero learning rate section can be used to more accurately measure the fully trained model's performance.
learning_rate: 3.e-5
lr_schedule_type: 'cosine'  # Options: 'cosine' or 'wsd'
learning_rate_final_fraction: 0.1  # Final LR as fraction of peak LR (applies to both cosine and WSD schedules)
wsd_decay_steps_fraction: 0.1  # Fraction of learning_rate_schedule_steps used for decay phase in WSD (e.g., 0.1 = 10%)
wsd_decay_style: 'linear'  # Decay style for WSD schedule: 'linear' or 'cosine'
warmup_steps_fraction: 0.1  # Fraction of learning_rate_schedule_steps used for warmup phase (applies to both schedules)
learning_rate_schedule_steps: -1 # By default the length of the schedule is set to the number of steps.
# However you may choose a longer schedule (learning_rate_schedule_steps > steps), in which case the training will end before
# dropping fully down. Or you may choose a shorter schedule, where the unspecified steps will have a learning rate of 0.

max_target_length: 2048 # Maximum sequence length
max_prefill_predict_length: 64 # Maximum length for the prefill when doing autoregression
prompt: "I love to" # Prompt for language model sampling.
load_from_prefill_dir: False # If true, decode.py doesn't "prefill" but just reads from directory
prefill_cache_dir: "" # If set and load_from_prefill_dir, decode.py reads from directory. If set, decode.py writes to directory
autoregressive_decode_assert: ""

# For nsys profiler, pass the training command to nsys command
# e.g. nsys profile -s none --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop {training command}
profiler: "" # Supported profiler: '', xplane, nsys
# If set to true, upload all profiler results from all hosts. Otherwise, only upload the profiler result from the first host.
upload_all_profiler_results: False
# Skip first n steps for profiling, to omit things like compilation and to give
# the iteration time a chance to stabilize.
skip_first_n_steps_for_profiler: 1
# Profile for a small number of steps to avoid a large profile file size.
profiler_steps: 5
hide_profiler_step_metric: False
profile_cleanly: True # If set to true, adds a block_until_ready on train state which aligns the profile for each step.
profile_periodically_period: -1 # If set to a positive integer, profile every profile_periodically_period steps.
# This is useful to debug scenarios where performance is changing.

# Managed ML diagnostics settings. If the feature is enabled, it will
# - create a managed ML diagnostics run with all the MaxText configs
# - upload xplane profiling, if it is enabled.
# - upload training metrics, at the defined log_period interval.
managed_mldiagnostics: False # Whether to enable the managed diagnostics
managed_mldiagnostics_run_group: ""  # Optional. Used to group multiple runs.

# Dump HLO and jaxpr options
dump_hlo: False
dump_step: -1 # Dump modules at the given step if set to a positive integer.
dump_hlo_local_dir: "/tmp/xla_dump/"
dump_hlo_delete_local_after: True # Cleans local directory after its uploaded
dump_hlo_gcs_dir: "" # Defaults to {base_output_directory}/{run_name}/xla_dump
dump_hlo_local_module_name: "jit_train_step" # Filter saving modules locally by this string. Set to empty string to remove any filter.
dump_hlo_module_name: "jit_train_step" # Filter uploading modules by this string. Set to empty string to remove any filter.
dump_hlo_xla_flags: "" # Defaults to "--xla_dump_to={dump_hlo_local_dir} --xla_dump_hlo_module_re={dump_hlo_local_module_name} --xla_dump_large_constants"
dump_hlo_upload_all: False # If true all hosts dump HLO, false only jax.process_index()==0
# All hosts should have identical HLO for SPMD programs, however we have encountered some bugs
# where this is not the case and it is helpful to compare HLO across hosts.
dump_jaxpr: False
dump_jaxpr_local_dir: "/tmp/jaxpr_dump/"
dump_jaxpr_delete_local_after: True
dump_jaxpr_gcs_dir: "" # Defaults to {base_output_directory}/{run_name}/jaxpr_dump

# When dropout is false the model is a deterministic function of the
# data_shuffle_seed and init_weights_seed (i.e. reproducible losses)
enable_dropout: True
enable_data_shuffling: True
data_shuffle_seed: 0
init_weights_seed: 0

# DiLoCo params.
enable_diloco: False
diloco_sync_period: 36
diloco_outer_lr: 0.3
diloco_outer_momentum: 0.9

# You may disable clipping by setting gradient_clipping_threshold to zero.
gradient_clipping_threshold: 1.0

# Instead of updating the weights every step, you may effectively use a larger
# batch by accumulating the gradient over a set of steps.
gradient_accumulation_steps: 1

opt_type: "adamw"  # one of "adamw", "adam_pax", "sgd", or "muon"

# If True, skip the training step when loss or gradient spike is detected
# No updates for both weights and momentums (if applies)
skip_step_on_spikes: False
# The rolling interval to calculate the mean and standard deviation
skip_step_interval: 128
# The scaling factor to determine if a spike occurred
skip_step_scaling_factor: 6.0

# List of parameter names/patterns to train.
# If non-empty, all other parameters will be frozen. Example: ['.*indexer.*'].
# If empty (default), all parameters are trained.
trainable_parameters_mask: []

# AdamW optimizer parameters
# We use AdamW following Llama2's training details, see https://arxiv.org/pdf/2307.09288.pdf section 2.2
adam_b1: 0.9 # Exponential decay rate to track the first moment of past gradients.
adam_b2: 0.95 # Exponential decay rate to track the second moment of past gradients.
adam_eps: 1.e-8 # A small constant applied to denominator outside of the square root.
adam_eps_root: 0. # A small constant applied to denominator inside the square root.
adam_weight_decay: 0.1 # AdamW Weight decay
adamw_mask: [] # List of parameter names/patterns to exclude from weight decay in AdamW, like ['bias', '.*norm', '.*ln.*'].
mu_dtype: "" # data type to store "mu" of AdamW tracking the first moment. Inherits from  weight_dtype if unset.
# Setting nu_dtype is not yet supported by optax, instead nu_dtype is always inherited from weights.
# See b/399961932 for more.

# Muon optimizer parameters
# https://github.com/google-deepmind/optax/blob/main/optax/contrib/_muon.py
# "mu_dtype", "adam_eps" are shared by AdamW
# "nesterov", "ns_coeffs", "ns_steps", "weight_decay_mask", "adaptive" use default
muon_beta: 0.95 # Decay rate for the exponentially weighted average of grads.
muon_weight_decay: 0 # Strength of the weight decay regularization. This is multiplied with the learning rate.
muon_consistent_rms: None # If None, apply width scaling to updates. If float, apply consistent rms scaling (recommend 0.2).

# Stack trace parameters
collect_stack_trace: False
stack_trace_to_cloud: False  # Uploads to cloud logging if True, else to the console if False.
stack_trace_interval_seconds: 600  # Stack trace collection frequency in seconds.

# Use iota operator in Embed
use_iota_embed: False
# use positional embedding
use_untrainable_positional_embedding: False
trainable_position_size: -1  # enable gpt3 position embedding with a positive trainable_position_size
# RoPE parameters
rope_type: "default" # one of "default", "llama3.1" or "yarn"
rope_linear_scaling_factor: 1.0 # linear scaling factor for "default" RoPE (see class `RotaryEmbedding` for more)
rope_use_scale: True # apply rope scaling for llama3.1 (see class `LLaMARotaryEmbedding` for more)
rope_min_timescale: 1
rope_max_timescale: 10_000 # Timescale For global Attention
local_rope_max_timescale: -1 # If positive used for local window Attention, otherwise `rope_max_timescale` is used for both local and global
global_rope_max_timescale: -1 # Timescale For global Attention (Gemma 4 specific)
global_rope_proportion: 0.25
local_rope_proportion: 1.0

# yarn RoPE parameters
max_position_embeddings: 163840
original_max_position_embeddings: 4096
rope_factor: 40
beta_fast: 32
beta_slow: 1
mscale: 1.0
rope_interleave: True # RoPE with sin/cos interleaved vs concatenated
rope_truncate: True # Floor lower bound and ceil upper bound for correction range
rope_attention_scaling: False # Scale the rotary embedding output

# Ahead of time Compilation (aka AOT)
# Only set these arguments if you are running train_compile or loading a compiled train step.
compiled_trainstep_file: "" # Name of saved serialized compiled train_step, e.g. compiled_train_v5e-256.pickle
compile_topology: '' # Target hardware version, e.g. 'v5e-256'
compile_topology_num_slices: -1 # Number of target slices, set to a positive integer.

decode_sampling_strategy: "greedy" # decode_sampling_strategy should be one of greedy, weighted, nucleus, topk, or composite(top_k -> top_p -> weighted temperature)
decode_sampling_nucleus_p: -1 # set if you're doing nucleus / top-p
decode_sampling_top_k: 0 # set if you're doing top-k
decode_sampling_temperature: 1.

eval_interval: -1  # the specific number of train step between eval_step
eval_steps: -1  # run this number of steps for eval, recommend setting this to prevent error due to running out of evel data
target_eval_loss: 0.  # early stop once reaching target eval_loss
abort_on_nan_loss: True # Check for NaN and abort if found in training loss
abort_on_inf_loss: True # Check for Inf and abort if found in training loss

# Goodput parameters
enable_goodput_recording: False
monitor_goodput: False
goodput_upload_interval_seconds: 30
enable_pathways_goodput: False
monitor_step_time_deviation: True
step_deviation_interval_seconds: 30
enable_gcp_goodput_metrics: True
enable_gcp_step_deviation_metrics: True

# GCP workload monitoring
report_heartbeat_metric_for_gcp_monitoring: False
heartbeat_reporting_interval_in_seconds: 5
report_performance_metric_for_gcp_monitoring: False

enable_tensorboard: True

# Vertex AI Tensorboard Configurations - https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/use_vertex_ai_tensorboard.md
# Set to True for GCE, False if running via XPK
use_vertex_tensorboard: False
# Project to create Vertex AI Tensorboard in for GCE, blank if project is set using 'gcloud config set project'
# Set this to blank if running via XPK
vertex_tensorboard_project: ""
# Region to create Vertex AI Tensorboard in for GCE, blank if running via XPK
# Vertex AI supported regions: https://cloud.google.com/vertex-ai/docs/general/locations#available-regions
vertex_tensorboard_region: ""

# If set to True, MaxText will perform extra checks using jax.checkify. Note that this will effect performance.
max_checkify: False

# Inference
inference_microbenchmark_prefill_lengths: "64,128,256,512,1024"
inference_microbenchmark_stages: "prefill,generate"
inference_microbenchmark_loop_iters: 10
inference_microbenchmark_log_file_path: ""
inference_microbenchmark_num_samples: [1, 2, 3, 4, 5]
inference_metadata_file: "" # path to a json file
inference_server: "MaxtextInterleavedServer"  # inference server to start
prefill_slice: "v5e-16" # slice to use for prefill in disaggregation mode
generate_slice: "v5e-16" # slice to use for generatation in disaggregation mode
inference_benchmark_test: False
enable_model_warmup: False
enable_llm_inference_pool: False          # Bool to launch inference server for llm_inference_gateway with their specified APIs
multi_sampling: False
return_log_prob: False

# Stack prefill cache across the layer to reduce the
# Python layer latency.
stack_prefill_result_cache: False

# KV Cache layout control
# Logical layout: 0,1,2,3 ; CACHE_BATCH, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV
# Default layout: 1,2,0,3 ; CACHE_SEQUENCE, CACHE_HEADS, CACHE_BATCH, CACHE_KV
prefill_cache_axis_order: "1,2,0,3"
ar_cache_axis_order: "1,2,0,3"

# Compute layout control
# Default layout: 0,1,2,3 ; BATCH, LENGTH, HEAD, D_KV
# Currently only support compute layout: 0,1,2,3 and 0,2,1,3
compute_axis_order: "0,1,2,3"

reshape_q: False

# Maxengine Metrics
prometheus_port: 0

# Maxengine server
enable_jax_profiler: False
jax_profiler_port: 9999

# TPU power trace level for xprof. 0:POWER_TRACE_NONE, 1:POWER_TRACE_NORMAL, or 2:POWER_TRACE_SPI
xprof_tpu_power_trace_level: 0
xprof_e2e_enable_fw_throttle_event: False
xprof_e2e_enable_fw_power_level_event: False
xprof_e2e_enable_fw_thermal_event: False
profile_power_events: False # Set to True to enable TPU-specific power/thermal profiling events. Defaults to False to avoid breaking GPU xplane tracing.

log_config: True # Prints the config (after defaults have been set by pyconfig logic)
debug_sharding: False # Prints model weights sharding info

# Checkpoint Structured logging
enable_checkpoint_cloud_logger: False

# Single-controller
enable_single_controller: False

custom_mesh: "" # Available options: ['hybrid_ring_64x4', 'hybrid_ring_32x8']
# Split physical axes for https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.mesh_utils.create_device_mesh.html
allow_split_physical_axes: False
# Apply transformations to the mesh to optimize for TPU v6e
optimize_mesh_for_tpu_v6e: False

shardy: True # Whether to use shardy XLA backend (default in Jax starting 0.7.0), or GSPMD (to be fully deprecated ~2026)

remove_size_one_mesh_axis_from_type: True # Whether to remove size one mesh axis from type through jax.config.

use_ragged_attention: False
ragged_block_size: 256

### Splash attention block sizes
# These can be tuned for specific hardware generations, and can be set up to
# the model's sequence length.
sa_block_q: 512
sa_block_kv: 512
sa_block_kv_compute: 512
sa_block_q_dkv: 512
sa_block_kv_dkv: 512
sa_block_kv_dkv_compute: 512
sa_block_q_dq: 512
sa_block_kv_dq: 512
sa_use_fused_bwd_kernel: False
sa_q_layout: "HEAD_DIM_MINOR"
sa_k_layout: "HEAD_DIM_MINOR"
sa_v_layout: "HEAD_DIM_MINOR"
use_max_logit_estimate: -1 # -1 means no estimate, any > 0 value will be used as max logit estimate
cost_estimate_flops_fwd: -1 # -1 means using splash default cost estmiation, any >= 0 value will be used as cost estmiation for splash to overlap for communication (forward)
cost_estimate_flops_bwd: -1 # -1 means using splash default cost estmiation, any >= 0 value will be used as cost estmiation for splash to overlap for communication (backward)
dq_reduction_steps: 0 #the number of reduction steps. For now, only 3 or all the kv steps are supported.
use_splash_scheduler: False # to use tokamax splash attention scheduler.
### Determine if we want to use load balance for context parallelism
context_parallel_load_balance: True
context_parallel_strategy: "all_gather" # "all_gather" or "ring"
context_parallel_reorder_strategy: "auto" # "auto", "dual_chunk_swap", or "striped"

### Paged Attention ###
# These settings take effect only when `attention=paged`.
# They should be adjusted based on the available HBM and model config.
# Note: one page group corresponds to one request/slot
pagedattn_num_pages: 64  # total number of pages to allocate
pagedattn_tokens_per_page: 32  # number of tokens each page can hold
pagedattn_pages_per_compute_block: 4  # number of pages processed together in pallas kernels
pagedattn_max_pages_per_group: -1  # defaults to number of pages needed to reach max_target_length
# Alignment of head_dim to the nearest multiple of this value, set to 0 to disable alignment. On
# TPUs, the head_dim is padded to the nearest multiple of 128.
pagedattn_head_dim_alignment: 128


# Chunked Prefill Parameters
prefill_chunk_size: 256
use_chunked_prefill: False

# Prefix Caching parameters in jetstream
enable_prefix_caching: False
prefix_caching_hbm_byte: 10_000_000_000 # 10 GB
prefix_caching_dram_byte: 100_000_000_000 # 100 GB

# This is a temporary flag that will be removed soon after the fix lands in TE
enable_padding_causal_mask: True

# Llama4-specific
# Whether to apply Query/Key normalization.
# NOTE: non-Llama4 models use RMSNorm before RoPE
# whereas Llama4 models use L2Norm after RoPE
use_qk_norm: False
# Every `X` layers will NOT use RoPE
nope_layer_interval: -1
# Every `X` layers is MoE layer
interleave_moe_layer_step: 1
# dynamically scale the attention temperature for each query token based on sequence length
# Recommended for long sequences (e.g., >32k tokens) to maintain stable output results
# See (https://arxiv.org/abs/2501.19399) for more details
temperature_tuning: False

# Multimodal flags
use_multimodal: False
use_audio: False
freeze_vision_encoder_params: True
freeze_audio_encoder_params: True
dtype_mm: "float32"  # Data type for multimodal model's vision encoder
remat_policy_for_vit: "minimal"  # Remat policy for multimodal model's vision encoder. Check `remat_policy` for options.
image_size_for_vit: 896 # Default for Gemma3, and should be overwritten by model's config
image_path: "" # Local image path used for decoding, can be multiple paths separated by comma, exp "/path/image1.jpg,/path/image2.jpg"
video_path: "" # Local video path used for decoding, can be multiple paths separated by comma, exp "/path/video1.mp4,/path/video2.mp4"
audio_path: "" # Local audio path used for decoding, can be multiple paths separated by comma, exp "/path/audio1.wav,/path/audio2.wav"
image_placeholder: "<|image|>"
video_placeholder: "<|video|>"
audio_placeholder: "<|audio|>"
use_audio_in_video: False
posemb_type_for_vit: "learn"
# max_num_images_per_example only applies for training when your image column is a list of images.
# -1 means no limit, and will pad to the max possible number of images determined by sequence length.
# Set it to avoid unnecessary padding if you know the maximum number of images per example.
max_num_images_per_example: -1
vision_output_length: -1 # The output length (number of soft tokens) from vision encoder, used in Gemma4.

### llama4 multi modal configs
hidden_size_for_vit: 1408
intermediate_size_for_vit: 5632
num_attention_heads_for_vit: 16
num_channels_for_vit: 3
tile_size_for_vit: 336
patch_size_for_vit: 14
conv_stride_for_vit: 14
num_hidden_layers_for_vit: 34
projector_input_dim_for_vit: 4096
projector_output_dim_for_vit: 4096
rope_theta_for_vit: 10000
vision_output_dim_for_vit: 4096
pixel_shuffle_ratio_for_vit: 0.5
projector_dropout_for_vit: 0.0

# Qwen3-OmniMoe vision encoder
spatial_merge_size_for_vit: 2
out_hidden_size_for_vit: 512
temporal_patch_size_for_vit: 2
num_position_embeddings_for_vit: 1024
deepstack_visual_indexes_for_vit: []

### Audio encoder configs (Qwen3-OmniMoe)
d_model_for_audio: 256
encoder_attention_heads_for_audio: 4
encoder_ffn_dim_for_audio: 512
encoder_layers_for_audio: 2
attention_dropout_for_audio: 0.0
activation_dropout_for_audio: 0.0
activation_function_for_audio: "gelu"
num_mel_bins_for_audio: 128
max_source_positions_for_audio: 1500
scale_embedding_for_audio: True
n_window_for_audio: 50
n_window_infer_for_audio: 800
conv_chunksize_for_audio: 500
downsample_hidden_size_for_audio: 256
output_dim_for_audio: 512
num_conv_layers_for_audio: 3
max_timescale_for_audio: 10000.0
max_sample_len_for_audio: 10000

use_mrope: false
mrope_section: [24, 20, 20]
position_id_per_seconds: 25

# Subslice shape in the form of "x,y,z" when using pathways (single controller).
# Example: "8,8" to use a 8x8 subgrid (64 chips) of a full pod (16x16) of trillium.
subslice_shape: ""

# NNX
enable_nnx: False
pure_nnx_decoder: False
pure_nnx: False

################################## Qwen3-Next Specific Configs ##################################
# Kernel size for the 1D convolution in the Gated Delta Net
gdn_conv_kernel_dim: 4
# Head dimension for the key/query in the Gated Delta Net
gdn_key_head_dim: 128
# Head dimension for the value in the Gated Delta Net
gdn_value_head_dim: 128
# Number of key/query heads in the Gated Delta Net
gdn_num_key_heads: 16
# Number of value heads in the Gated Delta Net
gdn_num_value_heads: 32
# Chunk size for the parallel scan algorithm in the Gated Delta Net.
gdn_chunk_size: 64
# Whether to apply L2 normalization to query and key tensors inside the Gated Delta Rule kernel.
use_qk_norm_in_gdn: True
# The ratio of dimension to apply ROPE on
partial_rotary_factor: 1.0

# Use tokamax library for gmm kernel implementation
use_tokamax_gmm: false
use_tokamax_splash: false
# Setting this flag will use a non-pallas implementation.
use_jax_splash: false

# vLLM Adapter Configurations
# Path to the HuggingFace-style config directory for the adapter (e.g. src/maxtext/integration/vllm/maxtext_vllm_adapter)
vllm_hf_config_path: ""
# A JSON string of overrides to apply to the HuggingFace-style config for the vLLM adapter.
# This can be used to override specific settings without modifying the original config file.
vllm_hf_overrides: {}
# JSON string containing additional configuration for the vLLM model (e.g. '{"maxtext_config": {...}}')
vllm_additional_config: {}
# When use_jax_splash=True, force the layout of the query tensor to be [..., NUM_HEADS, HEAD_DIM, SEQ_LENGTH]
force_q_layout: false

################################## DeepSeek Manifold-Constrained Hyper Connections (mHC) ##################################
# The number of parallel streams in Hyper Connection.
mhc_expansion_rate: 1
# The number of iterations for the Sinkhorn-Knopp algorithm.
sinkhorn_iterations: 20

################################## DeepSeek Engram ##################################
# Indices of transformer layers where Engram are integrated; leave empty [] to disable.
# Example: [1, 4] attaches to the 2nd and 5th layer.
engram_layers: []
# The max 'n' in N-gram. Example: n=3 means it covers both 2-grams and 3-grams.
engram_max_ngram_size: 3
# Number of heads dedicated to the Engram.
engram_num_heads: 8
# Head dimension for heads.
engram_head_dim: 1280
# List of minimum head vocab sizes for each n-gram order.
engram_vocab_bases: []
# Temporal window size for Engram convolution.
engram_kernel_size: 4
# The seed for Engram hash mapping.
engram_seed: 0

##### Distillation parameters
distill_alpha: 0.5
distill_temperature: 1.0
# distill_beta is used for cosine similarity loss between intermediate activataitions of out_proj in teacher/student models.
# 0.0 value disables this feature.
distill_beta: 0.0
# distill_feature_loss_type is the type of loss to use for feature distillation ("cosine" or "l2").
distill_feature_loss_type: "cosine"
distill_layer_indices: None
# Dynamic loss weight scheduling: set *_end to a target value and *_schedule to "linear" or "cosine".
# When *_end is None (default), the corresponding weight stays fixed throughout training.
distill_alpha_end: None
distill_alpha_schedule: "constant"
distill_temperature_end: None
distill_temperature_schedule: "constant"
distill_beta_end: None
distill_beta_schedule: "constant"

##### Elastic training parameters
# Elastic training is Pathways-specific and does not work on McJAX.
elastic_enabled: false
elastic_timeout_seconds: 300
elastic_max_retries: 10
elastic_min_slice_count: -1
\n"""


# File: src/maxtext/configs/models/deepseek3-671b.yml (commit 313890777)
DEEPSEEK_CONFIG_RAW = """\n# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# model config for DeepSeek V3 - 671B

# For DeepSeek default device-limited routing,
# please set n_routing_groups=8 and topk_routing_group=4 in your command-line arguments.

base_emb_dim: 7168
base_num_query_heads: 128
base_num_kv_heads: 128
base_mlp_dim: 18432
base_moe_mlp_dim: 2048
base_num_decoder_layers: 61
first_num_dense_layers: 3
mlp_activations: ["silu","linear"]
vocab_size: 129280
enable_dropout: False
logits_via_embedding: False
normalization_layer_epsilon: 1.0e-6
num_experts: 256
num_experts_per_tok: 8
shared_experts: 1
routed_scaling_factor: 2.5
routed_score_func: "sigmoid"
routed_bias: True
decoder_block: "deepseek"
# MLA
attention_type: "mla"
q_lora_rank: 1536
kv_lora_rank: 512
qk_nope_head_dim: 128
qk_rope_head_dim: 64
v_head_dim: 128
mscale: 1.0
# RoPE
rope_type: "yarn"
rope_max_timescale: 10_000 # DeepSeek uses  "rope_theta": 10000
max_position_embeddings: 163840
original_max_position_embeddings: 4096
rope_factor: 40
beta_fast: 32
rope_interleave: True
rope_truncate: True
rope_attention_scaling: False
\n"""


# File: src/maxtext/checkpoint_conversion/utils/param_mapping.py (commit 313890777)
PARAM_MAPPING_RAW = """\n#  Copyright 2023–2026 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

\"\"\"Parameter mappings and transformation hooks for checkpoint conversion.

This module defines the necessary components to convert model checkpoints between
MaxText and Hugging Face formats for various architectures (e.g., Gemma, Qwen).
It provides two key types of mappings for each model:

1.  **Parameter Name Mappings (`PARAM_MAPPING`)**: Dictionaries that map a MaxText
    parameter key to its corresponding Hugging Face parameter(s). These mappings are
    generated by functions like `GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING`.

    **Key: MaxText parameters, with following forms:**
    - `atomic_mt_key`: A single string representing one MaxText parameter.
    - `composite_mt_key`: A tuple of strings representing multiple MaxText parameters. (e.g., GPT-OSS)

    **Value: corresponding Hugging Face parameters, with following forms:**
    - `unscanned`: A single string.
    - `scanned`: A list of strings, to be stacked along the layer axis.
    - `unscanned with expert stacking`: A list of strings, to be stacked along the expert axis.
    - `scanned with expert stacking`: A nested list of strings, to be stacked along both layer and expert axes.
    Note: Expert stacking only applies a subset of MoE models (e.g., Qwen MoE, DeepSeek, Mixtral),
      but not others (e.g., GPT-OSS).

2.  **Hook Functions (`HOOK_FNS`)**: Dictionaries that map a MaxText parameter
    name to a specific transformation function (a "hook"). These hooks handle
    the actual value conversion, which can include operations like reshaping,
    transposing, scaling, or padding tensors to match the target format's
    requirements. These are generated by functions like
    `GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN`.

The main conversion script uses these mappings to systematically transform each
parameter from the source checkpoint and build the target checkpoint.
\"\"\"

import warnings
import numpy as np

import jax
import jax.numpy as jnp


def GEMMA3_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  \"\"\"Generates a parameter mapping from MaxText to Hugging Face for Gemma3.

  This function creates a dictionary that maps the parameter names from a
  MaxText Gemma3 checkpoint to their corresponding names in the Hugging Face
  `Gemma3ForCausalLM` format. It handles both the text and vision components
  of the model.

  Args:
    config (dict): The Hugging Face model configuration dictionary, which must
      contain 'text_config' and 'vision_config' sub-dictionaries.
    scan_layers (bool, optional): If True, generates mappings for scanned
      layers, where multiple layers are stacked into a single tensor. If False,
      generates mappings for individual, unscanned layers. Defaults to False.

  Returns:
    dict: A mapping where keys are `atomic_mt_key` (single MaxText parameter names). Values
      are either a single Hugging Face parameter name (unscanned form) or a list of
      Hugging Face parameter names (scanned form) for stacked text layers.
  \"\"\"
  tcfg = config["text_config"]
  vcfg = config["vision_config"]
  Ndec = tcfg["num_hidden_layers"]
  Nvis = vcfg["num_hidden_layers"]

  # pylint: disable=line-too-long
  mapping = {
      # Embedding & final norm
      "params-token_embedder-embedding": "model.language_model.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": "model.language_model.norm.weight",
      # Vision embed & pos
      "params-vision_encoder-Gemma3VisionEncoderLayer_0-embedding-kernel": "model.vision_tower.vision_model.embeddings.patch_embedding.weight",
      "params-vision_encoder-Gemma3VisionEncoderLayer_0-embedding-bias": "model.vision_tower.vision_model.embeddings.patch_embedding.bias",
      "params-vision_encoder-Gemma3VisionEncoderLayer_0-pos_embedding": "model.vision_tower.vision_model.embeddings.position_embedding.weight",
      "params-vision_encoder-Gemma3VisionEncoderLayer_0-Transformer-encoder_norm-scale": "model.vision_tower.vision_model.post_layernorm.weight",
      "params-vision_encoder-Gemma3VisionEncoderLayer_0-Transformer-encoder_norm-bias": "model.vision_tower.vision_model.post_layernorm.bias",
      # Multi-modal projector
      "params-vision_encoder-VisionEmbedder_0-mm_input_projection-w": "model.multi_modal_projector.mm_input_projection_weight",
      "params-vision_encoder-VisionEmbedder_0-mm_soft_embedding_norm-scale": "model.multi_modal_projector.mm_soft_emb_norm.weight",
  }

  vision_params = [
      ("LayerNorm_0-scale", "layer_norm1.weight"),
      ("LayerNorm_0-bias", "layer_norm1.bias"),
      ("LayerNorm_1-scale", "layer_norm2.weight"),
      ("LayerNorm_1-bias", "layer_norm2.bias"),
      ("MultiHeadDotProductAttention_0-query-kernel", "self_attn.q_proj.weight"),
      ("MultiHeadDotProductAttention_0-query-bias", "self_attn.q_proj.bias"),
      ("MultiHeadDotProductAttention_0-key-kernel", "self_attn.k_proj.weight"),
      ("MultiHeadDotProductAttention_0-key-bias", "self_attn.k_proj.bias"),
      ("MultiHeadDotProductAttention_0-value-kernel", "self_attn.v_proj.weight"),
      ("MultiHeadDotProductAttention_0-value-bias", "self_attn.v_proj.bias"),
      ("MultiHeadDotProductAttention_0-out-kernel", "self_attn.out_proj.weight"),
      ("MultiHeadDotProductAttention_0-out-bias", "self_attn.out_proj.bias"),
      ("MlpBlockViT_0-Dense_0-kernel", "mlp.fc1.weight"),
      ("MlpBlockViT_0-Dense_0-bias", "mlp.fc1.bias"),
      ("MlpBlockViT_0-Dense_1-kernel", "mlp.fc2.weight"),
      ("MlpBlockViT_0-Dense_1-bias", "mlp.fc2.bias"),
  ]

  # Vision layers mapping
  for i in range(Nvis):
    for mx, hf in vision_params:
      key = f"params-vision_encoder-Gemma3VisionEncoderLayer_0-Transformer-encoderblock_{i}-{mx}"
      mapping[key] = f"model.vision_tower.vision_model.encoder.layers.{i}.{hf}"

  # Text decoder mapping
  text_params = [
      ("pre_self_attention_norm-scale", "input_layernorm.weight"),
      ("post_self_attention_norm-scale", "post_attention_layernorm.weight"),
      ("self_attention-query_norm-scale", "self_attn.q_norm.weight"),
      ("self_attention-key_norm-scale", "self_attn.k_norm.weight"),
      ("pre_ffw_norm-scale", "pre_feedforward_layernorm.weight"),
      ("post_ffw_norm-scale", "post_feedforward_layernorm.weight"),
      ("self_attention-query-kernel", "self_attn.q_proj.weight"),
      ("self_attention-key-kernel", "self_attn.k_proj.weight"),
      ("self_attention-value-kernel", "self_attn.v_proj.weight"),
      ("self_attention-out-kernel", "self_attn.o_proj.weight"),
      ("mlp-wi_0-kernel", "mlp.gate_proj.weight"),
      ("mlp-wi_1-kernel", "mlp.up_proj.weight"),
      ("mlp-wo-kernel", "mlp.down_proj.weight"),
  ]

  if scan_layers:
    # Gemma3 repeats a 6-layer attention pattern (5 local + 1 global),
    # scanned as layers_0..layers_5 with leftovers in layers_remainder.
    attention_pattern_length = 6
    num_remaining = Ndec % attention_pattern_length
    num_scanned = Ndec - num_remaining

    # Main scanned blocks: params-decoder-layers-layers_{block_idx}-{param}
    for block_idx in range(attention_pattern_length):
      hf_indices = list(range(block_idx, num_scanned, attention_pattern_length))
      for mx, hf in text_params:
        key = f"params-decoder-layers-layers_{block_idx}-{mx}"
        mapping[key] = [f"model.language_model.layers.{i}.{hf}" for i in hf_indices]

    # Remainder layers (unscanned): params-decoder-layers_remainder-layers_{rem_idx}-{param}
    if num_remaining > 0:
      for rem_idx in range(num_remaining):
        hf_layer_idx = num_scanned + rem_idx
        for mx, hf in text_params:
          key = f"params-decoder-layers_remainder-layers_{rem_idx}-{mx}"
          mapping[key] = f"model.language_model.layers.{hf_layer_idx}.{hf}"
  else:
    for i in range(Ndec):
      for mx, hf in text_params:
        key = f"params-decoder-layers_{i}-{mx}"
        mapping[key] = f"model.language_model.layers.{i}.{hf}"

  return mapping


def GEMMA3_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  \"\"\"Hook functions for Gemma3 parameter conversion.

  This function provides a dictionary of transformation functions (hooks) for
  converting Gemma3 model parameters between MaxText and Hugging Face formats.
  It handles embedding padding/scaling, RMSNorm scaling, kernel reshaping, and
  vision-specific tensor manipulations.

  Args:
    config (dict): The Hugging Face model configuration dictionary.
    scan_layers (bool, optional): Whether the model uses scanned layers.
      Defaults to False.
    saving_to_hf (bool, optional): The direction of conversion. True for
      MaxText to Hugging Face, False for the reverse. Defaults to False.

  Returns:
    dict: A dictionary mapping MaxText parameter names to their corresponding
      transformation functions.
  \"\"\"
  hooks = {}

  # ---- Embedding pad & scale ----
  def pad_and_scale_embedding(input_tensor, target_shape):
    source_vocab_size, _ = input_tensor.shape
    target_vocab_size, target_hidden_size = target_shape

    # MaxText embedding = original_embedding * sqrt(hidden_size)
    # HF embedding = original_embedding (HF model forward pass applies scaling)
    # Note: config["hidden_size"] is the HF hidden size from the HF config object
    normalizer = np.dtype("bfloat16").type(config["text_config"]["hidden_size"] ** 0.5)

    # Apply scaling first
    if saving_to_hf:  # MaxText to HF
      scaled_tensor = (input_tensor / normalizer).astype(input_tensor.dtype)
    else:  # HF to MaxText
      scaled_tensor = (input_tensor * normalizer).astype(input_tensor.dtype)

    # Handle padding/truncation
    if source_vocab_size > target_vocab_size:
      warnings.warn(
          f"source vocab={source_vocab_size} > target vocab={target_vocab_size}, truncate output layer for MaxText."
      )
      output_tensor = scaled_tensor[:target_vocab_size, :]
    elif source_vocab_size < target_vocab_size:
      warnings.warn(f"source vocab={source_vocab_size} < target vocab={target_vocab_size}, pad output layer for MaxText.")
      padding_shape = (target_vocab_size - source_vocab_size, target_hidden_size)
      # Use jnp.zeros for JAX arrays, np.zeros for numpy arrays
      padding = (
          jnp.zeros(padding_shape, dtype=scaled_tensor.dtype)
          if isinstance(scaled_tensor, jax.Array)
          else np.zeros(padding_shape, dtype=scaled_tensor.dtype)
      )
      output_tensor = (
          jnp.concatenate([scaled_tensor, padding], axis=0)
          if isinstance(scaled_tensor, jax.Array)
          else np.concatenate([scaled_tensor, padding], axis=0)
      )
    else:  # Vocab sizes match
      output_tensor = scaled_tensor

    return output_tensor

  # ---- RMSNorm scale ----
  def scale_rmsnorm(x, target_shape):
    # MaxText norm = HF norm +1; HF norm = MaxText norm -1
    if saving_to_hf:
      return (x - 1.0).reshape(target_shape)
    return (x + 1.0).reshape(target_shape)

  # ---- Generic reshape ----
  def reshape_kernel(x, target_shape):
    if saving_to_hf:
      flipped = np.flip(np.array(target_shape))
      return x.reshape(flipped).T
    else:
      return x.T.reshape(target_shape)

  # ---- Vision reshape ----
  def vis_bias(x, target_shape):
    if saving_to_hf:
      return x.flatten()
    else:
      return x.reshape(target_shape)

  def vision_patch(x, target_shape):
    if saving_to_hf:
      return x.transpose(3, 2, 0, 1)
    else:
      return x.transpose(2, 3, 1, 0)

  def pos_embed(x, target_shape):
    if saving_to_hf:
      return x.squeeze(0)
    return x[None, :, :]

  # ---Embedding & final norm---
  hooks["params-token_embedder-embedding"] = pad_and_scale_embedding
  hooks["params-decoder-decoder_norm-scale"] = scale_rmsnorm
  # [1, 4096, 1152]
  hooks["params-vision_encoder-Gemma3VisionEncoderLayer_0-embedding-kernel"] = vision_patch
  hooks["params-vision_encoder-Gemma3VisionEncoderLayer_0-pos_embedding"] = pos_embed

  hooks["params-vision_encoder-VisionEmbedder_0-mm_input_projection-w"] = lambda x, _: x
  hooks["params-vision_encoder-VisionEmbedder_0-mm_soft_embedding_norm-scale"] = scale_rmsnorm

  # Text layers
  tc = config.get("text_config", {})
  nlayers = tc.get("num_hidden_layers", 0)
  if scan_layers:
    attention_pattern_length = 6
    num_remaining = nlayers % attention_pattern_length
    # Scanned sub-layer prefixes
    prefixes = [f"params-decoder-layers-layers_{block_idx}-" for block_idx in range(attention_pattern_length)]
    # Remainder sub-layer prefixes
    if num_remaining > 0:
      prefixes += [f"params-decoder-layers_remainder-layers_{rem_idx}-" for rem_idx in range(num_remaining)]
  else:
    prefixes = [f"params-decoder-layers_{i}-" for i in range(nlayers)]
  for pref in prefixes:
    # Attention Q/K/V/O
    hooks[pref + "self_attention-query-kernel"] = reshape_kernel
    hooks[pref + "self_attention-key-kernel"] = reshape_kernel
    hooks[pref + "self_attention-value-kernel"] = reshape_kernel
    hooks[pref + "self_attention-out-kernel"] = reshape_kernel
    # Norm scales
    for nm in [
        "pre_self_attention_norm-scale",
        "post_self_attention_norm-scale",
        "self_attention-query_norm-scale",
        "self_attention-key_norm-scale",
        "pre_ffw_norm-scale",
        "post_ffw_norm-scale",
    ]:
      hooks[pref + nm] = scale_rmsnorm
    # MLP
    hooks[pref + "mlp-wi_0-kernel"] = reshape_kernel
    hooks[pref + "mlp-wi_1-kernel"] = reshape_kernel
    hooks[pref + "mlp-wo-kernel"] = reshape_kernel

  # Vision layers
  vc = config.get("vision_config", {})
  nvis = vc.get("num_hidden_layers", 0)
  for i in range(nvis):
    base = f"params-vision_encoder-Gemma3VisionEncoderLayer_0-Transformer-encoderblock_{i}-"
    # Attention kernels & biases
    for qkv in ["query", "key", "value"]:
      hooks[base + f"MultiHeadDotProductAttention_0-{qkv}-kernel"] = reshape_kernel
      hooks[base + f"MultiHeadDotProductAttention_0-{qkv}-bias"] = vis_bias
    # [1152, 1152] -> [16, 72, 1152]
    hooks[base + "MultiHeadDotProductAttention_0-out-kernel"] = reshape_kernel
    hooks[base + "MultiHeadDotProductAttention_0-out-bias"] = vis_bias
    # MLP ViT kernels & biases
    for dense in ["Dense_0", "Dense_1"]:
      hooks[base + f"MlpBlockViT_0-{dense}-kernel"] = reshape_kernel

  return hooks


def GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  \"\"\"Returns mapping between MaxText and HuggingFace Gemma2 weight paths.

  Args:
      config (dict): Model configuration dictionary containing at least
        'num_hidden_layers'.
      scan_layers (bool, optional): Whether the MaxText model uses layer
        scanning optimization. When True, decoder layers are stacked into a
        single tensor. Defaults to False.

  Returns:
      dict: A mapping where keys are `atomic_mt_key` (single MaxText parameter name).
        Values are either a single string (unscanned form) or a list of strings
        (scanned form) for stacked layers when `scan_layers=True`.

  Notes:
      - MaxText uses a paired layer approach where two HF decoder layers are
        treated as one MaxText decoder layer.
      - MaxText layer `i` corresponds to HF layers `2i` and `2i+1`.
      - Local components map to even-numbered HF decoder layers (0, 2, 4...).
      - Global components map to odd-numbered HF decoder layers (1, 3, 5...).
  \"\"\"

  nlayers = config["num_hidden_layers"]
  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
  }
  if scan_layers:
    mapping = {
        **mapping,
        "params-decoder-layers-pre_self_attention_norm_global-scale": [
            f"model.layers.{i}.input_layernorm.weight" for i in range(1, nlayers, 2)
        ],
        "params-decoder-layers-mlp_global-wo-kernel": [
            f"model.layers.{i}.mlp.down_proj.weight" for i in range(1, nlayers, 2)
        ],
        "params-decoder-layers-mlp_global-wi_1-kernel": [
            f"model.layers.{i}.mlp.up_proj.weight" for i in range(1, nlayers, 2)
        ],
        "params-decoder-layers-mlp_global-wi_0-kernel": [
            f"model.layers.{i}.mlp.gate_proj.weight" for i in range(1, nlayers, 2)
        ],
        "params-decoder-layers-post_self_attention_norm_global-scale": [
            f"model.layers.{i}.post_attention_layernorm.weight" for i in range(1, nlayers, 2)
        ],
        "params-decoder-layers-post_ffw_norm_global-scale": [
            f"model.layers.{i}.post_feedforward_layernorm.weight" for i in range(1, nlayers, 2)
        ],
        "params-decoder-layers-pre_ffw_norm_global-scale": [
            f"model.layers.{i}.pre_feedforward_layernorm.weight" for i in range(1, nlayers, 2)
        ],
        "params-decoder-layers-self_attention_global-key-kernel": [
            f"model.layers.{i}.self_attn.k_proj.weight" for i in range(1, nlayers, 2)
        ],
        "params-decoder-layers-self_attention_global-out-kernel": [
            f"model.layers.{i}.self_attn.o_proj.weight" for i in range(1, nlayers, 2)
        ],
        "params-decoder-layers-self_attention_global-query-kernel": [
            f"model.layers.{i}.self_attn.q_proj.weight" for i in range(1, nlayers, 2)
        ],
        "params-decoder-layers-self_attention_global-value-kernel": [
            f"model.layers.{i}.self_attn.v_proj.weight" for i in range(1, nlayers, 2)
        ],
        "params-decoder-layers-pre_self_attention_norm_local-scale": [
            f"model.layers.{i}.input_layernorm.weight" for i in range(0, nlayers, 2)
        ],
        "params-decoder-layers-mlp_local-wo-kernel": [
            f"model.layers.{i}.mlp.down_proj.weight" for i in range(0, nlayers, 2)
        ],
        "params-decoder-layers-mlp_local-wi_1-kernel": [
            f"model.layers.{i}.mlp.up_proj.weight" for i in range(0, nlayers, 2)
        ],
        "params-decoder-layers-mlp_local-wi_0-kernel": [
            f"model.layers.{i}.mlp.gate_proj.weight" for i in range(0, nlayers, 2)
        ],
        "params-decoder-layers-post_self_attention_norm_local-scale": [
            f"model.layers.{i}.post_attention_layernorm.weight" for i in range(0, nlayers, 2)
        ],
        "params-decoder-layers-post_ffw_norm_local-scale": [
            f"model.layers.{i}.post_feedforward_layernorm.weight" for i in range(0, nlayers, 2)
        ],
        "params-decoder-layers-pre_ffw_norm_local-scale": [
            f"model.layers.{i}.pre_feedforward_layernorm.weight" for i in range(0, nlayers, 2)
        ],
        "params-decoder-layers-self_attention_local-key-kernel": [
            f"model.layers.{i}.self_attn.k_proj.weight" for i in range(0, nlayers, 2)
        ],
        "params-decoder-layers-self_attention_local-out-kernel": [
            f"model.layers.{i}.self_attn.o_proj.weight" for i in range(0, nlayers, 2)
        ],
        "params-decoder-layers-self_attention_local-query-kernel": [
            f"model.layers.{i}.self_attn.q_proj.weight" for i in range(0, nlayers, 2)
        ],
        "params-decoder-layers-self_attention_local-value-kernel": [
            f"model.layers.{i}.self_attn.v_proj.weight" for i in range(0, nlayers, 2)
        ],
    }
  # Case 2: scan_layer=False
  else:
    for maxtext_layer_idx in range(0, nlayers // 2):
      local_layer_idx = maxtext_layer_idx * 2
      global_layer_idx = maxtext_layer_idx * 2 + 1
      # pylint: disable=line-too-long
      layer_mapping = {
          f"params-decoder-layers_{maxtext_layer_idx}-pre_self_attention_norm_global-scale": f"model.layers.{global_layer_idx}.input_layernorm.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-mlp_global-wo-kernel": f"model.layers.{global_layer_idx}.mlp.down_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-mlp_global-wi_1-kernel": f"model.layers.{global_layer_idx}.mlp.up_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-mlp_global-wi_0-kernel": f"model.layers.{global_layer_idx}.mlp.gate_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-post_self_attention_norm_global-scale": f"model.layers.{global_layer_idx}.post_attention_layernorm.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-post_ffw_norm_global-scale": f"model.layers.{global_layer_idx}.post_feedforward_layernorm.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-pre_ffw_norm_global-scale": f"model.layers.{global_layer_idx}.pre_feedforward_layernorm.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_global-key-kernel": f"model.layers.{global_layer_idx}.self_attn.k_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_global-out-kernel": f"model.layers.{global_layer_idx}.self_attn.o_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_global-query-kernel": f"model.layers.{global_layer_idx}.self_attn.q_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_global-value-kernel": f"model.layers.{global_layer_idx}.self_attn.v_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-pre_self_attention_norm_local-scale": f"model.layers.{local_layer_idx}.input_layernorm.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-mlp_local-wo-kernel": f"model.layers.{local_layer_idx}.mlp.down_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-mlp_local-wi_1-kernel": f"model.layers.{local_layer_idx}.mlp.up_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-mlp_local-wi_0-kernel": f"model.layers.{local_layer_idx}.mlp.gate_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-post_self_attention_norm_local-scale": f"model.layers.{local_layer_idx}.post_attention_layernorm.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-post_ffw_norm_local-scale": f"model.layers.{local_layer_idx}.post_feedforward_layernorm.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-pre_ffw_norm_local-scale": f"model.layers.{local_layer_idx}.pre_feedforward_layernorm.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_local-key-kernel": f"model.layers.{local_layer_idx}.self_attn.k_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_local-out-kernel": f"model.layers.{local_layer_idx}.self_attn.o_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_local-query-kernel": f"model.layers.{local_layer_idx}.self_attn.q_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_local-value-kernel": f"model.layers.{local_layer_idx}.self_attn.v_proj.weight",
      }
      mapping = {**mapping, **layer_mapping}
  return mapping


def GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  \"\"\"Creates parameter transformation functions for Gemma2 conversion.

  This function generates a mapping of transformation functions that handle the
  necessary conversions between MaxText and HuggingFace parameter formats for
  Gemma2, including operations like padding, reshaping, and scaling.

  Args:
      config (dict): Model configuration dictionary that must contain:
          - num_hidden_layers (int): Number of layers in the model.
          - head_dim (int): Dimension of attention heads.
          - hidden_size (int): Model's hidden dimension size.
      scan_layers (bool, optional): Controls the output format for layer
        parameters. True for batched, False for individual. Defaults to False.
      saving_to_hf (bool, optional): Determines the direction of transformation.
        True for MaxText to HuggingFace, False for the reverse. Defaults to
        False.

  Returns:
      dict: A mapping from MaxText parameter names to transformation functions.
        The value can be a single function or a list of functions to be
        applied sequentially.
  \"\"\"
  nlayers = config["num_hidden_layers"]

  def pad_hf_embedding_layer(input_tensor, target_shape):
    \"\"\"Pads/unpads and scales the embedding layer.

    Note:
        HF embedding weights shape =  [256000, d_model]
        MaxText embedding weights shape = [256128, d_model]
        MaxText pads Gemma2 embedding to 256128 for better performance.
    \"\"\"
    # TODO(wenxindongwork), Perhaps, this dtype should be the activation dtype
    normalizer = np.dtype("float32").type(config["hidden_size"] ** 0.5)

    if saving_to_hf:
      target_tensor = input_tensor[: target_shape[0], : target_shape[1]]
      target_tensor = target_tensor / normalizer
      target_tensor = target_tensor.astype(input_tensor.dtype)
      return target_tensor
    else:
      target_tensor = np.zeros(target_shape, dtype=input_tensor.dtype)
      target_tensor[: input_tensor.shape[0], : input_tensor.shape[1]] = input_tensor
      target_tensor = target_tensor * normalizer
      target_tensor = target_tensor.astype(input_tensor.dtype)
      return target_tensor

  def reshape_kernel(input_tensor, target_shape):
    if saving_to_hf:
      flipped_target_shape = np.flip(np.array(target_shape))
      return input_tensor.reshape(flipped_target_shape).T
    else:
      return input_tensor.T.reshape(target_shape)

  def scale_rmsnorm_layer(input_tensor, target_shape):
    if saving_to_hf:
      return (input_tensor - 1.0).reshape(target_shape)
    else:
      return (input_tensor + 1.0).reshape(target_shape)

  def scale_query_layer(input_tensor, target_shape):
    if saving_to_hf:
      depth_scale = np.dtype("float32").type(np.sqrt(config["head_dim"]))
      return (input_tensor * depth_scale).astype(input_tensor.dtype)
    else:
      depth_scale = np.dtype("float32").type(1 / np.sqrt(config["head_dim"]))
      return (input_tensor * depth_scale).astype(input_tensor.dtype)

  # hook order does not affect result
  query_hook_chain = [reshape_kernel, scale_query_layer]

  mapping = {
      "params-token_embedder-embedding": pad_hf_embedding_layer,
      "params-decoder-decoder_norm-scale": scale_rmsnorm_layer,
  }
  if scan_layers:
    mapping = {
        **mapping,
        "params-decoder-layers-self_attention_global-query-kernel": query_hook_chain,
        "params-decoder-layers-self_attention_local-query-kernel": query_hook_chain,
        "params-decoder-layers-self_attention_global-key-kernel": reshape_kernel,
        "params-decoder-layers-self_attention_local-key-kernel": reshape_kernel,
        "params-decoder-layers-self_attention_global-value-kernel": reshape_kernel,
        "params-decoder-layers-self_attention_local-value-kernel": reshape_kernel,
        "params-decoder-layers-mlp_global-wo-kernel": reshape_kernel,
        "params-decoder-layers-mlp_global-wi_1-kernel": reshape_kernel,
        "params-decoder-layers-mlp_global-wi_0-kernel": reshape_kernel,
        "params-decoder-layers-self_attention_global-out-kernel": reshape_kernel,
        "params-decoder-layers-mlp_local-wo-kernel": reshape_kernel,
        "params-decoder-layers-mlp_local-wi_1-kernel": reshape_kernel,
        "params-decoder-layers-mlp_local-wi_0-kernel": reshape_kernel,
        "params-decoder-layers-self_attention_local-out-kernel": reshape_kernel,
        "params-decoder-layers-pre_self_attention_norm_global-scale": scale_rmsnorm_layer,
        "params-decoder-layers-post_self_attention_norm_global-scale": scale_rmsnorm_layer,
        "params-decoder-layers-post_ffw_norm_global-scale": scale_rmsnorm_layer,
        "params-decoder-layers-pre_ffw_norm_global-scale": scale_rmsnorm_layer,
        "params-decoder-layers-pre_self_attention_norm_local-scale": scale_rmsnorm_layer,
        "params-decoder-layers-post_self_attention_norm_local-scale": scale_rmsnorm_layer,
        "params-decoder-layers-post_ffw_norm_local-scale": scale_rmsnorm_layer,
        "params-decoder-layers-pre_ffw_norm_local-scale": scale_rmsnorm_layer,
    }
  else:
    for maxtext_layer_idx in range(nlayers // 2):
      mapping = {
          **mapping,
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_global-query-kernel": query_hook_chain,
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_local-query-kernel": query_hook_chain,
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_global-key-kernel": reshape_kernel,
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_local-key-kernel": reshape_kernel,
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_global-value-kernel": reshape_kernel,
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_local-value-kernel": reshape_kernel,
          f"params-decoder-layers_{maxtext_layer_idx}-mlp_global-wo-kernel": reshape_kernel,
          f"params-decoder-layers_{maxtext_layer_idx}-mlp_global-wi_1-kernel": reshape_kernel,
          f"params-decoder-layers_{maxtext_layer_idx}-mlp_global-wi_0-kernel": reshape_kernel,
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_global-out-kernel": reshape_kernel,
          f"params-decoder-layers_{maxtext_layer_idx}-mlp_local-wo-kernel": reshape_kernel,
          f"params-decoder-layers_{maxtext_layer_idx}-mlp_local-wi_1-kernel": reshape_kernel,
          f"params-decoder-layers_{maxtext_layer_idx}-mlp_local-wi_0-kernel": reshape_kernel,
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_local-out-kernel": reshape_kernel,
          f"params-decoder-layers_{maxtext_layer_idx}-pre_self_attention_norm_global-scale": scale_rmsnorm_layer,
          f"params-decoder-layers_{maxtext_layer_idx}-post_self_attention_norm_global-scale": scale_rmsnorm_layer,
          f"params-decoder-layers_{maxtext_layer_idx}-post_ffw_norm_global-scale": scale_rmsnorm_layer,
          f"params-decoder-layers_{maxtext_layer_idx}-pre_ffw_norm_global-scale": scale_rmsnorm_layer,
          f"params-decoder-layers_{maxtext_layer_idx}-pre_self_attention_norm_local-scale": scale_rmsnorm_layer,
          f"params-decoder-layers_{maxtext_layer_idx}-post_self_attention_norm_local-scale": scale_rmsnorm_layer,
          f"params-decoder-layers_{maxtext_layer_idx}-post_ffw_norm_local-scale": scale_rmsnorm_layer,
          f"params-decoder-layers_{maxtext_layer_idx}-pre_ffw_norm_local-scale": scale_rmsnorm_layer,
      }
  return mapping


def QWEN_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  \"\"\"Returns mapping from MaxText to HuggingFace Qwen weight paths.

  This function generates a dictionary that maps parameter names from a MaxText
  Qwen checkpoint to their corresponding names in the Hugging Face format.
  It handles both dense and Mixture-of-Experts (MoE) model variants.

  Args:
    config (dict): Model configuration dictionary, including
      'num_hidden_layers' and optionally 'num_experts'.
    scan_layers (bool, optional): Whether the MaxText model uses scanned
      layers. Defaults to False.

  Returns:
    dict: A mapping where keys are `atomic_mt_key` (single MaxText parameter names).
      Values are Hugging Face parameter names in one of four forms: unscanned (string),
      scanned (list of strings), unscanned with expert stacking (list of strings),
      or scanned with expert stacking (nested list of strings).
  \"\"\"
  n_layers = config["num_hidden_layers"]
  num_experts = config.get("num_experts", 0)

  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
      "params-decoder-logits_dense-kernel": "lm_head.weight",
  }

  if scan_layers:
    # This block handles scanned layers for both dense and MoE models.
    mapping.update(
        {
            "params-decoder-layers-pre_self_attention_layer_norm-scale": [
                f"model.layers.{i}.input_layernorm.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-self_attention-query-kernel": [
                f"model.layers.{i}.self_attn.q_proj.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-self_attention-key-kernel": [
                f"model.layers.{i}.self_attn.k_proj.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-self_attention-value-kernel": [
                f"model.layers.{i}.self_attn.v_proj.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-self_attention-query-bias": [
                f"model.layers.{i}.self_attn.q_proj.bias" for i in range(n_layers)
            ],
            "params-decoder-layers-self_attention-key-bias": [
                f"model.layers.{i}.self_attn.k_proj.bias" for i in range(n_layers)
            ],
            "params-decoder-layers-self_attention-value-bias": [
                f"model.layers.{i}.self_attn.v_proj.bias" for i in range(n_layers)
            ],
            "params-decoder-layers-self_attention-out-kernel": [
                f"model.layers.{i}.self_attn.o_proj.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-self_attention-query_norm-scale": [
                f"model.layers.{i}.self_attn.q_norm.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-self_attention-key_norm-scale": [
                f"model.layers.{i}.self_attn.k_norm.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-post_self_attention_layer_norm-scale": [
                f"model.layers.{i}.post_attention_layernorm.weight" for i in range(n_layers)
            ],
        }
    )
    if num_experts > 1:
      # For scanned MoE, we create a nested list: [[e0_l0, e0_l1..], [e1_l0, e1_l1..]..]
      # This follows the (experts, layers, ...) tensor layout.
      mapping.update(
          {
              "params-decoder-layers-moe_block-gate-kernel": [
                  f"model.layers.{i}.mlp.gate.weight" for i in range(n_layers)
              ],
              "params-decoder-layers-moe_block-wi_0": [
                  [f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight" for l in range(n_layers)]
                  for e in range(num_experts)
              ],
              "params-decoder-layers-moe_block-wi_1": [
                  [f"model.layers.{l}.mlp.experts.{e}.up_proj.weight" for l in range(n_layers)]
                  for e in range(num_experts)
              ],
              "params-decoder-layers-moe_block-wo": [
                  [f"model.layers.{l}.mlp.experts.{e}.down_proj.weight" for l in range(n_layers)]
                  for e in range(num_experts)
              ],
          }
      )
    else:  # Dense MLP
      mapping.update(
          {
              "params-decoder-layers-mlp-wi_0-kernel": [
                  f"model.layers.{i}.mlp.gate_proj.weight" for i in range(n_layers)
              ],
              "params-decoder-layers-mlp-wi_1-kernel": [f"model.layers.{i}.mlp.up_proj.weight" for i in range(n_layers)],
              "params-decoder-layers-mlp-wo-kernel": [f"model.layers.{i}.mlp.down_proj.weight" for i in range(n_layers)],
          }
      )
  else:  # unscanned layers
    for i in range(n_layers):
      # Common Attention and Norms
      # pylint: disable=line-too-long
      mapping.update(
          {
              f"params-decoder-layers_{i}-pre_self_attention_layer_norm-scale": f"model.layers.{i}.input_layernorm.weight",
              f"params-decoder-layers_{i}-self_attention-query-kernel": f"model.layers.{i}.self_attn.q_proj.weight",
              f"params-decoder-layers_{i}-self_attention-key-kernel": f"model.layers.{i}.self_attn.k_proj.weight",
              f"params-decoder-layers_{i}-self_attention-value-kernel": f"model.layers.{i}.self_attn.v_proj.weight",
              f"params-decoder-layers_{i}-self_attention-out-kernel": f"model.layers.{i}.self_attn.o_proj.weight",
              f"params-decoder-layers_{i}-self_attention-query-bias": f"model.layers.{i}.self_attn.q_proj.bias",
              f"params-decoder-layers_{i}-self_attention-key-bias": f"model.layers.{i}.self_attn.k_proj.bias",
              f"params-decoder-layers_{i}-self_attention-value-bias": f"model.layers.{i}.self_attn.v_proj.bias",
              f"params-decoder-layers_{i}-self_attention-query_norm-scale": f"model.layers.{i}.self_attn.q_norm.weight",
              f"params-decoder-layers_{i}-self_attention-key_norm-scale": f"model.layers.{i}.self_attn.k_norm.weight",
              f"params-decoder-layers_{i}-post_self_attention_layer_norm-scale": f"model.layers.{i}.post_attention_layernorm.weight",
              f"params-decoder-layers_{i}-post_self_attention_layer_norm-scale": f"model.layers.{i}.post_attention_layernorm.weight",
          }
      )
      if num_experts > 1:
        # For each unscanned MoE layer, map the MaxText parameter to a 1D list of all expert weights for that layer.
        mapping.update(
            {
                f"params-decoder-layers_{i}-moe_block-gate-kernel": f"model.layers.{i}.mlp.gate.weight",
                f"params-decoder-layers_{i}-moe_block-wi_0": [
                    f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight" for j in range(num_experts)
                ],
                f"params-decoder-layers_{i}-moe_block-wi_1": [
                    f"model.layers.{i}.mlp.experts.{j}.up_proj.weight" for j in range(num_experts)
                ],
                f"params-decoder-layers_{i}-moe_block-wo": [
                    f"model.layers.{i}.mlp.experts.{j}.down_proj.weight" for j in range(num_experts)
                ],
            }
        )
      else:  # Dense MLP
        mapping.update(
            {
                f"params-decoder-layers_{i}-mlp-wi_0-kernel": f"model.layers.{i}.mlp.gate_proj.weight",
                f"params-decoder-layers_{i}-mlp-wi_1-kernel": f"model.layers.{i}.mlp.up_proj.weight",
                f"params-decoder-layers_{i}-mlp-wo-kernel": f"model.layers.{i}.mlp.down_proj.weight",
            }
        )
  return mapping


def QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  \"\"\"Creates parameter transformation functions for Qwen.

  This function provides a dictionary of transformation functions (hooks) for
  converting Qwen model parameters between MaxText and Hugging Face formats.
  It handles embedding padding and kernel reshaping.

  Args:
    config (dict): Model configuration dictionary, including
      'num_hidden_layers' and optionally 'num_experts'.
    scan_layers (bool, optional): Whether the model uses scanned layers.
      Defaults to False.
    saving_to_hf (bool, optional): The direction of conversion. True for
      MaxText to Hugging Face, False for the reverse. Defaults to False.

  Returns:
    dict: A dictionary mapping MaxText parameter names to their corresponding
      transformation functions.
  \"\"\"
  n_layers = config["num_hidden_layers"]
  num_experts = config.get("num_experts", 0)

  def pad_embedding_layer(input_tensor, target_shape):
    \"\"\"Pads or truncates embedding layer to match target vocab size.\"\"\"
    source_vocab_size = input_tensor.shape[0]
    target_vocab_size = target_shape[0]

    if source_vocab_size == target_vocab_size:
      return input_tensor

    if saving_to_hf:  # MaxText to HF, truncate
      return input_tensor[:target_vocab_size, :]
    else:  # HF to MaxText, pad
      padded_tensor = np.zeros(target_shape, dtype=input_tensor.dtype)
      padded_tensor[:source_vocab_size, :] = input_tensor
      return padded_tensor

  def reshape_kernel(input_tensor, target_shape):
    \"\"\"Reshapes and transposes kernel weights between MaxText and HF.\"\"\"
    if saving_to_hf:
      flipped_target_shape = np.flip(np.array(target_shape))
      return input_tensor.reshape(flipped_target_shape).T
    else:
      return input_tensor.T.reshape(target_shape)

  def reshape_bias(input_tensor, target_shape=None):
    \"\"\"Reshapes biases between MaxText 2D (heads, dim) and HF 1D (hidden).\"\"\"
    # saving_to_hf: MaxText [heads, head_dim] -> HF [hidden_dim] (flatten)
    # loading_to_maxtext: HF [hidden_dim] -> MaxText [heads, head_dim]
    return input_tensor.reshape(target_shape)

  mapping = {
      "params-token_embedder-embedding": pad_embedding_layer,
      "params-decoder-logits_dense-kernel": reshape_kernel,
  }

  kernel_hooks = [
      "self_attention-query-kernel",
      "self_attention-key-kernel",
      "self_attention-value-kernel",
      "self_attention-out-kernel",
      "mlp-wi_0-kernel",
      "mlp-wi_1-kernel",
      "mlp-wo-kernel",
  ]
  bias_hooks = [
      "self_attention-query-bias",
      "self_attention-key-bias",
      "self_attention-value-bias",
  ]
  moe_kernel_hooks = [
      "moe_block-gate-kernel",
      "moe_block-wi_0-kernel",
      "moe_block-wi_1-kernel",
      "moe_block-wo-kernel",
      "moe_block-wi_0",
      "moe_block-wi_1",
      "moe_block-wo",
  ]

  if scan_layers:
    for key in kernel_hooks:
      mapping[f"params-decoder-layers-{key}"] = reshape_kernel
    for key in bias_hooks:
      mapping[f"params-decoder-layers-{key}"] = reshape_bias
    if num_experts > 1:
      for key in moe_kernel_hooks:
        mapping[f"params-decoder-layers-{key}"] = reshape_kernel
  else:
    for i in range(n_layers):
      for key in kernel_hooks:
        mapping[f"params-decoder-layers_{i}-{key}"] = reshape_kernel
      for key in bias_hooks:
        mapping[f"params-decoder-layers_{i}-{key}"] = reshape_bias
      if num_experts > 1:
        for key in moe_kernel_hooks:
          mapping[f"params-decoder-layers_{i}-{key}"] = reshape_kernel
  return mapping


def QWEN3_NEXT_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  \"\"\"
  Returns mapping from MaxText to HuggingFace Qwen3-Next weight paths.
  All MaxText keys start with 'params-' and use '-' separators for scanned layers.
  \"\"\"
  num_main_layers = config["num_hidden_layers"]
  num_experts = config["num_experts"]
  layer_cycle_interval = maxtext_config.inhomogeneous_layer_cycle_interval

  # 1. Non-layer specific weight mappings
  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
      "params-decoder-logits_dense-kernel": "lm_head.weight",
  }

  if scan_layers:
    # 2. Scan over block cycles
    for block_idx in range(layer_cycle_interval):
      hf_indices = list(range(block_idx, num_main_layers, layer_cycle_interval))
      prefix = f"params-decoder-layers-layer_{block_idx}"

      # Layer norms
      mapping[f"{prefix}-input_layernorm-scale"] = [f"model.layers.{i}.input_layernorm.weight" for i in hf_indices]
      mapping[f"{prefix}-post_attention_layernorm-scale"] = [
          f"model.layers.{i}.post_attention_layernorm.weight" for i in hf_indices
      ]

      # Handle Interleaved Attention (Linear vs Full)
      is_full_attention_layer = (block_idx + 1) % layer_cycle_interval == 0

      if is_full_attention_layer:
        mapping.update(
            {
                f"{prefix}-attention-attention-query-kernel": [
                    f"model.layers.{i}.self_attn.q_proj.weight" for i in hf_indices
                ],
                f"{prefix}-attention-attention-key-kernel": [
                    f"model.layers.{i}.self_attn.k_proj.weight" for i in hf_indices
                ],
                f"{prefix}-attention-attention-value-kernel": [
                    f"model.layers.{i}.self_attn.v_proj.weight" for i in hf_indices
                ],
                f"{prefix}-attention-attention-out-kernel": [
                    f"model.layers.{i}.self_attn.o_proj.weight" for i in hf_indices
                ],
                f"{prefix}-attention-attention-query_norm-scale": [
                    f"model.layers.{i}.self_attn.q_norm.weight" for i in hf_indices
                ],
                f"{prefix}-attention-attention-key_norm-scale": [
                    f"model.layers.{i}.self_attn.k_norm.weight" for i in hf_indices
                ],
            }
        )
      else:
        # Linear/Hybrid Attention Block
        mapping.update(
            {
                f"{prefix}-attention-in_proj_qkvz-kernel": [
                    f"model.layers.{i}.linear_attn.in_proj_qkvz.weight" for i in hf_indices
                ],
                f"{prefix}-attention-in_proj_ba-kernel": [
                    f"model.layers.{i}.linear_attn.in_proj_ba.weight" for i in hf_indices
                ],
                f"{prefix}-attention-conv1d-kernel": [f"model.layers.{i}.linear_attn.conv1d.weight" for i in hf_indices],
                f"{prefix}-attention-A_log": [f"model.layers.{i}.linear_attn.A_log" for i in hf_indices],
                f"{prefix}-attention-dt_bias": [f"model.layers.{i}.linear_attn.dt_bias" for i in hf_indices],
                f"{prefix}-attention-norm-rms_norm-scale": [
                    f"model.layers.{i}.linear_attn.norm.weight" for i in hf_indices
                ],
                f"{prefix}-attention-out_proj-kernel": [
                    f"model.layers.{i}.linear_attn.out_proj.weight" for i in hf_indices
                ],
            }
        )

      # 3. Handle MLP: Gates and Shared Experts
      mapping.update(
          {
              f"{prefix}-mlp-routed_experts-gate-kernel": [f"model.layers.{i}.mlp.gate.weight" for i in hf_indices],
              f"{prefix}-mlp-shared_expert-wi_0-kernel": [
                  f"model.layers.{i}.mlp.shared_expert.gate_proj.weight" for i in hf_indices
              ],
              f"{prefix}-mlp-shared_expert-wi_1-kernel": [
                  f"model.layers.{i}.mlp.shared_expert.up_proj.weight" for i in hf_indices
              ],
              f"{prefix}-mlp-shared_expert-wo-kernel": [
                  f"model.layers.{i}.mlp.shared_expert.down_proj.weight" for i in hf_indices
              ],
              f"{prefix}-mlp-shared_expert_gate-kernel": [
                  f"model.layers.{i}.mlp.shared_expert_gate.weight" for i in hf_indices
              ],
          }
      )

      # 4. Handle MoE Routed Experts
      mapping.update(
          {
              f"{prefix}-mlp-routed_experts-wi_0": [
                  [f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight" for i in hf_indices] for e in range(num_experts)
              ],
              f"{prefix}-mlp-routed_experts-wi_1": [
                  [f"model.layers.{i}.mlp.experts.{e}.up_proj.weight" for i in hf_indices] for e in range(num_experts)
              ],
              f"{prefix}-mlp-routed_experts-wo": [
                  [f"model.layers.{i}.mlp.experts.{e}.down_proj.weight" for i in hf_indices] for e in range(num_experts)
              ],
          }
      )
  else:
    # Unscanned layer mapping
    for i in range(num_main_layers):
      prefix = f"params-decoder-layers_{i}"

      # Layer Norms
      mapping[f"{prefix}-input_layernorm-scale"] = f"model.layers.{i}.input_layernorm.weight"
      mapping[f"{prefix}-post_attention_layernorm-scale"] = f"model.layers.{i}.post_attention_layernorm.weight"

      # Determine layer type based on cycle interval
      # Assuming block logic: layer i corresponds to block_idx = i % interval
      block_idx = i % layer_cycle_interval
      is_full_attention_layer = (block_idx + 1) % layer_cycle_interval == 0

      if is_full_attention_layer:
        mapping.update(
            {
                f"{prefix}-attention-attention-query-kernel": f"model.layers.{i}.self_attn.q_proj.weight",
                f"{prefix}-attention-attention-key-kernel": f"model.layers.{i}.self_attn.k_proj.weight",
                f"{prefix}-attention-attention-value-kernel": f"model.layers.{i}.self_attn.v_proj.weight",
                f"{prefix}-attention-attention-out-kernel": f"model.layers.{i}.self_attn.o_proj.weight",
                f"{prefix}-attention-attention-query_norm-scale": f"model.layers.{i}.self_attn.q_norm.weight",
                f"{prefix}-attention-attention-key_norm-scale": f"model.layers.{i}.self_attn.k_norm.weight",
            }
        )
      else:
        # Linear/Hybrid Attention Block
        mapping.update(
            {
                f"{prefix}-attention-in_proj_qkvz-kernel": f"model.layers.{i}.linear_attn.in_proj_qkvz.weight",
                f"{prefix}-attention-in_proj_ba-kernel": f"model.layers.{i}.linear_attn.in_proj_ba.weight",
                f"{prefix}-attention-conv1d-kernel": f"model.layers.{i}.linear_attn.conv1d.weight",
                f"{prefix}-attention-A_log": f"model.layers.{i}.linear_attn.A_log",
                f"{prefix}-attention-dt_bias": f"model.layers.{i}.linear_attn.dt_bias",
                f"{prefix}-attention-norm-rms_norm-scale": f"model.layers.{i}.linear_attn.norm.weight",
                f"{prefix}-attention-out_proj-kernel": f"model.layers.{i}.linear_attn.out_proj.weight",
            }
        )

      # MLP: Gates and Shared Experts
      mapping.update(
          {
              f"{prefix}-mlp-routed_experts-gate-kernel": f"model.layers.{i}.mlp.gate.weight",
              f"{prefix}-mlp-shared_expert-wi_0-kernel": f"model.layers.{i}.mlp.shared_expert.gate_proj.weight",
              f"{prefix}-mlp-shared_expert-wi_1-kernel": f"model.layers.{i}.mlp.shared_expert.up_proj.weight",
              f"{prefix}-mlp-shared_expert-wo-kernel": f"model.layers.{i}.mlp.shared_expert.down_proj.weight",
              f"{prefix}-mlp-shared_expert_gate-kernel": f"model.layers.{i}.mlp.shared_expert_gate.weight",
          }
      )

      # MoE Routed Experts (List of expert weights for this specific layer)
      mapping.update(
          {
              f"{prefix}-mlp-routed_experts-wi_0": [
                  f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight" for e in range(num_experts)
              ],
              f"{prefix}-mlp-routed_experts-wi_1": [
                  f"model.layers.{i}.mlp.experts.{e}.up_proj.weight" for e in range(num_experts)
              ],
              f"{prefix}-mlp-routed_experts-wo": [
                  f"model.layers.{i}.mlp.experts.{e}.down_proj.weight" for e in range(num_experts)
              ],
          }
      )
  return mapping


def QWEN3_NEXT_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  \"\"\"
  Transformation hooks for parameters using hyphenated 'params-' MaxText keys.
  \"\"\"

  def transpose(input_tensor, target_shape=None):
    return input_tensor.T

  def reshape_kernel(input_tensor, target_shape):
    if saving_to_hf:
      flipped_target_shape = np.flip(np.array(target_shape))
      return input_tensor.reshape(flipped_target_shape).T
    else:
      return input_tensor.T.reshape(target_shape)

  def permute_conv(input_tensor, target_shape=None):
    # MT: [K, 1, C] <-> HF: [C, 1, K]
    return input_tensor.transpose(2, 1, 0)

  # Initialize Hooks
  hooks = {
      "params-decoder-logits_dense-kernel": transpose,
  }

  layer_cycle_interval = maxtext_config.inhomogeneous_layer_cycle_interval
  num_main_layers = config["num_hidden_layers"]
  loop_indices = range(layer_cycle_interval) if scan_layers else range(num_main_layers)

  for i in loop_indices:
    if scan_layers:
      prefix = f"params-decoder-layers-layer_{i}"
      block_idx = i
    else:
      prefix = f"params-decoder-layers_{i}"
      block_idx = i % layer_cycle_interval
    is_full_attention_layer = (block_idx + 1) % layer_cycle_interval == 0

    if is_full_attention_layer:
      for key in ["query", "key", "value", "out"]:
        hooks[f"{prefix}-attention-attention-{key}-kernel"] = reshape_kernel
    else:
      hooks[f"{prefix}-attention-in_proj_qkvz-kernel"] = transpose
      hooks[f"{prefix}-attention-in_proj_ba-kernel"] = transpose
      hooks[f"{prefix}-attention-out_proj-kernel"] = transpose
      hooks[f"{prefix}-attention-conv1d-kernel"] = permute_conv

    mlp_prefix = f"{prefix}-mlp"
    hooks[f"{mlp_prefix}-routed_experts-gate-kernel"] = transpose
    hooks[f"{mlp_prefix}-shared_expert-wi_0-kernel"] = transpose
    hooks[f"{mlp_prefix}-shared_expert-wi_1-kernel"] = transpose
    hooks[f"{mlp_prefix}-shared_expert-wo-kernel"] = transpose
    hooks[f"{mlp_prefix}-shared_expert_gate-kernel"] = transpose

    hooks[f"{mlp_prefix}-routed_experts-wi_0"] = transpose
    hooks[f"{mlp_prefix}-routed_experts-wi_1"] = transpose
    hooks[f"{mlp_prefix}-routed_experts-wo"] = transpose

  return hooks


def DEEPSEEK_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  \"\"\"Generates a parameter mapping from MaxText to HuggingFace Deepseek weight paths.

  Returns:
    dict: A mapping where keys are `atomic_mt_key` (single MaxText parameter names).
      Values are Hugging Face parameter names in one of four forms: unscanned (string),
      scanned (list of strings), unscanned with expert stacking (list of strings),
      or scanned with expert stacking (nested list of strings).
  \"\"\"
  # Extract hf configuration parameters, without mtp
  num_main_layers = config["num_hidden_layers"]
  first_num_dense_layers = config["first_k_dense_replace"]
  num_experts = config.get("n_routed_experts", 0)

  # Mapping for non-layer-specific weights
  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
      "params-decoder-logits_dense-kernel": "lm_head.weight",
  }
  # Attention keys are shared by both dense and MoE
  attention_keys = {
      "pre_self_attention_layer_norm-scale": "input_layernorm.weight",
      "post_self_attention_layer_norm-scale": "post_attention_layernorm.weight",
      "self_attention-kv_norm-scale": "self_attn.kv_a_layernorm.weight",
      "self_attention-wkv_a-kernel": "self_attn.kv_a_proj_with_mqa.weight",
      "self_attention-wkv_b-kernel": "self_attn.kv_b_proj.weight",
      "self_attention-out-kernel": "self_attn.o_proj.weight",
      # v2
      "self_attention-query-kernel": "self_attn.q_proj.weight",
      # v3
      "self_attention-q_norm-scale": "self_attn.q_a_layernorm.weight",
      "self_attention-wq_a-kernel": "self_attn.q_a_proj.weight",
      "self_attention-wq_b-kernel": "self_attn.q_b_proj.weight",
      # v3.2
      "self_attention-indexer-k_norm-bias": "self_attn.indexer.k_norm.bias",
      "self_attention-indexer-k_norm-scale": "self_attn.indexer.k_norm.weight",
      "self_attention-indexer-weights_proj-kernel": "self_attn.indexer.weights_proj.weight",
      "self_attention-indexer-wk-kernel": "self_attn.indexer.wk.weight",
      "self_attention-indexer-wq_b-kernel": "self_attn.indexer.wq_b.weight",
  }
  # Dense Layers
  dense_layer_keys = attention_keys | {
      "mlp-wi_0-kernel": "mlp.gate_proj.weight",
      "mlp-wi_1-kernel": "mlp.up_proj.weight",
      "mlp-wo-kernel": "mlp.down_proj.weight",
  }
  # MoE Layers
  moe_layer_keys = attention_keys | {
      "DeepSeekMoeBlock_0-shared_experts-wi_0-kernel": "mlp.shared_experts.gate_proj.weight",
      "DeepSeekMoeBlock_0-shared_experts-wi_1-kernel": "mlp.shared_experts.up_proj.weight",
      "DeepSeekMoeBlock_0-shared_experts-wo-kernel": "mlp.shared_experts.down_proj.weight",
      "DeepSeekMoeBlock_0-MoeBlock_0-gate-kernel": "mlp.gate.weight",
      # v3
      "DeepSeekMoeBlock_0-MoeBlock_0-gate-bias": "mlp.gate.e_score_correction_bias",
  }
  # MoE Experts (nested list mapping: [[e0_l0, e0_l1..], [e1_l0, e1_l1..]..])
  moe_expert_keys = {
      "DeepSeekMoeBlock_0-MoeBlock_0-wi_0": "gate_proj.weight",
      "DeepSeekMoeBlock_0-MoeBlock_0-wi_1": "up_proj.weight",
      "DeepSeekMoeBlock_0-MoeBlock_0-wo": "down_proj.weight",
  }

  # scan
  if scan_layers:
    for maxtext_key, hf_key in dense_layer_keys.items():
      mapping[f"params-decoder-dense_layers-{maxtext_key}"] = [
          f"model.layers.{i}.{hf_key}" for i in range(first_num_dense_layers)
      ]

    for maxtext_key, hf_key in moe_layer_keys.items():
      mapping[f"params-decoder-moe_layers-{maxtext_key}"] = [
          f"model.layers.{i}.{hf_key}" for i in range(first_num_dense_layers, num_main_layers)
      ]

    for maxtext_key, hf_key in moe_expert_keys.items():
      mapping[f"params-decoder-moe_layers-{maxtext_key}"] = [
          [f"model.layers.{i}.mlp.experts.{e}.{hf_key}" for i in range(first_num_dense_layers, num_main_layers)]
          for e in range(num_experts)
      ]
  # unscan
  else:
    for i in range(first_num_dense_layers):
      for maxtext_key, hf_key in dense_layer_keys.items():
        mapping[f"params-decoder-dense_layers_{i}-{maxtext_key}"] = f"model.layers.{i}.{hf_key}"

    for i in range(first_num_dense_layers, num_main_layers):
      moe_layer_idx = i - first_num_dense_layers

      for maxtext_key, hf_key in moe_layer_keys.items():
        mapping[f"params-decoder-moe_layers_{moe_layer_idx}-{maxtext_key}"] = f"model.layers.{i}.{hf_key}"

      for maxtext_key, hf_key in moe_expert_keys.items():
        mapping[f"params-decoder-moe_layers_{moe_layer_idx}-{maxtext_key}"] = [
            f"model.layers.{i}.mlp.experts.{e}.{hf_key}" for e in range(num_experts)
        ]
  return mapping


def DEEPSEEK_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  \"\"\"Creates parameter transformation functions for Deepseek.\"\"\"

  def reshape_kernel(input_tensor, target_shape):
    \"\"\"Reshapes and transposes kernel weights between MaxText and HF.\"\"\"
    if saving_to_hf:
      flipped_target_shape = np.flip(np.array(target_shape))
      return input_tensor.reshape(flipped_target_shape).T
    else:
      return input_tensor.T.reshape(target_shape)

  num_main_layers = config["num_hidden_layers"]
  first_num_dense_layers = config["first_k_dense_replace"]

  mapping = {
      "params-decoder-logits_dense-kernel": reshape_kernel,
  }

  attention_need_reshape = {
      "self_attention-wkv_a-kernel",  # transpose
      "self_attention-wkv_b-kernel",
      "self_attention-out-kernel",
      # v2
      "self_attention-query-kernel",
      # v3
      "self_attention-wq_a-kernel",  # transpose
      "self_attention-wq_b-kernel",
      # v3.2
      "self_attention-indexer-weights_proj-kernel",  # transpose
      "self_attention-indexer-wk-kernel",  # transpose
      "self_attention-indexer-wq_b-kernel",
  }

  dense_need_reshape = attention_need_reshape | {
      "mlp-wi_0-kernel",  # transpose
      "mlp-wi_1-kernel",  # transpose
      "mlp-wo-kernel",  # transpose
  }

  moe_need_reshape = attention_need_reshape | {
      "DeepSeekMoeBlock_0-shared_experts-wi_0-kernel",  # transpose
      "DeepSeekMoeBlock_0-shared_experts-wi_1-kernel",  # transpose
      "DeepSeekMoeBlock_0-shared_experts-wo-kernel",  # transpose
      "DeepSeekMoeBlock_0-MoeBlock_0-gate-kernel",  # transpose
      "DeepSeekMoeBlock_0-MoeBlock_0-wi_0",  # transpose
      "DeepSeekMoeBlock_0-MoeBlock_0-wi_1",  # transpose
      "DeepSeekMoeBlock_0-MoeBlock_0-wo",  # transpose
  }

  # scan
  if scan_layers:
    for key in dense_need_reshape:
      mapping[f"params-decoder-dense_layers-{key}"] = reshape_kernel
    for key in moe_need_reshape:
      mapping[f"params-decoder-moe_layers-{key}"] = reshape_kernel
  # unscan
  else:
    for i in range(first_num_dense_layers):
      for key in dense_need_reshape:
        mapping[f"params-decoder-dense_layers_{i}-{key}"] = reshape_kernel
    for i in range(first_num_dense_layers, num_main_layers):
      moe_layer_idx = i - first_num_dense_layers
      for key in moe_need_reshape:
        mapping[f"params-decoder-moe_layers_{moe_layer_idx}-{key}"] = reshape_kernel

  return mapping


def DEEPSEEK_NNX_TO_VLLM_PARAM_HOOK_FN():
  \"\"\"Creates parameter transformation functions for Deepseek.\"\"\"
  return {}


def GPT_OSS_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  \"\"\"Generates mapping from MaxText gpt-oss to Hugging Face weight paths.

  Returns:
    dict: A mapping where keys are `atomic_mt_key` (single MaxText parameter) or
    `composite_mt_key` (a tuple of MaxText parameters). Values are Hugging Face parameter
    names either a single string (unscanned form) or a list of strings (scanned form).

  Notes:
  - Handles the inhomogeneous scan block structure, based on `inhomogeneous_layer_cycle_interval`
  - Handles `composite_mt_key`: multiple MaxText keys map to HF key(s)
    - (GptOssMlp-wi_0, GptOssMlp-wi_1): mlp.experts.gate_up_proj
    - (GptOssMlp-wi_0_bias, GptOssMlp-wi_1_bias): mlp.experts.gate_up_proj_bias
  \"\"\"
  n_layers = config["num_hidden_layers"]  # hf config
  layer_cycle_interval = maxtext_config.inhomogeneous_layer_cycle_interval

  # Base mapping for non-layer parameters (targeting standard HF keys)
  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
      "params-decoder-logits_dense-kernel": "lm_head.weight",
  }

  if scan_layers:
    # Scan over blocks
    for block_idx in range(layer_cycle_interval):
      # Identify all original HF layer indices that collapse into this block
      hf_indices = range(block_idx, n_layers, layer_cycle_interval)
      prefix = f"params-decoder-layers-layers_{block_idx}"
      block_mapping = {
          # Layer Norms
          f"{prefix}-pre_self_attention_layer_norm-scale": [
              f"model.layers.{i}.input_layernorm.weight" for i in hf_indices
          ],
          f"{prefix}-post_self_attention_layer_norm-scale": [
              f"model.layers.{i}.post_attention_layernorm.weight" for i in hf_indices
          ],
          # GptOssAttention
          f"{prefix}-GptOssAttention-query-kernel": [f"model.layers.{i}.self_attn.q_proj.weight" for i in hf_indices],
          f"{prefix}-GptOssAttention-query-bias": [f"model.layers.{i}.self_attn.q_proj.bias" for i in hf_indices],
          f"{prefix}-GptOssAttention-key-kernel": [f"model.layers.{i}.self_attn.k_proj.weight" for i in hf_indices],
          f"{prefix}-GptOssAttention-key-bias": [f"model.layers.{i}.self_attn.k_proj.bias" for i in hf_indices],
          f"{prefix}-GptOssAttention-value-kernel": [f"model.layers.{i}.self_attn.v_proj.weight" for i in hf_indices],
          f"{prefix}-GptOssAttention-value-bias": [f"model.layers.{i}.self_attn.v_proj.bias" for i in hf_indices],
          f"{prefix}-GptOssAttention-out-kernel": [f"model.layers.{i}.self_attn.o_proj.weight" for i in hf_indices],
          f"{prefix}-GptOssAttention-out-bias": [f"model.layers.{i}.self_attn.o_proj.bias" for i in hf_indices],
          f"{prefix}-GptOssAttention-sinks": [f"model.layers.{i}.self_attn.sinks" for i in hf_indices],
          # GptOssMlp
          # 1. Gate/Router
          f"{prefix}-GptOssMlp-gate-kernel": [f"model.layers.{i}.mlp.router.weight" for i in hf_indices],
          f"{prefix}-GptOssMlp-gate-bias": [f"model.layers.{i}.mlp.router.bias" for i in hf_indices],
          # 2. Experts (Down Projection)
          f"{prefix}-GptOssMlp-wo": [f"model.layers.{i}.mlp.experts.down_proj" for i in hf_indices],
          f"{prefix}-GptOssMlp-wo_bias": [f"model.layers.{i}.mlp.experts.down_proj_bias" for i in hf_indices],
          # 3. Experts (Gate/Up Fused Projection)
          # `composite_mt_key`: Multiple MaxText keys map to HF key(s).
          (f"{prefix}-GptOssMlp-wi_0", f"{prefix}-GptOssMlp-wi_1"): [
              f"model.layers.{i}.mlp.experts.gate_up_proj" for i in hf_indices
          ],
          (f"{prefix}-GptOssMlp-wi_0_bias", f"{prefix}-GptOssMlp-wi_1_bias"): [
              f"model.layers.{i}.mlp.experts.gate_up_proj_bias" for i in hf_indices
          ],
      }
      mapping.update(block_mapping)

  else:
    # Unscan
    for i in range(n_layers):
      prefix = f"params-decoder-layers_{i}"
      layer_mapping = {
          # Layer Norms
          f"{prefix}-pre_self_attention_layer_norm-scale": f"model.layers.{i}.input_layernorm.weight",
          f"{prefix}-post_self_attention_layer_norm-scale": f"model.layers.{i}.post_attention_layernorm.weight",
          # GptOssAttention
          f"{prefix}-GptOssAttention-query-kernel": f"model.layers.{i}.self_attn.q_proj.weight",
          f"{prefix}-GptOssAttention-query-bias": f"model.layers.{i}.self_attn.q_proj.bias",
          f"{prefix}-GptOssAttention-key-kernel": f"model.layers.{i}.self_attn.k_proj.weight",
          f"{prefix}-GptOssAttention-key-bias": f"model.layers.{i}.self_attn.k_proj.bias",
          f"{prefix}-GptOssAttention-value-kernel": f"model.layers.{i}.self_attn.v_proj.weight",
          f"{prefix}-GptOssAttention-value-bias": f"model.layers.{i}.self_attn.v_proj.bias",
          f"{prefix}-GptOssAttention-out-kernel": f"model.layers.{i}.self_attn.o_proj.weight",
          f"{prefix}-GptOssAttention-out-bias": f"model.layers.{i}.self_attn.o_proj.bias",
          f"{prefix}-GptOssAttention-sinks": f"model.layers.{i}.self_attn.sinks",
          # GptOssMlp
          # 1. Gate/Router
          f"{prefix}-GptOssMlp-gate-kernel": f"model.layers.{i}.mlp.router.weight",
          f"{prefix}-GptOssMlp-gate-bias": f"model.layers.{i}.mlp.router.bias",
          # 2. Experts (Down Projection)
          f"{prefix}-GptOssMlp-wo": f"model.layers.{i}.mlp.experts.down_proj",
          f"{prefix}-GptOssMlp-wo_bias": f"model.layers.{i}.mlp.experts.down_proj_bias",
          # 3. Experts (Gate/Up Fused Projection)
          # `composite_mt_key`: Multiple MaxText keys map to HF key(s).
          (f"{prefix}-GptOssMlp-wi_0", f"{prefix}-GptOssMlp-wi_1"): f"model.layers.{i}.mlp.experts.gate_up_proj",
          (
              f"{prefix}-GptOssMlp-wi_0_bias",
              f"{prefix}-GptOssMlp-wi_1_bias",
          ): f"model.layers.{i}.mlp.experts.gate_up_proj_bias",
      }
      mapping.update(layer_mapping)

  return mapping


def GPT_OSS_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  \"\"\"Transformation hooks for gpt-oss parameters.

  Notes:
  - Handles the inhomogeneous scan block structure (inhomogeneous_layer_cycle_interval)
  - Handles `composite_mt_key` where multiple MaxText keys map to HF key(s)
    - (GptOssMlp-wi_0, GptOssMlp-wi_1): mlp.experts.gate_up_proj
    - (GptOssMlp-wi_0_bias, GptOssMlp-wi_1_bias): mlp.experts.gate_up_proj_bias
    - The composite keys are transformed via `interleave` function
  \"\"\"

  def transpose(input_tensor, target_shape=None):
    return input_tensor.T

  def reshape_kernel(input_tensor, target_shape):
    \"\"\"Reshapes and transposes kernel weights between MaxText and HF.\"\"\"
    if saving_to_hf:
      flipped_target_shape = np.flip(np.array(target_shape))
      return input_tensor.reshape(flipped_target_shape).T
    else:
      return input_tensor.T.reshape(target_shape)

  def reshape_bias(input_tensor, target_shape=None):
    \"\"\"Reshapes biases between MaxText 2D (heads, dim) and HF 1D (hidden).\"\"\"
    if saving_to_hf:
      # MaxText [heads, head_dim] -> HF [hidden_dim] (flatten)
      return input_tensor.reshape(target_shape)
    else:
      # HF [hidden_dim] -> MaxText [heads, head_dim]
      return input_tensor.reshape(target_shape)

  def interleave(input_tensor, target_shape=None):
    \"\"\"
    Handles `composite_mt_key`: maxtext (wi_0, wi_1) <-> hf (wi_0_1)
    - if saving_to_hf: (wi_0, wi_1) -> wi_0_1
      - input_tensor is a list of two tensors, tensor ORDER must be same as key order
      - return a single tensor
    - otherwise: wi_0_1 -> (wi_0, wi_1)
      - input_tensor is a single tensor
      - return two tensors stack at LAST index -1, tensor ORDER must be same as key order
    \"\"\"
    if saving_to_hf:
      wi_0, wi_1 = input_tensor
      wi_0_1 = np.empty(target_shape, dtype=wi_0.dtype)
      wi_0_1[..., ::2] = wi_0
      wi_0_1[..., 1::2] = wi_1
      return wi_0_1
    else:
      wi_0_1 = input_tensor
      wi_0 = wi_0_1[..., ::2]
      wi_1 = wi_0_1[..., 1::2]
      return np.stack([wi_0, wi_1], axis=-1)

  n_layers = config["num_hidden_layers"]  # hf config
  layer_cycle_interval = maxtext_config.inhomogeneous_layer_cycle_interval

  hooks = {"params-decoder-logits_dense-kernel": transpose}

  indices = range(layer_cycle_interval) if scan_layers else range(n_layers)
  for idx in indices:
    prefix = f"params-decoder-layers-layers_{idx}" if scan_layers else f"params-decoder-layers_{idx}"
    # Attention Kernels & Biases
    for key in ["query", "key", "value"]:
      hooks[f"{prefix}-GptOssAttention-{key}-kernel"] = reshape_kernel
      hooks[f"{prefix}-GptOssAttention-{key}-bias"] = reshape_bias
    hooks[f"{prefix}-GptOssAttention-out-kernel"] = reshape_kernel
    # MLP Kernels & Biases
    hooks[f"{prefix}-GptOssMlp-gate-kernel"] = transpose
    # `composite_mt_key`: A hook for combining multiple MaxText params.
    hooks[(f"{prefix}-GptOssMlp-wi_0", f"{prefix}-GptOssMlp-wi_1")] = interleave
    hooks[(f"{prefix}-GptOssMlp-wi_0_bias", f"{prefix}-GptOssMlp-wi_1_bias")] = interleave

  return hooks


def QWEN3_OMNI_MOE_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  \"\"\"Returns mapping from MaxText to HuggingFace Qwen3-Omni weight paths.

  This function combines mappings from different modalities (text, vision, audio, etc.)
  into a unified parameter mapping for the multi-modal Qwen3-Omni model.

  Args:
    config (dict): Model configuration dictionary containing modality-specific configs.
    scan_layers (bool, optional): Whether the model uses scanned layers. Defaults to False.

  Returns:
    dict: Combined mapping from all modalities.
  \"\"\"
  # Collect all modality mappings
  mapping = {}

  # Text mapping with "thinker." prefix, reusing QWEN3-MOE mapping function
  num_experts_text = config["thinker_config"]["text_config"].get("num_experts", 0)
  n_layers_text = config["thinker_config"]["text_config"]["num_hidden_layers"]
  text_mapping = QWEN_MAXTEXT_TO_HF_PARAM_MAPPING(
      config={"num_hidden_layers": n_layers_text, "num_experts": num_experts_text},
      maxtext_config=maxtext_config,
      scan_layers=scan_layers,
  )

  # Add "thinker." prefix to text mapping values
  def add_prefix_recursive(value):
    \"\"\"Recursively add 'thinker.' prefix to strings, handling nested lists.\"\"\"
    if isinstance(value, list):
      return [add_prefix_recursive(v) for v in value]
    else:
      return f"thinker.{value}"

  for key, value in text_mapping.items():
    text_mapping[key] = add_prefix_recursive(value)
  mapping.update(text_mapping)

  # Vision mapping
  vision_config = config["thinker_config"]["vision_config"]
  n_vision_layers = vision_config["depth"]

  # Vision patch embedding
  mapping["params-vision_encoder-Qwen3OmniMoeVisionEncoder_0-patch_embed-proj-kernel"] = (
      "thinker.visual.patch_embed.proj.weight"
  )
  mapping["params-vision_encoder-Qwen3OmniMoeVisionEncoder_0-patch_embed-proj-bias"] = (
      "thinker.visual.patch_embed.proj.bias"
  )

  # Vision positional embedding
  mapping["params-vision_encoder-Qwen3OmniMoeVisionEncoder_0-pos_embed_interpolate-pos_embed"] = (
      "thinker.visual.pos_embed.weight"
  )

  # Vision blocks (27 layers)
  for i in range(n_vision_layers):
    prefix = f"params-vision_encoder-Qwen3OmniMoeVisionEncoder_0-blocks_{i}"
    hf_prefix = f"thinker.visual.blocks.{i}"

    # Layer norms
    mapping[f"{prefix}-ln1-scale"] = f"{hf_prefix}.norm1.weight"
    mapping[f"{prefix}-ln1-bias"] = f"{hf_prefix}.norm1.bias"
    mapping[f"{prefix}-ln2-scale"] = f"{hf_prefix}.norm2.weight"
    mapping[f"{prefix}-ln2-bias"] = f"{hf_prefix}.norm2.bias"

    # Attention (HF has fused QKV, MaxText has separate Q/K/V)
    # We'll handle the split/fusion in the hook functions
    mapping[f"{prefix}-attn-attn-query-kernel"] = f"{hf_prefix}.attn.qkv.weight"
    mapping[f"{prefix}-attn-attn-query-bias"] = f"{hf_prefix}.attn.qkv.bias"
    mapping[f"{prefix}-attn-attn-key-kernel"] = f"{hf_prefix}.attn.qkv.weight"
    mapping[f"{prefix}-attn-attn-key-bias"] = f"{hf_prefix}.attn.qkv.bias"
    mapping[f"{prefix}-attn-attn-value-kernel"] = f"{hf_prefix}.attn.qkv.weight"
    mapping[f"{prefix}-attn-attn-value-bias"] = f"{hf_prefix}.attn.qkv.bias"
    mapping[f"{prefix}-attn-attn-out-kernel"] = f"{hf_prefix}.attn.proj.weight"
    mapping[f"{prefix}-attn-attn-out-bias"] = f"{hf_prefix}.attn.proj.bias"

    # MLP
    mapping[f"{prefix}-mlp-kernel"] = f"{hf_prefix}.mlp.linear_fc1.weight"
    mapping[f"{prefix}-mlp-bias"] = f"{hf_prefix}.mlp.linear_fc1.bias"
    mapping[f"{prefix}-mlp_out-kernel"] = f"{hf_prefix}.mlp.linear_fc2.weight"
    mapping[f"{prefix}-mlp_out-bias"] = f"{hf_prefix}.mlp.linear_fc2.bias"

  # Vision merger_list (deep mergers at layers 8, 16, 24)
  deepstack_indexes = vision_config.get("deepstack_visual_indexes", [8, 16, 24])
  for merger_idx, _ in enumerate(deepstack_indexes):
    prefix = f"params-vision_encoder-Qwen3OmniMoeVisionEncoder_0-merger_{merger_idx}"
    hf_prefix = f"thinker.visual.merger_list.{merger_idx}"

    mapping[f"{prefix}-ln_q-scale"] = f"{hf_prefix}.ln_q.weight"
    mapping[f"{prefix}-ln_q-bias"] = f"{hf_prefix}.ln_q.bias"
    mapping[f"{prefix}-mlp_0-kernel"] = f"{hf_prefix}.mlp.0.weight"
    mapping[f"{prefix}-mlp_0-bias"] = f"{hf_prefix}.mlp.0.bias"
    mapping[f"{prefix}-mlp_2-kernel"] = f"{hf_prefix}.mlp.2.weight"
    mapping[f"{prefix}-mlp_2-bias"] = f"{hf_prefix}.mlp.2.bias"

  # Vision projector (final merger)
  mapping["params-vision_encoder-Qwen3OmniMoeVisionProjector_0-merger-ln_q-scale"] = "thinker.visual.merger.ln_q.weight"
  mapping["params-vision_encoder-Qwen3OmniMoeVisionProjector_0-merger-ln_q-bias"] = "thinker.visual.merger.ln_q.bias"
  mapping["params-vision_encoder-Qwen3OmniMoeVisionProjector_0-merger-mlp_0-kernel"] = (
      "thinker.visual.merger.mlp.0.weight"
  )
  mapping["params-vision_encoder-Qwen3OmniMoeVisionProjector_0-merger-mlp_0-bias"] = "thinker.visual.merger.mlp.0.bias"
  mapping["params-vision_encoder-Qwen3OmniMoeVisionProjector_0-merger-mlp_2-kernel"] = (
      "thinker.visual.merger.mlp.2.weight"
  )
  mapping["params-vision_encoder-Qwen3OmniMoeVisionProjector_0-merger-mlp_2-bias"] = "thinker.visual.merger.mlp.2.bias"

  # Audio mapping
  audio_config = config["thinker_config"]["audio_config"]
  n_audio_layers = audio_config["encoder_layers"]

  # Audio conv layers (3 Conv2D layers for downsampling)
  mapping["params-audio_encoder-Qwen3OmniAudioEncoder_0-conv2d1-kernel"] = "thinker.audio_tower.conv2d1.weight"
  mapping["params-audio_encoder-Qwen3OmniAudioEncoder_0-conv2d1-bias"] = "thinker.audio_tower.conv2d1.bias"
  mapping["params-audio_encoder-Qwen3OmniAudioEncoder_0-conv2d2-kernel"] = "thinker.audio_tower.conv2d2.weight"
  mapping["params-audio_encoder-Qwen3OmniAudioEncoder_0-conv2d2-bias"] = "thinker.audio_tower.conv2d2.bias"
  mapping["params-audio_encoder-Qwen3OmniAudioEncoder_0-conv2d3-kernel"] = "thinker.audio_tower.conv2d3.weight"
  mapping["params-audio_encoder-Qwen3OmniAudioEncoder_0-conv2d3-bias"] = "thinker.audio_tower.conv2d3.bias"

  # Audio conv output projection
  mapping["params-audio_encoder-Qwen3OmniAudioEncoder_0-conv_out-kernel"] = "thinker.audio_tower.conv_out.weight"

  # Audio encoder layers (32 layers)
  for i in range(n_audio_layers):
    prefix = f"params-audio_encoder-Qwen3OmniAudioEncoder_0-layers_{i}"
    hf_prefix = f"thinker.audio_tower.layers.{i}"

    # Layer norms
    mapping[f"{prefix}-input_layer_norm-scale"] = f"{hf_prefix}.self_attn_layer_norm.weight"
    mapping[f"{prefix}-input_layer_norm-bias"] = f"{hf_prefix}.self_attn_layer_norm.bias"
    mapping[f"{prefix}-post_attention_layer_norm-scale"] = f"{hf_prefix}.final_layer_norm.weight"
    mapping[f"{prefix}-post_attention_layer_norm-bias"] = f"{hf_prefix}.final_layer_norm.bias"

    # Attention (separate Q/K/V)
    mapping[f"{prefix}-self_attention_audio-query-kernel"] = f"{hf_prefix}.self_attn.q_proj.weight"
    mapping[f"{prefix}-self_attention_audio-query-bias"] = f"{hf_prefix}.self_attn.q_proj.bias"
    mapping[f"{prefix}-self_attention_audio-key-kernel"] = f"{hf_prefix}.self_attn.k_proj.weight"
    mapping[f"{prefix}-self_attention_audio-key-bias"] = f"{hf_prefix}.self_attn.k_proj.bias"
    mapping[f"{prefix}-self_attention_audio-value-kernel"] = f"{hf_prefix}.self_attn.v_proj.weight"
    mapping[f"{prefix}-self_attention_audio-value-bias"] = f"{hf_prefix}.self_attn.v_proj.bias"
    mapping[f"{prefix}-self_attention_audio-out-kernel"] = f"{hf_prefix}.self_attn.out_proj.weight"
    mapping[f"{prefix}-self_attention_audio-out-bias"] = f"{hf_prefix}.self_attn.out_proj.bias"

    # MLP (AudioMLP has 2 linear layers: fc1 and fc2)
    mapping[f"{prefix}-AudioMLP-wi-kernel"] = f"{hf_prefix}.fc1.weight"
    mapping[f"{prefix}-AudioMLP-wi-bias"] = f"{hf_prefix}.fc1.bias"
    mapping[f"{prefix}-AudioMLP-wo-kernel"] = f"{hf_prefix}.fc2.weight"
    mapping[f"{prefix}-AudioMLP-wo-bias"] = f"{hf_prefix}.fc2.bias"

  # Audio post layer norm
  mapping["params-audio_encoder-Qwen3OmniAudioEncoder_0-layernorm_post-scale"] = "thinker.audio_tower.ln_post.weight"
  mapping["params-audio_encoder-Qwen3OmniAudioEncoder_0-layernorm_post-bias"] = "thinker.audio_tower.ln_post.bias"

  # Audio projector (2 linear layers)
  mapping["params-audio_encoder-Qwen3OmniAudioProjector_0-proj1-kernel"] = "thinker.audio_tower.proj1.weight"
  mapping["params-audio_encoder-Qwen3OmniAudioProjector_0-proj1-bias"] = "thinker.audio_tower.proj1.bias"
  mapping["params-audio_encoder-Qwen3OmniAudioProjector_0-proj2-kernel"] = "thinker.audio_tower.proj2.weight"
  mapping["params-audio_encoder-Qwen3OmniAudioProjector_0-proj2-bias"] = "thinker.audio_tower.proj2.bias"

  return mapping


def QWEN3_OMNI_MOE_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  \"\"\"Creates parameter transformation functions for Qwen3-Omni.

  This function provides a dictionary of transformation functions (hooks) for
  converting Qwen3-Omni model parameters between MaxText and Hugging Face formats.
  It handles embedding padding and kernel reshaping.

  Args:
    config (dict): Model configuration dictionary, including
      'num_hidden_layers' and optionally 'num_experts'.
    scan_layers (bool, optional): Whether the model uses scanned layers.
      Defaults to False.
    saving_to_hf (bool, optional): The direction of conversion. True for
      MaxText to Hugging Face, False for the reverse. Defaults to False.

  Returns:
    dict: A dictionary mapping MaxText parameter names to their corresponding
      transformation functions.
  \"\"\"
  # Collect all modality hooks
  mapping = {}

  # Text hooks, reusing QWEN3-MOE hook function
  num_experts_text = config["thinker_config"]["text_config"].get("num_experts", 0)
  n_layers_text = config["thinker_config"]["text_config"]["num_hidden_layers"]
  text_hooks = QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN(
      config={"num_hidden_layers": n_layers_text, "num_experts": num_experts_text},
      maxtext_config=maxtext_config,
      scan_layers=scan_layers,
      saving_to_hf=saving_to_hf,
  )
  mapping.update(text_hooks)

  # Vision hooks
  vision_config = config["thinker_config"]["vision_config"]
  n_vision_layers = vision_config["depth"]
  hidden_size = vision_config["hidden_size"]

  def reshape_kernel_vision(input_tensor, target_shape):
    \"\"\"Reshape kernel for vision layers.\"\"\"
    if saving_to_hf:
      flipped_target_shape = np.flip(np.array(target_shape))
      return input_tensor.reshape(flipped_target_shape).T
    else:
      return input_tensor.T.reshape(target_shape)

  def reshape_conv3d_patch_embed(input_tensor, target_shape):
    \"\"\"Reshape 3D conv patch embedding weight.
    HF: (out_channels, in_channels, temporal, height, width)
    MaxText: (temporal, height, width, in_channels, out_channels)
    \"\"\"
    if saving_to_hf:
      # MaxText -> HF: (T, H, W, C_in, C_out) -> (C_out, C_in, T, H, W)
      return input_tensor.transpose(4, 3, 0, 1, 2)
    else:
      # HF -> MaxText: (C_out, C_in, T, H, W) -> (T, H, W, C_in, C_out)
      return input_tensor.transpose(2, 3, 4, 1, 0)

  def split_qkv_query(input_tensor, target_shape):
    \"\"\"Extract Q from fused QKV for HF->MaxText conversion.
    HF has fused QKV: (3*hidden_size, hidden_size)
    MaxText Q: (hidden_size, num_heads, head_dim)
    \"\"\"
    if saving_to_hf:
      # MaxText -> HF: will be handled by fusion hook
      raise NotImplementedError("Use fusion hook for MaxText->HF")
    else:
      # HF -> MaxText: Extract Q from fused QKV
      # input_tensor shape: (3*hidden_size, hidden_size)
      q_weight = input_tensor[:hidden_size, :]  # (hidden_size, hidden_size)
      return q_weight.T.reshape(target_shape)  # (hidden_size, num_heads, head_dim)

  def split_qkv_key(input_tensor, target_shape):
    \"\"\"Extract K from fused QKV for HF->MaxText conversion.\"\"\"
    if saving_to_hf:
      raise NotImplementedError("Use fusion hook for MaxText->HF")
    else:
      # Extract K from fused QKV
      k_weight = input_tensor[hidden_size : 2 * hidden_size, :]
      return k_weight.T.reshape(target_shape)

  def split_qkv_value(input_tensor, target_shape):
    \"\"\"Extract V from fused QKV for HF->MaxText conversion.\"\"\"
    if saving_to_hf:
      raise NotImplementedError("Use fusion hook for MaxText->HF")
    else:
      # Extract V from fused QKV
      v_weight = input_tensor[2 * hidden_size :, :]
      return v_weight.T.reshape(target_shape)

  def split_qkv_bias_query(input_tensor, target_shape):
    \"\"\"Extract Q bias from fused QKV bias.\"\"\"
    if saving_to_hf:
      raise NotImplementedError("Use fusion hook for MaxText->HF")
    else:
      q_bias = input_tensor[:hidden_size]
      return q_bias.reshape(target_shape)  # (num_heads, head_dim)

  def split_qkv_bias_key(input_tensor, target_shape):
    \"\"\"Extract K bias from fused QKV bias.\"\"\"
    if saving_to_hf:
      raise NotImplementedError("Use fusion hook for MaxText->HF")
    else:
      k_bias = input_tensor[hidden_size : 2 * hidden_size]
      return k_bias.reshape(target_shape)

  def split_qkv_bias_value(input_tensor, target_shape):
    \"\"\"Extract V bias from fused QKV bias.\"\"\"
    if saving_to_hf:
      raise NotImplementedError("Use fusion hook for MaxText->HF")
    else:
      v_bias = input_tensor[2 * hidden_size :]
      return v_bias.reshape(target_shape)

  def reshape_vision_attn_out(input_tensor, target_shape):
    \"\"\"Reshape vision attention output projection.
    HF: (hidden_size, hidden_size)
    MaxText: (num_heads, head_dim, hidden_size)
    \"\"\"
    if saving_to_hf:
      # MaxText -> HF: (num_heads, head_dim, hidden_size) -> (hidden_size, hidden_size)
      return input_tensor.reshape(hidden_size, hidden_size).T
    else:
      # HF -> MaxText: (hidden_size, hidden_size) -> (num_heads, head_dim, hidden_size)
      return input_tensor.T.reshape(target_shape)

  # Vision patch embedding
  mapping["params-vision_encoder-Qwen3OmniMoeVisionEncoder_0-patch_embed-proj-kernel"] = reshape_conv3d_patch_embed

  # Vision blocks
  for i in range(n_vision_layers):
    prefix = f"params-vision_encoder-Qwen3OmniMoeVisionEncoder_0-blocks_{i}"

    # Attention Q/K/V - split from fused QKV
    mapping[f"{prefix}-attn-attn-query-kernel"] = split_qkv_query
    mapping[f"{prefix}-attn-attn-query-bias"] = split_qkv_bias_query
    mapping[f"{prefix}-attn-attn-key-kernel"] = split_qkv_key
    mapping[f"{prefix}-attn-attn-key-bias"] = split_qkv_bias_key
    mapping[f"{prefix}-attn-attn-value-kernel"] = split_qkv_value
    mapping[f"{prefix}-attn-attn-value-bias"] = split_qkv_bias_value

    # Attention output
    mapping[f"{prefix}-attn-attn-out-kernel"] = reshape_vision_attn_out
    # attn-attn-out-bias doesn't need a hook (no reshape needed)

    # MLP
    mapping[f"{prefix}-mlp-kernel"] = reshape_kernel_vision
    mapping[f"{prefix}-mlp_out-kernel"] = reshape_kernel_vision

  # Vision merger_list and projector MLPs
  deepstack_indexes = vision_config.get("deepstack_visual_indexes", [8, 16, 24])
  for merger_idx, _ in enumerate(deepstack_indexes):
    prefix = f"params-vision_encoder-Qwen3OmniMoeVisionEncoder_0-merger_{merger_idx}"
    mapping[f"{prefix}-mlp_0-kernel"] = reshape_kernel_vision
    mapping[f"{prefix}-mlp_2-kernel"] = reshape_kernel_vision

  # Vision projector (final merger)
  mapping["params-vision_encoder-Qwen3OmniMoeVisionProjector_0-merger-mlp_0-kernel"] = reshape_kernel_vision
  mapping["params-vision_encoder-Qwen3OmniMoeVisionProjector_0-merger-mlp_2-kernel"] = reshape_kernel_vision

  # Audio hooks
  audio_config = config["thinker_config"]["audio_config"]
  n_audio_layers = audio_config["encoder_layers"]
  hidden_size_audio = audio_config["d_model"]

  def reshape_kernel_audio(input_tensor, target_shape):
    \"\"\"Reshape kernel for audio layers.\"\"\"
    if saving_to_hf:
      flipped_target_shape = np.flip(np.array(target_shape))
      return input_tensor.reshape(flipped_target_shape).T
    else:
      return input_tensor.T.reshape(target_shape)

  def reshape_conv2d_audio(input_tensor, target_shape):
    \"\"\"Reshape Conv2D weight for audio.

    HF: (out_channels, in_channels, height, width)
    MaxText: (height, width, in_channels, out_channels)
    \"\"\"
    if saving_to_hf:
      # MaxText -> HF: (H, W, C_in, C_out) -> (C_out, C_in, H, W)
      return input_tensor.transpose(3, 2, 0, 1)
    else:
      # HF -> MaxText: (C_out, C_in, H, W) -> (H, W, C_in, C_out)
      return input_tensor.transpose(2, 3, 1, 0)

  def reshape_audio_attn_qkv(input_tensor, target_shape):
    \"\"\"Reshape audio attention Q/K/V projection.

    HF: (hidden_size, hidden_size)
    MaxText: (hidden_size, num_heads, head_dim)
    \"\"\"
    if saving_to_hf:
      # MaxText -> HF: (hidden_size, num_heads, head_dim) -> (hidden_size, hidden_size)
      return input_tensor.reshape(hidden_size_audio, hidden_size_audio).T
    else:
      # HF -> MaxText: (hidden_size, hidden_size) -> (hidden_size, num_heads, head_dim)
      return input_tensor.T.reshape(target_shape)

  def reshape_audio_attn_out(input_tensor, target_shape):
    \"\"\"Reshape audio attention output projection.

    HF: (hidden_size, hidden_size)
    MaxText: (num_heads, head_dim, hidden_size)
    \"\"\"
    if saving_to_hf:
      # MaxText -> HF: (num_heads, head_dim, hidden_size) -> (hidden_size, hidden_size)
      return input_tensor.reshape(hidden_size_audio, hidden_size_audio).T
    else:
      # HF -> MaxText: (hidden_size, hidden_size) -> (num_heads, head_dim, hidden_size)
      return input_tensor.T.reshape(target_shape)

  def reshape_audio_attn_qkv_bias(input_tensor, target_shape):
    \"\"\"Reshape audio attention Q/K/V bias.

    HF: (hidden_size,)
    MaxText: (num_heads, head_dim)
    \"\"\"
    if saving_to_hf:
      # MaxText -> HF: (num_heads, head_dim) -> (hidden_size,)
      return input_tensor.reshape(hidden_size_audio)
    else:
      # HF -> MaxText: (hidden_size,) -> (num_heads, head_dim)
      return input_tensor.reshape(target_shape)

  # Audio conv layers
  mapping["params-audio_encoder-Qwen3OmniAudioEncoder_0-conv2d1-kernel"] = reshape_conv2d_audio
  mapping["params-audio_encoder-Qwen3OmniAudioEncoder_0-conv2d2-kernel"] = reshape_conv2d_audio
  mapping["params-audio_encoder-Qwen3OmniAudioEncoder_0-conv2d3-kernel"] = reshape_conv2d_audio

  # Audio conv output projection
  mapping["params-audio_encoder-Qwen3OmniAudioEncoder_0-conv_out-kernel"] = reshape_kernel_audio

  # Audio encoder layers
  for i in range(n_audio_layers):
    prefix = f"params-audio_encoder-Qwen3OmniAudioEncoder_0-layers_{i}"

    # Attention Q/K/V
    mapping[f"{prefix}-self_attention_audio-query-kernel"] = reshape_audio_attn_qkv
    mapping[f"{prefix}-self_attention_audio-query-bias"] = reshape_audio_attn_qkv_bias
    mapping[f"{prefix}-self_attention_audio-key-kernel"] = reshape_audio_attn_qkv
    mapping[f"{prefix}-self_attention_audio-key-bias"] = reshape_audio_attn_qkv_bias
    mapping[f"{prefix}-self_attention_audio-value-kernel"] = reshape_audio_attn_qkv
    mapping[f"{prefix}-self_attention_audio-value-bias"] = reshape_audio_attn_qkv_bias

    # Attention output
    mapping[f"{prefix}-self_attention_audio-out-kernel"] = reshape_audio_attn_out

    # MLP
    mapping[f"{prefix}-AudioMLP-wi-kernel"] = reshape_kernel_audio
    mapping[f"{prefix}-AudioMLP-wo-kernel"] = reshape_kernel_audio

  # Audio projector
  mapping["params-audio_encoder-Qwen3OmniAudioProjector_0-proj1-kernel"] = reshape_kernel_audio
  mapping["params-audio_encoder-Qwen3OmniAudioProjector_0-proj2-kernel"] = reshape_kernel_audio

  return mapping


def QWEN3_NNX_TO_VLLM_PARAM_HOOK_FN(target_shape=None):
  \"\"\"Creates parameter transformation functions for Qwen3.

  This function provides a dictionary of transformation functions (hooks) for
  converting Qwen3 model parameters between NNX and vLLM formats.

  Returns:
    dict: A dictionary mapping NNX parameter names to their corresponding
      transformation functions.
  \"\"\"
  return {}


def LLAMA31_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  \"\"\"
  Returns a dictionary mapping from MaxText parameter names to
  HuggingFace LLaMA3.1 parameter names.

  Args:
      config (dict): Model configuration dictionary containing:
          - num_hidden_layers (int): The number of decoder layers.
      scan_layers (bool, optional): If True, MaxText layers are 'stacked'
          into a single param. Defaults to False.

  Returns:
      dict: A mapping where keys are `atomic_mt_key` (single MaxText parameter names).
        Values are either a single string (unscanned form) or a list of strings
        (scanned form) for stacked layers when `scan_layers=True`.
  \"\"\"
  n_layers = config["num_hidden_layers"]

  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-logits_dense-kernel": "lm_head.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
  }

  if scan_layers:
    mapping["params-decoder-layers-self_attention-query-kernel"] = [
        f"model.layers.{layer_idx}.self_attn.q_proj.weight" for layer_idx in range(n_layers)
    ]
    mapping["params-decoder-layers-self_attention-key-kernel"] = [
        f"model.layers.{layer_idx}.self_attn.k_proj.weight" for layer_idx in range(n_layers)
    ]
    mapping["params-decoder-layers-self_attention-value-kernel"] = [
        f"model.layers.{layer_idx}.self_attn.v_proj.weight" for layer_idx in range(n_layers)
    ]
    mapping["params-decoder-layers-self_attention-out-kernel"] = [
        f"model.layers.{layer_idx}.self_attn.o_proj.weight" for layer_idx in range(n_layers)
    ]
    mapping["params-decoder-layers-mlp-wi_0-kernel"] = [
        f"model.layers.{layer_idx}.mlp.gate_proj.weight" for layer_idx in range(n_layers)
    ]
    mapping["params-decoder-layers-mlp-wi_1-kernel"] = [
        f"model.layers.{layer_idx}.mlp.up_proj.weight" for layer_idx in range(n_layers)
    ]
    mapping["params-decoder-layers-mlp-wo-kernel"] = [
        f"model.layers.{layer_idx}.mlp.down_proj.weight" for layer_idx in range(n_layers)
    ]
    mapping["params-decoder-layers-pre_self_attention_layer_norm-scale"] = [
        f"model.layers.{layer_idx}.input_layernorm.weight" for layer_idx in range(n_layers)
    ]
    mapping["params-decoder-layers-post_self_attention_layer_norm-scale"] = [
        f"model.layers.{layer_idx}.post_attention_layernorm.weight" for layer_idx in range(n_layers)
    ]
  else:
    for layer_idx in range(n_layers):
      mapping[f"params-decoder-layers_{layer_idx}-self_attention-query-kernel"] = (
          f"model.layers.{layer_idx}.self_attn.q_proj.weight"
      )
      mapping[f"params-decoder-layers_{layer_idx}-self_attention-key-kernel"] = (
          f"model.layers.{layer_idx}.self_attn.k_proj.weight"
      )
      mapping[f"params-decoder-layers_{layer_idx}-self_attention-value-kernel"] = (
          f"model.layers.{layer_idx}.self_attn.v_proj.weight"
      )
      mapping[f"params-decoder-layers_{layer_idx}-self_attention-out-kernel"] = (
          f"model.layers.{layer_idx}.self_attn.o_proj.weight"
      )
      mapping[f"params-decoder-layers_{layer_idx}-mlp-wi_0-kernel"] = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
      mapping[f"params-decoder-layers_{layer_idx}-mlp-wi_1-kernel"] = f"model.layers.{layer_idx}.mlp.up_proj.weight"
      mapping[f"params-decoder-layers_{layer_idx}-mlp-wo-kernel"] = f"model.layers.{layer_idx}.mlp.down_proj.weight"
      mapping[f"params-decoder-layers_{layer_idx}-pre_self_attention_layer_norm-scale"] = (
          f"model.layers.{layer_idx}.input_layernorm.weight"
      )
      mapping[f"params-decoder-layers_{layer_idx}-post_self_attention_layer_norm-scale"] = (
          f"model.layers.{layer_idx}.post_attention_layernorm.weight"
      )

  return mapping


def LLAMA31_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  \"\"\"Creates parameter transformation functions for converting between MaxText and
  HuggingFace formats.

  This function generates a mapping of transformation functions that handle the necessary
  conversions between MaxText and HuggingFace parameter formats, including operations like
  reshaping.
  \"\"\"
  nlayers = config["num_hidden_layers"]

  def scale_query_layer(input_tensor, target_shape):
    if saving_to_hf:
      depth_scale = np.dtype("float32").type(np.sqrt(config["head_dim"]))
      original_dtype = input_tensor.dtype
      output_tensor = input_tensor.astype(np.float32) * depth_scale
      return output_tensor.astype(original_dtype)
    else:
      depth_scale = np.dtype("float32").type(1 / np.sqrt(config["head_dim"]))
      original_dtype = input_tensor.dtype
      output_tensor = input_tensor.astype(np.float32) * depth_scale
      return output_tensor.astype(original_dtype)

  def adjust_rope(input_tensor, target_shape):
    arr = input_tensor
    if saving_to_hf:
      # Convert from MaxText's interleaved layout to HF's concatenated layout
      evens = arr[..., ::2]
      odds = arr[..., 1::2]
      return jax.numpy.concatenate((evens, odds), axis=arr.ndim - 1)
    else:
      # Convert from HF's concatenated layout to MaxText's interleaved layout
      half_dim = arr.shape[-1] // 2
      first_half = arr[..., :half_dim]
      second_half = arr[..., half_dim:]
      return jax.numpy.stack([first_half, second_half], axis=-1).reshape(arr.shape)

  def reshape_kernel(input_tensor, target_shape):
    if saving_to_hf:
      flipped_target_shape = np.flip(np.array(target_shape))
      return input_tensor.reshape(flipped_target_shape).transpose()
    else:
      return input_tensor.transpose().reshape(target_shape)

  # caveat: hook order does affect result
  # to_huggingface
  query_hook_chain = [scale_query_layer, adjust_rope, reshape_kernel]
  key_hook_chain = [adjust_rope, reshape_kernel]
  # to_maxtext
  if not saving_to_hf:
    query_hook_chain.reverse()
    key_hook_chain.reverse()

  hook_fns = {}

  hook_fns["params-decoder-logits_dense-kernel"] = reshape_kernel

  if scan_layers:
    hook_fns = {
        **hook_fns,
        "params-decoder-layers-self_attention-query-kernel": query_hook_chain,
        "params-decoder-layers-self_attention-key-kernel": key_hook_chain,
        "params-decoder-layers-self_attention-value-kernel": reshape_kernel,
        "params-decoder-layers-self_attention-out-kernel": reshape_kernel,
        "params-decoder-layers-mlp-wi_0-kernel": reshape_kernel,
        "params-decoder-layers-mlp-wi_1-kernel": reshape_kernel,
        "params-decoder-layers-mlp-wo-kernel": reshape_kernel,
    }
  else:
    for layer_idx in range(nlayers):
      hook_fns[f"params-decoder-layers_{layer_idx}-self_attention-query-kernel"] = query_hook_chain
      hook_fns[f"params-decoder-layers_{layer_idx}-self_attention-key-kernel"] = key_hook_chain
      hook_fns[f"params-decoder-layers_{layer_idx}-self_attention-value-kernel"] = reshape_kernel
      hook_fns[f"params-decoder-layers_{layer_idx}-self_attention-out-kernel"] = reshape_kernel
      hook_fns[f"params-decoder-layers_{layer_idx}-mlp-wi_0-kernel"] = reshape_kernel
      hook_fns[f"params-decoder-layers_{layer_idx}-mlp-wi_1-kernel"] = reshape_kernel
      hook_fns[f"params-decoder-layers_{layer_idx}-mlp-wo-kernel"] = reshape_kernel
  return hook_fns


def LLAMA31_NNX_TO_VLLM_PARAM_HOOK_FN():
  \"\"\"Defines and returns hook functions for weight transformations.

  These hooks are applied to specific weights during the conversion
  from MaxText to a HuggingFace-compatible format. They handle
  transformations like RoPE reordering and query scaling that are not
  simple re-mappings.

  Returns:
    A dictionary where keys are MaxText parameter names and values are
    the corresponding transformation functions.
  \"\"\"

  def reorder_rope(arr):
    \"\"\"Reorders Rotary Position Embedding (RoPE) weights.

    This function is necessary because MaxText and HuggingFace's vLLM
    implementations may have different orderings for RoPE dimensions.
    It splits the last dimension into even and odd indices and
    concatenates them.

    Args:
      arr: The input weight array.

    Returns:
      The reordered weight array.
    \"\"\"
    evens = arr[..., ::2]
    odds = arr[..., 1::2]
    return jax.numpy.concatenate((evens, odds), axis=arr.ndim - 1)

  def transform_query_kernel(arr):
    \"\"\"Transforms the query kernel.

    This involves scaling the kernel by the square root of the head
    dimension and then applying RoPE reordering.

    Args:
      arr: The query kernel weight array.

    Returns:
      The transformed query kernel array.
    \"\"\"
    head_dim = arr.shape[-1]
    depth_scale = np.dtype("float32").type(np.sqrt(head_dim))
    arr = arr * depth_scale
    return reorder_rope(arr)

  hook_fns = {
      "base.decoder.layers.self_attention.query.kernel": transform_query_kernel,
      "base.decoder.layers.self_attention.key.kernel": reorder_rope,
  }
  return hook_fns


def MIXTRAL_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  \"\"\"
  Generates the mapping of parameter names from MaxText to Hugging Face for Mixtral.

  Returns:
    dict: A mapping where keys are `atomic_mt_key` (single MaxText parameter names). Values
      are Hugging Face parameter names in one of four forms: unscanned string,
      scanned list of strings, unscanned with expert stacking (list of strings),
      or scanned with expert stacking (nested list of strings).
  \"\"\"
  mapping = {}

  # Top-level, non-layer-specific parameters
  mapping["params-token_embedder-embedding"] = "model.embed_tokens.weight"
  mapping["params-decoder-decoder_norm-scale"] = "model.norm.weight"
  mapping["params-decoder-logits_dense-kernel"] = "lm_head.weight"

  num_experts = maxtext_config.num_experts

  if scan_layers:
    # Initialize lists for scanned layer weights
    mapping.update(
        {
            "params-decoder-layers-self_attention-query-kernel": [],
            "params-decoder-layers-self_attention-key-kernel": [],
            "params-decoder-layers-self_attention-value-kernel": [],
            "params-decoder-layers-self_attention-out-kernel": [],
            "params-decoder-layers-pre_self_attention_layer_norm-scale": [],
            "params-decoder-layers-post_self_attention_layer_norm-scale": [],
            "params-decoder-layers-MoeBlock_0-gate-kernel": [],
            "params-decoder-layers-MoeBlock_0-wi_0": [],
            "params-decoder-layers-MoeBlock_0-wi_1": [],
            "params-decoder-layers-MoeBlock_0-wo": [],
        }
    )

    for i in range(config["num_hidden_layers"]):
      hf_prefix = f"model.layers.{i}"
      # Attention weights
      mapping["params-decoder-layers-self_attention-query-kernel"].append(f"{hf_prefix}.self_attn.q_proj.weight")
      mapping["params-decoder-layers-self_attention-key-kernel"].append(f"{hf_prefix}.self_attn.k_proj.weight")
      mapping["params-decoder-layers-self_attention-value-kernel"].append(f"{hf_prefix}.self_attn.v_proj.weight")
      mapping["params-decoder-layers-self_attention-out-kernel"].append(f"{hf_prefix}.self_attn.o_proj.weight")

      # RMSNorm weights
      mapping["params-decoder-layers-pre_self_attention_layer_norm-scale"].append(f"{hf_prefix}.input_layernorm.weight")
      mapping["params-decoder-layers-post_self_attention_layer_norm-scale"].append(
          f"{hf_prefix}.post_attention_layernorm.weight"
      )

      # MoE gate
      mapping["params-decoder-layers-MoeBlock_0-gate-kernel"].append(f"{hf_prefix}.block_sparse_moe.gate.weight")

    # Outer loop as experts and inner loop as layers to align with logic in _build_multi_axis_stacked_tensor()
    for j in range(num_experts):
      w1_layers = []
      w3_layers = []
      w2_layers = []

      for i in range(config["num_hidden_layers"]):
        hf_prefix = f"model.layers.{i}"
        w1_layers.append(f"{hf_prefix}.block_sparse_moe.experts.{j}.w1.weight")
        w3_layers.append(f"{hf_prefix}.block_sparse_moe.experts.{j}.w3.weight")
        w2_layers.append(f"{hf_prefix}.block_sparse_moe.experts.{j}.w2.weight")

      mapping["params-decoder-layers-MoeBlock_0-wi_0"].append(w1_layers)
      mapping["params-decoder-layers-MoeBlock_0-wi_1"].append(w3_layers)
      mapping["params-decoder-layers-MoeBlock_0-wo"].append(w2_layers)

  else:
    for i in range(config["num_hidden_layers"]):
      maxtext_prefix = f"params-decoder-layers_{i}"
      hf_prefix = f"model.layers.{i}"

      # Attention weights
      mapping[f"{maxtext_prefix}-self_attention-query-kernel"] = f"{hf_prefix}.self_attn.q_proj.weight"
      mapping[f"{maxtext_prefix}-self_attention-key-kernel"] = f"{hf_prefix}.self_attn.k_proj.weight"
      mapping[f"{maxtext_prefix}-self_attention-value-kernel"] = f"{hf_prefix}.self_attn.v_proj.weight"
      mapping[f"{maxtext_prefix}-self_attention-out-kernel"] = f"{hf_prefix}.self_attn.o_proj.weight"

      # RMSNorm weights
      mapping[f"{maxtext_prefix}-pre_self_attention_layer_norm-scale"] = f"{hf_prefix}.input_layernorm.weight"
      mapping[f"{maxtext_prefix}-post_self_attention_layer_norm-scale"] = f"{hf_prefix}.post_attention_layernorm.weight"

      # MoE gate
      mapping[f"{maxtext_prefix}-MoeBlock_0-gate-kernel"] = f"{hf_prefix}.block_sparse_moe.gate.weight"

      # MoE expert weights (1 MaxText param -> 8 HF params)
      w1_experts = [f"{hf_prefix}.block_sparse_moe.experts.{j}.w1.weight" for j in range(num_experts)]
      w3_experts = [f"{hf_prefix}.block_sparse_moe.experts.{j}.w3.weight" for j in range(num_experts)]
      w2_experts = [f"{hf_prefix}.block_sparse_moe.experts.{j}.w2.weight" for j in range(num_experts)]

      mapping[f"{maxtext_prefix}-MoeBlock_0-wi_0"] = w1_experts
      mapping[f"{maxtext_prefix}-MoeBlock_0-wi_1"] = w3_experts
      mapping[f"{maxtext_prefix}-MoeBlock_0-wo"] = w2_experts

  return mapping


def MIXTRAL_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  \"\"\"
  Generates parameter conversion hooks for Mixtral between MaxText and Hugging Face.
  \"\"\"
  hooks = {}

  def reshape_and_transpose_attention(x, target_shape):
    \"\"\"MaxText: [hidden, n_heads, h_dim] <-> HF: [n_heads * h_dim, hidden]\"\"\"
    if saving_to_hf:
      # (H, N, D) -> (H, N*D) -> (N*D, H)
      return x.reshape(config["hidden_size"], -1).transpose()
    else:
      # (N*D, H) -> (H, N*D) -> (H, N, D)
      return x.transpose().reshape(target_shape)

  def reshape_kernel(x, target_shape):
    return x.transpose()

  def scale_query_layer(input_tensor, target_shape):
    if saving_to_hf:
      depth_scale = np.dtype("float32").type(np.sqrt(maxtext_config.head_dim))
      return (input_tensor * depth_scale).astype(input_tensor.dtype)
    else:
      depth_scale = np.dtype("float32").type(1 / np.sqrt(maxtext_config.head_dim))
      return (input_tensor * depth_scale).astype(input_tensor.dtype)

  # hook order does not affect result
  query_hook_chain = [reshape_and_transpose_attention, scale_query_layer]

  if scan_layers:
    plan = [
        ("params-decoder-layers-self_attention-query-kernel", query_hook_chain),
        ("params-decoder-layers-self_attention-key-kernel", reshape_and_transpose_attention),
        ("params-decoder-layers-self_attention-value-kernel", reshape_and_transpose_attention),
        ("params-decoder-layers-self_attention-out-kernel", reshape_and_transpose_attention),
        ("params-decoder-layers-MoeBlock_0-wi_0", reshape_kernel),
        ("params-decoder-layers-MoeBlock_0-wi_1", reshape_kernel),
        ("params-decoder-layers-MoeBlock_0-wo", reshape_kernel),
        ("params-decoder-layers-MoeBlock_0-gate-kernel", reshape_kernel),
    ]
  else:
    plan = [
        ("params-decoder-layers_{i}-self_attention-query-kernel", query_hook_chain),
        ("params-decoder-layers_{i}-self_attention-key-kernel", reshape_and_transpose_attention),
        ("params-decoder-layers_{i}-self_attention-value-kernel", reshape_and_transpose_attention),
        ("params-decoder-layers_{i}-self_attention-out-kernel", reshape_and_transpose_attention),
        ("params-decoder-layers_{i}-MoeBlock_0-wi_0", reshape_kernel),
        ("params-decoder-layers_{i}-MoeBlock_0-wi_1", reshape_kernel),
        ("params-decoder-layers_{i}-MoeBlock_0-wo", reshape_kernel),
        ("params-decoder-layers_{i}-MoeBlock_0-gate-kernel", reshape_kernel),
    ]
  plan.append(("params-decoder-logits_dense-kernel", reshape_kernel))

  for maxtext_pattern, op_func in plan:
    if "{i}" in maxtext_pattern:
      for i in range(config["num_hidden_layers"]):
        hooks[maxtext_pattern.format(i=i)] = op_func
    else:
      hooks[maxtext_pattern] = op_func
  return hooks


def GEMMA4_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  \"\"\"Returns mapping between MaxText and HuggingFace Gemma4 weight paths.\"\"\"
  tcfg = config.get("text_config", config)
  nlayers = tcfg["num_hidden_layers"]
  share_kv_projections = maxtext_config.share_kv_projections
  # Gemma 4 uses a block pattern of length 6: 5 local, 1 global
  vcfg = config.get("vision_config", {})
  text_base = "model.language_model" if vcfg else "model"
  mapping = {
      "params-token_embedder-embedding": f"{text_base}.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": f"{text_base}.norm.weight",
  }
  if maxtext_config.use_multimodal and vcfg:
    nvis = vcfg.get("num_hidden_layers", 0)
    mapping.update(
        {
            "params-vision_encoder-Gemma4VisionEncoderLayer_0-vision_entry-input_projection-kernel": (
                "model.vision_tower.patch_embedder.input_proj.weight"
            ),
            "params-vision_encoder-Gemma4VisionEncoderLayer_0-vision_entry-pos_emb_param": (
                "model.vision_tower.patch_embedder.position_embedding_table"
            ),
            "params-vision_encoder-Gemma4VisionProjector_0-projection-kernel": (
                "model.embed_vision.embedding_projection.weight"
            ),
            "params-vision_encoder-Gemma4VisionEncoderLayer_0-std_scale": ("model.vision_tower.std_scale"),
            "params-vision_encoder-Gemma4VisionEncoderLayer_0-std_bias": ("model.vision_tower.std_bias"),
        }
    )
    for i in range(nvis):
      prefix = f"params-vision_encoder-Gemma4VisionEncoderLayer_0-layer_{i}"
      hf_prefix = f"model.vision_tower.encoder.layers.{i}"
      mapping.update(
          {
              f"{prefix}-attention-query-kernel": f"{hf_prefix}.self_attn.q_proj.linear.weight",
              f"{prefix}-attention-key-kernel": f"{hf_prefix}.self_attn.k_proj.linear.weight",
              f"{prefix}-attention-value-kernel": f"{hf_prefix}.self_attn.v_proj.linear.weight",
              f"{prefix}-attention-out-kernel": f"{hf_prefix}.self_attn.o_proj.linear.weight",
              f"{prefix}-attention-query_norm-scale": f"{hf_prefix}.self_attn.q_norm.weight",
              f"{prefix}-attention-key_norm-scale": f"{hf_prefix}.self_attn.k_norm.weight",
              f"{prefix}-pre_attention_norm-scale": f"{hf_prefix}.input_layernorm.weight",
              f"{prefix}-post_attention_norm-scale": f"{hf_prefix}.post_attention_layernorm.weight",
              f"{prefix}-pre_ffw_norm-scale": f"{hf_prefix}.pre_feedforward_layernorm.weight",
              f"{prefix}-post_ffw_norm-scale": f"{hf_prefix}.post_feedforward_layernorm.weight",
              f"{prefix}-mlp-wi_0-kernel": f"{hf_prefix}.mlp.gate_proj.linear.weight",
              f"{prefix}-mlp-wi_1-kernel": f"{hf_prefix}.mlp.up_proj.linear.weight",
              f"{prefix}-mlp-wo-kernel": f"{hf_prefix}.mlp.down_proj.linear.weight",
          }
      )
  if scan_layers:
    attention_pattern_length = 6
    num_remaining = nlayers % attention_pattern_length
    num_scanned = nlayers - num_remaining
    num_experts = tcfg.get("num_experts")
    num_experts = num_experts if num_experts is not None else 1

    # Main scanned blocks
    for layer_in_block in range(attention_pattern_length):
      hf_indices = list(range(layer_in_block, num_scanned, attention_pattern_length))
      prefix = f"params-decoder-scanned_blocks-layers_{layer_in_block}"
      mapping.update(
          {
              f"{prefix}-self_attention-query-kernel": [
                  f"{text_base}.layers.{i}.self_attn.q_proj.weight" for i in hf_indices
              ],
              f"{prefix}-self_attention-key-kernel": [
                  f"{text_base}.layers.{i}.self_attn.k_proj.weight" for i in hf_indices
              ],
              f"{prefix}-self_attention-value-kernel": (
                  None
                  if share_kv_projections and layer_in_block == 5
                  else [f"{text_base}.layers.{i}.self_attn.v_proj.weight" for i in hf_indices]
              ),
              f"{prefix}-self_attention-out-kernel": [
                  f"{text_base}.layers.{i}.self_attn.o_proj.weight" for i in hf_indices
              ],
              f"{prefix}-self_attention-query_norm-scale": [
                  f"{text_base}.layers.{i}.self_attn.q_norm.weight" for i in hf_indices
              ],
              f"{prefix}-self_attention-key_norm-scale": [
                  f"{text_base}.layers.{i}.self_attn.k_norm.weight" for i in hf_indices
              ],
          }
      )
      if maxtext_config.v_norm_with_scale:
        mapping.update(
            {
                f"{prefix}-self_attention-value_norm-scale": [
                    f"{text_base}.layers.{i}.self_attn.v_norm.weight" for i in hf_indices
                ]
            }
        )
      mapping.update(
          {
              f"{prefix}-pre_self_attention_norm-scale": [
                  f"{text_base}.layers.{i}.input_layernorm.weight" for i in hf_indices
              ],
              f"{prefix}-post_self_attention_norm-scale": [
                  f"{text_base}.layers.{i}.post_attention_layernorm.weight" for i in hf_indices
              ],
              f"{prefix}-pre_ffw_norm-scale": [
                  f"{text_base}.layers.{i}.pre_feedforward_layernorm.weight" for i in hf_indices
              ],
              f"{prefix}-post_ffw_norm-scale": [
                  f"{text_base}.layers.{i}.post_feedforward_layernorm.weight" for i in hf_indices
              ],
              f"{prefix}-mlp-pre_feedforward_layernorm_2-scale": [
                  f"{text_base}.layers.{i}.pre_feedforward_layernorm_2.weight" if num_experts > 1 else None
                  for i in hf_indices
              ],
              f"{prefix}-mlp-post_feedforward_layernorm_1-scale": [
                  f"{text_base}.layers.{i}.post_feedforward_layernorm_1.weight" if num_experts > 1 else None
                  for i in hf_indices
              ],
              f"{prefix}-mlp-post_feedforward_layernorm_2-scale": [
                  f"{text_base}.layers.{i}.post_feedforward_layernorm_2.weight" if num_experts > 1 else None
                  for i in hf_indices
              ],
              f"{prefix}-mlp-pre_forward_scale_2": [
                  f"{text_base}.layers.{i}.router.scale" if num_experts > 1 else None for i in hf_indices
              ],
              f"{prefix}-mlp-wi_0-kernel": [
                  f"{text_base}.layers.{i}.mlp.gate_proj.weight" if num_experts == 1 else None for i in hf_indices
              ],
              f"{prefix}-mlp-wi_1-kernel": [
                  f"{text_base}.layers.{i}.mlp.up_proj.weight" if num_experts == 1 else None for i in hf_indices
              ],
              f"{prefix}-mlp-wo-kernel": [
                  f"{text_base}.layers.{i}.mlp.down_proj.weight" if num_experts == 1 else None for i in hf_indices
              ],
              f"{prefix}-mlp-moe_block-MoeBlock_0-gate-kernel": [
                  f"{text_base}.layers.{i}.router.proj.weight" if num_experts > 1 else None for i in hf_indices
              ],
              f"{prefix}-mlp-moe_block-MoeBlock_0-wi_0": [
                  f"{text_base}.layers.{i}.experts.gate_up_proj" if num_experts > 1 else None for i in hf_indices
              ],
              f"{prefix}-mlp-moe_block-MoeBlock_0-wi_1": [
                  f"{text_base}.layers.{i}.experts.gate_up_proj" if num_experts > 1 else None for i in hf_indices
              ],
              f"{prefix}-mlp-moe_block-MoeBlock_0-wo": [
                  f"{text_base}.layers.{i}.experts.down_proj" if num_experts > 1 else None for i in hf_indices
              ],
              f"{prefix}-mlp-moe_block-MoeBlock_0-per_expert_scale": [
                  f"{text_base}.layers.{i}.router.per_expert_scale" if num_experts > 1 else None for i in hf_indices
              ],
              f"{prefix}-mlp-moe_block-shared_experts-wi_0-kernel": [
                  f"{text_base}.layers.{i}.mlp.gate_proj.weight" if num_experts > 1 else None for i in hf_indices
              ],
              f"{prefix}-mlp-moe_block-shared_experts-wi_1-kernel": [
                  f"{text_base}.layers.{i}.mlp.up_proj.weight" if num_experts > 1 else None for i in hf_indices
              ],
              f"{prefix}-mlp-moe_block-shared_experts-wo-kernel": [
                  f"{text_base}.layers.{i}.mlp.down_proj.weight" if num_experts > 1 else None for i in hf_indices
              ],
              f"{prefix}-layer_scalar": [f"{text_base}.layers.{i}.layer_scalar" for i in hf_indices],
          }
      )
      mapping = {
          k: v
          for k, v in mapping.items()
          if (isinstance(v, list) and len(v) > 0 and v[0] is not None) or (not isinstance(v, list) and v is not None)
      }

    # Remainder layers
    if num_remaining > 0:
      for rem_idx in range(num_remaining):
        hf_layer_idx = num_scanned + rem_idx
        # Remaining layers use local attention type logic
        is_global = False  # For gemma 4 it is unlikely the remainder is global but safe to determine
        layer_in_block = hf_layer_idx % 6
        is_global = layer_in_block == 5

        prefix = f"params-decoder-layers_remainder-layers_{rem_idx}"
        hf_prefix = f"{text_base}.layers.{hf_layer_idx}"
        mapping.update(
            {
                f"{prefix}-self_attention-query-kernel": (f"{hf_prefix}.self_attn.q_proj.weight"),
                f"{prefix}-self_attention-key-kernel": (f"{hf_prefix}.self_attn.k_proj.weight"),
                f"{prefix}-self_attention-value-kernel": (
                    f"{hf_prefix}.self_attn.k_proj.weight"
                    if share_kv_projections and is_global
                    else f"{hf_prefix}.self_attn.v_proj.weight"
                ),
                f"{prefix}-self_attention-out-kernel": (f"{hf_prefix}.self_attn.o_proj.weight"),
                f"{prefix}-self_attention-query_norm-scale": (f"{hf_prefix}.self_attn.q_norm.weight"),
                f"{prefix}-self_attention-key_norm-scale": (f"{hf_prefix}.self_attn.k_norm.weight"),
            }
        )
        if maxtext_config.v_norm_with_scale:
          mapping.update({f"{prefix}-self_attention-value_norm-scale": (f"{hf_prefix}.self_attn.v_norm.weight")})
        mapping.update(
            {
                f"{prefix}-pre_self_attention_norm-scale": (f"{hf_prefix}.input_layernorm.weight"),
                f"{prefix}-post_self_attention_norm-scale": (f"{hf_prefix}.post_attention_layernorm.weight"),
                f"{prefix}-pre_ffw_norm-scale": (f"{hf_prefix}.pre_feedforward_layernorm.weight"),
                f"{prefix}-post_ffw_norm-scale": (f"{hf_prefix}.post_feedforward_layernorm.weight"),
                f"{prefix}-mlp-pre_feedforward_layernorm_2-scale": (
                    f"{hf_prefix}.pre_feedforward_layernorm_2.weight" if num_experts > 1 else None
                ),
                f"{prefix}-mlp-post_feedforward_layernorm_1-scale": (
                    f"{hf_prefix}.post_feedforward_layernorm_1.weight" if num_experts > 1 else None
                ),
                f"{prefix}-mlp-post_feedforward_layernorm_2-scale": (
                    f"{hf_prefix}.post_feedforward_layernorm_2.weight" if num_experts > 1 else None
                ),
                f"{prefix}-mlp-pre_forward_scale_2": (f"{hf_prefix}.router.scale" if num_experts > 1 else None),
                f"{prefix}-mlp-wi_0-kernel": f"{hf_prefix}.mlp.gate_proj.weight" if num_experts == 1 else None,
                f"{prefix}-mlp-wi_1-kernel": f"{hf_prefix}.mlp.up_proj.weight" if num_experts == 1 else None,
                f"{prefix}-mlp-wo-kernel": f"{hf_prefix}.mlp.down_proj.weight" if num_experts == 1 else None,
                f"{prefix}-mlp-moe_block-MoeBlock_0-gate-kernel": f"{hf_prefix}.router.proj.weight"
                if num_experts > 1
                else None,
                f"{prefix}-mlp-moe_block-MoeBlock_0-wi_0": f"{hf_prefix}.experts.gate_up_proj"
                if num_experts > 1
                else None,
                f"{prefix}-mlp-moe_block-MoeBlock_0-wi_1": f"{hf_prefix}.experts.gate_up_proj"
                if num_experts > 1
                else None,
                f"{prefix}-mlp-moe_block-MoeBlock_0-wo": f"{hf_prefix}.experts.down_proj" if num_experts > 1 else None,
                f"{prefix}-mlp-moe_block-MoeBlock_0-per_expert_scale": f"{hf_prefix}.router.per_expert_scale"
                if num_experts > 1
                else None,
                f"{prefix}-mlp-moe_block-shared_experts-wi_0-kernel": f"{hf_prefix}.mlp.gate_proj.weight"
                if num_experts > 1
                else None,
                f"{prefix}-mlp-moe_block-shared_experts-wi_1-kernel": f"{hf_prefix}.mlp.up_proj.weight"
                if num_experts > 1
                else None,
                f"{prefix}-mlp-moe_block-shared_experts-wo-kernel": f"{hf_prefix}.mlp.down_proj.weight"
                if num_experts > 1
                else None,
                f"{prefix}-layer_scalar": f"{hf_prefix}.layer_scalar",
            }
        )
      mapping = {k: v for k, v in mapping.items() if v is not None}
  else:
    for i in range(nlayers):
      prefix = f"params-decoder-layers_{i}"
      hf_prefix = f"{text_base}.layers.{i}"
      is_global = i % 6 == 5
      num_experts = tcfg.get("num_experts")
      num_experts = num_experts if num_experts is not None else 1
      mapping.update(
          {
              f"{prefix}-self_attention-query-kernel": (f"{hf_prefix}.self_attn.q_proj.weight"),
              f"{prefix}-self_attention-key-kernel": (f"{hf_prefix}.self_attn.k_proj.weight"),
              f"{prefix}-self_attention-value-kernel": (
                  None if share_kv_projections and is_global else f"{hf_prefix}.self_attn.v_proj.weight"
              ),
              f"{prefix}-self_attention-out-kernel": (f"{hf_prefix}.self_attn.o_proj.weight"),
              f"{prefix}-self_attention-query_norm-scale": (f"{hf_prefix}.self_attn.q_norm.weight"),
              f"{prefix}-self_attention-key_norm-scale": (f"{hf_prefix}.self_attn.k_norm.weight"),
          }
      )
      if maxtext_config.v_norm_with_scale:
        mapping.update({f"{prefix}-self_attention-value_norm-scale": (f"{hf_prefix}.self_attn.v_norm.weight")})
      mapping.update(
          {
              f"{prefix}-pre_self_attention_norm-scale": (f"{hf_prefix}.input_layernorm.weight"),
              f"{prefix}-post_self_attention_norm-scale": (f"{hf_prefix}.post_attention_layernorm.weight"),
              f"{prefix}-pre_ffw_norm-scale": (f"{hf_prefix}.pre_feedforward_layernorm.weight"),
              f"{prefix}-post_ffw_norm-scale": (f"{hf_prefix}.post_feedforward_layernorm.weight"),
              f"{prefix}-mlp-pre_feedforward_layernorm_2-scale": (
                  f"{hf_prefix}.pre_feedforward_layernorm_2.weight" if num_experts > 1 else None
              ),
              f"{prefix}-mlp-post_feedforward_layernorm_1-scale": (
                  f"{hf_prefix}.post_feedforward_layernorm_1.weight" if num_experts > 1 else None
              ),
              f"{prefix}-mlp-post_feedforward_layernorm_2-scale": (
                  f"{hf_prefix}.post_feedforward_layernorm_2.weight" if num_experts > 1 else None
              ),
              f"{prefix}-mlp-pre_forward_scale_2": (f"{hf_prefix}.router.scale" if num_experts > 1 else None),
              f"{prefix}-mlp-wi_0-kernel": f"{hf_prefix}.mlp.gate_proj.weight" if num_experts == 1 else None,
              f"{prefix}-mlp-wi_1-kernel": f"{hf_prefix}.mlp.up_proj.weight" if num_experts == 1 else None,
              f"{prefix}-mlp-wo-kernel": f"{hf_prefix}.mlp.down_proj.weight" if num_experts == 1 else None,
              f"{prefix}-mlp-moe_block-MoeBlock_0-gate-kernel": f"{hf_prefix}.router.proj.weight"
              if num_experts > 1
              else None,
              f"{prefix}-mlp-moe_block-MoeBlock_0-wi_0": f"{hf_prefix}.experts.gate_up_proj" if num_experts > 1 else None,
              f"{prefix}-mlp-moe_block-MoeBlock_0-wi_1": f"{hf_prefix}.experts.gate_up_proj" if num_experts > 1 else None,
              f"{prefix}-mlp-moe_block-MoeBlock_0-wo": f"{hf_prefix}.experts.down_proj" if num_experts > 1 else None,
              f"{prefix}-mlp-moe_block-MoeBlock_0-per_expert_scale": f"{hf_prefix}.router.per_expert_scale"
              if num_experts > 1
              else None,
              f"{prefix}-mlp-moe_block-shared_experts-wi_0-kernel": f"{hf_prefix}.mlp.gate_proj.weight"
              if num_experts > 1
              else None,
              f"{prefix}-mlp-moe_block-shared_experts-wi_1-kernel": f"{hf_prefix}.mlp.up_proj.weight"
              if num_experts > 1
              else None,
              f"{prefix}-mlp-moe_block-shared_experts-wo-kernel": f"{hf_prefix}.mlp.down_proj.weight"
              if num_experts > 1
              else None,
              f"{prefix}-layer_scalar": f"{hf_prefix}.layer_scalar",
          }
      )
  mapping = {k: v for k, v in mapping.items() if v is not None}
  return mapping


def GEMMA4_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  \"\"\"Creates parameter transformation functions for Gemma4.\"\"\"
  tcfg = config.get("text_config", config)
  nlayers = tcfg["num_hidden_layers"]
  share_kv_projections = maxtext_config.share_kv_projections
  hooks = {}

  def pad_hf_embedding_layer(input_tensor, target_shape):
    normalizer = np.dtype("float32").type(tcfg["hidden_size"] ** 0.5)
    if saving_to_hf:
      target_tensor = input_tensor[: target_shape[0], : target_shape[1]]
      target_tensor = target_tensor / normalizer
      return target_tensor.astype(input_tensor.dtype)
    else:
      target_tensor = np.zeros(target_shape, dtype=input_tensor.dtype)
      target_tensor[: input_tensor.shape[0], : input_tensor.shape[1]] = input_tensor
      target_tensor = target_tensor * normalizer
      return target_tensor.astype(input_tensor.dtype)

  def reshape_kernel(input_tensor, target_shape):
    if saving_to_hf:
      flipped_target_shape = np.flip(np.array(target_shape))
      return input_tensor.reshape(flipped_target_shape).T
    else:
      return input_tensor.T.reshape(target_shape)

  def scale_rmsnorm_layer(input_tensor, target_shape):
    # Shift of 1.0 is now folded into Gemma 4 text and vision checkpoint weights
    return input_tensor.reshape(target_shape)

  def split_moe_wi0(input_tensor, target_shape):
    if saving_to_hf:
      raise NotImplementedError("Saving to HF for fused gate_up_proj requires custom concat hook.")
    # input_tensor: [E, 2*FF, H], target: [E, H, FF]
    _, two_FF, _ = input_tensor.shape
    FF = two_FF // 2
    return input_tensor[:, :FF, :].transpose(0, 2, 1)

  def split_moe_wi1(input_tensor, target_shape):
    if saving_to_hf:
      raise NotImplementedError("Saving to HF for fused gate_up_proj requires custom concat hook.")
    _, two_FF, _ = input_tensor.shape
    FF = two_FF // 2
    return input_tensor[:, FF:, :].transpose(0, 2, 1)

  def reshape_moe_wo(input_tensor, target_shape):
    # input_tensor: [E, H, FF], target: [E, FF, H]
    return input_tensor.transpose(0, 2, 1)

  hooks["params-token_embedder-embedding"] = pad_hf_embedding_layer
  hooks["params-decoder-decoder_norm-scale"] = scale_rmsnorm_layer
  # REMOVED: logits_dense-kernel hook (handled by logits_via_embedding: True)

  kernel_keys = [
      "self_attention-query-kernel",
      "self_attention-key-kernel",
      "self_attention-value-kernel",
      "self_attention-out-kernel",
      "mlp-wi_0-kernel",
      "mlp-wi_1-kernel",
      "mlp-wo-kernel",
      "mlp-moe_block-shared_experts-wi_0-kernel",
      "mlp-moe_block-shared_experts-wi_1-kernel",
      "mlp-moe_block-shared_experts-wo-kernel",
  ]
  moe_kernel_keys = [
      # `gate-kernel` (router) has shape [feature, num_experts] in MaxText, but [num_experts, feature] in HF
      "mlp-moe_block-MoeBlock_0-gate-kernel",
  ]

  norm_keys = [
      "self_attention-query_norm-scale",
      "self_attention-key_norm-scale",
  ]
  if maxtext_config.v_norm_with_scale:
    norm_keys.append("self_attention-value_norm-scale")

  norm_keys.extend(
      [
          "pre_self_attention_norm-scale",
          "post_self_attention_norm-scale",
          "pre_ffw_norm-scale",
          "post_ffw_norm-scale",
      ]
  )

  num_experts = tcfg.get("num_experts")
  num_experts = num_experts if num_experts is not None else 1
  if num_experts > 1:
    norm_keys.extend(
        [
            "mlp-pre_feedforward_layernorm_2-scale",
            "mlp-post_feedforward_layernorm_1-scale",
            "mlp-post_feedforward_layernorm_2-scale",
        ]
    )

  # Note: `pre_forward_scale_2`, `per_expert_scale`, and `layer_scalar`
  # are standard tensors being multiplied, not typical RMSNorms. They
  # do not need the `scale_rmsnorm_layer` hook, so leaving them out
  # of norm_keys means they perfectly default to the identity mapping.

  vcfg = config.get("vision_config", {})
  if maxtext_config.use_multimodal and vcfg:
    nvis = vcfg.get("num_hidden_layers", 0)

    def reshape_vision_patch(x, target_shape):
      # HF and MaxText both use (H, W, C) patch flattening now.
      # Just transpose between out_features/in_features.
      return x.T

    def reshape_pos_emb(x, target_shape):
      return x.transpose(1, 0, 2)

    hooks["params-vision_encoder-Gemma4VisionEncoderLayer_0-vision_entry-input_projection-kernel"] = reshape_vision_patch
    hooks["params-vision_encoder-Gemma4VisionEncoderLayer_0-vision_entry-pos_emb_param"] = reshape_pos_emb
    hooks["params-vision_encoder-Gemma4VisionProjector_0-projection-kernel"] = reshape_kernel

    for i in range(nvis):
      prefix = f"params-vision_encoder-Gemma4VisionEncoderLayer_0-layer_{i}"
      hooks[f"{prefix}-attention-query-kernel"] = reshape_kernel
      hooks[f"{prefix}-attention-key-kernel"] = reshape_kernel
      hooks[f"{prefix}-attention-value-kernel"] = reshape_kernel
      hooks[f"{prefix}-attention-out-kernel"] = reshape_kernel
      hooks[f"{prefix}-attention-query_norm-scale"] = scale_rmsnorm_layer
      hooks[f"{prefix}-attention-key_norm-scale"] = scale_rmsnorm_layer
      hooks[f"{prefix}-pre_attention_norm-scale"] = scale_rmsnorm_layer
      hooks[f"{prefix}-post_attention_norm-scale"] = scale_rmsnorm_layer
      hooks[f"{prefix}-pre_ffw_norm-scale"] = scale_rmsnorm_layer
      hooks[f"{prefix}-post_ffw_norm-scale"] = scale_rmsnorm_layer
      hooks[f"{prefix}-mlp-wi_0-kernel"] = reshape_kernel
      hooks[f"{prefix}-mlp-wi_1-kernel"] = reshape_kernel
      hooks[f"{prefix}-mlp-wo-kernel"] = reshape_kernel

  if scan_layers:
    attention_pattern_length = 6
    num_remaining = nlayers % attention_pattern_length

    # Scanned sub-layer prefixes
    for layer_in_block in range(attention_pattern_length):
      is_global = layer_in_block % 6 == 5
      prefix = f"params-decoder-scanned_blocks-layers_{layer_in_block}"
      for key in kernel_keys:
        if share_kv_projections and is_global and key == "self_attention-value-kernel":
          continue
        hooks[f"{prefix}-{key}"] = reshape_kernel
      for key in moe_kernel_keys:
        hooks[f"{prefix}-{key}"] = reshape_kernel
      for key in norm_keys:
        hooks[f"{prefix}-{key}"] = scale_rmsnorm_layer

      # Add these specialized 3D tensor hooks inside the loop
      hooks[f"{prefix}-mlp-moe_block-MoeBlock_0-wi_0"] = split_moe_wi0
      hooks[f"{prefix}-mlp-moe_block-MoeBlock_0-wi_1"] = split_moe_wi1
      hooks[f"{prefix}-mlp-moe_block-MoeBlock_0-wo"] = reshape_moe_wo

    # Remainder sub-layer prefixes
    if num_remaining > 0:
      for rem_idx in range(num_remaining):
        prefix = f"params-decoder-layers_remainder-layers_{rem_idx}"
        real_layer_idx = attention_pattern_length * (nlayers // attention_pattern_length) + rem_idx
        is_global = real_layer_idx % 6 == 5
        for key in kernel_keys:
          if share_kv_projections and is_global and key == "self_attention-value-kernel":
            continue
          hooks[f"{prefix}-{key}"] = reshape_kernel
        for key in moe_kernel_keys:
          hooks[f"{prefix}-{key}"] = reshape_kernel
        for key in norm_keys:
          hooks[f"{prefix}-{key}"] = scale_rmsnorm_layer

        # Add these specialized 3D tensor hooks inside the loop
        hooks[f"{prefix}-mlp-moe_block-MoeBlock_0-wi_0"] = split_moe_wi0
        hooks[f"{prefix}-mlp-moe_block-MoeBlock_0-wi_1"] = split_moe_wi1
        hooks[f"{prefix}-mlp-moe_block-MoeBlock_0-wo"] = reshape_moe_wo
  else:
    for i in range(nlayers):
      is_global = i % 6 == 5
      prefix = f"params-decoder-layers_{i}"
      for key in kernel_keys:
        if share_kv_projections and is_global and key == "self_attention-value-kernel":
          continue
        hooks[f"{prefix}-{key}"] = reshape_kernel
      for key in moe_kernel_keys:
        hooks[f"{prefix}-{key}"] = reshape_kernel
      for key in norm_keys:
        hooks[f"{prefix}-{key}"] = scale_rmsnorm_layer

      # Add these specialized 3D tensor hooks inside the loop
      hooks[f"{prefix}-mlp-moe_block-MoeBlock_0-wi_0"] = split_moe_wi0
      hooks[f"{prefix}-mlp-moe_block-MoeBlock_0-wi_1"] = split_moe_wi1
      hooks[f"{prefix}-mlp-moe_block-MoeBlock_0-wo"] = reshape_moe_wo
  return hooks


def OLMO3_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  \"\"\"Returns mapping from MaxText to HuggingFace Olmo3 weight paths.\"\"\"

  # Olmo3 uses an inhomogeneous layer cycle (4 layers: 3 sliding, 1 global).
  # MaxText handles this by defining sub-layers (layers_0, layers_1...) within a block.
  n_layers = config["num_hidden_layers"]

  # Default Olmo3 cycle length is 4 if not specified in config
  layer_cycle_interval = maxtext_config.inhomogeneous_layer_cycle_interval

  # Base mapping for embeddings and global norms
  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
      "params-decoder-logits_dense-kernel": "lm_head.weight",
  }

  if scan_layers:
    # Scanned: Map MaxText 'layers_k' to HF layers [k, k+cycle, k+2*cycle, ...]
    for cycle_idx in range(layer_cycle_interval):
      hf_indices = range(cycle_idx, n_layers, layer_cycle_interval)
      prefix = f"params-decoder-layers-layers_{cycle_idx}"

      mapping.update(
          {
              # Attention Projections
              f"{prefix}-attention-query-kernel": [f"model.layers.{i}.self_attn.q_proj.weight" for i in hf_indices],
              f"{prefix}-attention-key-kernel": [f"model.layers.{i}.self_attn.k_proj.weight" for i in hf_indices],
              f"{prefix}-attention-value-kernel": [f"model.layers.{i}.self_attn.v_proj.weight" for i in hf_indices],
              f"{prefix}-attention-out-kernel": [f"model.layers.{i}.self_attn.o_proj.weight" for i in hf_indices],
              # QK Norms (Olmo3 Specific)
              f"{prefix}-attention-query_norm-scale": [f"model.layers.{i}.self_attn.q_norm.weight" for i in hf_indices],
              f"{prefix}-attention-key_norm-scale": [f"model.layers.{i}.self_attn.k_norm.weight" for i in hf_indices],
              # MLP (wi_0=gate, wi_1=up, wo=down)
              f"{prefix}-mlp-wi_0-kernel": [f"model.layers.{i}.mlp.gate_proj.weight" for i in hf_indices],
              f"{prefix}-mlp-wi_1-kernel": [f"model.layers.{i}.mlp.up_proj.weight" for i in hf_indices],
              f"{prefix}-mlp-wo-kernel": [f"model.layers.{i}.mlp.down_proj.weight" for i in hf_indices],
              # Layer Norms
              f"{prefix}-post_self_attention_layer_norm-scale": [
                  f"model.layers.{i}.post_attention_layernorm.weight" for i in hf_indices
              ],
              f"{prefix}-post_mlp_layer_norm-scale": [
                  f"model.layers.{i}.post_feedforward_layernorm.weight" for i in hf_indices
              ],
          }
      )

  else:
    # Unscanned: Direct 1-to-1 mapping
    for i in range(n_layers):
      prefix = f"params-decoder-layers_{i}"
      hf_prefix = f"model.layers.{i}"

      mapping.update(
          {
              f"{prefix}-attention-query-kernel": f"{hf_prefix}.self_attn.q_proj.weight",
              f"{prefix}-attention-key-kernel": f"{hf_prefix}.self_attn.k_proj.weight",
              f"{prefix}-attention-value-kernel": f"{hf_prefix}.self_attn.v_proj.weight",
              f"{prefix}-attention-out-kernel": f"{hf_prefix}.self_attn.o_proj.weight",
              f"{prefix}-attention-query_norm-scale": f"{hf_prefix}.self_attn.q_norm.weight",
              f"{prefix}-attention-key_norm-scale": f"{hf_prefix}.self_attn.k_norm.weight",
              f"{prefix}-mlp-wi_0-kernel": f"{hf_prefix}.mlp.gate_proj.weight",
              f"{prefix}-mlp-wi_1-kernel": f"{hf_prefix}.mlp.up_proj.weight",
              f"{prefix}-mlp-wo-kernel": f"{hf_prefix}.mlp.down_proj.weight",
              f"{prefix}-post_self_attention_layer_norm-scale": f"{hf_prefix}.post_attention_layernorm.weight",
              f"{prefix}-post_mlp_layer_norm-scale": f"{hf_prefix}.post_feedforward_layernorm.weight",
          }
      )

  return mapping


def OLMO3_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  \"\"\"Creates parameter transformation functions for Olmo3.\"\"\"

  # Standard Transpose for Kernels (HF: [Out, In] <-> MaxText: [In, Out])
  def reshape_kernel(input_tensor, target_shape):
    if saving_to_hf:
      flipped_target_shape = np.flip(np.array(target_shape))
      return input_tensor.reshape(flipped_target_shape).T
    else:
      return input_tensor.T.reshape(target_shape)

  # Identity mapping for Norms
  # Olmo3 checkpoints typically have weights ~1.0.
  # If MaxText RMSNorm adds 1.0 (x * (1+w)), we might need w-1.0.
  # However, if weights are zeroed/mismatched, identity is safer to restore logic flow.
  def scale_rmsnorm_layer(input_tensor, target_shape):
    return input_tensor.reshape(target_shape)

  # Identity mapping for QK Norms (assuming MaxText attentions.py was patched to use global norm)
  def adapt_olmo3_qk_norm(input_tensor, target_shape):
    return input_tensor.reshape(target_shape)

  # Padding for Embedding layer (Vocab size adjustments)
  def pad_hf_embedding_layer(input_tensor, target_shape):
    source_vocab_size = input_tensor.shape[0]
    target_vocab_size = target_shape[0]
    if source_vocab_size == target_vocab_size:
      return input_tensor
    if saving_to_hf:
      return input_tensor[:target_vocab_size, :]
    else:
      padded_tensor = np.zeros(target_shape, dtype=input_tensor.dtype)
      padded_tensor[:source_vocab_size, :] = input_tensor
      return padded_tensor

  hooks = {
      "params-token_embedder-embedding": pad_hf_embedding_layer,
      "params-decoder-logits_dense-kernel": reshape_kernel,
      "params-decoder-decoder_norm-scale": scale_rmsnorm_layer,
  }

  kernel_keys = [
      "attention-query-kernel",
      "attention-key-kernel",
      "attention-value-kernel",
      "attention-out-kernel",
      "mlp-wi_0-kernel",
      "mlp-wi_1-kernel",
      "mlp-wo-kernel",
  ]

  norm_keys = [
      "post_self_attention_layer_norm-scale",
      "post_mlp_layer_norm-scale",
      "attention-query_norm-scale",
      "attention-key_norm-scale",
  ]

  cycle_len = getattr(maxtext_config, "inhomogeneous_layer_cycle_interval", 4)
  n_layers = config["num_hidden_layers"]

  if scan_layers:
    for cycle_idx in range(cycle_len):
      prefix = f"params-decoder-layers-layers_{cycle_idx}"
      for key in kernel_keys:
        hooks[f"{prefix}-{key}"] = reshape_kernel
      for key in norm_keys:
        # For QK norm, we use the specific adaptor (which is currently Identity
        # but separates logic if we need to revert to averaging later)
        if "attention-" in key and "_norm-scale" in key:
          hooks[f"{prefix}-{key}"] = adapt_olmo3_qk_norm
        else:
          hooks[f"{prefix}-{key}"] = scale_rmsnorm_layer
  else:
    for i in range(n_layers):
      prefix = f"params-decoder-layers_{i}"
      for key in kernel_keys:
        hooks[f"{prefix}-{key}"] = reshape_kernel
      for key in norm_keys:
        if "attention-" in key and "_norm-scale" in key:
          hooks[f"{prefix}-{key}"] = adapt_olmo3_qk_norm
        else:
          hooks[f"{prefix}-{key}"] = scale_rmsnorm_layer

  return hooks


# {maxtext model name: {maxtext weight name: hf weight name}}
PARAM_MAPPING = {
    "gemma2-2b": GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma2-9b": GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma2-27b": GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma3-4b": GEMMA3_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma3-12b": GEMMA3_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma3-27b": GEMMA3_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma4-26b": GEMMA4_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma4-31b": GEMMA4_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen2.5-1.5b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen2.5-7b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen2.5-14b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-0.6b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-1.7b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-1.7b-base": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-4b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-4b-base": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-4b-thinking-2507": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-8b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-8b-base": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-14b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-14b-base": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-32b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "llama3.1-8b": LLAMA31_MAXTEXT_TO_HF_PARAM_MAPPING,
    "llama3.1-70b": LLAMA31_MAXTEXT_TO_HF_PARAM_MAPPING,
    "llama3.1-405b": LLAMA31_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-30b-a3b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-30b-a3b-base": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-235b-a22b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-coder-480b-a35b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "deepseek2-16b": DEEPSEEK_MAXTEXT_TO_HF_PARAM_MAPPING,
    "deepseek3-671b": DEEPSEEK_MAXTEXT_TO_HF_PARAM_MAPPING,
    "deepseek3.2-671b": DEEPSEEK_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gpt-oss-20b": GPT_OSS_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gpt-oss-120b": GPT_OSS_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-omni-30b-a3b": QWEN3_OMNI_MOE_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-next-80b-a3b": QWEN3_NEXT_MAXTEXT_TO_HF_PARAM_MAPPING,
    "mixtral-8x7b": MIXTRAL_MAXTEXT_TO_HF_PARAM_MAPPING,
    "mixtral-8x22b": MIXTRAL_MAXTEXT_TO_HF_PARAM_MAPPING,
    "olmo3-7b": OLMO3_MAXTEXT_TO_HF_PARAM_MAPPING,
    "olmo3-7b-pt": OLMO3_MAXTEXT_TO_HF_PARAM_MAPPING,
    "olmo3-32b": OLMO3_MAXTEXT_TO_HF_PARAM_MAPPING,
}

# {maxtext model name: {maxtext weight name: bi-directional transform}}
HOOK_FNS = {
    "gemma2-2b": GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma2-9b": GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma2-27b": GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma3-4b": GEMMA3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma3-12b": GEMMA3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma3-27b": GEMMA3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma4-26b": GEMMA4_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma4-31b": GEMMA4_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen2.5-1.5b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen2.5-7b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen2.5-14b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-0.6b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-1.7b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-1.7b-base": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-4b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-4b-base": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-4b-thinking-2507": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-8b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-8b-base": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-14b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-14b-base": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-32b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "llama3.1-8b": LLAMA31_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "llama3.1-70b": LLAMA31_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "llama3.1-405b": LLAMA31_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-30b-a3b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-30b-a3b-base": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-235b-a22b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-coder-480b-a35b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "deepseek2-16b": DEEPSEEK_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "deepseek3-671b": DEEPSEEK_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "deepseek3.2-671b": DEEPSEEK_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gpt-oss-20b": GPT_OSS_TO_HF_PARAM_HOOK_FN,
    "gpt-oss-120b": GPT_OSS_TO_HF_PARAM_HOOK_FN,
    "qwen3-omni-30b-a3b": QWEN3_OMNI_MOE_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-next-80b-a3b": QWEN3_NEXT_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "mixtral-8x7b": MIXTRAL_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "mixtral-8x22b": MIXTRAL_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "olmo3-7b": OLMO3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "olmo3-7b-pt": OLMO3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "olmo3-32b": OLMO3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
}

VLLM_HOOK_FNS = {
    "qwen3": QWEN3_NNX_TO_VLLM_PARAM_HOOK_FN,
    "llama3.1": LLAMA31_NNX_TO_VLLM_PARAM_HOOK_FN,
    "deepseek3": DEEPSEEK_NNX_TO_VLLM_PARAM_HOOK_FN,
}
\n"""


# File: src/maxtext/checkpoint_conversion/to_maxtext.py (commit 313890777)
TO_MAXTEXT_CONVERTER_RAW = """\n# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

\"\"\"
This script converts a HuggingFace model checkpoint to a MaxText-compatible
Orbax checkpoint.

Key Parameters (to be set in the config file or as command-line overrides):
  model_name: (Required) The name of the model to convert (e.g., "gemma2-2b").
              Must be a key in `maxtext.utils.globals.HF_IDS`.
  base_output_directory: (Optional) The directory where the converted HuggingFace
                         checkpoint will be saved. Can be a local path, a GCS
                         path (gs://...), or a HuggingFace Hub repo ID (hf://...).
                         Defaults to "./mt_output/".
  scan_layers: (bool) Whether the MaxText model was trained with scanned layers.
               This must match the training configuration of the checkpoint.
  --lazy_load_tensors: (bool) If True, uses an on-demand loading strategy to minimize RAM
             usage during conversion. Recommended if, 2 * model_size (GB) >= system RAM
             Defaults to False.
  --hf_model_path: (Optional) Specifies a local or remote directory containing the model weights.
      If unspecified, we use the default Hugging Face repository ID
      (e.g., openai/gpt-oss-20b; see `HF_IDS[model_name]` in `maxtext.utils.globals`).
      This is necessary for locally dequantized models like GPT-OSS or DeepSeek.
  --save_dtype: (Optional) Specifies the data type of saved model weights.
             Default to `bfloat16` to save memory.

Environment Variables:
  HF_AUTH_TOKEN: (Required) HuggingFace authentication token, needed to
                 download models from HuggingFace Hub.

Example Usage:
  To convert a gemma2-2b model and save it to a specific directory:

   python -m maxtext.checkpoint_conversion.to_maxtext \
    maxtext/configs/base.yml model_name="gemma2-2b" \
    base_output_directory="/path/to/your/output/directory" \
    hf_access_token=${HF_TOKEN?} hardware=cpu skip_jax_distributed_system=True \
    scan_layers=False

  For models with scanned layers (e.g., some custom architectures), you might
  need to set scan_layers=True and param_scan_axis accordingly.

  To convert a 70B model with minimal RAM usage:

   python -m maxtext.checkpoint_conversion.to_maxtext \
    maxtext/configs/base.yml model_name="llama3.1-70b" \
    base_output_directory="gs://my-bucket/maxtext-checkpoints" \
    hf_access_token=${HF_TOKEN?} hardware=cpu skip_jax_distributed_system=True \
    --lazy_load_tensors=True
\"\"\"

import argparse
from functools import partial
import json
import os
import sys
import threading
import time
from typing import Any, Callable, List, Sequence
import absl
import ml_dtypes
import torch
import flax.linen as nn
from huggingface_hub import hf_hub_download, list_repo_files
import jax
from maxtext.configs import pyconfig
from maxtext.configs.types import DType
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.checkpoint_conversion.standalone_scripts.llama_or_mistral_ckpt import save_weights_to_checkpoint
from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
from maxtext.checkpoint_conversion.utils.param_mapping import HOOK_FNS, PARAM_MAPPING
from maxtext.checkpoint_conversion.utils.utils import MemoryMonitorTqdm, apply_hook_fns, load_hf_dict_from_transformers, load_hf_dict_from_safetensors, print_peak_memory, print_ram_usage, validate_and_filter_param_map_keys
from maxtext.inference.inference_utils import str2bool
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import max_logging, max_utils, maxtext_utils
from maxtext.utils.globals import HF_IDS
import numpy as np
from orbax.checkpoint import type_handlers
from safetensors import safe_open


absl.logging.set_verbosity(absl.logging.INFO)  # for max_logging.log


class LazyHFLoader:
  \"\"\"
  Loads Hugging Face weights on-demand to minimize RAM usage.

  This class is the core of the "lazy loading" feature. Instead of loading the
  entire model into memory at once, it reads the model's index file (e.g.,
  `model.safetensors.index.json`) to understand the mapping between tensor names
  and the shard files they belong to.

  When a specific tensor is requested via `get_tensor`, this class:
  1. Identifies the correct shard file.
  2. Downloads the shard file if not already cached by `huggingface_hub`.
  3. Opens the shard and extracts *only* the requested tensor into memory.

  This approach is highly memory-efficient, especially for `safetensors`, as
  it avoids loading entire multi-gigabyte shard files when only a small piece
  is needed. A threading lock (`_ram_lock`) is used to ensure that memory-intensive
  file-opening operations are serialized to prevent RAM spikes, while downloads
  can still occur in parallel.
  \"\"\"

  def __init__(self, model_id, token, revision=None):
    self.model_id = model_id
    self.token = token
    self.revision = revision
    # Whether loads from local directory
    self.is_local = os.path.isdir(self.model_id)
    self.shard_map = {}
    self.current_shard_name = None
    self.current_shard_content = {}
    # Use a lock to serialize heavy RAM operations, but NOT downloads
    self._ram_lock = threading.Lock()
    self._initialize_index()

  def __getstate__(self):
    \"\"\"Allows pickling/copying by excluding the non-pickleable lock.\"\"\"
    state = self.__dict__.copy()
    del state["_ram_lock"]
    return state

  def __setstate__(self, state):
    \"\"\"Restores state after pickling/copying and recreates a new lock.\"\"\"
    self.__dict__.update(state)
    self._ram_lock = threading.Lock()

  def _initialize_index(self):
    \"\"\"Fetches and parses the Hugging Face model index file to build a shard map.\"\"\"
    if self.is_local:
      files = os.listdir(self.model_id)
    else:
      files = list_repo_files(self.model_id, token=self.token)

    # Prefer safetensors
    if "model.safetensors.index.json" in files:
      index_file = "model.safetensors.index.json"
    elif "model.safetensors" in files:
      # Single file case
      self.shard_map = {None: "model.safetensors"}
      return
    else:
      raise ValueError("Could not find recognized model weights (safetensors) in HF repo.")

    # Download and parse the index
    max_logging.log(f"Loading index file: {index_file}")
    if self.is_local:
      index_path = os.path.join(self.model_id, index_file)
    else:
      index_path = hf_hub_download(
          repo_id=self.model_id,
          filename=index_file,
          token=self.token,
          revision=self.revision,
      )
    with open(index_path, "r", encoding="utf-8") as f:
      index_data = json.load(f)
    self.shard_map = index_data["weight_map"]

  def get_tensor(self, key: str) -> np.ndarray:
    \"\"\"
    Retrieves a specific tensor by name, lazily loading its shard if necessary.

    This is the main entry point for accessing model weights. It determines
    which shard file contains the tensor, ensures it's downloaded, and then
    reads the tensor data.

    For safetensors, this is extremely efficient as it memory-maps the file
    and reads only the required tensor's data from disk.
    \"\"\"
    # Handle single-file models (shard map key might be None or we just know the filename)
    shard_name = self.shard_map.get(key)
    if shard_name is None and None in self.shard_map:
      shard_name = self.shard_map[None]
    elif shard_name is None:
      # Fallback: sometimes keys in index don't perfectly match requested keys if there are prefix mismatches.
      # You might need advanced fuzzy matching here if you encounter errors.
      raise ValueError(f"Key {key} not found in HF checkpoint index.")

    if self.is_local:
      local_path = os.path.join(self.model_id, shard_name)
    else:
      # STEP 1: Download outside the lock.
      # multiple threads can download different shards at the same time.
      local_path = hf_hub_download(
          repo_id=self.model_id,
          filename=shard_name,
          token=self.token,
          revision=self.revision,
      )

    # STEP 2: Lock ONLY the reading into RAM.
    # This prevents multiple threads from simultaneously allocating large chunks of RAM.
    with self._ram_lock:
      with safe_open(local_path, framework="np", device="cpu") as f:
        return f.get_tensor(key)


class LazyTensor:
  \"\"\"
  A proxy object that looks like a NumPy array but delays actual loading
  and transformation until __array__ is called (e.g., by Orbax during save).
  \"\"\"

  def __init__(
      self,
      load_fn: Callable[[], np.ndarray],
      shape: tuple,
      dtype,
      name: str = "unknown",
  ):
    self._load_fn = load_fn
    self.shape = shape
    self.dtype = np.dtype(dtype)
    self.ndim = len(shape)
    self.name = name

  @property
  def size(self):
    \"\"\"Total number of elements in the tensor.\"\"\"
    return np.prod(self.shape)

  @property
  def nbytes(self):
    \"\"\"Return estimated nbytes so Orbax doesn't need to load the real array to find out.\"\"\"
    return self.size * self.dtype.itemsize

  @property
  def itemsize(self):
    return self.dtype.itemsize

  def __array__(self, dtype=None):
    \"\"\"
    Materializes the tensor data.

    When this method is invoked, it finally calls the `_load_fn` that was
    provided during initialization. This function executes the actual loading
    and transformation of the tensor from the Hugging Face checkpoint. The
    resulting NumPy array is then returned to the caller.
    \"\"\"
    # This method is called just-in-time by Orbax when saving this specific leaf.
    try:
      arr = self._load_fn()
    except Exception as e:
      max_logging.log(f"FATAL ERROR: Failed to load tensor '{self.name}' (shape {self.shape}). Error: {e}")
      # Re-raise the original exception so it doesn't get masked by "object __array__..."
      raise

    if not isinstance(self.shape, list) and arr.shape != self.shape:
      raise ValueError(f"Shape mismatch for tensor '{self.name}'. Expected {self.shape}, but got {arr.shape}.")

    # Ensure it's a standard numpy array (converts JAX arrays if necessary)
    if not isinstance(arr, np.ndarray):
      arr = np.array(arr)

    if dtype is not None and arr.dtype != dtype:
      return arr.astype(dtype)
    return arr

  def __repr__(self):
    return f"LazyTensor(name={self.name}, shape={self.shape}, dtype={self.dtype})"


class LazyTensorHandler(type_handlers.NumpyHandler):
  \"\"\"
  Custom Orbax handler for LazyTensor.

  It masquerades as a standard NumpyHandler so that the resulting checkpoint
  has the standard 'array_metadatas' structure and can be loaded by
  standard MaxText instances.
  \"\"\"

  async def serialize(self, value, *args, **kwargs):
    # MATERIALIZE: Trigger the lazy load (__array__) explicitly before saving.
    # This ensures the parent NumpyHandler receives a real np.ndarray.
    if hasattr(value, "__array__"):
      value = np.array(value)

    return await super().serialize(value, *args, **kwargs)


# Register LazyTensor with the custom handler.
# It's safe to register this globally even if eager loading is used.
type_handlers.register_type_handler(LazyTensor, LazyTensorHandler(), override=True)


def get_maxtext_model_info(config):
  \"\"\"Initializes the abstract MaxText model and returns parameter mapping information.

  Args:
    config: The MaxText configuration object.

  Returns:
    maxtext_abstract_dict: A dictionary mapping MaxText parameter keys to a tuple
      (index, target_shape), where 'index' is the position of the parameter in the
      flattened parameter list.
    abstract_params_treedef: The tree structure definition of the abstract model parameters.
  \"\"\"
  # Setup JAX distributed system and mesh
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

  max_logging.log("Initializing MaxText abstract model...")
  quant = quantizations.configure_quantization(config)
  maxtext_model_flax = models.transformer_as_linen(config, mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)

  # Get abstract model structure (name, shape) without materializing the weights to save memory
  abstract_params_tree = maxtext_utils.get_abstract_param(maxtext_model_flax, config)["params"]

  abstract_params_flat, _ = jax.tree_util.tree_flatten_with_path(abstract_params_tree)
  # Standardize abstract tree for later unflattening
  abstract_params_tree = jax.tree.map(
      lambda _: 0,
      abstract_params_tree,
      is_leaf=lambda x: isinstance(x, nn.LogicallyPartitioned),
  )
  abstract_params_treedef = jax.tree_util.tree_structure(abstract_params_tree)

  max_logging.log("MaxText abstract model and state initialized.")

  # preprocess state
  maxtext_abstract_dict = {}
  for mt_target_idx, (path_tuple, abstract_leaf_value) in enumerate(abstract_params_flat):
    key_parts = [k.key for k in path_tuple if hasattr(k, "key")]
    mt_param_key = "params-" + "-".join(key_parts)
    mt_target_shape = abstract_leaf_value.shape
    maxtext_abstract_dict[mt_param_key] = (mt_target_idx, mt_target_shape)

  return maxtext_abstract_dict, abstract_params_treedef


def _build_multi_axis_stacked_tensor(
    hf_source_keys: List[List[str]],
    tensor_getter_fn: Callable[[str], np.ndarray],
    hook_fns: Any,
    target_shape: tuple,
    config,
) -> np.ndarray:
  \"\"\"Builds a MaxText tensor by stacking HF weights along two axes (experts and layers).

  This function handles the complex case for scanned MoE layers, producing a tensor
  with the shape (num_experts, num_layers, ...).

  Args:
      hf_source_keys: A nested (2D) list of Hugging Face parameter names.
                      Outer list iterates experts, inner list iterates layers.
      tensor_getter_fn: A callable that takes a HF key and returns the tensor (as numpy array).
      hook_fns: The hook function(s) to apply to each individual weight.
      target_shape: The final shape of the target MaxText tensor.
      config: The MaxText pyconfig object.

  Returns:
      The final, assembled NumPy array for the MaxText parameter.
  \"\"\"
  all_expert_tensors = []
  # The hook function needs the shape of an individual slice, not the full stacked tensor.
  # For multi-axis stacking (experts, layers, ...), the slice shape is target_shape[2:]
  mt_slice_shape = target_shape[2:]

  # Outer loop iterates through experts
  for layer_keys_for_expert in hf_source_keys:
    layer_tensors_for_expert = []
    # Inner loop iterates through layers for the current expert
    for hf_key_single in layer_keys_for_expert:
      hf_tensor_numpy = tensor_getter_fn(hf_key_single)
      processed_hf_tensor = apply_hook_fns(hf_tensor_numpy, mt_slice_shape, hook_fns)
      layer_tensors_for_expert.append(processed_hf_tensor)
    all_expert_tensors.append(np.stack(layer_tensors_for_expert, axis=0))
  return np.stack(all_expert_tensors, axis=0)


def _build_single_axis_stacked_tensor(
    hf_source_keys: List[str],
    tensor_getter_fn: Callable[[str], np.ndarray],
    hook_fns: Any,
    target_shape: tuple,
    config,
) -> np.ndarray:
  \"\"\"Builds a MaxText tensor by stacking HF weights along a single axis.

  This function handles both standard scanned layers (e.g., attention) and
  unscanned MoE layers (which are stacked along the expert axis).

  Args:
      hf_source_keys: A 1D list of Hugging Face parameter names.
      tensor_getter_fn: A callable that takes a HF key and returns the tensor (as numpy array).
      hook_fns: The hook function(s) to apply to each individual weight.
      target_shape: The final shape of the target MaxText tensor.
      config: The MaxText pyconfig object.

  Returns:
      The final, assembled NumPy array for the MaxText parameter.
  \"\"\"
  tensors_to_stack = []

  if config.scan_layers:
    # If it's a standard scanned layer, we use the configured param_scan_axis.
    axis_to_stack = config.param_scan_axis
  else:
    # Otherwise, if an unscanned MoE layer, and we stack along the expert axis (0).
    axis_to_stack = 0

  # The hook function needs the shape of an individual slice, not the full stacked tensor.
  # We calculate it by removing the stacking dimension from the final target shape.
  mt_slice_shape_list = list(target_shape)
  del mt_slice_shape_list[axis_to_stack]
  mt_slice_shape = tuple(mt_slice_shape_list)

  for hf_key_single in hf_source_keys:
    hf_tensor_numpy = tensor_getter_fn(hf_key_single)
    processed_hf_tensor = apply_hook_fns(hf_tensor_numpy, mt_slice_shape, hook_fns)
    tensors_to_stack.append(processed_hf_tensor)

  # Stack all processed tensors along the determined axis.
  return np.stack(tensors_to_stack, axis=axis_to_stack)


def _get_hf_loading_function(hf_source_keys_or_key, tensor_getter, hook_fn, mt_target_shape_or_shapes, config):
  \"\"\"Determine the loading function for HF keys.
  HF keys can take four forms:
    Case 1: Unscanned (single string)
    Case 2: Scanned (list of strings)
    Case 3: Unscanned with expert stacking (list of strings)
    Case 4: Scanned with expert stacking (nested list of strings)
  \"\"\"
  load_fn = None
  if not isinstance(hf_source_keys_or_key, list):
    # Case 1: Single hf key (str)
    def _loader(getter, key, shape, hook):
      return apply_hook_fns(getter(key), shape, hook)

    load_fn = partial(
        _loader,
        tensor_getter,
        hf_source_keys_or_key,
        mt_target_shape_or_shapes,
        hook_fn,
    )
  # Stacked mapping
  elif not isinstance(hf_source_keys_or_key[0], list):
    # Case 2 or 3: Single-Axis Stacked hf keys (un-nested list)
    load_fn = partial(
        _build_single_axis_stacked_tensor,
        hf_source_keys_or_key,
        tensor_getter,
        hook_fn,
        mt_target_shape_or_shapes,
        config,
    )
  else:
    # isinstance(hf_source_keys_or_key[0], list)
    # Case 4: Multi-Axis Stacked hf keys (nested list)
    load_fn = partial(
        _build_multi_axis_stacked_tensor,
        hf_source_keys_or_key,
        tensor_getter,
        hook_fn,
        mt_target_shape_or_shapes,
        config,
    )
  return load_fn


def _get_maxtext_indices_and_shapes(mt_param_key_or_keys, maxtext_abstract_dict):
  \"\"\"Resolves MaxText key(s) to target indices and shapes.

  The index is the parameter's order in `maxtext_abstract_dict.keys()`.
  This function handles two forms of MaxText keys:
  - `atomic_mt_key`: A single string representing one MaxText parameter that map to HF parameter(s).
  - `composite_mt_key`: A tuple of strings for multiple MaxText parameters that map to HF parameter(s).
  \"\"\"
  is_composite_mt_key = isinstance(mt_param_key_or_keys, tuple)
  # atomic_mt_key
  if not is_composite_mt_key:
    mt_target_idx, mt_target_shape = maxtext_abstract_dict[mt_param_key_or_keys]
    return mt_target_idx, mt_target_shape
  # composite_mt_key
  mt_target_indices, mt_target_shapes = [], []
  for mt_param_key in mt_param_key_or_keys:
    mt_target_idx, mt_target_shape = maxtext_abstract_dict[mt_param_key]
    mt_target_indices.append(mt_target_idx)
    mt_target_shapes.append(mt_target_shape)
  return mt_target_indices, mt_target_shapes


def _get_maxtext_weight(
    load_fn,
    mt_target_idx_or_indices,
    mt_target_shape_or_shapes,
    mt_param_key_or_keys,
    final_mt_weights,
    save_dtype,
    use_lazy_load,
):
  \"\"\"Loads Hugging Face parameters and converts them to MaxText parameters.

  This function handles loading based on tensor mode (eager or lazy) and
  processes MaxText keys, which can be `atomic_mt_key` or `composite_mt_key`.
  \"\"\"
  is_composite_mt_key = isinstance(mt_param_key_or_keys, tuple)
  if not use_lazy_load:
    # Case 1: Eager mode
    # In eager mode, we execute the function immediately to get the
    # NumPy array and append it to our list of weights.
    final_mt_tensor_numpy = load_fn()
    if not is_composite_mt_key:
      # Case 1.1: Eager mode, `atomic_mt_key`
      final_mt_weights[mt_target_idx_or_indices] = final_mt_tensor_numpy
      if final_mt_tensor_numpy.shape != mt_target_shape_or_shapes:
        raise ValueError(
            f"Shape mismatch for {mt_param_key_or_keys}: Expected {mt_target_shape_or_shapes}, "
            f"got {final_mt_tensor_numpy.shape}"
        )
    else:
      # Case 1.2: Eager mode, `composite_mt_key`
      # The hook returns a tensor that can be split in last dim.
      # In eager mode, we can just split the materialized tensor.
      for i, mt_target_idx in enumerate(mt_target_idx_or_indices):
        final_mt_weights[mt_target_idx] = final_mt_tensor_numpy[..., i]
        if final_mt_weights[mt_target_idx].shape != mt_target_shape_or_shapes[i]:
          raise ValueError(
              f"Shape mismatch for {mt_param_key_or_keys[i]}: Expect {mt_target_shape_or_shapes[i]}, "
              f"got {final_mt_weights[mt_target_idx].shape}"
          )
  else:
    # Case 2: Lazy mode
    # In lazy mode, we don't execute the loading/transformation function
    # immediately. Instead, we wrap it in a `LazyTensor` object. This
    # object acts as a placeholder that holds all the information needed
    # to load the tensor later (the `load_fn`, shape, dtype).
    # The actual data will only be loaded when Orbax calls `__array__`
    # on this object during the saving process.
    final_mt_tensor_numpy = LazyTensor(
        load_fn,
        mt_target_shape_or_shapes,
        save_dtype,
        name=mt_param_key_or_keys,
    )
    if not is_composite_mt_key:
      # Case 2.1: Lazy mode, `atomic_mt_key`
      final_mt_weights[mt_target_idx_or_indices] = final_mt_tensor_numpy
    else:
      # Case 2.2: Lazy mode, `composite_mt_key`
      # For a composite key, the hook returns a tensor that can be split in last dim.
      # For lazy loading, we can't split the tensor until it's loaded.
      # We create multiple LazyTensors, each responsible for loading the
      # full source tensor but then slicing its piece. Parent HF tensor is loaded repeatedly.
      for i, mt_target_idx in enumerate(mt_target_idx_or_indices):

        def _slicing_loader(base_loader, slice_idx):
          return np.array(base_loader)[..., slice_idx]

        # Each LazyTensor gets a new load_fn that wraps the original and applies the slice.
        slicing_load_fn = partial(_slicing_loader, final_mt_tensor_numpy, i)
        final_mt_weights[mt_target_idx] = LazyTensor(
            slicing_load_fn,
            mt_target_shape_or_shapes[i],
            save_dtype,
            name=mt_param_key_or_keys[i],
        )


def main(
    args: Sequence[str],
    lazy_load_tensors: bool = False,
    eager_load_method: str = "transformers",
    hf_model_path: str | None = None,
    revision: str | None = None,
    save_dtype: str = "bfloat16",
    simulated_cpu_devices_count: int = 16,
) -> None:
  overall_start = time.time()
  # Check if the user is using an Instruct version. If so, use the base model architecture
  for i, arg in enumerate(args):
    if arg.startswith("model_name="):
      model_name_arg = args[i].split("=")[1]
      model_name_original = model_name_arg
      if "-Instruct" in model_name_arg:
        max_logging.log("Warning: You want an Instruct version, so we are using the base model architecture instead.")
        model_name_arg = model_name_arg.replace("-Instruct", "")
        args[i] = f"model_name={model_name_arg}"
      break

  # check the supported model ids
  if model_name_original not in HF_IDS:
    raise ValueError(
        f"Unsupported model name: {model_name_original}.\
                      Supported models are: {list(HF_IDS.keys())}"
    )

  model_id = hf_model_path or HF_IDS[model_name_original]

  # Initialize maxtext config
  config = pyconfig.initialize(args)
  max_utils.print_system_information()

  if not config.base_output_directory:
    output_directory = f"tmp/{config.run_name}"
  else:
    output_directory = config.base_output_directory

  hf_token = config.hf_access_token

  if lazy_load_tensors and config.use_multimodal:
    raise ValueError("lazy loading of HF tensors is not supported for multimodal models yet.")

  hf_state_dict_numpy = None
  hf_loader = None

  # Define the appropriate tensor getter based on mode
  if lazy_load_tensors:
    max_logging.log(f"Lazy loading ENABLED. Initializing LazyHFLoader for: {model_id}...")
    hf_loader = LazyHFLoader(model_id, hf_token, revision=revision)

    print_ram_usage("After LazyLoader init")
    tensor_getter = hf_loader.get_tensor
  else:
    max_logging.log(f"Lazy loading DISABLED. Loading full HuggingFace model: {model_id}...")

    # Eager load methods:
    # - Method 1: transformers_class.from_pretrained(..., dtype="auto")
    # - Method 2: safetensors.safe_open(..., framework="pt")
    #
    # Comparison:
    # - Both methods result in the same dtype (usually bfloat16) and model structure
    #   for most models (e.g., DeepSeek-V2), with similar loading times.
    # - Exception: Gemma-3 uses different internal naming (prefixes) between
    #   Method 1 and Method 2. Current MaxText 'param_mapping' for Gemma-3 assumes
    #   the Transformers-style structure (Method 1).
    # - The 'safetensors' method is a necessary fallback for:
    #   1. "Day-0" models where the official Transformers code hasn't been merged yet
    #      (e.g., DeepSeek-V3.2 during its initial release).
    #   2. Weights omitted by official Transformers class
    #      (e.g., Multi-Token Prediction weights (`layers.61`) in DeepSeek-V3).
    #
    # Recommendation:
    # - Use 'transformers' as the default for backward compatibility of mapping.
    # - 'safetensors' is an interchangeable and valid alternative for most models,
    #   and is strictly required if the model or specific weights lack Transformers support.
    if eager_load_method == "transformers":
      max_logging.log("Eager load with Transformers backend, from_pretrained with auto dtype")
      # For auto mode, loaded dtype is the same as `dtype` specified in config.json (or `torch_dtype` for older version)
      # e.g., https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json#L54
      hf_state_dict_numpy = load_hf_dict_from_transformers(model_id, token=hf_token, revision=revision, dtype="auto")
    elif eager_load_method == "safetensors":
      max_logging.log("Eager load with Safetensors backend, safe_open with pt framework")
      # For safe_open, loaded dtype is the same as original safetensor
      # e.g., https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/model.safetensors.index.json
      hf_state_dict_numpy = load_hf_dict_from_safetensors(model_id, token=hf_token, revision=revision, framework="pt")
    else:
      raise NotImplementedError

    unique_dtypes = {tensor.dtype for tensor in hf_state_dict_numpy.values()}
    max_logging.log(f"HuggingFace model loaded. dtypes: {unique_dtypes}")
    print_ram_usage("After full HF model load")

    def _eager_getter(key):
      if key not in hf_state_dict_numpy:
        raise ValueError(f"HuggingFace key {key} not found in state_dict.")
      v = hf_state_dict_numpy[key]
      # target dtype is "float32"
      if save_dtype == DType.FLOAT32:
        return v.to(torch.float32).numpy()
      # target dtype is "bfloat16"
      elif save_dtype == DType.BFLOAT16:
        # - torch.bfloat16 -> torch.float32 -> np.float32 -> ml_dtypes.bfloat16
        #   As numpy doesn't accept bfloat16 directly, we convert to float32 first
        # - torch.float16 -> np.float16 -> ml_dtypes.bfloat16
        # - torch.float32 -> np.float32 -> ml_dtypes.bfloat16
        if v.dtype == torch.bfloat16:
          v = v.to(torch.float32)
        return v.numpy().astype(ml_dtypes.bfloat16)
      raise NotImplementedError(f"Save dtype {save_dtype} is not currently implemented.")

    tensor_getter = _eager_getter

  # Get parameter mappings and hooks
  model_key = config.model_name
  # load config
  hf_config_obj = HF_MODEL_CONFIGS[model_key]
  hf_config_dict = hf_config_obj.to_dict()
  # example of param mapping (gemma2, maxtext:huggingface):
  # "params-decoder-layers_{maxtext_layer_idx}-pre_self_attention_norm_global-scale":
  #   f"model.layers.{global_layer_idx}.input_layernorm.weight",
  param_map_mt_to_hf = PARAM_MAPPING[model_key](hf_config_dict, config, config.scan_layers)
  # Example of Hook FN mapping, to perform reshape:
  # f"params-decoder-layers_{maxtext_layer_idx}-self_attention_global-key-kernel": reshape_kernel,
  hook_fn_map_mt = HOOK_FNS[model_key](hf_config_dict, config, config.scan_layers, saving_to_hf=False)
  max_logging.log("Parameter mappings and hooks obtained.")

  maxtext_abstract_dict, abstract_params_treedef = get_maxtext_model_info(config)

  # Weight transformation
  max_logging.log("Starting weight transformation...")
  start = time.time()
  # Stores MaxText weights: numpy.ndarray
  final_mt_weights = [None] * len(maxtext_abstract_dict)

  # Preprocess key
  filtered_map_keys = validate_and_filter_param_map_keys(param_map_mt_to_hf.keys(), maxtext_abstract_dict.keys())

  for mt_param_key_or_keys in MemoryMonitorTqdm(
      filtered_map_keys,
      desc="Transforming weights",
      unit="param",
      leave=True,
      dynamic_ncols=True,
      smoothing=0,
  ):
    if not lazy_load_tensors:
      max_logging.log(f"maxtext param: {mt_param_key_or_keys}")

    hf_source_keys_or_key = param_map_mt_to_hf.get(mt_param_key_or_keys)
    if hf_source_keys_or_key is None:
      raise ValueError(f"MaxText parameter {mt_param_key_or_keys} not found in mapping.")
    hook_fn = hook_fn_map_mt.get(mt_param_key_or_keys)

    # Step 1: Resolves MaxText key(s) to target indices and shapes
    # based on MaxText key form (`atomic_mt_key` or `composite_mt_key`)
    mt_target_idx_or_indices, mt_target_shape_or_shapes = _get_maxtext_indices_and_shapes(
        mt_param_key_or_keys, maxtext_abstract_dict
    )

    # Step 2: Determine the loading function for hf key
    # based on hf_key form (unscanned, scanned, unscanned with expert stacking, or scanned with expert stacking)
    load_fn = _get_hf_loading_function(
        hf_source_keys_or_key,
        tensor_getter,
        hook_fn,
        mt_target_shape_or_shapes,
        config,
    )

    # Step 3: Load hf keys and convert to maxtext keys
    # based on tensor load mode (lazy, eager) and MaxText key form (`atomic_mt_key` or `composite_mt_key`)
    _get_maxtext_weight(
        load_fn,
        mt_target_idx_or_indices,
        mt_target_shape_or_shapes,
        mt_param_key_or_keys,
        final_mt_weights,
        save_dtype,
        lazy_load_tensors,
    )

  del hf_state_dict_numpy
  max_logging.log("Weight transformation preparation complete.")
  max_logging.log(f"Elapse for transform: {(time.time() - start) / 60:.2f} min")
  print_ram_usage("Before creating full JAX tree")

  # Create final MaxText parameters tree
  jax_weights = jax.tree_util.tree_unflatten(abstract_params_treedef, final_mt_weights)
  del final_mt_weights, abstract_params_treedef

  print_ram_usage("Before saving")
  if lazy_load_tensors:
    max_logging.log("Starting checkpoint save (loading weights just-in-time)...")
  else:
    max_logging.log("Starting checkpoint save...")

  # Save the converted weights to a MaxText checkpoint.
  # If simulated_cpu_devices_count > 1, weights are promoted from NumPy to JAX arrays
  # and sharded across virtual devices.
  save_weights_to_checkpoint(
      output_directory,
      jax_weights,
      simulated_cpu_devices_count,
      config.checkpoint_storage_use_ocdbt,
      config.checkpoint_storage_use_zarr3,
  )

  print_ram_usage("Program Ends")
  max_logging.log(f"Conversion complete. Checkpoint saved to {output_directory}")
  max_logging.log(f"Overall Elapse: {(time.time() - overall_start) / 60:.2f} min")
  print_peak_memory()


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Suppress TensorFlow logging

  # Define local parser
  parser = argparse.ArgumentParser()
  # Lazy load uses `safetensors.safe_open` with np
  parser.add_argument(
      "--lazy_load_tensors",
      type=str2bool,
      required=False,
      default=False,
      help="Whether to use lazy loading of HF tensors",
  )
  # Eager load uses `transformers_class.from_pretrained` with auto dtype or `safetensors.safe_open` with pt.
  # The two methods are interchangeable in most cases.
  # Must use "transformers" for gemma3-4b due to mapping compatibility.
  # Must use "safetensors" for models without official transformers support, like DeepSeek-V3.2.
  # Must use "safetensors" for weights omitted by transformers class,
  #   like Multi-Token Prediction weights (`layers.61`) in DeepSeek-V3.
  parser.add_argument(
      "--eager_load_method",
      type=str,
      required=False,
      default="transformers",
      choices=["transformers", "safetensors"],
      help="Backend to use for eager loading: `transformers_class.from_pretrained` or `safetensors.safe_open` with pt",
  )
  # If not specified, default to maxtext.utils.globals.HF_IDS[model_name]
  parser.add_argument(
      "--hf_model_path",
      type=str,
      required=False,
      default=None,
      help="Customized remote HF repo, or local path to HF model",
  )
  # If hf_model_path is set to a local path, this is ignored.
  parser.add_argument(
      "--revision",
      type=str,
      required=False,
      default=None,
      help="Specific Hugging Face revision (branch/tag/commit)",
  )
  parser.add_argument(
      "--save_dtype",
      type=str,
      required=False,
      default="bfloat16",
      choices=["float32", "bfloat16"],
      help="Save MaxText weights in specified dtype",
  )
  # Determines the logical sharding of the output checkpoint by partitioning
  # weights across virtual XLA devices.
  # - Even on a single CPU host, JAX can simulate multiple devices (e.g., 16)
  # - If set to 1, sharding is skipped.
  # - Sharding is preferred. For downstream loading on TPU pods, this helps prevent OOM and speedup.
  #
  # Example: Embedding Layer shape=(151936, 1024)
  # Case 1: simulated_cpu_devices_count=16 (Sharded)
  #   sharding: NamedShardingMetadata(shape=[16], ...)
  #   storage:  chunk_shape=(9496, 1024)  <-- 1/16th of rows per chunk
  # Case 2: simulated_cpu_devices_count=1 (Monolith)
  #   sharding: None
  #   storage:  chunk_shape=(151936, 1024) <-- Full layer in one chunk
  parser.add_argument(
      "--simulated_cpu_devices_count", type=int, required=False, default=16, help="Sharding of checkpoint"
  )
  # Parse local arguments
  # Parse known args returns the namespace AND the list of remaining arguments
  local_args, remaining_args = parser.parse_known_args()
  # Reconstruct model_args (script name + the args MaxText needs)
  model_args = [sys.argv[0]] + remaining_args

  # Set jax environment
  jax.config.update("jax_platforms", "cpu")
  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={local_args.simulated_cpu_devices_count}"
  main(
      args=model_args,
      lazy_load_tensors=local_args.lazy_load_tensors,
      eager_load_method=local_args.eager_load_method,
      hf_model_path=local_args.hf_model_path,
      revision=local_args.revision,
      save_dtype=local_args.save_dtype,
      simulated_cpu_devices_count=local_args.simulated_cpu_devices_count,
  )
\n"""


# File: src/maxtext/checkpoint_conversion/to_huggingface.py (commit 313890777)
TO_HF_CONVERTER_RAW = """\n# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

\"\"\"Converts a MaxText checkpoint to a HuggingFace-compatible model checkpoint.

It is invoked using MaxText's pyconfig, which means you provide a base config
file and can override parameters on the command line.

Key Parameters (to be set in the config file or as command-line overrides):
  model_name: (Required) The name of the model to convert (e.g., "gemma2-2b").
              Must be a key in `maxtext.utils.globals.HF_IDS`.
  load_parameters_path: (Required) Path to the MaxText checkpoint directory
                        containing the parameter-only checkpoint.
  base_output_directory: (Optional) The directory where the converted HuggingFace
                         checkpoint will be saved. Can be a local path, a GCS
                         path (gs://...), or a HuggingFace Hub repo ID (hf://...).
                         Defaults to "./mt_output/".
  scan_layers: (bool) Whether the MaxText model was trained with scanned layers.
               This must match the training configuration of the checkpoint.
  weight_dtype: (Optional) It affects the resulting Hugging Face weight dtype.
                Default value is `float32`. We recommend using `bfloat16`
                to save memory and speed up conversion.

Optional Flags:
  --override_model_architecture: If set, overrides the HF model configuration
                                 with values from the MaxText configuration
                                 (e.g., num_heads, hidden_size) instead of failing.

Environment Variables:
  HF_AUTH_TOKEN: (Required) A HuggingFace authentication token. This is needed
                 to download the correct tokenizer configuration and to upload
                 the converted model to the HuggingFace Hub if `base_output_directory`
                 is a Hub repo ID (e.g., "hf://my-user/my-model").

Example Usage:
  To convert a gemma2-2b MaxText checkpoint and save it to a local directory:

  export HF_AUTH_TOKEN="hf_YOUR_TOKEN"
  python src/maxtext/checkpoint_conversion/to_huggingface.py \
    src/maxtext/configs/base.yml \
    model_name="gemma2-2b" \
    load_parameters_path="/path/to/your/maxtext/checkpoint/" \
    base_output_directory="/path/to/your/output/directory" \
    scan_layers=False

  Note: Other parameters in base.yml (like per_device_batch_size, max_target_length, etc.)
  are used to initialize the model structure and should be consistent with the
  checkpoint being converted, but often don't need to be changed from their defaults.
\"\"\"

import jax
import os
from typing import Sequence
import time

from transformers import AutoTokenizer, AutoProcessor

from absl import app
from absl import flags

from maxtext.configs import pyconfig
from maxtext.checkpoint_conversion.utils.param_mapping import (
    HOOK_FNS,
    PARAM_MAPPING,
)
from maxtext.checkpoint_conversion.utils.hf_shape import HF_SHAPE
from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
from maxtext.checkpoint_conversion.utils.utils import (
    validate_and_filter_param_map_keys,
    process_maxtext_param,
    save_model_files,
    load_orbax_checkpoint,
    detect_and_extract_checkpoint,
    MemoryMonitorTqdm,
    print_peak_memory,
)
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils.globals import HF_IDS


flags.DEFINE_bool(
    "override_model_architecture",
    False,
    "If True, overrides Hugging Face config architecture parameters (heads, layers, dims) "
    "with values from the MaxText config. If False, raises a ValueError on mismatch.",
)

FLAGS = flags.FLAGS


def _get_model_mappings(
    model_name: str, scan_layers: bool, hf_config_dict: dict, maxtext_config: pyconfig.HyperParameters
):
  \"\"\"Retrieves parameter, shape, and hook function mappings for the model.

  Args:
    model_name: The name of the model (e.g., "gemma2-2b").
    scan_layers: Boolean indicating if the model was trained with scanned layers.
    hf_config_dict: The Hugging Face model configuration dictionary.
    maxtext_config: The maxtext model configuration.

  Returns:
    A dictionary containing the parameter mapping, shape mapping, and hook
    function mapping required for the conversion.

  Raises:
    ValueError: If mappings for the specified `model_name` are not found.
  \"\"\"
  if model_name not in PARAM_MAPPING or model_name not in HF_SHAPE or model_name not in HOOK_FNS:
    raise ValueError(f"Mappings not found for model: {model_name}. Available PARAM_MAPPING keys: {PARAM_MAPPING.keys()}")

  return {
      "param_mapping": PARAM_MAPPING[model_name](hf_config_dict, maxtext_config, scan_layers),
      "shape_mapping": HF_SHAPE[model_name](hf_config_dict),
      "hook_fn_mapping": HOOK_FNS[model_name](hf_config_dict, maxtext_config, scan_layers, saving_to_hf=True),
  }


def _validate_or_update_architecture(hf_config, max_config, override: bool):
  \"\"\"Validates consistency between HF and MaxText configs or overrides HF config if requested.

  Args:
    hf_config: The Hugging Face configuration object.
    max_config: The MaxText configuration object (HyperParameters).
    override: Boolean, if True, update hf_config with max_config values.
              If False, raise error on mismatch.
  \"\"\"
  # Mapping from Hugging Face config attribute -> MaxText config attribute
  # Note: We use derived MaxText attributes (e.g. emb_dim) which account for scale factors.
  attributes_to_check = [
      ("hidden_size", "emb_dim"),
      ("intermediate_size", "mlp_dim"),
      ("kv_lora_rank", "kv_lora_rank"),
      ("moe_intermediate_size", "moe_mlp_dim"),
      ("n_routed_experts", "num_experts"),
      ("n_shared_experts", "shared_experts"),
      ("num_attention_heads", "num_query_heads"),
      ("num_experts", "num_experts"),
      ("num_experts_per_tok", "num_experts_per_tok"),
      ("num_hidden_layers", "num_decoder_layers"),
      ("num_key_value_heads", "num_kv_heads"),
      ("num_local_experts", "num_experts"),
      ("q_lora_rank", "q_lora_rank"),
      ("qk_nope_head_dim", "qk_nope_head_dim"),
      ("qk_rope_head_dim", "qk_rope_head_dim"),
      ("v_head_dim", "v_head_dim"),
      ("vocab_size", "vocab_size"),
  ]

  if max_config.attention_type == "mla":
    attributes_to_check.extend(
        [
            ("qk_nope_head_dim", "qk_nope_head_dim"),
            ("qk_rope_head_dim", "qk_rope_head_dim"),
            ("v_head_dim", "v_head_dim"),
            ("kv_lora_rank", "kv_lora_rank"),
            ("q_lora_rank", "q_lora_rank"),
        ]
    )
  else:
    attributes_to_check.append(("head_dim", "head_dim"))

  mismatches = []

  for hf_attr, mt_attr in attributes_to_check:
    # Skip checks if the HF config doesn't have this attribute (e.g. layer_norm_eps vs rms_norm_eps)
    if not hasattr(hf_config, hf_attr):
      continue

    # Skip checks if MaxText config doesn't have the attribute (shouldn't happen for valid configs)
    if not hasattr(max_config, mt_attr):
      continue

    hf_value = getattr(hf_config, hf_attr)
    mt_value = getattr(max_config, mt_attr)

    # Handle None values
    if hf_value is None or mt_value is None:
      continue

    # Special handling for Gemma 2 where local and global layers are bundled
    if max_config.model_name.startswith("gemma2") and hf_attr == "num_hidden_layers":
      if isinstance(mt_value, int):
        mt_value = mt_value * 2

    # Handle vocab size padding
    if hf_attr == "vocab_size" and isinstance(mt_value, int) and isinstance(hf_value, int):
      # MaxText often pads vocab size to a multiple of 128 or 256 for TPU efficiency
      if mt_value >= hf_value and (mt_value - hf_value) < 256:
        mt_value = hf_value

    # Compare values (with tolerance for floats)
    is_match = False
    if isinstance(hf_value, float) or isinstance(mt_value, float):
      try:
        is_match = abs(float(hf_value) - float(mt_value)) < 1e-6
      except (ValueError, TypeError):
        is_match = hf_value == mt_value
    else:
      is_match = hf_value == mt_value

    if not is_match:
      if override:
        max_logging.log(f"⚠️ Overwriting HF Config '{hf_attr}': {hf_value} -> {mt_value} (from MaxText '{mt_attr}')")
        setattr(hf_config, hf_attr, mt_value)
      else:
        mismatches.append(f"{hf_attr} (HF={hf_value} vs MaxText={mt_value})")

  if mismatches:
    error_msg = (
        "Architecture mismatches detected between standard Hugging Face config and provided MaxText config:\n  - "
        + "\n  - ".join(mismatches)
        + "\n\nAction Required: Pass the flag `--override_model_architecture` to force the conversion using MaxText values."
    )
    raise ValueError(error_msg)


def main(argv: Sequence[str]) -> None:
  \"\"\"Main function to convert a MaxText checkpoint to HuggingFace format.

  This function orchestrates the entire conversion process. It loads the
  MaxText checkpoint, transforms the parameter keys and weights according to
  pre-defined mappings, and saves the resulting model, configuration, and
  tokenizer in a format compatible with the Hugging Face ecosystem.

  Args:
    argv: Command-line arguments, which are parsed by `pyconfig`.
  \"\"\"
  # Initialize maxtext config
  config = pyconfig.initialize(argv)
  assert (
      config.load_full_state_path == ""
  ), "This script expects parameters, not a full state. Use generate_param_only_checkpoint first if needed."
  max_utils.print_system_information()
  overall_start = time.time()

  # Load Maxtext checkpoint using Orbax to get full parameter dict
  max_logging.log(f"\nLoading Orbax checkpoint from: {config.load_parameters_path}")
  start = time.time()
  checkpoint_dict = load_orbax_checkpoint(config)
  max_logging.log(f"Elapse for checkpoint load: {(time.time() - start) / 60:.2f} min")

  # Define output directory
  if not config.base_output_directory:
    output_directory = f"tmp/{config.run_name}"
  else:
    output_directory = config.base_output_directory

  # 1. Get HuggingFace Model Configuration
  model_key = config.model_name
  if model_key not in HF_IDS:
    raise ValueError(f"Unsupported model name: {config.model_name}. Supported models are: {list(HF_IDS.keys())}")
  hf_config_obj = HF_MODEL_CONFIGS[model_key]

  # Validate architecture consistency (raising ValueError on mismatch) or override HF config if specified.
  _validate_or_update_architecture(hf_config_obj, config, override=FLAGS.override_model_architecture)

  # 2. Load Tokenizer
  if model_key not in HF_IDS:
    raise ValueError(f"HF Tokenizer ID not found for model key: {model_key}")
  hf_token = config.hf_access_token
  hf_tokenizer_id = HF_IDS[model_key]
  tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_id, token=hf_token)

  # For multi-modal case:
  processor = AutoProcessor.from_pretrained(hf_tokenizer_id, token=hf_token) if config.use_multimodal else None

  # 3. Get parameter mappings
  mappings = _get_model_mappings(model_key, config.scan_layers, hf_config_obj.to_dict(), config)
  param_map = mappings["param_mapping"]
  shape_map = mappings["shape_mapping"]  # HF target shapes
  hook_fn_map = mappings["hook_fn_mapping"]

  # 4. Extract and transform weights for Linen/NNX-SFT/NNX-RL checkpoints
  maxtext_state_dict = detect_and_extract_checkpoint(checkpoint_dict)

  # Validate that checkpoint keys match the parameter mapping
  filtered_map_keys = validate_and_filter_param_map_keys(param_map.keys(), maxtext_state_dict.keys())

  # Iterate through the parameter map to transform and collect weights.
  # This loop handles both simple 1-to-1 mappings and complex N-to-1 mappings
  # (where multiple MaxText weights are combined into a single HF weight).
  max_logging.log("\nProccessing weight...")
  start = time.time()
  processed_params_list = []

  for key in MemoryMonitorTqdm(filtered_map_keys, total=len(filtered_map_keys), leave=True):
    if isinstance(key, tuple):
      # if key is tuple of param names, weight is list of param weights
      weight = [maxtext_state_dict[subkey] for subkey in key]
    else:
      # if key is single param name, weight is single param weight
      weight = maxtext_state_dict[key]

    processed_params = process_maxtext_param(key, weight, param_map, hook_fn_map, shape_map, config)
    processed_params_list.extend(processed_params)

  max_logging.log(f"Weight dtype after transform: {type(processed_params[0][1].dtype)}")

  transformed_hf_weights = dict(processed_params_list)
  max_logging.log(f"Elapse for transform: {(time.time() - start) / 60:.2f} min")

  # 5. Save in HuggingFace Format
  if not transformed_hf_weights:
    print("Error: No weights were transformed. Check mappings and parameter paths.")
    return

  max_logging.log("\nSaving HuggingFace model...")
  start = time.time()
  save_model_files(
      weight_arrays=transformed_hf_weights,
      config=hf_config_obj,
      tokenizer=tokenizer,
      processor=processor,
      output_dir=output_directory,
  )
  max_logging.log(f"✅ MaxText model successfully saved in HuggingFace format at {output_directory}")
  max_logging.log(f"Elapse for save: {(time.time() - start) / 60:.2f} min")
  max_logging.log(f"Overall Elapse: {(time.time() - overall_start) / 60:.2f} min")
  print_peak_memory()


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  jax.config.update("jax_platforms", "cpu")
  os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

  app.run(main)
\n"""


# File: src/maxtext/utils/globals.py (commit 313890777)
GLOBALS_RAW = """\n# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

\"\"\"Global variable constants used throughout the codebase\"\"\"

import os.path

# This is the maxtext package root (src/maxtext)
# Since this file is at src/maxtext/utils/globals.py, we need to go up 2 levels
MAXTEXT_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# This is the maxtext repo root: with ".git" folder; "README.md"; "pyproject.toml"; &etc.
MAXTEXT_REPO_ROOT = os.environ.get(
    "MAXTEXT_REPO_ROOT",
    r
    if os.path.isdir(
        os.path.join(r := os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), ".git")
    )
    else MAXTEXT_PKG_DIR,
)

# This is the configs root: with "base.yml"; "models/"; &etc.
MAXTEXT_CONFIGS_DIR = os.environ.get("MAXTEXT_CONFIGS_DIR", os.path.join(MAXTEXT_PKG_DIR, "configs"))

# This is the assets root: with "tokenizers/"; &etc.
MAXTEXT_ASSETS_ROOT = os.environ.get("MAXTEXT_ASSETS_ROOT", os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "assets"))

# This is the test assets root: with "golden_logits"; &etc.
MAXTEXT_TEST_ASSETS_ROOT = os.environ.get("MAXTEXT_TEST_ASSETS_ROOT", os.path.join(MAXTEXT_REPO_ROOT, "tests", "assets"))

EPS = 1e-8  # Epsilon to calculate loss
DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = 2 * 1024**3  # Default checkpoint file size

# Mapping from MaxText model key to Hugging Face tokenizer identifiers
HF_IDS = {
    "gemma2-2b": "google/gemma-2-2b",
    "gemma2-9b": "google/gemma-2-9b",
    "gemma2-27b": "google/gemma-2-27b",
    "gemma3-4b": "google/gemma-3-4b-it",  # hf multi-modal should also support the pure-text
    "gemma3-12b": "google/gemma-3-12b-it",
    "gemma3-27b": "google/gemma-3-27b-it",
    "gemma4-26b": "google/gemma-4-26b-a4b-it",
    "gemma4-31b": "google/gemma-4-31b-it",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-4b-thinking-2507": "Qwen/Qwen3-4B-Thinking-2507",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama3.1-8b-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1-70b-Instruct": "meta-llama/Llama-3.1-70B-Instruct",
    "llama3.1-70b": "meta-llama/Llama-3.1-70B",
    "llama3.1-405b": "meta-llama/Llama-3.1-405B",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "qwen3-30b-a3b-base": "Qwen/Qwen3-30B-A3B-Base",
    "qwen3-235b-a22b": "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "qwen3-480b-a35b": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "deepseek2-16b": "deepseek-ai/DeepSeek-V2-Lite",
    "deepseek3-671b": "deepseek-ai/DeepSeek-V3",
    "deepseek3.2-671b": "deepseek-ai/DeepSeek-V3.2",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "qwen3-omni-30b-a3b": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "qwen3-next-80b-a3b": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mixtral-8x22b": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "olmo3-7b": "allenai/Olmo-3-7B-Instruct",
    "olmo3-7b-pt": "allenai/Olmo-3-1025-7B",
    "olmo3-32b": "allenai/Olmo-3-32B-Think",
    # "default" is not HF model, but adding to to avoid confusing warning about tokenizer_path
    "default": os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers/tokenizer.llama2"),
}

__all__ = [
    "DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE",
    "EPS",
    "MAXTEXT_ASSETS_ROOT",
    "MAXTEXT_CONFIGS_DIR",
    "MAXTEXT_PKG_DIR",
    "MAXTEXT_REPO_ROOT",
    "MAXTEXT_TEST_ASSETS_ROOT",
    "HF_IDS",
]
\n"""


# File: src/maxtext/optimizers/optimizers.py (commit 313890777)
OPTIMIZERS_RAW = """\n# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=bare-except, consider-using-generator, too-many-positional-arguments
\"\"\" Utils that are only interesting to MaxText. \"\"\"

import re
import jax
import jax.numpy as jnp

import optax
from optax.contrib._muon import muon
from maxtext.utils.muon_utils import get_muon_weight_dimension_numbers


def _get_path_mask_fn(patterns, match_returns_true=True):
  \"\"\"Helper to create a mask function from a list of regex patterns.\"\"\"
  if not patterns:
    return None

  compiled_patterns = [re.compile(pattern) for pattern in patterns]

  def mask_fn(params):
    def _is_masked(path, _):
      # Join path keys into a single string for pattern matching (e.g., "layer1/bias")
      path_str = jax.tree_util.keystr(path, simple=True, separator="/")
      matched = any(pattern.search(path_str) for pattern in compiled_patterns)
      return matched if match_returns_true else not matched

    return jax.tree_util.tree_map_with_path(_is_masked, params)

  return mask_fn


def get_adamw_mask(config):
  \"\"\"Create a mask function for AdamW optimizer to exclude certain parameters from weight decay.\"\"\"
  return _get_path_mask_fn(getattr(config, "adamw_mask", None), match_returns_true=False)


def _compute_rolling_stats(arr: jax.Array, count: jax.Array, interval: int):
  \"\"\"Computes mean and unbiased std (Bessel's correction) over a rolling window.\"\"\"
  valid_elements = jnp.minimum(count, interval)
  safe_elements = jnp.maximum(1, valid_elements)
  mask = jnp.arange(interval) < valid_elements

  mean = jnp.sum(jnp.where(mask, arr, 0.0)) / safe_elements
  sq_diff = jnp.where(mask, (arr - mean) ** 2, 0.0)

  # Use Bessel's correction (N - 1) for unbiased variance to align with torch.std
  variance = jnp.sum(sq_diff) / jnp.maximum(1, valid_elements - 1)
  std = jnp.sqrt(variance)
  return mean, std


def skip_step_on_spikes(
    inner_opt: optax.GradientTransformation, interval: int, scaling_factor: float
) -> optax.GradientTransformationExtraArgs:
  \"\"\"Wrapper that skips updates when loss or grad_norm spike.

  This wrapper calculates a rolling mean and standard deviation (using
  Bessel's correction) over the last `interval` steps for both the loss
  and the gradient norm. If the current step's loss or gradient norm
  exceeds `mean + scaling_factor * std`, the update is zeroed and the
  optimizer state is not advanced, effectively skipping the step.

  Reference implementation:
  https://github.com/allenai/OLMo-core/blob/c757b7c3c15197154c753d883330afbfa4869dcc/src/olmo_core/optim/skip_step_optimizer.py#L12

  Args:
    inner_opt: The inner Optax gradient transformation to wrap.
    interval: The number of recent steps to use for calculating mean and std.
    scaling_factor: The multiplier for standard deviation to set the spike threshold.

  Returns:
    An optax.GradientTransformationExtraArgs that skips spikes.
  \"\"\"

  def init_fn(params):
    return {
        "inner_state": inner_opt.init(params),
        "losses": jnp.zeros(interval, dtype=jnp.float32),
        "grad_norms": jnp.zeros(interval, dtype=jnp.float32),
        "count": jnp.zeros((), dtype=jnp.int32),
        "is_skipped": jnp.array(False, dtype=jnp.bool_),
    }

  def update_fn(updates, state, params=None, **extra_args):
    # Using `pop()` removes `loss` and `grad_norm` from `extra_args` before they are
    # passed downstream to `inner_opt.update()`. This prevents `TypeError` if the
    # inner optimizer doesn't explicitly accept these as `kwargs`.
    loss = extra_args.pop("loss", None)
    grad_norm = extra_args.pop("grad_norm", None)

    # Fallback to standard update if loss is not provided
    if loss is None:
      inner_updates, new_inner_state = inner_opt.update(updates, state["inner_state"], params, **extra_args)
      return inner_updates, {
          "inner_state": new_inner_state,
          "losses": state["losses"],
          "grad_norms": state["grad_norms"],
          "count": state["count"],
          "is_skipped": jnp.array(False, dtype=jnp.bool_),
      }

    count = state["count"]
    losses = state["losses"]
    grad_norms = state["grad_norms"]

    # Compute rolling stats
    loss_mean, loss_std = _compute_rolling_stats(losses, count, interval)
    grad_norm_mean, grad_norm_std = _compute_rolling_stats(grad_norms, count, interval)

    # Check if the current metrics are within the allowed thresholds
    is_loss_ok = (loss - loss_mean) <= scaling_factor * loss_std
    if grad_norm is not None:
      is_grad_norm_ok = (grad_norm - grad_norm_mean) <= scaling_factor * grad_norm_std
      is_ok = jnp.logical_and(is_loss_ok, is_grad_norm_ok)
    else:
      is_ok = is_loss_ok

    # Only enforce skip if we have at least half the interval filled (or 2 elements minimum)
    min_history = max(2, interval // 2)
    is_warmup = (count + 1) < min_history
    is_ok = jnp.logical_or(is_warmup, is_ok)

    # Conditionally execute the inner optimizer to prevent momentum poisoning
    def do_update():
      return inner_opt.update(updates, state["inner_state"], params, **extra_args)

    def skip_update():
      # b/500923599: Investigate logging compatible with jax.jit, jax.lax.cond, and Pathway
      inner_updates = jax.tree_util.tree_map(jnp.zeros_like, updates)
      return inner_updates, state["inner_state"]

    inner_updates, new_inner_state = jax.lax.cond(is_ok, do_update, skip_update)

    # Update rolling buffers (append even if skipped so spikes can become the new baseline)
    idx = count % interval
    new_losses = losses.at[idx].set(loss)

    new_grad_norms = grad_norms
    if grad_norm is not None:
      new_grad_norms = grad_norms.at[idx].set(grad_norm)

    new_state = {
        "inner_state": new_inner_state,
        "losses": new_losses,
        "grad_norms": new_grad_norms,
        "count": count + 1,
        "is_skipped": jnp.logical_not(is_ok),
    }
    return inner_updates, new_state

  return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def get_optimizer(config, learning_rate_schedule, model=None):
  \"\"\"Create optimizer.\"\"\"
  if config.opt_type == "adamw":
    # Create AdamW Optimizer following Llama2's training details, see https://arxiv.org/pdf/2307.09288.pdf section 2.2
    base_opt = optax.adamw(
        learning_rate_schedule,
        b1=config.adam_b1,
        b2=config.adam_b2,
        eps=config.adam_eps,
        eps_root=config.adam_eps_root,
        weight_decay=config.adam_weight_decay,
        mu_dtype=config.mu_dtype,
        mask=get_adamw_mask(config),
    )
  elif config.opt_type == "adam_pax":
    base_opt = adam_pax(
        learning_rate_schedule,
        beta1=config.adam_b1,
        beta2=config.adam_b2,
        epsilon=config.adam_eps,
        epsilon_root=config.adam_eps_root,
        weight_decay=config.adam_weight_decay,
        mask=get_adamw_mask(config),
    )
  elif config.opt_type == "sgd":
    base_opt = optax.sgd(learning_rate_schedule)
  elif config.opt_type == "muon":
    # extract muon dimension number from model structure
    if model is not None:
      muon_weight_dimension_numbers = get_muon_weight_dimension_numbers(model, config)
    else:
      raise ValueError("Please specify model to extract muon dimension number.")
    muon_kwargs = {
        # Shared parameters: "nesterov" uses default
        "learning_rate": learning_rate_schedule,
        "eps": config.adam_eps,
        "mu_dtype": config.mu_dtype,
        # Muon-specific parameters: "ns_coeffs", "ns_steps", "weight_decay_mask", "adaptive" uses default
        "beta": config.muon_beta,
        "weight_decay": config.muon_weight_decay,
        "muon_weight_dimension_numbers": muon_weight_dimension_numbers,
        "consistent_rms": config.muon_consistent_rms,
        # AdamW-specific parameters
        "adam_b1": config.adam_b1,
        "adam_b2": config.adam_b2,
        "adam_eps_root": config.adam_eps_root,
        "adam_weight_decay": config.adam_weight_decay,
    }
    base_opt = muon(**muon_kwargs)
  else:
    raise ValueError(f"{config.opt_type=} is not a supported.")

  if getattr(config, "skip_step_on_spikes", False):
    base_opt = skip_step_on_spikes(
        base_opt,
        interval=config.skip_step_interval,
        scaling_factor=config.skip_step_scaling_factor,
    )

  # If a whitelist of trainable parameters is provided, freeze everything else.
  # When trainable_parameters_mask is empty, freeze_mask_fn is None and all parameters are trained.
  trainable_patterns = getattr(config, "trainable_parameters_mask", None)
  freeze_mask_fn = _get_path_mask_fn(trainable_patterns, match_returns_true=False)
  if freeze_mask_fn is not None:
    # Use optax.multi_transform to explicitly map frozen parameters to a stateless set_to_zero() optimizer.
    # If we simply wrapped base_opt in optax.masked() or chained it, Optax would still allocate
    # massive states (momentum, variance) for the entire model before zeroing the updates.
    # By using multi_transform, only the trainable parameters get states allocated.
    return optax.multi_transform(
        {"trainable": base_opt, "frozen": optax.set_to_zero()},
        lambda params: jax.tree_util.tree_map(lambda x: "frozen" if x else "trainable", freeze_mask_fn(params)),
    )

  return base_opt


def adam_pax(
    learning_rate_fn: optax.Schedule,
    beta1: float,
    beta2: float,
    epsilon: float,
    epsilon_root: float,
    weight_decay: float,
    mask=None,
) -> optax.GradientTransformation:
  \"\"\"Standard Adam optimizer that supports weight decay.

  Follows the implementation in pax/praxis sharded_adam
  https://github.com/google/praxis/blob/545e00ab126b823265d70c715950d39333484f38/praxis/optimizers.py#L621

  Args:
    learning_rate_fn: a callable that given the current training step, returns
      the learning rate to apply.
    beta1: decay rate to track the first moment.
    beta2: decay rate to track the second moment.
    epsilon: Small constant applied to the denominator outside of the square
      root to avoid dividing by zero when rescaling.
    epsilon_root: Small constant applied to the denominator inside of the square
      root to avoid dividing by zero when rescaling.
    weight_decay: If > 0, weight decay to apply.

  Returns:
    A `optax.GradientTransformation`.
  \"\"\"

  def init_fn(params):
    mu = jax.tree_util.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
    return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def bias_corrected_decay(step: jnp.int32, decay: float):
    \"\"\"Incorporates bias correction into decay.

    Please see section 7.1 in https://arxiv.org/pdf/1804.04235.pdf for the
    derivation of the formulas below. With bias-corrected decay, we can simply
    do

    m_{t} = decay1 * m_{t-1} + (1 - decay1) * g
    v_{t} = decay2 * v_{t-1} + (1 - decay2) * g ^ 2

    without further bias correction.

    Args:
      step: current step, 0-based.
      decay: the raw decay. As t -> infinity, bias corrected decay converges to
        this value.

    Returns:
      Bias corrected decay.
    \"\"\"
    t = step.astype(jnp.float32) + 1.0
    return decay * (1.0 - jnp.power(decay, t - 1.0)) / (1.0 - jnp.power(decay, t))

  def update_fn(updates, state, params=None):
    # Sanitize updates just in case.
    if weight_decay > 0:
      assert params is not None
    count = state.count

    class _slot_opt_state:

      def __init__(self, mu, nu):
        self.mu = mu
        self.nu = nu

    def _update_momentum(update, mu, nu):
      # The conversion to the data type of the update ensures that bfloat16 remains
      # bfloat16 in the optimizer state. This conversion has to be done after
      # `bias_corrected_dacay` is calculated as calculating `jnp.power(decay, t)` in low
      # precision can result in it being rounded to 1 and subsequently a
      # "division by zero" error.
      beta1_decay = bias_corrected_decay(count, beta1).astype(update.dtype)
      beta2_decay = bias_corrected_decay(count, beta2).astype(update.dtype)
      mu = (1.0 - beta1_decay) * update + beta1_decay * mu
      nu = (1.0 - beta2_decay) * (update**2) + beta2_decay * nu
      return _slot_opt_state(mu=mu, nu=nu)

    updated_moments = jax.tree_util.tree_map(_update_momentum, updates, state.mu, state.nu)

    mu = jax.tree_util.tree_map(lambda x: x.mu, updated_moments)
    nu = jax.tree_util.tree_map(lambda x: x.nu, updated_moments)

    updates = jax.tree_util.tree_map(lambda mu, nu: mu / (jnp.sqrt(nu + epsilon_root) + epsilon), mu, nu)

    if weight_decay > 0:
      if mask is not None:
        mask_tree = mask(params) if callable(mask) else mask
        updates = jax.tree_util.tree_map(lambda x, v, m: x + weight_decay * v if m else x, updates, params, mask_tree)
      else:
        updates = jax.tree_util.tree_map(lambda x, v: x + weight_decay * v, updates, params)

    step_size = -1.0 * learning_rate_fn(count)
    # Finally, fold in step size.
    updates = jax.tree_util.tree_map(lambda x: step_size * x, updates)

    updated_states = optax.ScaleByAdamState(count=count + 1, mu=mu, nu=nu)
    return updates, updated_states

  return optax.GradientTransformation(init_fn, update_fn)
\n"""


# File: src/maxtext/trainers/pre_train/train.py (commit 313890777)
TRAIN_ENTRYPOINT_RAW = """\n# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=g-bad-todo, abstract-method, consider-using-with
\"\"\"Training loop and Decoding of the model.\"\"\"

# Calling jax.device_count here prevents a "TPU platform already registered" error.
# See github.com/google/maxtext/issues/20 for more

from typing import Any, Sequence
import contextlib
import datetime
import functools
import os

from absl import app

import numpy as np
import optax

import pathwaysutils  # pylint: disable=unused-import

import tensorflow as tf

import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.linen import partitioning as nn_partitioning

from maxtext.configs import pyconfig
from maxtext.utils.globals import EPS
from maxtext.utils import elastic_utils
# Placeholder: internal

# pylint: disable=too-many-positional-arguments
from maxtext.layers.multi_token_prediction import calculate_mtp_acceptance_rate, calculate_mtp_loss
from maxtext.common import checkpointing, profiler
from maxtext.common.goodput import (
    GoodputEvent,
    RECORD_JOB_END_TIME,
    RECORD_JOB_START_TIME,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
    record_goodput,
)
from maxtext.common.gcloud_stub import cloud_diagnostics as _cloud_diag, is_decoupled
from maxtext.common.gcloud_stub import vertex_tensorboard_modules
from maxtext.common.metric_logger import MetricLogger, record_activation_metrics
from maxtext.trainers.post_train.dpo.dpo_utils import _merge_dpo_state, _split_dpo_state, dpo_loss_fn
from maxtext.utils import exceptions
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import qk_clip_utils
from maxtext.utils import sharding
from maxtext.utils import train_utils
from maxtext.utils.gradient_accumulation import gradient_accumulation_loss_and_grad
from maxtext.utils.vocabulary_tiling import vocab_tiling_linen_loss

_diag_modules = _cloud_diag()
diagnostic, debug_configuration, diagnostic_configuration, stack_trace_configuration = _diag_modules
VertexTensorboardManager, _vertex_tb_is_stub = vertex_tensorboard_modules()


def get_first_step(model, state):
  if isinstance(model, nn.Module):
    return int(state.step)
  return int(state.optimizer.step.get_value())


# -----------------------------------------------------------------------------
# Top-level Functions
# -----------------------------------------------------------------------------


def loss_fn(model, config, data, dropout_rng, params, sparsity_state=None, is_train=True):
  \"\"\"loss_fn for both train and eval.

  Args:
    model: A nn.Module
    config: Config of parameters
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout
    params: Model params
    is_train: True for train_step and False for eval_step

  Returns:
    loss: average loss
    aux: a dictionary including intermediate_outputs, xent_sum, and total_weights
  \"\"\"
  # decimate proportion of data when per_device_batch_size<1
  if is_train:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_train_on, :]
  else:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_eval_on, :]
  mutable_collections = ["intermediates"]
  if config.mtp_num_layers > 0 and is_train:
    # The single model.apply call now triggers the entire chain if MTP is enabled:
    # Decoder runs -> returns hidden_state -> MTPBlock uses it -> MTPBlock sows losses -> we reap them here.
    mutable_collections.append("mtp_losses")

  # During evaluation, if the acceptance rate test is enabled, we must
  # make its specific collection mutable so the MTPBlock can sow into it.
  if config.mtp_eval_target_module > 0 and not is_train:
    mutable_collections.append("mtp_acceptance")
  sparsity_enabled = is_train and config.weight_sparsity_n and config.weight_sparsity_m
  if sparsity_enabled:
    mutable_collections.append("batch_stats")
  if isinstance(model, nn.Module):
    # inputs, targets, segments, positions = apply_args
    rng1, aqt_rng = jax.random.split(dropout_rng)

    # Flax Linen model
    if sparsity_enabled:
      model_vars = {"params": params}
      if sparsity_state:
        model_vars["batch_stats"] = sparsity_state
    else:
      model_vars = params
    logits, intermediate_outputs = model.apply(
        model_vars,
        data["inputs"],
        data["inputs_position"],
        decoder_segment_ids=data["inputs_segmentation"],
        encoder_images=data["images"] if config.use_multimodal else None,
        encoder_image_masks=data["image_masks"] if config.use_multimodal and "image_masks" in data else None,
        enable_dropout=config.enable_dropout if is_train else False,
        rngs={"dropout": rng1, "params": aqt_rng},
        mutable=mutable_collections,
        decoder_target_tokens=data["targets"],
        decoder_target_mask=data["targets_segmentation"],
    )

    if (config.use_indexer and not config.indexer_sparse_training) and is_train:
      # In Dense Warm-up stage, we skip main model loss calculation for efficiency.
      # The main model parameters are frozen and only the indexer is trained via KL divergence.
      xent_sum = 0.0
      total_z_loss = 0.0
    elif config.num_vocab_tiling > 1:
      hidden_state_key = ("intermediates", "decoder", "hidden_states")
      hidden_states = maxtext_utils.get_nested_value(intermediate_outputs, hidden_state_key)[0]
      xent_sum, total_z_loss = vocab_tiling_linen_loss(hidden_states, data, config, model, params, is_train)
    else:
      one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
      xent, z_loss = max_utils.cross_entropy_with_logits(logits, one_hot_targets, z_loss=config.z_loss_multiplier)

      xent = sharding.maybe_shard_with_logical(
          xent,
          ("activation_embed_and_logits_batch", "activation_length"),
          model.mesh,
          config.shard_mode,
          debug_sharding=config.debug_sharding,
      )
      z_loss = sharding.maybe_shard_with_logical(
          z_loss,
          ("activation_embed_and_logits_batch", "activation_length"),
          model.mesh,
          config.shard_mode,
          debug_sharding=config.debug_sharding,
      )

      # Mask out paddings at the end of each example.
      xent = xent * (data["targets_segmentation"] != 0)
      z_loss = z_loss * (data["targets_segmentation"] != 0)

      xent_sum = jnp.sum(xent)
      total_z_loss = jnp.sum(z_loss)
  else:
    # Flax NNX model
    logits = model(
        decoder_input_tokens=data["inputs"],
        decoder_positions=data["inputs_position"],
        decoder_segment_ids=data["inputs_segmentation"],
        encoder_images=data["images"] if config.use_multimodal else None,
        encoder_image_masks=data["image_masks"] if config.use_multimodal and "image_masks" in data else None,
        enable_dropout=config.enable_dropout if is_train else False,
        decoder_target_tokens=data["targets"],
        decoder_target_mask=data["targets_segmentation"],
    )
    intermediate_outputs = {}

    if (config.use_indexer and not config.indexer_sparse_training) and is_train:
      # In Dense Warm-up stage, we skip main model loss calculation for efficiency.
      # The main model parameters are frozen and only the indexer is trained via KL divergence.
      xent_sum = 0.0
      total_z_loss = 0.0
    else:
      one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
      xent, z_loss = max_utils.cross_entropy_with_logits(logits, one_hot_targets, z_loss=config.z_loss_multiplier)

      xent = nn.with_logical_constraint(xent, ("activation_embed_and_logits_batch", "activation_length"))
      z_loss = nn.with_logical_constraint(z_loss, ("activation_embed_and_logits_batch", "activation_length"))

      # Mask out paddings at the end of each example.
      xent = xent * (data["targets_segmentation"] != 0)
      z_loss = z_loss * (data["targets_segmentation"] != 0)

      xent_sum = jnp.sum(xent)
      total_z_loss = jnp.sum(z_loss)

  total_weights = jnp.sum(data["targets_segmentation"] != 0)
  # If gradient accumulation is enabled, we don't need to divide xent_sum
  # by total_weights and then multiply the computed gradient by total_weights,
  # since it's equivalent to computing the gradient from xent_sum.
  # This simplification reduces the number of operations and makes it easier
  # for XLA to move all-reduce out of the gradient accumulation loop when use
  # Zero1+GA to reduce communication overhead.
  # EPS was used to avoid division by zero, but it's not needed when gradient
  # accumulation is enabled since there's no division.
  if config.gradient_accumulation_steps > 1 and not config.use_tunix_gradient_accumulation:
    loss = xent_sum
  else:
    # When using Tunix gradient accumulation, we revert to standard normalization.
    # Unlike the manual accumulation path above, Tunix (via optax.MultiSteps) expects
    # a normalized loss for each step. It handles the accumulation state
    # updates and scaling internally.
    loss = xent_sum / (total_weights + EPS)

  # We keep z-loss normalized by total_weights.
  total_z_loss = total_z_loss / (total_weights + EPS)

  # Calculate and Add MTP Loss
  mtp_loss = 0.0
  if config.mtp_num_layers > 0 and is_train:
    mtp_loss = calculate_mtp_loss(intermediate_outputs, config)
    loss += mtp_loss

  # get indexer loss
  indexer_loss = 0.0
  if config.use_indexer and config.indexer_loss_scaling_factor > 0.0:
    indexer_losses = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "self_attention", "indexer_loss")
    if indexer_losses:
      indexer_loss = jnp.mean(jnp.concatenate(indexer_losses))
      loss += indexer_loss
    else:
      max_logging.debug("No indexer loss found.")

  # get MoE load balance loss
  moe_lb_loss = 0.0
  if config.num_experts > 1:
    moe_lb_losses = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "moe_lb_loss")
    if moe_lb_losses:
      moe_lb_loss = jnp.mean(jnp.concatenate(moe_lb_losses))
      loss += moe_lb_loss
    else:
      max_logging.debug("\nNo MoE load balance loss found. Defaulting to 0.0.")

  # get MoE routed bias term updates
  moe_bias_updates = None
  if config.routed_bias and config.routed_bias_update_rate > 0.0:
    nested_key = ("intermediates", "decoder", "moe_layers", "moe_bias_updates")
    moe_bias_updates = maxtext_utils.get_nested_value(intermediate_outputs, nested_key, None)

  # Add the model's primary output to the intermediates dict so it can be used
  # by the acceptance rate calculation in eval_step.
  intermediate_outputs["logits"] = logits

  aux = {
      "intermediate_outputs": intermediate_outputs,
      "xent_sum": xent_sum,
      "z_loss": total_z_loss,
      "total_weights": total_weights,
      "moe_lb_loss": moe_lb_loss,
      "indexer_loss": indexer_loss,
      "moe_bias_updates": moe_bias_updates,
      "mtp_loss": mtp_loss,
      "batch_stats": (intermediate_outputs.get("batch_stats", None) if hasattr(intermediate_outputs, "get") else None),
  }
  return loss, aux


def train_step(model, config, state_mesh_shardings, params_shardings, state, data, dropout_rng):
  \"\"\"

  Args:
    model: A nn.Module
    state: A pytree of the current state of the model
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout

  Returns:
    new_state: Same format as state.
    metrics: Dictionary of model metrics such as loss, training rate, etc.
    rng2: A new rng key that can be used in future calls.

  \"\"\"
  reference_params, reference_params_sharding, extra_dpo_args, _loss_fn = (
      [],
      [],
      [],
      loss_fn,
  )
  if config.use_dpo:
    state, reference_params = _split_dpo_state(state)
    state_mesh_shardings, reference_params_sharding = _split_dpo_state(state_mesh_shardings)
    extra_dpo_args = [reference_params]
    _loss_fn = dpo_loss_fn

  params = state.params
  if config.gradient_accumulation_steps > 1:
    loss, aux, raw_grads = gradient_accumulation_loss_and_grad(
        _loss_fn,
        config,
        model,
        params,
        params_shardings,
        data,
        dropout_rng,
        extra_dpo_args,
    )
  else:
    if config.optimizer_memory_host_offload:
      if config.use_dpo:
        reference_params = jax.device_put(
            reference_params,
            max_utils.with_memory_kind(reference_params_sharding, "device"),
        )
        extra_dpo_args = [reference_params]
    if config.shard_optimizer_over_data:
      params = jax.tree.map(
          functools.partial(sharding.maybe_shard_with_name, shard_mode=config.shard_mode),
          params,
          params_shardings,
      )
    sparsity_enabled = config.weight_sparsity_n and config.weight_sparsity_m
    pure_params = params["params"] if sparsity_enabled else params
    batch_stats = params.get("batch_stats", {})

    grad_func = jax.value_and_grad(_loss_fn, argnums=4, has_aux=True)
    (loss, aux), raw_grads = grad_func(
        model,
        config,
        data,
        dropout_rng,
        pure_params,
        *extra_dpo_args,
        sparsity_state=batch_stats,
        is_train=True,
    )

  raw_grads = jax.tree_util.tree_map(
      lambda x: x.astype(config.grad_dtype) if x.dtype == jnp.float32 else x,
      raw_grads,
  )
  if config.parameter_memory_host_offload:
    raw_grads = jax.device_put(
        raw_grads,
        max_utils.with_memory_kind(params_shardings, "device"),
    )
  intermediate_outputs = aux["intermediate_outputs"]
  xent_sum = aux["xent_sum"]
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]
  indexer_loss = aux.get("indexer_loss", 0.0)
  z_loss = aux.get("z_loss", 0.0)
  moe_bias_updates = aux.get("moe_bias_updates")
  mtp_loss = aux.get("mtp_loss", 0.0)

  if config.gradient_clipping_threshold > 0:
    grads = maxtext_utils.apply_gradient_clipping(raw_grads, state, config.gradient_clipping_threshold)
  else:
    grads = raw_grads

  if config.optimizer_memory_host_offload:
    state = state.replace(
        opt_state=jax.device_put(
            state.opt_state,
            jax.tree_util.tree_map(
                lambda x: x.with_memory_kind(kind="device"),
                state_mesh_shardings.opt_state,
            ),
        )
    )
  # Move all parameters to device before optimizer update
  if config.parameter_memory_host_offload:
    max_logging.log("\nMoving all parameters to device before optimizer update")

    def move(path, value):
      max_logging.log(f"train.py: Moving f{path} to device")
      return value.with_memory_kind(kind="device")

    state = state.replace(
        params=jax.device_put(
            state.params,
            jax.tree_util.tree_map_with_path(move, state_mesh_shardings.params),
        )
    )
  # Re-wrap grads to match state.params structure if it's a dict of collections
  sparsity_enabled = config.weight_sparsity_n and config.weight_sparsity_m
  if sparsity_enabled:
    full_grads = {"params": grads}
    if sparsity_enabled and "batch_stats" in state.params:
      batch_stats_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params.get("batch_stats", {}))
      full_grads["batch_stats"] = batch_stats_grads
    full_grads = max_utils.unbox_logicallypartioned(full_grads)
  else:
    full_grads = grads

  if getattr(config, "skip_step_on_spikes", False):
    grad_norm = max_utils.l2norm_pytree(grads)
    # TrainState.apply_gradients doesn't pass **kwargs to tx.update, so we unpack it manually.
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params, loss=loss, grad_norm=grad_norm)
    new_params = optax.apply_updates(state.params, updates)

    new_state = state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
    )
  else:
    new_state = state.apply_gradients(grads=full_grads)

  # Apply updates for Auxiliary-Loss-Free load balancing for DeepSeek family
  if config.routed_bias and config.routed_bias_update_rate > 0.0 and moe_bias_updates is not None:
    target_path = ("params", "decoder", "moe_layers", "DeepSeekMoeBlock_0", "MoeBlock_0", "gate", "bias")
    # Flax 'sow' returns a tuple, so we take the first element [0].
    # Updates the shape to be aligned with state.
    moe_bias_updates = jnp.array(moe_bias_updates[0]).transpose()
    new_state = maxtext_utils.update_state_param(new_state, target_path, moe_bias_updates)

  lm_loss = xent_sum / (total_weights + EPS)
  scalar_metrics = {
      "learning/loss": loss,
      "learning/lm_loss": lm_loss,
      "learning/perplexity": jnp.exp(lm_loss),
      "learning/z_loss": z_loss,
      "learning/moe_lb_loss": moe_lb_loss,
      "learning/indexer_loss": indexer_loss,
      "learning/mtp_loss": mtp_loss,
      "learning/total_weights": total_weights,
  }
  if config.use_qk_clip:
    # Apply QK-Clip
    new_state = qk_clip_utils.apply_qk_clip(new_state, intermediate_outputs, config)

    # Report max_logits metric
    global_max_logit = qk_clip_utils.calculate_max_logit_metric(intermediate_outputs)
    if global_max_logit is not None:
      scalar_metrics["learning/max_logits"] = global_max_logit

  if not config.optimizer_memory_host_offload:
    scalar_metrics["learning/grad_norm"] = max_utils.l2norm_pytree(grads)
    scalar_metrics["learning/raw_grad_norm"] = max_utils.l2norm_pytree(raw_grads)
    scalar_metrics["learning/param_norm"] = max_utils.l2norm_pytree(new_state.params)
  if config.use_dpo:
    scalar_metrics["learning/dpo_loss"] = aux["dpo_loss"]
    scalar_metrics["learning/dpo_reward_accuracy"] = aux["reward_accuracy"]
  metrics = {
      "scalar": scalar_metrics,
      "scalars": {},
  }

  if config.record_internal_nn_metrics:
    record_activation_metrics(metrics, intermediate_outputs, config)

  if config.use_dpo:
    new_state = _merge_dpo_state(new_state, reference_params)

  return new_state, metrics


def eval_step(model, config, state, data, dropout_rng):
  \"\"\"eval_step no backprop and new state compared with train_step.\"\"\"

  reference_params, extra_dpo_args, _loss_fn = [], [], loss_fn
  if config.use_dpo:
    state, reference_params = _split_dpo_state(state)
    extra_dpo_args = [reference_params]
    _loss_fn = dpo_loss_fn

  sparsity_enabled = config.weight_sparsity_n and config.weight_sparsity_m
  pure_params = state.params["params"] if sparsity_enabled else state.params
  batch_stats = state.params.get("batch_stats", {})

  eval_loss_fn = functools.partial(_loss_fn, model, config, data, dropout_rng, is_train=False)
  loss, aux = eval_loss_fn(pure_params, *extra_dpo_args, sparsity_state=batch_stats)

  mtp_acceptance_rate = 0.0
  if config.mtp_eval_target_module > 0:
    mtp_acceptance_rate = calculate_mtp_acceptance_rate(aux["intermediate_outputs"], config)

  xent_sum = aux["xent_sum"]
  z_loss = aux.get("z_loss", 0.0)
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]
  indexer_loss = aux.get("indexer_loss", 0.0)
  mtp_loss = aux.get("mtp_loss", 0.0)
  # For DPO, report the unnormalized sum of per-sample preference losses so that
  # MetricLogger (which divides eval/total_loss by eval/total_weights) recovers
  # the correct mean DPO loss. xent_sum is always 0 for DPO and must not be used.
  eval_total_loss = aux["dpo_loss"] * total_weights if config.use_dpo else xent_sum
  metrics = {
      "scalar": {
          "evaluation/loss": loss,
          "evaluation/z_loss": z_loss,
          "evaluation/total_loss": eval_total_loss,
          "evaluation/total_weights": total_weights,
          "evaluation/moe_lb_loss": moe_lb_loss,
          "evaluation/indexer_loss": indexer_loss,
          "evaluation/mtp_loss": mtp_loss,
          "evaluation/mtp_acceptance_rate_percent": mtp_acceptance_rate,
      },
  }
  if config.use_dpo:
    metrics["scalar"]["evaluation/dpo_reward_accuracy"] = aux["reward_accuracy"]

  return metrics


def train_loop(config, recorder, state=None):
  \"\"\"Main Training loop.\"\"\"
  (
      init_rng,
      checkpoint_manager,
      state_mesh_shardings,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      data_loader,
      rampup_manager,
      eval_data_iterator,
      state,
  ) = train_utils.setup_train_loop(config, recorder)

  if config.use_dpo:
    if "reference_params" not in state.params:
      reference_params = jax.tree.map(jnp.copy, state.params["params"])
      state = _merge_dpo_state(state, reference_params)
    state_mesh_shardings = _merge_dpo_state(state_mesh_shardings, state_mesh_shardings.params["params"])

  params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)

  with jax.set_mesh(mesh), mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    p_train_step, p_eval_step = train_utils.jit_train_and_eval_step(
        config,
        model,
        mesh,
        state,
        state_mesh_shardings,
        train_step,
        eval_step,
        eval_data_iterator,
        params_shardings,
    )
    shaped_batch = maxtext_utils.get_shaped_batch(config)
    if config.shard_optimizer_over_data:
      state = sharding.maybe_shard_with_name(state, state_mesh_shardings, config.shard_mode)
    maxtext_utils.maybe_dump_jaxpr(config, p_train_step, (state, shaped_batch, init_rng))
    if config.compiled_trainstep_file == "":  # compile only when there is no pre-compiled file loaded
      compiled = p_train_step.lower(state, shaped_batch, init_rng).compile()
      compiled_stats = compiled.memory_analysis()
      max_utils.print_compiled_memory_stats(compiled_stats)

  start_step = get_first_step(model, state)  # this is the start_step for training
  prof = profiler.Profiler(config, offset_step=start_step)
  metric_logger = MetricLogger(config=config, learning_rate_schedule=learning_rate_schedule)

  # Write train config params, num model params, and XLA flags to tensorboard
  metric_logger.write_setup_info_to_tensorboard(state.params)

  _job_completed_gracefully = False
  try:
    last_step_completion = datetime.datetime.now()
    for step in np.arange(start_step, config.steps):
      prof.maybe_activate_profiler(step, state)

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        example_batch = data_loader.load_next_batch(rampup_manager=rampup_manager)
        # pylint: disable=not-callable
        nextrng = jax.jit(jax.random.fold_in)(init_rng, step)
        with maybe_record_goodput(recorder, GoodputEvent.STEP, step):
          with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
            if config.shard_optimizer_over_data:
              state = sharding.maybe_shard_with_name(state, state_mesh_shardings, config.shard_mode)
            state, metrics = p_train_step(state, example_batch, nextrng)

      step_time_delta = datetime.datetime.now() - last_step_completion
      last_step_completion = datetime.datetime.now()

      state_to_save = state if not config.use_dpo else _split_dpo_state(state)[0]
      checkpointing.maybe_save_checkpoint(checkpoint_manager, state_to_save, config, data_iterator, step)

      if config.dump_hlo and step == (config.dump_step if config.dump_step >= 0 else start_step):
        jax.block_until_ready(state)  # Ensure compilation has finished.
        gcs_utils.upload_dump(
            config.dump_hlo_local_dir,
            config.dump_hlo_gcs_dir,
            module_name=config.dump_hlo_module_name,
            delete_local_after=config.dump_hlo_delete_local_after,
            all_host_upload=config.dump_hlo_upload_all,
        )

      if config.eval_interval > 0 and step > start_step and (step + 1) % config.eval_interval == 0:
        assert eval_data_iterator
        # Explicitly reset the eval iterator and counters before starting the eval loop
        eval_data_iterator.reset()
        metric_logger.reset_eval_metrics()

        eval_step_count = 0
        # pylint: disable=not-callable
        for eval_batch in eval_data_iterator:
          # Shard input eval data
          eval_batch = jax.device_put(eval_batch, sharding.get_input_data_sharding(config, mesh))
          if config.eval_steps > 0 and eval_step_count >= config.eval_steps:
            break
          with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
            eval_metrics = p_eval_step(state, eval_batch, nextrng)
          metric_logger.record_eval_metrics(step, metrics=eval_metrics)
          max_logging.log(f"Completed eval step {eval_step_count}")
          eval_step_count += 1
        metric_logger.record_eval_metrics(step, eval_step_count=eval_step_count)
        if metric_logger.cumulative_eval_metrics["scalar"]["eval/avg_loss"] <= config.target_eval_loss:
          prof.deactivate()
          raise exceptions.StopTraining(f"Target loss {config.target_eval_loss=} is achieved.")

      prof.maybe_deactivate_profiler(step, state)

      if step == start_step:
        max_utils.print_mem_stats("After params initialized")

      metric_logger.buffer_and_write_train_metrics(metrics, step, step_time_delta)

    if config.save_checkpoint_on_completion:
      state_to_save = state if not config.use_dpo else _split_dpo_state(state)[0]
      checkpointing.maybe_save_checkpoint(checkpoint_manager, state_to_save, config, data_iterator)
    if checkpoint_manager is not None:
      # in case the last checkpoint_period checkpoint is still in progress
      checkpoint_manager.wait_until_finished()
    _job_completed_gracefully = True
  except exceptions.StopTraining as e:
    max_logging.log(f"Training stopped: {str(e)}")
    _job_completed_gracefully = True
  finally:
    if _job_completed_gracefully:
      record_goodput(recorder, RECORD_JOB_END_TIME)
    metric_logger.flush_metrics_and_cleanup()

  return state


def initialize(argv: Sequence[str]) -> tuple[pyconfig.HyperParameters, Any, Any]:
  \"\"\"Initialization of hyperparameters and utilities\"\"\"
  pathwaysutils.initialize()
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  # TF allocates extraneous GPU memory when using TFDS data
  # this leads to CUDA OOMs. WAR for now is to hide GPUs from TF
  tf.config.set_visible_devices([], "GPU")
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )
  # or fill in here
  config = pyconfig.initialize(argv)
  max_utils.print_system_information()
  train_utils.validate_train_config(config)
  jax.config.update("jax_use_shardy_partitioner", config.shardy)
  jax.config.update("jax_remove_size_one_mesh_axis_from_type", config.remove_size_one_mesh_axis_from_type)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path or ""
  vertex_tensorboard_manager = VertexTensorboardManager()
  if config.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
    vertex_tensorboard_manager.configure_vertex_tensorboard(config)

  # Create the Goodput recorder
  recorder = create_goodput_recorder(config)

  # Stack traces configurations
  debug_config = debug_configuration.DebugConfig(
      stack_trace_config=stack_trace_configuration.StackTraceConfig(
          collect_stack_trace=config.collect_stack_trace,
          stack_trace_to_cloud=config.stack_trace_to_cloud,
          stack_trace_interval_seconds=config.stack_trace_interval_seconds,
      )
  )
  diagnostic_config = diagnostic_configuration.DiagnosticConfig(debug_config)
  return config, recorder, diagnostic_config


def run(config, recorder, diagnostic_config):
  \"\"\"Run the job given hyperparameters and utilities.

  In decoupled mode (DECOUPLE_GCLOUD=TRUE) cloud diagnostics may be stubbed; if so, skip wrapping.
  \"\"\"
  # Use nullcontext when diagnostics are stubbed or in decoupled mode
  diagnostics_context = (
      contextlib.nullcontext()
      if is_decoupled() or getattr(diagnostic, "__class__", None).__name__ == "_StubDiag"
      else diagnostic.diagnose(diagnostic_config)
  )

  if is_decoupled() or getattr(diagnostic, "__class__", None).__name__ == "_StubDiag":
    max_logging.log("[DECOUPLED NO-OP] skipping cloud diagnostics wrapper.")

  with (
      diagnostics_context,
      max_utils.maybe_get_transformer_engine_context(config),
  ):
    train_loop(config, recorder)


def get_train_func(config, recorder, diagnostic_config, argv):
  \"\"\"Returns the train function, wrapping in elastic_retry if elastic training is enabled.\"\"\"
  if config.elastic_enabled:
    max_logging.log("Elastic utils: Elastic training enabled.")

    def elastic_train_wrapper(argv: Sequence[str]) -> None:
      \"\"\"Wrapper for elastic training initializes variables and runs the train loop.\"\"\"
      elastic_config, elastic_recorder, elastic_diagnostic_config = initialize(argv)
      run(
          elastic_config,
          elastic_recorder,
          elastic_diagnostic_config,
      )

    train_func = elastic_utils.elastic_retry(config)(functools.partial(elastic_train_wrapper, argv=argv))
  else:
    # Use the already initialized variables
    def train_func():
      run(config, recorder, diagnostic_config)

  return train_func


def main(argv: Sequence[str]) -> None:
  config, recorder, diagnostic_config = initialize(argv)
  record_goodput(recorder, RECORD_JOB_START_TIME)
  train_func = get_train_func(config, recorder, diagnostic_config, argv)
  with maybe_monitor_goodput(config):
    train_func()


if __name__ == "__main__":
  app.run(main)
\n"""


# File: src/maxtext/inference/decode.py (commit 313890777)
DECODE_ENTRYPOINT_RAW = """\n# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

\"\"\"CLI utility for running inference on a single/multi stream(s).\"\"\"

import os
from typing import Sequence, Any
import numpy as np
import jax
import jax.numpy as jnp

from absl import app

from maxtext.configs import pyconfig
from maxtext.common import profiler
from maxtext.common.gcloud_stub import jetstream, is_decoupled
from maxtext.inference.maxengine import maxengine
from maxtext.multimodal import processor as mm_processor
from maxtext.multimodal import utils as mm_utils
from maxtext.utils import max_utils

_config_lib, engine_api, _token_utils, _tokenizer_api, _token_params_ns = jetstream()
# Placeholder: internal

# Number of text sequences to process in a single batch.
_NUM_STREAMS = 1


def _batch_first_result_token(first_tokens: list[Any], batch_size: int):
  \"\"\"Batches together a list of first result tokens from prefill calls.

  This is needed because prefill currently returns the first token as a batch of size 1
  to optimize latency to first token without padding to the configured batch size, while
  generate returns a batch of configured size. This function batches a list of
  such single-element first tokens into one batch to simulate the normal processing
  that first tokens are generated by generate.

  Args:
    first_tokens: A list of `ResultTokens` representing first token returned by `prefill`
    batch_size: The target batch size to pad to. This should be from the config.
  Return:
    A `ResultTokens` with all first tokens batched as if they are produced by a single
    `generate` step.
  \"\"\"
  data = jnp.vstack([first_token.data for first_token in first_tokens])

  def _pad_to_batch_size(data: jax.Array, batch_size: int):
    pad_width = [(0, batch_size - data.shape[0]), (0, 0)]
    data = jnp.pad(data, pad_width, mode="constant", constant_values=0)
    return data

  result_tokens = engine_api.ResultTokens(
      data=_pad_to_batch_size(data, batch_size),
      tokens_idx=(0, 1),
      valid_idx=(1, 2),
      length_idx=(2, 3),
      samples_per_slot=1,
  )

  def _all_equals(elements: Sequence[jax.Array], target: jax.Array):
    \"\"\"Checks if each element equals the given target.\"\"\"
    stacked = jnp.stack(elements)
    row_comparisons = stacked == target
    return jnp.all(row_comparisons)

  # `tokens_idx`, `valid_idx`, `length_idx` and `samples_per_slot` are hardcoded
  # and should be the same for all first tokens returned from prefill.
  assert _all_equals([jnp.array(t.tokens_idx) for t in first_tokens], jnp.array(result_tokens.tokens_idx))
  assert _all_equals([jnp.array(t.valid_idx) for t in first_tokens], jnp.array(result_tokens.valid_idx))
  assert _all_equals([jnp.array(t.length_idx) for t in first_tokens], jnp.array(result_tokens.length_idx))
  assert _all_equals([jnp.array(t.samples_per_slot) for t in first_tokens], jnp.array(result_tokens.samples_per_slot))

  return result_tokens


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  config = pyconfig.initialize(argv)
  _validate_config(config)
  jax.config.update("jax_use_shardy_partitioner", config.shardy)
  max_utils.print_system_information()

  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(1234)
  rng, rng_load_params = jax.random.split(rng)
  params = engine.load_params(rng_load_params)
  prof = profiler.Profiler(config)

  text = config.prompt
  prefill_length = config.max_prefill_predict_length
  processor_outputs = mm_utils.PreprocessorOutput()
  if config.use_multimodal:
    processor_outputs = mm_processor.preprocess_mm_data(config)
    image_offsets = mm_processor.get_image_offsets(config=config, processor_output=processor_outputs)

    prefill_length -= image_offsets
    text = mm_processor.reformat_prompt(
        prompt=config.prompt,
        image_placeholder=config.image_placeholder,
        video_placeholder=config.video_placeholder,
        model_name=config.model_name,
        num_images=processor_outputs.num_images,
        num_videos=getattr(processor_outputs, "num_videos", 0),
    )

  metadata = engine.get_tokenizer()
  tokenizer_model = engine.build_tokenizer(metadata)
  token_params_is_stub = getattr(_token_params_ns, "_IS_STUB", False)
  engine_api_is_stub = getattr(engine_api, "_IS_STUB", False)
  if is_decoupled() and (token_params_is_stub or engine_api_is_stub):
    raise RuntimeError(
        "JetStream disabled by DECOUPLE_GCLOUD=TRUE or stubbed; decode requires the JetStream tokenizer. "
        "Unset DECOUPLE_GCLOUD or install JetStream to run decode."
    )

  try:
    has_chat_template = getattr(tokenizer_model.tokenizer, "chat_template", False)  # pytype: disable=attribute-error
  except AttributeError as _:
    has_chat_template = False
  is_bos = config.add_bos and not has_chat_template
  tokens, true_length = tokenizer_model.encode(text, is_bos=is_bos, prefill_lengths=[prefill_length])

  position_ids = None
  mrope_position_deltas = None

  if config.use_multimodal:
    tokens = mm_processor.prepare_text_for_image_fusion(tokens=tokens, config=config, processor_output=processor_outputs)
    true_length += image_offsets

    if config.use_mrope:
      from maxtext.multimodal import processor_qwen3_omni  # pylint: disable=import-outside-toplevel

      position_ids, mrope_position_deltas = processor_qwen3_omni.get_rope_index(
          input_ids=tokens[np.newaxis, :],  # Add batch dimension for processing
          image_grid_thw=processor_outputs.pixel_grid_thw,  # pytype: disable=attribute-error
          video_grid_thw=processor_outputs.video_grid_thw,  # pytype: disable=attribute-error
          attention_mask=np.ones_like(tokens)[np.newaxis, :],
          use_audio_in_video=config.use_audio and getattr(processor_outputs, "num_videos", 0) > 0,
          audio_lengths=processor_outputs.audio_lengths,  # pytype: disable=attribute-error
          second_per_grids=processor_outputs.video_second_per_grid,  # pytype: disable=attribute-error
          spatial_merge_size=config.spatial_merge_size_for_vit,  # pytype: disable=attribute-error
          position_id_per_seconds=config.position_id_per_seconds,
      )

  assert (
      true_length <= config.max_prefill_predict_length
  ), f"Input token length {true_length} is longer than {config.max_prefill_predict_length=}"
  assert config.quantization != "fp8", "fp8 on NVIDIA GPUs is not supported in decode.py yet"
  assert config.quantization != "nanoo_fp8", "NANOO fp8 on AMD MI300/MI325 GPUs is not supported in decode.py yet"

  batch_size = int(config.per_device_batch_size * jax.device_count())
  assert (
      0 < _NUM_STREAMS <= batch_size
  ), f"The number of streams {_NUM_STREAMS} must be > 0 and <= batch size {batch_size}"

  prefill_result_list = []
  first_token_list = []
  sampled_tokens_list = []

  prof.activate(optional_postfix="trace")

  # Prefill
  rng, rng_prefill = jax.random.split(rng)  # Split RNG before calling prefill
  for i in range(_NUM_STREAMS):
    with jax.profiler.StepTraceAnnotation("prefill", stream=i):
      prefill_result, first_token = engine.prefill(
          params=params,
          padded_tokens=tokens,
          positions=position_ids,
          mrope_deltas=mrope_position_deltas,
          images=processor_outputs.pixel_values if config.use_multimodal else None,
          image_masks=processor_outputs.pixel_mask if config.use_multimodal and "llama4" in config.model_name else None,
          audio_values=processor_outputs.audio_values if config.use_audio else None,
          audio_masks=processor_outputs.audio_mask if config.use_audio else None,
          true_length=true_length,
          rng=rng_prefill,
          slot=i,
      )
    prefill_result_list.append(prefill_result)
    first_token_list.append(first_token)

  # Insert
  rng, rng_init_decode = jax.random.split(rng)
  decode_state = engine.init_decode_state(rng_init_decode)
  for i in range(_NUM_STREAMS):
    decode_state = engine.insert(prefill_result_list[i], decode_state, slot=i)

  # Generate
  prof_deactivated = False
  steps = range(config.max_prefill_predict_length, config.max_target_length)
  sampled_tokens_list.append(_batch_first_result_token(first_token_list, batch_size))
  for i in steps:
    rng, rng_generate = jax.random.split(rng)
    with jax.profiler.StepTraceAnnotation("generate", step=i):
      decode_state, sampled_tokens = engine.generate(params, decode_state, rng=rng_generate)

    # Automatically deactivate profiler after profiler_steps steps
    if i > config.max_prefill_predict_length + config.profiler_steps:
      prof.deactivate(blocking_object=sampled_tokens)
      prof_deactivated = True

    sampled_tokens_list.append(sampled_tokens)

  # Get results
  for i in range(_NUM_STREAMS):
    results = [t.get_result_at_slot(i).tokens.item() for t in sampled_tokens_list]
    output = tokenizer_model.decode(results)
    print(f"Input `{text}` -> `{output}`")

  assert output.startswith(
      config.autoregressive_decode_assert
  ), f"generated text mismatch {output=}, {config.autoregressive_decode_assert=}"

  # Deactivate profiler
  if not prof_deactivated:
    prof.deactivate(blocking_object=output)

  prof.post_process()


def _validate_config(config):
  assert config.load_full_state_path == "", (
      "Decode doesn't operate on full states! Convert to parameter checkpoint first."
      "Using generate_param_only_checkpoint."
  )


if __name__ == "__main__":
  app.run(main)
\n"""


MAXTEXT_STATIC_INSTRUCTION = f"""
================================================================================
RAW MAXTEXT REFERENCE SOURCE CODE & CORE BRING-UP SPECIFICATIONS
================================================================================

1. COMMON TYPES (src/maxtext/common/common_types.py):
{COMMON_TYPES_RAW}

2. NORMALIZATIONS (src/maxtext/layers/normalizations.py):
{NORMALIZATIONS_RAW}

3. EMBEDDINGS (src/maxtext/layers/embeddings.py):
{EMBEDDINGS_RAW}

4. LINEARS (src/maxtext/layers/linears.py):
{LINEARS_RAW}

5. ATTENTION MLA REFERENCE (src/maxtext/layers/attention_mla.py):
{ATTENTION_MLA_RAW}

6. MOE PRIMITIVES (src/maxtext/layers/moe.py):
{MOE_RAW}

7. DECODER STACK ASSEMBLY (src/maxtext/layers/decoders.py):
{DECODERS_STACK_RAW}

8. PARAMETER INITIALIZERS (src/maxtext/layers/initializers.py):
{INITIALIZERS_RAW}

9. DEEPSEEK MODEL WIRING (src/maxtext/models/deepseek.py):
{DEEPSEEK_MODEL_RAW}

10. MASTER MODELS CONTAINER (src/maxtext/models/models.py):
{MODELS_RAW}

11. SPMD MESH SHARDING UTILS (src/maxtext/utils/sharding.py):
{SHARDING_RAW}

12. MASTER BASE CONFIGURATION (src/maxtext/configs/base.yml):
{BASE_CONFIG_RAW}

13. MODEL CONFIGURATION (src/maxtext/configs/models/deepseek3-671b.yml):
{DEEPSEEK_CONFIG_RAW}

14. PARAMETER MAPPING (src/maxtext/checkpoint_conversion/utils/param_mapping.py):
{PARAM_MAPPING_RAW}

15. TO_MAXTEXT CHECKPOINT CONVERTER (src/maxtext/checkpoint_conversion/to_maxtext.py):
{TO_MAXTEXT_CONVERTER_RAW}

16. TO_HUGGINGFACE CHECKPOINT CONVERTER (src/maxtext/checkpoint_conversion/to_huggingface.py):
{TO_HF_CONVERTER_RAW}

17. GLOBALS & REPO MAPPING (src/maxtext/utils/globals.py):
{GLOBALS_RAW}

18. OPTIMIZERS & LEARNING RATE SCHEDULES (src/maxtext/optimizers/optimizers.py):
{OPTIMIZERS_RAW}

19. MAIN TRAINING LOOP ENTRYPOINT (src/maxtext/trainers/pre_train/train.py):
{TRAIN_ENTRYPOINT_RAW}

20. MAIN INFERENCE DECODE ENTRYPOINT (src/maxtext/inference/decode.py):
{DECODE_ENTRYPOINT_RAW}
================================================================================
"""
