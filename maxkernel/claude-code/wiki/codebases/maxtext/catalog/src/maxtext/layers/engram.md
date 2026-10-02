---
title: 'Module: src/maxtext/layers/engram.py'
type: catalog
provenance: extracted
module: src/maxtext/layers/engram.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.layers.engram`/
symbols:
  Engram.__call__: Engram#__call__().
  Engram.value_proj: Engram#value_proj.
  Engram.key_proj: Engram#key_proj.
  NgramHashMapping._calculate_vocab_size_across_layers: NgramHashMapping#_calculate_vocab_size_across_layers().
  NgramHashMapping._get_ngram_hashes: NgramHashMapping#_get_ngram_hashes().
  NgramHashMapping._calculate_multipliers_across_layers: NgramHashMapping#_calculate_multipliers_across_layers().
  Engram.short_conv: Engram#short_conv.
  NgramHashMapping.__call__: NgramHashMapping#__call__().
  MultiHeadEmbedding.__call__: MultiHeadEmbedding#__call__().
  ShortConv.__call__: ShortConv#__call__().
  Engram.config: Engram#config.
  NgramHashMapping.pad_id: NgramHashMapping#pad_id.
  NgramHashMapping.compressed_tokenizer: NgramHashMapping#compressed_tokenizer.
  NgramHashMapping.vocab_size_across_layers: NgramHashMapping#vocab_size_across_layers.
  Engram.dtype: Engram#dtype.
  Engram.weight_dtype: Engram#weight_dtype.
  NgramHashMapping.max_ngram_size: NgramHashMapping#max_ngram_size.
  CompressedTokenizer.num_new_token: CompressedTokenizer#num_new_token.
  CompressedTokenizer.lookup_table: CompressedTokenizer#lookup_table.
  NgramHashMapping.tokenizer_vocab_size: NgramHashMapping#tokenizer_vocab_size.
  NgramHashMapping.layer_multipliers: NgramHashMapping#layer_multipliers.
  MultiHeadEmbedding.offsets: MultiHeadEmbedding#offsets.
  ShortConv.norm: ShortConv#norm.
  Engram.conv_kernel_size: Engram#conv_kernel_size.
  Engram.multi_head_embedding: Engram#multi_head_embedding.
  Engram.k_norm: Engram#k_norm.
  Engram.q_norm: Engram#q_norm.
  NgramHashMapping.layer_ids: NgramHashMapping#layer_ids.
  CompressedTokenizer.__init__: CompressedTokenizer#__init__().
  CompressedTokenizer.__len__: CompressedTokenizer#__len__().
  CompressedTokenizer.__call__: CompressedTokenizer#__call__().
  NgramHashMapping.get_vocab_sizes: NgramHashMapping#get_vocab_sizes().
  CompressedTokenizer.lookup_table_np: CompressedTokenizer#lookup_table_np.
  Engram.kernel_init: Engram#kernel_init.
  Engram.quant: Engram#quant.
  Engram.max_ngram_size: Engram#max_ngram_size.
  Engram.engram_dim: Engram#engram_dim.
  Engram.create_norms: Engram#create_norms().
  Engram.apply_norms: Engram#apply_norms().
  CompressedTokenizer: CompressedTokenizer#
  CompressedTokenizer._build_normalizer: CompressedTokenizer#_build_normalizer().
  CompressedTokenizer._build_lookup_table: CompressedTokenizer#_build_lookup_table().
  NgramHashMapping.min_head_vocab_size_per_ngram: NgramHashMapping#min_head_vocab_size_per_ngram.
  NgramHashMapping.n_head_per_ngram: NgramHashMapping#n_head_per_ngram.
  NgramHashMapping.find_next_unseen_prime: NgramHashMapping#find_next_unseen_prime().
  StaticWrapper: StaticWrapper#
  StaticWrapper.val: StaticWrapper#val.
  MultiHeadEmbedding: MultiHeadEmbedding#
  MultiHeadEmbedding.embedding: MultiHeadEmbedding#embedding.
  ShortConv: ShortConv#
  ShortConv.create_norms: ShortConv#create_norms().
  ShortConv.conv: ShortConv#conv.
  ShortConv.apply_norms: ShortConv#apply_norms().
  Engram.rngs: Engram#rngs.
  NgramHashMapping: NgramHashMapping#
  NgramHashMapping.__init__: NgramHashMapping#__init__().
  StaticWrapper.__init__: StaticWrapper#__init__().
  MultiHeadEmbedding.__init__: MultiHeadEmbedding#__init__().
  MultiHeadEmbedding.num_heads: MultiHeadEmbedding#num_heads.
  ShortConv.__init__: ShortConv#__init__().
  ShortConv.mhc_expansion_rate: ShortConv#mhc_expansion_rate.
  Engram: Engram#
  Engram.__init__: Engram#__init__().
  Engram.mesh: Engram#mesh.
  Engram.mhc_expansion_rate: Engram#mhc_expansion_rate.
---
# Module: [`src/maxtext/layers/engram.py`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py)

## Classes
### `CompressedTokenizer`
- def: [`src/maxtext/layers/engram.py:40`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L40)
- doc: A canonicalizing wrapper that reduces vocabulary sparsity for n-gram lookup.
- signature: `class CompressedTokenizer:`
- members:
  - `__call__(self, input_ids)` — [`L124`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L124) — Maps original token IDs to compressed IDs.
  - `_build_lookup_table(self, tokenizer: HFTokenizer, normalizer: normalizers.Sequence)` — [`L91`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L91) — Builds the mapping from the original vocabulary to the compressed vocabulary.
  - `_build_normalizer(self)` — [`L61`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L61) — Builds the normalization pipeline for text processing.
  - `lookup_table` — [`L56`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L56)
  - `lookup_table_np` — [`L55`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L55)
  - `num_new_token` — [`L55`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L55)
- protocol/private: `__init__`[`L53`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L53), `__len__`[`L58`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L58)
- used by: [`pad_id`](engram.md#NgramHashMapping.pad_id), [`compressed_tokenizer`](engram.md#NgramHashMapping.compressed_tokenizer)

### `Engram`  ·  implements/extends Module
- def: [`src/maxtext/layers/engram.py:496`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L496)
- doc: Engram Memory Layer with n-gram embedding, with multi-branch integration.
- signature: `class Engram(nnx.Module):`
- members:
  - `__call__(self, hidden_states: Array, hash_input_ids: Array)` — [`L621`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L621) — Computes the Engram output by retrieving, gating, and smoothing n-gram memory.
  - `__init__(self, config: Config, mesh: Mesh, vocab_sizes: List[int], engram_num_heads: int, engram_head_dim: int, engram_max_ngram_size: int, engram_kernel_size: int, mhc_expansion_rate: int, kernel_init: NdInitializer = nd_dense_init(1, "fan_in", "normal"), quant: Optional[Quant] = None, rngs: nnx.Rngs = None)` — [`L506`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L506) — Args:
  - `apply_norms(norms, x)` — [`L660`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L660)
  - `create_norms(rngs)` — [`L579`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L579)
  - `config` — [`L535`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L535)
  - `conv_kernel_size` — [`L546`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L546)
  - `dtype` — [`L537`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L537)
  - `engram_dim` — [`L549`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L549)
  - `k_norm` — [`L590`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L590)
  - `kernel_init` — [`L539`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L539)
  - `key_proj` — [`L558`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L558)
  - `max_ngram_size` — [`L545`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L545)
  - `mesh` — [`L536`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L536)
  - `mhc_expansion_rate` — [`L542`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L542)
  - `multi_head_embedding` — [`L552`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L552)
  - `q_norm` — [`L592`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L592)
  - `quant` — [`L540`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L540)
  - `rngs` — [`L541`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L541)
  - `short_conv` — [`L612`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L612)
  - `value_proj` — [`L595`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L595)
  - `weight_dtype` — [`L538`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L538)
- uses (calls/refs, reference-scoped): [`MultiHeadEmbedding`](engram.md#MultiHeadEmbedding), [`ShortConv`](engram.md#ShortConv)

### `MultiHeadEmbedding`  ·  implements/extends Module
- def: [`src/maxtext/layers/engram.py:339`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L339)
- doc: A flattened table representation for multi-head embedding spaces across n-gram orders.
- signature: `class MultiHeadEmbedding(nnx.Module):`
- members:
  - `__call__(self, input_ids: Array, model_mode: str = MODEL_MODE_TRAIN)` — [`L371`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L371) — Retrieves embeddings for multi-head indices.
  - `__init__(self, config: Config, mesh: Mesh, vocab_sizes: List[int], head_dim: int, rngs: nnx.Rngs = None)` — [`L344`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L344) — Args:
  - `embedding` — [`L369`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L369)
  - `num_heads` — [`L361`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L361)
  - `offsets` — [`L366`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L366)
- uses (calls/refs, reference-scoped): [`StaticWrapper`](engram.md#StaticWrapper), [`val`](engram.md#StaticWrapper.val)
- used by: [`multi_head_embedding`](engram.md#Engram.multi_head_embedding)

### `NgramHashMapping`
- def: [`src/maxtext/layers/engram.py:139`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L139)
- doc: Deterministically maps token indices to n-gram hash indices for embedding lookups.
- signature: `class NgramHashMapping:`
- members:
  - `__init__(self, engram_vocab_bases: List[int], max_ngram_size: int, engram_num_heads: int, layer_ids: List[int], tokenizer: HFTokenizer, pad_id: int, seed: int)` — [`L154`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L154) — Args:
  - `_calculate_multipliers_across_layers(self, seed: int)` — [`L197`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L197) — Pre-calculates random odd multipliers for each layer and n-gram position.
  - `_calculate_vocab_size_across_layers(self)` — [`L222`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L222) — Calculates unique prime vocabulary sizes for every head in every layer.
  - `_get_ngram_hashes(self, compressed_ids: Array, layer_id: int)` — [`L264`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L264) — Computes hash indices for all n-grams in the input batch.
  - `find_next_unseen_prime(start: int, seen_primes: set)` — [`L228`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L228)
  - `get_vocab_sizes(self, layer_id: int)` — [`L258`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L258) — Returns a flattened list of prime vocabulary sizes for a specific layer.
  - `compressed_tokenizer` — [`L180`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L180)
  - `layer_ids` — [`L177`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L177)
  - `layer_multipliers` — [`L189`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L189)
  - `max_ngram_size` — [`L175`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L175)
  - `min_head_vocab_size_per_ngram` — [`L174`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L174)
  - `n_head_per_ngram` — [`L176`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L176)
  - `pad_id` — [`L185`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L185)
  - `tokenizer_vocab_size` — [`L181`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L181)
  - `vocab_size_across_layers` — [`L195`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L195)
- protocol/private: `__call__`[`L322`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L322)
- uses (calls/refs, reference-scoped): [`lookup_table_np`](engram.md#CompressedTokenizer.lookup_table_np), [`CompressedTokenizer`](engram.md#CompressedTokenizer)

### `ShortConv`  ·  implements/extends Module
- def: [`src/maxtext/layers/engram.py:390`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L390)
- doc: Depthwise causal 1D convolution, with multi-branch integration.
- signature: `class ShortConv(nnx.Module):`
- members:
  - `__call__(self, x: Array)` — [`L459`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L459) — Compute y^i = SiLU(Conv1D(RMSNorm^i(x^i))) for each branch i.
  - `__init__(self, config: Config, hidden_size: int, kernel_size: int, dilation: int, mhc_expansion_rate: int, rngs: nnx.Rngs = None)` — [`L399`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L399) — Args:
  - `apply_norms(norms, x)` — [`L481`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L481)
  - `create_norms(rngs)` — [`L425`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L425)
  - `conv` — [`L442`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L442)
  - `mhc_expansion_rate` — [`L417`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L417)
  - `norm` — [`L435`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L435)
- used by: [`short_conv`](engram.md#Engram.short_conv)

### `StaticWrapper`
- def: [`src/maxtext/layers/engram.py:332`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L332)
- doc: Wrapper to prevent nnx from treating the value as a variable.
- signature: `class StaticWrapper:`
- members:
  - `val` — [`L336`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L336)
- protocol/private: `__init__`[`L335`](../../../../../../../raw/code/maxtext/src/maxtext/layers/engram.py#L335)
- used by: [`__call__`](engram.md#MultiHeadEmbedding.__call__), [`offsets`](engram.md#MultiHeadEmbedding.offsets)

