---
title: 'Module: easydel/trainers/embedding_trainer/_fn.py'
type: catalog
provenance: extracted
module: easydel/trainers/embedding_trainer/_fn.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.embedding_trainer._fn`/
symbols:
  embedding_training_step: embedding_training_step().
  embedding_training_step.loss_fn: embedding_training_step().loss_fn().
  mnrl_loss: mnrl_loss().
  _embed_batch: _embed_batch().
  infonce_loss: infonce_loss().
  triplet_loss: triplet_loss().
  matryoshka_loss: matryoshka_loss().
---
# Module: [`easydel/trainers/embedding_trainer/_fn.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/_fn.py)

## Functions
- `_embed_batch(module, input_ids: jax.Array, attention_mask: jax.Array, normalize: bool = True)` — [`L178`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/_fn.py#L178) — Forward-pass a batch through the embedding model and return embeddings.
- `embedding_training_step(state: EasyDeLState, batch: dict[str, jax.Array], loss_type: str = "infonce", temperature: float = 0.05, margin: float = 0.2, normalize: bool = True, matryoshka_dims: list[int] | None = None, learning_rate_fn: optax.Schedule | None = None, partition_spec: PartitionSpec | None = None, gradient_accumulation_steps: int = 1)` — [`L202`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/_fn.py#L202) — Training step for contrastive embedding learning.
- `infonce_loss(query_embeds: jax.Array, positive_embeds: jax.Array, negative_embeds: jax.Array | None = None, temperature: float = 0.05)` — [`L37`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/_fn.py#L37) — Compute InfoNCE contrastive loss with in-batch negatives.
- `loss_fn(tree, minibatch)` — [`L250`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/_fn.py#L250)
- `matryoshka_loss(loss_fn: tp.Callable, query_embeds: jax.Array, positive_embeds: jax.Array, negative_embeds: jax.Array | None, dims: list[int], **loss_kwargs)` — [`L131`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/_fn.py#L131) — Compute Matryoshka Representation Learning loss.
- `mnrl_loss(query_embeds: jax.Array, positive_embeds: jax.Array, negative_embeds: jax.Array | None = None, temperature: float = 0.05)` — [`L108`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/_fn.py#L108) — Compute Multiple Negatives Ranking Loss (MNRL).
- `triplet_loss(query_embeds: jax.Array, positive_embeds: jax.Array, negative_embeds: jax.Array, margin: float = 0.2)` — [`L77`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/_fn.py#L77) — Compute triplet margin loss.

