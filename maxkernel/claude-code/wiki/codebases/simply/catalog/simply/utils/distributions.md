---
title: 'Module: simply/utils/distributions.py'
type: catalog
provenance: extracted
module: simply/utils/distributions.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.distributions`/
symbols:
  Array: Array.
  Categorical: Categorical#
  MaskedCategorical.masked_logits: MaskedCategorical#masked_logits().
  MaskedCategorical: MaskedCategorical#
  Categorical.log_prob: Categorical#log_prob().
  MaskedCategorical.log_prob: MaskedCategorical#log_prob().
  Categorical.log_probs: Categorical#log_probs().
  MaskedCategorical.masked_log_probs: MaskedCategorical#masked_log_probs().
  Distribution.sample: Distribution#sample().
  Categorical.sample: Categorical#sample().
  MaskedCategorical.sample: MaskedCategorical#sample().
  Categorical._entropy: Categorical#_entropy().
  Categorical.logits: Categorical#logits.
  Distribution.entropy: Distribution#entropy().
  Distribution.prob: Distribution#prob().
  Distribution.log_prob: Distribution#log_prob().
  Categorical.entropy: Categorical#entropy().
  Categorical.prob: Categorical#prob().
  MaskedCategorical.prob: MaskedCategorical#prob().
  MaskedCategorical._entropy: MaskedCategorical#_entropy().
  DistributionRegistry: DistributionRegistry#
  _gather: _gather().
  MaskedCategorical.mask: MaskedCategorical#mask.
  Shape: Shape.
  Categorical.dtype: Categorical#dtype().
  Distribution: Distribution#
  MaskedCategorical.neg_inf: MaskedCategorical#neg_inf.
  DistributionRegistry.namespace: DistributionRegistry#namespace.
---
# Module: [`simply/utils/distributions.py`](../../../../../../raw/code/simply/simply/utils/distributions.py)

## Classes
### `Categorical`
- def: [`simply/utils/distributions.py:64`](../../../../../../raw/code/simply/simply/utils/distributions.py#L64)
- doc: Categorical distribution.
- signature: `class Categorical(Distribution):`
- members:
  - `dtype(self)` — [`L70`](../../../../../../raw/code/simply/simply/utils/distributions.py#L70)
  - `entropy(self)` — [`L77`](../../../../../../raw/code/simply/simply/utils/distributions.py#L77)
  - `log_prob(self, value: Array)` — [`L88`](../../../../../../raw/code/simply/simply/utils/distributions.py#L88)
  - `log_probs(self)` — [`L74`](../../../../../../raw/code/simply/simply/utils/distributions.py#L74)
  - `prob(self, value: Array)` — [`L85`](../../../../../../raw/code/simply/simply/utils/distributions.py#L85)
  - `sample(self, key: jax.typing.ArrayLike, shape: Shape | None = None)` — [`L80`](../../../../../../raw/code/simply/simply/utils/distributions.py#L80)
  - `logits` — [`L67`](../../../../../../raw/code/simply/simply/utils/distributions.py#L67)
- protocol/private: `_entropy`[`L92`](../../../../../../raw/code/simply/simply/utils/distributions.py#L92)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`Array`](distributions.md#Array), [`DistributionRegistry`](distributions.md#DistributionRegistry), [`_gather`](distributions.md#_gather), [`Shape`](distributions.md#Shape), [`Distribution`](distributions.md#Distribution)
- used by: [`compute_ppo_loss`](../rl_lib.md#compute_ppo_loss), [`MaskedCategorical`](distributions.md#MaskedCategorical), [`_compute_logprobs`](../rl_lib.md#compute_logprobs._compute_logprobs), [`sample`](distributions.md#Distribution.sample), [`test_categorical`](distributions_test.md#DistributionsTest.test_categorical), [`entropy`](distributions.md#Distribution.entropy), [`log_prob`](distributions.md#Distribution.log_prob), [`prob`](distributions.md#Distribution.prob), [`simple_sample_fn`](sampling_lib.md#sample_from_logits.simple_sample_fn), [`simple_sample_score_fn`](sampling_lib.md#compute_log_likelihood.simple_sample_score_fn)

### `Distribution`  ·  implements/extends ABC
- def: [`simply/utils/distributions.py:39`](../../../../../../raw/code/simply/simply/utils/distributions.py#L39)
- doc: Base class of Distribution.
- signature: `class Distribution(abc.ABC):`
- members:
  - `entropy(self)` — [`L43`](../../../../../../raw/code/simply/simply/utils/distributions.py#L43) — Computes the entropy of the distribution.
  - `log_prob(self, value: Array)` — [`L57`](../../../../../../raw/code/simply/simply/utils/distributions.py#L57) — Computes the log probability of the input value.
  - `prob(self, value: Array)` — [`L53`](../../../../../../raw/code/simply/simply/utils/distributions.py#L53) — Computes the probability of the input value.
  - `sample(self, key: jax.typing.ArrayLike, shape: Shape | None = None)` — [`L47`](../../../../../../raw/code/simply/simply/utils/distributions.py#L47) — Samples random values from the distribution.
- uses (calls/refs, reference-scoped): [`Array`](distributions.md#Array), [`log_prob`](distributions.md#Categorical.log_prob), [`sample`](distributions.md#Categorical.sample), [`entropy`](distributions.md#Categorical.entropy), [`prob`](distributions.md#Categorical.prob), [`Shape`](distributions.md#Shape)
- used by: [`Categorical`](distributions.md#Categorical)

### `DistributionRegistry`  ·  implements/extends RootRegistry
- def: [`simply/utils/distributions.py:33`](../../../../../../raw/code/simply/simply/utils/distributions.py#L33)
- doc: Registry for distributions.
- signature: `class DistributionRegistry(registry.RootRegistry):`
- members:
  - `namespace` — [`L36`](../../../../../../raw/code/simply/simply/utils/distributions.py#L36)
- uses (calls/refs, reference-scoped): [`RootRegistry`](registry.md#RootRegistry)
- used by: [`RootRegistry`](registry.md#RootRegistry), [`Categorical`](distributions.md#Categorical), [`MaskedCategorical`](distributions.md#MaskedCategorical)

### `MaskedCategorical`
- def: [`simply/utils/distributions.py:98`](../../../../../../raw/code/simply/simply/utils/distributions.py#L98)
- doc: Masked categorical distribution.
- signature: `class MaskedCategorical(Categorical):`
- members:
  - `log_prob(self, value: Array)` — [`L120`](../../../../../../raw/code/simply/simply/utils/distributions.py#L120)
  - `masked_log_probs(self)` — [`L109`](../../../../../../raw/code/simply/simply/utils/distributions.py#L109)
  - `masked_logits(self)` — [`L105`](../../../../../../raw/code/simply/simply/utils/distributions.py#L105)
  - `prob(self, value: Array)` — [`L117`](../../../../../../raw/code/simply/simply/utils/distributions.py#L117)
  - `sample(self, key: jax.typing.ArrayLike, shape: Shape | None = None)` — [`L112`](../../../../../../raw/code/simply/simply/utils/distributions.py#L112)
  - `mask` — [`L101`](../../../../../../raw/code/simply/simply/utils/distributions.py#L101)
  - `neg_inf` — [`L102`](../../../../../../raw/code/simply/simply/utils/distributions.py#L102)
- protocol/private: `_entropy`[`L124`](../../../../../../raw/code/simply/simply/utils/distributions.py#L124)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`Array`](distributions.md#Array), [`masked`](masked.md#masked), [`Categorical`](distributions.md#Categorical), [`DistributionRegistry`](distributions.md#DistributionRegistry), [`_gather`](distributions.md#_gather), [`Shape`](distributions.md#Shape)
- used by: [`masked_sample_fn`](sampling_lib.md#sample_from_logits.masked_sample_fn), [`masked_sample_score_fn`](sampling_lib.md#compute_log_likelihood.masked_sample_score_fn), [`test_masked_categorical`](distributions_test.md#DistributionsTest.test_masked_categorical)

## Functions
- `_gather(x: Array, indices: Array)` — [`L130`](../../../../../../raw/code/simply/simply/utils/distributions.py#L130) — Gathers x values along the last dim.

## Module values
- `Array` — [`L30`](../../../../../../raw/code/simply/simply/utils/distributions.py#L30)
- `Shape` — [`L29`](../../../../../../raw/code/simply/simply/utils/distributions.py#L29)

