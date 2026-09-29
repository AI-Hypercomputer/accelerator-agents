---
title: 'Module: jaxite/jaxite_cggi/random_source.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/random_source.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.random_source`/
symbols:
  RandomSource: RandomSource#
  PseudorandomSource: PseudorandomSource#
  RandomSource.uniform: RandomSource#uniform().
  RandomSource.rounded_normal: RandomSource#rounded_normal().
  _shape_generator: _shape_generator().
  RandomSource.sk_uniform: RandomSource#sk_uniform().
  ALL_RNGS: ALL_RNGS.
  CycleRng: CycleRng#
  NormalOnlyRng.rounded_normal: NormalOnlyRng#rounded_normal().
  ConstantUniformRng.rounded_normal: ConstantUniformRng#rounded_normal().
  SystemRandomSource.uniform: SystemRandomSource#uniform().
  SystemRandomSource.rounded_normal: SystemRandomSource#rounded_normal().
  PseudorandomSource.uniform: PseudorandomSource#uniform().
  PseudorandomSource.rounded_normal: PseudorandomSource#rounded_normal().
  ZeroRng: ZeroRng#
  CycleRng.uniform: CycleRng#uniform().
  VARYING_MAGNITUDE_TEST_RNGS: VARYING_MAGNITUDE_TEST_RNGS.
  CycleRng.rounded_normal: CycleRng#rounded_normal().
  SystemRandomSource: SystemRandomSource#
  SystemRandomSource.sk_uniform: SystemRandomSource#sk_uniform().
  PseudorandomSource.sk_uniform: PseudorandomSource#sk_uniform().
  NormalOnlyRng.sk_uniform: NormalOnlyRng#sk_uniform().
  ConstantUniformRng.sk_uniform: ConstantUniformRng#sk_uniform().
  NormalOnlyRng: NormalOnlyRng#
  ConstantUniformRng: ConstantUniformRng#
  CycleRng.noise: CycleRng#noise.
  ConstantUniformRng.uniform: ConstantUniformRng#uniform().
  SystemRandomSource.rng: SystemRandomSource#rng.
  PseudorandomSource.rng: PseudorandomSource#rng.
  ZeroRng.rounded_normal: ZeroRng#rounded_normal().
  CycleRng.sk_uniform: CycleRng#sk_uniform().
  NormalOnlyRng.rng: NormalOnlyRng#rng.
  ConstantUniformRng.rng: ConstantUniformRng#rng.
  _normalize_value: _normalize_value().
  CycleRng.const_normal_noise: CycleRng#const_normal_noise.
  CycleRng.uniform_data: CycleRng#uniform_data.
  SystemRandomSource.uniform_bounds: SystemRandomSource#uniform_bounds.
  SystemRandomSource.normal_std: SystemRandomSource#normal_std.
  PseudorandomSource.uniform_bounds: PseudorandomSource#uniform_bounds.
  PseudorandomSource.normal_std: PseudorandomSource#normal_std.
  NormalOnlyRng.normal_std: NormalOnlyRng#normal_std.
  NormalOnlyRng.uniform: NormalOnlyRng#uniform().
  ConstantUniformRng.const_uniform: ConstantUniformRng#const_uniform.
  ConstantUniformRng.normal_std: ConstantUniformRng#normal_std.
  ZeroRng.uniform: ZeroRng#uniform().
  _varying_params: _varying_params.
  CycleRng.__init__: CycleRng#__init__().
  SystemRandomSource.__init__: SystemRandomSource#__init__().
  PseudorandomSource.__init__: PseudorandomSource#__init__().
  NormalOnlyRng.__init__: NormalOnlyRng#__init__().
  ConstantUniformRng.__init__: ConstantUniformRng#__init__().
  ZeroRng.sk_uniform: ZeroRng#sk_uniform().
---
# Module: [`jaxite/jaxite_cggi/random_source.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py)

## Classes
### `ConstantUniformRng`  ·  implements/extends RandomSource
- def: [`jaxite/jaxite_cggi/random_source.py:214`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L214) — documented in [jaxite-jaxite_cggi-random_source](../../../concepts/jaxite-jaxite_cggi-random_source.md)
- doc: A test-only RNG that sets the output of uniform() to a fixed value.
- signature: `class ConstantUniformRng(RandomSource):`
- members:
  - `rounded_normal(self, shape: Sequence[int] = (1,), dtype: jnp.dtype = jnp.uint32)` — [`L229`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L229)
  - `sk_uniform(self, shape: Sequence[int] = (1,), dtype: jnp.dtype = jnp.uint32)` — [`L236`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L236)
  - `uniform(self, shape: Sequence[int] = (1,), dtype: jnp.dtype = jnp.uint32)` — [`L224`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L224)
  - `const_uniform` — [`L220`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L220)
  - `normal_std` — [`L221`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L221)
  - `rng` — [`L222`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L222)
- protocol/private: `__init__`[`L217`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L217)
- uses (calls/refs, reference-scoped): [`RandomSource`](random_source.md#RandomSource), [`_shape_generator`](random_source.md#_shape_generator)
- used by: [`RandomSource`](random_source.md#RandomSource), [`rounded_normal`](random_source.md#RandomSource.rounded_normal), [`uniform`](random_source.md#RandomSource.uniform), [`sk_uniform`](random_source.md#RandomSource.sk_uniform), [`ALL_RNGS`](random_source.md#ALL_RNGS), [`rng`](random_source_test.md#ConstantUniformRandomSourceTest.rng), [`test_rounded_normal_correct_type`](random_source_test.md#ConstantUniformRandomSourceTest.test_rounded_normal_correct_type), [`test_rounded_normal_valid_and_correct_shape`](random_source_test.md#ConstantUniformRandomSourceTest.test_rounded_normal_valid_and_correct_shape), [`test_uniform_valid_and_correct_shape`](random_source_test.md#ConstantUniformRandomSourceTest.test_uniform_valid_and_correct_shape)

### `CycleRng`  ·  implements/extends RandomSource
- def: [`jaxite/jaxite_cggi/random_source.py:89`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L89) — documented in [jaxite-jaxite_cggi-random_source](../../../concepts/jaxite-jaxite_cggi-random_source.md)
- doc: An RNG that cycles through a deterministic set of values.
- signature: `class CycleRng(RandomSource):`
- members:
  - `rounded_normal(self, shape: Sequence[int] = (1,), dtype: jnp.dtype = jnp.uint32)` — [`L108`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L108)
  - `sk_uniform(self, shape: Sequence[int] = (1,), dtype: jnp.dtype = jnp.uint32)` — [`L103`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L103)
  - `uniform(self, shape: Sequence[int] = (1,), dtype: jnp.dtype = jnp.uint32)` — [`L98`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L98) — documented in [jaxite-jaxite_cggi-random_source](../../../concepts/jaxite-jaxite_cggi-random_source.md)
  - `const_normal_noise` — [`L93`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L93)
  - `noise` — [`L96`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L96)
  - `uniform_data` — [`L95`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L95)
- protocol/private: `__init__`[`L92`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L92)
- uses (calls/refs, reference-scoped): [`RandomSource`](random_source.md#RandomSource), [`_shape_generator`](random_source.md#_shape_generator)
- used by: [`RandomSource`](random_source.md#RandomSource), [`test_deterministic_noise_encrypt`](lwe_test.md#LweEncryptDecryptTest.test_deterministic_noise_encrypt), [`rounded_normal`](random_source.md#RandomSource.rounded_normal), [`uniform`](random_source.md#RandomSource.uniform), [`sk_uniform`](random_source.md#RandomSource.sk_uniform), [`ALL_RNGS`](random_source.md#ALL_RNGS), [`noise_free_rng`](blind_rotate_test.md#BlindRotateTest.noise_free_rng), [`test_max_error_encrypt_decrypt`](lwe_test.md#LweEncryptDecryptTest.test_max_error_encrypt_decrypt), [`test_too_much_error_cannot_decrypt`](lwe_test.md#LweEncryptDecryptTest.test_too_much_error_cannot_decrypt), [`test_rounded_normal_matches_const_normal_noise`](random_source_test.md#CycleRngTest.test_rounded_normal_matches_const_normal_noise), [`noise_free_rng`](lwe_test.md#LweEncryptDecryptTest.noise_free_rng), [`test_uniform_matches_random_data`](random_source_test.md#CycleRngTest.test_uniform_matches_random_data), [`rng`](random_source_test.md#CycleRngTest.rng), [`noise_free_rng`](rgsw_test.md#RgswTest.noise_free_rng), [`noise_free_rng`](rlwe_test.md#RlweTest.noise_free_rng)

### `NormalOnlyRng`  ·  implements/extends RandomSource
- def: [`jaxite/jaxite_cggi/random_source.py:184`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L184) — documented in [jaxite-jaxite_cggi-random_source](../../../concepts/jaxite-jaxite_cggi-random_source.md)
- doc: A test-only RNG.
- signature: `class NormalOnlyRng(RandomSource):`
- members:
  - `rounded_normal(self, shape: Sequence[int] = (1,), dtype: jnp.dtype = jnp.uint32)` — [`L199`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L199)
  - `sk_uniform(self, shape: Sequence[int] = (1,), dtype: jnp.dtype = jnp.uint32)` — [`L208`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L208)
  - `uniform(self, shape: Sequence[int] = (1,), dtype: jnp.dtype = jnp.uint32)` — [`L194`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L194)
  - `normal_std` — [`L191`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L191)
  - `rng` — [`L192`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L192)
- protocol/private: `__init__`[`L190`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L190)
- uses (calls/refs, reference-scoped): [`RandomSource`](random_source.md#RandomSource), [`_shape_generator`](random_source.md#_shape_generator)
- used by: [`RandomSource`](random_source.md#RandomSource), [`rounded_normal`](random_source.md#RandomSource.rounded_normal), [`uniform`](random_source.md#RandomSource.uniform), [`sk_uniform`](random_source.md#RandomSource.sk_uniform), [`ALL_RNGS`](random_source.md#ALL_RNGS), [`rng`](random_source_test.md#NormalOnlyRandomSourceTest.rng), [`test_rounded_normal_correct_type`](random_source_test.md#NormalOnlyRandomSourceTest.test_rounded_normal_correct_type), [`test_rounded_normal_valid_and_correct_shape`](random_source_test.md#NormalOnlyRandomSourceTest.test_rounded_normal_valid_and_correct_shape), [`test_uniform_valid_and_correct_shape`](random_source_test.md#NormalOnlyRandomSourceTest.test_uniform_valid_and_correct_shape)

### `PseudorandomSource`  ·  implements/extends RandomSource
- def: [`jaxite/jaxite_cggi/random_source.py:151`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L151) — documented in [jaxite-jaxite_cggi-random_source](../../../concepts/jaxite-jaxite_cggi-random_source.md)
- doc: An insecure random source based on the Python stdlib.
- signature: `class PseudorandomSource(RandomSource):`
- members:
  - `rounded_normal(self, shape: Sequence[int] = (1,), dtype: jnp.dtype = jnp.uint32)` — [`L171`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L171) — documented in [jaxite-jaxite_cggi-random_source](../../../concepts/jaxite-jaxite_cggi-random_source.md)
  - `sk_uniform(self, shape: Sequence[int] = (1,), dtype: jnp.dtype = jnp.uint32)` — [`L178`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L178)
  - `uniform(self, shape: Sequence[int] = (1,), dtype: jnp.dtype = jnp.uint32)` — [`L164`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L164) — documented in [jaxite-jaxite_cggi-random_source](../../../concepts/jaxite-jaxite_cggi-random_source.md)
  - `normal_std` — [`L161`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L161)
  - `rng` — [`L162`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L162)
  - `uniform_bounds` — [`L160`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L160)
- protocol/private: `__init__`[`L154`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L154)
- uses (calls/refs, reference-scoped): [`RandomSource`](random_source.md#RandomSource), [`_shape_generator`](random_source.md#_shape_generator)
- used by: [`test_blind_rotate_with_noise`](blind_rotate_test.md#BlindRotateTest.test_blind_rotate_with_noise), [`test_cmux_noisy`](blind_rotate_test.md#BlindRotateTest.test_cmux_noisy), [`test_blind_rotate_doesnt_set_padding_bit`](blind_rotate_test.md#BlindRotateTest.test_blind_rotate_doesnt_set_padding_bit), [`test_sample_extract_noisy`](blind_rotate_test.md#BlindRotateTest.test_sample_extract_noisy), [`test_external_product_noisy_multiply`](blind_rotate_test.md#BlindRotateTest.test_external_product_noisy_multiply), [`test_modulus_switch_preserves_message`](lwe_test.md#LweModulusSwitchingTest.test_modulus_switch_preserves_message), [`test_encrypt_decrypt_with_varying_rngs`](rgsw_test.md#RgswTest.test_encrypt_decrypt_with_varying_rngs), [`test_encrypt_decrypt_with_varying_nonzero_sk_entry`](rgsw_test.md#RgswTest.test_encrypt_decrypt_with_varying_nonzero_sk_entry), [`RandomSource`](random_source.md#RandomSource), [`test_switch_key_error_free`](key_switch_test.md#KeySwitchTest.test_switch_key_error_free), [`test_switch_key_with_error`](key_switch_test.md#KeySwitchTest.test_switch_key_with_error), [`test_key_switch_gen`](key_switch_test.md#KeySwitchTest.test_key_switch_gen), [`rounded_normal`](random_source.md#RandomSource.rounded_normal), [`uniform`](random_source.md#RandomSource.uniform), [`sk_uniform`](random_source.md#RandomSource.sk_uniform), [`ALL_RNGS`](random_source.md#ALL_RNGS), [`test_3_bit_consistency`](bootstrap_test.md#ConsistencyTest.test_3_bit_consistency), [`test_uniform_elements_within_bounds`](random_source_test.md#test_uniform_elements_within_bounds), [`get_lwe_rng_for_128_bit_security`](../jaxite_bool/bool_params.md#get_lwe_rng_for_128_bit_security), [`get_rlwe_rng_for_128_bit_security`](../jaxite_bool/bool_params.md#get_rlwe_rng_for_128_bit_security), [`VARYING_MAGNITUDE_TEST_RNGS`](random_source.md#VARYING_MAGNITUDE_TEST_RNGS), [`CryptographicallySecureRandomSourceTest`](random_source_test.md#CryptographicallySecureRandomSourceTest), [`get_rng_for_test`](../jaxite_bool/bool_params.md#get_rng_for_test), [`test_3_bit_bootstrap`](bootstrap_test.md#BootstrapTest.test_3_bit_bootstrap), [`test_3_bit_bootstrap_larger_lwe_dimension`](bootstrap_test.md#BootstrapTest.test_3_bit_bootstrap_larger_lwe_dimension), [`test_3_bit_bootstrap_prod_decomp_params`](bootstrap_test.md#BootstrapTest.test_3_bit_bootstrap_prod_decomp_params)

### `RandomSource`  ·  implements/extends ABC
- def: [`jaxite/jaxite_cggi/random_source.py:66`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L66) — documented in [jaxite-jaxite_cggi-random_source](../../../concepts/jaxite-jaxite_cggi-random_source.md)
- doc: An interface for a source of randomness.
- signature: `class RandomSource(abc.ABC):`
- members:
  - `rounded_normal(self, shape: Sequence[int] = (1,), dtype: jnp.dtype = jnp.uint32)` — [`L76`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L76) — documented in [jaxite-jaxite_cggi-random_source](../../../concepts/jaxite-jaxite_cggi-random_source.md)
  - `sk_uniform(self, shape: Sequence[int] = (1,), dtype: jnp.dtype = jnp.uint32)` — [`L82`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L82) — Generates random numbers in {0, 1}, for use in generating secret keys. — documented in [jaxite-jaxite_cggi-lwe](../../../concepts/jaxite-jaxite_cggi-lwe.md)
  - `uniform(self, shape: Sequence[int] = (1,), dtype: jnp.dtype = jnp.uint32)` — [`L70`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L70) — documented in [jaxite-jaxite_cggi-random_source](../../../concepts/jaxite-jaxite_cggi-random_source.md)
- uses (calls/refs, reference-scoped): [`PseudorandomSource`](random_source.md#PseudorandomSource), [`CycleRng`](random_source.md#CycleRng), [`rounded_normal`](random_source.md#ConstantUniformRng.rounded_normal), [`rounded_normal`](random_source.md#NormalOnlyRng.rounded_normal), [`ZeroRng`](random_source.md#ZeroRng), [`rounded_normal`](random_source.md#PseudorandomSource.rounded_normal), [`rounded_normal`](random_source.md#SystemRandomSource.rounded_normal), [`uniform`](random_source.md#PseudorandomSource.uniform), [`uniform`](random_source.md#SystemRandomSource.uniform), [`uniform`](random_source.md#CycleRng.uniform), [`rounded_normal`](random_source.md#CycleRng.rounded_normal), [`SystemRandomSource`](random_source.md#SystemRandomSource), [`sk_uniform`](random_source.md#ConstantUniformRng.sk_uniform), [`sk_uniform`](random_source.md#NormalOnlyRng.sk_uniform), [`sk_uniform`](random_source.md#PseudorandomSource.sk_uniform), [`sk_uniform`](random_source.md#SystemRandomSource.sk_uniform), [`ConstantUniformRng`](random_source.md#ConstantUniformRng), [`NormalOnlyRng`](random_source.md#NormalOnlyRng), [`uniform`](random_source.md#ConstantUniformRng.uniform), [`rounded_normal`](random_source.md#ZeroRng.rounded_normal), [`sk_uniform`](random_source.md#CycleRng.sk_uniform), [`uniform`](random_source.md#NormalOnlyRng.uniform), [`uniform`](random_source.md#ZeroRng.uniform), [`sk_uniform`](random_source.md#ZeroRng.sk_uniform)
- used by: [`run_bootstrap_test`](bootstrap_test.md#BootstrapBaseTest.run_bootstrap_test), [`encrypt`](rgsw.md#encrypt), [`gen_bootstrapping_key`](bootstrap.md#gen_bootstrapping_key), [`encrypt`](rlwe.md#encrypt), [`gen_key`](key_switch.md#gen_key), [`encrypt`](../jaxite_bool/jaxite_bool.md#encrypt), [`encrypt`](lwe.md#encrypt), [`gen_key`](lwe.md#gen_key), [`gen_key`](rlwe.md#gen_key), [`PseudorandomSource`](random_source.md#PseudorandomSource), [`gen_key`](rgsw.md#gen_key), [`CycleRng`](random_source.md#CycleRng), [`test_uniform_elements_within_bounds`](random_source_test.md#test_uniform_elements_within_bounds), [`ZeroRng`](random_source.md#ZeroRng), [`__init__`](../jaxite_bool/jaxite_bool.md#ServerKeySet.__init__), [`SystemRandomSource`](random_source.md#SystemRandomSource), [`__init__`](../jaxite_bool/jaxite_bool.md#ClientKeySet.__init__), [`test_rounded_normal_correct_type`](random_source_test.md#CryptographicallySecureRandomSourceTest.test_rounded_normal_correct_type), [`test_rounded_normal_valid_and_correct_shape`](random_source_test.md#CryptographicallySecureRandomSourceTest.test_rounded_normal_valid_and_correct_shape), [`test_uniform_valid_and_correct_shape`](random_source_test.md#CryptographicallySecureRandomSourceTest.test_uniform_valid_and_correct_shape), [`ConstantUniformRng`](random_source.md#ConstantUniformRng), [`NormalOnlyRng`](random_source.md#NormalOnlyRng)

### `SystemRandomSource`  ·  implements/extends RandomSource
- def: [`jaxite/jaxite_cggi/random_source.py:114`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L114) — documented in [jaxite-jaxite_cggi-random_source](../../../concepts/jaxite-jaxite_cggi-random_source.md)
- doc: A cryptographically secure random source that runs on CPU serially.
- signature: `class SystemRandomSource(RandomSource):`
- members:
  - `rounded_normal(self, shape: Sequence[int] = (1,), dtype: jnp.dtype = jnp.uint32)` — [`L136`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L136)
  - `sk_uniform(self, shape: Sequence[int] = (1,), dtype: jnp.dtype = jnp.uint32)` — [`L145`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L145)
  - `uniform(self, shape: Sequence[int] = (1,), dtype: jnp.dtype = jnp.uint32)` — [`L129`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L129)
  - `normal_std` — [`L126`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L126)
  - `rng` — [`L127`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L127)
  - `uniform_bounds` — [`L125`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L125)
- protocol/private: `__init__`[`L120`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L120)
- uses (calls/refs, reference-scoped): [`RandomSource`](random_source.md#RandomSource), [`_shape_generator`](random_source.md#_shape_generator)
- used by: [`RandomSource`](random_source.md#RandomSource), [`rounded_normal`](random_source.md#RandomSource.rounded_normal), [`uniform`](random_source.md#RandomSource.uniform), [`sk_uniform`](random_source.md#RandomSource.sk_uniform), [`ALL_RNGS`](random_source.md#ALL_RNGS), [`test_uniform_elements_within_bounds`](random_source_test.md#test_uniform_elements_within_bounds), [`CryptographicallySecureRandomSourceTest`](random_source_test.md#CryptographicallySecureRandomSourceTest)

### `ZeroRng`  ·  implements/extends RandomSource
- def: [`jaxite/jaxite_cggi/random_source.py:242`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L242) — documented in [jaxite-jaxite_cggi-lwe](../../../concepts/jaxite-jaxite_cggi-lwe.md)
- doc: An random generator that always outputs zero.
- signature: `class ZeroRng(RandomSource):`
- members:
  - `rounded_normal(self, shape: Sequence[int] = (1,), dtype: jnp.dtype = jnp.uint32)` — [`L250`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L250)
  - `sk_uniform(self, shape: Sequence[int] = (1,), dtype: jnp.dtype = jnp.uint32)` — [`L255`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L255)
  - `uniform(self, shape: Sequence[int] = (1,), dtype: jnp.dtype = jnp.uint32)` — [`L245`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L245)
- uses (calls/refs, reference-scoped): [`RandomSource`](random_source.md#RandomSource)
- used by: [`RandomSource`](random_source.md#RandomSource), [`rounded_normal`](random_source.md#RandomSource.rounded_normal), [`uniform`](random_source.md#RandomSource.uniform), [`sk_uniform`](random_source.md#RandomSource.sk_uniform), [`ALL_RNGS`](random_source.md#ALL_RNGS), [`rng`](random_source_test.md#ZeroRandomSourceTest.rng), [`test_rounded_normal_correct_type`](random_source_test.md#ZeroRandomSourceTest.test_rounded_normal_correct_type), [`test_rounded_normal_elements_equal_zero`](random_source_test.md#ZeroRandomSourceTest.test_rounded_normal_elements_equal_zero), [`test_rounded_normal_valid_and_correct_shape`](random_source_test.md#ZeroRandomSourceTest.test_rounded_normal_valid_and_correct_shape), [`test_uniform_valid_and_correct_shape`](random_source_test.md#ZeroRandomSourceTest.test_uniform_valid_and_correct_shape), [`ZERO_RNG`](bootstrap_test.md#ZERO_RNG), [`ZERO_RNG`](key_switch_test.md#ZERO_RNG)  (1 test-only)

## Functions
- `_normalize_value(value: Any, dtype: jnp.dtype)` — [`L15`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L15) — Normalize value to the correct bounds for dtype.
- `_shape_generator(generator: Callable[[], Any], shape: Sequence[int], dtype: jnp.dtype = jnp.uint32)` — [`L33`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L33) — Generates an array of data generated by a given function. — documented in [jaxite-jaxite_cggi-random_source](../../../concepts/jaxite-jaxite_cggi-random_source.md)

## Module values
- `ALL_RNGS` — [`L279`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L279) — documented in [jaxite-jaxite_cggi-random_source](../../../concepts/jaxite-jaxite_cggi-random_source.md)
- `VARYING_MAGNITUDE_TEST_RNGS` — [`L274`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L274) — documented in [jaxite-jaxite_cggi-random_source](../../../concepts/jaxite-jaxite_cggi-random_source.md)
- `_varying_params` — [`L261`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source.py#L261)

