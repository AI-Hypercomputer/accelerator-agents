---
title: 'Module: jaxite/jaxite_cggi/random_source_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/random_source_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.random_source_test`/
symbols:
  CycleRngTest.test_rounded_normal_matches_const_normal_noise: CycleRngTest#test_rounded_normal_matches_const_normal_noise().
  test_uniform_elements_within_bounds: test_uniform_elements_within_bounds().
  ConstantUniformRandomSourceTest.rng: ConstantUniformRandomSourceTest#rng.
  CycleRngTest.test_uniform_matches_random_data: CycleRngTest#test_uniform_matches_random_data().
  CycleRngTest.rng: CycleRngTest#rng.
  ZeroRandomSourceTest.rng: ZeroRandomSourceTest#rng.
  NormalOnlyRandomSourceTest.rng: NormalOnlyRandomSourceTest#rng.
  ShapeGeneratorTest.test_valid_shape: ShapeGeneratorTest#test_valid_shape().
  ShapeGeneratorTest.test_1d_shape_is_valid: ShapeGeneratorTest#test_1d_shape_is_valid().
  ShapeGeneratorTest.test_nd_shape_is_valid: ShapeGeneratorTest#test_nd_shape_is_valid().
  ShapeGeneratorTest.test_invalid_shape: ShapeGeneratorTest#test_invalid_shape().
  CryptographicallySecureRandomSourceTest: CryptographicallySecureRandomSourceTest#
  CryptographicallySecureRandomSourceTest.test_uniform_valid_and_correct_shape: CryptographicallySecureRandomSourceTest#test_uniform_valid_and_correct_shape().
  CryptographicallySecureRandomSourceTest.test_rounded_normal_valid_and_correct_shape: CryptographicallySecureRandomSourceTest#test_rounded_normal_valid_and_correct_shape().
  CryptographicallySecureRandomSourceTest.test_rounded_normal_correct_type: CryptographicallySecureRandomSourceTest#test_rounded_normal_correct_type().
  NormalOnlyRandomSourceTest.test_uniform_valid_and_correct_shape: NormalOnlyRandomSourceTest#test_uniform_valid_and_correct_shape().
  NormalOnlyRandomSourceTest.test_rounded_normal_valid_and_correct_shape: NormalOnlyRandomSourceTest#test_rounded_normal_valid_and_correct_shape().
  NormalOnlyRandomSourceTest.test_rounded_normal_correct_type: NormalOnlyRandomSourceTest#test_rounded_normal_correct_type().
  ConstantUniformRandomSourceTest.test_uniform_valid_and_correct_shape: ConstantUniformRandomSourceTest#test_uniform_valid_and_correct_shape().
  ConstantUniformRandomSourceTest.test_rounded_normal_valid_and_correct_shape: ConstantUniformRandomSourceTest#test_rounded_normal_valid_and_correct_shape().
  ConstantUniformRandomSourceTest.test_rounded_normal_correct_type: ConstantUniformRandomSourceTest#test_rounded_normal_correct_type().
  ZeroRandomSourceTest.test_uniform_valid_and_correct_shape: ZeroRandomSourceTest#test_uniform_valid_and_correct_shape().
  ZeroRandomSourceTest.test_rounded_normal_valid_and_correct_shape: ZeroRandomSourceTest#test_rounded_normal_valid_and_correct_shape().
  ZeroRandomSourceTest.test_rounded_normal_elements_equal_zero: ZeroRandomSourceTest#test_rounded_normal_elements_equal_zero().
  ZeroRandomSourceTest.test_rounded_normal_correct_type: ZeroRandomSourceTest#test_rounded_normal_correct_type().
  ShapeGeneratorTest.constant_function: ShapeGeneratorTest#constant_function.
  CycleRngTest.test_shape: CycleRngTest#test_shape.
  AllRngsTest.test_sk_uniform_is_binary: AllRngsTest#test_sk_uniform_is_binary().
  CycleRngTest.const_normal_noise: CycleRngTest#const_normal_noise.
  ConstantUniformRandomSourceTest.const_uniform: ConstantUniformRandomSourceTest#const_uniform.
  ShapeGeneratorTest: ShapeGeneratorTest#
  ShapeGeneratorTest.setUp: ShapeGeneratorTest#setUp().
  AllRngsTest: AllRngsTest#
  CycleRngTest: CycleRngTest#
  CycleRngTest.setUp: CycleRngTest#setUp().
  NormalOnlyRandomSourceTest: NormalOnlyRandomSourceTest#
  NormalOnlyRandomSourceTest.setUp: NormalOnlyRandomSourceTest#setUp().
  ConstantUniformRandomSourceTest: ConstantUniformRandomSourceTest#
  ConstantUniformRandomSourceTest.setUp: ConstantUniformRandomSourceTest#setUp().
  ZeroRandomSourceTest: ZeroRandomSourceTest#
  ZeroRandomSourceTest.setUp: ZeroRandomSourceTest#setUp().
---
# Module: [`jaxite/jaxite_cggi/random_source_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py)

## Classes
### `AllRngsTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_cggi/random_source_test.py:36`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L36)
- signature: `class AllRngsTest(absltest.TestCase):`
- members:
  - `test_sk_uniform_is_binary(self)` — [`L38`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L38)
- uses (calls/refs, reference-scoped): [`ALL_RNGS`](random_source.md#ALL_RNGS)

### `ConstantUniformRandomSourceTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_cggi/random_source_test.py:131`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L131)
- signature: `class ConstantUniformRandomSourceTest(absltest.TestCase):`
- members:
  - `setUp(self)` — [`L133`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L133)
  - `test_rounded_normal_correct_type(self)` — [`L152`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L152)
  - `test_rounded_normal_valid_and_correct_shape(self)` — [`L147`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L147)
  - `test_uniform_valid_and_correct_shape(self)` — [`L142`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L142)
  - `const_uniform` — [`L137`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L137)
  - `rng` — [`L138`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L138)
- uses (calls/refs, reference-scoped): [`rounded_normal`](random_source.md#ConstantUniformRng.rounded_normal), [`ConstantUniformRng`](random_source.md#ConstantUniformRng), [`uniform`](random_source.md#ConstantUniformRng.uniform)

### `CryptographicallySecureRandomSourceTest`
- def: [`jaxite/jaxite_cggi/random_source_test.py:70`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L70)
- doc: Test cryptographically-secure random sources.
- signature: `class CryptographicallySecureRandomSourceTest(parameterized.TestCase):`
- members:
  - `test_rounded_normal_correct_type(self, rng: random_source.RandomSource)` — [`L91`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L91)
  - `test_rounded_normal_valid_and_correct_shape(self, rng: random_source.RandomSource)` — [`L84`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L84)
  - `test_uniform_valid_and_correct_shape(self, rng: random_source.RandomSource)` — [`L77`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L77)
- uses (calls/refs, reference-scoped): [`RandomSource`](random_source.md#RandomSource), [`PseudorandomSource`](random_source.md#PseudorandomSource), [`rounded_normal`](random_source.md#RandomSource.rounded_normal), [`uniform`](random_source.md#RandomSource.uniform), [`SystemRandomSource`](random_source.md#SystemRandomSource)

### `CycleRngTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_cggi/random_source_test.py:45`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L45)
- signature: `class CycleRngTest(absltest.TestCase):`
- members:
  - `setUp(self)` — [`L47`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L47)
  - `test_rounded_normal_matches_const_normal_noise(self)` — [`L57`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L57)
  - `test_uniform_matches_random_data(self)` — [`L53`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L53)
  - `const_normal_noise` — [`L50`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L50)
  - `rng` — [`L51`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L51)
  - `test_shape` — [`L49`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L49)
- uses (calls/refs, reference-scoped): [`CycleRng`](random_source.md#CycleRng), [`uniform`](random_source.md#CycleRng.uniform), [`rounded_normal`](random_source.md#CycleRng.rounded_normal)

### `NormalOnlyRandomSourceTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_cggi/random_source_test.py:107`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L107)
- signature: `class NormalOnlyRandomSourceTest(absltest.TestCase):`
- members:
  - `setUp(self)` — [`L109`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L109)
  - `test_rounded_normal_correct_type(self)` — [`L125`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L125)
  - `test_rounded_normal_valid_and_correct_shape(self)` — [`L120`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L120)
  - `test_uniform_valid_and_correct_shape(self)` — [`L115`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L115)
  - `rng` — [`L113`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L113)
- uses (calls/refs, reference-scoped): [`rounded_normal`](random_source.md#NormalOnlyRng.rounded_normal), [`NormalOnlyRng`](random_source.md#NormalOnlyRng), [`uniform`](random_source.md#NormalOnlyRng.uniform)

### `ShapeGeneratorTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_cggi/random_source_test.py:9`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L9)
- signature: `class ShapeGeneratorTest(absltest.TestCase):`
- members:
  - `setUp(self)` — [`L11`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L11)
  - `test_1d_shape_is_valid(self)` — [`L20`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L20)
  - `test_invalid_shape(self)` — [`L30`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L30)
  - `test_nd_shape_is_valid(self)` — [`L25`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L25)
  - `test_valid_shape(self)` — [`L15`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L15)
  - `constant_function` — [`L13`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L13)
- uses (calls/refs, reference-scoped): [`_shape_generator`](random_source.md#_shape_generator)

### `ZeroRandomSourceTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_cggi/random_source_test.py:158`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L158)
- signature: `class ZeroRandomSourceTest(absltest.TestCase):`
- members:
  - `setUp(self)` — [`L160`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L160)
  - `test_rounded_normal_correct_type(self)` — [`L181`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L181)
  - `test_rounded_normal_elements_equal_zero(self)` — [`L176`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L176)
  - `test_rounded_normal_valid_and_correct_shape(self)` — [`L171`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L171)
  - `test_uniform_valid_and_correct_shape(self)` — [`L166`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L166)
  - `rng` — [`L164`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L164)
- uses (calls/refs, reference-scoped): [`ZeroRng`](random_source.md#ZeroRng), [`rounded_normal`](random_source.md#ZeroRng.rounded_normal), [`uniform`](random_source.md#ZeroRng.uniform)

## Functions
- `test_uniform_elements_within_bounds(self, rng: random_source.RandomSource)` — [`L101`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/random_source_test.py#L101)

