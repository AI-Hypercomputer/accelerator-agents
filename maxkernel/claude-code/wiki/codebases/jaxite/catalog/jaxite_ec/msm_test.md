---
title: 'Module: jaxite_ec/msm_test.py'
type: catalog
provenance: extracted
module: jaxite_ec/msm_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite_ec.msm_test`/
symbols:
  script_dir: script_dir.
  TEST_PARAMS: TEST_PARAMS.
  MSMTest.test_pippenger_index_selection_twisted_edwards: MSMTest#test_pippenger_index_selection_twisted_edwards().
  MSMTest.test_pippenger_signed_index_selection_twisted_edwards: MSMTest#test_pippenger_signed_index_selection_twisted_edwards().
  MSMTest.test_pippenger_index_selection_rns_twisted_edwards: MSMTest#test_pippenger_index_selection_rns_twisted_edwards().
  MSMTest.read_external_file: MSMTest#read_external_file().
  MSMTest.test_pippenger_index_selection: MSMTest#test_pippenger_index_selection().
  MSMTest.test_pippenger_index_rns_selection: MSMTest#test_pippenger_index_rns_selection().
  twist_coordinates_list: twist_coordinates_list().
  script_path: script_path.
  MSMTest: MSMTest#
---
# Module: [`jaxite_ec/msm_test.py`](../../../../../raw/code/jaxite/jaxite_ec/msm_test.py)

## Classes
### `MSMTest`  ·  implements/extends TestCase
- def: [`jaxite_ec/msm_test.py:68`](../../../../../raw/code/jaxite/jaxite_ec/msm_test.py#L68)
- signature: `class MSMTest(parameterized.TestCase):`
- members:
  - `read_external_file(self, scalar_path, base_path, result_path)` — [`L69`](../../../../../raw/code/jaxite/jaxite_ec/msm_test.py#L69)
  - `test_pippenger_index_rns_selection(self, scalar_path, base_path, result_path)` — [`L483`](../../../../../raw/code/jaxite/jaxite_ec/msm_test.py#L483) — RNS version Pippenger - XYZZ.
  - `test_pippenger_index_selection(self, scalar_path, base_path, result_path)` — [`L97`](../../../../../raw/code/jaxite/jaxite_ec/msm_test.py#L97) — Normal version Pippenger.
  - `test_pippenger_index_selection_rns_twisted_edwards(self, scalar_path, base_path, result_path)` — [`L592`](../../../../../raw/code/jaxite/jaxite_ec/msm_test.py#L592)
  - `test_pippenger_index_selection_twisted_edwards(self, scalar_path, base_path, result_path)` — [`L205`](../../../../../raw/code/jaxite/jaxite_ec/msm_test.py#L205)
  - `test_pippenger_signed_index_selection_twisted_edwards(self, scalar_path, base_path, result_path)` — [`L342`](../../../../../raw/code/jaxite/jaxite_ec/msm_test.py#L342)
- uses (calls/refs, reference-scoped): [`TEST_PARAMS`](msm_test.md#TEST_PARAMS), [`twist_coordinates_list`](msm_test.md#twist_coordinates_list)

## Functions
- `twist_coordinates_list(ec_config, coordinates_list)` — [`L56`](../../../../../raw/code/jaxite/jaxite_ec/msm_test.py#L56)

## Module values
- `TEST_PARAMS` — [`L28`](../../../../../raw/code/jaxite/jaxite_ec/msm_test.py#L28)
- `script_dir` — [`L19`](../../../../../raw/code/jaxite/jaxite_ec/msm_test.py#L19)
- `script_path` — [`L18`](../../../../../raw/code/jaxite/jaxite_ec/msm_test.py#L18)

