---
title: 'Module: src/alphafold3/model/confidence_types.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/confidence_types.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.confidence_types`/
symbols:
  AtomConfidence.from_inference_result: AtomConfidence#from_inference_result().
  StructureConfidenceFull.from_inference_result: StructureConfidenceFull#from_inference_result().
  StructureConfidenceFull.to_json: StructureConfidenceFull#to_json().
  ConfidenceCategory.from_char: ConfidenceCategory#from_char().
  ConfidenceCategory.from_confidence_score: ConfidenceCategory#from_confidence_score().
  ConfidenceCategory.to_char: ConfidenceCategory#to_char().
  AtomConfidence.__post_init__: AtomConfidence#__post_init__().
  StructureConfidenceSummary.from_inference_result: StructureConfidenceSummary#from_inference_result().
  StructureConfidenceSummary.to_json: StructureConfidenceSummary#to_json().
  AtomConfidence.from_json: AtomConfidence#from_json().
  AtomConfidence.confidence_category: AtomConfidence#confidence_category.
  ConfidenceCategory: ConfidenceCategory#
  ConfidenceCategory.HIGH: ConfidenceCategory#HIGH.
  ConfidenceCategory.MEDIUM: ConfidenceCategory#MEDIUM.
  ConfidenceCategory.LOW: ConfidenceCategory#LOW.
  ConfidenceCategory.DISORDERED: ConfidenceCategory#DISORDERED.
  AtomConfidence.to_json: AtomConfidence#to_json().
  _dump_json: _dump_json().
  AtomConfidence.confidence: AtomConfidence#confidence.
  AtomConfidence: AtomConfidence#
  AtomConfidence.chain_id: AtomConfidence#chain_id.
  AtomConfidence.atom_number: AtomConfidence#atom_number.
  StructureConfidenceSummary: StructureConfidenceSummary#
  StructureConfidenceSummary.convert: StructureConfidenceSummary#convert().
  StructureConfidenceFull: StructureConfidenceFull#
  StructureConfidenceFull.pae: StructureConfidenceFull#pae.
  StructureConfidenceFull.token_chain_ids: StructureConfidenceFull#token_chain_ids.
  StructureConfidenceFull.token_res_ids: StructureConfidenceFull#token_res_ids.
  StructureConfidenceFull.atom_plddts: StructureConfidenceFull#atom_plddts.
  StructureConfidenceFull.atom_chain_ids: StructureConfidenceFull#atom_chain_ids.
  StructureConfidenceFull.contact_probs: StructureConfidenceFull#contact_probs.
  StructureConfidenceSummary.ptm: StructureConfidenceSummary#ptm.
  StructureConfidenceSummary.iptm: StructureConfidenceSummary#iptm.
  StructureConfidenceSummary.ranking_score: StructureConfidenceSummary#ranking_score.
  StructureConfidenceSummary.fraction_disordered: StructureConfidenceSummary#fraction_disordered.
  StructureConfidenceSummary.has_clash: StructureConfidenceSummary#has_clash.
  StructureConfidenceSummary.chain_pair_pae_min: StructureConfidenceSummary#chain_pair_pae_min.
  StructureConfidenceSummary.chain_pair_iptm: StructureConfidenceSummary#chain_pair_iptm.
  StructureConfidenceSummary.chain_ptm: StructureConfidenceSummary#chain_ptm.
  StructureConfidenceSummary.chain_iptm: StructureConfidenceSummary#chain_iptm.
  StructureConfidenceSummary.from_json: StructureConfidenceSummary#from_json().
  StructureConfidenceFull.from_json: StructureConfidenceFull#from_json().
---
# Module: [`src/alphafold3/model/confidence_types.py`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py)

## Classes
### `AtomConfidence`
- def: [`src/alphafold3/model/confidence_types.py:95`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L95)
- doc: Dataclass for 1D per-atom confidences from AlphaFold.
- signature: `class AtomConfidence:`
- members:
  - `from_inference_result(cls, inference_result: model.InferenceResult)` — [`L112`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L112) — Instantiates an AtomConfidence from a structure. — documented in [alphafold3-model-confidence_types](../../../../concepts/alphafold3-model-confidence_types.md)
  - `from_json(cls, json_string: str)` — [`L141`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L141) — Instantiates a AtomConfidence from a json string.
  - `to_json(self)` — [`L150`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L150)
  - `atom_number` — [`L99`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L99)
  - `chain_id` — [`L98`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L98)
  - `confidence` — [`L100`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L100)
  - `confidence_category` — [`L101`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L101)
- protocol/private: `__post_init__`[`L103`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L103)
- uses (calls/refs, reference-scoped): [`iter_atoms`](../structure/structure.md#Structure.iter_atoms), [`from_char`](confidence_types.md#ConfidenceCategory.from_char), [`from_confidence_score`](confidence_types.md#ConfidenceCategory.from_confidence_score), [`atom_b_factor`](../structure/structure.md#Structure.atom_b_factor), [`InferenceResult`](model.md#InferenceResult), [`predicted_structure`](model.md#InferenceResult.predicted_structure), [`ConfidenceCategory`](confidence_types.md#ConfidenceCategory), [`_dump_json`](confidence_types.md#_dump_json)
- used by: [`post_process_inference_result`](post_processing.md#post_process_inference_result)

### `ConfidenceCategory`  ·  implements/extends Enum
- def: [`src/alphafold3/model/confidence_types.py:46`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L46) — documented in [alphafold3-model-confidence_types](../../../../concepts/alphafold3-model-confidence_types.md)
- doc: Confidence categories for AlphaFold predictions.
- signature: `class ConfidenceCategory(enum.Enum):`
- members:
  - `from_char(cls, char: str)` — [`L55`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L55)
  - `from_confidence_score(cls, confidence: float)` — [`L82`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L82) — documented in [alphafold3-model-confidence_types](../../../../concepts/alphafold3-model-confidence_types.md)
  - `to_char(self)` — [`L70`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L70)
  - `DISORDERED` — [`L52`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L52) — documented in [alphafold3-model-confidence_types](../../../../concepts/alphafold3-model-confidence_types.md)
  - `HIGH` — [`L49`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L49) — documented in [alphafold3-model-confidence_types](../../../../concepts/alphafold3-model-confidence_types.md)
  - `LOW` — [`L51`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L51) — documented in [alphafold3-model-confidence_types](../../../../concepts/alphafold3-model-confidence_types.md)
  - `MEDIUM` — [`L50`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L50) — documented in [alphafold3-model-confidence_types](../../../../concepts/alphafold3-model-confidence_types.md)
- used by: [`from_inference_result`](confidence_types.md#AtomConfidence.from_inference_result), [`from_json`](confidence_types.md#AtomConfidence.from_json), [`confidence_category`](confidence_types.md#AtomConfidence.confidence_category)

### `StructureConfidenceFull`
- def: [`src/alphafold3/model/confidence_types.py:223`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L223)
- doc: Dataclass for full structure data from AlphaFold.
- signature: `class StructureConfidenceFull:`
- members:
  - `from_inference_result(cls, inference_result: model.InferenceResult)` — [`L234`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L234) — Returns a new instance based on a given inference result. — documented in [alphafold3-model-confidence_types](../../../../concepts/alphafold3-model-confidence_types.md)
  - `from_json(cls, json_string: str)` — [`L269`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L269) — Returns a new instance from a given json string.
  - `to_json(self)` — [`L273`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L273) — Converts StructureConfidenceFull to json string. — documented in [alphafold3-model-confidence_types](../../../../concepts/alphafold3-model-confidence_types.md)
  - `atom_chain_ids` — [`L230`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L230) — documented in [alphafold3-model-confidence_types](../../../../concepts/alphafold3-model-confidence_types.md)
  - `atom_plddts` — [`L229`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L229) — documented in [alphafold3-model-confidence_types](../../../../concepts/alphafold3-model-confidence_types.md)
  - `contact_probs` — [`L231`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L231) — documented in [alphafold3-model-confidence_types](../../../../concepts/alphafold3-model-confidence_types.md)
  - `pae` — [`L226`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L226) — documented in [alphafold3-model-confidence_types](../../../../concepts/alphafold3-model-confidence_types.md)
  - `token_chain_ids` — [`L227`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L227) — documented in [alphafold3-model-confidence_types](../../../../concepts/alphafold3-model-confidence_types.md)
  - `token_res_ids` — [`L228`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L228) — documented in [alphafold3-model-confidence_types](../../../../concepts/alphafold3-model-confidence_types.md)
- uses (calls/refs, reference-scoped): [`chain_id`](../structure/structure.md#Structure.chain_id), [`metadata`](model.md#InferenceResult.metadata), [`atom_b_factor`](../structure/structure.md#Structure.atom_b_factor), [`InferenceResult`](model.md#InferenceResult), [`predicted_structure`](model.md#InferenceResult.predicted_structure), [`numerical_data`](model.md#InferenceResult.numerical_data)
- used by: [`post_process_inference_result`](post_processing.md#post_process_inference_result)

### `StructureConfidenceSummary`
- def: [`src/alphafold3/model/confidence_types.py:160`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L160)
- doc: Dataclass for the summary of structure scores from AlphaFold.
- signature: `class StructureConfidenceSummary:`
- members:
  - `convert(data)` — [`L210`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L210)
  - `from_inference_result(cls, inference_result: model.InferenceResult)` — [`L186`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L186) — Returns a new instance based on a given inference result.
  - `from_json(cls, json_string: str)` — [`L205`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L205) — Returns a new instance from a given json string.
  - `to_json(self)` — [`L209`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L209)
  - `chain_iptm` — [`L183`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L183)
  - `chain_pair_iptm` — [`L181`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L181)
  - `chain_pair_pae_min` — [`L180`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L180)
  - `chain_ptm` — [`L182`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L182)
  - `fraction_disordered` — [`L178`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L178)
  - `has_clash` — [`L179`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L179)
  - `iptm` — [`L176`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L176)
  - `ptm` — [`L175`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L175)
  - `ranking_score` — [`L177`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L177)
- uses (calls/refs, reference-scoped): [`metadata`](model.md#InferenceResult.metadata), [`InferenceResult`](model.md#InferenceResult), [`_dump_json`](confidence_types.md#_dump_json)
- used by: [`post_process_inference_result`](post_processing.md#post_process_inference_result)

## Functions
- `_dump_json(data: Any, indent: int | None = None)` — [`L34`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/confidence_types.py#L34) — Dumps a json string with JSON compatible NaN representation.

