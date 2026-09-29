---
title: 'Module: easydel/inference/vwhisper/server.py'
type: catalog
provenance: extracted
module: easydel/inference/vwhisper/server.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.vwhisper.server`/
symbols:
  create_whisper_app.create_transcription: create_whisper_app().create_transcription().
  WhisperModel.inference: WhisperModel#inference.
  create_whisper_app.create_translation: create_whisper_app().create_translation().
  ResponseFormat: ResponseFormat#
  run_server: run_server().
  WhisperModel._instance: WhisperModel#_instance.
  WhisperModel.model: WhisperModel#model.
  WhisperModel.__new__: WhisperModel#__new__().
  create_whisper_app: create_whisper_app().
  ResponseFormat.json: ResponseFormat#json.
  WhisperModel: WhisperModel#
  ResponseFormat.text: ResponseFormat#text.
  WhisperModel._initialize: WhisperModel#_initialize().
  WhisperModel.tokenizer: WhisperModel#tokenizer.
  WhisperModel.processor: WhisperModel#processor.
  TranscriptionResponse: TranscriptionResponse#
  ResponseFormat.srt: ResponseFormat#srt.
  ResponseFormat.verbose_json: ResponseFormat#verbose_json.
  ResponseFormat.vtt: ResponseFormat#vtt.
  TranscriptionResponse.text: TranscriptionResponse#text.
  TranscriptionResponse.segments: TranscriptionResponse#segments.
  create_whisper_app.read_root: create_whisper_app().read_root().
---
# Module: [`easydel/inference/vwhisper/server.py`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py)

## Classes
### `ResponseFormat`  ·  implements/extends StrEnum
- def: [`easydel/inference/vwhisper/server.py:204`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py#L204)
- doc: Supported response formats for transcription API.
- signature: `class ResponseFormat(StrEnum):`
- members:
  - `json` — [`L231`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py#L231)
  - `srt` — [`L233`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py#L233)
  - `text` — [`L232`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py#L232)
  - `verbose_json` — [`L234`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py#L234)
  - `vtt` — [`L235`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py#L235)
- used by: [`create_transcription`](server.md#create_whisper_app.create_transcription), [`create_translation`](server.md#create_whisper_app.create_translation)

### `TranscriptionResponse`  ·  implements/extends BaseModel
- def: [`easydel/inference/vwhisper/server.py:238`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py#L238)
- doc: Pydantic model for transcription API responses.
- signature: `class TranscriptionResponse(BaseModel):`
- members:
  - `segments` — [`L271`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py#L271)
  - `text` — [`L270`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py#L270)
- used by: [`create_transcription`](server.md#create_whisper_app.create_transcription)

### `WhisperModel`
- def: [`easydel/inference/vwhisper/server.py:81`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py#L81)
- doc: Singleton wrapper for the Whisper model.
- signature: `class WhisperModel:`
- members:
  - `__new__(cls, model_name=None, dtype=jnp.bfloat16)` — [`L130`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py#L130) — Create or return the singleton WhisperModel instance.
  - `_initialize(self)` — [`L165`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py#L165) — Initialize model components.
  - `inference` — [`L196`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py#L196)
  - `model` — [`L188`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py#L188)
  - `processor` — [`L194`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py#L194)
  - `tokenizer` — [`L193`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py#L193)
- protocol/private: `_instance`[`L128`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py#L128)
- uses (calls/refs, reference-scoped): [`from_pretrained`](../../modules/auto/auto_modeling.md#BaseAutoEasyModel.from_pretrained), [`vWhisperInference`](core.md#vWhisperInference), [`AutoEasyDeLModelForSpeechSeq2Seq`](../../modules/auto/auto_modeling.md#AutoEasyDeLModelForSpeechSeq2Seq)
- used by: [`create_transcription`](server.md#create_whisper_app.create_transcription), [`create_translation`](server.md#create_whisper_app.create_translation), [`create_whisper_app`](server.md#create_whisper_app)

## Functions
- `create_transcription(file: UploadFile = File(...), model: str = Form(model_name), prompt: str | None = Form(None), response_format: ResponseFormat = Form(ResponseFormat.json), temperature: float = Form(0), language: str | None = Form(None), timestamp_granularities: list[str] | None = Form(None))` — [`L346`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py#L346) — Transcribe audio to text using the Whisper model.
- `create_translation(file: UploadFile = File(...), model: str = Form(model_name), prompt: str | None = Form(None), response_format: ResponseFormat = Form(ResponseFormat.json), temperature: float = Form(0), timestamp_granularities: list[str] | None = Form(None))` — [`L442`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py#L442) — Translate audio to English text using the Whisper model.
- `create_whisper_app(model_name: str = "openai/whisper-large-v3-turbo", dtype=jnp.bfloat16)` — [`L274`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py#L274) — Create a FastAPI application for Whisper transcription.
- `read_root()` — [`L335`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py#L335) — Root endpoint for health check and API information.
- `run_server(model_name: str = "openai/whisper-large-v3-turbo", host: str = "0.0.0.0", port: int = 8000, dtype=jnp.bfloat16)` — [`L540`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/server.py#L540) — Run the vWhisper FastAPI server.

