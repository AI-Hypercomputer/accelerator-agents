---
title: 'Module: src/maxdiffusion/models/ltx_video/utils/prompt_enhance_utils.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/utils/prompt_enhance_utils.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.utils.prompt_enhance_utils`/
symbols:
  generate_cinematic_prompt: generate_cinematic_prompt().
  _generate_i2v_prompt: _generate_i2v_prompt().
  _get_first_frames_from_conditioning_item: _get_first_frames_from_conditioning_item().
  _generate_t2v_prompt: _generate_t2v_prompt().
  _generate_and_decode_prompts: _generate_and_decode_prompts().
  logger: logger.
  T2V_CINEMATIC_PROMPT: T2V_CINEMATIC_PROMPT.
  I2V_CINEMATIC_PROMPT: I2V_CINEMATIC_PROMPT.
  tensor_to_pil: tensor_to_pil().
  _generate_image_captions: _generate_image_captions().
---
# Module: [`src/maxdiffusion/models/ltx_video/utils/prompt_enhance_utils.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/prompt_enhance_utils.py)

## Functions
- `_generate_and_decode_prompts(prompt_enhancer_model, prompt_enhancer_tokenizer, model_inputs, max_new_tokens: int)` — [`L198`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/prompt_enhance_utils.py#L198)
- `_generate_i2v_prompt(image_caption_model, image_caption_processor, prompt_enhancer_model, prompt_enhancer_tokenizer, prompts: List[str], first_frames: List[Image.Image], max_new_tokens: int, system_prompt: str)` — [`L151`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/prompt_enhance_utils.py#L151)
- `_generate_image_captions(image_caption_model, image_caption_processor, images: List[Image.Image], system_prompt: str = "<DETAILED_CAPTION>")` — [`L177`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/prompt_enhance_utils.py#L177)
- `_generate_t2v_prompt(prompt_enhancer_model, prompt_enhancer_tokenizer, prompts: List[str], max_new_tokens: int, system_prompt: str)` — [`L130`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/prompt_enhance_utils.py#L130)
- `_get_first_frames_from_conditioning_item(conditioning_item)` — [`L125`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/prompt_enhance_utils.py#L125)
- `generate_cinematic_prompt(image_caption_model, image_caption_processor, prompt_enhancer_model, prompt_enhancer_tokenizer, prompt: Union[str, List[str]], conditioning_items: Optional[List] = None, max_new_tokens: int = 256)` — [`L80`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/prompt_enhance_utils.py#L80)
- `tensor_to_pil(tensor)` — [`L63`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/prompt_enhance_utils.py#L63)

## Module values
- `I2V_CINEMATIC_PROMPT` — [`L43`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/prompt_enhance_utils.py#L43)
- `T2V_CINEMATIC_PROMPT` — [`L25`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/prompt_enhance_utils.py#L25)
- `logger` — [`L23`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/prompt_enhance_utils.py#L23)

