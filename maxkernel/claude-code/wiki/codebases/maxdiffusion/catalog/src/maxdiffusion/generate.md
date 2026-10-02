---
title: 'Module: src/maxdiffusion/generate.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/generate.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.generate`/
symbols:
  run_inference: run_inference().
  run: run().
  _run_with_mesh: _run_with_mesh().
  get_unet_inputs: get_unet_inputs().
  main: main().
  get_batch_sharding: get_batch_sharding().
  tokenize: tokenize().
  GenerateSD: GenerateSD#
  loop_body: loop_body().
  vae_decode: vae_decode().
  GenerateSD.__init__: GenerateSD#__init__().
  GenerateSD.post_training_steps: GenerateSD#post_training_steps().
---
# Module: [`src/maxdiffusion/generate.py`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/generate.py)

## Classes
### `GenerateSD`  ·  implements/extends StableDiffusionTrainer
- def: [`src/maxdiffusion/generate.py:41`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/generate.py#L41)
- signature: `class GenerateSD(StableDiffusionTrainer):`
- members:
  - `post_training_steps(self, pipeline, params, train_states)` — [`L46`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/generate.py#L46)
- protocol/private: `__init__`[`L43`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/generate.py#L43)
- used by: [`run`](generate.md#run)

## Functions
- `_run_with_mesh(config, checkpoint_loader)` — [`L184`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/generate.py#L184)
- `get_batch_sharding(mesh, config)` — [`L50`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/generate.py#L50) — Sharding for the batch dimension.
- `get_unet_inputs(pipeline, params, states, config, rng, batch_sharding, batch_size)` — [`L101`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/generate.py#L101)
- `loop_body(step, args, model, pipeline, prompt_embeds, guidance_scale, guidance_rescale, batch_sharding)` — [`L60`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/generate.py#L60)
- `main(argv: Sequence[str])` — [`L288`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/generate.py#L288)
- `run(config)` — [`L178`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/generate.py#L178)
- `run_inference(states, pipeline, params, config, rng, mesh, batch_size)` — [`L145`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/generate.py#L145)
- `tokenize(prompt, tokenizer)` — [`L94`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/generate.py#L94) — Tokenizes prompt.
- `vae_decode(latents, state, pipeline)` — [`L138`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/generate.py#L138)

