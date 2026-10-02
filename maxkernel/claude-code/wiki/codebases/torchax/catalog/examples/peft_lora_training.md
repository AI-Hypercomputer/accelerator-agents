---
title: 'Module: examples/peft_lora_training.py'
type: catalog
provenance: extracted
module: examples/peft_lora_training.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `examples.peft_lora_training`/
symbols:
  main: main().
  train_model: train_model().
  run_inference: run_inference().
  tmp_dir: tmp_dir.
  load_datasets: load_datasets().
  load_datasets.format_example: load_datasets().format_example().
  create_peft_model: create_peft_model().
  train_model.model_fn: train_model().model_fn().
  train_model.loss_fn: train_model().loss_fn().
  save_model_and_tokenizer: save_model_and_tokenizer().
---
# Module: [`examples/peft_lora_training.py`](../../../../../raw/code/torchax/examples/peft_lora_training.py)

## Functions
- `create_peft_model(model_path: str)` — [`L107`](../../../../../raw/code/torchax/examples/peft_lora_training.py#L107) — Loads base model and applies PEFT configuration.
- `format_example(example)` — [`L72`](../../../../../raw/code/torchax/examples/peft_lora_training.py#L72)
- `load_datasets(model_path: str, dataset_path: str)` — [`L62`](../../../../../raw/code/torchax/examples/peft_lora_training.py#L62) — Loads tokenizer and training dataloader.
- `loss_fn(model_output, labels)` — [`L147`](../../../../../raw/code/torchax/examples/peft_lora_training.py#L147)
- `main()` — [`L227`](../../../../../raw/code/torchax/examples/peft_lora_training.py#L227)
- `model_fn(weights, buffers, batch)` — [`L142`](../../../../../raw/code/torchax/examples/peft_lora_training.py#L142)
- `run_inference(model_path: str, peft_path: str, dataloader: DataLoader, device: torch.device)` — [`L185`](../../../../../raw/code/torchax/examples/peft_lora_training.py#L185) — Loads the saved model and runs inference on one batch.
- `save_model_and_tokenizer(model: torch.nn.Module, tokenizer: transformers.PreTrainedTokenizer, params: Dict[str, Any], save_path: str)` — [`L165`](../../../../../raw/code/torchax/examples/peft_lora_training.py#L165) — Saves the trained PEFT model and tokenizer.
- `train_model(model: torch.nn.Module, dataloader: DataLoader, device: torch.device)` — [`L125`](../../../../../raw/code/torchax/examples/peft_lora_training.py#L125) — Runs the training loop and returns trained parameters.

## Module values
- `tmp_dir` — [`L45`](../../../../../raw/code/torchax/examples/peft_lora_training.py#L45)

