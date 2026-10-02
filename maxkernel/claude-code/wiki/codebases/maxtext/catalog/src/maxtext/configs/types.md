---
title: 'Module: src/maxtext/configs/types.py'
type: catalog
provenance: extracted
module: src/maxtext/configs/types.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.configs.types`/
symbols:
  MaxTextConfig.set_derived_and_validate_values: MaxTextConfig#set_derived_and_validate_values().
  MaxTextConfig: MaxTextConfig#
  RematLocation: RematLocation#
  PathStr: PathStr.
  RematLocation.REMAT: RematLocation#REMAT.
  MaxTextConfig.validate_and_set_hlo_dump_defaults: MaxTextConfig#validate_and_set_hlo_dump_defaults().
  MaxTextConfig.get_num_target_devices: MaxTextConfig#get_num_target_devices().
  Quantization.quantization: Quantization#quantization.
  MaxTextConfig._validate_check_vma_is_supported: MaxTextConfig#_validate_check_vma_is_supported().
  ModelArchitecture.decoder_block: ModelArchitecture#decoder_block.
  RunInfo.model_name: RunInfo#model_name.
  HardwareAndMesh.mesh_axes: HardwareAndMesh#mesh_axes.
  PipelineParallelism.num_pipeline_microbatches: PipelineParallelism#num_pipeline_microbatches.
  MultimodalGeneral.validate_video_max_grid: MultimodalGeneral#validate_video_max_grid().
  MaxTextConfig.validate_ragged_buffer_factor: MaxTextConfig#validate_ragged_buffer_factor().
  DatasetGeneral.dataset_type: DatasetGeneral#dataset_type.
  RunInfo.run_name: RunInfo#run_name.
  LayoutAndSharding.data_sharding: LayoutAndSharding#data_sharding.
  PipelineParallelism.pipeline_parallel_layers: PipelineParallelism#pipeline_parallel_layers.
  Distillation.distill_beta_end: Distillation#distill_beta_end.
  RunInfo.base_output_directory: RunInfo#base_output_directory.
  HloDump.dump_hlo_gcs_dir: HloDump#dump_hlo_gcs_dir.
  HloDump.dump_jaxpr_gcs_dir: HloDump#dump_jaxpr_gcs_dir.
  MaxTextConfig.dpo: MaxTextConfig#dpo.
  DType: DType#
  DataTypes.weight_dtype: DataTypes#weight_dtype.
  Quantization.kv_quant_axis: Quantization#kv_quant_axis.
  HardwareAndMesh.num_slices: HardwareAndMesh#num_slices.
  HardwareAndMesh.shard_mode: HardwareAndMesh#shard_mode.
  RematAndOffload.remat_policy: RematAndOffload#remat_policy.
  Tokenizer.tokenizer_type: Tokenizer#tokenizer_type.
  Optimizer.opt_type: Optimizer#opt_type.
  Optimizer.lr_schedule_type: Optimizer#lr_schedule_type.
  InferenceGeneral.max_target_length: InferenceGeneral#max_target_length.
  HloDump.dump_hlo_xla_flags: HloDump#dump_hlo_xla_flags.
  DerivedValues.num_target_devices: DerivedValues#num_target_devices.
  RL: RL#
  logger: logger.
  DatasetType: DatasetType#
  DataTypes.dtype: DataTypes#dtype.
  DataTypes.grad_dtype: DataTypes#grad_dtype.
  DataTypes.matmul_precision: DataTypes#matmul_precision.
  ModelArchitecture.base_mlp_dim: ModelArchitecture#base_mlp_dim.
  Attention.attention: Attention#attention.
  Attention.attention_type: Attention#attention_type.
  HardwareAndMesh.hardware: HardwareAndMesh#hardware.
  LayoutAndSharding.context_sharding: LayoutAndSharding#context_sharding.
  DcnParallelism.dcn_diloco_parallelism: DcnParallelism#dcn_diloco_parallelism.
  PipelineParallelism.num_layers_per_pipeline_stage: PipelineParallelism#num_layers_per_pipeline_stage.
  PipelineParallelism.num_pipeline_repeats: PipelineParallelism#num_pipeline_repeats.
  RematAndOffload.decoder_layer_input: RematAndOffload#decoder_layer_input.
  RematAndOffload.context: RematAndOffload#context.
  RematAndOffload.mlpwi: RematAndOffload#mlpwi.
  RematAndOffload.mlpwi_0: RematAndOffload#mlpwi_0.
  RematAndOffload.mlpwi_1: RematAndOffload#mlpwi_1.
  RematAndOffload.mlpwo: RematAndOffload#mlpwo.
  RematAndOffload.moe_mlpwi_0: RematAndOffload#moe_mlpwi_0.
  RematAndOffload.moe_mlpwi_1: RematAndOffload#moe_mlpwi_1.
  RematAndOffload.moe_mlpwo: RematAndOffload#moe_mlpwo.
  RematAndOffload.query_proj: RematAndOffload#query_proj.
  RematAndOffload.key_proj: RematAndOffload#key_proj.
  RematAndOffload.value_proj: RematAndOffload#value_proj.
  RematAndOffload.query_wa_proj: RematAndOffload#query_wa_proj.
  RematAndOffload.kv_wa_proj: RematAndOffload#kv_wa_proj.
  RematAndOffload.qkv_proj: RematAndOffload#qkv_proj.
  RematAndOffload.out_proj: RematAndOffload#out_proj.
  RematAndOffload.mla_q: RematAndOffload#mla_q.
  RematAndOffload.mla_kv: RematAndOffload#mla_kv.
  RematAndOffload.attention_out: RematAndOffload#attention_out.
  RematAndOffload.engram: RematAndOffload#engram.
  DatasetGeneral.packing: DatasetGeneral#packing.
  Optimizer.wsd_decay_style: Optimizer#wsd_decay_style.
  Rope.rope_type: Rope#rope_type.
  Decoding.decode_sampling_strategy: Decoding#decode_sampling_strategy.
  DevelopmentAndDebugging.enable_single_controller: DevelopmentAndDebugging#enable_single_controller.
  Profiling.profiler: Profiling#profiler.
  Profiling.xprof_tpu_power_trace_level: Profiling#xprof_tpu_power_trace_level.
  Checkpointing.load_parameters_path: Checkpointing#load_parameters_path.
  Checkpointing.load_full_state_path: Checkpointing#load_full_state_path.
  EmergencyCheckpointing.local_checkpoint_directory: EmergencyCheckpointing#local_checkpoint_directory.
  IciParallelism: IciParallelism#
  Tokenizer.tokenizer_path: Tokenizer#tokenizer_path.
  GrainDataset.grain_train_files: GrainDataset#grain_train_files.
  GrainDataset.grain_train_mixture_config_path: GrainDataset#grain_train_mixture_config_path.
  MultimodalGeneral: MultimodalGeneral#
  MoEGeneral.base_moe_mlp_dim: MoEGeneral#base_moe_mlp_dim.
  DcnParallelism.dcn_pipeline_parallelism: DcnParallelism#dcn_pipeline_parallelism.
  DatasetGeneral.per_device_batch_size: DatasetGeneral#per_device_batch_size.
  DatasetGeneral.expansion_factor_real_data: DatasetGeneral#expansion_factor_real_data.
  GrainDataset.grain_use_elastic_iterator: GrainDataset#grain_use_elastic_iterator.
  DPO.max_prompt_length: DPO#max_prompt_length.
  FineTuning.use_dpo: FineTuning#use_dpo.
  Distillation.distill_alpha_end: Distillation#distill_alpha_end.
  Distillation.distill_temperature_end: Distillation#distill_temperature_end.
  TrainingLoop.eval_interval: TrainingLoop#eval_interval.
  Optimizer.gradient_accumulation_steps: Optimizer#gradient_accumulation_steps.
  DerivedValues.dcn_parallelism: DerivedValues#dcn_parallelism.
  DerivedValues.micro_batch_size_to_train_on: DerivedValues#micro_batch_size_to_train_on.
  DerivedValues.rampup_end_step: DerivedValues#rampup_end_step.
  DerivedValues.global_batch_size_to_load_increment: DerivedValues#global_batch_size_to_load_increment.
  MaxTextConfig.calculate_global_batch_sizes: MaxTextConfig#calculate_global_batch_sizes().
  RunInfo: RunInfo#
  Checkpointing: Checkpointing#
  OrbaxStorage: OrbaxStorage#
  EmergencyCheckpointing: EmergencyCheckpointing#
  DataTypes: DataTypes#
  Quantization: Quantization#
  ModelArchitecture: ModelArchitecture#
  MTP: MTP#
  Logits: Logits#
  Attention: Attention#
  MoBa: MoBa#
  MlaAttention: MlaAttention#
  CompressedAttention: CompressedAttention#
  AttentionIndexer: AttentionIndexer#
  Llama4Attention: Llama4Attention#
  SplashAttention: SplashAttention#
  MoEGeneral: MoEGeneral#
  MoEKernels: MoEKernels#
  DeepSeekMoE: DeepSeekMoE#
  Qwen3Next: Qwen3Next#
  HardwareAndMesh: HardwareAndMesh#
  LayoutAndSharding: LayoutAndSharding#
  DcnParallelism: DcnParallelism#
  PipelineParallelism: PipelineParallelism#
  RematAndOffload: RematAndOffload#
  Tokenizer: Tokenizer#
  DatasetGeneral: DatasetGeneral#
  TfdsDataset: TfdsDataset#
  HfDataset: HfDataset#
  GrainDataset: GrainDataset#
  GrainDataset.grain_eval_files: GrainDataset#grain_eval_files.
  OlmoGrainDataset: OlmoGrainDataset#
  FineTuning: FineTuning#
  Distillation: Distillation#
  TrainingLoop: TrainingLoop#
  ManifoldConstrainedHyperConnections: ManifoldConstrainedHyperConnections#
  DilocoParams: DilocoParams#
  Optimizer: Optimizer#
  AdamW: AdamW#
  Muon: Muon#
  PositionalEmbedding: PositionalEmbedding#
  Rope: Rope#
  YarnRope: YarnRope#
  InferenceGeneral: InferenceGeneral#
  Decoding: Decoding#
  InferenceLayout: InferenceLayout#
  InferenceServer: InferenceServer#
  InferenceBenchmark: InferenceBenchmark#
  PrefixCaching: PrefixCaching#
  AOT: AOT#
  DevelopmentAndDebugging: DevelopmentAndDebugging#
  Profiling: Profiling#
  HloDump: HloDump#
  HloDump.dump_hlo_local_dir: HloDump#dump_hlo_local_dir.
  Metrics: Metrics#
  ManagedMLDiagnostics: ManagedMLDiagnostics#
  Goodput: Goodput#
  ElasticTraining: ElasticTraining#
  GcpMonitoring: GcpMonitoring#
  Tensorboard: Tensorboard#
  VisionTower: VisionTower#
  VisionProjector: VisionProjector#
  AudioEncoder: AudioEncoder#
  RLHardware: RLHardware#
  VLLM: VLLM#
  RLDataset: RLDataset#
  RLEvaluation: RLEvaluation#
  Reward: Reward#
  SpecialTokens: SpecialTokens#
  Engram: Engram#
  DerivedValues: DerivedValues#
  MaxTextConfig.rl: MaxTextConfig#rl.
  LearningRateScheduleType: LearningRateScheduleType#
  TokenizerType: TokenizerType#
  Quantization.quantization_local_shard_count: Quantization#quantization_local_shard_count.
  Quantization.use_qwix_quantization: Quantization#use_qwix_quantization.
  ModelArchitecture.base_num_decoder_layers: ModelArchitecture#base_num_decoder_layers.
  Attention.share_kv_projections: Attention#share_kv_projections.
  Attention.attn_logits_soft_cap: Attention#attn_logits_soft_cap.
  Attention.use_tokamax_gmm: Attention#use_tokamax_gmm.
  MoEGeneral.use_ring_of_experts: MoEGeneral#use_ring_of_experts.
  MoEGeneral.interleave_moe_layer_step: MoEGeneral#interleave_moe_layer_step.
  Qwen3Next.partial_rotary_factor: Qwen3Next#partial_rotary_factor.
  HardwareAndMesh.scan_layers: HardwareAndMesh#scan_layers.
  LayoutAndSharding.logical_axis_rules: LayoutAndSharding#logical_axis_rules.
  DcnParallelism.dcn_tensor_parallelism: DcnParallelism#dcn_tensor_parallelism.
  IciParallelism.ici_pipeline_parallelism: IciParallelism#ici_pipeline_parallelism.
  PipelineParallelism.pipeline_delay_activation_forwarding: PipelineParallelism#pipeline_delay_activation_forwarding.
  TrainingLoop.steps: TrainingLoop#steps.
  Optimizer.learning_rate_schedule_steps: Optimizer#learning_rate_schedule_steps.
  ElasticTraining.elastic_enabled: ElasticTraining#elastic_enabled.
  DerivedValues.num_decoder_layers: DerivedValues#num_decoder_layers.
  DerivedValues.global_batch_size_to_load_start: DerivedValues#global_batch_size_to_load_start.
  Checkpointing.lora_input_adapters_path: Checkpointing#lora_input_adapters_path.
  Checkpointing.hf_lora_adapter_path: Checkpointing#hf_lora_adapter_path.
  Checkpointing.save_quantized_params_path: Checkpointing#save_quantized_params_path.
  Quantization.quant_cfg_path: Quantization#quant_cfg_path.
  TfdsDataset.dataset_path: TfdsDataset#dataset_path.
  HfDataset.hf_data_dir: HfDataset#hf_data_dir.
  OlmoGrainDataset.olmo_index_path: OlmoGrainDataset#olmo_index_path.
  OlmoGrainDataset.olmo_path_remap_from: OlmoGrainDataset#olmo_path_remap_from.
  OlmoGrainDataset.olmo_path_remap_to: OlmoGrainDataset#olmo_path_remap_to.
  LoRA.lora_restore_path: LoRA#lora_restore_path.
  InferenceGeneral.prefill_cache_dir: InferenceGeneral#prefill_cache_dir.
  InferenceBenchmark.inference_microbenchmark_log_file_path: InferenceBenchmark#inference_microbenchmark_log_file_path.
  InferenceBenchmark.inference_metadata_file: InferenceBenchmark#inference_metadata_file.
  AOT.compiled_trainstep_file: AOT#compiled_trainstep_file.
  DevelopmentAndDebugging.jax_cache_dir: DevelopmentAndDebugging#jax_cache_dir.
  DevelopmentAndDebugging._validate_config: DevelopmentAndDebugging#_validate_config.
  HloDump.dump_jaxpr_local_dir: HloDump#dump_jaxpr_local_dir.
  Metrics.metrics_file: Metrics#metrics_file.
  MultimodalGeneral.image_path: MultimodalGeneral#image_path.
  MultimodalGeneral.video_path: MultimodalGeneral#video_path.
  MultimodalGeneral.audio_path: MultimodalGeneral#audio_path.
  MaxTextConfig.debug: MaxTextConfig#debug.
  MaxTextConfig.lora: MaxTextConfig#lora.
  XProfTPUPowerTraceMode: XProfTPUPowerTraceMode#
  DType.FLOAT32: DType#FLOAT32.
  MatmulPrecision: MatmulPrecision#
  QuantizationType: QuantizationType#
  KvQuantAxis: KvQuantAxis#
  OptimizerType: OptimizerType#
  WsdDecayStyle: WsdDecayStyle#
  RopeType: RopeType#
  DatasetType.TFDS: DatasetType#TFDS.
  SamplingStrategy: SamplingStrategy#
  ProfilerType: ProfilerType#
  EmergencyCheckpointing.local_checkpoint_period: EmergencyCheckpointing#local_checkpoint_period.
  ModelArchitecture.global_parameter_scale: ModelArchitecture#global_parameter_scale.
  ModelArchitecture.attention_output_dim: ModelArchitecture#attention_output_dim.
  ModelArchitecture.fused_qkv: ModelArchitecture#fused_qkv.
  Logits.final_logits_soft_cap: Logits#final_logits_soft_cap.
  Attention.sliding_window_size: Attention#sliding_window_size.
  Attention.chunk_attn_window_size: Attention#chunk_attn_window_size.
  Attention.use_gmm_v2: Attention#use_gmm_v2.
  Attention.use_qk_clip: Attention#use_qk_clip.
  AttentionIndexer.indexer_topk: AttentionIndexer#indexer_topk.
  SplashAttention.local_sa_block_q: SplashAttention#local_sa_block_q.
  SplashAttention.local_sa_block_kv: SplashAttention#local_sa_block_kv.
  SplashAttention.local_sa_block_kv_compute: SplashAttention#local_sa_block_kv_compute.
  SplashAttention.local_sa_block_q_dkv: SplashAttention#local_sa_block_q_dkv.
  SplashAttention.local_sa_block_kv_dkv: SplashAttention#local_sa_block_kv_dkv.
  SplashAttention.local_sa_block_kv_dkv_compute: SplashAttention#local_sa_block_kv_dkv_compute.
  SplashAttention.local_sa_block_q_dq: SplashAttention#local_sa_block_q_dq.
  SplashAttention.local_sa_block_kv_dq: SplashAttention#local_sa_block_kv_dq.
  SplashAttention.local_sa_use_fused_bwd_kernel: SplashAttention#local_sa_use_fused_bwd_kernel.
  SplashAttention.local_sa_q_layout: SplashAttention#local_sa_q_layout.
  SplashAttention.local_sa_k_layout: SplashAttention#local_sa_k_layout.
  SplashAttention.local_sa_v_layout: SplashAttention#local_sa_v_layout.
  SplashAttention.local_use_splash_scheduler: SplashAttention#local_use_splash_scheduler.
  SplashAttention.local_sa_fuse_reciprocal: SplashAttention#local_sa_fuse_reciprocal.
  SplashAttention.local_sa_use_base2_exp: SplashAttention#local_sa_use_base2_exp.
  MoEGeneral.capacity_factor: MoEGeneral#capacity_factor.
  MoEGeneral.moe_expert_input_dim: MoEGeneral#moe_expert_input_dim.
  MoEGeneral.use_ragged_sort: MoEGeneral#use_ragged_sort.
  DeepSeekMoE.first_num_dense_layers: DeepSeekMoE#first_num_dense_layers.
  DeepSeekMoE.use_batch_split_schedule: DeepSeekMoE#use_batch_split_schedule.
  HardwareAndMesh.custom_mesh_and_rule: HardwareAndMesh#custom_mesh_and_rule.
  LayoutAndSharding.sharding_tolerance: LayoutAndSharding#sharding_tolerance.
  LayoutAndSharding.internal_compile_num_devices: LayoutAndSharding#internal_compile_num_devices.
  DcnParallelism.dcn_data_parallelism: DcnParallelism#dcn_data_parallelism.
  DcnParallelism.dcn_fsdp_parallelism: DcnParallelism#dcn_fsdp_parallelism.
  DcnParallelism.dcn_fsdp_transpose_parallelism: DcnParallelism#dcn_fsdp_transpose_parallelism.
  DcnParallelism.dcn_sequence_parallelism: DcnParallelism#dcn_sequence_parallelism.
  DcnParallelism.dcn_context_parallelism: DcnParallelism#dcn_context_parallelism.
  DcnParallelism.dcn_context_autoregressive_parallelism: DcnParallelism#dcn_context_autoregressive_parallelism.
  DcnParallelism.dcn_tensor_sequence_parallelism: DcnParallelism#dcn_tensor_sequence_parallelism.
  DcnParallelism.dcn_expert_parallelism: DcnParallelism#dcn_expert_parallelism.
  DcnParallelism.dcn_autoregressive_parallelism: DcnParallelism#dcn_autoregressive_parallelism.
  IciParallelism.ici_diloco_parallelism: IciParallelism#ici_diloco_parallelism.
  IciParallelism.ici_tensor_parallelism: IciParallelism#ici_tensor_parallelism.
  DatasetGeneral.eval_per_device_batch_size: DatasetGeneral#eval_per_device_batch_size.
  DatasetGeneral.colocated_python_data_input: DatasetGeneral#colocated_python_data_input.
  HfDataset.hf_eval_split: HfDataset#hf_eval_split.
  GrainDataset.grain_file_type: GrainDataset#grain_file_type.
  DPO: DPO#
  FineTuning.use_sft: FineTuning#use_sft.
  FineTuning.use_grpo: FineTuning#use_grpo.
  LoRA: LoRA#
  Distillation.distill_beta: Distillation#distill_beta.
  Optimizer.wsd_decay_steps_fraction: Optimizer#wsd_decay_steps_fraction.
  Optimizer.warmup_steps_fraction: Optimizer#warmup_steps_fraction.
  AdamW.mu_dtype: AdamW#mu_dtype.
  AOT.compile_topology: AOT#compile_topology.
  AOT.compile_topology_num_slices: AOT#compile_topology_num_slices.
  DevelopmentAndDebugging.constant_bound_config: DevelopmentAndDebugging#constant_bound_config.
  DevelopmentAndDebugging.subslice_shape: DevelopmentAndDebugging#subslice_shape.
  HloDump.dump_hlo_local_module_name: HloDump#dump_hlo_local_module_name.
  Debug: Debug#
  Engram.engram_vocab_bases: Engram#engram_vocab_bases.
  Engram.engram_max_ngram_size: Engram#engram_max_ngram_size.
  DerivedValues.mlp_dim: DerivedValues#mlp_dim.
  DerivedValues.moe_mlp_dim: DerivedValues#moe_mlp_dim.
  DerivedValues.using_pipeline_parallelism: DerivedValues#using_pipeline_parallelism.
  DerivedValues.global_batch_size_to_load: DerivedValues#global_batch_size_to_load.
  DerivedValues.checkpoint_dir: DerivedValues#checkpoint_dir.
  DerivedValues.metrics_dir: DerivedValues#metrics_dir.
  DerivedValues.tensorboard_dir: DerivedValues#tensorboard_dir.
  get_individual_scales: get_individual_scales().
  XProfTPUPowerTraceMode.POWER_TRACE_NONE: XProfTPUPowerTraceMode#POWER_TRACE_NONE.
  DType.BFLOAT16: DType#BFLOAT16.
  MatmulPrecision.DEFAULT: MatmulPrecision#DEFAULT.
  QuantizationType.NONE: QuantizationType#NONE.
  KvQuantAxis.HEADS_AND_DKV: KvQuantAxis#HEADS_AND_DKV.
  RematPolicy: RematPolicy#
  RematPolicy.FULL: RematPolicy#FULL.
  RematLocation.DEVICE: RematLocation#DEVICE.
  OptimizerType.ADAMW: OptimizerType#ADAMW.
  LearningRateScheduleType.COSINE: LearningRateScheduleType#COSINE.
  LearningRateScheduleType.WSD: LearningRateScheduleType#WSD.
  WsdDecayStyle.LINEAR: WsdDecayStyle#LINEAR.
  RopeType.DEFAULT: RopeType#DEFAULT.
  TokenizerType.SENTENCEPIECE: TokenizerType#SENTENCEPIECE.
  TokenizerType.TIKTOKEN: TokenizerType#TIKTOKEN.
  DatasetType.HF: DatasetType#HF.
  DatasetType.GRAIN: DatasetType#GRAIN.
  SamplingStrategy.GREEDY: SamplingStrategy#GREEDY.
  ProfilerType.NONE: ProfilerType#NONE.
  ModelName: ModelName.
  Checkpointing.enable_checkpointing: Checkpointing#enable_checkpointing.
  Checkpointing.source_checkpoint_layout: Checkpointing#source_checkpoint_layout.
  Checkpointing.colocated_python_checkpointing: Checkpointing#colocated_python_checkpointing.
  EmergencyCheckpointing.enable_multi_tier_checkpointing: EmergencyCheckpointing#enable_multi_tier_checkpointing.
  EmergencyCheckpointing.multi_tier_checkpointing_backup_interval_minutes: EmergencyCheckpointing#multi_tier_checkpointing_backup_interval_minutes.
  EmergencyCheckpointing.enable_emergency_checkpoint: EmergencyCheckpointing#enable_emergency_checkpoint.
  Quantization.quantize_kvcache: Quantization#quantize_kvcache.
  ModelArchitecture.base_emb_dim: ModelArchitecture#base_emb_dim.
  ModelArchitecture.base_num_query_heads: ModelArchitecture#base_num_query_heads.
  ModelArchitecture.base_num_kv_heads: ModelArchitecture#base_num_kv_heads.
  ModelArchitecture.head_dim: ModelArchitecture#head_dim.
  Attention.num_kv_shared_layers: Attention#num_kv_shared_layers.
  Attention.use_tokamax_splash: Attention#use_tokamax_splash.
  Attention.use_jax_splash: Attention#use_jax_splash.
  Attention.force_q_layout: Attention#force_q_layout.
  MoBa.moba: MoBa#moba.
  MlaAttention.q_lora_rank: MlaAttention#q_lora_rank.
  CompressedAttention.compress_ratios: CompressedAttention#compress_ratios.
  AttentionIndexer.use_indexer: AttentionIndexer#use_indexer.
  AttentionIndexer.indexer_loss_scaling_factor: AttentionIndexer#indexer_loss_scaling_factor.
  SplashAttention.sa_block_q: SplashAttention#sa_block_q.
  SplashAttention.sa_block_kv: SplashAttention#sa_block_kv.
  SplashAttention.sa_block_kv_compute: SplashAttention#sa_block_kv_compute.
  SplashAttention.sa_block_q_dkv: SplashAttention#sa_block_q_dkv.
  SplashAttention.sa_block_kv_dkv: SplashAttention#sa_block_kv_dkv.
  SplashAttention.sa_block_kv_dkv_compute: SplashAttention#sa_block_kv_dkv_compute.
  SplashAttention.sa_block_q_dq: SplashAttention#sa_block_q_dq.
  SplashAttention.sa_block_kv_dq: SplashAttention#sa_block_kv_dq.
  SplashAttention.sa_use_fused_bwd_kernel: SplashAttention#sa_use_fused_bwd_kernel.
  SplashAttention.sa_q_layout: SplashAttention#sa_q_layout.
  SplashAttention.sa_k_layout: SplashAttention#sa_k_layout.
  SplashAttention.sa_v_layout: SplashAttention#sa_v_layout.
  SplashAttention.use_splash_scheduler: SplashAttention#use_splash_scheduler.
  SplashAttention.sa_fuse_reciprocal: SplashAttention#sa_fuse_reciprocal.
  SplashAttention.sa_use_base2_exp: SplashAttention#sa_use_base2_exp.
  MoEGeneral.num_experts: MoEGeneral#num_experts.
  MoEGeneral.num_experts_per_tok: MoEGeneral#num_experts_per_tok.
  MoEGeneral.ragged_buffer_factor: MoEGeneral#ragged_buffer_factor.
  MoEKernels.sparse_matmul: MoEKernels#sparse_matmul.
  DeepSeekMoE.routed_bias: DeepSeekMoE#routed_bias.
  DeepSeekMoE.routed_bias_update_rate: DeepSeekMoE#routed_bias_update_rate.
  DeepSeekMoE.first_num_hash_layers: DeepSeekMoE#first_num_hash_layers.
  Qwen3Next.gdn_num_key_heads: Qwen3Next#gdn_num_key_heads.
  Qwen3Next.gdn_num_value_heads: Qwen3Next#gdn_num_value_heads.
  HardwareAndMesh.inhomogeneous_layer_cycle_interval: HardwareAndMesh#inhomogeneous_layer_cycle_interval.
  HardwareAndMesh.context_parallel_load_balance: HardwareAndMesh#context_parallel_load_balance.
  HardwareAndMesh.context_parallel_strategy: HardwareAndMesh#context_parallel_strategy.
  HardwareAndMesh.context_parallel_reorder_strategy: HardwareAndMesh#context_parallel_reorder_strategy.
  HardwareAndMesh.enable_nnx: HardwareAndMesh#enable_nnx.
  HardwareAndMesh.pure_nnx_decoder: HardwareAndMesh#pure_nnx_decoder.
  HardwareAndMesh.pure_nnx: HardwareAndMesh#pure_nnx.
  LayoutAndSharding.check_vma: LayoutAndSharding#check_vma.
  LayoutAndSharding.internal_compile: LayoutAndSharding#internal_compile.
  DcnParallelism.dcn_tensor_transpose_parallelism: DcnParallelism#dcn_tensor_transpose_parallelism.
  IciParallelism.ici_data_parallelism: IciParallelism#ici_data_parallelism.
  IciParallelism.ici_fsdp_parallelism: IciParallelism#ici_fsdp_parallelism.
  IciParallelism.ici_fsdp_transpose_parallelism: IciParallelism#ici_fsdp_transpose_parallelism.
  IciParallelism.ici_sequence_parallelism: IciParallelism#ici_sequence_parallelism.
  IciParallelism.ici_context_parallelism: IciParallelism#ici_context_parallelism.
  IciParallelism.ici_context_autoregressive_parallelism: IciParallelism#ici_context_autoregressive_parallelism.
  IciParallelism.ici_tensor_transpose_parallelism: IciParallelism#ici_tensor_transpose_parallelism.
  IciParallelism.ici_tensor_sequence_parallelism: IciParallelism#ici_tensor_sequence_parallelism.
  IciParallelism.ici_autoregressive_parallelism: IciParallelism#ici_autoregressive_parallelism.
  IciParallelism.ici_expert_parallelism: IciParallelism#ici_expert_parallelism.
  PipelineParallelism.pipeline_fsdp_ag_per_repeat: PipelineParallelism#pipeline_fsdp_ag_per_repeat.
  PipelineParallelism.scan_layers_per_stage: PipelineParallelism#scan_layers_per_stage.
  Tokenizer.use_truncation: Tokenizer#use_truncation.
  Tokenizer.num_vocab_tiling: Tokenizer#num_vocab_tiling.
  DatasetGeneral.max_segments_per_seq: DatasetGeneral#max_segments_per_seq.
  DatasetGeneral.generate_padding_batch_eval: DatasetGeneral#generate_padding_batch_eval.
  DatasetGeneral.enable_rampup_batch_size: DatasetGeneral#enable_rampup_batch_size.
  DatasetGeneral.per_device_batch_size_start: DatasetGeneral#per_device_batch_size_start.
  DatasetGeneral.per_device_batch_size_increment: DatasetGeneral#per_device_batch_size_increment.
  DatasetGeneral.global_rampup_samples: DatasetGeneral#global_rampup_samples.
  TfdsDataset.dataset_name: TfdsDataset#dataset_name.
  TfdsDataset.eval_split: TfdsDataset#eval_split.
  HfDataset.hf_path: HfDataset#hf_path.
  HfDataset.hf_eval_files: HfDataset#hf_eval_files.
  HfDataset.hf_access_token: HfDataset#hf_access_token.
  FineTuning.sft_train_on_completion_only: FineTuning#sft_train_on_completion_only.
  Distillation.distill_alpha_schedule: Distillation#distill_alpha_schedule.
  Distillation.distill_temperature_schedule: Distillation#distill_temperature_schedule.
  Distillation.distill_beta_schedule: Distillation#distill_beta_schedule.
  TrainingLoop.eval_steps: TrainingLoop#eval_steps.
  DilocoParams.enable_diloco: DilocoParams#enable_diloco.
  DevelopmentAndDebugging._clean_empty_string_for_list: DevelopmentAndDebugging#_clean_empty_string_for_list().
  MultimodalGeneral.use_multimodal: MultimodalGeneral#use_multimodal.
  MultimodalGeneral.video_max_grid_t: MultimodalGeneral#video_max_grid_t.
  MultimodalGeneral.video_max_grid_h: MultimodalGeneral#video_max_grid_h.
  MultimodalGeneral.video_max_grid_w: MultimodalGeneral#video_max_grid_w.
  VisionTower.deepstack_visual_indexes_for_vit: VisionTower#deepstack_visual_indexes_for_vit.
  RL.loss_algo: RL#loss_algo.
  Engram.engram_layers: Engram#engram_layers.
  DerivedValues.emb_dim: DerivedValues#emb_dim.
  DerivedValues.num_kv_heads: DerivedValues#num_kv_heads.
  DerivedValues.num_query_heads: DerivedValues#num_query_heads.
  DerivedValues.num_diloco_replicas: DerivedValues#num_diloco_replicas.
  DerivedValues.ici_parallelism: DerivedValues#ici_parallelism.
  DerivedValues.global_batch_size_to_train_on: DerivedValues#global_batch_size_to_train_on.
  DerivedValues.global_batch_size_to_eval_on: DerivedValues#global_batch_size_to_eval_on.
  DerivedValues.global_batch_size_to_load_eval: DerivedValues#global_batch_size_to_load_eval.
  DerivedValues.micro_batch_size_to_eval_on: DerivedValues#micro_batch_size_to_eval_on.
  DerivedValues.managed_mldiagnostics_dir: DerivedValues#managed_mldiagnostics_dir.
  DerivedValues.tensors_on_device: DerivedValues#tensors_on_device.
  DerivedValues.tensors_to_offload: DerivedValues#tensors_to_offload.
  DerivedValues.rampup_samples_per_increment_to_load: DerivedValues#rampup_samples_per_increment_to_load.
  XProfTPUPowerTraceMode.POWER_TRACE_NORMAL: XProfTPUPowerTraceMode#POWER_TRACE_NORMAL.
  XProfTPUPowerTraceMode.POWER_TRACE_SPI: XProfTPUPowerTraceMode#POWER_TRACE_SPI.
  AxisNames: AxisNames.
  DType.FLOAT16: DType#FLOAT16.
  MatmulPrecision.HIGH: MatmulPrecision#HIGH.
  MatmulPrecision.HIGHEST: MatmulPrecision#HIGHEST.
  MatmulPrecision.BFLOAT16: MatmulPrecision#BFLOAT16.
  MatmulPrecision.FLOAT32: MatmulPrecision#FLOAT32.
  QuantizationType.INT4: QuantizationType#INT4.
  QuantizationType.INT8: QuantizationType#INT8.
  QuantizationType.INTMP: QuantizationType#INTMP.
  QuantizationType.FP8_E5M2: QuantizationType#FP8_E5M2.
  QuantizationType.FP8_E4M3: QuantizationType#FP8_E4M3.
  QuantizationType.FP8: QuantizationType#FP8.
  QuantizationType.NANOO_FP8: QuantizationType#NANOO_FP8.
  QuantizationType.FP8_NANO_V2: QuantizationType#FP8_NANO_V2.
  QuantizationType.FP8_GPU: QuantizationType#FP8_GPU.
  QuantizationType.FP8_FULL: QuantizationType#FP8_FULL.
  QuantizationType.TE_FP8_DS: QuantizationType#TE_FP8_DS.
  QuantizationType.TE_FP8_CS: QuantizationType#TE_FP8_CS.
  QuantizationType.TE_MXFP8: QuantizationType#TE_MXFP8.
  QuantizationType.TE_NVFP4: QuantizationType#TE_NVFP4.
  QuantizationType.TE_NVFP4_NO_RHT: QuantizationType#TE_NVFP4_NO_RHT.
  KvQuantAxis.NONE: KvQuantAxis#NONE.
  KvQuantAxis.DKV: KvQuantAxis#DKV.
  RematPolicy.MINIMAL: RematPolicy#MINIMAL.
  RematPolicy.SAVE_DOT_WITH_CONTEXT_EXCEPT_MLP: RematPolicy#SAVE_DOT_WITH_CONTEXT_EXCEPT_MLP.
  RematPolicy.SAVE_DOT_EXCEPT_MLPWI: RematPolicy#SAVE_DOT_EXCEPT_MLPWI.
  RematPolicy.SAVE_DOT_EXCEPT_MLP: RematPolicy#SAVE_DOT_EXCEPT_MLP.
  RematPolicy.SAVE_QKV_PROJ: RematPolicy#SAVE_QKV_PROJ.
  RematPolicy.QKV_PROJ_OFFLOADED: RematPolicy#QKV_PROJ_OFFLOADED.
  RematPolicy.CUSTOM: RematPolicy#CUSTOM.
  RematPolicy.MINIMAL_OFFLOADED: RematPolicy#MINIMAL_OFFLOADED.
  RematPolicy.SAVE_OUT_PROJ: RematPolicy#SAVE_OUT_PROJ.
  RematLocation.OFFLOAD: RematLocation#OFFLOAD.
  OptimizerType.ADAM_PAX: OptimizerType#ADAM_PAX.
  OptimizerType.SGD: OptimizerType#SGD.
  OptimizerType.MUON: OptimizerType#MUON.
  WsdDecayStyle.COSINE: WsdDecayStyle#COSINE.
  RopeType.LLAMA3_1: RopeType#LLAMA3_1.
  RopeType.YARN: RopeType#YARN.
  TokenizerType.HUGGINGFACE: TokenizerType#HUGGINGFACE.
  DatasetType.SYNTHETIC: DatasetType#SYNTHETIC.
  DatasetType.C4MLPERF: DatasetType#C4MLPERF.
  DatasetType.OLMO_GRAIN: DatasetType#OLMO_GRAIN.
  SamplingStrategy.WEIGHTED: SamplingStrategy#WEIGHTED.
  SamplingStrategy.NUCLEUS: SamplingStrategy#NUCLEUS.
  SamplingStrategy.TOPK: SamplingStrategy#TOPK.
  SamplingStrategy.COMPOSITE: SamplingStrategy#COMPOSITE.
  ProfilerType.XPLANE: ProfilerType#XPLANE.
  ProfilerType.NSYS: ProfilerType#NSYS.
  RunInfo.base_config: RunInfo#base_config.
  RunInfo.override_model_config: RunInfo#override_model_config.
  RunInfo.override_logical_axis_rules: RunInfo#override_logical_axis_rules.
  RunInfo.log_config: RunInfo#log_config.
  RunInfo.debug_sharding: RunInfo#debug_sharding.
  RunInfo.sharding_strategy: RunInfo#sharding_strategy.
  Checkpointing.load_checkpoint_only_once: Checkpointing#load_checkpoint_only_once.
  Checkpointing.async_checkpointing: Checkpointing#async_checkpointing.
  Checkpointing.checkpoint_period: Checkpointing#checkpoint_period.
  Checkpointing.max_num_checkpoints_to_keep: Checkpointing#max_num_checkpoints_to_keep.
  Checkpointing.enable_single_replica_ckpt_restoring: Checkpointing#enable_single_replica_ckpt_restoring.
  Checkpointing.checkpoint_todelete_subdir: Checkpointing#checkpoint_todelete_subdir.
  Checkpointing.checkpoint_todelete_full_path: Checkpointing#checkpoint_todelete_full_path.
  Checkpointing.force_unroll: Checkpointing#force_unroll.
  Checkpointing.checkpoint_is_quantized: Checkpointing#checkpoint_is_quantized.
  Checkpointing.enable_orbax_v1: Checkpointing#enable_orbax_v1.
  Checkpointing.checkpoint_conversion_fn: Checkpointing#checkpoint_conversion_fn.
  Checkpointing.save_checkpoint_on_completion: Checkpointing#save_checkpoint_on_completion.
  Checkpointing.enable_continuous_checkpointing: Checkpointing#enable_continuous_checkpointing.
  Checkpointing.enable_autocheckpoint: Checkpointing#enable_autocheckpoint.
  OrbaxStorage.checkpoint_storage_target_data_file_size_bytes: OrbaxStorage#checkpoint_storage_target_data_file_size_bytes.
  OrbaxStorage.checkpoint_storage_use_ocdbt: OrbaxStorage#checkpoint_storage_use_ocdbt.
  OrbaxStorage.checkpoint_storage_use_zarr3: OrbaxStorage#checkpoint_storage_use_zarr3.
  OrbaxStorage.checkpoint_storage_concurrent_gb: OrbaxStorage#checkpoint_storage_concurrent_gb.
  EmergencyCheckpointing.mtc_data_parallelism: EmergencyCheckpointing#mtc_data_parallelism.
  EmergencyCheckpointing.use_replicator_service: EmergencyCheckpointing#use_replicator_service.
  EmergencyCheckpointing.replicator_backup_interval_minutes: EmergencyCheckpointing#replicator_backup_interval_minutes.
  DataTypes.activations_in_float32: DataTypes#activations_in_float32.
  DataTypes.dtype_mm: DataTypes#dtype_mm.
  Quantization.replicate_quant_scale: Quantization#replicate_quant_scale.
  Quantization.kv_quant_dtype: Quantization#kv_quant_dtype.
  Quantization.use_manual_quantization: Quantization#use_manual_quantization.
  Quantization.weight_quantization_calibration_method: Quantization#weight_quantization_calibration_method.
  Quantization.act_quantization_calibration_method: Quantization#act_quantization_calibration_method.
  Quantization.bwd_quantization_calibration_method: Quantization#bwd_quantization_calibration_method.
  Quantization.weight_sparsity_n: Quantization#weight_sparsity_n.
  Quantization.weight_sparsity_m: Quantization#weight_sparsity_m.
  Quantization.weight_sparsity_update_step: Quantization#weight_sparsity_update_step.
  Quantization.weight_sparsity_start_step: Quantization#weight_sparsity_start_step.
  ModelArchitecture.dense_init_scale: ModelArchitecture#dense_init_scale.
  ModelArchitecture.global_head_dim: ModelArchitecture#global_head_dim.
  ModelArchitecture.mlp_activations: ModelArchitecture#mlp_activations.
  ModelArchitecture.mlp_activations_limit: ModelArchitecture#mlp_activations_limit.
  ModelArchitecture.normalization_layer_epsilon: ModelArchitecture#normalization_layer_epsilon.
  ModelArchitecture.attention_bias: ModelArchitecture#attention_bias.
  ModelArchitecture.fused_mlp: ModelArchitecture#fused_mlp.
  ModelArchitecture.qk_norm_with_scale: ModelArchitecture#qk_norm_with_scale.
  ModelArchitecture.v_norm_with_scale: ModelArchitecture#v_norm_with_scale.
  MTP.mtp_num_layers: MTP#mtp_num_layers.
  MTP.mtp_loss_scaling_factor: MTP#mtp_loss_scaling_factor.
  MTP.mtp_eval_target_module: MTP#mtp_eval_target_module.
  Logits.logits_via_embedding: Logits#logits_via_embedding.
  Logits.normalize_embedding_logits: Logits#normalize_embedding_logits.
  Logits.logits_dot_in_fp32: Logits#logits_dot_in_fp32.
  Logits.cast_logits_to_fp32: Logits#cast_logits_to_fp32.
  Logits.z_loss_multiplier: Logits#z_loss_multiplier.
  Attention.global_num_kv_heads: Attention#global_num_kv_heads.
  Attention.hidden_size_per_layer_input: Attention#hidden_size_per_layer_input.
  Attention.vocab_size_per_layer_input: Attention#vocab_size_per_layer_input.
  Attention.use_double_wide_mlp: Attention#use_double_wide_mlp.
  Attention.attention_sink: Attention#attention_sink.
  Attention.float32_qk_product: Attention#float32_qk_product.
  Attention.float32_logits: Attention#float32_logits.
  Attention.use_post_attn_norm: Attention#use_post_attn_norm.
  Attention.use_post_ffw_norm: Attention#use_post_ffw_norm.
  Attention.use_ragged_attention: Attention#use_ragged_attention.
  Attention.ragged_block_size: Attention#ragged_block_size.
  Attention.enable_padding_causal_mask: Attention#enable_padding_causal_mask.
  Attention.qk_clip_threshold: Attention#qk_clip_threshold.
  MoBa.moba_chunk_size: MoBa#moba_chunk_size.
  MoBa.moba_topk: MoBa#moba_topk.
  MlaAttention.mla_naive_kvcache: MlaAttention#mla_naive_kvcache.
  MlaAttention.kv_lora_rank: MlaAttention#kv_lora_rank.
  MlaAttention.qk_nope_head_dim: MlaAttention#qk_nope_head_dim.
  MlaAttention.qk_rope_head_dim: MlaAttention#qk_rope_head_dim.
  MlaAttention.v_head_dim: MlaAttention#v_head_dim.
  CompressedAttention.o_lora_rank: CompressedAttention#o_lora_rank.
  CompressedAttention.o_groups: CompressedAttention#o_groups.
  CompressedAttention.compressed_rope_max_timescale: CompressedAttention#compressed_rope_max_timescale.
  AttentionIndexer.indexer_head_dim: AttentionIndexer#indexer_head_dim.
  AttentionIndexer.indexer_n_heads: AttentionIndexer#indexer_n_heads.
  AttentionIndexer.indexer_sparse_training: AttentionIndexer#indexer_sparse_training.
  AttentionIndexer.indexer_use_approx_top_k: AttentionIndexer#indexer_use_approx_top_k.
  AttentionIndexer.indexer_approx_top_k_recall: AttentionIndexer#indexer_approx_top_k_recall.
  Llama4Attention.use_qk_norm: Llama4Attention#use_qk_norm.
  Llama4Attention.temperature_tuning: Llama4Attention#temperature_tuning.
  SplashAttention.use_max_logit_estimate: SplashAttention#use_max_logit_estimate.
  SplashAttention.cost_estimate_flops_fwd: SplashAttention#cost_estimate_flops_fwd.
  SplashAttention.cost_estimate_flops_bwd: SplashAttention#cost_estimate_flops_bwd.
  SplashAttention.dq_reduction_steps: SplashAttention#dq_reduction_steps.
  MoEGeneral.padded_base_moe_mlp_dim: MoEGeneral#padded_base_moe_mlp_dim.
  MoEGeneral.load_balance_loss_weight: MoEGeneral#load_balance_loss_weight.
  MoEGeneral.use_custom_sort_vjp: MoEGeneral#use_custom_sort_vjp.
  MoEGeneral.moe_dispatch_no_expert_sharding: MoEGeneral#moe_dispatch_no_expert_sharding.
  MoEGeneral.use_gather_mosaic_kernel: MoEGeneral#use_gather_mosaic_kernel.
  MoEGeneral.ragged_gather_fallback: MoEGeneral#ragged_gather_fallback.
  MoEGeneral.ragged_gather_reduce_fallback: MoEGeneral#ragged_gather_reduce_fallback.
  MoEGeneral.ragged_gather_cost_estimate_flops: MoEGeneral#ragged_gather_cost_estimate_flops.
  MoEGeneral.ragged_gather_reduce_cost_estimate_flops: MoEGeneral#ragged_gather_reduce_cost_estimate_flops.
  MoEGeneral.ragged_gather_cost_estimate_bytes_accessed: MoEGeneral#ragged_gather_cost_estimate_bytes_accessed.
  MoEGeneral.ragged_gather_reduce_cost_estimate_bytes_accessed: MoEGeneral#ragged_gather_reduce_cost_estimate_bytes_accessed.
  MoEGeneral.use_random_routing: MoEGeneral#use_random_routing.
  MoEGeneral.moe_fsdp_use_two_stage_all_gather: MoEGeneral#moe_fsdp_use_two_stage_all_gather.
  MoEGeneral.shard_exp_on_fsdp: MoEGeneral#shard_exp_on_fsdp.
  MoEGeneral.use_2d_fsdp_sharding: MoEGeneral#use_2d_fsdp_sharding.
  MoEGeneral.norm_topk_prob: MoEGeneral#norm_topk_prob.
  MoEGeneral.float32_weight_sum: MoEGeneral#float32_weight_sum.
  MoEGeneral.float32_gate_logits: MoEGeneral#float32_gate_logits.
  MoEGeneral.prefuse_moe_weights: MoEGeneral#prefuse_moe_weights.
  MoEGeneral.fuse_expert_scales: MoEGeneral#fuse_expert_scales.
  MoEKernels.megablox: MoEKernels#megablox.
  MoEKernels.wi_tile_fwd_batch_seq: MoEKernels#wi_tile_fwd_batch_seq.
  MoEKernels.wi_tile_fwd_embed_dim: MoEKernels#wi_tile_fwd_embed_dim.
  MoEKernels.wi_tile_fwd_mlp_dim: MoEKernels#wi_tile_fwd_mlp_dim.
  MoEKernels.wi_tile_dlhs_batch_seq: MoEKernels#wi_tile_dlhs_batch_seq.
  MoEKernels.wi_tile_dlhs_embed_dim: MoEKernels#wi_tile_dlhs_embed_dim.
  MoEKernels.wi_tile_dlhs_mlp_dim: MoEKernels#wi_tile_dlhs_mlp_dim.
  MoEKernels.wi_tile_drhs_batch_seq: MoEKernels#wi_tile_drhs_batch_seq.
  MoEKernels.wi_tile_drhs_embed_dim: MoEKernels#wi_tile_drhs_embed_dim.
  MoEKernels.wi_tile_drhs_mlp_dim: MoEKernels#wi_tile_drhs_mlp_dim.
  MoEKernels.wo_tile_fwd_batch_seq: MoEKernels#wo_tile_fwd_batch_seq.
  MoEKernels.wo_tile_fwd_embed_dim: MoEKernels#wo_tile_fwd_embed_dim.
  MoEKernels.wo_tile_fwd_mlp_dim: MoEKernels#wo_tile_fwd_mlp_dim.
  MoEKernels.wo_tile_dlhs_batch_seq: MoEKernels#wo_tile_dlhs_batch_seq.
  MoEKernels.wo_tile_dlhs_embed_dim: MoEKernels#wo_tile_dlhs_embed_dim.
  MoEKernels.wo_tile_dlhs_mlp_dim: MoEKernels#wo_tile_dlhs_mlp_dim.
  MoEKernels.wo_tile_drhs_batch_seq: MoEKernels#wo_tile_drhs_batch_seq.
  MoEKernels.wo_tile_drhs_embed_dim: MoEKernels#wo_tile_drhs_embed_dim.
  MoEKernels.wo_tile_drhs_mlp_dim: MoEKernels#wo_tile_drhs_mlp_dim.
  MoEKernels.merge_gating_gmm: MoEKernels#merge_gating_gmm.
  DeepSeekMoE.shared_experts: DeepSeekMoE#shared_experts.
  DeepSeekMoE.routed_scaling_factor: DeepSeekMoE#routed_scaling_factor.
  DeepSeekMoE.routed_score_func: DeepSeekMoE#routed_score_func.
  DeepSeekMoE.mlp_bias: DeepSeekMoE#mlp_bias.
  DeepSeekMoE.n_routing_groups: DeepSeekMoE#n_routing_groups.
  DeepSeekMoE.topk_routing_group: DeepSeekMoE#topk_routing_group.
  DeepSeekMoE.batch_split_factor: DeepSeekMoE#batch_split_factor.
  Qwen3Next.gdn_conv_kernel_dim: Qwen3Next#gdn_conv_kernel_dim.
  Qwen3Next.gdn_key_head_dim: Qwen3Next#gdn_key_head_dim.
  Qwen3Next.gdn_value_head_dim: Qwen3Next#gdn_value_head_dim.
  Qwen3Next.gdn_chunk_size: Qwen3Next#gdn_chunk_size.
  Qwen3Next.use_qk_norm_in_gdn: Qwen3Next#use_qk_norm_in_gdn.
  HardwareAndMesh.param_scan_axis: HardwareAndMesh#param_scan_axis.
  HardwareAndMesh.custom_mesh: HardwareAndMesh#custom_mesh.
  HardwareAndMesh.allow_split_physical_axes: HardwareAndMesh#allow_split_physical_axes.
  HardwareAndMesh.optimize_mesh_for_tpu_v6e: HardwareAndMesh#optimize_mesh_for_tpu_v6e.
  HardwareAndMesh.shardy: HardwareAndMesh#shardy.
  HardwareAndMesh.remove_size_one_mesh_axis_from_type: HardwareAndMesh#remove_size_one_mesh_axis_from_type.
  LayoutAndSharding.input_data_sharding_logical_axes: LayoutAndSharding#input_data_sharding_logical_axes.
  LayoutAndSharding.shard_optimizer_over_data: LayoutAndSharding#shard_optimizer_over_data.
  LayoutAndSharding.compile_xla_flags: LayoutAndSharding#compile_xla_flags.
  PipelineParallelism.pipeline_fsdp_ag_once: PipelineParallelism#pipeline_fsdp_ag_once.
  PipelineParallelism.scan_pipeline_iterations: PipelineParallelism#scan_pipeline_iterations.
  PipelineParallelism.scan_pipeline_repeats: PipelineParallelism#scan_pipeline_repeats.
  PipelineParallelism.set_remat_policy_on_pipeline_iterations: PipelineParallelism#set_remat_policy_on_pipeline_iterations.
  PipelineParallelism.set_remat_policy_on_layers_per_stage: PipelineParallelism#set_remat_policy_on_layers_per_stage.
  RematAndOffload.remat_policy_for_vit: RematAndOffload#remat_policy_for_vit.
  RematAndOffload.optimizer_memory_host_offload: RematAndOffload#optimizer_memory_host_offload.
  RematAndOffload.parameter_memory_host_offload: RematAndOffload#parameter_memory_host_offload.
  Tokenizer.vocab_size: Tokenizer#vocab_size.
  Tokenizer.use_chat_template: Tokenizer#use_chat_template.
  Tokenizer.chat_template_path: Tokenizer#chat_template_path.
  Tokenizer.chat_template: Tokenizer#chat_template.
  Tokenizer.data_template_path: Tokenizer#data_template_path.
  Tokenizer.tokenize_train_data: Tokenizer#tokenize_train_data.
  Tokenizer.tokenize_eval_data: Tokenizer#tokenize_eval_data.
  Tokenizer.add_bos: Tokenizer#add_bos.
  Tokenizer.add_eos: Tokenizer#add_eos.
  DatasetGeneral.max_corpus_chars: DatasetGeneral#max_corpus_chars.
  DatasetGeneral.train_data_columns: DatasetGeneral#train_data_columns.
  DatasetGeneral.train_image_column: DatasetGeneral#train_image_column.
  DatasetGeneral.eval_data_columns: DatasetGeneral#eval_data_columns.
  DatasetGeneral.eval_image_column: DatasetGeneral#eval_image_column.
  DatasetGeneral.grain_packing_type: DatasetGeneral#grain_packing_type.
  DatasetGeneral.num_epoch: DatasetGeneral#num_epoch.
  DatasetGeneral.reuse_example_batch: DatasetGeneral#reuse_example_batch.
  DatasetGeneral.generate_padding_batch_train: DatasetGeneral#generate_padding_batch_train.
  TfdsDataset.eval_dataset_name: TfdsDataset#eval_dataset_name.
  TfdsDataset.train_split: TfdsDataset#train_split.
  HfDataset.hf_name: HfDataset#hf_name.
  HfDataset.hf_train_files: HfDataset#hf_train_files.
  GrainDataset.grain_worker_count: GrainDataset#grain_worker_count.
  GrainDataset.grain_per_worker_buffer_size: GrainDataset#grain_per_worker_buffer_size.
  GrainDataset.grain_worker_count_eval: GrainDataset#grain_worker_count_eval.
  GrainDataset.grain_per_worker_buffer_size_eval: GrainDataset#grain_per_worker_buffer_size_eval.
  GrainDataset.grain_ram_budget_mb: GrainDataset#grain_ram_budget_mb.
  GrainDataset.grain_num_threads: GrainDataset#grain_num_threads.
  GrainDataset.grain_prefetch_buffer_size: GrainDataset#grain_prefetch_buffer_size.
  GrainDataset.grain_num_threads_eval: GrainDataset#grain_num_threads_eval.
  GrainDataset.grain_prefetch_buffer_size_eval: GrainDataset#grain_prefetch_buffer_size_eval.
  GrainDataset.grain_data_source_max_workers: GrainDataset#grain_data_source_max_workers.
  GrainDataset.grain_shuffle_buffer_size: GrainDataset#grain_shuffle_buffer_size.
  OlmoGrainDataset.olmo_apply_ngram_filter: OlmoGrainDataset#olmo_apply_ngram_filter.
  DPO.algo: DPO#algo.
  DPO.dpo_beta: DPO#dpo_beta.
  DPO.orpo_lambda: DPO#orpo_lambda.
  DPO.dpo_label_smoothing: DPO#dpo_label_smoothing.
  FineTuning.formatting_func_path: FineTuning#formatting_func_path.
  FineTuning.formatting_func_kwargs: FineTuning#formatting_func_kwargs.
  LoRA.model_config: LoRA#model_config.
  LoRA.enable_lora: LoRA#enable_lora.
  LoRA.lora_rank: LoRA#lora_rank.
  LoRA.lora_alpha: LoRA#lora_alpha.
  LoRA.lora_module_path: LoRA#lora_module_path.
  LoRA.lora_weight_qtype: LoRA#lora_weight_qtype.
  LoRA.lora_tile_size: LoRA#lora_tile_size.
  Distillation.student_overrides: Distillation#student_overrides.
  Distillation.teacher_overrides: Distillation#teacher_overrides.
  Distillation.offline_data_dir: Distillation#offline_data_dir.
  Distillation.distill_alpha: Distillation#distill_alpha.
  Distillation.distill_temperature: Distillation#distill_temperature.
  Distillation.distill_feature_loss_type: Distillation#distill_feature_loss_type.
  Distillation.distill_layer_indices: Distillation#distill_layer_indices.
  Distillation.learn_to_init_mode: Distillation#learn_to_init_mode.
  Distillation.lti_use_general_linear_map: Distillation#lti_use_general_linear_map.
  Distillation.distill_weights_copy_map: Distillation#distill_weights_copy_map.
  Distillation.distill_student_weights_share_map: Distillation#distill_student_weights_share_map.
  Distillation.attn_module_name: Distillation#attn_module_name.
  Distillation.lti_layer_indices: Distillation#lti_layer_indices.
  Distillation.student_params_to_update: Distillation#student_params_to_update.
  TrainingLoop.log_period: TrainingLoop#log_period.
  TrainingLoop.target_eval_loss: TrainingLoop#target_eval_loss.
  TrainingLoop.abort_on_nan_loss: TrainingLoop#abort_on_nan_loss.
  TrainingLoop.abort_on_inf_loss: TrainingLoop#abort_on_inf_loss.
  TrainingLoop.enable_dropout: TrainingLoop#enable_dropout.
  TrainingLoop.dropout_rate: TrainingLoop#dropout_rate.
  TrainingLoop.enable_data_shuffling: TrainingLoop#enable_data_shuffling.
  TrainingLoop.data_shuffle_seed: TrainingLoop#data_shuffle_seed.
  TrainingLoop.init_weights_seed: TrainingLoop#init_weights_seed.
  ManifoldConstrainedHyperConnections.mhc_expansion_rate: ManifoldConstrainedHyperConnections#mhc_expansion_rate.
  ManifoldConstrainedHyperConnections.sinkhorn_iterations: ManifoldConstrainedHyperConnections#sinkhorn_iterations.
  ManifoldConstrainedHyperConnections.enable_mhc_lite: ManifoldConstrainedHyperConnections#enable_mhc_lite.
  DilocoParams.diloco_sync_period: DilocoParams#diloco_sync_period.
  DilocoParams.diloco_outer_lr: DilocoParams#diloco_outer_lr.
  DilocoParams.diloco_outer_momentum: DilocoParams#diloco_outer_momentum.
  DilocoParams.dcn_bandwidth_limit: DilocoParams#dcn_bandwidth_limit.
  DilocoParams.dcn_bandwidth_burst: DilocoParams#dcn_bandwidth_burst.
  DilocoParams.dcn_bandwidth_latency: DilocoParams#dcn_bandwidth_latency.
  DilocoParams.dcn_bandwidth_interface: DilocoParams#dcn_bandwidth_interface.
  Optimizer.skip_step_on_spikes: Optimizer#skip_step_on_spikes.
  Optimizer.skip_step_interval: Optimizer#skip_step_interval.
  Optimizer.skip_step_scaling_factor: Optimizer#skip_step_scaling_factor.
  Optimizer.use_tunix_gradient_accumulation: Optimizer#use_tunix_gradient_accumulation.
  Optimizer.gradient_clipping_threshold: Optimizer#gradient_clipping_threshold.
  Optimizer.learning_rate: Optimizer#learning_rate.
  Optimizer.learning_rate_final_fraction: Optimizer#learning_rate_final_fraction.
  Optimizer.trainable_parameters_mask: Optimizer#trainable_parameters_mask.
  AdamW.adam_b1: AdamW#adam_b1.
  AdamW.adam_b2: AdamW#adam_b2.
  AdamW.adam_eps: AdamW#adam_eps.
  AdamW.adam_eps_root: AdamW#adam_eps_root.
  AdamW.adam_weight_decay: AdamW#adam_weight_decay.
  AdamW.adamw_mask: AdamW#adamw_mask.
  Muon.muon_beta: Muon#muon_beta.
  Muon.muon_weight_decay: Muon#muon_weight_decay.
  Muon.muon_consistent_rms: Muon#muon_consistent_rms.
  PositionalEmbedding.use_iota_embed: PositionalEmbedding#use_iota_embed.
  PositionalEmbedding.use_untrainable_positional_embedding: PositionalEmbedding#use_untrainable_positional_embedding.
  PositionalEmbedding.trainable_position_size: PositionalEmbedding#trainable_position_size.
  PositionalEmbedding.nope_layer_interval: PositionalEmbedding#nope_layer_interval.
  Rope.rope_use_scale: Rope#rope_use_scale.
  Rope.rope_min_timescale: Rope#rope_min_timescale.
  Rope.rope_max_timescale: Rope#rope_max_timescale.
  Rope.rope_linear_scaling_factor: Rope#rope_linear_scaling_factor.
  Rope.local_rope_max_timescale: Rope#local_rope_max_timescale.
  Rope.global_rope_max_timescale: Rope#global_rope_max_timescale.
  Rope.global_rope_proportion: Rope#global_rope_proportion.
  Rope.local_rope_proportion: Rope#local_rope_proportion.
  YarnRope.max_position_embeddings: YarnRope#max_position_embeddings.
  YarnRope.original_max_position_embeddings: YarnRope#original_max_position_embeddings.
  YarnRope.rope_factor: YarnRope#rope_factor.
  YarnRope.beta_fast: YarnRope#beta_fast.
  YarnRope.beta_slow: YarnRope#beta_slow.
  YarnRope.mscale: YarnRope#mscale.
  YarnRope.rope_interleave: YarnRope#rope_interleave.
  YarnRope.rope_truncate: YarnRope#rope_truncate.
  YarnRope.rope_attention_scaling: YarnRope#rope_attention_scaling.
  InferenceGeneral.max_prefill_predict_length: InferenceGeneral#max_prefill_predict_length.
  InferenceGeneral.prompt: InferenceGeneral#prompt.
  InferenceGeneral.system_prompt: InferenceGeneral#system_prompt.
  InferenceGeneral.load_from_prefill_dir: InferenceGeneral#load_from_prefill_dir.
  InferenceGeneral.autoregressive_decode_assert: InferenceGeneral#autoregressive_decode_assert.
  InferenceGeneral.model_call_mode: InferenceGeneral#model_call_mode.
  InferenceGeneral.use_chunked_prefill: InferenceGeneral#use_chunked_prefill.
  InferenceGeneral.prefill_chunk_size: InferenceGeneral#prefill_chunk_size.
  InferenceGeneral.enable_model_warmup: InferenceGeneral#enable_model_warmup.
  InferenceGeneral.enable_llm_inference_pool: InferenceGeneral#enable_llm_inference_pool.
  InferenceGeneral.multi_sampling: InferenceGeneral#multi_sampling.
  InferenceGeneral.return_log_prob: InferenceGeneral#return_log_prob.
  Decoding.decode_sampling_nucleus_p: Decoding#decode_sampling_nucleus_p.
  Decoding.decode_sampling_top_k: Decoding#decode_sampling_top_k.
  Decoding.decode_sampling_temperature: Decoding#decode_sampling_temperature.
  InferenceLayout.stack_prefill_result_cache: InferenceLayout#stack_prefill_result_cache.
  InferenceLayout.prefill_cache_axis_order: InferenceLayout#prefill_cache_axis_order.
  InferenceLayout.ar_cache_axis_order: InferenceLayout#ar_cache_axis_order.
  InferenceLayout.compute_axis_order: InferenceLayout#compute_axis_order.
  InferenceLayout.reshape_q: InferenceLayout#reshape_q.
  InferenceServer.inference_server: InferenceServer#inference_server.
  InferenceServer.prefill_slice: InferenceServer#prefill_slice.
  InferenceServer.generate_slice: InferenceServer#generate_slice.
  InferenceBenchmark.inference_microbenchmark_prefill_lengths: InferenceBenchmark#inference_microbenchmark_prefill_lengths.
  InferenceBenchmark.inference_microbenchmark_stages: InferenceBenchmark#inference_microbenchmark_stages.
  InferenceBenchmark.inference_microbenchmark_loop_iters: InferenceBenchmark#inference_microbenchmark_loop_iters.
  InferenceBenchmark.inference_microbenchmark_num_samples: InferenceBenchmark#inference_microbenchmark_num_samples.
  InferenceBenchmark.inference_benchmark_test: InferenceBenchmark#inference_benchmark_test.
  PrefixCaching.enable_prefix_caching: PrefixCaching#enable_prefix_caching.
  PrefixCaching.prefix_caching_hbm_byte: PrefixCaching#prefix_caching_hbm_byte.
  PrefixCaching.prefix_caching_dram_byte: PrefixCaching#prefix_caching_dram_byte.
  AOT.write_estimator_result: AOT#write_estimator_result.
  DevelopmentAndDebugging.jax_distributed_initialization_timeout: DevelopmentAndDebugging#jax_distributed_initialization_timeout.
  DevelopmentAndDebugging.jax_debug_log_modules: DevelopmentAndDebugging#jax_debug_log_modules.
  DevelopmentAndDebugging.skip_jax_distributed_system: DevelopmentAndDebugging#skip_jax_distributed_system.
  DevelopmentAndDebugging.max_checkify: DevelopmentAndDebugging#max_checkify.
  Profiling.upload_all_profiler_results: Profiling#upload_all_profiler_results.
  Profiling.skip_first_n_steps_for_profiler: Profiling#skip_first_n_steps_for_profiler.
  Profiling.profiler_steps: Profiling#profiler_steps.
  Profiling.profile_cleanly: Profiling#profile_cleanly.
  Profiling.profile_periodically_period: Profiling#profile_periodically_period.
  Profiling.hide_profiler_step_metric: Profiling#hide_profiler_step_metric.
  Profiling.enable_jax_profiler: Profiling#enable_jax_profiler.
  Profiling.jax_profiler_port: Profiling#jax_profiler_port.
  Profiling.enable_tpu_profiling_options: Profiling#enable_tpu_profiling_options.
  Profiling.tpu_num_chips_to_profile_per_task: Profiling#tpu_num_chips_to_profile_per_task.
  Profiling.tpu_num_sparse_cores_to_trace: Profiling#tpu_num_sparse_cores_to_trace.
  Profiling.tpu_num_sparse_core_tiles_to_trace: Profiling#tpu_num_sparse_core_tiles_to_trace.
  Profiling.xprof_e2e_enable_fw_throttle_event: Profiling#xprof_e2e_enable_fw_throttle_event.
  Profiling.xprof_e2e_enable_fw_power_level_event: Profiling#xprof_e2e_enable_fw_power_level_event.
  Profiling.xprof_e2e_enable_fw_thermal_event: Profiling#xprof_e2e_enable_fw_thermal_event.
  Profiling.profile_power_events: Profiling#profile_power_events.
  HloDump.dump_hlo: HloDump#dump_hlo.
  HloDump.dump_step: HloDump#dump_step.
  HloDump.dump_hlo_delete_local_after: HloDump#dump_hlo_delete_local_after.
  HloDump.dump_hlo_module_name: HloDump#dump_hlo_module_name.
  HloDump.dump_hlo_upload_all: HloDump#dump_hlo_upload_all.
  HloDump.dump_jaxpr: HloDump#dump_jaxpr.
  HloDump.dump_jaxpr_delete_local_after: HloDump#dump_jaxpr_delete_local_after.
  Metrics.gcs_metrics: Metrics#gcs_metrics.
  Metrics.save_config_to_gcs: Metrics#save_config_to_gcs.
  Metrics.record_internal_nn_metrics: Metrics#record_internal_nn_metrics.
  Metrics.prometheus_port: Metrics#prometheus_port.
  Metrics.enable_checkpoint_cloud_logger: Metrics#enable_checkpoint_cloud_logger.
  Metrics.enable_tunix_perf_metrics: Metrics#enable_tunix_perf_metrics.
  ManagedMLDiagnostics.managed_mldiagnostics: ManagedMLDiagnostics#managed_mldiagnostics.
  ManagedMLDiagnostics.managed_mldiagnostics_on_demand_profiling: ManagedMLDiagnostics#managed_mldiagnostics_on_demand_profiling.
  ManagedMLDiagnostics.managed_mldiagnostics_run_group: ManagedMLDiagnostics#managed_mldiagnostics_run_group.
  ManagedMLDiagnostics.managed_mldiagnostics_region: ManagedMLDiagnostics#managed_mldiagnostics_region.
  Goodput.enable_goodput_recording: Goodput#enable_goodput_recording.
  Goodput.monitor_goodput: Goodput#monitor_goodput.
  Goodput.goodput_upload_interval_seconds: Goodput#goodput_upload_interval_seconds.
  Goodput.enable_pathways_goodput: Goodput#enable_pathways_goodput.
  Goodput.monitor_step_time_deviation: Goodput#monitor_step_time_deviation.
  Goodput.step_deviation_interval_seconds: Goodput#step_deviation_interval_seconds.
  Goodput.enable_gcp_goodput_metrics: Goodput#enable_gcp_goodput_metrics.
  Goodput.enable_gcp_step_deviation_metrics: Goodput#enable_gcp_step_deviation_metrics.
  ElasticTraining.elastic_timeout_seconds: ElasticTraining#elastic_timeout_seconds.
  ElasticTraining.elastic_max_retries: ElasticTraining#elastic_max_retries.
  ElasticTraining.elastic_min_slice_count: ElasticTraining#elastic_min_slice_count.
  GcpMonitoring.report_heartbeat_metric_for_gcp_monitoring: GcpMonitoring#report_heartbeat_metric_for_gcp_monitoring.
  GcpMonitoring.heartbeat_reporting_interval_in_seconds: GcpMonitoring#heartbeat_reporting_interval_in_seconds.
  GcpMonitoring.report_performance_metric_for_gcp_monitoring: GcpMonitoring#report_performance_metric_for_gcp_monitoring.
  Tensorboard.enable_tensorboard: Tensorboard#enable_tensorboard.
  Tensorboard.use_vertex_tensorboard: Tensorboard#use_vertex_tensorboard.
  Tensorboard.vertex_tensorboard_project: Tensorboard#vertex_tensorboard_project.
  Tensorboard.vertex_tensorboard_region: Tensorboard#vertex_tensorboard_region.
  MultimodalGeneral.attention_for_vit: MultimodalGeneral#attention_for_vit.
  MultimodalGeneral.freeze_vision_encoder_params: MultimodalGeneral#freeze_vision_encoder_params.
  MultimodalGeneral.freeze_audio_encoder_params: MultimodalGeneral#freeze_audio_encoder_params.
  MultimodalGeneral.use_audio: MultimodalGeneral#use_audio.
  MultimodalGeneral.image_size_for_vit: MultimodalGeneral#image_size_for_vit.
  MultimodalGeneral.image_placeholder: MultimodalGeneral#image_placeholder.
  MultimodalGeneral.posemb_type_for_vit: MultimodalGeneral#posemb_type_for_vit.
  MultimodalGeneral.max_num_images_per_example: MultimodalGeneral#max_num_images_per_example.
  MultimodalGeneral.video_placeholder: MultimodalGeneral#video_placeholder.
  MultimodalGeneral.audio_placeholder: MultimodalGeneral#audio_placeholder.
  MultimodalGeneral.use_audio_in_video: MultimodalGeneral#use_audio_in_video.
  MultimodalGeneral.use_mrope: MultimodalGeneral#use_mrope.
  MultimodalGeneral.mrope_section: MultimodalGeneral#mrope_section.
  MultimodalGeneral.position_id_per_seconds: MultimodalGeneral#position_id_per_seconds.
  VisionTower.hidden_size_for_vit: VisionTower#hidden_size_for_vit.
  VisionTower.intermediate_size_for_vit: VisionTower#intermediate_size_for_vit.
  VisionTower.num_attention_heads_for_vit: VisionTower#num_attention_heads_for_vit.
  VisionTower.num_channels_for_vit: VisionTower#num_channels_for_vit.
  VisionTower.tile_size_for_vit: VisionTower#tile_size_for_vit.
  VisionTower.patch_size_for_vit: VisionTower#patch_size_for_vit.
  VisionTower.conv_stride_for_vit: VisionTower#conv_stride_for_vit.
  VisionTower.num_hidden_layers_for_vit: VisionTower#num_hidden_layers_for_vit.
  VisionTower.rope_theta_for_vit: VisionTower#rope_theta_for_vit.
  VisionTower.vision_output_dim_for_vit: VisionTower#vision_output_dim_for_vit.
  VisionTower.spatial_merge_size_for_vit: VisionTower#spatial_merge_size_for_vit.
  VisionTower.out_hidden_size_for_vit: VisionTower#out_hidden_size_for_vit.
  VisionTower.temporal_patch_size_for_vit: VisionTower#temporal_patch_size_for_vit.
  VisionTower.num_position_embeddings_for_vit: VisionTower#num_position_embeddings_for_vit.
  VisionTower.vision_output_length: VisionTower#vision_output_length.
  VisionProjector.projector_input_dim_for_vit: VisionProjector#projector_input_dim_for_vit.
  VisionProjector.projector_output_dim_for_vit: VisionProjector#projector_output_dim_for_vit.
  VisionProjector.pixel_shuffle_ratio_for_vit: VisionProjector#pixel_shuffle_ratio_for_vit.
  VisionProjector.projector_dropout_for_vit: VisionProjector#projector_dropout_for_vit.
  AudioEncoder.d_model_for_audio: AudioEncoder#d_model_for_audio.
  AudioEncoder.encoder_attention_heads_for_audio: AudioEncoder#encoder_attention_heads_for_audio.
  AudioEncoder.encoder_ffn_dim_for_audio: AudioEncoder#encoder_ffn_dim_for_audio.
  AudioEncoder.encoder_layers_for_audio: AudioEncoder#encoder_layers_for_audio.
  AudioEncoder.attention_dropout_for_audio: AudioEncoder#attention_dropout_for_audio.
  AudioEncoder.activation_dropout_for_audio: AudioEncoder#activation_dropout_for_audio.
  AudioEncoder.activation_function_for_audio: AudioEncoder#activation_function_for_audio.
  AudioEncoder.num_mel_bins_for_audio: AudioEncoder#num_mel_bins_for_audio.
  AudioEncoder.max_source_positions_for_audio: AudioEncoder#max_source_positions_for_audio.
  AudioEncoder.scale_embedding_for_audio: AudioEncoder#scale_embedding_for_audio.
  AudioEncoder.n_window_for_audio: AudioEncoder#n_window_for_audio.
  AudioEncoder.n_window_infer_for_audio: AudioEncoder#n_window_infer_for_audio.
  AudioEncoder.conv_chunksize_for_audio: AudioEncoder#conv_chunksize_for_audio.
  AudioEncoder.downsample_hidden_size_for_audio: AudioEncoder#downsample_hidden_size_for_audio.
  AudioEncoder.output_dim_for_audio: AudioEncoder#output_dim_for_audio.
  AudioEncoder.num_conv_layers_for_audio: AudioEncoder#num_conv_layers_for_audio.
  AudioEncoder.max_timescale_for_audio: AudioEncoder#max_timescale_for_audio.
  AudioEncoder.max_sample_len_for_audio: AudioEncoder#max_sample_len_for_audio.
  Debug.rl: Debug#rl.
  RLHardware.trainer_devices_fraction: RLHardware#trainer_devices_fraction.
  RLHardware.sampler_devices_fraction: RLHardware#sampler_devices_fraction.
  RLHardware.chips_per_vm: RLHardware#chips_per_vm.
  RLHardware.use_pathways: RLHardware#use_pathways.
  RLHardware.num_trainer_slices: RLHardware#num_trainer_slices.
  RLHardware.num_samplers_slices: RLHardware#num_samplers_slices.
  RLHardware.rollout_data_parallelism: RLHardware#rollout_data_parallelism.
  RLHardware.rollout_tensor_parallelism: RLHardware#rollout_tensor_parallelism.
  RLHardware.rollout_expert_parallelism: RLHardware#rollout_expert_parallelism.
  RLHardware.inference_replicas: RLHardware#inference_replicas.
  RLHardware.inference_devices_per_replica: RLHardware#inference_devices_per_replica.
  RLHardware.inference_rollouts: RLHardware#inference_rollouts.
  RLHardware.use_pathways_reshard: RLHardware#use_pathways_reshard.
  VLLM.kv_cache_buffer: VLLM#kv_cache_buffer.
  VLLM.hbm_utilization_vllm: VLLM#hbm_utilization_vllm.
  VLLM.swap_space_vllm_gb: VLLM#swap_space_vllm_gb.
  VLLM.enable_dp_attention: VLLM#enable_dp_attention.
  VLLM.enable_expert_parallel: VLLM#enable_expert_parallel.
  VLLM.async_scheduling: VLLM#async_scheduling.
  VLLM.max_num_batched_tokens: VLLM#max_num_batched_tokens.
  VLLM.max_num_seqs: VLLM#max_num_seqs.
  VLLM.stop_strings: VLLM#stop_strings.
  VLLM.vllm_additional_config: VLLM#vllm_additional_config.
  VLLM.vllm_hf_overrides: VLLM#vllm_hf_overrides.
  VLLM.vllm_hf_config_path: VLLM#vllm_hf_config_path.
  VLLM.use_standalone_converter: VLLM#use_standalone_converter.
  VLLM.vllm_load_format: VLLM#vllm_load_format.
  VLLM.debug_converter: VLLM#debug_converter.
  VLLM.gcs_debug_path: VLLM#gcs_debug_path.
  RL.num_generations: RL#num_generations.
  RL.num_iterations: RL#num_iterations.
  RL.grpo_beta: RL#grpo_beta.
  RL.grpo_epsilon: RL#grpo_epsilon.
  RL.loss_agg_mode: RL#loss_agg_mode.
  RL.use_agentic_rollout: RL#use_agentic_rollout.
  RL.max_concurrency: RL#max_concurrency.
  RL.off_policy_steps: RL#off_policy_steps.
  RL.system_prompt: RL#system_prompt.
  RL.degenerate_group_masking: RL#degenerate_group_masking.
  RL.epsilon_high: RL#epsilon_high.
  RL.reshard_chunk_size: RL#reshard_chunk_size.
  RLDataset.batch_size: RLDataset#batch_size.
  RLDataset.num_batches: RLDataset#num_batches.
  RLDataset.num_test_batches: RLDataset#num_test_batches.
  RLDataset.eval_batch_size: RLDataset#eval_batch_size.
  RLDataset.test_batch_start_index: RLDataset#test_batch_start_index.
  RLDataset.train_fraction: RLDataset#train_fraction.
  RLDataset.train_micro_batch_size: RLDataset#train_micro_batch_size.
  RLDataset.rollout_micro_batch_size: RLDataset#rollout_micro_batch_size.
  RLDataset.dataset_processor_path: RLDataset#dataset_processor_path.
  RLEvaluation.eval_sampling_strategy: RLEvaluation#eval_sampling_strategy.
  RLEvaluation.generation_configs: RLEvaluation#generation_configs.
  RLEvaluation.num_eval_passes: RLEvaluation#num_eval_passes.
  RLEvaluation.eval_corr_lst: RLEvaluation#eval_corr_lst.
  RLEvaluation.eval_make_lst: RLEvaluation#eval_make_lst.
  RLEvaluation.eval_mode: RLEvaluation#eval_mode.
  Reward.reward_exact_answer: Reward#reward_exact_answer.
  Reward.reward_exact_format_match: Reward#reward_exact_format_match.
  Reward.reward_white_space_format_match: Reward#reward_white_space_format_match.
  Reward.reward_partial_format_match: Reward#reward_partial_format_match.
  Reward.reward_ratio_guess_to_answer_high: Reward#reward_ratio_guess_to_answer_high.
  Reward.reward_ratio_guess_to_answer_low: Reward#reward_ratio_guess_to_answer_low.
  Reward.penalty_incorrect_format: Reward#penalty_incorrect_format.
  Reward.penalty_incorrect_answer: Reward#penalty_incorrect_answer.
  Reward.math_verify_timeout: Reward#math_verify_timeout.
  Reward.math_verify_num_procs: Reward#math_verify_num_procs.
  SpecialTokens.reasoning_start_token: SpecialTokens#reasoning_start_token.
  SpecialTokens.reasoning_end_token: SpecialTokens#reasoning_end_token.
  SpecialTokens.solution_start_token: SpecialTokens#solution_start_token.
  SpecialTokens.solution_end_token: SpecialTokens#solution_end_token.
  Engram.engram_num_heads: Engram#engram_num_heads.
  Engram.engram_head_dim: Engram#engram_head_dim.
  Engram.engram_kernel_size: Engram#engram_kernel_size.
  Engram.engram_seed: Engram#engram_seed.
  DerivedValues.convert_checkpoint_if_possible: DerivedValues#convert_checkpoint_if_possible.
  MaxTextConfig.model_config: MaxTextConfig#model_config.
  MaxTextConfig.load_model_specific_defaults: MaxTextConfig#load_model_specific_defaults().
---
# Module: [`src/maxtext/configs/types.py`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py)

## Classes
### `AOT`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1770`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1770)
- doc: Ahead of Time (AOT) Compilation settings.
- signature: `class AOT(BaseModel):`
- members:
  - `compile_topology` — [`L1774`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1774)
  - `compile_topology_num_slices` — [`L1775`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1775)
  - `compiled_trainstep_file` — [`L1773`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1773)
  - `write_estimator_result` — [`L1776`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1776)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`PathStr`](types.md#PathStr)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig), [`get_num_target_devices`](types.md#MaxTextConfig.get_num_target_devices)

### `AdamW`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1595`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1595)
- doc: Configuration specific to the AdamW optimizer.
- signature: `class AdamW(BaseModel):`
- members:
  - `adam_b1` — [`L1598`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1598)
  - `adam_b2` — [`L1602`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1602)
  - `adam_eps` — [`L1606`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1606)
  - `adam_eps_root` — [`L1610`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1610)
  - `adam_weight_decay` — [`L1614`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1614)
  - `adamw_mask` — [`L1615`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1615)
  - `mu_dtype` — [`L1621`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1621)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `Attention`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:552`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L552)
- doc: General configuration for the attention mechanism.
- signature: `class Attention(BaseModel):`
- members:
  - `attention` — [`L555`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L555) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `attention_sink` — [`L598`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L598)
  - `attention_type` — [`L559`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L559) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `attn_logits_soft_cap` — [`L606`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L606)
  - `chunk_attn_window_size` — [`L605`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L605)
  - `enable_padding_causal_mask` — [`L624`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L624)
  - `float32_logits` — [`L600`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L600)
  - `float32_qk_product` — [`L599`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L599)
  - `force_q_layout` — [`L627`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L627)
  - `global_num_kv_heads` — [`L566`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L566)
  - `hidden_size_per_layer_input` — [`L570`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L570)
  - `num_kv_shared_layers` — [`L584`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L584)
  - `qk_clip_threshold` — [`L629`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L629)
  - `ragged_block_size` — [`L623`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L623)
  - `share_kv_projections` — [`L562`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L562)
  - `sliding_window_size` — [`L604`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L604)
  - `use_double_wide_mlp` — [`L591`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L591)
  - `use_gmm_v2` — [`L616`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L616)
  - `use_jax_splash` — [`L626`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L626)
  - `use_post_attn_norm` — [`L609`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L609)
  - `use_post_ffw_norm` — [`L610`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L610)
  - `use_qk_clip` — [`L628`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L628)
  - `use_ragged_attention` — [`L611`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L611)
  - `use_tokamax_gmm` — [`L612`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L612)
  - `use_tokamax_splash` — [`L625`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L625)
  - `vocab_size_per_layer_input` — [`L577`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L577)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig), [`_validate_check_vma_is_supported`](types.md#MaxTextConfig._validate_check_vma_is_supported)

### `AttentionIndexer`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:662`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L662)
- doc: Configuration for DeepSeek Sparse Attention (DSA): DeepSeek3.2-style MLA with indexer.
- signature: `class AttentionIndexer(BaseModel):`
- members:
  - `indexer_approx_top_k_recall` — [`L677`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L677)
  - `indexer_head_dim` — [`L666`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L666)
  - `indexer_loss_scaling_factor` — [`L673`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L673)
  - `indexer_n_heads` — [`L667`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L667)
  - `indexer_sparse_training` — [`L669`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L669)
  - `indexer_topk` — [`L668`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L668)
  - `indexer_use_approx_top_k` — [`L674`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L674)
  - `use_indexer` — [`L665`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L665)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `AudioEncoder`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:2038`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2038)
- doc: Configuration for the Audio Encoder in a multimodal model.
- signature: `class AudioEncoder(BaseModel):`
- members:
  - `activation_dropout_for_audio` — [`L2046`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2046)
  - `activation_function_for_audio` — [`L2047`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2047)
  - `attention_dropout_for_audio` — [`L2045`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2045)
  - `conv_chunksize_for_audio` — [`L2053`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2053)
  - `d_model_for_audio` — [`L2041`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2041)
  - `downsample_hidden_size_for_audio` — [`L2054`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2054)
  - `encoder_attention_heads_for_audio` — [`L2042`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2042)
  - `encoder_ffn_dim_for_audio` — [`L2043`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2043)
  - `encoder_layers_for_audio` — [`L2044`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2044)
  - `max_sample_len_for_audio` — [`L2058`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2058)
  - `max_source_positions_for_audio` — [`L2049`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2049)
  - `max_timescale_for_audio` — [`L2057`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2057)
  - `n_window_for_audio` — [`L2051`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2051)
  - `n_window_infer_for_audio` — [`L2052`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2052)
  - `num_conv_layers_for_audio` — [`L2056`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2056)
  - `num_mel_bins_for_audio` — [`L2048`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2048)
  - `output_dim_for_audio` — [`L2055`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2055)
  - `scale_embedding_for_audio` — [`L2050`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2050)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `Checkpointing`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:314`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L314)
- doc: Core configuration for checkpointing and run restoration.
- signature: `class Checkpointing(BaseModel):`
- members:
  - `async_checkpointing` — [`L329`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L329)
  - `checkpoint_conversion_fn` — [`L350`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L350)
  - `checkpoint_is_quantized` — [`L344`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L344)
  - `checkpoint_period` — [`L330`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L330)
  - `checkpoint_todelete_full_path` — [`L339`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L339)
  - `checkpoint_todelete_subdir` — [`L335`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L335)
  - `colocated_python_checkpointing` — [`L358`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L358)
  - `enable_autocheckpoint` — [`L362`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L362)
  - `enable_checkpointing` — [`L327`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L327)
  - `enable_continuous_checkpointing` — [`L357`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L357)
  - `enable_orbax_v1` — [`L349`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L349)
  - `enable_single_replica_ckpt_restoring` — [`L332`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L332)
  - `force_unroll` — [`L340`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L340)
  - `hf_lora_adapter_path` — [`L319`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L319)
  - `load_checkpoint_only_once` — [`L328`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L328)
  - `load_full_state_path` — [`L326`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L326)
  - `load_parameters_path` — [`L317`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L317)
  - `lora_input_adapters_path` — [`L318`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L318)
  - `max_num_checkpoints_to_keep` — [`L331`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L331)
  - `save_checkpoint_on_completion` — [`L354`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L354)
  - `save_quantized_params_path` — [`L348`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L348)
  - `source_checkpoint_layout` — [`L351`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L351)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`PathStr`](types.md#PathStr)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `CompressedAttention`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:651`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L651)
- doc: Configuration for Compressed Attention.
- signature: `class CompressedAttention(BaseModel):`
- members:
  - `compress_ratios` — [`L656`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L656)
  - `compressed_rope_max_timescale` — [`L657`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L657)
  - `o_groups` — [`L655`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L655)
  - `o_lora_rank` — [`L654`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L654)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `DPO`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1330`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1330)
- doc: Configuration for DPO and ORPO preference optimization algorithms.
- signature: `class DPO(BaseModel):`
- members:
  - `algo` — [`L1333`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1333)
  - `dpo_beta` — [`L1334`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1334)
  - `dpo_label_smoothing` — [`L1336`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1336)
  - `max_prompt_length` — [`L1337`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1337)
  - `orpo_lambda` — [`L1335`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1335)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`dpo`](types.md#MaxTextConfig.dpo)

### `DType`  ·  implements/extends Enum, str
- def: [`src/maxtext/configs/types.py:63`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L63)
- doc: Supported data types for weights and activations.
- signature: `class DType(str, Enum):`
- members:
  - `BFLOAT16` — [`L66`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L66)
  - `FLOAT16` — [`L68`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L68)
  - `FLOAT32` — [`L67`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L67)
- used by: [`weight_dtype`](types.md#DataTypes.weight_dtype), [`dtype`](types.md#DataTypes.dtype), [`grad_dtype`](types.md#DataTypes.grad_dtype)

### `DataTypes`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:410`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L410)
- doc: Configuration for data types and precision.
- signature: `class DataTypes(BaseModel):`
- members:
  - `activations_in_float32` — [`L420`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L420)
  - `dtype` — [`L413`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L413)
  - `dtype_mm` — [`L424`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L424)
  - `grad_dtype` — [`L414`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L414)
  - `matmul_precision` — [`L416`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L416)
  - `weight_dtype` — [`L415`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L415) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`DType`](types.md#DType), [`MatmulPrecision`](types.md#MatmulPrecision), [`FLOAT32`](types.md#DType.FLOAT32), [`BFLOAT16`](types.md#DType.BFLOAT16), [`DEFAULT`](types.md#MatmulPrecision.DEFAULT)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `DatasetGeneral`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1200`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1200)
- doc: General configuration for dataset and data loading.
- signature: `class DatasetGeneral(BaseModel):`
- members:
  - `colocated_python_data_input` — [`L1241`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1241)
  - `dataset_type` — [`L1203`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1203)
  - `enable_rampup_batch_size` — [`L1237`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1237)
  - `eval_data_columns` — [`L1212`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1212)
  - `eval_image_column` — [`L1213`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1213)
  - `eval_per_device_batch_size` — [`L1205`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1205)
  - `expansion_factor_real_data` — [`L1227`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1227)
  - `generate_padding_batch_eval` — [`L1233`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1233)
  - `generate_padding_batch_train` — [`L1229`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1229)
  - `global_rampup_samples` — [`L1240`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1240)
  - `grain_packing_type` — [`L1218`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1218)
  - `max_corpus_chars` — [`L1209`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1209)
  - `max_segments_per_seq` — [`L1222`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1222)
  - `num_epoch` — [`L1226`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1226)
  - `packing` — [`L1214`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1214) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `per_device_batch_size` — [`L1204`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1204)
  - `per_device_batch_size_increment` — [`L1239`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1239)
  - `per_device_batch_size_start` — [`L1238`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1238)
  - `reuse_example_batch` — [`L1228`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1228)
  - `train_data_columns` — [`L1210`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1210)
  - `train_image_column` — [`L1211`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1211)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`DatasetType`](types.md#DatasetType), [`TFDS`](types.md#DatasetType.TFDS)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `DatasetType`  ·  implements/extends Enum, str
- def: [`src/maxtext/configs/types.py:174`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L174) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
- doc: Supported data loading pipelines.
- signature: `class DatasetType(str, Enum):`
- members:
  - `C4MLPERF` — [`L181`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L181)
  - `GRAIN` — [`L179`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L179)
  - `HF` — [`L178`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L178)
  - `OLMO_GRAIN` — [`L182`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L182)
  - `SYNTHETIC` — [`L177`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L177)
  - `TFDS` — [`L180`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L180)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`dataset_type`](types.md#DatasetGeneral.dataset_type)

### `DcnParallelism`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1028`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1028)
- doc: Parallelism dimensions across the DCN (Data Center Network).
- signature: `class DcnParallelism(BaseModel):`
- members:
  - `dcn_autoregressive_parallelism` — [`L1045`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1045)
  - `dcn_context_autoregressive_parallelism` — [`L1037`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1037)
  - `dcn_context_parallelism` — [`L1036`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1036)
  - `dcn_data_parallelism` — [`L1032`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1032)
  - `dcn_diloco_parallelism` — [`L1031`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1031) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `dcn_expert_parallelism` — [`L1044`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1044)
  - `dcn_fsdp_parallelism` — [`L1033`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1033)
  - `dcn_fsdp_transpose_parallelism` — [`L1034`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1034)
  - `dcn_pipeline_parallelism` — [`L1043`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1043)
  - `dcn_sequence_parallelism` — [`L1035`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1035)
  - `dcn_tensor_parallelism` — [`L1038`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1038)
  - `dcn_tensor_sequence_parallelism` — [`L1040`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1040)
  - `dcn_tensor_transpose_parallelism` — [`L1039`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1039)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `Debug`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:2061`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2061)
- doc: Configuration for debugging options.
- signature: `class Debug(BaseModel):`
- members:
  - `rl` — [`L2064`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2064)
- used by: [`debug`](types.md#MaxTextConfig.debug)

### `Decoding`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1718`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1718)
- doc: Configuration for decoding and sampling strategies.
- signature: `class Decoding(BaseModel):`
- members:
  - `decode_sampling_nucleus_p` — [`L1722`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1722)
  - `decode_sampling_strategy` — [`L1721`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1721)
  - `decode_sampling_temperature` — [`L1724`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1724)
  - `decode_sampling_top_k` — [`L1723`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1723)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`SamplingStrategy`](types.md#SamplingStrategy), [`GREEDY`](types.md#SamplingStrategy.GREEDY)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `DeepSeekMoE`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:903`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L903)
- doc: Configuration specific to DeepSeek-style MoE layers.
- signature: `class DeepSeekMoE(BaseModel):`
- members:
  - `batch_split_factor` — [`L926`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L926)
  - `first_num_dense_layers` — [`L906`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L906)
  - `first_num_hash_layers` — [`L918`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L918)
  - `mlp_bias` — [`L912`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L912)
  - `n_routing_groups` — [`L917`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L917)
  - `routed_bias` — [`L910`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L910)
  - `routed_bias_update_rate` — [`L911`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L911)
  - `routed_scaling_factor` — [`L908`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L908)
  - `routed_score_func` — [`L909`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L909)
  - `shared_experts` — [`L907`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L907)
  - `topk_routing_group` — [`L921`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L921)
  - `use_batch_split_schedule` — [`L922`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L922)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `DerivedValues`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:2268`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2268)
- doc: Holds all fields that are derived from other config values for perfect legacy compatibility.
- signature: `class DerivedValues(BaseModel):`
- members:
  - `checkpoint_dir` — [`L2346`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2346)
  - `convert_checkpoint_if_possible` — [`L2350`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2350)
  - `dcn_parallelism` — [`L2302`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2302)
  - `emb_dim` — [`L2271`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2271)
  - `global_batch_size_to_eval_on` — [`L2322`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2322)
  - `global_batch_size_to_load` — [`L2327`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2327)
  - `global_batch_size_to_load_eval` — [`L2331`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2331)
  - `global_batch_size_to_load_increment` — [`L2376`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2376)
  - `global_batch_size_to_load_start` — [`L2375`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2375)
  - `global_batch_size_to_train_on` — [`L2317`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2317)
  - `ici_parallelism` — [`L2298`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2298)
  - `managed_mldiagnostics_dir` — [`L2363`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2363)
  - `metrics_dir` — [`L2355`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2355)
  - `micro_batch_size_to_eval_on` — [`L2340`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2340)
  - `micro_batch_size_to_train_on` — [`L2335`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2335)
  - `mlp_dim` — [`L2275`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2275)
  - `moe_mlp_dim` — [`L2276`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2276)
  - `num_decoder_layers` — [`L2280`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2280)
  - `num_diloco_replicas` — [`L2293`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2293)
  - `num_kv_heads` — [`L2284`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2284)
  - `num_query_heads` — [`L2288`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2288)
  - `num_target_devices` — [`L2312`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2312) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `rampup_end_step` — [`L2368`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2368)
  - `rampup_samples_per_increment_to_load` — [`L2379`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2379)
  - `tensorboard_dir` — [`L2359`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2359)
  - `tensors_on_device` — [`L2369`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2369)
  - `tensors_to_offload` — [`L2372`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2372)
  - `using_pipeline_parallelism` — [`L2307`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2307)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `DevelopmentAndDebugging`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1779`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1779)
- doc: General settings for development and debugging.
- signature: `class DevelopmentAndDebugging(BaseModel):`
- members:
  - `_clean_empty_string_for_list(cls, v: Any)` — [`L1798`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1798) — Coerces an empty string from YAML into an empty list before validation.
  - `constant_bound_config` — [`L1782`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1782)
  - `enable_single_controller` — [`L1790`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1790) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `jax_cache_dir` — [`L1783`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1783)
  - `jax_debug_log_modules` — [`L1788`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1788)
  - `jax_distributed_initialization_timeout` — [`L1787`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1787)
  - `max_checkify` — [`L1792`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1792)
  - `skip_jax_distributed_system` — [`L1789`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1789)
  - `subslice_shape` — [`L1791`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1791)
- protocol/private: `_validate_config`[`L1807`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1807)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`PathStr`](types.md#PathStr)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig), [`get_num_target_devices`](types.md#MaxTextConfig.get_num_target_devices)

### `DilocoParams`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1521`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1521)
- doc: Diloco Hyperparameters
- signature: `class DilocoParams(BaseModel):`
- members:
  - `dcn_bandwidth_burst` — [`L1531`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1531)
  - `dcn_bandwidth_interface` — [`L1535`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1535)
  - `dcn_bandwidth_latency` — [`L1532`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1532)
  - `dcn_bandwidth_limit` — [`L1528`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1528)
  - `diloco_outer_lr` — [`L1526`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1526)
  - `diloco_outer_momentum` — [`L1527`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1527)
  - `diloco_sync_period` — [`L1525`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1525)
  - `enable_diloco` — [`L1524`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1524)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `Distillation`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1393`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1393)
- doc: Configuration for Knowledge Distillation.
- signature: `class Distillation(BaseModel):`
- members:
  - `attn_module_name` — [`L1459`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1459)
  - `distill_alpha` — [`L1415`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1415)
  - `distill_alpha_end` — [`L1423`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1423)
  - `distill_alpha_schedule` — [`L1424`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1424)
  - `distill_beta` — [`L1417`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1417)
  - `distill_beta_end` — [`L1433`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1433)
  - `distill_beta_schedule` — [`L1436`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1436)
  - `distill_feature_loss_type` — [`L1418`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1418)
  - `distill_layer_indices` — [`L1422`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1422)
  - `distill_student_weights_share_map` — [`L1454`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1454)
  - `distill_temperature` — [`L1416`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1416)
  - `distill_temperature_end` — [`L1427`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1427)
  - `distill_temperature_schedule` — [`L1430`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1430)
  - `distill_weights_copy_map` — [`L1449`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1449)
  - `learn_to_init_mode` — [`L1441`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1441)
  - `lti_layer_indices` — [`L1463`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1463)
  - `lti_use_general_linear_map` — [`L1443`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1443)
  - `offline_data_dir` — [`L1409`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1409)
  - `student_overrides` — [`L1399`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1399)
  - `student_params_to_update` — [`L1469`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1469)
  - `teacher_overrides` — [`L1403`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1403)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `ElasticTraining`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1909`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1909)
- doc: Configuration for elastic training and fault tolerance.
- signature: `class ElasticTraining(BaseModel):`
- members:
  - `elastic_enabled` — [`L1915`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1915)
  - `elastic_max_retries` — [`L1923`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1923)
  - `elastic_min_slice_count` — [`L1927`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1927)
  - `elastic_timeout_seconds` — [`L1916`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1916)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig), [`get_num_target_devices`](types.md#MaxTextConfig.get_num_target_devices)

### `EmergencyCheckpointing`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:381`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L381)
- doc: Configuration for emergency (local) checkpointing.
- signature: `class EmergencyCheckpointing(BaseModel):`
- members:
  - `enable_emergency_checkpoint` — [`L397`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L397)
  - `enable_multi_tier_checkpointing` — [`L384`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L384)
  - `local_checkpoint_directory` — [`L387`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L387)
  - `local_checkpoint_period` — [`L388`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L388)
  - `mtc_data_parallelism` — [`L393`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L393)
  - `multi_tier_checkpointing_backup_interval_minutes` — [`L389`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L389)
  - `replicator_backup_interval_minutes` — [`L405`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L405)
  - `use_replicator_service` — [`L401`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L401)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`PathStr`](types.md#PathStr)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `Engram`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:2250`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2250)
- doc: Configuration for DeepSeek Engram (https://www.arxiv.org/pdf/2601.07372).
- signature: `class Engram(BaseModel):`
- members:
  - `engram_head_dim` — [`L2258`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2258)
  - `engram_kernel_size` — [`L2264`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2264)
  - `engram_layers` — [`L2253`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2253)
  - `engram_max_ngram_size` — [`L2263`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2263)
  - `engram_num_heads` — [`L2257`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2257)
  - `engram_seed` — [`L2265`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2265)
  - `engram_vocab_bases` — [`L2259`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2259)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `FineTuning`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1344`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1344)
- doc: Configuration for fine-tuning methods like DPO, SFT, and GRPO.
- signature: `class FineTuning(BaseModel):`
- members:
  - `formatting_func_kwargs` — [`L1356`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1356)
  - `formatting_func_path` — [`L1352`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1352)
  - `sft_train_on_completion_only` — [`L1349`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1349)
  - `use_dpo` — [`L1347`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1347)
  - `use_grpo` — [`L1360`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1360)
  - `use_sft` — [`L1348`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1348)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `GcpMonitoring`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1933`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1933)
- doc: Configuration for GCP-specific workload monitoring.
- signature: `class GcpMonitoring(BaseModel):`
- members:
  - `heartbeat_reporting_interval_in_seconds` — [`L1939`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1939)
  - `report_heartbeat_metric_for_gcp_monitoring` — [`L1936`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1936)
  - `report_performance_metric_for_gcp_monitoring` — [`L1940`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1940)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `Goodput`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1896`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1896)
- doc: Configuration for goodput monitoring.
- signature: `class Goodput(BaseModel):`
- members:
  - `enable_gcp_goodput_metrics` — [`L1905`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1905)
  - `enable_gcp_step_deviation_metrics` — [`L1906`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1906)
  - `enable_goodput_recording` — [`L1899`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1899)
  - `enable_pathways_goodput` — [`L1902`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1902)
  - `goodput_upload_interval_seconds` — [`L1901`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1901)
  - `monitor_goodput` — [`L1900`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1900)
  - `monitor_step_time_deviation` — [`L1903`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1903)
  - `step_deviation_interval_seconds` — [`L1904`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1904)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `GrainDataset`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1266`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1266)
- doc: Configuration specific to Grain datasets.
- signature: `class GrainDataset(BaseModel):`
- members:
  - `grain_data_source_max_workers` — [`L1297`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1297)
  - `grain_eval_files` — [`L1270`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1270)
  - `grain_file_type` — [`L1275`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1275)
  - `grain_num_threads` — [`L1291`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1291)
  - `grain_num_threads_eval` — [`L1293`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1293)
  - `grain_per_worker_buffer_size` — [`L1287`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1287)
  - `grain_per_worker_buffer_size_eval` — [`L1289`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1289)
  - `grain_prefetch_buffer_size` — [`L1292`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1292)
  - `grain_prefetch_buffer_size_eval` — [`L1294`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1294)
  - `grain_ram_budget_mb` — [`L1290`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1290)
  - `grain_shuffle_buffer_size` — [`L1301`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1301)
  - `grain_train_files` — [`L1269`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1269)
  - `grain_train_mixture_config_path` — [`L1271`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1271)
  - `grain_use_elastic_iterator` — [`L1279`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1279)
  - `grain_worker_count` — [`L1286`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1286)
  - `grain_worker_count_eval` — [`L1288`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1288)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`PathStr`](types.md#PathStr)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `HardwareAndMesh`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:951`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L951)
- doc: Configuration for hardware and parallelism mesh.
- signature: `class HardwareAndMesh(BaseModel):`
- members:
  - `allow_split_physical_axes` — [`L990`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L990)
  - `context_parallel_load_balance` — [`L977`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L977)
  - `context_parallel_reorder_strategy` — [`L982`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L982)
  - `context_parallel_strategy` — [`L978`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L978)
  - `custom_mesh` — [`L986`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L986)
  - `custom_mesh_and_rule` — [`L987`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L987)
  - `enable_nnx` — [`L991`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L991)
  - `hardware` — [`L954`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L954) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `inhomogeneous_layer_cycle_interval` — [`L974`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L974)
  - `mesh_axes` — [`L956`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L956) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `num_slices` — [`L955`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L955) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `optimize_mesh_for_tpu_v6e` — [`L992`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L992)
  - `param_scan_axis` — [`L976`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L976)
  - `pure_nnx` — [`L995`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L995)
  - `pure_nnx_decoder` — [`L994`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L994)
  - `remove_size_one_mesh_axis_from_type` — [`L996`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L996)
  - `scan_layers` — [`L975`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L975)
  - `shard_mode` — [`L973`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L973) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `shardy` — [`L993`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L993)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig), [`_validate_check_vma_is_supported`](types.md#MaxTextConfig._validate_check_vma_is_supported)

### `HfDataset`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1254`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1254)
- doc: Configuration specific to HuggingFace datasets.
- signature: `class HfDataset(BaseModel):`
- members:
  - `hf_access_token` — [`L1263`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1263)
  - `hf_data_dir` — [`L1259`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1259)
  - `hf_eval_files` — [`L1262`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1262)
  - `hf_eval_split` — [`L1261`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1261)
  - `hf_name` — [`L1258`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1258)
  - `hf_path` — [`L1257`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1257)
  - `hf_train_files` — [`L1260`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1260)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`PathStr`](types.md#PathStr)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `HloDump`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1845`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1845)
- doc: Configuration for dumping HLO modules for debugging.
- signature: `class HloDump(BaseModel):`
- members:
  - `dump_hlo` — [`L1848`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1848)
  - `dump_hlo_delete_local_after` — [`L1854`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1854)
  - `dump_hlo_gcs_dir` — [`L1855`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1855)
  - `dump_hlo_local_dir` — [`L1850`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1850)
  - `dump_hlo_local_module_name` — [`L1857`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1857)
  - `dump_hlo_module_name` — [`L1856`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1856)
  - `dump_hlo_upload_all` — [`L1859`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1859)
  - `dump_hlo_xla_flags` — [`L1858`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1858)
  - `dump_jaxpr` — [`L1860`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1860)
  - `dump_jaxpr_delete_local_after` — [`L1865`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1865)
  - `dump_jaxpr_gcs_dir` — [`L1866`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1866)
  - `dump_jaxpr_local_dir` — [`L1861`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1861)
  - `dump_step` — [`L1849`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1849)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`PathStr`](types.md#PathStr)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig), [`validate_and_set_hlo_dump_defaults`](types.md#MaxTextConfig.validate_and_set_hlo_dump_defaults)

### `IciParallelism`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1048`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1048)
- doc: Parallelism dimensions within the ICI (Inter-Chip Interconnect).
- signature: `class IciParallelism(BaseModel):`
- members:
  - `ici_autoregressive_parallelism` — [`L1061`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1061)
  - `ici_context_autoregressive_parallelism` — [`L1057`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1057)
  - `ici_context_parallelism` — [`L1056`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1056)
  - `ici_data_parallelism` — [`L1052`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1052)
  - `ici_diloco_parallelism` — [`L1051`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1051)
  - `ici_expert_parallelism` — [`L1063`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1063)
  - `ici_fsdp_parallelism` — [`L1053`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1053)
  - `ici_fsdp_transpose_parallelism` — [`L1054`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1054)
  - `ici_pipeline_parallelism` — [`L1062`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1062)
  - `ici_sequence_parallelism` — [`L1055`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1055)
  - `ici_tensor_parallelism` — [`L1058`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1058)
  - `ici_tensor_sequence_parallelism` — [`L1060`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1060)
  - `ici_tensor_transpose_parallelism` — [`L1059`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1059)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig), [`_validate_check_vma_is_supported`](types.md#MaxTextConfig._validate_check_vma_is_supported)

### `InferenceBenchmark`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1745`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1745)
- doc: Configuration for running inference microbenchmarks.
- signature: `class InferenceBenchmark(BaseModel):`
- members:
  - `inference_benchmark_test` — [`L1756`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1756)
  - `inference_metadata_file` — [`L1755`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1755)
  - `inference_microbenchmark_log_file_path` — [`L1753`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1753)
  - `inference_microbenchmark_loop_iters` — [`L1752`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1752)
  - `inference_microbenchmark_num_samples` — [`L1754`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1754)
  - `inference_microbenchmark_prefill_lengths` — [`L1748`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1748)
  - `inference_microbenchmark_stages` — [`L1751`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1751)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`PathStr`](types.md#PathStr)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `InferenceGeneral`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1689`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1689)
- doc: General configuration for inference.
- signature: `class InferenceGeneral(BaseModel):`
- members:
  - `autoregressive_decode_assert` — [`L1705`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1705)
  - `enable_llm_inference_pool` — [`L1713`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1713)
  - `enable_model_warmup` — [`L1712`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1712)
  - `load_from_prefill_dir` — [`L1703`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1703)
  - `max_prefill_predict_length` — [`L1693`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1693)
  - `max_target_length` — [`L1692`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1692)
  - `model_call_mode` — [`L1709`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1709)
  - `multi_sampling` — [`L1714`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1714)
  - `prefill_cache_dir` — [`L1704`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1704)
  - `prefill_chunk_size` — [`L1711`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1711)
  - `prompt` — [`L1694`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1694)
  - `return_log_prob` — [`L1715`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1715)
  - `system_prompt` — [`L1695`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1695)
  - `use_chunked_prefill` — [`L1710`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1710)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`PathStr`](types.md#PathStr)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `InferenceLayout`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1727`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1727)
- doc: Configuration for KV cache and compute layouts during inference.
- signature: `class InferenceLayout(BaseModel):`
- members:
  - `ar_cache_axis_order` — [`L1732`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1732)
  - `compute_axis_order` — [`L1733`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1733)
  - `prefill_cache_axis_order` — [`L1731`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1731)
  - `reshape_q` — [`L1734`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1734)
  - `stack_prefill_result_cache` — [`L1730`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1730)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `InferenceServer`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1737`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1737)
- doc: Configuration for running as an inference server.
- signature: `class InferenceServer(BaseModel):`
- members:
  - `generate_slice` — [`L1742`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1742)
  - `inference_server` — [`L1740`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1740)
  - `prefill_slice` — [`L1741`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1741)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `KvQuantAxis`  ·  implements/extends Enum, str
- def: [`src/maxtext/configs/types.py:104`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L104)
- doc: Axes to quantize over for the Key-Value cache.
- signature: `class KvQuantAxis(str, Enum):`
- members:
  - `DKV` — [`L108`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L108)
  - `HEADS_AND_DKV` — [`L109`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L109)
  - `NONE` — [`L107`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L107)
- used by: [`kv_quant_axis`](types.md#Quantization.kv_quant_axis)

### `LayoutAndSharding`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1001`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1001)
- doc: Configuration for data and model sharding rules.
- signature: `class LayoutAndSharding(BaseModel):`
- members:
  - `check_vma` — [`L1017`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1017)
  - `compile_xla_flags` — [`L1025`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1025)
  - `context_sharding` — [`L1006`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1006) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `data_sharding` — [`L1005`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1005) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `input_data_sharding_logical_axes` — [`L1007`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1007)
  - `internal_compile` — [`L1023`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1023)
  - `internal_compile_num_devices` — [`L1024`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1024)
  - `logical_axis_rules` — [`L1004`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1004)
  - `shard_optimizer_over_data` — [`L1022`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1022)
  - `sharding_tolerance` — [`L1011`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1011)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig), [`get_num_target_devices`](types.md#MaxTextConfig.get_num_target_devices), [`_validate_check_vma_is_supported`](types.md#MaxTextConfig._validate_check_vma_is_supported)

### `LearningRateScheduleType`  ·  implements/extends Enum, str
- def: [`src/maxtext/configs/types.py:144`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L144)
- doc: Supported learning rate schedule types.
- signature: `class LearningRateScheduleType(str, Enum):`
- members:
  - `COSINE` — [`L147`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L147)
  - `WSD` — [`L148`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L148)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`lr_schedule_type`](types.md#Optimizer.lr_schedule_type)

### `Llama4Attention`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:680`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L680)
- doc: Configuration specific to Llama4-style models.
- signature: `class Llama4Attention(BaseModel):`
- members:
  - `temperature_tuning` — [`L687`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L687)
  - `use_qk_norm` — [`L683`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L683)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `LoRA`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1363`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1363)
- doc: Configuration for LoRA / QLoRA adapters.
- signature: `class LoRA(BaseModel):`
- members:
  - `enable_lora` — [`L1368`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1368)
  - `lora_alpha` — [`L1370`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1370)
  - `lora_module_path` — [`L1371`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1371)
  - `lora_rank` — [`L1369`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1369)
  - `lora_restore_path` — [`L1387`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1387)
  - `lora_tile_size` — [`L1383`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1383)
  - `lora_weight_qtype` — [`L1379`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1379)
  - `model_config` — [`L1366`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1366)
- uses (calls/refs, reference-scoped): [`PathStr`](types.md#PathStr)
- used by: [`lora`](types.md#MaxTextConfig.lora)

### `Logits`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:535`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L535)
- doc: Configuration for the final logits computation.
- signature: `class Logits(BaseModel):`
- members:
  - `cast_logits_to_fp32` — [`L544`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L544)
  - `final_logits_soft_cap` — [`L545`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L545)
  - `logits_dot_in_fp32` — [`L543`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L543)
  - `logits_via_embedding` — [`L538`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L538)
  - `normalize_embedding_logits` — [`L539`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L539)
  - `z_loss_multiplier` — [`L549`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L549)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `MTP`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:521`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L521)
- doc: Multi-Token Prediction Configs.
- signature: `class MTP(BaseModel):`
- members:
  - `mtp_eval_target_module` — [`L529`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L529)
  - `mtp_loss_scaling_factor` — [`L525`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L525)
  - `mtp_num_layers` — [`L524`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L524)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `ManagedMLDiagnostics`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1884`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1884)
- doc: Configuration for managed mldiagnostics.
- signature: `class ManagedMLDiagnostics(BaseModel):`
- members:
  - `managed_mldiagnostics` — [`L1887`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1887)
  - `managed_mldiagnostics_on_demand_profiling` — [`L1888`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1888)
  - `managed_mldiagnostics_region` — [`L1893`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1893)
  - `managed_mldiagnostics_run_group` — [`L1892`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1892)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `ManifoldConstrainedHyperConnections`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1506`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1506)
- doc: Configuration for DeepSeek Manifold-Constrained Hyper Connections (mHC).
- signature: `class ManifoldConstrainedHyperConnections(BaseModel):`
- members:
  - `enable_mhc_lite` — [`L1511`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1511)
  - `mhc_expansion_rate` — [`L1509`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1509)
  - `sinkhorn_iterations` — [`L1510`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1510)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `MatmulPrecision`  ·  implements/extends Enum, str
- def: [`src/maxtext/configs/types.py:71`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L71)
- doc: Precision levels for matrix multiplications.
- signature: `class MatmulPrecision(str, Enum):`
- members:
  - `BFLOAT16` — [`L78`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L78)
  - `DEFAULT` — [`L74`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L74)
  - `FLOAT32` — [`L80`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L80)
  - `HIGH` — [`L75`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L75)
  - `HIGHEST` — [`L76`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L76)
- used by: [`matmul_precision`](types.md#DataTypes.matmul_precision)

### `MaxTextConfig`  ·  implements/extends AOT, AdamW, Attention, AttentionIndexer, AudioEncoder, Checkpointing, CompressedAttention, DataTypes, DatasetGeneral, DcnParallelism, Decoding, DeepSeekMoE, DerivedValues, DevelopmentAndDebugging, DilocoParams, Distillation, ElasticTraining, EmergencyCheckpointing, Engram, FineTuning, GcpMonitoring, Goodput, GrainDataset, HardwareAndMesh, HfDataset, HloDump, IciParallelism, InferenceBenchmark, InferenceGeneral, InferenceLayout, InferenceServer, LayoutAndSharding, Llama4Attention, Logits, MTP, ManagedMLDiagnostics, ManifoldConstrainedHyperConnections, Metrics, MlaAttention, MoBa, MoEGeneral, MoEKernels, ModelArchitecture, MultimodalGeneral, Muon, OlmoGrainDataset, Optimizer, OrbaxStorage, PipelineParallelism, PositionalEmbedding, PrefixCaching, Profiling, Quantization, Qwen3Next, RL, RLDataset, RLEvaluation, RLHardware, RematAndOffload, Reward, Rope, RunInfo, SpecialTokens, SplashAttention, Tensorboard, TfdsDataset, Tokenizer, TrainingLoop, VLLM, VisionProjector, VisionTower, YarnRope
- def: [`src/maxtext/configs/types.py:2409`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2409) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
- doc: The main configuration object for MaxText.
- signature: `class MaxTextConfig(RunInfo, Checkpointing, OrbaxStorage, EmergencyCheckpointing, ElasticTraining, DataTypes, Quantization, ModelArchitecture, Engram, MTP, Logits, Attention, MlaAttention, CompressedAttention, MoBa, AttentionIndexer, Llama4Attention, SplashAttention, MoEGeneral, MoEKernels, DeepSeekMoE, Qwen3Next, HardwareAndMesh, LayoutAndSharding, DcnParallelism, IciParallelism, PipelineParallelism, RematAndOffload, TrainingLoop, ManifoldConstrainedHyperConnections, DilocoParams, Optimizer, AdamW, Muon, FineTuning, Distillation, RLHardware, VLLM, RL, RLDataset, RLEvaluation, Reward, SpecialTokens, PositionalEmbedding, Rope, YarnRope, DatasetGeneral, TfdsDataset, HfDataset, GrainDataset, OlmoGrainDataset, Tokenizer, InferenceGeneral, Decoding, InferenceLayout, InferenceServer, InferenceBenchmark, PrefixCaching, AOT, DevelopmentAndDebugging, Profiling, HloDump, Metrics, Goodput, GcpMonitoring, Tensorboard, ManagedMLDiagnostics, MultimodalGeneral, VisionTower, VisionProjector, AudioEncoder, DerivedValues):`
- members:
  - `_validate_check_vma_is_supported(self)` — [`L2527`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2527) — Validates that check_vma is used with supported settings. — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `calculate_global_batch_sizes(per_device_batch_size, expansion_factor, num_devices, grad_accum_steps)` — [`L2808`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2808) — Helper to calculate global and micro batch sizes for training and loading.
  - `get_num_target_devices()` — [`L2760`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2760) — Get the number of devices for the target topology, handling AOT compilation and single-controller modes. — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `load_model_specific_defaults(cls, values: dict[str, Any])` — [`L2523`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2523) — This method is a no-op because `pyconfig` handles model-specific config loading.
  - `set_derived_and_validate_values(self)` — [`L2559`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2559) — Computes all derived values and runs all cross-field validations after initial parsing. — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `validate_and_set_hlo_dump_defaults()` — [`L2682`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2682) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `validate_ragged_buffer_factor(self)` — [`L2547`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2547) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `debug` — [`L2506`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2506)
  - `dpo` — [`L2507`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2507) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `lora` — [`L2515`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2515)
  - `model_config` — [`L2519`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2519)
  - `rl` — [`L2511`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2511)
- uses (calls/refs, reference-scoped): [`quantization`](types.md#Quantization.quantization), [`decoder_block`](types.md#ModelArchitecture.decoder_block), [`model_name`](types.md#RunInfo.model_name), [`mesh_axes`](types.md#HardwareAndMesh.mesh_axes), [`num_pipeline_microbatches`](types.md#PipelineParallelism.num_pipeline_microbatches), [`dataset_type`](types.md#DatasetGeneral.dataset_type), [`data_sharding`](types.md#LayoutAndSharding.data_sharding), [`distill_beta_end`](types.md#Distillation.distill_beta_end), [`pipeline_parallel_layers`](types.md#PipelineParallelism.pipeline_parallel_layers), [`run_name`](types.md#RunInfo.run_name), [`base_output_directory`](types.md#RunInfo.base_output_directory), [`dump_hlo_gcs_dir`](types.md#HloDump.dump_hlo_gcs_dir), [`dump_jaxpr_gcs_dir`](types.md#HloDump.dump_jaxpr_gcs_dir), [`dump_hlo_xla_flags`](types.md#HloDump.dump_hlo_xla_flags), [`kv_quant_axis`](types.md#Quantization.kv_quant_axis), [`lr_schedule_type`](types.md#Optimizer.lr_schedule_type), [`max_target_length`](types.md#InferenceGeneral.max_target_length), [`num_slices`](types.md#HardwareAndMesh.num_slices), [`num_target_devices`](types.md#DerivedValues.num_target_devices), [`opt_type`](types.md#Optimizer.opt_type), [`remat_policy`](types.md#RematAndOffload.remat_policy), [`shard_mode`](types.md#HardwareAndMesh.shard_mode), [`tokenizer_type`](types.md#Tokenizer.tokenizer_type), [`weight_dtype`](types.md#DataTypes.weight_dtype), [`RL`](types.md#RL), [`DatasetType`](types.md#DatasetType), [`attention`](types.md#Attention.attention), [`attention_type`](types.md#Attention.attention_type), [`base_mlp_dim`](types.md#ModelArchitecture.base_mlp_dim), [`context_sharding`](types.md#LayoutAndSharding.context_sharding), [`dcn_diloco_parallelism`](types.md#DcnParallelism.dcn_diloco_parallelism), [`enable_single_controller`](types.md#DevelopmentAndDebugging.enable_single_controller), [`hardware`](types.md#HardwareAndMesh.hardware), [`logger`](types.md#logger), [`num_layers_per_pipeline_stage`](types.md#PipelineParallelism.num_layers_per_pipeline_stage), [`num_pipeline_repeats`](types.md#PipelineParallelism.num_pipeline_repeats), [`packing`](types.md#DatasetGeneral.packing), [`IciParallelism`](types.md#IciParallelism), [`MultimodalGeneral`](types.md#MultimodalGeneral), [`grain_train_files`](types.md#GrainDataset.grain_train_files)  (+291 more)
- used by: [`RL`](types.md#RL), [`IciParallelism`](types.md#IciParallelism), [`MultimodalGeneral`](types.md#MultimodalGeneral), [`AOT`](types.md#AOT), [`AdamW`](types.md#AdamW), [`Attention`](types.md#Attention), [`AttentionIndexer`](types.md#AttentionIndexer), [`AudioEncoder`](types.md#AudioEncoder), [`Checkpointing`](types.md#Checkpointing), [`CompressedAttention`](types.md#CompressedAttention), [`DataTypes`](types.md#DataTypes), [`DatasetGeneral`](types.md#DatasetGeneral), [`DcnParallelism`](types.md#DcnParallelism), [`Decoding`](types.md#Decoding), [`DeepSeekMoE`](types.md#DeepSeekMoE), [`DerivedValues`](types.md#DerivedValues), [`DevelopmentAndDebugging`](types.md#DevelopmentAndDebugging), [`DilocoParams`](types.md#DilocoParams), [`Distillation`](types.md#Distillation), [`ElasticTraining`](types.md#ElasticTraining), [`EmergencyCheckpointing`](types.md#EmergencyCheckpointing), [`Engram`](types.md#Engram), [`FineTuning`](types.md#FineTuning), [`GcpMonitoring`](types.md#GcpMonitoring), [`Goodput`](types.md#Goodput), [`GrainDataset`](types.md#GrainDataset), [`HardwareAndMesh`](types.md#HardwareAndMesh), [`HfDataset`](types.md#HfDataset), [`HloDump`](types.md#HloDump), [`InferenceBenchmark`](types.md#InferenceBenchmark), [`InferenceGeneral`](types.md#InferenceGeneral), [`InferenceLayout`](types.md#InferenceLayout), [`InferenceServer`](types.md#InferenceServer), [`LayoutAndSharding`](types.md#LayoutAndSharding), [`Llama4Attention`](types.md#Llama4Attention), [`Logits`](types.md#Logits), [`MTP`](types.md#MTP), [`ManagedMLDiagnostics`](types.md#ManagedMLDiagnostics), [`ManifoldConstrainedHyperConnections`](types.md#ManifoldConstrainedHyperConnections), [`Metrics`](types.md#Metrics)  (+32 more)

### `Metrics`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1869`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1869)
- doc: General configuration for metrics and monitoring.
- signature: `class Metrics(BaseModel):`
- members:
  - `enable_checkpoint_cloud_logger` — [`L1877`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1877)
  - `enable_tunix_perf_metrics` — [`L1878`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1878)
  - `gcs_metrics` — [`L1873`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1873)
  - `metrics_file` — [`L1872`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1872)
  - `prometheus_port` — [`L1876`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1876)
  - `record_internal_nn_metrics` — [`L1875`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1875)
  - `save_config_to_gcs` — [`L1874`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1874)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`PathStr`](types.md#PathStr)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `MlaAttention`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:640`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L640)
- doc: Configuration for Multi-Layer Attention (MLA).
- signature: `class MlaAttention(BaseModel):`
- members:
  - `kv_lora_rank` — [`L645`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L645)
  - `mla_naive_kvcache` — [`L643`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L643)
  - `q_lora_rank` — [`L644`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L644)
  - `qk_nope_head_dim` — [`L646`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L646)
  - `qk_rope_head_dim` — [`L647`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L647)
  - `v_head_dim` — [`L648`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L648)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `MoBa`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:632`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L632)
- doc: Configuration for Mixture of Block Attention (MoBA).
- signature: `class MoBa(BaseModel):`
- members:
  - `moba` — [`L635`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L635)
  - `moba_chunk_size` — [`L636`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L636)
  - `moba_topk` — [`L637`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L637)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `MoEGeneral`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:751`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L751)
- doc: General configuration for Mixture of Experts (MoE) layers.
- signature: `class MoEGeneral(BaseModel):`
- members:
  - `base_moe_mlp_dim` — [`L762`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L762)
  - `capacity_factor` — [`L756`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L756)
  - `float32_gate_logits` — [`L842`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L842)
  - `float32_weight_sum` — [`L838`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L838)
  - `fuse_expert_scales` — [`L851`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L851)
  - `interleave_moe_layer_step` — [`L820`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L820)
  - `load_balance_loss_weight` — [`L766`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L766)
  - `moe_dispatch_no_expert_sharding` — [`L775`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L775)
  - `moe_expert_input_dim` — [`L758`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L758)
  - `moe_fsdp_use_two_stage_all_gather` — [`L821`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L821)
  - `norm_topk_prob` — [`L834`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L834)
  - `num_experts` — [`L754`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L754)
  - `num_experts_per_tok` — [`L755`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L755)
  - `padded_base_moe_mlp_dim` — [`L763`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L763)
  - `prefuse_moe_weights` — [`L846`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L846)
  - `ragged_buffer_factor` — [`L757`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L757)
  - `ragged_gather_cost_estimate_bytes_accessed` — [`L809`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L809)
  - `ragged_gather_cost_estimate_flops` — [`L799`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L799)
  - `ragged_gather_fallback` — [`L789`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L789)
  - `ragged_gather_reduce_cost_estimate_bytes_accessed` — [`L814`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L814)
  - `ragged_gather_reduce_cost_estimate_flops` — [`L804`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L804)
  - `ragged_gather_reduce_fallback` — [`L794`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L794)
  - `shard_exp_on_fsdp` — [`L825`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L825)
  - `use_2d_fsdp_sharding` — [`L830`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L830)
  - `use_custom_sort_vjp` — [`L767`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L767)
  - `use_gather_mosaic_kernel` — [`L785`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L785)
  - `use_ragged_sort` — [`L782`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L782)
  - `use_random_routing` — [`L819`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L819)
  - `use_ring_of_experts` — [`L771`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L771)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig), [`_validate_check_vma_is_supported`](types.md#MaxTextConfig._validate_check_vma_is_supported), [`validate_ragged_buffer_factor`](types.md#MaxTextConfig.validate_ragged_buffer_factor)

### `MoEKernels`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:858`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L858)
- doc: Configuration for MoE-specific kernels like Megablox.
- signature: `class MoEKernels(BaseModel):`
- members:
  - `megablox` — [`L861`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L861)
  - `merge_gating_gmm` — [`L900`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L900)
  - `sparse_matmul` — [`L862`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L862)
  - `wi_tile_dlhs_batch_seq` — [`L869`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L869)
  - `wi_tile_dlhs_embed_dim` — [`L873`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L873)
  - `wi_tile_dlhs_mlp_dim` — [`L874`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L874)
  - `wi_tile_drhs_batch_seq` — [`L875`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L875)
  - `wi_tile_drhs_embed_dim` — [`L879`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L879)
  - `wi_tile_drhs_mlp_dim` — [`L880`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L880)
  - `wi_tile_fwd_batch_seq` — [`L863`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L863)
  - `wi_tile_fwd_embed_dim` — [`L867`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L867)
  - `wi_tile_fwd_mlp_dim` — [`L868`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L868)
  - `wo_tile_dlhs_batch_seq` — [`L887`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L887)
  - `wo_tile_dlhs_embed_dim` — [`L891`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L891)
  - `wo_tile_dlhs_mlp_dim` — [`L892`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L892)
  - `wo_tile_drhs_batch_seq` — [`L893`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L893)
  - `wo_tile_drhs_embed_dim` — [`L897`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L897)
  - `wo_tile_drhs_mlp_dim` — [`L898`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L898)
  - `wo_tile_fwd_batch_seq` — [`L881`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L881)
  - `wo_tile_fwd_embed_dim` — [`L885`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L885)
  - `wo_tile_fwd_mlp_dim` — [`L886`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L886)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `ModelArchitecture`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:475`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L475)
- doc: Core model architecture parameters.
- signature: `class ModelArchitecture(BaseModel):`
- members:
  - `attention_bias` — [`L509`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L509)
  - `attention_output_dim` — [`L493`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L493)
  - `base_emb_dim` — [`L483`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L483)
  - `base_mlp_dim` — [`L486`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L486) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `base_num_decoder_layers` — [`L488`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L488)
  - `base_num_kv_heads` — [`L485`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L485)
  - `base_num_query_heads` — [`L484`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L484)
  - `decoder_block` — [`L478`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L478) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `dense_init_scale` — [`L487`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L487)
  - `fused_mlp` — [`L513`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L513)
  - `fused_qkv` — [`L508`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L508)
  - `global_head_dim` — [`L497`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L497)
  - `global_parameter_scale` — [`L482`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L482)
  - `head_dim` — [`L489`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L489)
  - `mlp_activations` — [`L501`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L501)
  - `mlp_activations_limit` — [`L502`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L502)
  - `normalization_layer_epsilon` — [`L507`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L507)
  - `qk_norm_with_scale` — [`L514`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L514)
  - `v_norm_with_scale` — [`L518`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L518)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `MultimodalGeneral`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1954`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1954)
- doc: General configuration for Multimodal models.
- signature: `class MultimodalGeneral(BaseModel):`
- members:
  - `validate_video_max_grid(self)` — [`L1992`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1992)
  - `attention_for_vit` — [`L1958`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1958)
  - `audio_path` — [`L1971`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1971)
  - `audio_placeholder` — [`L1973`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1973)
  - `freeze_audio_encoder_params` — [`L1960`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1960)
  - `freeze_vision_encoder_params` — [`L1959`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1959)
  - `image_path` — [`L1963`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1963)
  - `image_placeholder` — [`L1964`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1964)
  - `image_size_for_vit` — [`L1962`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1962)
  - `max_num_images_per_example` — [`L1966`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1966)
  - `mrope_section` — [`L1976`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1976)
  - `posemb_type_for_vit` — [`L1965`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1965)
  - `position_id_per_seconds` — [`L1977`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1977)
  - `use_audio` — [`L1961`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1961)
  - `use_audio_in_video` — [`L1974`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1974)
  - `use_mrope` — [`L1975`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1975)
  - `use_multimodal` — [`L1957`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1957)
  - `video_max_grid_h` — [`L1982`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1982)
  - `video_max_grid_t` — [`L1978`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1978)
  - `video_max_grid_w` — [`L1986`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1986)
  - `video_path` — [`L1970`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1970)
  - `video_placeholder` — [`L1972`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1972)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`PathStr`](types.md#PathStr)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `Muon`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1627`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1627)
- doc: Configuration specific to the Muon optimizer.
- signature: `class Muon(BaseModel):`
- members:
  - `muon_beta` — [`L1630`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1630)
  - `muon_consistent_rms` — [`L1635`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1635)
  - `muon_weight_decay` — [`L1631`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1631)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `OlmoGrainDataset`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1304`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1304)
- doc: Configuration for the OLMo numpy fixed-seq-length input pipeline (dataset_type=olmo_grain).
- signature: `class OlmoGrainDataset(BaseModel):`
- members:
  - `olmo_apply_ngram_filter` — [`L1327`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1327)
  - `olmo_index_path` — [`L1318`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1318)
  - `olmo_path_remap_from` — [`L1319`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1319)
  - `olmo_path_remap_to` — [`L1323`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1323)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`PathStr`](types.md#PathStr)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `Optimizer`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1538`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1538)
- doc: Configuration for the optimizer and learning rate schedule.
- signature: `class Optimizer(BaseModel):`
- members:
  - `gradient_accumulation_steps` — [`L1551`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1551)
  - `gradient_clipping_threshold` — [`L1558`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1558)
  - `learning_rate` — [`L1561`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1561)
  - `learning_rate_final_fraction` — [`L1566`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1566)
  - `learning_rate_schedule_steps` — [`L1581`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1581)
  - `lr_schedule_type` — [`L1562`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1562) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `opt_type` — [`L1541`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1541) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `skip_step_interval` — [`L1546`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1546)
  - `skip_step_on_spikes` — [`L1542`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1542)
  - `skip_step_scaling_factor` — [`L1550`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1550)
  - `trainable_parameters_mask` — [`L1586`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1586)
  - `use_tunix_gradient_accumulation` — [`L1554`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1554)
  - `warmup_steps_fraction` — [`L1580`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1580)
  - `wsd_decay_steps_fraction` — [`L1570`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1570)
  - `wsd_decay_style` — [`L1576`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1576)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`LearningRateScheduleType`](types.md#LearningRateScheduleType), [`OptimizerType`](types.md#OptimizerType), [`WsdDecayStyle`](types.md#WsdDecayStyle), [`ADAMW`](types.md#OptimizerType.ADAMW), [`COSINE`](types.md#LearningRateScheduleType.COSINE), [`LINEAR`](types.md#WsdDecayStyle.LINEAR)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `OptimizerType`  ·  implements/extends Enum, str
- def: [`src/maxtext/configs/types.py:135`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L135)
- doc: Supported optimizer algorithms.
- signature: `class OptimizerType(str, Enum):`
- members:
  - `ADAMW` — [`L138`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L138)
  - `ADAM_PAX` — [`L139`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L139)
  - `MUON` — [`L141`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L141)
  - `SGD` — [`L140`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L140)
- used by: [`opt_type`](types.md#Optimizer.opt_type)

### `OrbaxStorage`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:368`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L368)
- doc: Configuration for Orbax checkpoint storage options.
- signature: `class OrbaxStorage(BaseModel):`
- members:
  - `checkpoint_storage_concurrent_gb` — [`L378`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L378)
  - `checkpoint_storage_target_data_file_size_bytes` — [`L371`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L371)
  - `checkpoint_storage_use_ocdbt` — [`L374`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L374)
  - `checkpoint_storage_use_zarr3` — [`L375`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L375)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `PipelineParallelism`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1066`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1066)
- doc: Configuration for pipeline parallelism.
- signature: `class PipelineParallelism(BaseModel):`
- members:
  - `num_layers_per_pipeline_stage` — [`L1072`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1072) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `num_pipeline_microbatches` — [`L1078`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1078) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `num_pipeline_repeats` — [`L1073`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1073) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `pipeline_delay_activation_forwarding` — [`L1082`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1082)
  - `pipeline_fsdp_ag_once` — [`L1085`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1085)
  - `pipeline_fsdp_ag_per_repeat` — [`L1069`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1069)
  - `pipeline_parallel_layers` — [`L1077`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1077) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `scan_layers_per_stage` — [`L1088`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1088)
  - `scan_pipeline_iterations` — [`L1086`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1086)
  - `scan_pipeline_repeats` — [`L1087`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1087)
  - `set_remat_policy_on_layers_per_stage` — [`L1090`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1090)
  - `set_remat_policy_on_pipeline_iterations` — [`L1089`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1089)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `PositionalEmbedding`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1641`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1641)
- doc: General configuration for positional embeddings.
- signature: `class PositionalEmbedding(BaseModel):`
- members:
  - `nope_layer_interval` — [`L1655`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1655)
  - `trainable_position_size` — [`L1651`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1651)
  - `use_iota_embed` — [`L1644`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1644)
  - `use_untrainable_positional_embedding` — [`L1648`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1648)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `PrefixCaching`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1759`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1759)
- doc: Configuration for Prefix Caching in JetStream.
- signature: `class PrefixCaching(BaseModel):`
- members:
  - `enable_prefix_caching` — [`L1762`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1762)
  - `prefix_caching_dram_byte` — [`L1764`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1764)
  - `prefix_caching_hbm_byte` — [`L1763`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1763)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `ProfilerType`  ·  implements/extends Enum, str
- def: [`src/maxtext/configs/types.py:195`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L195)
- doc: Supported performance profilers.
- signature: `class ProfilerType(str, Enum):`
- members:
  - `NONE` — [`L198`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L198)
  - `NSYS` — [`L200`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L200)
  - `XPLANE` — [`L199`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L199)
- used by: [`profiler`](types.md#Profiling.profiler)

### `Profiling`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1810`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1810)
- doc: Configuration for performance profiling.
- signature: `class Profiling(BaseModel):`
- members:
  - `enable_jax_profiler` — [`L1820`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1820)
  - `enable_tpu_profiling_options` — [`L1822`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1822)
  - `hide_profiler_step_metric` — [`L1819`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1819)
  - `jax_profiler_port` — [`L1821`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1821)
  - `profile_cleanly` — [`L1817`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1817)
  - `profile_periodically_period` — [`L1818`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1818)
  - `profile_power_events` — [`L1838`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1838)
  - `profiler` — [`L1813`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1813)
  - `profiler_steps` — [`L1816`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1816)
  - `skip_first_n_steps_for_profiler` — [`L1815`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1815)
  - `tpu_num_chips_to_profile_per_task` — [`L1823`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1823)
  - `tpu_num_sparse_core_tiles_to_trace` — [`L1825`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1825)
  - `tpu_num_sparse_cores_to_trace` — [`L1824`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1824)
  - `upload_all_profiler_results` — [`L1814`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1814)
  - `xprof_e2e_enable_fw_power_level_event` — [`L1836`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1836)
  - `xprof_e2e_enable_fw_thermal_event` — [`L1837`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1837)
  - `xprof_e2e_enable_fw_throttle_event` — [`L1835`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1835)
  - `xprof_tpu_power_trace_level` — [`L1828`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1828)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`ProfilerType`](types.md#ProfilerType), [`XProfTPUPowerTraceMode`](types.md#XProfTPUPowerTraceMode), [`NONE`](types.md#ProfilerType.NONE), [`POWER_TRACE_NONE`](types.md#XProfTPUPowerTraceMode.POWER_TRACE_NONE)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `Quantization`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:427`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L427)
- doc: Configuration for model quantization.
- signature: `class Quantization(BaseModel):`
- members:
  - `act_quantization_calibration_method` — [`L452`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L452)
  - `bwd_quantization_calibration_method` — [`L456`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L456)
  - `kv_quant_axis` — [`L440`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L440) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `kv_quant_dtype` — [`L441`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L441)
  - `quant_cfg_path` — [`L438`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L438)
  - `quantization` — [`L430`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L430) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `quantization_local_shard_count` — [`L442`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L442)
  - `quantize_kvcache` — [`L439`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L439)
  - `replicate_quant_scale` — [`L434`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L434)
  - `use_manual_quantization` — [`L444`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L444)
  - `use_qwix_quantization` — [`L443`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L443)
  - `weight_quantization_calibration_method` — [`L448`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L448)
  - `weight_sparsity_m` — [`L464`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L464)
  - `weight_sparsity_n` — [`L460`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L460)
  - `weight_sparsity_start_step` — [`L469`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L469)
  - `weight_sparsity_update_step` — [`L468`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L468)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`PathStr`](types.md#PathStr), [`KvQuantAxis`](types.md#KvQuantAxis), [`QuantizationType`](types.md#QuantizationType), [`HEADS_AND_DKV`](types.md#KvQuantAxis.HEADS_AND_DKV), [`NONE`](types.md#QuantizationType.NONE)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `QuantizationType`  ·  implements/extends Enum, str
- def: [`src/maxtext/configs/types.py:83`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L83)
- doc: Supported quantization schemes.
- signature: `class QuantizationType(str, Enum):`
- members:
  - `FP8` — [`L92`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L92)
  - `FP8_E4M3` — [`L91`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L91)
  - `FP8_E5M2` — [`L90`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L90)
  - `FP8_FULL` — [`L96`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L96)
  - `FP8_GPU` — [`L95`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L95)
  - `FP8_NANO_V2` — [`L94`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L94)
  - `INT4` — [`L87`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L87)
  - `INT8` — [`L88`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L88)
  - `INTMP` — [`L89`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L89)
  - `NANOO_FP8` — [`L93`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L93)
  - `NONE` — [`L86`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L86)
  - `TE_FP8_CS` — [`L98`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L98)
  - `TE_FP8_DS` — [`L97`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L97)
  - `TE_MXFP8` — [`L99`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L99)
  - `TE_NVFP4` — [`L100`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L100)
  - `TE_NVFP4_NO_RHT` — [`L101`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L101)
- used by: [`quantization`](types.md#Quantization.quantization)

### `Qwen3Next`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:932`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L932)
- doc: Configuration specific to Qwen3-Next models with Gated Delta Net.
- signature: `class Qwen3Next(BaseModel):`
- members:
  - `gdn_chunk_size` — [`L940`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L940)
  - `gdn_conv_kernel_dim` — [`L935`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L935)
  - `gdn_key_head_dim` — [`L936`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L936)
  - `gdn_num_key_heads` — [`L938`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L938)
  - `gdn_num_value_heads` — [`L939`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L939)
  - `gdn_value_head_dim` — [`L937`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L937)
  - `partial_rotary_factor` — [`L948`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L948)
  - `use_qk_norm_in_gdn` — [`L944`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L944)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `RL`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:2131`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2131)
- doc: Configuration for RL algorithms like Group Relative Policy Optimization (GRPO) among others.
- signature: `class RL(BaseModel):`
- members:
  - `degenerate_group_masking` — [`L2159`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2159)
  - `epsilon_high` — [`L2163`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2163)
  - `grpo_beta` — [`L2136`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2136)
  - `grpo_epsilon` — [`L2137`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2137)
  - `loss_agg_mode` — [`L2139`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2139)
  - `loss_algo` — [`L2138`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2138)
  - `max_concurrency` — [`L2147`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2147)
  - `num_generations` — [`L2134`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2134)
  - `num_iterations` — [`L2135`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2135)
  - `off_policy_steps` — [`L2151`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2151)
  - `reshard_chunk_size` — [`L2167`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2167)
  - `system_prompt` — [`L2155`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2155)
  - `use_agentic_rollout` — [`L2143`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2143)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig), [`rl`](types.md#MaxTextConfig.rl)

### `RLDataset`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:2176`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2176)
- doc: Dataset settings for RL training.
- signature: `class RLDataset(BaseModel):`
- members:
  - `batch_size` — [`L2179`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2179)
  - `dataset_processor_path` — [`L2187`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2187)
  - `eval_batch_size` — [`L2182`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2182)
  - `num_batches` — [`L2180`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2180)
  - `num_test_batches` — [`L2181`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2181)
  - `rollout_micro_batch_size` — [`L2186`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2186)
  - `test_batch_start_index` — [`L2183`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2183)
  - `train_fraction` — [`L2184`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2184)
  - `train_micro_batch_size` — [`L2185`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2185)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `RLEvaluation`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:2196`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2196)
- doc: Settings for RL evaluation.
- signature: `class RLEvaluation(BaseModel):`
- members:
  - `eval_corr_lst` — [`L2205`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2205)
  - `eval_make_lst` — [`L2209`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2209)
  - `eval_mode` — [`L2213`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2213)
  - `eval_sampling_strategy` — [`L2199`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2199)
  - `generation_configs` — [`L2200`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2200)
  - `num_eval_passes` — [`L2204`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2204)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `RLHardware`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:2067`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2067)
- doc: Hardware settings specific to RL training.
- signature: `class RLHardware(BaseModel):`
- members:
  - `chips_per_vm` — [`L2072`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2072)
  - `inference_devices_per_replica` — [`L2087`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2087)
  - `inference_replicas` — [`L2086`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2086)
  - `inference_rollouts` — [`L2090`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2090)
  - `num_samplers_slices` — [`L2075`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2075)
  - `num_trainer_slices` — [`L2074`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2074)
  - `rollout_data_parallelism` — [`L2076`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2076)
  - `rollout_expert_parallelism` — [`L2085`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2085)
  - `rollout_tensor_parallelism` — [`L2081`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2081)
  - `sampler_devices_fraction` — [`L2071`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2071)
  - `trainer_devices_fraction` — [`L2070`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2070)
  - `use_pathways` — [`L2073`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2073)
  - `use_pathways_reshard` — [`L2093`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2093)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `RematAndOffload`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1093`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1093)
- doc: Configuration for gradient checkpointing (rematerialization) and offloading.
- signature: `class RematAndOffload(BaseModel):`
- members:
  - `attention_out` — [`L1157`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1157)
  - `context` — [`L1104`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1104) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `decoder_layer_input` — [`L1101`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1101) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `engram` — [`L1161`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1161) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `key_proj` — [`L1134`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1134)
  - `kv_wa_proj` — [`L1140`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1140)
  - `mla_kv` — [`L1153`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1153) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `mla_q` — [`L1149`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1149)
  - `mlpwi` — [`L1105`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1105) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `mlpwi_0` — [`L1109`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1109)
  - `mlpwi_1` — [`L1113`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1113)
  - `mlpwo` — [`L1117`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1117)
  - `moe_mlpwi_0` — [`L1121`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1121) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `moe_mlpwi_1` — [`L1125`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1125)
  - `moe_mlpwo` — [`L1129`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1129)
  - `optimizer_memory_host_offload` — [`L1163`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1163)
  - `out_proj` — [`L1145`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1145)
  - `parameter_memory_host_offload` — [`L1164`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1164)
  - `qkv_proj` — [`L1144`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1144) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `query_proj` — [`L1133`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1133)
  - `query_wa_proj` — [`L1136`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1136)
  - `remat_policy` — [`L1096`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1096) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `remat_policy_for_vit` — [`L1100`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1100)
  - `value_proj` — [`L1135`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1135)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`RematLocation`](types.md#RematLocation), [`REMAT`](types.md#RematLocation.REMAT), [`RematPolicy`](types.md#RematPolicy), [`DEVICE`](types.md#RematLocation.DEVICE), [`FULL`](types.md#RematPolicy.FULL)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `RematLocation`  ·  implements/extends Enum, str
- def: [`src/maxtext/configs/types.py:127`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L127) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
- doc: Specifies where to store activations for rematerialization.
- signature: `class RematLocation(str, Enum):`
- members:
  - `DEVICE` — [`L131`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L131)
  - `OFFLOAD` — [`L132`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L132)
  - `REMAT` — [`L130`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L130)
- used by: [`attention_out`](types.md#RematAndOffload.attention_out), [`context`](types.md#RematAndOffload.context), [`decoder_layer_input`](types.md#RematAndOffload.decoder_layer_input), [`engram`](types.md#RematAndOffload.engram), [`key_proj`](types.md#RematAndOffload.key_proj), [`kv_wa_proj`](types.md#RematAndOffload.kv_wa_proj), [`mla_kv`](types.md#RematAndOffload.mla_kv), [`mla_q`](types.md#RematAndOffload.mla_q), [`mlpwi`](types.md#RematAndOffload.mlpwi), [`mlpwi_0`](types.md#RematAndOffload.mlpwi_0), [`mlpwi_1`](types.md#RematAndOffload.mlpwi_1), [`mlpwo`](types.md#RematAndOffload.mlpwo), [`moe_mlpwi_0`](types.md#RematAndOffload.moe_mlpwi_0), [`moe_mlpwi_1`](types.md#RematAndOffload.moe_mlpwi_1), [`moe_mlpwo`](types.md#RematAndOffload.moe_mlpwo), [`out_proj`](types.md#RematAndOffload.out_proj), [`qkv_proj`](types.md#RematAndOffload.qkv_proj), [`query_proj`](types.md#RematAndOffload.query_proj), [`query_wa_proj`](types.md#RematAndOffload.query_wa_proj), [`value_proj`](types.md#RematAndOffload.value_proj)

### `RematPolicy`  ·  implements/extends Enum, str
- def: [`src/maxtext/configs/types.py:112`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L112)
- doc: Available rematerialization (gradient checkpointing) policies.
- signature: `class RematPolicy(str, Enum):`
- members:
  - `CUSTOM` — [`L122`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L122)
  - `FULL` — [`L115`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L115)
  - `MINIMAL` — [`L116`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L116)
  - `MINIMAL_OFFLOADED` — [`L123`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L123)
  - `QKV_PROJ_OFFLOADED` — [`L121`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L121)
  - `SAVE_DOT_EXCEPT_MLP` — [`L119`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L119)
  - `SAVE_DOT_EXCEPT_MLPWI` — [`L118`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L118)
  - `SAVE_DOT_WITH_CONTEXT_EXCEPT_MLP` — [`L117`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L117)
  - `SAVE_OUT_PROJ` — [`L124`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L124)
  - `SAVE_QKV_PROJ` — [`L120`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L120)
- used by: [`remat_policy`](types.md#RematAndOffload.remat_policy)

### `Reward`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:2219`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2219)
- doc: Configuration for the reward/penalty model in RL.
- signature: `class Reward(BaseModel):`
- members:
  - `math_verify_num_procs` — [`L2235`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2235)
  - `math_verify_timeout` — [`L2234`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2234)
  - `penalty_incorrect_answer` — [`L2233`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2233)
  - `penalty_incorrect_format` — [`L2232`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2232)
  - `reward_exact_answer` — [`L2222`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2222)
  - `reward_exact_format_match` — [`L2223`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2223)
  - `reward_partial_format_match` — [`L2225`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2225)
  - `reward_ratio_guess_to_answer_high` — [`L2226`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2226)
  - `reward_ratio_guess_to_answer_low` — [`L2229`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2229)
  - `reward_white_space_format_match` — [`L2224`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2224)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `Rope`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1658`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1658)
- doc: Configuration for Rotary Positional Embedding (RoPE).
- signature: `class Rope(BaseModel):`
- members:
  - `global_rope_max_timescale` — [`L1667`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1667)
  - `global_rope_proportion` — [`L1668`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1668)
  - `local_rope_max_timescale` — [`L1666`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1666)
  - `local_rope_proportion` — [`L1669`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1669)
  - `rope_linear_scaling_factor` — [`L1665`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1665)
  - `rope_max_timescale` — [`L1664`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1664)
  - `rope_min_timescale` — [`L1663`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1663)
  - `rope_type` — [`L1661`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1661)
  - `rope_use_scale` — [`L1662`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1662)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`RopeType`](types.md#RopeType), [`DEFAULT`](types.md#RopeType.DEFAULT)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `RopeType`  ·  implements/extends Enum, str
- def: [`src/maxtext/configs/types.py:158`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L158)
- doc: Supported Rotary Positional Embedding (RoPE) implementations.
- signature: `class RopeType(str, Enum):`
- members:
  - `DEFAULT` — [`L161`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L161)
  - `LLAMA3_1` — [`L162`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L162)
  - `YARN` — [`L163`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L163)
- used by: [`rope_type`](types.md#Rope.rope_type)

### `RunInfo`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:285`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L285)
- doc: Configuration for the overall run, model identity, and logging.
- signature: `class RunInfo(BaseModel):`
- members:
  - `base_config` — [`L288`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L288)
  - `base_output_directory` — [`L307`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L307) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `debug_sharding` — [`L306`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L306)
  - `log_config` — [`L302`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L302)
  - `model_name` — [`L296`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L296) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `override_logical_axis_rules` — [`L298`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L298)
  - `override_model_config` — [`L297`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L297)
  - `run_name` — [`L292`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L292) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `sharding_strategy` — [`L308`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L308)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`PathStr`](types.md#PathStr), [`ModelName`](types.md#ModelName)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig), [`validate_and_set_hlo_dump_defaults`](types.md#MaxTextConfig.validate_and_set_hlo_dump_defaults)

### `SamplingStrategy`  ·  implements/extends Enum, str
- def: [`src/maxtext/configs/types.py:185`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L185)
- doc: Supported decoding and sampling strategies.
- signature: `class SamplingStrategy(str, Enum):`
- members:
  - `COMPOSITE` — [`L192`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L192)
  - `GREEDY` — [`L188`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L188)
  - `NUCLEUS` — [`L190`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L190)
  - `TOPK` — [`L191`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L191)
  - `WEIGHTED` — [`L189`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L189)
- used by: [`decode_sampling_strategy`](types.md#Decoding.decode_sampling_strategy)

### `SpecialTokens`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:2241`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2241)
- doc: Special tokens used for formatting prompts and responses in RL.
- signature: `class SpecialTokens(BaseModel):`
- members:
  - `reasoning_end_token` — [`L2245`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2245)
  - `reasoning_start_token` — [`L2244`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2244)
  - `solution_end_token` — [`L2247`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2247)
  - `solution_start_token` — [`L2246`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2246)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `SplashAttention`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:693`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L693)
- doc: Tunable block sizes for Splash Attention kernels.
- signature: `class SplashAttention(BaseModel):`
- members:
  - `cost_estimate_flops_bwd` — [`L740`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L740)
  - `cost_estimate_flops_fwd` — [`L735`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L735)
  - `dq_reduction_steps` — [`L745`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L745)
  - `local_sa_block_kv` — [`L713`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L713)
  - `local_sa_block_kv_compute` — [`L714`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L714)
  - `local_sa_block_kv_dkv` — [`L716`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L716)
  - `local_sa_block_kv_dkv_compute` — [`L717`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L717)
  - `local_sa_block_kv_dq` — [`L721`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L721)
  - `local_sa_block_q` — [`L712`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L712)
  - `local_sa_block_q_dkv` — [`L715`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L715)
  - `local_sa_block_q_dq` — [`L720`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L720)
  - `local_sa_fuse_reciprocal` — [`L729`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L729)
  - `local_sa_k_layout` — [`L726`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L726)
  - `local_sa_q_layout` — [`L725`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L725)
  - `local_sa_use_base2_exp` — [`L730`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L730)
  - `local_sa_use_fused_bwd_kernel` — [`L722`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L722)
  - `local_sa_v_layout` — [`L727`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L727)
  - `local_use_splash_scheduler` — [`L728`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L728)
  - `sa_block_kv` — [`L697`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L697)
  - `sa_block_kv_compute` — [`L698`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L698)
  - `sa_block_kv_dkv` — [`L700`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L700)
  - `sa_block_kv_dkv_compute` — [`L701`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L701)
  - `sa_block_kv_dq` — [`L703`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L703)
  - `sa_block_q` — [`L696`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L696)
  - `sa_block_q_dkv` — [`L699`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L699)
  - `sa_block_q_dq` — [`L702`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L702)
  - `sa_fuse_reciprocal` — [`L709`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L709)
  - `sa_k_layout` — [`L706`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L706)
  - `sa_q_layout` — [`L705`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L705)
  - `sa_use_base2_exp` — [`L710`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L710)
  - `sa_use_fused_bwd_kernel` — [`L704`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L704)
  - `sa_v_layout` — [`L707`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L707)
  - `use_max_logit_estimate` — [`L731`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L731)
  - `use_splash_scheduler` — [`L708`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L708)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `Tensorboard`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1945`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1945)
- doc: Configuration for Tensorboard logging.
- signature: `class Tensorboard(BaseModel):`
- members:
  - `enable_tensorboard` — [`L1948`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1948)
  - `use_vertex_tensorboard` — [`L1949`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1949)
  - `vertex_tensorboard_project` — [`L1950`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1950)
  - `vertex_tensorboard_region` — [`L1951`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1951)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `TfdsDataset`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1244`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1244)
- doc: Configuration specific to TFDS datasets.
- signature: `class TfdsDataset(BaseModel):`
- members:
  - `dataset_name` — [`L1248`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1248)
  - `dataset_path` — [`L1247`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1247)
  - `eval_dataset_name` — [`L1249`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1249)
  - `eval_split` — [`L1251`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1251)
  - `train_split` — [`L1250`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1250)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`PathStr`](types.md#PathStr)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `Tokenizer`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1167`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1167)
- doc: Configuration for the tokenizer.
- signature: `class Tokenizer(BaseModel):`
- members:
  - `add_bos` — [`L1188`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1188)
  - `add_eos` — [`L1189`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1189)
  - `chat_template` — [`L1178`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1178)
  - `chat_template_path` — [`L1177`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1177)
  - `data_template_path` — [`L1182`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1182)
  - `num_vocab_tiling` — [`L1194`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1194)
  - `tokenize_eval_data` — [`L1187`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1187)
  - `tokenize_train_data` — [`L1186`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1186)
  - `tokenizer_path` — [`L1171`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1171)
  - `tokenizer_type` — [`L1175`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1175) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
  - `use_chat_template` — [`L1176`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1176)
  - `use_truncation` — [`L1190`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1190)
  - `vocab_size` — [`L1170`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1170)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig), [`PathStr`](types.md#PathStr), [`TokenizerType`](types.md#TokenizerType), [`SENTENCEPIECE`](types.md#TokenizerType.SENTENCEPIECE)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `TokenizerType`  ·  implements/extends Enum, str
- def: [`src/maxtext/configs/types.py:166`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L166)
- doc: Supported tokenizer libraries.
- signature: `class TokenizerType(str, Enum):`
- members:
  - `HUGGINGFACE` — [`L170`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L170)
  - `SENTENCEPIECE` — [`L169`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L169)
  - `TIKTOKEN` — [`L171`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L171)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`tokenizer_type`](types.md#Tokenizer.tokenizer_type)

### `TrainingLoop`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1476`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1476)
- doc: Configuration for the main training loop, evaluation, and reproducibility.
- signature: `class TrainingLoop(BaseModel):`
- members:
  - `abort_on_inf_loss` — [`L1498`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1498)
  - `abort_on_nan_loss` — [`L1497`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1497)
  - `data_shuffle_seed` — [`L1502`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1502)
  - `dropout_rate` — [`L1500`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1500)
  - `enable_data_shuffling` — [`L1501`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1501)
  - `enable_dropout` — [`L1499`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1499)
  - `eval_interval` — [`L1485`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1485)
  - `eval_steps` — [`L1489`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1489)
  - `init_weights_seed` — [`L1503`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1503)
  - `log_period` — [`L1484`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1484)
  - `steps` — [`L1479`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1479)
  - `target_eval_loss` — [`L1493`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1493)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `VLLM`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:2098`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2098)
- doc: vLLM-specific configuration for rollouts.
- signature: `class VLLM(BaseModel):`
- members:
  - `async_scheduling` — [`L2106`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2106)
  - `debug_converter` — [`L2121`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2121)
  - `enable_dp_attention` — [`L2104`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2104)
  - `enable_expert_parallel` — [`L2105`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2105)
  - `gcs_debug_path` — [`L2125`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2125)
  - `hbm_utilization_vllm` — [`L2102`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2102)
  - `kv_cache_buffer` — [`L2101`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2101)
  - `max_num_batched_tokens` — [`L2107`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2107)
  - `max_num_seqs` — [`L2108`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2108)
  - `stop_strings` — [`L2109`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2109)
  - `swap_space_vllm_gb` — [`L2103`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2103)
  - `use_standalone_converter` — [`L2116`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2116)
  - `vllm_additional_config` — [`L2110`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2110)
  - `vllm_hf_config_path` — [`L2115`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2115)
  - `vllm_hf_overrides` — [`L2111`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2111)
  - `vllm_load_format` — [`L2117`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2117)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `VisionProjector`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:2029`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2029)
- doc: Configuration for the Vision Projector in a multimodal model.
- signature: `class VisionProjector(BaseModel):`
- members:
  - `pixel_shuffle_ratio_for_vit` — [`L2034`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2034)
  - `projector_dropout_for_vit` — [`L2035`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2035)
  - `projector_input_dim_for_vit` — [`L2032`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2032)
  - `projector_output_dim_for_vit` — [`L2033`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2033)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

### `VisionTower`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1999`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1999)
- doc: Configuration for the Vision Tower (Encoder) in a multimodal model.
- signature: `class VisionTower(BaseModel):`
- members:
  - `conv_stride_for_vit` — [`L2011`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2011)
  - `deepstack_visual_indexes_for_vit` — [`L2022`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2022)
  - `hidden_size_for_vit` — [`L2002`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2002)
  - `intermediate_size_for_vit` — [`L2003`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2003)
  - `num_attention_heads_for_vit` — [`L2004`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2004)
  - `num_channels_for_vit` — [`L2005`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2005)
  - `num_hidden_layers_for_vit` — [`L2015`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2015)
  - `num_position_embeddings_for_vit` — [`L2021`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2021)
  - `out_hidden_size_for_vit` — [`L2019`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2019)
  - `patch_size_for_vit` — [`L2010`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2010)
  - `rope_theta_for_vit` — [`L2016`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2016)
  - `spatial_merge_size_for_vit` — [`L2018`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2018)
  - `temporal_patch_size_for_vit` — [`L2020`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2020)
  - `tile_size_for_vit` — [`L2009`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2009)
  - `vision_output_dim_for_vit` — [`L2017`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2017)
  - `vision_output_length` — [`L2023`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2023)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`set_derived_and_validate_values`](types.md#MaxTextConfig.set_derived_and_validate_values), [`MaxTextConfig`](types.md#MaxTextConfig)

### `WsdDecayStyle`  ·  implements/extends Enum, str
- def: [`src/maxtext/configs/types.py:151`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L151)
- doc: Supported decay styles for WSD schedule.
- signature: `class WsdDecayStyle(str, Enum):`
- members:
  - `COSINE` — [`L155`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L155)
  - `LINEAR` — [`L154`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L154)
- used by: [`wsd_decay_style`](types.md#Optimizer.wsd_decay_style)

### `XProfTPUPowerTraceMode`  ·  implements/extends IntEnum
- def: [`src/maxtext/configs/types.py:45`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L45)
- doc: Enum for XProfTPUPowerTraceMode.
- signature: `class XProfTPUPowerTraceMode(enum.IntEnum):`
- members:
  - `POWER_TRACE_NONE` — [`L48`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L48)
  - `POWER_TRACE_NORMAL` — [`L49`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L49)
  - `POWER_TRACE_SPI` — [`L50`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L50)
- used by: [`xprof_tpu_power_trace_level`](types.md#Profiling.xprof_tpu_power_trace_level)

### `YarnRope`  ·  implements/extends BaseModel
- def: [`src/maxtext/configs/types.py:1672`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1672)
- doc: Configuration specific to YaRN (Yet another RoPE) scaling.
- signature: `class YarnRope(BaseModel):`
- members:
  - `beta_fast` — [`L1678`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1678)
  - `beta_slow` — [`L1679`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1679)
  - `max_position_embeddings` — [`L1675`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1675)
  - `mscale` — [`L1680`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1680)
  - `original_max_position_embeddings` — [`L1676`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1676)
  - `rope_attention_scaling` — [`L1683`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1683)
  - `rope_factor` — [`L1677`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1677)
  - `rope_interleave` — [`L1681`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1681)
  - `rope_truncate` — [`L1682`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L1682)
- uses (calls/refs, reference-scoped): [`MaxTextConfig`](types.md#MaxTextConfig)
- used by: [`MaxTextConfig`](types.md#MaxTextConfig)

## Functions
- `get_individual_scales(scale: int)` — [`L2385`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L2385) — Choose appropriate scales for individual dimensions based on global scale.

## Module values
- `AxisNames` — [`L60`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L60)
- `ModelName` — [`L207`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L207)
- `PathStr` — [`L59`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L59) — documented in [maxtext-configs-types](../../../../concepts/maxtext-configs-types.md)
- `logger` — [`L53`](../../../../../../../raw/code/maxtext/src/maxtext/configs/types.py#L53)

