import sys
import yaml

with open('/usr/local/google/home/shangkunwang/kernel_agent/accelerator-agents/MaxKernel/evaluation/jaxbench_adapted_dataset/level2/51p_DeepSeek_V4_CSA/reference.py', 'r') as f:
    ref_lines = f.readlines()

# Extract get_inputs() code
start_idx = -1
end_idx = -1
for i, line in enumerate(ref_lines):
    if line.startswith('def get_inputs():'):
        start_idx = i
    elif start_idx != -1 and line.startswith('# Computation'):
        end_idx = i - 1
        break

input_gen_code = "".join(ref_lines[start_idx:end_idx]).strip()

yaml_data = {
    'task_id': 'dsv4_csa_sparse_ragged_paged_attention',
    'description': 'DeepSeek-V4 compressed sparse attention (CSA): a fused SparseCore gather of a DSV4-FP8 paged MLA cache followed by ragged flash attention over the gathered top-k tokens, merged with a sliding-window partial (swa_accumution/l/m) that a previous kernel produced.\n\nEach query token carries its own top-k list, so the gather is per-token, not per-sequence: `topk_indices[t]` names up to `csa_topk` compressed KV tokens out of the sequence\'s `kv_len` and the attention kernel then treats every query token as an independent one-token sequence over its own `csa_topk`-long KV buffer. Trailing -1 entries mark padding; `kv_lens` is derived from them and the gathered rows beyond it are zeroed (they still cost a full einsum row, and they still inflate the softmax denominator -- that is the reference semantics, not an approximation to fix).\n\nA compressed KV token is 512 bytes on the wire (448 float8_e4m3fn nope values, 7 float8_e8m0fnu scales, 57 pad bytes). Rope is stored separately as 64 bfloat16 values. Dequantised head_dim is 448 + 64 = 512, shared by every query head.',
    'input_gen_code': input_gen_code,
    'rtol': 0.02,
    'atol': 0.02
}

class LiteralStr(str):
    pass

def literal_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')

yaml.add_representer(LiteralStr, literal_presenter)

yaml_data['input_gen_code'] = LiteralStr(yaml_data['input_gen_code'])
yaml_data['description'] = LiteralStr(yaml_data['description'])

with open('/usr/local/google/home/shangkunwang/kernel_agent/accelerator-agents/MaxKernel/evaluation/jaxbench_adapted_dataset/level2/51p_DeepSeek_V4_CSA/kernel_task.yaml', 'w') as f:
    yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True)
