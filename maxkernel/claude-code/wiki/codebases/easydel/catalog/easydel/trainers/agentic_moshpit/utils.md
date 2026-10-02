---
title: 'Module: easydel/trainers/agentic_moshpit/utils.py'
type: catalog
provenance: extracted
module: easydel/trainers/agentic_moshpit/utils.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.agentic_moshpit.utils`/
symbols:
  compute_advantages_gigpo: compute_advantages_gigpo().
  compute_advantages_step: compute_advantages_step().
  compute_advantages_episode: compute_advantages_episode().
  compute_discounted_returns: compute_discounted_returns().
  normalize_rewards_group: normalize_rewards_group().
  normalize_rewards_batch: normalize_rewards_batch().
  compute_segment_discounted_returns: compute_segment_discounted_returns().
---
# Module: [`easydel/trainers/agentic_moshpit/utils.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/utils.py)

## Functions
- `compute_advantages_episode(rewards: jax.Array, group_size: int, scale_rewards: str = "group")` — [`L138`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/utils.py#L138) — Compute episode-level advantages (standard GRPO).
- `compute_advantages_gigpo(episode_rewards: jax.Array, step_rewards_list: list[np.ndarray], group_size: int, episode_weight: float = 1, step_weight: float = 1, gamma: float = 0.95, scale_rewards: str = "group")` — [`L210`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/utils.py#L210) — Compute GiGPO-style combined episode + step advantages.
- `compute_advantages_step(step_rewards_list: list[np.ndarray], group_size: int, gamma: float = 0.95)` — [`L180`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/utils.py#L180) — Compute step-level discounted advantages.
- `compute_discounted_returns(step_rewards: np.ndarray, response_mask: np.ndarray, gamma: float = 0.95)` — [`L36`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/utils.py#L36) — Compute discounted returns for each step in a trajectory.
- `compute_segment_discounted_returns(step_rewards: np.ndarray, segment_ids: np.ndarray, gamma: float = 0.95)` — [`L65`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/utils.py#L65) — Compute discounted returns respecting segment (response turn) boundaries.
- `normalize_rewards_batch(rewards: jax.Array)` — [`L122`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/utils.py#L122) — Normalize rewards across the entire batch.
- `normalize_rewards_group(rewards: jax.Array, group_size: int)` — [`L97`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/utils.py#L97) — Normalize rewards within groups (GRPO-style).

