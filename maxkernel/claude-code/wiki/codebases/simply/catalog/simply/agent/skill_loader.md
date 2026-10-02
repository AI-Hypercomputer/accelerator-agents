---
title: 'Module: simply/agent/skill_loader.py'
type: catalog
provenance: extracted
module: simply/agent/skill_loader.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.agent.skill_loader`/
symbols:
  load_skills: load_skills().
  load_skills_cached: load_skills_cached().
  _load_skill_files: _load_skill_files().
  SkillInfo: SkillInfo#
  SkillInfo.name: SkillInfo#name.
  _parse_llm_response: _parse_llm_response().
  _compute_skills_hash: _compute_skills_hash().
  SkillInfo.__str__: SkillInfo#__str__().
  RatedSkill.skill: RatedSkill#skill.
  SkillInfo.description: SkillInfo#description.
  _batch_skills: _batch_skills().
  SkillInfo.content: SkillInfo#content.
  RatedSkill.rating: RatedSkill#rating.
  _build_relevance_prompt: _build_relevance_prompt().
  _prune_old_caches: _prune_old_caches().
  RatedSkill.updated_description: RatedSkill#updated_description.
  SkillInfo.folder_name: SkillInfo#folder_name.
  RatedSkill: RatedSkill#
  _RELEVANCE_SYSTEM_PROMPT: _RELEVANCE_SYSTEM_PROMPT.
  _CACHE_MAX_AGE_DAYS: _CACHE_MAX_AGE_DAYS.
---

# Module: [`simply/agent/skill_loader.py`](../../../../../../raw/code/simply/simply/agent/skill_loader.py)

## Classes

### `RatedSkill`

-   def:
    [`simply/agent/skill_loader.py:43`](../../../../../../raw/code/simply/simply/agent/skill_loader.py#L43)
-   doc: A skill with a relevance rating from the system LLM.
-   signature: `class RatedSkill:`
-   members:
    -   `rating` —
        [`L47`](../../../../../../raw/code/simply/simply/agent/skill_loader.py#L47)
    -   `skill` —
        [`L46`](../../../../../../raw/code/simply/simply/agent/skill_loader.py#L46)
    -   `updated_description` —
        [`L48`](../../../../../../raw/code/simply/simply/agent/skill_loader.py#L48)
-   uses (calls/refs, reference-scoped):
    [`SkillInfo`](skill_loader.md#SkillInfo)
-   used by: [`load_skills`](skill_loader.md#load_skills)

### `SkillInfo`

-   def:
    [`simply/agent/skill_loader.py:26`](../../../../../../raw/code/simply/simply/agent/skill_loader.py#L26)
-   doc: Information about a skill parsed from SKILL.md.
-   signature: `class SkillInfo:`
-   members:
    -   `content` —
        [`L31`](../../../../../../raw/code/simply/simply/agent/skill_loader.py#L31)
    -   `description` —
        [`L30`](../../../../../../raw/code/simply/simply/agent/skill_loader.py#L30)
    -   `folder_name` —
        [`L32`](../../../../../../raw/code/simply/simply/agent/skill_loader.py#L32)
    -   `name` —
        [`L29`](../../../../../../raw/code/simply/simply/agent/skill_loader.py#L29)
-   protocol/private:
    `__str__`[`L34`](../../../../../../raw/code/simply/simply/agent/skill_loader.py#L34)
-   used by: [`load_skills`](skill_loader.md#load_skills),
    [`_load_skill_files`](skill_loader.md#_load_skill_files),
    [`_compute_skills_hash`](skill_loader.md#_compute_skills_hash),
    [`_parse_llm_response`](skill_loader.md#_parse_llm_response),
    [`skill`](skill_loader.md#RatedSkill.skill),
    [`_batch_skills`](skill_loader.md#_batch_skills),
    [`_build_relevance_prompt`](skill_loader.md#_build_relevance_prompt)

## Functions

-   `_batch_skills(task: str, skills: list[SkillInfo], max_tokens_per_call: int,
    token_counter: Callable[[str], int])` —
    [`L115`](../../../../../../raw/code/simply/simply/agent/skill_loader.py#L115)
    — Split skills into batches that fit within the token budget.
-   `_build_relevance_prompt(task: str, skills: Sequence[SkillInfo])` —
    [`L104`](../../../../../../raw/code/simply/simply/agent/skill_loader.py#L104)
    — Build the user prompt for relevance evaluation.
-   `_compute_skills_hash(task: str, all_skills: list[SkillInfo], max_skills:
    int)` —
    [`L348`](../../../../../../raw/code/simply/simply/agent/skill_loader.py#L348)
    — Combined hash of the task, max_skills, and all loaded skill contents.
-   `_load_skill_files(skills_dir: epath.Path)` —
    [`L51`](../../../../../../raw/code/simply/simply/agent/skill_loader.py#L51)
    — Load all SKILL.md files from skill directories.
-   `_parse_llm_response(response_text: str, skills: Sequence[SkillInfo])` —
    [`L147`](../../../../../../raw/code/simply/simply/agent/skill_loader.py#L147)
    — Parse LLM response into (rating, description) pairs.
-   `_prune_old_caches(cache_dir: epath.Path)` —
    [`L363`](../../../../../../raw/code/simply/simply/agent/skill_loader.py#L363)
    — Removes cache files older than _CACHE_MAX_AGE_DAYS days.
-   `load_skills(skills_dir: str, task: str, system_llm: llm_lib.LLMBase,
    max_skills: int = 10, all_skills: list[SkillInfo] | None = None)` —
    [`L211`](../../../../../../raw/code/simply/simply/agent/skill_loader.py#L211)
    — Load and filter skills relevant to the given task.
-   `load_skills_cached(skills_dir: str, task: str, system_llm: llm_lib.LLMBase,
    max_skills: int = 10, cache_dir: epath.Path | None = None)` —
    [`L380`](../../../../../../raw/code/simply/simply/agent/skill_loader.py#L380)
    — Load and filter skills with caching.

## Module values

-   `_CACHE_MAX_AGE_DAYS` —
    [`L345`](../../../../../../raw/code/simply/simply/agent/skill_loader.py#L345)
-   `_RELEVANCE_SYSTEM_PROMPT` —
    [`L84`](../../../../../../raw/code/simply/simply/agent/skill_loader.py#L84)
