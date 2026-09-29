---
title: 'Module: simply/agent/main.py'
type: catalog
provenance: extracted
module: simply/agent/main.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.agent.main`/
symbols:
  main: main().
  run_agent_loop: run_agent_loop().
  parse_predefined_knowledge: parse_predefined_knowledge().
  _ENV: _ENV.
  _SKILLS_DIR: _SKILLS_DIR.
  _LLM: _LLM.
  _SYSTEM_LLM: _SYSTEM_LLM.
  _MAX_NUM_SKILLS: _MAX_NUM_SKILLS.
  load_task: load_task().
  _RUN_ID: _RUN_ID.
  _TASK_FILE: _TASK_FILE.
  _DEFAULT_ENV: _DEFAULT_ENV.
  _RESUME: _RESUME.
  _BASE_SYSTEM_DIR: _BASE_SYSTEM_DIR.
  _DEFAULT_SKILLS_DIR: _DEFAULT_SKILLS_DIR.
  _DISPLAY_FULL: _DISPLAY_FULL.
  setup_logging: setup_logging().
---

# Module: [`simply/agent/main.py`](../../../../../../raw/code/simply/simply/agent/main.py)

## Functions

-   `load_task(task_file_path: epath.Path)` —
    [`L116`](../../../../../../raw/code/simply/simply/agent/main.py#L116) —
    Loads a task description from a markdown file.
-   `main(argv: Sequence[str])` —
    [`L174`](../../../../../../raw/code/simply/simply/agent/main.py#L174)
-   `parse_predefined_knowledge(knowledge_file_path: epath.Path)` —
    [`L123`](../../../../../../raw/code/simply/simply/agent/main.py#L123) —
    Parses the predefined knowledge.
-   `run_agent_loop(agent: agent_lib.Agent, critique:
    critique_lib.TrajectoryCritique, system_llm: llm_lib.LLMBase, run_info:
    dict[str, str])` —
    [`L139`](../../../../../../raw/code/simply/simply/agent/main.py#L139) — Runs
    the agent loop, calling step() and saving after each step.
-   `setup_logging(session_dir: epath.Path)` —
    [`L97`](../../../../../../raw/code/simply/simply/agent/main.py#L97) — Sets
    up logging for the agent.

## Module values

-   `_BASE_SYSTEM_DIR` —
    [`L73`](../../../../../../raw/code/simply/simply/agent/main.py#L73)
-   `_DEFAULT_ENV` —
    [`L62`](../../../../../../raw/code/simply/simply/agent/main.py#L62)
-   `_DEFAULT_SKILLS_DIR` —
    [`L83`](../../../../../../raw/code/simply/simply/agent/main.py#L83)
-   `_DISPLAY_FULL` —
    [`L90`](../../../../../../raw/code/simply/simply/agent/main.py#L90)
-   `_ENV` — [`L63`](../../../../../../raw/code/simply/simply/agent/main.py#L63)
-   `_LLM` — [`L38`](../../../../../../raw/code/simply/simply/agent/main.py#L38)
-   `_MAX_NUM_SKILLS` —
    [`L78`](../../../../../../raw/code/simply/simply/agent/main.py#L78)
-   `_RESUME` —
    [`L68`](../../../../../../raw/code/simply/simply/agent/main.py#L68)
-   `_RUN_ID` —
    [`L48`](../../../../../../raw/code/simply/simply/agent/main.py#L48)
-   `_SKILLS_DIR` —
    [`L84`](../../../../../../raw/code/simply/simply/agent/main.py#L84)
-   `_SYSTEM_LLM` —
    [`L43`](../../../../../../raw/code/simply/simply/agent/main.py#L43)
-   `_TASK_FILE` —
    [`L53`](../../../../../../raw/code/simply/simply/agent/main.py#L53)
