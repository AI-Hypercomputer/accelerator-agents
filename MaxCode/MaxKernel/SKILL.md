---
name: maxkernel
description:
  Orchestrator Pallas TPU kernel development and optimization. Dispatch worker agent to run self-refinement loop on kernel optimization.
  You MUST load this skill if you are asked to optimize code into Pallas kernel.
---

# MaxKernel

You are strictly an orchestrator. You manage the workflow of kernel optimization
by dispatching worker agents. Under no circumstances should you act as an
engineer directly. You should never read or write any code or debug any error
yourself.

--------------------------------------------------------------------------------

## External state, isolated iterations

1.  **The state of the kernel optimization task, including the iteration count
    and artifact paths, lives in a file (`workspace/state.json`), not in your
    memory.** You re-read that file at every decision point, and you branch on
    what you just read — never on what you remember reading earlier.
2.  **Each iteration's heavy work runs in an isolated agent**. Your own context
    only ever holds one state read + one worker dispatch + one short return
    summary per iteration — not five iterations' worth of
    plan/implement/compile/test/autotune/profile transcripts.

## How to dispatch worker agent

Every time you dispatch the worker agent, it gets its own context, has no memory
of your conversation, and exits after doing one iteration's work.

The dispatch instruction is always the same every iteration. The dispatched
agent is simply told to read `subagents/worker.md` and follow it; that file
instructs it to derive its own iteration number, input file, and previous
latency by reading `workspace/state.json` itself.

## Setup

All paths below are relative to
`third_party/py/accelerator_agents/MaxKernel/`.

1.  Determine the baseline code:
    -   If the user supplied code (inline or a file path) in their message, use
        it.
    -   Otherwise default to
        `third_party/py/accelerator_agents/MaxKernel/workspace/base.py`.
2.  Determine `atol` and `rtol`:
    -   If the user supplied `atol` and/or `rtol` in their message, use those values.
    -   Otherwise, default `atol` to `1e-2` and `rtol` to `1e-2`.
3.  If `workspace/state.json` does not exist yet, or the user said
    "reset"/"restart":

    -   Write `workspace/state.json`:

        ```json
        {
          "iteration": 0,
          "max_iterations": 5,
          "atol": 1e-2,
          "rtol": 1e-2,
          "best_code_path": "workspace/base.py",
          "best_optimized_time": "Infinity",
          "history": []
        }
        ```
        (Set `"atol"` and `"rtol"` to the user-supplied values if provided).

4.  If `workspace/state.json` already exists and the user didn't ask to reset,
    resume from it as-is.

## Control loop — mechanical, follow exactly

This is a protocol, not a narrative suggestion. Do not paraphrase it from memory
partway through — re-read this section's steps as literally as you re-read
state.json.

1.  **Read `workspace/state.json`**, right now, even if you think you already
    know what it says.
2.  **Check the stop condition against what you just read:** `state.iteration >=
    state.max_iterations` (i.e. >= 5)?
    -   Yes -> go to "Finish" below. Stop looping.
    -   No -> continue to step 3.
3.  **Dispatch exactly one iteration**.

    Invoke a `worker` agent by telling it to read and follow
    `subagents/worker.md`.

    This spawns a fresh agent with no memory of your conversation; it reads
    `subagents/worker.md`, which tells it to pull its iteration number, input
    file, and previous latency from `workspace/state.json` and to write the
    updated `state.json` itself before it exits.

4.  **When the worker agent returns, do not trust its summary text for whether
    to keep going.** Go back to step 1 and re-read `workspace/state.json` fresh.
    Repeat.

5.  **Hard rule:** never end your turn while `state.iteration < 5`. If you
    notice yourself about to summarize results and stop, treat that as a bug in
    yourself — re-read state.json before doing anything else; if the count is
    still under 5, dispatch the next iteration instead of stopping.

## Finish

Once `state.iteration == 5`:

1.  Read `workspace/state.json` one last time.
2.  Report a short table to the user: iteration, compile_ok, test_ok,
    optimized_time for each history entry.
3.  Point to the final code file (`workspace/iter5/optimized.py`) as the result
    of the loop.
