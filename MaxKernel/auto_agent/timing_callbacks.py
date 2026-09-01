import time
from typing import Any, Dict, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools import BaseTool, ToolContext


def get_current_iteration(state: dict) -> str:
  return str(state.get("iteration", 0))


def _ensure_iteration_metrics(state: dict):
  if "timing_metrics" not in state:
    state["timing_metrics"] = {"overall_pipeline_time": 0, "iterations": {}}

  metrics = state["timing_metrics"]
  iteration_str = get_current_iteration(state)

  if iteration_str not in metrics["iterations"]:
    metrics["iterations"][iteration_str] = {
      "iteration_total_time": 0,
      "agents": {},  # Restored to native dict for pipeline loop
      "agent_events": [],  # Explicit array for chronological tracking
      "llm_calls": [],
      "tools": [],
      "framework_overhead": 0,
    }
  return metrics, metrics["iterations"][iteration_str]


def _get_agent_name(ctx: Any) -> str:
  name = getattr(ctx, "agent_name", None)
  if not name and hasattr(ctx, "agent"):
    name = getattr(ctx.agent, "name", None)

  if not name:
    # Fallback to console debug if perfectly opaque
    print(f"DEBUG_CTX_DIR: {dir(ctx)}", flush=True)
    try:
      print(f"DEBUG_CTX_VARS: {vars(ctx)}", flush=True)
    except Exception:
      pass
    return "Unknown"
  return str(name)


async def before_agent_callback(callback_context: CallbackContext) -> None:
  state = callback_context.session.state
  agent_name = _get_agent_name(callback_context)
  state[f"_start_agent_{agent_name}"] = time.time()


async def after_agent_callback(callback_context: CallbackContext) -> None:
  state = callback_context.session.state
  agent_name = _get_agent_name(callback_context)
  start_time = state.pop(f"_start_agent_{agent_name}", None)

  if start_time is not None:
    end_time = time.time()
    duration = end_time - start_time
    metrics, iter_metrics = _ensure_iteration_metrics(state)

    # 1. Maintain aggregated duration dictionary for compatibility with pipeline_agent.py
    if not isinstance(iter_metrics.get("agents"), dict):
      iter_metrics["agents"] = {}
    current_dur = iter_metrics["agents"].get(agent_name, 0.0)
    iter_metrics["agents"][agent_name] = current_dur + duration

    # 2. Append granular events array for frontend visualization tooling
    if "agent_events" not in iter_metrics:
      iter_metrics["agent_events"] = []

    iter_metrics["agent_events"].append(
      {
        "agent": agent_name,
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
      }
    )


def _extract_tokens(llm_response):
  prompt_tokens = 0
  completion_tokens = 0
  try:
    raw = getattr(llm_response, "raw_response", llm_response)
    usage_metadata = getattr(raw, "usage_metadata", None)
    usage = getattr(llm_response, "usage", None)

    if usage_metadata is not None:
      prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0)
      completion_tokens = getattr(usage_metadata, "candidates_token_count", 0)
    elif usage is not None:
      prompt_tokens = getattr(
        usage, "prompt_tokens", getattr(usage, "prompt_token_count", 0)
      )
      completion_tokens = getattr(
        usage, "completion_tokens", getattr(usage, "candidates_token_count", 0)
      )
    elif isinstance(raw, dict) and isinstance(raw.get("usage"), dict):
      prompt_tokens = raw["usage"].get("prompt_tokens", 0)
      completion_tokens = raw["usage"].get("completion_tokens", 0)
  except Exception as e:
    print(f"FAILED TOKEN EXTRACTION: {e}", flush=True)
  return prompt_tokens, completion_tokens


async def before_model_callback(
  callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
  state = callback_context.session.state
  agent_name = _get_agent_name(callback_context)
  state[f"_llm_start_{agent_name}"] = time.time()


async def after_model_callback(
  callback_context: CallbackContext, llm_response: LlmResponse
) -> None:
  state = callback_context.session.state
  agent_name = _get_agent_name(callback_context)
  start_time = state.pop(f"_llm_start_{agent_name}", None)

  if start_time is not None:
    end_time = time.time()
    duration = end_time - start_time
    agent_name = _get_agent_name(callback_context)
    metrics, iter_metrics = _ensure_iteration_metrics(state)

    # --- INJECTED TOKEN TRACKING ---
    try:
      if "token_metrics" not in state:
        state["token_metrics"] = {"iterations": {}}
      t_metrics = state["token_metrics"]
      it_str = str(state.get("iteration", 0))
      if it_str not in t_metrics["iterations"]:
        t_metrics["iterations"][it_str] = {"agents": {}, "llm_calls": []}
      t_iter = t_metrics["iterations"][it_str]

      pt, ct = _extract_tokens(llm_response)
      tt = pt + ct

      if agent_name not in t_iter["agents"]:
        t_iter["agents"][agent_name] = {
          "prompt_tokens": 0,
          "completion_tokens": 0,
          "total_tokens": 0,
          "calls": 0,
        }

      t_iter["agents"][agent_name]["prompt_tokens"] += pt
      t_iter["agents"][agent_name]["completion_tokens"] += ct
      t_iter["agents"][agent_name]["total_tokens"] += tt
      t_iter["agents"][agent_name]["calls"] += 1

      t_iter["llm_calls"].append(
        {
          "agent": agent_name,
          "prompt_tokens": pt,
          "completion_tokens": ct,
          "total_tokens": tt,
          "timestamp": time.time(),
        }
      )
    except Exception as e:
      print(f"FAILED TOKEN TRACKING LOGIC: {e}", flush=True)
    # -------------------------------

    iter_metrics["llm_calls"].append(
      {
        "agent": agent_name,
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
      }
    )


async def before_tool_callback(
  tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext
) -> None:
  tool_context.session.state[f"_tool_start_{tool.name}"] = time.time()


async def after_tool_callback(
  tool: BaseTool,
  args: Dict[str, Any],
  tool_context: ToolContext,
  tool_response: Optional[Dict],
) -> None:
  start_time = tool_context.session.state.pop(f"_tool_start_{tool.name}", None)
  if start_time is not None:
    end_time = time.time()
    duration = end_time - start_time
    metrics, iter_metrics = _ensure_iteration_metrics(
      tool_context.session.state
    )

    status = "success"
    if tool_response and isinstance(tool_response, dict):
      if "error" in tool_response or not tool_response.get("success", True):
        status = "fail"

    iter_metrics["tools"].append(
      {
        "tool_name": tool.name,
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "status": status,
      }
    )
