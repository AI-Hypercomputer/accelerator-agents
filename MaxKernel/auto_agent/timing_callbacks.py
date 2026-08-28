import time
from typing import Any, Dict, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools import BaseTool, ToolContext

def get_current_iteration(state: dict) -> str:
    return str(state.get('iteration', 0))

def _ensure_iteration_metrics(state: dict):
    if 'timing_metrics' not in state:
        state['timing_metrics'] = {
            'overall_pipeline_time': 0,
            'iterations': {}
        }
    
    metrics = state['timing_metrics']
    iteration_str = get_current_iteration(state)
    
    if iteration_str not in metrics['iterations']:
        metrics['iterations'][iteration_str] = {
            'iteration_total_time': 0,
            'agents': {},
            'llm_calls': [],
            'tools': [],
            'framework_overhead': 0
        }
    return metrics, metrics['iterations'][iteration_str]

async def before_agent_callback(callback_context: CallbackContext):
    state = callback_context.state
    agent_name = callback_context.agent.name if hasattr(callback_context, 'agent') else "Unknown"
    state[f'_start_agent_{agent_name}'] = time.time()

async def after_agent_callback(callback_context: CallbackContext):
    state = callback_context.state
    agent_name = callback_context.agent.name if hasattr(callback_context, 'agent') else "Unknown"
    start_time = state.pop(f'_start_agent_{agent_name}', None)
    
    if start_time is not None:
        duration = time.time() - start_time
        metrics, iter_metrics = _ensure_iteration_metrics(state)
        current = iter_metrics['agents'].get(agent_name, 0)
        iter_metrics['agents'][agent_name] = current + duration

async def before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmRequest]:
    state = callback_context.state
    state['_current_llm_start'] = time.time()
    return llm_request
    

async def after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    state = callback_context.state
    start_time = state.pop('_current_llm_start', None)
    
    if start_time is not None:
        duration = time.time() - start_time
        agent_name = callback_context.agent.name if hasattr(callback_context, 'agent') else "Unknown"
        metrics, iter_metrics = _ensure_iteration_metrics(state)
        iter_metrics['llm_calls'].append({
            "agent": agent_name,
            "duration": duration
        })
    return llm_response
    

async def before_tool_callback(tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext) -> None:
    tool_context.state[f'_tool_start_{tool.name}'] = time.time()

async def after_tool_callback(tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext, tool_response: Optional[Dict]) -> Optional[Dict]:
    start_time = tool_context.state.pop(f'_tool_start_{tool.name}', None)
    if start_time is not None:
        duration = time.time() - start_time
        metrics, iter_metrics = _ensure_iteration_metrics(tool_context.state)
        
        status = "success"
        if tool_response and isinstance(tool_response, dict):
            if "error" in tool_response or not tool_response.get("success", True):
                status = "fail"
                
        iter_metrics['tools'].append({
            "tool_name": tool.name,
            "duration": duration,
            "status": status
        })
    return tool_response
    
