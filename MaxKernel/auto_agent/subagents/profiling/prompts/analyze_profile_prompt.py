# auto_agent/subagents/profiling/prompts/analyze_profile_prompt.py
"""Prompt for analyzing profiling results using offline XProf tools."""

PROMPT = """
Your goal is to provide the results from the profiling execution and perform deep analysis.
Your response should have two parts:
1) A summary of the profiling results.
2) Deep analysis using the available offline XProf tools.

For context, here are the profiling results (might contain the xplane.pb path or direct output):
{profiling_results}

Attributes of a good analysis:
*   Observe the "DMAs_and_memory_transfers_ratio and compute_ratio".
*   Use the `load_xplane_and_query` tool to explore the profiling data if you have an xplane.pb file path.
    *   Table schemas:
        *   planes (id, name)
        *   lines (id, plane_id, display_id, name, timestamp_ns)
        *   events (plane_id, line_id, name, offset_ps, duration_ps, start_ps, end_ps)
*   Look for top ops by duration (sum(duration_ps)).
*   Use `get_hlo_dump` to check for specific HLO instructions if needed.
*   Use `get_overview_page_metrics` to retrieve high-level metrics (e.g., duty cycle, step time) for the summary.
*   Use `create_chart_from_xplane` to visualize distributions.
*   Provide actionable recommendations for performance improvement based on the analysis (e.g., specific HLO ops to optimize, memory bandwidth issues, or low duty cycle causes).
*   Use `search_api_tool` (Vertex RAG Engine) to find relevant optimization guides or similar HLO patterns in the knowledge base.

If the profiling results contain a path to an `xplane.pb` file, prioritize using the tools to get more insights.
"""
