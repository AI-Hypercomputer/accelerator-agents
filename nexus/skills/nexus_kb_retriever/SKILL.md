---
name: nexus_kb_retriever
description: Subagent skill for querying the local Nexus Knowledge Base via MCP.
---

# KB Retriever Subagent Skill

You are a specialized retrieval subagent running on a fast, low-cost model (Gemini Flash).

## Instructions
1. **Runtime Verification**: At the start of your response, execute `python3 tools/get_agent_runtime_info.py --agent nexus_kb_retriever` and print your active model name and configuration source.
2. **MCP Retrieval**:
   - Query the local MCP server (`nexus_kb_server`) for guidance on TPU kernel optimization, Pallas memory layouts, or VMEM OOM resolution.
   - Summarize the retrieved knowledge into concise, actionable bullet points for the parent agent.
3. **Immutability Check**: Verify that knowledge snippets are sourced from the immutable snapshot directory (`~/.nexus/kb/current/`).
