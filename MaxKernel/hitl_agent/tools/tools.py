"""Tool setup for HITL kernel generation agents."""

import os
import logging
from google.adk.tools import ToolContext
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from google.adk.models import LlmRequest
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from vertexai.preview import rag
from hitl_agent.tools.search_api_tool import search_api_tool
from hitl_agent.config import WORKDIR, RAG_CORPUS


# Custom VertexAiRagRetrieval that forces function_declarations mode to avoid
# incompatibility with MCPToolset when using Gemini 2.x models
class CompatibleVertexAiRagRetrieval(VertexAiRagRetrieval):
  """VertexAiRagRetrieval that uses function_declarations instead of retrieval mode.

    This avoids the 400 INVALID_ARGUMENT error when mixing with MCPToolset.
    """

  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    # Always use function_declarations mode, even for Gemini 2+
    # to maintain compatibility with MCPToolset
    from google.adk.tools.retrieval.base_retrieval_tool import BaseRetrievalTool
    await BaseRetrievalTool.process_llm_request(self,
                                                tool_context=tool_context,
                                                llm_request=llm_request)


# Read-only filesystem tool for orchestration agent (no write access)
filesystem_tool_r = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command='npx',
            args=[
                "-y",  # Argument for npx to auto-confirm install
                "@modelcontextprotocol/server-filesystem@0.5.1",
                os.path.abspath(WORKDIR),
            ],
        ),),
    # Optional: Filter which tools from the MCP server are exposed
    tool_filter=['list_directory', 'read_file'])

# Read-write filesystem tool for sub-agents
filesystem_tool_rw = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command='npx',
            args=[
                "-y",  # Argument for npx to auto-confirm install
                "@modelcontextprotocol/server-filesystem@0.5.1",
                os.path.abspath(WORKDIR),
            ],
        ),),
    # Optional: Filter which tools from the MCP server are exposed
    tool_filter=['list_directory', 'read_file', 'write_file'])

# Vertex AI RAG Engine tool
vertex_ai_rag_tool = None
if RAG_CORPUS:
  vertex_ai_rag_tool = CompatibleVertexAiRagRetrieval(
      name='retrieval_tool',
      description=
      'Use this tool to retrieve Pallas/JAX/TPU documentation and examples from the RAG corpus. This is helpful for answering questions about Pallas concepts, JAX APIs, TPU architecture, best practices, and implementation patterns.',
      rag_resources=[rag.RagResource(rag_corpus=RAG_CORPUS)],
      similarity_top_k=10,
      vector_distance_threshold=0.6,
  )
  logging.info(
      f"Initialized CompatibleVertexAiRagRetrieval with corpus: {RAG_CORPUS}")
else:
  logging.warning(
      "RAG_CORPUS not set. VertexAiRagRetrieval tool will not be available.")

__all__ = [
    'search_api_tool',
    'filesystem_tool_r',
    'filesystem_tool_rw',
    'vertex_ai_rag_tool',
    'CompatibleVertexAiRagRetrieval',
]
