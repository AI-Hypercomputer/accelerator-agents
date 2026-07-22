import json
import logging
import os
import sys
from typing import Optional

import graphviz

logger = logging.getLogger(__name__)


def visualize_graph(
  graph_json_path: str, output_path: Optional[str] = None
) -> None:
  """Reads a search graph JSON file and exports it to a PNG."""
  if not os.path.exists(graph_json_path):
    logger.error(f"Graph JSON file not found: {graph_json_path}")
    return

  if output_path is None:
    output_base = os.path.splitext(graph_json_path)[0]
  else:
    output_base = os.path.splitext(output_path)[0]

  with open(graph_json_path, "r") as f:
    data = json.load(f)

  nodes_data = data.get("nodes", {})
  best_node_id = data.get("best_node_id")

  dot = graphviz.Digraph(name="SearchGraph")
  dot.attr("node", shape="box", style="filled", fontname="Helvetica")

  for node_id, node_data in nodes_data.items():
    eval_data = node_data.get("evaluation") or {}
    latency_ms = eval_data.get("latency_ms")
    latency = f"{latency_ms:.4f} ms" if latency_ms is not None else "N/A"

    execution_status = node_data.get("execution_status", "UNKNOWN")
    depth = node_data.get("depth", 0)

    color = "lightblue"
    if node_id == best_node_id:
      color = "lightgreen"
    elif execution_status != "SUCCESS":
      color = "lightpink"

    label = f"ID: {node_id}\nDepth: {depth}\nLatency: {latency}"
    dot.node(node_id, label, fillcolor=color)

    parent_id = node_data.get("parent_id")
    if parent_id:
      dot.edge(parent_id, node_id)

  try:
    dot.render(filename=output_base, format="png", cleanup=True)
    logger.info(f"Successfully rendered graph to PNG at {output_base}.png")
  except Exception as e:
    logger.error(f"Failed to generate PNG: {e}")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

  if len(sys.argv) < 2:
    print(
      "Usage: python -m auto_search.utils.visualize_graph <path_to_graph.json> [output_graph_path]"
    )
    sys.exit(1)

  output = sys.argv[2] if len(sys.argv) > 2 else None
  visualize_graph(sys.argv[1], output)
