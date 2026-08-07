#!/usr/bin/env python3
"""
Local Nexus Knowledge Base MCP Server (stdio/CLI).
Serves technical guidance snippets from immutable snapshots in ~/.nexus/kb/current/.
Supports concurrent session reads without locking.
"""
import os
import sys
import json
import glob

KB_DIR = os.path.expanduser("~/.nexus/kb/current")
FALLBACK_KB_DIR = os.path.expanduser("~/.nexus/kb")

def ensure_kb_dir():
    target_dir = KB_DIR if os.path.exists(KB_DIR) else FALLBACK_KB_DIR
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        # Create a sample immutable snapshot document if directory is empty
        sample_path = os.path.join(target_dir, "tpu_attention_optimization.md")
        with open(sample_path, "w") as f:
            f.write("""# TPU Attention Optimization Guidance (v20260805)
## VMEM OOM in Pallas
- Cause: Oversized BLOCK_SIZE causing scratchpad VMEM overflow.
- Recommendation: Reduce BLOCK_SIZE from 256 to 64 and enable ring attention.
## Ring Attention Pattern
- Set USE_RING_ATTENTION = True to shard QKV blocks across TPU mesh.
""")
    return target_dir

def search_kb(query: str):
    kb_path = ensure_kb_dir()
    results = []
    seen_snippets = set()
    keywords = query.lower().split()
    for filepath in glob.glob(os.path.join(kb_path, "**/*.md"), recursive=True):
        try:
            with open(filepath, "r") as f:
                lines = f.readlines()
            for idx, line in enumerate(lines):
                if any(kw in line.lower() for kw in keywords):
                    snippet = "".join(lines[max(0, idx-2):min(len(lines), idx+3)]).strip()
                    dedup_key = (os.path.basename(filepath), snippet)
                    if dedup_key not in seen_snippets:
                        seen_snippets.add(dedup_key)
                        results.append({
                            "file": os.path.basename(filepath),
                            "line": idx + 1,
                            "snippet": snippet
                        })
        except Exception:
            continue
    return results

def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        res = search_kb(query)
        print(json.dumps({"query": query, "results": res}, indent=2))
        return

    # Stdio loop supporting standard MCP JSON-RPC 2.0 & simple line requests
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")

            # Handle Standard MCP JSON-RPC 2.0 Protocol
            if method == "initialize":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "nexus_kb_server", "version": "1.0.0"}
                    }
                }
            elif method == "notifications/initialized":
                continue
            elif method == "tools/list":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": [
                            {
                                "name": "search_kb",
                                "description": "Search local Nexus Knowledge Base for Pallas, TPU optimization, and VMEM guidance.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string", "description": "Search keywords or topic"}
                                    },
                                    "required": ["query"]
                                }
                            }
                        ]
                    }
                }
            elif method == "tools/call":
                params = req.get("params", {})
                tool_name = params.get("name")
                args = params.get("arguments", {})
                query = args.get("query", "")
                res = search_kb(query)
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(res, indent=2)}]
                    }
                }
            else:
                # Direct or Legacy Query Request
                query = req.get("query", "")
                res = search_kb(query)
                resp = {"id": req_id, "result": res}

            print(json.dumps(resp))
            sys.stdout.flush()
        except Exception as e:
            err_resp = {"jsonrpc": "2.0", "id": req.get("id") if isinstance(req, dict) else None, "error": {"code": -32603, "message": str(e)}}
            print(json.dumps(err_resp))
            sys.stdout.flush()

if __name__ == "__main__":
    main()

