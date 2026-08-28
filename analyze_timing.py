import argparse
import json
import sys
from pathlib import Path

def calculate_union_time(intervals):
    if not intervals:
        return 0.0
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    union_time = 0.0
    current_start, current_end = sorted_intervals[0]
    
    for start, end in sorted_intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            union_time += (current_end - current_start)
            current_start, current_end = start, end
    union_time += (current_end - current_start)
    return union_time

def process_file_metrics(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return None

    overall = data.get("overall_pipeline_time", 0.0)
    iters = data.get("iterations", {})
    
    file_stats = {
        "overall_time": overall,
        "llm_wall": 0.0,
        "tool_wall": 0.0,
        "overhead_wall": 0.0,
        "llm_calls": 0,
        "tool_calls": 0,
        "iterations": []
    }
    
    for iter_id, iter_data in sorted(iters.items(), key=lambda x: int(x[0])):
        llm_calls = iter_data.get("llm_calls", [])
        tools = iter_data.get("tools", [])
        agent_events = iter_data.get("agent_events", [])
        
        # FIX: The orchestrator's 'iteration_total_time' only measures the core while-loop.
        # But pre-loop setup (e.g. preparation LLMs) and post-loop summaries are all defaulting 
        # to Iteration 0 since they share the state. This artificially deflates the bounds. 
        # We calculate the TRUE chronological span by finding the min/max timestamp of every 
        # event bucketed into this iteration.
        all_events = llm_calls + tools + agent_events
        starts = [x["start_time"] for x in all_events if "start_time" in x]
        ends = [x["end_time"] for x in all_events if "end_time" in x]
        
        if starts and ends:
            true_iter_total = max(ends) - min(starts)
        else:
            true_iter_total = iter_data.get("iteration_total_time", 0.0)
        
        llm_intervals = [(c["start_time"], c["end_time"]) for c in llm_calls if "start_time" in c]
        tool_intervals = [(t["start_time"], t["end_time"]) for t in tools if "start_time" in t]
        
        llm_wall = calculate_union_time(llm_intervals)
        tool_wall = calculate_union_time(tool_intervals)
        
        all_io_intervals = llm_intervals + tool_intervals
        active_io_wall = calculate_union_time(all_io_intervals)
        
        true_overhead = true_iter_total - active_io_wall
        true_overhead = max(0.0, true_overhead)

        # OVERHEAD BREAKDOWN DYNAMICS: EXCLUSIVE COMPUTE
        # Agents are hierarchical (Parent calls Child). If we just sum them, it will >100%.
        # To find 'Self Time' (Exclusive Compute), we subtract from each agent the intervals of:
        # 1. Any LLM calls inside it
        # 2. Any Tools inside it
        # 3. Any OTHER Agents strictly nested inside it (children)
        exclusive_agent_overheads = {}
        for i, a in enumerate(agent_events):
            a_name = a.get("agent")
            if not a_name or "start_time" not in a: continue
            
            a_start, a_end = a["start_time"], a["end_time"]
            a_dur = a_end - a_start
            if a_dur <= 0: continue
            
            minus_ints = []
            for l in llm_calls:
                if "start_time" in l and "end_time" in l and l["start_time"] >= a_start and l["end_time"] <= a_end:
                    minus_ints.append((l["start_time"], l["end_time"]))
            for t in tools:
                if "start_time" in t and "end_time" in t and t["start_time"] >= a_start and t["end_time"] <= a_end:
                    minus_ints.append((t["start_time"], t["end_time"]))
            
            for j, c in enumerate(agent_events):
                if i == j or "start_time" not in c or "end_time" not in c: continue
                c_start, c_end = c["start_time"], c["end_time"]
                c_dur = c_end - c_start
                # Map as a child if it falls within our envelope
                if c_start >= a_start and c_end <= a_end:
                    if c_dur < a_dur - 0.0001:
                        minus_ints.append((c_start, c_end))
                    elif abs(c_dur - a_dur) <= 0.0001 and j > i:
                        # Tie-breaker for perfect wrapping envelopes
                        minus_ints.append((c_start, c_end))
            
            # Subtraction logic using Interval Unions: |A - B| = |A U B| - |B|
            union_all = calculate_union_time([(a_start, a_end)] + minus_ints)
            union_minus = calculate_union_time(minus_ints)
            exclusive_compute = union_all - union_minus
            
            if exclusive_compute >= 0.01:
                exclusive_agent_overheads[a_name] = exclusive_agent_overheads.get(a_name, 0.0) + exclusive_compute

        # Guarantee the sum perfectly accounts for True Overhead. 
        # Missing time is typically ADK setup/bridge code routing between agent hooks, 
        # or tiny fractional agents filtered out for being <= 0.01s.
        assigned_ovh = sum(exclusive_agent_overheads.values())
        unaccounted_ovh = true_overhead - assigned_ovh
        if unaccounted_ovh >= 0.01:
            exclusive_agent_overheads["<Unaccounted Orchestrator Bridge>"] = unaccounted_ovh

        file_stats["llm_wall"] += llm_wall
        file_stats["tool_wall"] += tool_wall
        file_stats["overhead_wall"] += true_overhead
        file_stats["llm_calls"] += len(llm_calls)
        file_stats["tool_calls"] += len(tools)
        
        file_stats["iterations"].append({
            "id": iter_id,
            "total_time": true_iter_total,
            "llm_wall": llm_wall,
            "tool_wall": tool_wall,
            "overhead_wall": true_overhead,
            "llm_count": len(llm_calls),
            "tool_count": len(tools),
            "agent_overheads": exclusive_agent_overheads
        })
        
    return file_stats

def analyze_path(target_path: str):
    path = Path(target_path)
    if path.is_file():
        files_to_process = [path]
    elif path.is_dir():
        # Crawl the directory recursively for every telemetry file
        files_to_process = list(path.glob('**/timing_metrics.json'))
    else:
        print(f"Error: {target_path} is not a valid file or directory.")
        sys.exit(1)

    if not files_to_process:
        print(f"No timing_metrics.json files found in {target_path}")
        sys.exit(1)

    output_lines = []
    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    # Global aggregators for massive directory summaries
    total_runs = len(files_to_process)
    global_pipeline = 0.0
    global_llm_wall = 0.0
    global_tool_wall = 0.0
    global_overhead_wall = 0.0
    global_llm_calls = 0
    global_tool_calls = 0

    global_agent_ovh = {}

    for fpath in files_to_process:
        metrics = process_file_metrics(fpath)
        if not metrics:
            continue
            
        log(f"\n============================================================")
        log(f"  FILE: {fpath.name} (Node: {fpath.parent.name})")
        log(f"  PATH: {fpath}")
        log(f"============================================================")
        
        global_pipeline += metrics["overall_time"]
        global_llm_wall += metrics["llm_wall"]
        global_tool_wall += metrics["tool_wall"]
        global_overhead_wall += metrics["overhead_wall"]
        global_llm_calls += metrics["llm_calls"]
        global_tool_calls += metrics["tool_calls"]
        
        for iter_m in metrics["iterations"]:
            it_tot = iter_m["total_time"]
            log(f"--- Iteration {iter_m['id']} ---")
            log(f"  Total Iteration Time (Wall) : {it_tot:>7.2f}s")
            if it_tot > 0:
                llm = iter_m['llm_wall']
                tool = iter_m['tool_wall']
                ovh = iter_m['overhead_wall']
                log(f"  LLM Wait Time (Wall)        : {llm:>7.2f}s  ({llm/it_tot*100:>4.1f}%)  [Calls: {iter_m['llm_count']}]")
                log(f"  Tool Exec Time (Wall)       : {tool:>7.2f}s  ({tool/it_tot*100:>4.1f}%)  [Calls: {iter_m['tool_count']}]")
                log(f"  True Framework Overhead     : {ovh:>7.2f}s  ({ovh/it_tot*100:>4.1f}%)")
                
                if iter_m.get("agent_overheads"):
                    log(f"      -> Exclusive Non-IO Compute Breakdown (Excludes Child Logic):")
                    log(f"         [Note: Agents with < 0.01s exclusive time are hidden/grouped into Orchestrator Bridge]")
                    max_name_len = max((len(str(k)) for k in iter_m["agent_overheads"].keys()), default=30)
                    for ag_name, ag_ovh in sorted(iter_m["agent_overheads"].items(), key=lambda x: x[1], reverse=True):
                        log(f"         * {ag_name:<{max_name_len}} : {ag_ovh:>6.2f}s")
                        global_agent_ovh[ag_name] = global_agent_ovh.get(ag_name, 0.0) + ag_ovh
            log()

    log("\n\n============================================================")
    log("        MACRO SUMMARY (ACROSS ALL DISCOVERED NODES)         ")
    log("============================================================")
    log(f"Total Nodes/Attempts Analyzed : {total_runs}")
    log(f"Aggregated Pipeline Time   : {global_pipeline:>7.2f}s computation-hours")
    
    if global_pipeline > 0:
        log(f"Total LLM Wait Time        : {global_llm_wall:>7.2f}s  ({global_llm_wall/global_pipeline*100:>4.1f}%)  [Calls: {global_llm_calls}]")
        log(f"Total Tool Exec Time       : {global_tool_wall:>7.2f}s  ({global_tool_wall/global_pipeline*100:>4.1f}%)  [Calls: {global_tool_calls}]")
        log(f"Total Framework Overhead   : {global_overhead_wall:>7.2f}s  ({global_overhead_wall/global_pipeline*100:>4.1f}%)")
        
        if global_agent_ovh:
            log(f"  -> Breakdown of Framework Overhead by Exclusive Component Compute:")
            log(f"       [Note: Agents with < 0.01s exclusive time are hidden/grouped into Orchestrator Bridge]")
            max_name_len = max((len(str(k)) for k in global_agent_ovh.keys()), default=30)
            for ag_name, ag_ovh in sorted(global_agent_ovh.items(), key=lambda x: x[1], reverse=True):
                log(f"       * {ag_name:<{max_name_len}} : {ag_ovh:>7.2f}s")
        
        unaccounted = global_pipeline - (global_llm_wall + global_tool_wall + global_overhead_wall)
        if unaccounted > 0:
            log(f"Other / Orchestrator Setup : {unaccounted:>7.2f}s  ({unaccounted/global_pipeline*100:>4.1f}%)")
    log("============================================================\n")

    return "\n".join(output_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze auto_agent timing metrics from a file or directory.")
    parser.add_argument("target", help="Path to timing_metrics.json OR a root directory (e.g. /tmp/.../search_runs/)")
    parser.add_argument("--write_md", action="store_true", help="Write timing_summary.md in the target directory")
    args = parser.parse_args()
    
    summary_text = analyze_path(args.target)
    
    if args.write_md:
        target_path = Path(args.target)
        out_dir = target_path if target_path.is_dir() else target_path.parent
        out_file = out_dir / "timing_summary.md"
        with open(out_file, "w") as f:
            f.write("```text\n")
            f.write(summary_text)
            f.write("\n```\n")
        print(f"Saved metric summary to: {out_file}")
