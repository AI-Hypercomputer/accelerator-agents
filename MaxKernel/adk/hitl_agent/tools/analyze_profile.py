import json

from xprof.convert import raw_to_tool_data


def analyze_trace(path):
  tool_data_result, _ = raw_to_tool_data.xspace_to_tool_data(
    [path], "trace_viewer", {}
  )
  trace_data = json.loads(tool_data_result)
  events = trace_data.get("traceEvents", [])
  pid = None
  start_last = None
  end_last = None
  sync_wait_total = 0

  events_for_tpu_0 = []
  jit_computation_events = []
  for event in events:
    if "args" in event and event["args"].get("name", None) == "/device:TPU:0":
      pid = event.get("pid", -1)
    if event.get("pid", -1) == pid:
      events_for_tpu_0.append(event)
      if "jit_computation" in event.get("name", None):
        jit_computation_events.append(event)

  start_last = (
    jit_computation_events[-2]["ts"] + jit_computation_events[-2]["dur"]
  )
  end_last = (
    jit_computation_events[-1]["ts"] + jit_computation_events[-1]["dur"]
  )

  for event in events_for_tpu_0:
    if "dur" in event:
      if event["ts"] >= start_last and (event["ts"] + event["dur"]) <= end_last:
        if "SyncWait" in event.get("name", None):
          sync_wait_total += event["dur"]

  total_computation_time = end_last - start_last
  if total_computation_time > 0:
    ratio = sync_wait_total / total_computation_time
    print(
      f"We see that kernel spends {ratio * 100:.4f}% waiting for synchronization and {(1 - ratio) * 100:.4f}% computing."
    )
    return ratio


if __name__ == "__main__":
  path = "/tmp/profile-data/plugins/profile/2025_09_17_23_21_59/t1v-n-ab373229-w-0.xplane.pb"
  analyze_trace(path)
