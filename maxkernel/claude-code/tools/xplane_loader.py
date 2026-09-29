#!/usr/bin/env python3
"""Shared loader for .xplane.pb traces, built on xprof's supported API.

Why this module exists
----------------------
The trace-reading tools here originally imported `xplane_pb2` from
`tensorflow.tsl.profiler.protobuf`, which pulls in a full TensorFlow install
purely to obtain one protobuf module -- and then fails anyway, because xprof
deliberately does NOT re-export `xplane_pb2`. See the OSS build of
`xprof/cli/internal/oss/xplane_tools.py`, whose `get_xspace_proto()` raises
`NotImplementedError("as_text=True is not supported in OSS because xplane_pb2
is not exposed.")`. Depending on a TF-internal proto path is therefore both
heavy and unsupported, which is why `tensorflow` is no longer a requirement.

xprof's supported reader is `xprof.profile_data.ProfileData`, which is what
xprof's own CLI tools use (`iter_planes()` in `xplane_tools.py`). This module
wraps it and rebuilds the exact SQLite schema the subagent prompts document,
so agent-authored SQL keeps working unchanged.

Schema notes (`planes` / `lines` / `events` columns)
---------------------------------------------------
`ProfileData` exposes a deliberately narrow view:
    plane -> .name, .lines, .stats
    line  -> .name, .events
    event -> .name, .start_ns, .duration_ns, .stats

The raw proto's `plane.id`, `line.id`, `line.display_id` and
`line.timestamp_ns` are not exposed. To keep the documented schema intact,
those columns are populated with stable surrogate values:

  * `planes.id`        -- 0-based index of the plane in the trace
  * `lines.id`         -- 0-based index of the line within its plane
  * `lines.display_id` -- mirrors `lines.id`
  * `lines.timestamp_ns` -- 0

They remain valid JOIN keys between `planes`, `lines` and `events` (which is
what queries actually use them for), but they are NOT the profiler's original
numeric ids and must not be compared against ids from any other source.

Event times follow xprof's own convention in `xplane_tools.py`:
`offset_ps = start_ns * 1000`, `duration_ps = duration_ns * 1000`.
"""

import gzip
import os
import pathlib
import sqlite3

from xprof import profile_data

SCHEMA = """
CREATE TABLE planes (id INTEGER, name TEXT);
CREATE TABLE lines (id INTEGER, plane_id INTEGER, display_id INTEGER,
                    name TEXT, timestamp_ns INTEGER);
CREATE TABLE events (
    plane_id INTEGER, line_id INTEGER,
    name TEXT, offset_ps INTEGER, duration_ps INTEGER,
    start_ps INTEGER, end_ps INTEGER
);
"""


def _profile_data_for(path: str):
  """Returns a ProfileData for `path`, transparently handling gzip."""
  if path.endswith(".gz"):
    with gzip.open(path, "rb") as f:
      return profile_data.ProfileData.from_serialized_xspace(f.read())
  return profile_data.ProfileData.from_file(path)


def resolve_xplane_path(path: str) -> str:
  """Accepts a file or a directory and returns a concrete .xplane.pb path.

  jax.profiler.trace() writes to a nested
  `<logdir>/plugins/profile/<timestamp>/<host>.xplane.pb`, so agents routinely
  hold the logdir rather than the file. Resolving here means every tool accepts
  either form instead of failing with a confusing parse error.
  """
  p = pathlib.Path(path)
  if p.is_dir():
    matches = sorted(p.glob("**/*.xplane.pb")) + sorted(p.glob("**/*.xspace.pb"))
    if not matches:
      raise FileNotFoundError(f"No .xplane.pb/.xspace.pb found under {path!r}")
    return str(matches[0])
  if not p.exists():
    raise FileNotFoundError(f"Path does not exist: {path!r}")
  return str(p)


def iter_planes(path: str):
  """Yields every XPlane in the trace at `path`."""
  yield from _profile_data_for(resolve_xplane_path(path)).planes


def load_into_sqlite(path: str) -> sqlite3.Connection:
  """Loads a trace into an in-memory SQLite DB using the documented schema."""
  conn = sqlite3.connect(":memory:")
  c = conn.cursor()
  c.executescript(SCHEMA)

  for plane_id, plane in enumerate(iter_planes(path)):
    c.execute("INSERT INTO planes VALUES (?, ?)", (plane_id, plane.name))
    for line_id, line in enumerate(plane.lines):
      c.execute(
          "INSERT INTO lines VALUES (?, ?, ?, ?, ?)",
          (line_id, plane_id, line_id, line.name, 0),
      )
      rows = []
      for event in line.events:
        start_ps = int(event.start_ns * 1000)
        duration_ps = int(event.duration_ns * 1000)
        rows.append((
            plane_id,
            line_id,
            event.name,
            start_ps,
            duration_ps,
            start_ps,
            start_ps + duration_ps,
        ))
      if rows:
        c.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)", rows)

  conn.commit()
  return conn
