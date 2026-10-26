"""Unit tests for incremental metric saving in AutonomousPipelineAgent."""

import json
import os
import tempfile
import time
from unittest.mock import Mock

import pytest
from google.adk.sessions import Session

from auto_agent.subagents.pipeline_agent import AutonomousPipelineAgent


@pytest.fixture
def temp_workdir():
  with tempfile.TemporaryDirectory() as tmpdir:
    yield tmpdir


@pytest.fixture
def mock_invocation_context(temp_workdir):
  session = Session(
    id="test-session-metrics",
    user_id="test-user",
    appName="auto_agent",
  )
  session.state["workdir"] = temp_workdir
  session.state["pipeline_start_time"] = time.time() - 100
  session.state["token_metrics"] = {
    "iterations": {
      "0": {
        "agents": {"PlanAgent": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "calls": 1}}
      }
    }
  }
  session.state["timing_metrics"] = {
    "overall_pipeline_time": 0,
    "iterations": {
      "0": {
        "agents": {"PlanAgent": 10.0},
        "llm_calls": [{"duration": 5.0}],
        "tools": [{"duration": 2.0}],
      }
    },
  }

  ctx = Mock()
  ctx.session = session
  return ctx


def test_save_metrics_to_disk(temp_workdir, mock_invocation_context):
  """Verify that _save_metrics_to_disk writes both token and timing files."""
  agent = Mock()
  agent.name = "TestAutonomousPipelineAgent"

  token_file = os.path.join(temp_workdir, "token_metrics.json")
  timing_file = os.path.join(temp_workdir, "timing_metrics.json")

  assert not os.path.exists(token_file)
  assert not os.path.exists(timing_file)

  AutonomousPipelineAgent._save_metrics_to_disk(agent, mock_invocation_context)

  assert os.path.exists(token_file)
  assert os.path.exists(timing_file)

  with open(token_file, "r") as f:
    token_data = json.load(f)
  assert "iterations" in token_data
  assert "0" in token_data["iterations"]

  with open(timing_file, "r") as f:
    timing_data = json.load(f)
  assert "overall_pipeline_time" in timing_data
  assert timing_data["overall_pipeline_time"] > 0


def test_update_timing_metrics_flushes_incrementally(temp_workdir, mock_invocation_context):
  """Verify that _update_timing_metrics flushes metrics on each iteration."""
  agent = Mock()
  agent.name = "TestAutonomousPipelineAgent"
  agent._save_metrics_to_disk = lambda ctx: AutonomousPipelineAgent._save_metrics_to_disk(agent, ctx)

  mock_invocation_context.session.state["iter_start_time"] = time.time() - 10.0

  token_file = os.path.join(temp_workdir, "token_metrics.json")
  timing_file = os.path.join(temp_workdir, "timing_metrics.json")

  AutonomousPipelineAgent._update_timing_metrics(agent, mock_invocation_context, iteration=0)

  assert os.path.exists(token_file)
  assert os.path.exists(timing_file)

  with open(timing_file, "r") as f:
    timing_data = json.load(f)
  iter_total = timing_data["iterations"]["0"]["iteration_total_time"]
  assert iter_total >= 9.0
