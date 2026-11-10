# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Registry sanity checks for the probe suite. Needs jax, not a TPU."""

import collections
import os
import sys

import pytest

pytest.importorskip("jax")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402
import run_probes  # noqa: E402


@pytest.fixture(scope="module")
def probes():
  run_probes._import_all_check_modules()
  return harness.registered_probes()


def test_registry_is_not_empty(probes):
  assert probes


def test_probe_names_are_unique(probes):
  counts = collections.Counter(p.name for p in probes)
  duplicates = sorted(name for name, count in counts.items() if count > 1)
  assert not duplicates, f"duplicate probe names: {duplicates}"


def test_every_probe_has_a_question(probes):
  missing = [p.name for p in probes if not p.question.strip()]
  assert not missing, f"probes with an empty question: {missing}"
