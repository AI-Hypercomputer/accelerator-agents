#!/bin/bash

# Navigate to the experiment directory
cd "$(dirname "$0")"

# Launch the experiment using xmanager CLI with the specified GQM flags
xmanager launch launch.py -- \
  --xm_resource_alloc="group:mlacc-gqm-dyn" \
  --xm_resource_pool="msca-dynamic" \
  --cell="yulhrp"
