#!/usr/bin/env bash
#
# Copyright 2026 Sony  Semiconductor Solutions Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Run aicamera evaluation for every zoo model. Logs and summary in OUTDIR (default /tmp/eval_results).
# Run from the evaluation directory (where pyproject.toml lives). Requires: uv.

set -euo pipefail

MODELS=$(uv run evaluator list-models --format machine)
OUTDIR="/tmp/eval_results"
mkdir -p "$OUTDIR"

MODEL_COUNT=$(printf '%s\n' $MODELS | wc -l)
echo "Results will be saved to: $OUTDIR"
echo "Testing $MODEL_COUNT models."
echo ""

n=0
for m in $MODELS; do
  n=$((n + 1))
  echo "=== Running model $n/$MODEL_COUNT: $m ==="
  START=$(date +%s)
  uv run evaluator aicamera -m "$m" > "$OUTDIR/${m}.log" 2>&1
  EC=$?
  END=$(date +%s)
  ELAPSED=$((END - START))
  echo "$m|$EC|${ELAPSED}s" >> "$OUTDIR/summary.csv"
  echo "  exit_code=$EC  time=${ELAPSED}s"
done
echo "=== All done ==="
echo "Results saved to: $OUTDIR (logs: ${OUTDIR}/*.log, summary: ${OUTDIR}/summary.csv)"
