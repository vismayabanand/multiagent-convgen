#!/usr/bin/env bash
# Runs the full A/B diversity experiment (Run B then Run A) with Groq.
# Set N=10 for a quick smoke test; increase to 100 for the full experiment.

set -e

N=${N:-10}

echo "=== Run B (steering ON, seed=42, n=$N) ==="
python3 -m toolgen.cli generate \
  --artifacts-dir ./artifacts \
  --output ./output/run_B.jsonl \
  --n "$N" \
  --seed 42 \
  --provider openai

echo ""
echo "=== Run A (steering OFF, seed=42, n=$N) ==="
python3 -m toolgen.cli generate \
  --artifacts-dir ./artifacts \
  --output ./output/run_A.jsonl \
  --n "$N" \
  --seed 42 \
  --no-cross-conversation-steering \
  --provider openai

echo ""
echo "=== Evaluate Run B ==="
python3 -m toolgen.cli evaluate --dataset ./output/run_B.jsonl --output ./output/report_B.json

echo ""
echo "=== Evaluate Run A ==="
python3 -m toolgen.cli evaluate --dataset ./output/run_A.jsonl --output ./output/report_A.json

echo ""
echo "=== Updating DESIGN.md with results ==="
python3 update_design_results.py \
  --report-a ./output/report_A.json \
  --report-b ./output/report_B.json

echo ""
echo "Done."
