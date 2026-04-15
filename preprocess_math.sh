#!/bin/bash
# Preprocess the MATH-lighteval dataset into parquet format for GRPO training.
# Run this on your training server where the verl conda env is available.
#
# Usage:
#   bash preprocess_math.sh
#
set -euo pipefail

SAVE_DIR="$(pwd)/data/math"
mkdir -p "$SAVE_DIR"

echo "Preprocessing MATH dataset → ${SAVE_DIR} ..."
python verl/examples/data_preprocess/math_dataset.py \
    --local_save_dir "$SAVE_DIR"

echo ""
echo "Done. Files created:"
ls -lh "$SAVE_DIR"
echo ""
echo "Dataset stats:"
python -c "
import pandas as pd
train = pd.read_parquet('${SAVE_DIR}/train.parquet')
test  = pd.read_parquet('${SAVE_DIR}/test.parquet')
print(f'Train: {len(train)} rows, columns: {list(train.columns)}')
print(f'Test:  {len(test)} rows')
print(f'data_source: {train[\"data_source\"].iloc[0]}')
print(f'Sample prompt: {str(train[\"prompt\"].iloc[0])[:200]}')
"
