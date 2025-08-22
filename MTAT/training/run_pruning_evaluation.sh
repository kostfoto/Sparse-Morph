#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: bash run_pruning_evaluation.sh <method> <prune1> <prune2>"
    exit 1
fi

METHOD=$1
PRUNE1=$2
PRUNE2=$3
EVAL_SCRIPT="python -u eval.py --data_path ../data/ --model_load_path"
MODEL_DIR="../models/final/compressed"
METRICS_FILE="../models/final/compressed/metrics_${METHOD}_${PRUNE1}_${PRUNE2}.txt"

# If metrics file exists, skip pruning and evaluation, compute only statistics
if [ -f "$METRICS_FILE" ]; then
    echo "Metrics file $METRICS_FILE already exists. Skipping pruning and evaluation."
else
    rm -f $METRICS_FILE
    python prune_and_evaluate.py $METHOD $PRUNE1 $PRUNE2

    # Prune and evaluate models
    for i in {1..5}; do
        cat "$MODEL_DIR/compressed_model_${METHOD}_${i}_params.txt" >> $METRICS_FILE
        echo "Evaluating model $METHOD, $i at pruning ratios $PRUNE1 $PRUNE2"
        OUTPUT=$($EVAL_SCRIPT $MODEL_DIR/compressed_model_${METHOD}_$i.pth --model_type short --method ${METHOD})
        echo "$OUTPUT" >> $METRICS_FILE
    done
fi

# Extract metrics and compute statistics
python - <<EOF
import re
import numpy as np

with open("$METRICS_FILE", "r") as f:
    content = f.read()

roc_values = [float(m.group(1)) for m in re.finditer(r'roc_auc:\\s*([0-9\\.]+)', content)]
pr_values = [float(m.group(1)) for m in re.finditer(r'pr_auc:\\s*([0-9\\.]+)', content)]

roc_mean, roc_std = np.mean(roc_values), np.std(roc_values, ddof=1)
pr_mean, pr_std = np.mean(pr_values), np.std(pr_values, ddof=1)

print(f"ROC AUC - Mean: {roc_mean:.4f}, Std: {roc_std:.4f}")
print(f"PR AUC - Mean: {pr_mean:.4f}, Std: {pr_std:.4f}")
EOF
