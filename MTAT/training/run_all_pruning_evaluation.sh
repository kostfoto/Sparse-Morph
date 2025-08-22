#!/bin/bash

METHODS=("relu" "maxout" "lmpl" "lmpl2" "lmpl2bn")  
PRUNING_RATIOS=("0.8" "0.9" "0.95" "0.98") 

for METHOD in "${METHODS[@]}"; do
    for PRUNE1 in "${PRUNING_RATIOS[@]}"; do
        for PRUNE2 in "${PRUNING_RATIOS[@]}"; do
            bash run_pruning_evaluation.sh $METHOD $PRUNE1 $PRUNE2
        done
    done
done
