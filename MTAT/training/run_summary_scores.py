import os
import re
import numpy as np

def parse_score_file(filename):
    """Parses a score file and extracts roc_auc and pr_auc values."""
    with open(filename, "r") as f:
        content = f.read()
    
    roc_match = re.search(r'roc_auc:\s*([0-9\.]+)', content)
    pr_match = re.search(r'pr_auc:\s*([0-9\.]+)', content)
    
    if roc_match and pr_match:
        return float(roc_match.group(1)), float(pr_match.group(1))
    return None, None

def compute_statistics(methods):
    """Computes mean and standard deviation for each method over five runs."""
    for method in methods:
        roc_values = []
        pr_values = []
        
        for i in range(1, 6):
            filename = f"../models/final/best_model_scores_{method}_{i}.txt"
            if os.path.exists(filename):
                roc_auc, pr_auc = parse_score_file(filename)
                if roc_auc is not None and pr_auc is not None:
                    roc_values.append(roc_auc)
                    pr_values.append(pr_auc)
                
        if roc_values and pr_values:
            roc_mean, roc_std = np.mean(roc_values), np.std(roc_values, ddof=1)
            pr_mean, pr_std = np.mean(pr_values), np.std(pr_values, ddof=1)
            print(f"{method}: roc_auc mean: {roc_mean:.4f}, roc_auc std: {roc_std:.4f}, pr_auc mean: {pr_mean:.4f}, pr_auc std: {pr_std:.4f}")
        else:
            print(f"{method}: No valid data found.")

if __name__ == "__main__":
    methods = ["relu", "maxout", "lmpl", "lmpl2", "lmpl2bn", "zhang"]  # Replace with your list of methods
    compute_statistics(methods)
