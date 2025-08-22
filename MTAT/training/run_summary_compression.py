import re
import os
from collections import defaultdict

def parse_metrics_file(filename):
    metrics = defaultdict(dict)
    with open(filename, "r") as f:
        content = f.read()
    
    matches = re.findall(
        r"Metrics file .*?metrics_(\w+)_(\d\.\d+)_(\d\.\d+)\.txt already exists.*?" 
        r"ROC AUC - Mean: ([0-9\.]+), Std: ([0-9\.]+).*?PR AUC - Mean: ([0-9\.]+), Std: ([0-9\.]+)", 
        content, re.DOTALL
    )
    
    for method, r1, r2, roc_mean, roc_std, pr_mean, pr_std in matches:
        metrics[(r2, r1)][method] = (roc_mean, roc_std, pr_mean, pr_std)
    
    return metrics

def generate_latex_table(metrics):
    methods = ["relu", "maxout", "lmpl", "lmpl2bn"]
    method_names = {"relu": "ReLU", "maxout": "Maxout", "lmpl": "Dense-Morph", "lmpl2bn": "Sparse-Morph. (ours)"}
    
    table = """
    \\begin{table*}[t]
        \\begin{center}
            \caption{}
            \label{table:2}
            \\vspace{-0.15cm}
            \\begin{tabular}{llcccccccc}
            \\toprule
            \multicolumn{2}{c}{Pruning ratio}& \multicolumn{2}{c}{ReLU} & \multicolumn{2}{c}{Maxout} & \multicolumn{2}{c}{Dense-Morph.} & \multicolumn{2}{c}{Sparse-Morph. (ours)} \\\\
            \cmidrule(lr){1-2} \cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}
            $r_2$ & $r_1$ & ROC-AUC & PR-AUC & ROC-AUC & PR-AUC & ROC-AUC & PR-AUC & ROC-AUC & PR-AUC \\\\
            \midrule
    """
    
    sorted_r_values = sorted(set(k[0] for k in metrics.keys()), key=float, reverse=False)
    
    for r2 in sorted_r_values:
        table += f"        \\multirow{{4}}{{*}}{{{r2}}}"
        sorted_r1_values = sorted(set(k[1] for k in metrics.keys() if k[0] == r2), key=float)
        for i, r1 in enumerate(sorted_r1_values):
            row = f"  & {r1} "
            for method in methods:
                if method in metrics[(r2, r1)]:
                    roc_mean, roc_std, pr_mean, pr_std = metrics[(r2, r1)][method]
                    row += f" & {roc_mean} $\\pm$ {roc_std} & {pr_mean} $\\pm$ {pr_std} "
                else:
                    row += " & -- & -- "
            row += "\\\\\n"
            if i == 0:
                table += row
            else:
                table += "        " + row
        table += "        \midrule\n"
    
    table += """
            \\bottomrule
            \end{tabular}
        \end{center}
    \end{table*}
    """
    
    return table

if __name__ == "__main__":
    input_file = "../models/final/compressed/all_metrics_2.txt"  # Update this with the actual filename
    metrics = parse_metrics_file(input_file)
    latex_table = generate_latex_table(metrics)
    print(latex_table)
