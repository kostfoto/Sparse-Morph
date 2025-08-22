import torch
import sys
import os
import re
import numpy as np
import model as Model  # Adjust import based on your model structure
import copy

def prune_weights(model, pruning_ratio=[0.2, 0.2], method = "relu"):
    model = copy.deepcopy(model)  # Clone model to avoid modifying the original

    def prune_layer_linear(layer, r, extra=0):
        """Zero out the smallest magnitude weights in 'layer'."""
        with torch.no_grad():
            weight = layer.weight.data
            num_to_prune = int(r * weight.numel()) + extra

            # Get absolute values and sort
            flat_weights = weight.abs().flatten()
            threshold = torch.topk(flat_weights, num_to_prune, largest=False).values.max()

            # Zero out weights below the threshold
            weight[weight.abs() <= threshold] = 0.0

            remaining = weight.numel()
            if layer.bias is not None:
                remaining += layer.bias.data.numel()
            remaining -= num_to_prune
            return remaining

    def prune_layer_maxplus(layer, r, extra=0):
        """Zero out the smallest magnitude weights in 'layer'."""
        with torch.no_grad():
            weight = layer.weight.data
            num_to_prune = int(r * weight.numel()) + extra

            # Get absolute values and so
            flat_weights = weight.flatten()
            threshold = torch.topk(flat_weights, num_to_prune, largest=False).values.max()

            # Zero out weights below the threshold
            weight[weight <= threshold] = -1e9

            remaining = weight.numel()
            if layer.bias:
                remaining += layer.b.data.numel()
            remaining -= num_to_prune
            return remaining

    # Apply pruning to all linear layers in the model
    total_params = 0
    if method == "relu":
        total_params += prune_layer_linear(model.dense1, pruning_ratio[0])
        total_params += prune_layer_linear(model.dense2, pruning_ratio[1])  
    elif method == "maxout":
        for lin_layer in model.dense1.linear:
            total_params += prune_layer_linear(lin_layer, 1-(1-pruning_ratio[0])/len(model.dense1.linear), extra=lin_layer.bias.size(0)//2)
        total_params += prune_layer_linear(model.dense2, pruning_ratio[1])
    elif method == "lmpl":
        total_params += prune_layer_linear(model.dense1, 1-(1-pruning_ratio[0])/2)
        total_params += prune_layer_maxplus(model.relu, 1-(1-pruning_ratio[0])/2)
        total_params += prune_layer_linear(model.dense2, pruning_ratio[1])
    elif method == "lmpl2" or method == "lmpl2bn":
        total_params += prune_layer_linear(model.dense1, pruning_ratio[0], extra=2*model.dense1.weight.size(0))
        total_params += 3*model.dense1.weight.size(0)
        total_params += prune_layer_linear(model.dense2, pruning_ratio[1])

    return model, total_params  # Return the pruned model

def prune_and_save_model(method, i, pruning_ratios):
    model_path = f"../models/final/best_model_{method}_{i}.pth"
    save_path = f"../models/final/compressed/compressed_model_{method}_{i}.pth"
    
    # Load the model
    state_dict = torch.load(model_path)
    model = Model.ShortChunkCNN(method=method)
    model.load_state_dict(state_dict)
    
    # Prune the model
    pruned_model, total_params = prune_weights(model, pruning_ratio=pruning_ratios, method=method)
    
    # Save the pruned model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(pruned_model.state_dict(), save_path)
    
    print(f"Saved pruned model to {save_path}")
    with open(f"../models/final/compressed/compressed_model_{method}_{i}_params.txt", 'w') as f:
        f.write(f"Compressed model {method}_{i}, total_params: {total_params}\n")
    return save_path

# def parse_metrics(output):
#     roc_values = [float(m.group(1)) for m in re.finditer(r'roc_auc:\s*([0-9\.]+)', output)]
#     pr_values = [float(m.group(1)) for m in re.finditer(r'pr_auc:\s*([0-9\.]+)', output)]
#     return roc_values, pr_values

# def compute_statistics(values):
#     return np.mean(values), np.std(values, ddof=1)

def main():
    if len(sys.argv) != 4:
        print("Usage: python prune_and_evaluate.py <method> <prune1> <prune2>")
        sys.exit(1)
    
    method = sys.argv[1]
    pruning_ratios = [float(sys.argv[2]), float(sys.argv[3])]
    
    for i in range(1, 6):
    # for i in range(1, 3):
        prune_and_save_model(method, i, pruning_ratios)

if __name__ == "__main__":
    main()
