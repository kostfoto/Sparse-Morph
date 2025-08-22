# Sparse-Morph CIFAR-10 experiments

The CIFAR-10 experiments, which focus on pruning, are fully contained in the provided notebook. To get started, create a conda environment using the `cifar_req.txt` file, then run the notebook sequentially.

The notebook includes the definition of our Sparse-Morph morphological layer. Please note that in our implementation, creating the explicitly sparse max-plus layer can remove all connections from a morphological neuron. For this reason, the bias term in the sparse max-plus layer is required; without it, the model may produce invalid outputs.
