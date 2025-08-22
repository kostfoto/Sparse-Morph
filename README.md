# Sparse-Morph
Pytorch implementation of the EUSIPCO 25 paper "Sparse Hybrid Linear-Morphological Networks".

Sparse-Morph replaces standard activation layers with explicitly sparse max-plus layers. This design offers benefits in terms of induced sparsity, pruning efficiency, and faster training convergence.

The repository includes experiments on CIFAR-10 and MTAT, each organized in a separate folder with a dedicated README for reproduction instructions.

If you are only interested in the definition of the Sparse-Morph layer, we recommend starting from the CIFAR-10 notebook.
