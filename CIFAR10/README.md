# Sparse-Morph CIFAR-10 experiments

The **CIFAR-10 experiments**, which focus on pruning, are fully contained in the provided notebook. To get started, create a conda environment using the `cifar_req.txt` file, then run the notebook sequentially. 

The notebook includes the definition of our **Sparse-Morph** morphological layer. Please note that in our implementation, creating the explicitly sparse max-plus layer can remove all connections from a morphological neuron. For this reason, **the bias term in the sparse max-plus layer is required**; without it, the model may produce invalid outputs.

### Notes on Reporting Results  

- **Pruned models**: Mean and variance are reported automatically by the code. A latex table is generated automatically with all the results. 
- **Unpruned models**: The testing cells output the accuracies of the 5 unpruned models. Unfortunately, when writing the code we did not include automatic reporting of their mean and variance. To obtain these statistics, copy the reported test accuracies into the cells at the bottom of the notebook, where the mean and variance will be computed.  
