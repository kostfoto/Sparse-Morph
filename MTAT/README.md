# Sparse-Morph MTAT Experiments  

For the MTAT experiments, we build on the repository:  
[Evaluation of CNN-based Automatic Music Tagging Models, SMC 2020](https://github.com/minzwon/sota-music-tagging-models.git).  

Since that repository has excellent documentation, we recommend first setting it up and successfully training the **Short-chunk CNN** model before reproducing our experiments.  

Our modifications are limited to the **Short-chunk CNN classification head**, where we replace the original ReLU layers with **maxout**, **dense max-plus**, and **sparse max-plus (Sparse-Morph)** layers. We also provide code for pruning and reporting results.  

If you want to reproduce the experiments, please make sure to first delete the files inside of the `models/`, `models/final/`, and `models/final/compressed/` folders. 

Note: We have not written scripts for fully automating the reproduction of results, so some manual steps are required.  

---

## Conda Environment  

You can use the same conda environment as for the `sota-music-tagging-models` repository, which should also work for our code.  
If not, create a new environment using the provided `mtat_req.txt` file. 

## Preprocessing the Dataset  

1. Download the MTAT dataset into the `data/` folder.  
2. Run the preprocessing script to convert `.mp3` files to `.npy` files:  

```bash
cd preprocessing/
python -u mtat_read.py run ../data/mtat
```
These steps mirror those in the forked `sota-music-tagging-models` repository. If needed, follow their documentation for additional guidance.

## Training models

We provide only the modified Short-chunk CNN. Training other model types will result in errors.
In our paper, we trained for 100 epochs.

Run the following commands to start training:

```bash
cd training/

python -u main.py --data_path ../data/ --model_type short --n_epochs 100 --method <desired method>
```

Method options: 
```
--method relu     : Standard ReLU-based short-chunk CNN head  
--method maxout   : Short-chunk CNN head with maxout  
--method zhang    : Classification head replacing a linear layer with max-plus  
--method lmpl     : Dense max-plus layer (no batchnorm)  
--method lmplbn   : Dense max-plus layer (with batchnorm)  
--method lmpl2    : Sparse max-plus layer (Sparse-Morph) without batchnorm  
--method lmpl2bn  : Sparse max-plus layer (Sparse-Morph) with batchnorm  
```

To reproduce our paper’s results, use: `relu, maxout, zhang, lmpl, lmpl2bn`

After training, two files will appear in the `models/` folder:

1) best_model.pth

2) best_model_scores.txt

Keep them as is, you’ll need them for evaluation.

## Evaluating models

To evaluate a trained model:

```
cd training/

python -u eval.py --data_path ../data/ --model_load_path ../models/best_model.pth --model_type short --method <method used when training the model>
```

Make sure `<method>` matches the one used during training.

This script outputs the model’s scores. Manually append these results to the end of `best_model_scores.txt`.
Afterwards, that file should contain both the validation scores (across epochs) and the final evaluation scores.

## Moving Files to `/models/final/`

Once a model has been trained and evaluated, move the files manually to `models/final/` and rename them:
```
'best_model.pth' -> 'best_model_{method}_{index}.pth' 
'best_model_scores.txt' -> 'best_model_scores_{method}_{index}.txt' 
```

* `{method}`: the method used to train the model (see above). 
* `{index}`: model number.

In the paper, we trained 5 models per method, so index takes values in the set {1,2,3,4,5}. For example, for `method=relu` and `index=1` you should rename the model to `best_model_relu_1.pth` and the scores to `best_model_scores_relu_1.txt`.

## Pruning models

Once training is complete, pruning can be applied. We provide the following scripts in training/:
* `run_all_pruning_evaluation.sh`
* `run_pruning_evaluation.sh`
* `prune_and_evaluate.py`

### Running pruning for a specific method and pruning raio:
```bash
cd training/ 

bash run_pruning_evaluation.sh <method> <prune1> <prune2>
```
Arguments:
* `<method>`: The method for which the 5 trained models you wish to prune (note that if you train less than 5 models, you will have to change the for loop in prune_and_evaluate.py so that it does not throws an error trying to read from a file that does not exist
* `<prune1>`: The pruning ratio of the first layer of the head. 
* `<prune2>`: The pruning ratio of the second layer of the head. 

### Running pruning for all methods and pruning ratios:
```bash
cd training/ 

bash run_all_pruning_evaluation.sh
```

After running, files of the form:
```
metrics_{method}_{prune1}_{prune2}.txt
```
will appear in `models/final/compressed/`. 
These files contain the pruned model scores and parameters. 

Once all metrics files are created, re-run
```bash
bash run_all_pruning_evaluation.sh > ../models/final/compressed/all_metrics.txt
```
This won't re-run evaluations if all metrics files already exist. 

## Reporting results

Inside `training/`, we provide three scripts/notebooks for summarizing results:

- `run_summary_compression.py` -> summarizes pruning results  
- `run_plot_convergence.ipynb` -> generates convergence plots  
- `run_summary_scores.py` -> summarizes test scores  

### How to use  

- **Pruning and test scores**: run the Python scripts directly inside `training/`:  
  ```bash
  python run_summary_compression.py
  python run_summary_scores.py
  ```
- **Convergence plots**: Open and run the notebook `run_plot_convergence.ipynb`. By default, the notebook outputs the ROC plot. 
  
  To instead generate the PR plot, manually adjust the lines marked with 
  ```python
  ## CHANGE THIS TO OUTPUT ROC or PR PLOTS
  ```
  Make sure to update both the plotting commands **and** the corresponding regular expression lines so that the correct score type is plotted.
