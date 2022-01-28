## Introduction

This package is a code supplement to the paper *Anomalous Jet Identification Via Sequence Modeling*: https://arxiv.org/abs/2105.09274

The package contains the same model used to produce the results of the paper, with a small dataset to provide a proof-of-concept scenario of the model.

## Contents

* main.py
    * Primary script to run all aspected of pre-processing, training, and evaluation
* model.py 
    * Model definition in PyTorch
* helpers/processing.py
    * Implementation of the alignment algorithm (Algorithm 1) in the paper
* helpers/train.py
    * Functions to train and evaluate the model
* helpers/plotting.py
    * Code to plot ROC curves, other evaluation plots using Matplotlib
* helpers/eval.py
    * Helper functions to handle evaluation data
*helpers/defaults.yaml
    * Definition of default argument values
* unprocessed_data/
    * Small datasets for both 2 and 3 prong signal hypotheses. Contains both contaminated datasets as well as background and signal sets independently. Number of events: 1000 Background, 100 signal for both 2 and 3 prong


## Usage

``` python main.py [arguments] ```

Example: ``` python main.py -p -t -d ```

#### Arguments

| **Option** | **Action** |
| ---------- | ---------- |
| `-p` | Pre-process the datasets in unprocessed_data/ using the alignment algorithm (Algorithm 1) |
| `-t` | Train and evaluate the model (requires processing first) |
| `-d` | Draw evaluation plots (requires training first) |
| `-j` | Maximum number of jets per event to consider, sorted in pT descending (Default 1) |
| `-c` | Maximum number of constituents per jet to consider, sorted in pT descending (Default 20)  |
| `-k` | KL-Divergence weight term in the VRNN loss function (Default 0.1) |
| `-l` | Hidden layer dimensionality (Default 16) |
| `-z` | Latent layer dimensionality (Default 2) |
| `-s` | Signal sample to process/train/evaluate (Default "2Prong") |

## Output

Processed data in .hdf5 format stored in the Output_h5/ directory
Network weights in .pth format stored in the saves/ directory
Evaluation data in .npy format stored in the eval_data/ directory
Evaluation plots stored in the plots/ directory




