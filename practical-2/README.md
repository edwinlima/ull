

# Practical 2: Learning Word Representations

## Running the models
The main models can be trained using different files:

1. skip-gram.py for Skip-gram

2. skip-gram_without_hot.py for a more efficient code of the Skip-gram to run large datasets without memory failure

3. vae_bsg.py for Bayesian Skip-gram


## Evaluation
LexicalSubstitution.ipynb : a python notebook to obtain the rankings file needed to perform the lexical substitution task


## util
util.py: helpful functions to read the input files and created vocabulary and model inputs

## Data files
  data/hansards/ or the hansards dataset 
  
  data/lst for the lst dataset and evaluation scripts to obtain the GAP given the model output rankings
