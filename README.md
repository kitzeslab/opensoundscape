`openbird`
---

1. `legacy-Lasseck2013` - contains Lasseck's 2013 code
  - Modified to on a modern machine w/ Python 3 (very slowly)
2. `bmooreii-Lasseck2013` - Modernize Lasseck's code towards mix-and-match models
  - Key Features:
    - Parallelism: Use all the cores on your machine when possible
    - Pluggable: Easily add new preprocessing, model fit, and prediction routines
  - Current State:
    - Only available with `lasseck2013` preprocessing and model_fitting
    - Models aren't actually generated, simply storing all data in MongoDB
      for Jupyter Notebook shenanigans.
    - `spect_gen`: Alpha, preprocess training, testing, and prediction files
    - `view`: Alpha, should be able to see segments and spectrograms for
      training, testing, and prediction files
    - `model_fit`: Not done, only generates statistics necessary for fitting a model
    - `predict`: Not done, only generates statistics necessary for making predictions
  - To Do:
    - Look for opportunities for abstracting parameters into the INI files
    - `model_fit`:
      - Actually fit a model
      - Test the model if testing files are requested, else cross-validate
      - Save the model for later recollection
    - `predict`:
      - Recall, or train, a model from `model_fit`
      - Make the predictions with statistics
    - Build a testing/validation framework, automate with Travis CI
    - Deploy Anaconda packages
    - Deploy Docker and Singularity containers (maybe others?)
