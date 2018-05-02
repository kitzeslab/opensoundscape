Quick Instructions
---

1. `pip install -r requirements.txt`
2. Define `openbird.ini` with updated parameters from `config/openbird.ini`
3. Start `mongod` (if using `db_rw = True`)
4. Preprocess `./openbird.py preprocess`
    - This will preprocess in parallel using all cores on your machine, to limit
      please define `num_processors = N`, where `N` is the number of
      processors you would like to use.
5. Fit a Model `./openbird.py model_fit`
    - This will generate all file and file-file statistics necessary for training
    - To do:
        - Actually train the model
        - Save the model for predictions later
6. Make a prediction `./openbird predict`
    - This will generate all file and file-file statistics necessary for predictions
    - To do:
        - Recall the model (or train it)
        - Make a prediction
