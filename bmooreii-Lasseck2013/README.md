Quick Instructions
---

1. `pip install -r requirements.txt`
2. Define `openbird.ini` with updated parameters from `config/openbird.ini`
3. Start `mongod` (if using `db_rw = True`)
4. Preprocess `./openbird.py preprocess`
    - This will preprocess in parallel using all cores on your machine, to
      limit please define `num_processors = N`, where `N` is the number of
      processors you would like to use.
