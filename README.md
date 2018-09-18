Quick Instructions
---

### Prerequisites

Install:
1. `libsamplerate`:
  - Arch Linux: `pacman -S libsamplerate`
  - OSX: `brew install libsamplerate`
1. MongoDB:
  - Arch Linux: `pacman -S mongodb`
  - OSX: `brew install mongodb`
2. Python 3:
  - Arch Linux: `pacman -S python`
  - OSX: `brew install python3` or Anaconda
3. Python 3 Dependencies:
  - `pip3 install -r requirements.txt`

### Setup

1. Start MongoDB:
  - Arch Linux: `systemctl start mongodb`
  - OSX: `mongod --dbpath <path>` # Default <path>=/data/db

### Running the Code

1. Define `opensoundscape.ini` with updated parameters from `config/opensoundscape.ini`
  - Minimally need to define `data_dir` and `train_file`
2. Generate spectrograms `./opensoundscape.py spect_gen`
    - This will preprocess in parallel using all cores (minus 1) on your
      machine, to further limit please define `num_processors` in your
      `opensoundscape.ini`
3. Fit a Model `./opensoundscape.py model_fit`
    - This will generate all file and file-file statistics necessary for training
    - To do:
        - Actually train a model
        - Save the model for later predictions
7. Make a prediction `./opensoundscape predict`
    - This will generate all file and file-file statistics necessary for predictions
    - To do:
        - Recall the model (or train it)
        - Make a prediction
