# Mac and Linux

OpenSoundscape can be installed on Mac and Linux machines with Python >=3.10 using the pip command `pip install opensoundscape==0.12.1`. We strongly recommend installing OpenSoundscape in a virtual environment to prevent dependency conflicts.

Below are instructions for installation with two package managers:
* `conda`: Python and package management through Anaconda, a package manager popular among scientific programmers
* `venv`: Python's included virtual environment manager, `venv`

Feel free to use another virtual environment manager (e.g. `virtualenvwrapper`) if desired.

## Installation via Anaconda

* Install Anaconda if you don't already have it.
   * Download the installer [here](https://www.anaconda.com/products/individual), or
   * follow the [installation instructions](https://docs.anaconda.com/anaconda/install/) for your operating system.
* Create a Python (>=3.10) conda environment for opensoundscape: `conda create --name opensoundscape pip python=3.10` (you can leave out the requirement of python 3.10, just make sure you have at least python 3.10)
* Activate the environment: `conda activate opensoundscape`
* Install opensoundscape using pip: `pip install opensoundscape==0.12.1`
* Deactivate the environment when you're done using it: `conda deactivate`

### Intel-chip Macs (not Apple Silicon)

You may need to install pytorch from conda-forge since newer versions are not visible to pip. Run these lines in Termina, replacing `NAME` with the desired name of your environment:

```
conda create -n NAME python=3.11
conda activate NAME
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -c conda-forge
pip install opensoundscape==0.12.1
```

## Installation via `venv`

Download Python (>=3.10) from [this website](https://www.python.org/downloads/).

Run the following commands in your bash terminal:
* Check that you have installed Python >=3.10: `python3 --version`
* Change directories to where you wish to store the environment: `cd [path for environments folder]`
    * Tip:  You can use this folder to store virtual environments for other projects as well, so put it somewhere that makes sense for you, e.g. in your home directory.
* Make a directory for virtual environments and `cd` into it: `mkdir .venv && cd .venv`
* Create an environment called `opensoundscape` in the directory: `python3 -m venv opensoundscape`
* Activate/use the environment: `source opensoundscape/bin/activate`
* Install OpenSoundscape in the environment: `pip install opensoundscape==0.12.1`
* Once you are done with OpenSoundscape, deactivate the environment: `deactivate`
* To use the environment again, you will have to refer to absolute path of the virtual environments folder. For instance, if I were on a Mac and created `.venv` inside a directory `/Users/MyFiles/Code` I would activate the virtual environment using: `source /Users/MyFiles/Code/.venv/opensoundscape/bin/activate`

For some of our functions, you will need a version of `ffmpeg >= 0.4.1`. On Mac machines, `ffmpeg` can be installed via `brew`.
