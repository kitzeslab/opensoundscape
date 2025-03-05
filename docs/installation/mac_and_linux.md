# Mac and Linux

OpenSoundscape can be installed on Mac and Linux machines with Python >=3.9 using the pip command `pip install opensoundscape==0.12.0`. We recommend installing OpenSoundscape in a virtual environment to prevent dependency conflicts.

Below are instructions for installation with two package managers:
* `conda`: Python and package management through Anaconda, a package manager popular among scientific programmers
* `venv`: Python's included virtual environment manager, `venv`

Feel free to use another virtual environment manager (e.g. `virtualenvwrapper`) if desired.

## Installation via Anaconda

* Install Anaconda if you don't already have it.
   * Download the installer [here](https://www.anaconda.com/products/individual), or
   * follow the [installation instructions](https://docs.anaconda.com/anaconda/install/) for your operating system.
* Create a Python (>=3.9) conda environment for opensoundscape: `conda create --name opensoundscape pip python=3.10` (you can leave out the requirement of python 3.10, just make sure you have at least python 3.9)
* Activate the environment: `conda activate opensoundscape`
* Install opensoundscape using pip: `pip install opensoundscape==0.12.0`
* Deactivate the environment when you're done using it: `conda deactivate`

## Installation via `venv`

Download Python (>=3.9) from [this website](https://www.python.org/downloads/).

Run the following commands in your bash terminal:
* Check that you have installed Python >=3.9: `python3 --version`
* Change directories to where you wish to store the environment: `cd [path for environments folder]`
    * Tip:  You can use this folder to store virtual environments for other projects as well, so put it somewhere that makes sense for you, e.g. in your home directory.
* Make a directory for virtual environments and `cd` into it: `mkdir .venv && cd .venv`
* Create an environment called `opensoundscape` in the directory: `python3 -m venv opensoundscape`
* Activate/use the environment: `source opensoundscape/bin/activate`
* Install OpenSoundscape in the environment: `pip install opensoundscape==0.12.0`
* Once you are done with OpenSoundscape, deactivate the environment: `deactivate`
* To use the environment again, you will have to refer to absolute path of the virtual environments folder. For instance, if I were on a Mac and created `.venv` inside a directory `/Users/MyFiles/Code` I would activate the virtual environment using: `source /Users/MyFiles/Code/.venv/opensoundscape/bin/activate`

For some of our functions, you will need a version of `ffmpeg >= 0.4.1`. On Mac machines, `ffmpeg` can be installed via `brew`.
