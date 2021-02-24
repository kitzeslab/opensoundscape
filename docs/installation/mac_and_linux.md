# Mac or Linux

OpenSoundscape can be installed either via pip (for users) or poetry (for
developers contributing to the code). Either way, Python 3.7 or higher is required.

## Pip command

Already familiar with installing python packages via pip? The pip command to install OpenSoundscape is `pip install opensoundscape==0.4.6`.

## Conda

## Virtualenvwrapper

Python 3.7 is required to run OpenSoundscape. Download it from [this website](https://www.python.org/downloads/).

We recommend installing OpenSoundscape in a virtual environment to prevent dependency conflicts. Below are instructions for installation with Python's included virtual environment manager, `venv`, but feel free to use another virtual environment manager (e.g. `conda`, `virtualenvwrapper`) if desired.

Run the following commands in your bash terminal:
* Check that you have installed Python 3.7.\_: `python3 --version`
* Change directories to where you wish to store the environment: `cd [path for environments folder]`
    * Tip:  You can use this folder to store virtual environments for other projects as well, so put it somewhere that makes sense for you, e.g. in your home directory.
* Make a directory for virtual environments and `cd` into it: `mkdir .venv && cd .venv`
* Create an environment called `opensoundscape` in the directory: `python3 -m venv opensoundscape`
* Activate/use the environment `source opensoundscape/bin/activate`
* Install OpenSoundscape in the environment: `pip install opensoundscape==0.4.6`
* Once you are done with OpenSoundscape, deactivate the environment: `deactivate`
* To use the environment again, you will have to refer to absolute path of the virtual environments folder. For instance, if I were on a Mac and created `.venv` inside a directory `/Users/MyFiles/Code` I would activate the virtual environment using: `source /Users/MyFiles/Code/.venv/opensoundscape/bin/activate`

For some of our functions, you will need a version of `ffmpeg >= 0.4.1`. On Mac machines, `ffmpeg` can be installed via `brew`.
