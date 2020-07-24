[![CI Status](https://github.com/kitzeslab/opensoundscape/workflows/CI/badge.svg)](https://github.com/kitzeslab/opensoundscape/actions?query=workflow%3ACI)
[![Documentation Status](https://readthedocs.org/projects/opensoundscape/badge/?version=latest)](http://opensoundscape.org/en/latest/?badge=latest)

# OpenSoundscape
---

OpenSoundscape is a utility library for analyzing bioacoustic data.
It consists of command line scripts for tasks such as preprocessing audio data,
training machine learning models to classify vocalizations, estimating the
spatial location of sounds, identifying which species' sounds are present in
acoustic data, and more.

These utilities can be strung together to create data analysis pipelines.
OpenSoundscape is designed to be run on any scale of computer:
laptop, desktop, or computing cluster.

OpenSoundscape is currently in active development. If you find a bug, please submit an issue. If you have another question about OpenSoundscape, please email Sam Lapp (`sam.lapp` at `pitt.edu`) or Tessa Rhinehart (`tessa.rhinehart` at `pitt.edu`).

For examples of some of the utilities offered, please see the `notebooks/` directory. We plan to add more vignettes and documentation soon.

# Installation

OpenSoundscape can be installed either via pip (for users) or poetry (for
developers contributing to the code). Either way, Python 3.7 or higher is required.

## Installation via pip (most users)
Python 3.7 is required to run OpenSoundscape. Download it from [this website](https://www.python.org/downloads/).

We recommend installing OpenSoundscape in a virtual environment to prevent dependency conflicts. Below are instructions for installation with Python's included virtual environment manager, `venv`, but feel free to use another virtual environment manager (e.g. `conda`, `virtualenvwrapper`) if desired.

Run the following commands in your bash terminal:
* Check that you have installed Python 3.7.\_: `python3 --version`
* Change directories to where you wish to store the environment: `cd [path for environments folder]`
    * Tip:  You can use this folder to store virtual environments for other projects as well, so put it somewhere that makes sense for you, e.g. in your home directory.
* Make a directory for virtual environments and `cd` into it: `mkdir .venv && cd .venv`
* Create an environment called `opensoundscape` in the directory: `python3 -m venv opensoundscape`
* **For Windows computers:** activate/use the environment: `opensoundscape\Scripts\activate.bat`
* **For Mac computers:** activate/use the environment `source opensoundscape/bin/activate`
* Install OpenSoundscape in the environment: `pip install opensoundscape==0.4.0`
* Once you are done with OpenSoundscape, deactivate the environment: `deactivate`
* To use the environment again, you will have to refer to absolute path of the virtual environments folder. For instance, if I were on a Mac and created `.venv` inside a directory `/Users/MyFiles/Code` I would activate the virtual environment using: `source /Users/MyFiles/Code/.venv/opensoundscape/bin/activate`
      
## Installation via poetry (contributors and advanced users)
Poetry installation allows direct use of the most recent version of the code.
This workflow allows advanced users to use the newest features in OpenSoundscape,
and allows developers/contributors to build and test their contributions.

To install via poetry, do the following:
* Download [poetry](https://poetry.eustace.io/docs/#installation)
* Download
  [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/install.html)
* Link `poetry` and `virtualenvwrapper`:
  - Figure out where the virtualenvwrapper.sh file is: `which virtualenvwrapper.sh`
  - Add the following to your `~/.bashrc` and source it.
    ```
    # virtualenvwrapper + poetry
    export PATH=~/.local/bin:$PATH
    export WORKON_HOME=~/Library/Caches/pypoetry/virtualenvs
    source [insert path to virtualenvwrapper.sh, e.g. ~/.local/bin/virtualenvwrapper_lazy.sh]
    ```
* **Users**: clone this github repository to your machine:
`git clone https://github.com/kitzeslab/opensoundscape.git`
* **Contributors**: fork this github repository and clone the fork to your machine
* Ensure you are in the top-level directory of the clone
* Switch to the development branch of OpenSoundscape: `git checkout develop`
* Build the virtual environment for opensoundscape: `poetry install`
  - If poetry install outputs the following error, make sure to download Python 3.7:

     ```
     Installing build dependencies: started
     Installing build dependencies: finished with status 'done'
     opensoundscape requires Python '>=3.7,<4.0' but the running Python is 3.6.10
     ```
     If you are using `conda`, install Python 3.7 using `conda install python==3.7`
  - If you are on a Mac and poetry install fails to install `numba`, contact one
    of the developers for help troubleshooting your issues.
* Activate the virtual environment with the name provided at install e.g.: `workon opensoundscape-dxMTH98s-py3.7` or `poetry shell`
* Check that OpenSoundscape runs: `opensoundscape -h`
* Run tests (from the top-level directory): `poetry run pytest`
* Go back to your system's Python when you are done: `deactivate`

### Jupyter
To use OpenSoundscape within JupyterLab, you will have to make an `ipykernel`
for the OpenSoundscape virtual environment.

- Activate poetry virtual environment, e.g.: `workon opensoundscape-dxMTH98s-py3.7`
    - Use `poetry env list` if you're not sure what the name of the environment is
- Create ipython kernel: `python -m ipykernel install --user --name=[name of poetry environment] --display-name=OpenSoundscape`
- Now when you make a new document on JupyterLab, you should see a Python kernel available called OpenSoundscape.
- Contributors: if you include Jupyter's `autoreload`, any changes you make to the source code
  installed via poetry will be reflected whenever you run the `%autoreload` line magic in a cell:

  ```
  %load_ext autoreload
  %autoreload
  ```

<!-- ### Conda notes

Error in installing numba during poetry install:
```
 compile options: '-I/Users/tessa/Library/Caches/pypoetry/virtualenvs/opensoundscape-dxMTH98s-py3.7/include -I/anaconda3/include/python3.7m -c'
    extra options: '-fopenmp -std=c++11'
    gcc: numba/np/ufunc/gufunc_scheduler.cpp
    gcc: numba/np/ufunc/omppool.cpp
    clang: error: unsupported option '-fopenmp'
    clang: error: unsupported option '-fopenmp'
    error: Command "gcc -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -I/anaconda3/include -arch x86_64 -I/anaconda3/include -arch x86_64 -I/Users/tessa/Library/Caches/pypoetry/virtualenvs/opensoundscape-dxMTH98s-py3.7/include -I/anaconda3/include/python3.7m -c numba/np/ufunc/omppool.cpp -o build/temp.macosx-10.7-x86_64-3.7/numba/np/ufunc/omppool.o -fopenmp -std=c++11" failed with exit status 1
```

Fixed by using conda clang
```
 conda install clang_osx-64 clangxx_osx-64
```

Alternatively can do using brew:
```
brew install llvm libomp
```

but have to add 3 lines to bash_profile (see here: https://embeddedartistry.com/blog/2017/02/24/installing-llvm-clang-on-osx/)

Might also be solved by updating xcode -->

### Contributing to code

Make contributions by editing the code in your fork. Create branches
for features using `git checkout -b feature_branch_name` and push these
changes to remote using `git push -u origin feature_branch_name`. To merge a
feature branch into the development branch, use the GitHub
web interface to create a merge request.

When contributions in your fork are complete, open a pull request using the
GitHub web interface. Before opening a PR, do the following to
ensure the code is consistent with the rest of the package:
* Run tests: `poetry run pytest`
* Format the code with `black` style (from the top level of the repo): `black .`
* Additional libraries to be installed should be installed with `poetry add`, but
  in most cases contributors should not add libraries.


### Contributing to documentation

Build the documentation using either poetry or sphinx-build
- With poetry: `poetry run build_docs`
- With sphinx-build: `sphinx-build doc doc/_build`

Publish the documentation with the following commands:

```
rm -rf /tmp/gh-pages
cp -r doc/_build /tmp/gh-pages
git checkout gh-pages
rm -rf *
cp -r /tmp/gh-pages/* .
git add .
git commit -m "Updated gh-pages"
git push
git checkout master
```
