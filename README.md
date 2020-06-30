# OpenSounsdcape
---

OpenSoundscape is a utility library for analyzing bioacoustic data.
It consists of command line scripts for tasks such as preprocessing audio data,
training machine learning models to classify vocalizations, estimating the
spatial location of sounds, identifying which species' sounds are present in
acoustic data, and more.

These utilities can be strung together to create data analysis pipelines.
OpenSoundscape is designed to be run on any scale of computer:
laptop, desktop, or computing cluster.

OpenSoundscape is currently in active development. For examples of some of the
utilities offered, please see the `notebooks/` directory. If you have
a question about OpenSoundscape, please open an issue in this repository
or email Sam Lapp (`sam.lapp` at `pitt.edu`)
or Tessa Rhinehart (`tessa.rhinehart` at `pitt.edu`).

# Installation

OpenSoundscape can be installed either via pip (for users) or poetry (for
developers contributing to the code). Either way, Python 3.7 or higher is required.

## Installation via pip (most users)
Most users should install via `pip` (see below). This installation represents the
latest stable version of OpenSoundscape. However, OpenSoundscape is still in
active development, so we do not promise it is bug-free; if you find a bug,
please submit an issue.

We recommend installing OpenSoundscape in a virtual environment using
[`virtualenvwrapper`](https://virtualenvwrapper.readthedocs.io/en/latest/install.html).
Using virtual environments prevents conflicts between the packages needed to
run OpenSoundscape (its *dependencies*) and packages used to run other programs
on your computer.

After downloading `virtualenvwrapper`, use the following commands to install
OpenSoundscape in a virtual environment.

```
mkvirtualenv opensoundscape #make environment and work on it
pip install opensoundscape==1.0.0.alpha0 #install opensoundscape in environment
deactivate #deactivate environment when finished using it
workon opensoundscape #activate environment to start using it again
```

## Installation via poetry (contributors and advanced users)
Poetry installation allows direct use of the most recent version of the code.
This workflow allows advanced users to use the newest features in OpenSoundscape,
and allows developers/contributors to build and test their contributions.

To install via poetry, do the following:
- Get [poetry](https://poetry.eustace.io/docs/#installation)
- Get
  [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/install.html)
- Link `poetry` and `virtualenvwrapper`:
  - Figure out where the virtualenvwrapper.sh file is: `which virtualenvwrapper.sh`
  - Add the following to your `~/.bashrc` and source it.

    ```
    # virtualenvwrapper + poetry
    export PATH=~/.local/bin:$PATH
    export WORKON_HOME=~/Library/Caches/pypoetry/virtualenvs
    source [insert path to virtualenvwrapper.sh, e.g. ~/.local/bin/virtualenvwrapper_lazy.sh]
    ```

**Contributors**: fork the github repository and clone the fork to your machine.
We use `black` pre-commit hooks for formatting.

**Users**: clone this github repository to your machine:
`git clone https://github.com/kitzeslab/opensoundscape.git`

**After the repository is on your machine:**
- Ensure you are in the top-level directory of the clone
- Switch to the development branch of OpenSoundscape: `git checkout develop`
- Build the virtual environment for opensoundscape: `poetry install`
  - If poetry install fails with the following error, make sure to install Python 3.7:

     ```
     Installing build dependencies: started
     Installing build dependencies: finished with status 'done'
     opensoundscape requires Python '>=3.7,<4.0' but the running Python is 3.6.10
     ```
     If you are using `conda`, install Python 3.7 using `conda install python==3.7`
  - If you are on a Mac and poetry install fails to install `numba`, contact one
    of the developers for help troubleshooting your issues.
- Activate the virtual environment with the name provided at install e.g.: `workon opensoundscape-dxMTH98s-py3.6`
- Check OpenSoundscape runs: `opensoundscape -h`
- To go back to your system's Python: `deactivate`
- Running the tests (from top-level directory): `poetry run pytest`

### Jupyter
To use OpenSoundscape within JupyterLab, you will have to make an `ipykernel`
for the OpenSoundscape virtual environment.

- Activate poetry virtual environment, e.g.: `workon opensoundscape-dxMTH98s-py3.6`
    - Use `poetry list` if you're not sure what the name of the environment is
- Create ipython kernel: `python -m ipykernel install --user --name=[name of poetry environment] --display-name=OpenSoundscape`
- Now when you make a new document on JupyterLab, you should see a Python kernel available called OpenSoundscape.
- Contributors: if you include Jupyter's `autoreload`, any changes you make to the source code
  installed via poetry will be reflected whenever you run the `%autoreload` line magic in a cell:

  ```
  %load_ext autoreload
  %autoreload
  ```

<! -- ### Conda notes

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
  * Create any needed tests
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
