# Contributors

Contributors and advanced users can use this workflow to install OpenSoundscape using Poetry. Poetry installation allows direct use of the most recent version of the code. This workflow allows advanced users to use the newest features in OpenSoundscape, and allows developers/contributors to build and test their contributions.

## Poetry installation

* Install [poetry](https://poetry.eustace.io/docs/#installation)
* Create a new virtual environment for the OpenSoundscape installation. If you are using Anaconda, you can create a new environment with `conda create -n opso-dev python=3.10` where `opso-dev` is the name of the new virtual environment. (You can leave out the requirement of python 3.10, just make sure you have at least python 3.9). Use `conda activate opso-dev` to enter the environment to work on OpenSoundscape and `conda deactivate opso-dev` to return to your base Python installation. If you are not using Anaconda, other packages such as `virtualenv` should work as well. Ensure that the Python version is compatible with the current version of OpenSoundscape.
* **Internal Contributors**: Clone this github repository to your machine:
`git clone https://github.com/kitzeslab/opensoundscape.git`
* **External Contributors**: Fork this github repository and clone the fork to your machine
* Ensure you are in the top-level directory of the clone
* Switch to the development branch of OpenSoundscape: `git checkout develop`
* Install OpenSoundscape using `poetry install`. This will install OpenSoundscape and its dependencies into the `opso-dev` virtual environment. By default it will install OpenSoundscape in develop mode, so that updated code in the respository can be imported without reinstallation. 
  - If you are on a Mac and poetry install fails to install `numba`, contact one
    of the developers for help troubleshooting your issues.
* Install the `ffmpeg` dependency. On a Mac, `ffmpeg` can be installed using Homebrew.
* Run the test suite to ensure that everything installed properly. From the top-level directory, run the command `pytest`.

## Contribution workflow

### Contributing to code
Make contributions by editing the code in your repo. Create branches
for features by starting with the `develop` branch and then running
`git checkout -b feature_branch_name`. Once work is complete, push the new
branch to remote using `git push -u origin feature_branch_name`. To merge a
feature branch into the development branch, use the GitHub
web interface to create a merge or a pull request. Before opening a PR, do the following to
ensure the code is consistent with the rest of the package:
* Run the test suite using `pytest`
* Format the code with `black` style (from the top level of the repo): `black .`

### Contributing to documentation

Build the documentation using `sphinx-build docs docs/_build`
