# Contributors
Contributors and advanced users can use this workflow to install via Poetry. Poetry installation allows direct use of the most recent version of the code. This workflow allows advanced users to use the newest features in OpenSoundscape, and allows developers/contributors to build and test their contributions.


## Poetry installation
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

## Contribution workflow

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
* Format the code with `black` style (from the top level of the repo): `poetry run black .`
  * To automatically handle this, `poetry run pre-commit install`
* Additional libraries to be installed should be installed with `poetry add`, but
  in most cases contributors should not add libraries.

### Contributing to documentation

Build the documentation using either poetry or sphinx-build
- With poetry: `poetry run build_docs`
- With sphinx-build: `sphinx-build doc doc/_build`
