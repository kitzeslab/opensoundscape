# Contributors

## Poetry installation
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
