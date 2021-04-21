# Jupyter
To use OpenSoundscape in JupyterLab or in a Jupyter Notebook, you may either start Jupyter from within your OpenSoundscape virtual environment and use the "Python 3" kernel in your notebooks, or create a separate "OpenSoundscape" kernel using the instructions below

The following steps assume you have already used your operating system-specific installation instructions to create a virtual environement containing OpenSoundscape and its dependencies.

## Use virtual environment
* Activate your virtual environment
* Start JupyterLab or Jupyter Notebook from inside the conda environment, e.g.: `jupyter lab`
* Copy and paste the JupyterLab link into your web browser

With this method, the default "Python 3" kernel will be able to import `opensoundscape` modules.

## Create independent kernel
Use the following steps to create a kernel that appears in any notebook you open, not just notebooks opened from your virtual environment.

* Activate your virtual environment to have access to the `ipykernel` package
* Create ipython kernel with the following command, replacing `ENV_NAME` with the name of your OpenSoundscape virtual environment.

    ```
    python -m ipykernel install --user --name=ENV_NAME --display-name=OpenSoundscape
    ```

* Now when you make a new notebook on JupyterLab, or change kernels on an existing notebook, you can choose to use the "OpenSoundscape" Python kernel

Contributors: if you include Jupyter's `autoreload`, any changes you make to the source code installed via poetry will be reflected whenever you run the `%autoreload` line magic in a cell:
```
%load_ext autoreload
%autoreload
```
