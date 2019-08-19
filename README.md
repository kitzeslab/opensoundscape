Opensoundscape
---

### Motivation

We originally wanted Opensoundscape to be an end-to-end machine/deep learning
preprocessing, training, and prediction pipeline. However, we have recently
realized this doesn't make sense for most. We have decided for 1.0.0 we wanted
to create to move to a utility library...

### Installation

We recommend installing Opensoundscape in a virtual environment. Barry
uses `virtualenv/virtualenvwrapper` and Tessa uses `conda`. Either works.
This is to avoid conflicting dependencies with your current environment.

- `pip install opensoundscape==1.0.0.alpha0`

### Contributions

Contributions are highly encouraged! Our development workflow is a combination
of `virtualenvwrapper` and `poetry`. 

- Get [poetry](https://poetry.eustace.io/docs/#installation)
- Get
  [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/install.html)
- Link `poetry` and `virtualenvwrapper`, add something like the following to
  your `~/.bashrc` (don't forget to source it!)

```
# virtualenvwrapper + poetry
export WORKON_HOME=~/.cache/pypoetry/virtualenvs
source /usr/bin/virtualenvwrapper_lazy.sh
```

- Fork the github repository, and clone it
    - We use `black` pre-commit hooks for formatting
- Build the virtual environment for opensoundscape: `poetry install`
- Activate your opensoundscape environment: `workon opensoundscape-py3.7`
- Check Opensoundscape runs: `opensoundscape -h`
- To go back to your system's Python: `deactivate`
- Running the tests: `poetry run pytest`

### Building the Documentation

- With poetry: `poetry run build_docs`
- With sphinx-build: `sphinx-build doc doc/_build`
