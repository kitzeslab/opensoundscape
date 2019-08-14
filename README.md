Opensoundscape
---

### Installation

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
- Running the tests: `poetry run pytest tests`
