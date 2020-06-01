Opensoundscape
---

### Installation

We recommend you run Opensoundscape into a Python 3.7 virtual environment. If
you don't already have Python installed you can use [the Anaconda
distribution](https://www.anaconda.com/distribution/#download-section). Simply
select your platform (Windows, macOS, or Linux) and download a Python 3.7
64-bit installer. You will also need to run a MongoDB server which is used to
store data for Opensoundscape. The documentation for this process is available
[here](https://docs.mongodb.com/manual/installation/#mongodb-community-edition).
The documentation also explains how to start the service on various platforms.

Now, you can get Opensoundscape via
[PyPI](https://pypi.org/project/opensoundscape/0.3.0.1).  I will show the
`conda` commands below, but we use `virtualenvwrapper` internally.

1. Create the environment and install Opensoundscape: `conda create --name
   opensoundscape python=3.7 opensoundscape=0.3.0.1`
2. Activate the environment: `conda activate opensoundscape`
3. Check if everything worked: `opensoundscape -h`
4. Deactivate the environment when finished: `conda deactivate`

### Singularity Container

Currently, Singularity is only working on Linux. The developers recently showed
a development version which works on
[macOS](https://www.linkedin.com/feed/update/urn:li:activity:6505987087735623680/).
You can pull our container from
[here](https://cloud.sylabs.io/library/_container/5c7d4c0f5cf3490001ca7987).

1. Get the container: `singularity pull
   library://barrymoo/default/opensoundscape:0.3.0.1`
2. Check if Opensoundscape can run: `singularity run --app opensoundscape
   opensoundscape_0.3.0.1.sif -h`
3. Check if MongoDB can run: `singularity run --app mongodb
   opensoundscape_0.3.0.1.sif -h`

### Quick Start Guide

Going to run through a quick example of running Opensoundscape. First, we need
some data

- The [CLO-43SD-AUDIO](https://datadryad.org/resource/doi:10.5061/dryad.j2t92)
  dataset:

```
cd ~/Downloads
wget https://datadryad.org/bitstream/handle/10255/dryad.111783/CLO-43SD-AUDIO.tar.gz
tar -xzf CLO-43SD-AUDIO.tar.gz
rm CLO-43SD-AUDIO.tar.gz
```

- Download our training & prediction split of a subset of the CLO-43SD dataset:

```
cd ~/Downloads/CLO-43SD-AUDIO/
wget https://raw.github.com/rhine3/opso-support/master/clo-43sd-train-small.csv
wget https://raw.github.com/rhine3/opso-support/master/clo-43sd-predict-small.csv
```

- Make a new directory to run Opensoundscape in, using `~/clo-43sd` below

```
cd ~/clo-43sd
wget https://raw.github.com/rhine3/opso-support/master/opso-test-small.ini
```

- Edit the `.ini` to reflect the absolute path of your `Downloads` folder, e.g.
  with `vim`: `vim opso-test-small.ini`
- Start the MongoDB daemon in another terminal: `mongod --config
  /usr/local/etc/mongod.conf`
- Run Opensoundscape:

```
opensoundscape init -i opso-test-small.ini
opensoundscape spect_gen -i opso-test-small.ini > spect-gen-output-small.txt
opensoundscape model_fit -i opso-test-small.ini > model-fit-output-small.txt
opensoundscape predict -i opso-test-small.ini > predict-output-small.txt
```

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
