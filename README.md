[![https://www.singularity-hub.org/static/img/hosted-singularity--hub-%23e32929.svg](https://www.singularity-hub.org/static/img/hosted-singularity--hub-%23e32929.svg)](https://singularity-hub.org/collections/1681)

# OpenSoundscape

## Quick start guide

*Note: installation instructions are for MacOS systems only.*

* Install [Anaconda for Python 3](https://www.anaconda.com/download/#macos) and [HomeBrew](https://brew.sh/).
* Use HomeBrew to install a few other packages: `brew install libsamplerate mongodb git wget`
* Set up the Python environment:

        conda install -c conda-forge python=3.6 pip=18.0 pandas=0.23.4 numpy=1.15.1 matplotlib=2.1.2 docopt=0.6.2 scipy=1.0.0 scikit-image=0.13.1 pymongo=3.4.0 progressbar2=3.36.0 opencv=3.4.3 scikit-learn=0.20.0 #for dev: pytest==3.6.1 black==18.9b0

	pip install librosa==0.6.2 #for dev: pre-commit==1.12.0

* Download data files, the [CLO-43SD-AUDIO](https://datadryad.org/resource/doi:10.5061/dryad.j2t92) dataset:

        cd ~/Downloads
        wget "https://datadryad.org/bitstream/handle/10255/dryad.111783/CLO-43SD-AUDIO.tar.gz"
        tar -xzf CLO-43SD-AUDIO.tar.gz
        rm CLO-43SD-AUDIO.tar.gz
        

* Download our training & prediction split of a subset of the CLO-43SD dataset:
    
        cd ~/Downloads/CLO-43SD-AUDIO/
        wget https://raw.github.com/rhine3/opso-support/master/clo-43sd-train-small.csv
        wget https://raw.github.com/rhine3/opso-support/master/clo-43sd-predict-small.csv


* Download OpenSoundscape: 

        mkdir ~/Code && cd ~/Code
        git clone https://github.com/jkitzes/opensoundscape


* Download our config file, `opso-test-small.ini`
        
        cd ~/Code/opensoundscape/
        wget https://raw.github.com/rhine3/opso-support/master/opso-test-small.ini  
        
  
* Edit the `.ini` to reflect the absolute path of your `Downloads` folder, e.g. with `vim`: `vim opso-test-small.ini`
* Start the MongoDB daemon in another terminal: `mongod --config /usr/local/etc/mongod.conf`
* Run OpenSoundscape:

        ./opensoundscape.py init -i opso-test-small.ini 
        ./opensoundscape.py spect_gen -i opso-test-small.ini > spect-gen-output-small.txt
        ./opensoundscape.py model_fit -i opso-test-small.ini > model-fit-output-small.txt
        ./opensoundscape.py predict -i opso-test-small.ini > predict-output-small.txt
