## Create a conda environment from the included yml file
We provide a .yml file that allows a user to re-create a working conda environment for a specific version of OpenSoundscape. This can be useful if you get errors because of changes within dependencies, or if you get package conflicts. 

* download the `opensoundscape.yml` file from GitHub
* Install Anaconda if you don't already have it.
   * Download the installer [here](https://www.anaconda.com/products/individual), or
   * follow the [installation instructions](https://docs.anaconda.com/anaconda/install/) for your operating system.
* Create the conda environment from the file by running `conda env create -f opensoundscape.yml` in the command line. 

We also provide an environment configuration that includes tensorflow: `opensoundscape_tf.yml`, which you can use to load and run BirdNET and Perch from the Bioacoustics Model Zoo. See associated tutorials if you need Tensorflow to have GPU acceleration, or check requirements on [this page](https://www.tensorflow.org/install/source#gpu).