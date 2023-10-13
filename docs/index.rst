OpenSoundscape
==============

OpenSoundscape (OPSO) is free and open source Python utility library analyzing bioacoustic data. 

OpenSoundscape includes utilities which can be strung together to create data analysis pipelines, including functions to:

* load and manipulate audio files
* create and manipulate spectrograms
* train convolutional neural networks (CNNs) on spectrograms with PyTorch
* run pre-trained CNNs to detect vocalizations
* detect periodic vocalizations with RIBBIT
* load and manipulate Raven annotations
* estimate the location of sound sources from synchronized recordings


OpenSoundscape's source code can be found on `GitHub <https://github.com/kitzeslab/opensoundscape>`__ and its documentation can be found on `OpenSoundscape.org <https://opensoundscape.org>`__.


Table of Contents
~~~~~~~~~~~~~~~~~~

The documentation is organized into the following sections:

**Introduction**: 

   * Quick Start Guide
   * Links to resources 
   * Orientation for PyTorch users

**Installation**: 

    * Installation instructions for Windows, Mac, and Linux operating systems. For cluster or cloud environments, follow the Linux operating system instructions
    * Instructions for using OpenSoundscape with different utilities, e.g. Jupyter, Google Colab, and Poetry (for contributors)

**Tutorials**: 

  * Step-by-step guides for how to use OpenSoundscape's common functions. 
  * These tutorials include code, examples, and downloadable data. 
  * All tutorials are written as Jupyter Notebooks that can be downloaded and run on your own computer or run on Google Colab.

**Classifiers 101**:

    * An introduction to a philosophy for training and using classifiers, influenced by our lab's work using bioacoustic classifiers for large-scale bioacoustic monitoring of animal sounds

**Codebase Documentation**:

    * Documentation for the entire API ("application programming interface") of OpenSoundscape: its functions (and their arguments) and classes (and their methods).


Contact & Citation
~~~~~~~~~~~~~~~~~~

OpenSoundcape is developed and maintained by the `Kitzes Lab <http://www.kitzeslab.org/>`_ at the University of Pittsburgh. It is currently in active development. 

If you find a bug, please `submit an issue <https://github.com/kitzeslab/opensoundscape/issues>`__ on the GitHub repository. If you have another question about OpenSoundscape, please use the `OpenSoundscape Discussions board <https://github.com/kitzeslab/opensoundscape/discussions>`__ or email Sam Lapp (``sam.lapp at pitt.edu``)


Suggested citation:

    Lapp, Sam; Rhinehart, Tessa; Freeland-Haynes, Louis; 
    Khilnani, Jatin; Syunkova, Alexandra; Kitzes, Justin. 
    “OpenSoundscape: An Open-Source Bioacoustics Analysis Package for Python.” 
    Methods in Ecology and Evolution 2023. https://doi.org/10.1111/2041-210X.14196.


.. toctree::
   :hidden:
   
   self

.. toctree::
   :maxdepth: 2
   :hidden: 
   :caption: Introduction

   intro/quick-start.md
   intro/resources.md
   intro/pytorch_orientation.md

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Installation

   installation/mac_and_linux.md
   installation/windows.md
   installation/jupyter.md
   installation/google_colab.md
   installation/contributors.md


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorials

   tutorials/audio
   tutorials/spectrogram
   tutorials/annotations
   tutorials/preprocess_audio_dataset
   tutorials/train_cnn
   tutorials/customize_cnn_training
   tutorials/predict_with_cnn
   tutorials/acoustic_localization
   tutorials/signal_processing

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Classifiers 101

   classifier_guide/overview
   classifier_guide/why_classify
   classifier_guide/data_organization
   classifier_guide/create_training_dataset
   classifier_guide/train_cnn
   classifier_guide/evaluate_cnn
   classifier_guide/retrain_cnn

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Codebase Documentation

   api/modules


.. toctree::
   :maxdepth: 0
   :caption: Index
   :hidden:

   General Index <genindex>
   Module Index <py-modindex.html>
