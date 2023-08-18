OpenSoundscape
==============

OpenSoundscape is free and open source Python utility library analyzing bioacoustic data. It includes utilities which can be strung together to create data analysis pipelines, including functions to:

* load and manipulate audio files
* create and manipulate spectrograms
* train convolutional neural networks (CNNs) on spectrograms with PyTorch
* run pre-trained CNNs to detect vocalizations
* detect periodic vocalizations with RIBBIT
* load and manipulate Raven annotations
* estimate the location of sound sources from synchronized recordings


OpenSoundscape's source code can be found on `GitHub <https://github.com/kitzeslab/opensoundscape>`__ and its documentation can be found on `OpenSoundscape.org <https://opensoundscape.org>`__.


############################ 
Show me the code!
############################

Just want to see quick examples of how to use OpenSoundscape? See the `Quick-start Guide </intro/quick-start>`


############################ 
Contact us
############################

OpenSoundcape is developed and maintained by the `Kitzes Lab <http://www.kitzeslab.org/>`_ at the University of Pittsburgh. It is currently in active development. If you find a bug, please `submit an issue <https://github.com/kitzeslab/opensoundscape/issues>`__ on the GitHub respository. If you have another question about OpenSoundscape, please use the `OpenSoundscape Discussions board <https://github.com/kitzeslab/opensoundscape/discussions>`__ or email Sam Lapp (``sam.lapp at pitt.edu``)


Suggested citation:

    Lapp, Sam; Rhinehart, Tessa; Freeland-Haynes, Louis; 
    Khilnani, Jatin; Syunkova, Alexandra; Kitzes, Justin. 
    “OpenSoundscape: An Open-Source Bioacoustics Analysis Package for Python.” 
    Methods in Ecology and Evolution 2023. https://doi.org/10.1111/2041-210X.14196.



############## 
Introduction
##############

   :doc:`intro/quick-start.md`
   :doc:`intro/resources.md`


############## 
Installation
##############

OpenSoundscape can be installed and run on a single computer or in a cluster or cloud environment. For cluster environments, follow the Linux operating system instructions. 

   :doc:`installation/mac_and_linux.md`
   :doc:`installation/windows.md`
   :doc:`installation/contributors.md`
   :doc:`installation/google_colab.md`
   :doc:`installation/jupyter.md`


##############
Tutorials
##############

Step-by-step guides for how to use OpenSoundscape's common functions. 

These tutorials include code, examples, and downloadable data. All tutorials are written as Jupyter Notebooks that can also be downloaded `here <https://github.com/kitzeslab/opensoundscape/tree/master/docs/tutorials>` and run on your own computer.

   :doc:`tutorials/audio_and_spectrogram`
   :doc:`tutorials/format_data`
   :doc:`tutorials/create_training_dataset`
   :doc:`tutorials/quick_start_cnn_training`
   :doc:`tutorials/evaluate_cnn_performance`
   :doc:`tutorials/customize_cnn_training`
   :doc:`tutorials/predict_with_cnn`
   :doc:`tutorials/review_cnn_predictions`
   :doc:`tutorials/acoustic_localization`
   :doc:`tutorials/signal_processing`

############################
Classifiers 101
############################
An introduction to a philosophy for training and using classifiers, influenced by our lab's work using bioacoustic classifiers for large-scale bioacoustic monitoring of animal sounds

   :doc:`classifier_guide/overview`
   :doc:`classifier_guide/why_classify`
   :doc:`classifier_guide/data_organization`
   :doc:`classifier_guide/create_training_dataset`
   :doc:`classifier_guide/train_cnn`
   :doc:`classifier_guide/evaluate_cnn`
   :doc:`classifier_guide/retrain_cnn`


############################
Codebase Documentation
############################

This documentation contains documentation for the entire API of OpenSoundscape: its functions (and their arguments) and classes (and their methods).

   :doc:`api/modules`


############################
Index
############################

The General Index is an alphabetized list of every function and class in OpenSoundscape. The Module Index is a list of all of the modules. Each module is an individual file that defines functions that have a similar theme.

   :doc:`General Index <genindex>`
   :doc:`Module Index <py-modindex.html>`



.. Below is the code for creating the sidebar

.. toctree::
   :maxdepth: 0
   :hidden: 
   :caption: Introduction
   
   index.rst
   intro/quick-start.md
   intro/resources.md

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Installation

   installation/mac_and_linux.md
   installation/windows.md
   installation/contributors.md
   installation/google_colab.md
   installation/jupyter.md


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Tutorials

   tutorials/audio_and_spectrogram
   tutorials/format_data
   tutorials/create_training_dataset
   tutorials/quick_start_cnn_training
   tutorials/evaluate_cnn_performance
   tutorials/customize_cnn_training
   tutorials/predict_with_cnn
   tutorials/review_cnn_predictions
   tutorials/acoustic_localization
   tutorials/signal_processing

.. toctree::
   :maxdepth: 1
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
