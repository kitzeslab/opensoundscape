OpenSoundscape
==============

OpenSoundscape is free and open source software for the analysis of bioacoustic recordings (`GitHub <https://github.com/kitzeslab/opensoundscape>`_). Its main goals are to allow users to train their own custom species classification models using a variety of frameworks (including convolutional neural networks) and to use trained models to predict whether species are present in field recordings. OpSo can be installed and run on a single computer or in a cluster or cloud environment.

OpenSoundcape is developed and maintained by the `Kitzes Lab <http://www.kitzeslab.org/>`_ at the University of Pittsburgh.

The Installation section below provides guidance on installing OpSo. The Tutorials pages below are written as Jupyter Notebooks that can also be downloaded from the `project repository <http://github.com/kitzeslab/opensoundscape/>`_ on GitHub.

.. toctree::
   :hidden:

   self

.. toctree::
   :maxdepth: 2
   :caption: Installation

   installation/mac_and_linux.md
   installation/windows.md
   installation/contributors.md

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/audio_and_spectrogram
   tutorials/train
   tutorials/predict
   tutorials/RIBBIT_pulse_rate_demo

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   api
   
.. toctree::
   :hidden: #hides from index

   genindex
