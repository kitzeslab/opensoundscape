[![CI Status](https://github.com/kitzeslab/opensoundscape/workflows/CI/badge.svg)](https://github.com/kitzeslab/opensoundscape/actions?query=workflow%3ACI)
[![Documentation Status](https://readthedocs.org/projects/opensoundscape/badge/?version=latest)](http://opensoundscape.org/en/latest/?badge=latest)

# OpenSoundscape

OpenSoundscape is a utility library for analyzing bioacoustic data. It consists of Python modules for tasks such as preprocessing audio data, training machine learning models to classify vocalizations, estimating the spatial location of sounds, identifying which species' sounds are present in acoustic data, and more.

These utilities can be strung together to create data analysis pipelines. OpenSoundscape is designed to be run on any scale of computer: laptop, desktop, or computing cluster.

OpenSoundscape is currently in active development. If you find a bug, please submit an issue. If you have another question about OpenSoundscape, please email Sam Lapp (`sam.lapp` at `pitt.edu`) or Tessa Rhinehart (`tessa.rhinehart` at `pitt.edu`).

# Installation

OpenSoundscape can be installed on Windows, Mac, and Linux machines. It has been tested on Python 3.7 and 3.8.

Most users should install OpenSoundscape via pip: `pip install opensoundscape==0.6.1`. Contributors and advanced users can also use Poetry to install OpenSoundscape.

For more detailed instructions on how to install OpenSoundscape and use it in Jupyter, see the [documentation](http://opensoundscape.org).

# Features & Tutorials
OpenSoundscape includes functions to:
* trim, split, and manipulate audio files
* create and manipulate spectrograms
* train CNNs on spectrograms with PyTorch
* run pre-trained CNNs to detect vocalizations
* detect periodic vocalizations with RIBBIT
* load and manipulate Raven annotations

OpenSoundscape can also be used with our library of publicly available trained machine learning models for the detection of 500 common North American bird species.

For full API documentation and tutorials on how to use OpenSoundscape to work with audio and spectrograms, train machine learning models, apply trained machine learning models to acoustic data, and detect periodic vocalizations using RIBBIT, see the [documentation](http://opensoundscape.org).

# Quick Start

Using Audio and Spectrogram classes #tldr
```
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram

#load an audio file and trim out a 5 second clip
my_audio = Audio.from_file("/path/to/audio.wav")
clip_5s = my_audio.trim(0,5)

#create a spectrogram and plot it
my_spec = Spectrogram.from_audio(clip_5s)
my_spec.plot()
```

Using a pre-trained CNN to make predictions on long audio files
```
from opensoundscape.torch.models.cnn import load_model
from opensoundscape.preprocess.preprocessors import ClipLoadingSpectrogramPreprocessor
from opensoundscape.helpers import make_clip_df
from glob import glob

#get list of audio files
files = glob('./dir/*.WAV')

#generate clip df
clip_df = make_clip_df(files,clip_duration=5.0,clip_overlap=0)

#create dataset
dataset = ClipLoadingSpectrogramPreprocessor(clip_df)
#you may need to change preprocessing params to match model

#generate predictions with a model
model = load_model('/path/to/saved.model')
scores, _, _ = model.predict(dataset)

#scores is a dataframe with MultiIndex: file, start_time, end_time
#containing inference scores for each class and each audio window
```

Training a CNN with labeled audio data
```
from opensoundscape.torch.models.cnn import PytorchModel
from opensoundscape.preprocess.preprocessors import CnnPreprocessor

#load a DataFrame of one-hot audio clip labels
#(index: file paths, columns: classes)
df = pd.read_csv('my_labels.csv')

#create a preprocessor that will create and augment samples for the CNN
train_dataset = CnnPreprocessor(df)

#create a CNN and train for 2 epochs
#for simplicity, using the training set as validation (not recommended!)
#the best model is automatically saved to `./best.model`
model = PytorchModel('resnet18',classes=df.columns)
model.train(
  train_dataset=train_dataset,
  valid_dataset=train_dataset,
  epochs=2
)
```
