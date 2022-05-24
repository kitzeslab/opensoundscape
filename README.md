[![CI Status](https://github.com/kitzeslab/opensoundscape/workflows/CI/badge.svg)](https://github.com/kitzeslab/opensoundscape/actions?query=workflow%3ACI)
[![Documentation Status](https://readthedocs.org/projects/opensoundscape/badge/?version=latest)](http://opensoundscape.org/en/latest/?badge=latest)

# OpenSoundscape

OpenSoundscape is a utility library for analyzing bioacoustic data. It consists of Python modules for tasks such as preprocessing audio data, training machine learning models to classify vocalizations, estimating the spatial location of sounds, identifying which species' sounds are present in acoustic data, and more.

These utilities can be strung together to create data analysis pipelines. OpenSoundscape is designed to be run on any scale of computer: laptop, desktop, or computing cluster.

OpenSoundscape is currently in active development. If you find a bug, please submit an issue. If you have another question about OpenSoundscape, please email Sam Lapp (`sam.lapp` at `pitt.edu`).


#### Suggested Citation
```
Lapp, Rhinehart, Freeland-Haynes, Khilnani, and Kitzes, 2022. "OpenSoundscape v0.7.0".
```

# Installation

OpenSoundscape can be installed on Windows, Mac, and Linux machines. It has been tested on Python 3.7 and 3.8.

Most users should install OpenSoundscape via pip: `pip install opensoundscape==0.7.0`. Contributors and advanced users can also use Poetry to install OpenSoundscape.

For more detailed instructions on how to install OpenSoundscape and use it in Jupyter, see the [documentation](http://opensoundscape.org).

# Features & Tutorials
OpenSoundscape includes functions to:
* load and manipulate audio files
* create and manipulate spectrograms
* train CNNs on spectrograms with PyTorch
* run pre-trained CNNs to detect vocalizations
* detect periodic vocalizations with RIBBIT
* load and manipulate Raven annotations

OpenSoundscape can also be used with our library of publicly available trained machine learning models for the detection of 500 common North American bird species.

For full API documentation and tutorials on how to use OpenSoundscape to work with audio and spectrograms, train machine learning models, apply trained machine learning models to acoustic data, and detect periodic vocalizations using RIBBIT, see the [documentation](http://opensoundscape.org).

# Quick Start

Using Audio and Spectrogram classes
```python
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram

#load an audio file and trim out a 5 second clip
my_audio = Audio.from_file("/path/to/audio.wav")
clip_5s = my_audio.trim(0,5)

#create a spectrogram and plot it
my_spec = Spectrogram.from_audio(clip_5s)
my_spec.plot()
```

Load audio starting at a real-world timestamp
```python
from datetime import datetime; import pytz

start_time = pytz.timezone('UTC').localize(datetime(2020,4,4,10,25))
audio_length = 5 #seconds  
path = '/path/to/audiomoth_file.WAV' #an AudioMoth recording

Audio.from_file(path, start_timestamp=start_time,duration=audio_length)
```

Using a pre-trained CNN to make predictions on long audio files
```python
from opensoundscape.torch.models.cnn import load_model

#get list of audio files
files = glob('./dir/*.WAV')

#generate predictions with a model
model = load_model('/path/to/saved.model')
scores, _, _ = model.predict(files)

#scores is a dataframe with MultiIndex: file, start_time, end_time
#containing inference scores for each class and each audio window
```

Training a CNN with labeled audio data
```python
from opensoundscape.torch.models.cnn import CNN
from sklearn.model_selection import train_test_split

#load a DataFrame of one-hot audio clip labels
df = pd.read_csv('my_labels.csv') #index: paths; columns: classes
train_df, validation_df = train_test_split(df,test_size=0.2)

#create a CNN and train on 2-second spectrograms for 2 epochs
model = CNN('resnet18',classes=df.columns,sample_duration=2.0)
model.train(
  train_df,
  validation_df,
  epochs=2
)
#the best model is automatically saved to a file `./best.model`
```
