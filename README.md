[![CI Status](https://github.com/kitzeslab/opensoundscape/workflows/CI/badge.svg)](https://github.com/kitzeslab/opensoundscape/actions?query=workflow%3ACI)
[![Documentation Status](https://readthedocs.org/projects/opensoundscape/badge/?version=latest)](http://opensoundscape.org/en/latest/?badge=latest)

OpenSoundscape (OPSO) is free and open source Python utility library analyzing bioacoustic data. 

OpenSoundscape includes utilities which can be strung together to create data analysis pipelines, including functions to:

* load and manipulate audio files
* create and manipulate spectrograms
* train convolutional neural networks (CNNs) on spectrograms with PyTorch
* run pre-trained CNNs to detect vocalizations
* detect periodic vocalizations with RIBBIT
* load and manipulate Raven annotations
* estimate the location of sound sources from synchronized recordings


OpenSoundscape's source code can be found on [GitHub] (https://github.com/kitzeslab/opensoundscape) and its documentation can be found on [OpenSoundscape.org](https://opensoundscape.org).

## Show me the code!

For examples of how to use OpenSoundscape, see the [Quick Start Guide](#quick-start-guide) below.

For full API documentation and tutorials on how to use OpenSoundscape to work with audio and spectrograms, train machine learning models, apply trained machine learning models to acoustic data, and detect periodic vocalizations using RIBBIT, see the [documentation](http://opensoundscape.org).


## Contact & Citation

OpenSoundcape is developed and maintained by the [Kitzes Lab](http://www.kitzeslab.org/) at the University of Pittsburgh. It is currently in active development. If you find a bug, please [submit an issue](https://github.com/kitzeslab/opensoundscape/issues) on the GitHub repository. If you have another question about OpenSoundscape, please use the (OpenSoundscape Discussions board)[https://github.com/kitzeslab/opensoundscape/discussions] or email Sam Lapp (`sam.lapp at pitt.edu`)


Suggested citation:

    Lapp, Sam; Rhinehart, Tessa; Freeland-Haynes, Louis; 
    Khilnani, Jatin; Syunkova, Alexandra; Kitzes, Justin. 
    “OpenSoundscape: An Open-Source Bioacoustics Analysis Package for Python.” 
    Methods in Ecology and Evolution 2023. https://doi.org/10.1111/2041-210X.14196.


## Quick Start Guide

A guide to the most commonly used features of OpenSoundscape.


### Installation

Details about installation are available on the OpenSoundscape documentation at [OpenSoundscape.org](https://opensoundscape.org). FAQs:

#### How do I install OpenSoundscape?

* Most users should install OpenSoundscape via pip, preferably within a virtual environment: `pip install opensoundscape==0.10.1`. 
* To use OpenSoundscape in Jupyter Notebooks (e.g. for tutorials), follow the installation instructions for your operating system, then follow the "Jupyter" instructions.
* Contributors and advanced users can also use Poetry to install OpenSoundscape using the "Contributor" instructions

#### Will OpenSoundscape work on my machine?

* OpenSoundscape can be installed on Windows, Mac, and Linux machines.
* It has been tested on Python 3.9, 3.10, and 3.11.
* For Apple Silicon (M1 chip) users, Python >=3.9 is recommended and may be required to avoid dependency issues.
* Most computer cluster users should follow the Linux installation instructions


### Use Audio and Spectrogram classes
```python
from opensoundscape import Audio, Spectrogram

#load an audio file and trim out a 5 second clip
my_audio = Audio.from_file("/path/to/audio.wav")
clip_5s = my_audio.trim(0,5)

#create a spectrogram and plot it
my_spec = Spectrogram.from_audio(clip_5s)
my_spec.plot()
```

### Load audio starting at a real-world timestamp
```python
from datetime import datetime; import pytz

start_time = pytz.timezone('UTC').localize(datetime(2020,4,4,10,25))
audio_length = 5 #seconds  
path = '/path/to/audiomoth_file.WAV' #an AudioMoth recording

Audio.from_file(path, start_timestamp=start_time,duration=audio_length)
```

### Use a pre-trained CNN to make predictions on long audio files
```python
from opensoundscape import load_model

#get list of audio files
files = glob('./dir/*.WAV')

#generate predictions with a model
model = load_model('/path/to/saved.model')
scores = model.predict(files)

#scores is a dataframe with MultiIndex: file, start_time, end_time
#containing inference scores for each class and each audio window
```

### Train a CNN using audio files and Raven annotations 
```python
from sklearn.model_selection import train_test_split
from opensoundscape import BoxedAnnotations, CNN

# assume we have a list of raven annotation files and corresponding audio files
# load the annotations into OpenSoundscape
all_annotations = BoxedAnnotations.from_raven_files(raven_file_paths,audio_file_paths)

# pick classes to train the model on. These should occur in the annotated data
class_list = ['IBWO','BLJA']

# create labels for fixed-duration (2 second) clips 
labels = all_annotations.one_hot_clip_labels(
  cip_duration=2,
  clip_overlap=0,
  min_label_overlap=0.25,
  class_subset=class_list
)

# split the labels into training and validation sets
train_df, validation_df = train_test_split(labels, test_size=0.3)

# create a CNN and train on the labeled data
model = CNN(architecture='resnet18', sample_duration=2, classes=class_list)
model.train(train_df, validation_df, epochs=20, num_workers=8, batch_size=256)
```

### Train a CNN with labeled audio data (one label per audio file):
```python
from opensoundscape import CNN
from sklearn.model_selection import train_test_split

#load a DataFrame of one-hot audio clip labels
df = pd.read_csv('my_labels.csv') #index: paths; columns: classes
train_df, validation_df = train_test_split(df,test_size=0.2)

#create a CNN and train on 2-second spectrograms for 20 epochs
model = CNN('resnet18', classes=df.columns, sample_duration=2.0)
model.train(train_df, validation_df, epochs=20)
#the best model is automatically saved to a file `./best.model`
```
