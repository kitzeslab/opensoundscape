
# Quick Start

A guide to the most commonly used features of OpenSoundscape.

## Using Audio and Spectrogram classes
```python
from opensoundscape import Audio, Spectrogram

#load an audio file and trim out a 5 second clip
my_audio = Audio.from_file("/path/to/audio.wav")
clip_5s = my_audio.trim(0,5)

#create a spectrogram and plot it
my_spec = Spectrogram.from_audio(clip_5s)
my_spec.plot()
```

## Load audio starting at a real-world timestamp
```python
from datetime import datetime; import pytz

start_time = pytz.timezone('UTC').localize(datetime(2020,4,4,10,25))
audio_length = 5 #seconds  
path = '/path/to/audiomoth_file.WAV' #an AudioMoth recording

Audio.from_file(path, start_timestamp=start_time,duration=audio_length)
```

## Using a pre-trained CNN to make predictions on long audio files
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

## Training a CNN using audio files and Raven annotations 
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

## Training a CNN with labeled audio data (one label per audio file):
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
