__version__ = "0.10.0"

from . import annotations
from . import audio
from . import data_selection
from . import utils
from . import localization
from . import metrics
from . import ribbit
from . import sample
from . import signal_processing
from . import spectrogram
from . import ml
from . import preprocess
from . import logging

# expose some modules at the top level
from .ml import cnn
from .preprocess import preprocessors, actions

# expose some classes at the top level
from .audio import Audio
from .spectrogram import Spectrogram
from .ml.cnn import CNN, load_model
from .ml.datasets import AudioFileDataset, AudioSplittingDataset
from .preprocess.actions import Action
from .preprocess.preprocessors import SpectrogramPreprocessor
from .sample import AudioSample
from .annotations import BoxedAnnotations
