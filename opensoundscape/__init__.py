__version__ = "0.11.0"

from . import annotations
from . import audio
from . import data_selection
from . import utils
from . import metrics
from . import ribbit
from . import sample
from . import signal_processing
from . import spectrogram
from . import ml
from . import preprocess
from . import logging
from . import localization

# expose some modules at the top level
from .ml import cnn, bioacoustics_model_zoo, cnn_architectures
from .preprocess import preprocessors, actions

# expose some classes at the top level
from .audio import Audio
from .spectrogram import Spectrogram, MelSpectrogram

from .ml.cnn import SpectrogramClassifier, CNN
from .ml.lightning import LightningSpectrogramModule
from .ml.datasets import AudioFileDataset, AudioSplittingDataset
from .preprocess.actions import Action
from .preprocess.preprocessors import SpectrogramPreprocessor, AudioPreprocessor
from .sample import AudioSample
from .annotations import BoxedAnnotations
from .preprocess.utils import show_tensor, show_tensor_grid
from .localization import SpatialEvent, SynchronizedRecorderArray
from .utils import set_seed
