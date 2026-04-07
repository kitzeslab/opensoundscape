__version__ = "0.12.1"

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
from . import sample_data
from . import vector_database
from . import visualization

# expose some modules at the top level
from .ml import cnn, cnn_architectures, shallow_classifier
from .preprocess import preprocessors, actions

# expose some classes at the top level
from .audio import Audio
from .sample_data import birds, birds_path
from .spectrogram import Spectrogram, MelSpectrogram

from .ml.cnn import SpectrogramClassifier, CNN, load_model
from .ml.onnx_model import ONNXModel
from .ml.lightning import LightningSpectrogramModule
from .ml.datasets import AudioFileDataset, AudioFileDataset
from .preprocess.actions import Action
from .preprocess.preprocessors import SpectrogramPreprocessor, AudioPreprocessor
from .sample import AudioSample
from .annotations import BoxedAnnotations
from .preprocess.utils import show_tensor, show_tensor_grid
from .localization import SpatialEvent, SynchronizedRecorderArray
from .utils import set_seed
from .ml.shallow_classifier import MLPClassifier
