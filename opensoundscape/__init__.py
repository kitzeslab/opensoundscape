__version__ = "0.8.0"

from . import annotations
from . import audio_tools
from . import audio
from . import audiomoth
from . import data_selection
from . import helpers
from . import localization
from . import metrics
from . import ribbit
from . import signal_processing
from . import spectrogram
from . import taxa
from . import torch
from . import preprocess
from . import resources
from . import wandb

# expose some modules at the top level
from .torch.models import cnn
from .preprocess import preprocessors, actions

# expose some classes at the top level
from .audio import Audio
from .spectrogram import Spectrogram
from .torch.models.cnn import CNN
from .torch.datasets import AudioFileDataset, AudioSplittingDataset
from .preprocess.actions import Action
from .preprocess.preprocessors import SpectrogramPreprocessor
