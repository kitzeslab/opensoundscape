import torch
import pandas as pd
import numpy as np
import tensorflow_hub

from opensoundscape.preprocess.preprocessors import AudioPreprocessor
from opensoundscape.ml.dataloaders import SafeAudioDataloader
from tqdm.autonotebook import tqdm
from opensoundscape.ml.cnn import BaseClassifier


def collate_to_np_array(audio_samples):
    """
    takes list of AudioSample objects with type(sample.data)==opensoundscape.Audio
    and returns np.array of shape [batch, length of audio signal]
    """
    try:
        return np.array([a.data.samples for a in audio_samples])
    except Exception as exc:
        raise ValueError(
            "Must pass list of AudioSample with Audio object as .data"
        ) from exc


class PerchDataLoader(SafeAudioDataloader):
    def __init__(self, *args, **kwargs):
        """Load samples with specific collate function

        Collate function takes list of AudioSample objects with type(.data)=opensoundscape.Audio
        and returns np.array of shape [batch, length of audio signal]

        Args:
            see SafeAudioDataloader
        """
        kwargs.update({"collate_fn": collate_to_np_array})
        super(PerchDataLoader, self).init(*args, **kwargs)


def google_perch(url="https://tfhub.dev/google/bird-vocalization-classifier/3"):
    """load TF model hub google Perch model, wrap in OpSo TensorFlowHubModel class

    Args:
        url to model path (default is Perch v3)

    Returns:
        opensoundscape.TensorFlowHubModel object with .predict() method for inference

    Methods:
        predict: get per-audio-clip per-class scores in dataframe format; includes WandB logging
        generate_embeddings: make embeddings for audio data (feature vectors from penultimate layer)
        generate_logits_and_embeddings: returns (logits, embeddings)
    """
    return GoogleBirdVocalizationClassifier(url)


class GoogleBirdVocalizationClassifier(BaseClassifier):
    def __init__(self, url="https://tfhub.dev/google/bird-vocalization-classifier/3"):
        """load TF model hub google Perch model, wrap in OpSo TensorFlowHubModel class

        Args:
            url to model path (default is Perch v3)

        Returns:
            opensoundscape.TensorFlowHubModel object with .predict() method for inference
        """
        self.network = tensorflow_hub.load(url)
        self.preprocessor = AudioPreprocessor(sample_duration=5, sample_rate=32000)
        self.inference_dataloader_cls = PerchDataLoader

        # class list
        label_csv = Path(__file__) / 
        self.classes = pd.read_csv(label_csv)["ebird2021"].values

    def __call__(
        self, dataloader, return_embeddings=False, return_logits=True, **kwargs
    ):
        """kwargs are passed to SafeAudioDataloader init (num_workers, batch_size, etc)"""

        if not return_logits and not return_embeddings:
            raise ValueError("Both return_logits and return_embeddings cannot be False")

        # iterate batches, running inference on each
        logits = []
        embeddings = []
        for batch in tqdm(dataloader):
            batch_logits, batch_embeddings = self.network.infer_tf(batch)
            logits.extend(batch_logits)
            embeddings.extend(batch_embeddings)

        if return_logits and return_embeddings:
            return embeddings, logits
        elif return_logits:
            return logits
        elif return_embeddings:
            return embeddings

    def generate_embeddings(self, samples, **kwargs):
        """Generate embeddings for audio data

        Args:
            samples: any of the following:
                - list of file paths
                - Dataframe with file as index
                - Dataframe with file, start_time, end_time of clips as index
            **kwargs: any arguments to SafeAudioDataloader

        Returns:
            list of embeddings
        """
        dataloader = self.inference_dataloader_cls(samples, self.preprocessor, **kwargs)
        return self(dataloader, return_embeddings=True, return_logits=False)

    def generate_logits_and_embeddings(self, samples, **kwargs):
        """Return (logits, embeddings) for audio data

        Args:
            samples: any of the following:
                - list of file paths
                - Dataframe with file as index
                - Dataframe with file, start_time, end_time of clips as index
            **kwargs: any arguments to SafeAudioDataloader
        """
        dataloader = self.inference_dataloader_cls(samples, self.preprocessor, **kwargs)
        return self(dataloader, return_embeddings=True, return_logits=True)
