import pandas as pd
import numpy as np
import torch
from deprecated import deprecated
from pathlib import Path
import copy
import warnings

from opensoundscape.preprocess.utils import PreprocessingError, _run_pipeline
from opensoundscape.preprocess import actions
from opensoundscape.preprocess.actions import (
    BaseAction,
    Action,
    Augmentation,
    ImgOverlay,
)

# from opensoundscape.preprocess.overlay import ImgOverlay
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram


class BasePreprocessor(torch.utils.data.Dataset):
    """Base class for Preprocessing pipelines (use in place of torch Dataset)

    Custom Preprocessor classes should subclass this class or its children

    Args:
        label_df:
            dataframe of audio clips. label_df must have audio paths in the index.
            If label_df has labels, the class names should be the columns, and
            the values of each row should be 0 or 1.
            If data does not have labels, label_df will have no columns
        return_labels:
            if True, the __getitem__ method will return {X:sample,y:labels}
            If False, the __getitem__ method will return {X:sample}
            If label_df has no labels (no columns), use return_labels=False
            [default: True]

    Raises:
        PreprocessingError if exception is raised during __getitem__
    """

    def __init__(self, label_df, clip_times_df=None, return_labels=True):

        # give helpful warnings for incorret inputs, but don't raise Exception
        if not Path(label_df.index[0]).exists():
            warnings.warn(
                "Index of dataframe passed to "
                f"preprocessor must be a file path. Got {label_df.index[0]}."
            )
        if return_labels and not label_df.values[0, 0] in (0, 1):
            warnings.warn(
                "if return_labels=True, label_df must have labels that take values of 0 and 1"
            )

        self.label_df = label_df
        self.clip_times_df = clip_times_df
        self.return_labels = return_labels
        self.classes = label_df.columns
        self.perform_augmentations = (
            True
        )  # if False, Actions that subclass Augmentation are skipped

        # pipeline: a dictionary listing operations to conduct on each sample
        self.pipeline = {}

    def __len__(self):
        return self.label_df.shape[0]

    def __getitem__(self, item_idx, break_on_key=None, break_on_type=None):

        try:
            label_df_row = self.label_df.iloc[item_idx]

            x, sample_info = _run_pipeline(
                self.pipeline,
                label_df_row,
                perform_augmentations=self.perform_augmentations,
                break_on_key=break_on_key,
                break_on_type=break_on_type,
            )

            # Return sample & label pairs (training/validation)
            if self.return_labels:
                labels = torch.from_numpy(sample_info["_labels"].values)
                return {"X": x, "y": labels}

            # Return sample only (prediction)
            return {"X": x}

        except:
            raise PreprocessingError(
                f"failed to preprocess sample from path: {self.label_df.index[item_idx]}"
            )

    def class_counts_cal(self):
        """count number of each label"""
        labels = self.label_df.columns
        counts = np.sum(self.label_df.values, 0)
        return labels, counts

    def sample(self, **kwargs):
        """out-of-place random sample

        creates copy of object with n rows randomly sampled from dataframe

        Args: see pandas.DataFrame.sample()

        Returns:
            a new dataset object
        """
        new_ds = copy.deepcopy(self)
        new_ds.label_df = new_ds.label_df.sample(**kwargs)
        if new_ds.clip_times_df is not None:
            new_ds.clip_times_df = new_ds.clip_times_df[new_ds.label_df.index]
        return new_ds

    def head(self, n=5):
        """out-of-place copy of first n samples

        performs df.head(n) on self.label_df

        Args:
            n: number of first samples to return, see pandas.DataFrame.head()
            [default: 5]

        Returns:
            a new dataset object
        """
        new_ds = copy.deepcopy(self)
        new_ds.label_df = new_ds.label_df.head(n)
        if new_ds.clip_times_df is not None:
            new_ds.clip_times_df = new_ds.clip_times_df[new_ds.label_df.index]
        return new_ds


class AudioLoadingPreprocessor(BasePreprocessor):
    """creates Audio objects from file paths

    Args:
        label_df:
            dataframe of audio clips. label_df must have audio paths in the index.
            If label_df has labels, the class names should be the columns, and
            the values of each row should be 0 or 1.
            If data does not have labels, label_df will have no columns
        return_labels:
            if True, __getitem__ returns {"X":batch_tensors,"y":labels}
            if False, __getitem__ returns {"X":batch_tensors}
            [default: True]
        audio_length:
            length in seconds of audio to return
            - None: do not trim the original audio
            - seconds (float): trim longer audio to this length. Shorter
            audio input will raise a ValueError.
    """

    def __init__(self, label_df, return_labels=True, audio_length=None):

        super(AudioLoadingPreprocessor, self).__init__(
            label_df, return_labels=return_labels
        )
        self.audio_length = (
            audio_length
        )  # TODO: may get out of date with trim_audio.params.audio_length

        # TODO: should the Action get parameters from the function as attributes or put them in .params?
        self.pipeline = {
            "load_audio": Action(Audio.from_file),
            "trim_audio": Action(actions.trim_audio, audio_length=self.audio_length),
        }


class AudioToSpectrogramPreprocessor(BasePreprocessor):
    """
    loads audio paths, creates spectrogram, returns tensor

    by default, does not resample audio, but bandpasses to 0-11025 Hz
    (to ensure all outputs have same scale in y-axis)
    can change with .actions.load_audio.set(sample_rate=sr)

    Args:
        label_df:
            dataframe of audio clips. label_df must have audio paths in the index.
            If label_df has labels, the class names should be the columns, and
            the values of each row should be 0 or 1.
            If data does not have labels, label_df will have no columns
        audio_length:
            length in seconds of audio clips [default: None]
            If provided, longer clips trimmed to this length. By default,
            shorter clips will not be extended (modify actions.AudioTrimmer
            to change behavior).
        out_shape:
            output shape of tensor in pixels [default: [224,224]]
        return_labels:
            if True, the __getitem__ method will return {X:sample,y:labels}
            If False, the __getitem__ method will return {X:sample}
            If label_df has no labels (no columns), use return_labels=False
            [default: True]
    """

    def __init__(
        self, label_df, audio_length=None, out_shape=[224, 224], return_labels=True
    ):

        super(AudioToSpectrogramPreprocessor, self).__init__(
            label_df, return_labels=return_labels
        )

        self.audio_length = audio_length
        self.return_labels = return_labels

        self.pipeline = {
            "load_audio": Action(Audio.from_file),
            "trim_audio": Action(actions.trim_audio, audio_length=self.audio_length),
            "to_spec": Action(Spectrogram.from_audio),
            "bandpass": Action(
                Spectrogram.bandpass, min_f=0, max_f=11025, out_of_bounds_ok=False
            ),
            "to_img": Action(Spectrogram.to_image, shape=out_shape),
            "to_tensor": Action(actions.image_to_tensor),
            "normalize": Action(actions.tensor_normalize),
        }


class AlphaSampleGenerator(BasePreprocessor):
    """Child of BasePreprocessor with full augmentation pipeline

    loads audio, creates spectrogram, performs augmentations, returns tensor

    by default, does not resample audio, but bandpasses to 0-11.025 kHz
    (to ensure all outputs have same scale in y-axis)
    can change with .actions.load_audio.set(sample_rate=sr)

    Args:
        label_df:
            dataframe of audio clips. label_df must have audio paths in the index.
            If label_df has labels, the class names should be the columns, and
            the values of each row should be 0 or 1.
            If data does not have labels, label_df will have no columns
        audio_length:
            length in seconds of audio clips [default: None]
            If provided, longer clips trimmed to this length. By default,
            shorter clips will not be extended (modify actions.AudioTrimmer
            to change behavior).
        out_shape:
            output shape of tensor in pixels [default: [224,224]]
        return_labels:
            if True, the __getitem__ method will return {X:sample,y:labels}
            If False, the __getitem__ method will return {X:sample}
            If label_df has no labels (no columns), use return_labels=False
            [default: True]
        # debug:
        #     If a path is provided, generated samples (after all augmentation)
        #     will be saved to the path as an image. This is useful for checking
        #     that the sample provided to the model matches expectations.
        #     [default: None]
    """

    def __init__(
        self,
        label_df,
        audio_length=None,
        return_labels=True,
        # debug=None,
        overlay_df=None,
        out_shape=[224, 224],
    ):

        super(AlphaSampleGenerator, self).__init__(
            label_df, return_labels=return_labels
        )
        self.audio_length = audio_length  # TODO can get out of date

        self.pipeline = {
            "load_audio": Action(Audio.from_file),
            "trim_audio": Action(actions.trim_audio, audio_length=self.audio_length),
            "to_spec": Action(Spectrogram.from_audio),
            "bandpass": Action(
                Spectrogram.bandpass, min_f=0, max_f=11025, out_of_bounds_ok=False
            ),
            "to_img": Action(Spectrogram.to_image, shape=out_shape),
            "overlay": BaseAction()
            if overlay_df is None
            else ImgOverlay(overlay_df=overlay_df, update_labels=False),
            "to_tensor": Action(actions.image_to_tensor),
            "time_mask": Augmentation(actions.time_mask),
            "frequency_mask": Augmentation(actions.frequency_mask),
            "add_noise": Augmentation(actions.tensor_add_noise, std=0.005),
            "normalize": Action(actions.tensor_normalize),
            "random_affine": Augmentation(actions.torch_random_affine),
        }
        # TODO: remove overlay action if overlay_df is None?


class PredictionPreprocessor(BasePreprocessor):
    """load audio samples from long audio files

    Directly loads a part of an audio file, eg 5-10 seconds, without loading
    entire file. This alows for prediction on long audio files without needing to
    pre-split or load large files into memory.

    It will load the requested audio segments into samples, regardless of length

    Args:
        clip_times_df: a dataframe with file paths as index and 2 columns:
            ['start_time','end_time'] (seconds since beginning of file)
    Returns:
        ClipLoadingSpectrogramPreprocessor object

    Examples:
    You can quickly create such a df for a set of audio files like this:

    ```
    update this #TODO
    ```

    If you use this preprocessor with model.predict(), it will work, but
    the scores/predictions df will only have file paths not the times of clips.
    You will want to re-add the start and end times of clips as multi-index:

    ```
    score_df = model.predict(clip_loading_ds) #for instance
    score_df.index = pd.MultiIndex.from_arrays(
        [clip_df.index,clip_df['start_time'],clip_df['end_time']]
    )
    ```
    """

    def __init__(self, clip_times_df):
        assert clip_times_df.columns[0] == "start_time"
        assert clip_times_df.columns[1] == "end_time"
        super(ClipLoadingSpectrogramPreprocessor, self).__init__(
            clip_times_df[[]], return_labels=False, clip_times_df=clip_times_df
        )

        self.pipeline = {
            "load_audio": AudioClipLoader(),
            "trim_audio": Action(actions.trim_audio, audio_length=self.audio_length),
            "to_spec": Action(Spectrogram.from_audio),
            "bandpass": Action(
                Spectrogram.bandpass, min_f=0, max_f=11025, out_of_bounds_ok=False
            ),
            "to_img": Action(Spectrogram.to_image, shape=out_shape),
            "to_tensor": Action(actions.image_to_tensor),
            "normalize": Action(actions.tensor_normalize),
        }

    def __getitem__(self, item_idx):

        try:
            clip_times = self.clip_times_df.iloc[item_idx]
            label_df_row = self.clip_times_df[[]].iloc[item_idx]  # no columns
            x, sample_info = _run_pipeline(
                self.pipeline,
                label,
                perform_augmentations=self.perform_augmentations,
                clip_times=clip_times,
            )
            return {"X": x}
        except:
            raise PreprocessingError(
                f"failed to preprocess sample: {self.df.index[item_idx]}"
            )
