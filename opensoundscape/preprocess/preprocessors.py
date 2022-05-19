import pandas as pd
import numpy as np
import torch
from deprecated import deprecated
from pathlib import Path
import copy
import warnings

from opensoundscape.preprocess.utils import (
    PreprocessingError,
    _run_pipeline,
    insert_before,
    insert_after,
)
from opensoundscape.preprocess import actions
from opensoundscape.preprocess.actions import (
    BaseAction,
    Action,
    Overlay,
    AudioClipLoader,
    AudioTrim,
)

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
        sample_duration: duration of samples in seconds [default: None will
            load full length samples]

    Raises:
        PreprocessingError if exception is raised during __getitem__
    """

    def __init__(self, label_df, return_labels=True, sample_duration=None):
        # give helpful warnings for incorret inputs, but don't raise Exception
        if len(label_df) > 0 and not Path(label_df.index[0]).exists():
            warnings.warn(
                "Index of dataframe passed to "
                f"preprocessor must be a file path. Got {label_df.index[0]}."
            )
        if return_labels and len(label_df.columns) == 0:
            warnings.warn("return_labels=True but df has no columns!")
        elif (
            len(label_df) > 0 and return_labels and not label_df.values[0, 0] in (0, 1)
        ):
            warnings.warn(
                "if return_labels=True, label_df must have labels that take values of 0 and 1"
            )

        self.label_df = label_df
        self.clip_times_df = None
        self.return_labels = return_labels
        self.classes = label_df.columns
        self.sample_duration = sample_duration
        # if augmentation_on False, skips Actions with .is_augmentation=True
        self.augmentation_on = True

        # pipeline: a dictionary listing operations to conduct on each sample
        self.pipeline = pd.Series(dtype="object")

    def __len__(self):
        return self.label_df.shape[0]

    def __getitem__(self, item_idx, break_on_key=None, break_on_type=None):

        label_df_row = self.label_df.iloc[item_idx]

        clip_times = (
            None if self.clip_times_df is None else self.clip_times_df.iloc[item_idx]
        )

        # _run_pipeline will raise PreprocessingError if something fails
        x, sample_info = _run_pipeline(
            self.pipeline,
            label_df_row,
            augmentation_on=self.augmentation_on,
            break_on_key=break_on_key,
            break_on_type=break_on_type,
            clip_times=clip_times,
            sample_duration=self.sample_duration,
        )

        # Return sample & label pairs (training/validation)
        if self.return_labels:
            labels = torch.from_numpy(sample_info["_labels"].values)
            return {"X": x, "y": labels}

        # Return sample only (prediction)
        return {"X": x}

    def __repr__(self):
        return f"{self.__class__} object with pipeline: {self.pipeline}"

    def class_counts(self):
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
            new_ds.clip_times_df = new_ds.clip_times_df.loc[new_ds.label_df.index]
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
            new_ds.clip_times_df = new_ds.clip_times_df.loc[new_ds.label_df.index]
        return new_ds

    def insert_action(self, action_index, action, after_key=None, before_key=None):
        """insert an action into the pipeline in specific specific position

        This is an in-place operation

        Inserts a new action before or after a specific key. If after_key and
        before_key are both None, action is appended to the end of the pipeline.

        Args:
            action_index: string key for new action in pipeline dictionary
            action: the action object, must be subclass of BaseAction
            after_key: insert the action immediately after this key in pipeline
            before_key: insert the action immediately before this key in pipeline
                Note: only one of (after_key, before_key) can be specified
        """
        if after_key is not None and before_key is not None:
            raise ValueError("Specifying both before_key and after_key is not allowed")

        assert (
            not action_index in self.pipeline.index
        ), f"action_index must be unique, but {action_index} is already in the pipeline. Provide a different name for this action."

        if after_key is None and before_key is None:
            # put this action at the end of the pipeline
            self.pipeline[action_index] = action
        elif before_key is not None:
            self.pipeline = insert_before(
                self.pipeline, after_key, action_index, action
            )
        elif after_key is not None:
            self.pipeline = insert_after(self.pipeline, after_key, action_index, action)

    def remove_action(self, action_index):
        """remove an action into the pipeline based on its index (name)

        This is an in-place operation

        Args:
            action_index: index of action to remove in pipeline
        """
        self.pipeline = self.pipeline.drop(action_index)

    def make_dataset(self, label_df):
        """Generates a copy of itself, with `self.label_df = label_df`"""
        new_dataset = self.sample(n=0)
        new_dataset.label_df = label_df
        return new_dataset


class SpecPreprocessor(BasePreprocessor):
    """Child of BasePreprocessor that creates specrograms with augmentation

    loads audio, creates spectrogram, performs augmentations, returns tensor

    by default, does not resample audio, but bandpasses to 0-11.025 kHz
    (to ensure all outputs have same scale in y-axis)
    can change with .actions.load_audio.set(sample_rate=sr)

    during prediction, will load clips from long audio files rather than entire
    audio files.

    Args:
        label_df:
            dataframe of audio clips. label_df must have audio paths in the index.
            If label_df has labels, the class names should be the columns, and
            the values of each row should be 0 or 1.
            If data does not have labels, label_df will have no columns
        sample_duration:
            length in seconds of audio samples generated
            If not None, longer clips trimmed to this length. By default,
            shorter clips will be extended (modify random_trim_audio and
            trim_audio to change behavior).
        out_shape:
            output shape of tensor h,w,channels [default: [224,224,3]]
        return_labels:
            if True, the __getitem__ method will return {X:sample,y:labels}
            If False, the __getitem__ method will return {X:sample}
            If label_df has no labels (no columns), use return_labels=False
            [default: True]
    """

    def __init__(
        self,
        label_df,
        sample_duration,
        return_labels=True,
        overlay_df=None,
        out_shape=[224, 224, 3],
    ):
        self.sample_duration = sample_duration
        super(SpecPreprocessor, self).__init__(
            label_df, return_labels=return_labels, sample_duration=sample_duration
        )
        self.pipeline = pd.Series(
            {
                "load_audio": AudioClipLoader(),
                # if we are augmenting and get a long file, take a random trim from it
                "random_trim_audio": AudioTrim(is_augmentation=True, random_trim=True),
                # otherwise, we expect to get the correct duration. no random trim
                "trim_audio": AudioTrim(),  # trim or extend (w/silence) clips to correct length
                "to_spec": Action(Spectrogram.from_audio),
                "bandpass": Action(
                    Spectrogram.bandpass, min_f=0, max_f=11025, out_of_bounds_ok=False
                ),
                "to_img": Action(
                    Spectrogram.to_image,
                    shape=out_shape[0:2],
                    channels=out_shape[2],
                    return_type="torch",
                ),
                "time_mask": Action(actions.time_mask, is_augmentation=True),
                "frequency_mask": Action(actions.frequency_mask, is_augmentation=True),
                "add_noise": Action(
                    actions.tensor_add_noise, is_augmentation=True, std=0.005
                ),
                "rescale": Action(actions.scale_tensor),
                "random_affine": Action(
                    actions.torch_random_affine, is_augmentation=True
                ),
            }
        )
        # add overlay augmentation if overlay_df is provided
        if overlay_df is not None:
            overlay = Overlay(
                is_augmentation=True, overlay_df=overlay_df, update_labels=False
            )
            self.insert_action("overlay", overlay, after_key="to_img")
