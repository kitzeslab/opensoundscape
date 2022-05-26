import pandas as pd
import numpy as np
import torch
from pathlib import Path
import copy
import warnings

from opensoundscape.preprocess.utils import PreprocessingError
from opensoundscape.preprocess import actions
from opensoundscape.preprocess.actions import (
    Action,
    Overlay,
    AudioClipLoader,
    AudioTrim,
)

from opensoundscape.spectrogram import Spectrogram


class BasePreprocessor:
    """Class for defining an ordered set of Actions and a way to run them

    Custom Preprocessor classes should subclass this class or its children

    Preprocessors have one job: to transform samples from some input (eg
    a file path) to some output (eg a torch.Tensor) using a specific procedure.
    The procedure consists of Actions ordered by the Preprocessor's index.
    Preprocessors have a forward() method which runs the set of Actions
    specified in the index.

    Args:
        action_dict: dictionary of name:Action actions to perform sequentially
        sample_duration: length of audio samples to generate (seconds)
    """

    def __init__(self, sample_duration=None):
        self.pipeline = pd.Series({}, dtype=object)
        self.sample_duration = sample_duration

    def __repr__(self):
        return f"Preprocessor with pipeline: {self.pipeline}"

    def insert_action(self, action_index, action, after_key=None, before_key=None):
        """insert an action in specific specific position

        This is an in-place operation

        Inserts a new action before or after a specific key. If after_key and
        before_key are both None, action is appended to the end of the index.

        Args:
            action_index: string key for new action in index
            action: the action object, must be subclass of BaseAction
            after_key: insert the action immediately after this key in index
            before_key: insert the action immediately before this key in index
                Note: only one of (after_key, before_key) can be specified
        """
        if after_key is not None and before_key is not None:
            raise ValueError("Specifying both before_key and after_key is not allowed")

        assert not action_index in self.pipeline, (
            f"action_index must be unique, but {action_index} is already"
            "in the pipeline. Provide a different name for this action."
        )

        if after_key is None and before_key is None:
            # put this action at the end of the index
            self.pipeline = self.pipeline.append(pd.Series({action_index: action}))
        elif before_key is not None:
            self._insert_action_before(before_key, action_index, action)
        elif after_key is not None:
            self._insert_action_after(after_key, action_index, action)

    def remove_action(self, action_index):
        """alias for self.drop(...,inplace=True), removes an action

        This is an in-place operation

        Args:
            action_index: index of action to remove
        """
        self.pipeline.drop(action_index, inplace=True)

    def forward(
        self,
        sample,
        break_on_type=None,
        break_on_key=None,
        clip_times=None,
        bypass_augmentations=False,
    ):
        """perform actions in self.pipeline on a sample (until a break point)

        Actions with .bypass = True are skipped. Actions with .is_augmentation
        = True can be skipped by passing bypass_augmentations=True.

        Args:
            sample: either:
                - pd.Series with file path as index (.name) and labels
                - OR a file path as pathlib.Path or string
            break_on_type: if not None, the pipeline will be stopped when it
                reaches an Action of this class. The matching action is not
                performed.
            break_on_key: if not None, the pipeline will be stopped when it
                reaches an Action whose index equals this value. The matching
                action is not performed.
            bypass_augmentations: if True, actions with .is_augmentatino=True
                are skipped

        Returns:
            {'X':preprocessed sample, 'y':labels} if return_labels==True,
            otherwise {'X':preprocessed sample}

        """
        if break_on_key is not None:
            assert (
                break_on_key in self.pipeline
            ), f"break_on_key was {break_on_key} but no matching action found in pipeline"

        # handle paths or pd.Series as input for `sample`
        if type(sample) == str or issubclass(type(sample), Path):
            label_df_row = pd.Series(dtype=object, name=sample)
        else:
            assert type(sample) == pd.Series, (
                "sample must be pd.Series with "
                "path as .name OR file path (str or pathlib.Path), "
                f"was {type(sample)}"
            )
            label_df_row = sample

        # Series.name (dataframe index) contains a path to a file
        x = Path(label_df_row.name)

        # a list of additional variables that an action may request from the preprocessor
        sample_info = {
            "_path": Path(label_df_row.name),
            "_labels": copy.deepcopy(label_df_row),
            "_start_time": None if clip_times is None else clip_times["start_time"],
            "_sample_duration": self.sample_duration,
            "_preprocessor": self,
        }

        try:
            # perform each action in the pipeline
            for k, action in self.pipeline.items():
                if type(action) == break_on_type or k == break_on_key:
                    break
                if action.bypass:
                    continue
                if action.is_augmentation and bypass_augmentations:
                    continue
                extra_args = {key: sample_info[key] for key in action.extra_args}
                if action.returns_labels:
                    x, labels = action.go(x, **extra_args)
                    sample_info["_labels"] = labels
                else:
                    x = action.go(x, **extra_args)
        except:
            # treat any exceptions raised during forward as PreprocessingErrors
            raise PreprocessingError(
                f"failed to preprocess sample from path: {label_df_row.name}"
            )

        return x, sample_info

    def _insert_action_before(self, idx, name, value):
        """insert an item before a spcific index in a series"""
        i = list(self.pipeline.index).index(idx)
        part1 = self.pipeline[0:i]
        part2 = self.pipeline[i:]
        self.pipeline = part1.append(pd.Series([value], index=[name])).append(part2)

    def _insert_action_after(self, idx, name, value):
        """insert an item after a spcific index in a series"""
        i = list(self.pipeline.index).index(idx)
        part1 = self.pipeline[0 : i + 1]
        part2 = self.pipeline[i + 1 :]
        self.pipeline = part1.append(pd.Series([value], index=[name])).append(part2)


class SpectrogramPreprocessor(BasePreprocessor):
    """Child of BasePreprocessor that creates specrogram Tensors w/augmentation

    loads audio, creates spectrogram, performs augmentations, returns tensor

    by default, does not resample audio, but bandpasses to 0-11.025 kHz
    (to ensure all outputs have same scale in y-axis)
    can change with .load_audio.set(sample_rate=sr)

    during prediction, will load clips from long audio files rather than entire
    audio files.

    Args:
        sample_duration:
            length in seconds of audio samples generated
            If not None, longer clips trimmed to this length. By default,
            shorter clips will be extended (modify random_trim_audio and
            trim_audio to change behavior).
        overlay_df: if not None, will include an overlay action drawing
            samples from this df
        out_shape:
            output shape of tensor h,w,channels [default: [224,224,3]]
    """

    def __init__(self, sample_duration, overlay_df=None, out_shape=[224, 224, 3]):

        super(SpectrogramPreprocessor, self).__init__(sample_duration=sample_duration)

        # define a default set of Actions
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
                "overlay": Overlay(
                    is_augmentation=True, overlay_df=overlay_df, update_labels=False
                )
                if overlay_df is not None
                else None,
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

        # remove overlay if overlay_df was not specified
        if overlay_df is None:
            self.pipeline.drop("overlay", inplace=True)
