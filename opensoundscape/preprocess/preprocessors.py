"""Preprocessor classes: tools for preparing and augmenting audio samples"""
from pathlib import Path
import pandas as pd
import copy

from opensoundscape.preprocess import actions
from opensoundscape.preprocess.actions import (
    Action,
    Overlay,
    AudioClipLoader,
    AudioTrim,
    SpectrogramToTensor,
)
from opensoundscape.preprocess.utils import PreprocessingError
from opensoundscape.spectrogram import Spectrogram
from opensoundscape.sample import AudioSample


class BasePreprocessor:
    """Class for defining an ordered set of Actions and a way to run them

    Custom Preprocessor classes should subclass this class or its children

    Preprocessors have one job: to transform samples from some input (eg
    a file path) to some output (eg an AudioSample with .data as torch.Tensor)
    using a specific procedure defined by the .pipeline attribute.
    The procedure consists of Actions ordered by the Preprocessor's .pipeline.
    Preprocessors have a forward() method which sequentially applies the Actions
    in the pipeline to produce a sample.

    Args:
        action_dict: dictionary of name:Action actions to perform sequentially
        sample_duration: length of audio samples to generate (seconds)
    """

    def __init__(self, sample_duration=None):
        self.pipeline = pd.Series({}, dtype=object)
        self.sample_duration = sample_duration

    def __repr__(self):
        return f"Preprocessor with pipeline:\n{self.pipeline}"

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
            new_item = pd.Series({action_index: action})
            self.pipeline = pd.concat([self.pipeline, new_item])
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
        bypass_augmentations=False,
        trace=False,
    ):
        """perform actions in self.pipeline on a sample (until a break point)

        Actions with .bypass = True are skipped. Actions with .is_augmentation
        = True can be skipped by passing bypass_augmentations=True.

        Args:
            sample: any of
                - (path, start time) tuple
                - pd.Series with (file, start_time, end_time) as .name
                (eg index of a pd.DataFrame from which row was taken)
                - AudioSample object
            break_on_type: if not None, the pipeline will be stopped when it
                reaches an Action of this class. The matching action is not
                performed.
            break_on_key: if not None, the pipeline will be stopped when it
                reaches an Action whose index equals this value. The matching
                action is not performed.
            clip_times: can be either
                - None: the file is treated as a single sample
                - dictionary {"start_time":float,"end_time":float}:
                    the start and end time of clip in audio
            bypass_augmentations: if True, actions with .is_augmentatino=True
                are skipped
            trace (boolean - default False): if True, saves the output of each pipeline step in the `sample_info` output argument - should be utilized for analysis/debugging on samples of interest

        Returns:
            sample (instance of AudioSample class)

        """
        if break_on_key is not None:
            assert (
                break_on_key in self.pipeline
            ), f"break_on_key was {break_on_key} but no matching action found in pipeline"

        # create AudioSample from input path
        sample = self._generate_sample(sample)
        if trace:
            sample.trace = pd.Series(index=self.pipeline.index)

        # run the pipeline by performing each Action on the AudioSample
        try:
            # perform each action in the pipeline
            for k, action in self.pipeline.items():
                if type(action) == break_on_type or k == break_on_key:
                    if trace:
                        # saved "output" of this step informs user pipeline was stopped
                        sample.trace[k] = f"## Pipeline terminated ## {sample.trace[k]}"
                    break
                if action.bypass:
                    continue
                if action.is_augmentation and bypass_augmentations:
                    if trace:
                        sample.trace[k] = f"## Bypassed ## {sample.trace[k]}"
                    continue

                # perform the action (modifies the AudioSample in-place)
                action.go(sample)

                if trace:  # user requested record of preprocessing steps
                    # save the current state of the sample's data
                    # (trace is a Series with index matching self.pipeline)
                    try:
                        sample.trace[k] = copy.deepcopy(sample.data)
                    # this will fail on Spectrogram and Audio class, which are immutable, implemented by
                    # raising an AttributeError if .__setattr__ is called. Since deepcopy calls setattr,
                    # we can't deepcopy those. As a temporary fix, we can add the original object because
                    # it is immutable. However, we should re-factor immmutable classes to avoid this issue
                    # (see Issue #671)
                    except AttributeError:
                        sample.trace[k] = sample.data

        except Exception as exc:
            # treat any exceptions raised during forward as PreprocessingErrors
            raise PreprocessingError(
                f"failed to preprocess sample from path: {sample.source}"
            ) from exc

        # remove temporary attributes from sample
        del sample.preprocessor, sample.target_duration
        return sample

    def _generate_sample(self, sample):
        """create AudioSample object from initial input: any of
            (path, start time) tuple
            pd.Series with (file, start_time, end_time) as .name
                (eg index of a pd.DataFrame from which row was taken)
            AudioSample object

        can override this method in subclasses to modify how samples
        are created, or to add additional attributes to samples
        """
        # handle paths or pd.Series as input for `sample`
        if isinstance(sample, tuple):
            path, start = sample
            assert isinstance(
                path, (str, Path)
            ), "if passing tuple, first element must be str or pathlib.Path"
            sample = AudioSample(path, start_time=start, duration=self.sample_duration)
        elif isinstance(sample, pd.Series):
            # .name should contain (path, start_time, end_time)
            # note: end is not used, uses start_time self.sample_duration
            path, start, _ = sample.name
            assert isinstance(
                path, (str, Path)
            ), "if passing a series, series.name must contain (path, start_time, end_time)"
            sample = AudioSample(path, start_time=start, duration=self.sample_duration)
        else:
            assert isinstance(sample, AudioSample), (
                "sample must be AudioSample, tuple of (path, start_time), "
                "or pd.Series with (path, start_time, end_time) as .name. "
                f"was {type(sample)}"
            )
            pass  # leave it as an AudioSample

        # add attributes to the sample that might be needed by actions in the pipeline
        sample.preprocessor = self
        sample.target_duration = self.sample_duration

        return sample

    def _insert_action_before(self, idx, name, value):
        """insert an item before a spcific index in a series"""
        i = list(self.pipeline.index).index(idx)
        part1 = self.pipeline[0:i]
        new_item = pd.Series([value], index=[name])
        part2 = self.pipeline[i:]
        self.pipeline = pd.concat([part1, new_item, part2])

    def _insert_action_after(self, idx, name, value):
        """insert an item after a spcific index in a series"""
        i = list(self.pipeline.index).index(idx)
        part1 = self.pipeline[0 : i + 1]
        new_item = pd.Series([value], index=[name])
        part2 = self.pipeline[i + 1 :]
        self.pipeline = pd.concat([part1, new_item, part2])


class SpectrogramPreprocessor(BasePreprocessor):
    """Child of BasePreprocessor that creates specrogram Tensors w/augmentation

    loads audio, creates spectrogram, performs augmentations, creates tensor

    by default, does not resample audio, but bandpasses to 0-11.025 kHz
    (to ensure all outputs have same scale in y-axis)
    can change with .pipeline.bandpass.set(min_f=,max_f=)

    Args:
        sample_duration:
            length in seconds of audio samples generated
            If not None, longer clips are trimmed to this length. By default,
            shorter clips will be extended (modify random_trim_audio and
            trim_audio to change behavior).

        overlay_df: if not None, will include an overlay action drawing
            samples from this df
        height: height of output sample (frequency axis)
            - default None will use the original height of the spectrogram
        width: width of output sample (time axis)
            -  default None will use the originalwidth of the spectrogram
        channels: number of channels in output sample (default 1)
    """

    def __init__(
        self, sample_duration, overlay_df=None, height=None, width=None, channels=1
    ):
        super(SpectrogramPreprocessor, self).__init__(sample_duration=sample_duration)
        self.height = height
        self.width = width
        self.channels = channels

        # define a default set of Actions
        # each Action's .go() method is called during preprocessing
        # the .go() method takes an AudioSample object as an argument
        # and modifies it _in place_.
        self.pipeline = pd.Series(
            {
                # load a segment of an audio file into an Audio object
                # references AudioSample attributes: start_time and duration
                "load_audio": AudioClipLoader(),
                # if we are augmenting and get a long file, take a random trim from it
                "random_trim_audio": AudioTrim(is_augmentation=True, random_trim=True),
                # otherwise, we expect to get the correct duration. no random trim
                "trim_audio": AudioTrim(),  # trim or extend (w/silence) clips to correct length
                # convert Audio object to Spectrogram
                "to_spec": Action(Spectrogram.from_audio),
                # bandpass to 0-11.025 kHz (to ensure all outputs have same scale in y-axis)
                "bandpass": Action(
                    Spectrogram.bandpass, min_f=0, max_f=11025, out_of_bounds_ok=False
                ),
                # convert Spectrogram to torch.Tensor and re-size to desired output shape
                # references AudioSample attributes: target_height, target_width, target_channels
                "to_tensor": SpectrogramToTensor(),  # uses sample.target_shape
                ##  augmentations ##
                # Overlay is a version of "mixup" that draws samples from a user-specified dataframe
                # and overlays them on the current sample
                "overlay": Overlay(
                    is_augmentation=True, overlay_df=overlay_df, update_labels=False
                )
                if overlay_df is not None
                else None,
                # add vertical (time) and horizontal (frequency) masking bars
                "time_mask": Action(actions.time_mask, is_augmentation=True),
                "frequency_mask": Action(actions.frequency_mask, is_augmentation=True),
                # add noise to the sample
                "add_noise": Action(
                    actions.tensor_add_noise, is_augmentation=True, std=0.005
                ),
                # linearly scale the _values_ of the sample
                "rescale": Action(actions.scale_tensor),
                # apply random affine (rotation, translation, scaling, shearing) augmentation
                # default values are reasonable for spectrograms: no shearing or rotation
                "random_affine": Action(
                    actions.torch_random_affine, is_augmentation=True
                ),
            }
        )

        # remove overlay if overlay_df was not specified
        if overlay_df is None:
            self.pipeline.drop("overlay", inplace=True)

    def _generate_sample(self, sample):
        """add attributes to the sample specifying desired shape of output sample

        these will be used by the SpectrogramToTensor action in the pipeline

        otherwise, generate AudioSamples from paths as normal
        """
        sample = super()._generate_sample(sample)

        # add attributes specifying desired shape of output sample
        sample.height = self.height
        sample.width = self.width
        sample.channels = self.channels

        return sample


class AudioPreprocessor(BasePreprocessor):
    """Child of BasePreprocessor that only loads audio and resamples

    Args:
        sample_duration:
            length in seconds of audio samples generated
        sample_rate: target sample rate. [default: None] does not resample
    """

    def __init__(self, sample_duration, sample_rate):
        super(AudioPreprocessor, self).__init__(sample_duration=sample_duration)
        self.pipeline = pd.Series(
            {
                # load a segment of an audio file into an Audio object
                # references AudioSample attributes: start_time and duration
                "load_audio": AudioClipLoader(sample_rate=sample_rate),
            }
        )
