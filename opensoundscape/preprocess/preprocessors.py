"""Preprocessor classes: tools for preparing and augmenting audio samples"""

from pathlib import Path
import pandas as pd
import copy
import time
import json
import yaml
import pathlib
import warnings

from opensoundscape.preprocess import actions, action_functions, io
from opensoundscape.preprocess.actions import (
    Action,
    AudioClipLoader,
    AudioTrim,
    SpectrogramToTensor,
)
from opensoundscape.preprocess.overlay import Overlay
from opensoundscape.preprocess.utils import PreprocessingError, get_args, get_reqd_args
from opensoundscape.spectrogram import Spectrogram
from opensoundscape.audio import Audio
from opensoundscape.sample import AudioSample
from opensoundscape.preprocess.actions import ACTION_CLS_DICT
from opensoundscape.preprocess import io
from opensoundscape.preprocess import preprocessors, actions


PREPROCESSOR_CLS_DICT = {}


def register_preprocessor_cls(cls):
    """add class to PREPROCESSOR_CLS_DICT"""
    # register the model in dictionary
    PREPROCESSOR_CLS_DICT[io.build_name(cls)] = cls

    # return the class (use as decorator)
    return cls


def preprocessor_from_dict(dict):
    """load a preprocessor from a dictionary saved with pre.to_dict()

    looks up class name using the "class" key in PREPROCESSOR_CLS_DICT
    requires that the class was decorated with @register_preprocessor_cls
    so that it is listed in PREPROCESSOR_CLS_DICT.

    If you write a custom preprocessor class, you must decorate it with
    @register_preprocessor_cls so that it can be looked up by name during from_dict

    Args:
        dict: dictionary created with a preprocessor class's .to_dict() method

    Returns:
        initialized preprocessor with same configuration and parameters as original
        - some caveats: Overlay augentation will not re-load fully, as overlay sample
            dataframes and `criterion_fn`s are not saved

    See also: BasePreprocessor.from_dict(), .save_json(), load_json()
    """
    preprocessor_class = PREPROCESSOR_CLS_DICT[dict["class"]]
    return preprocessor_class.from_dict(dict)


@register_preprocessor_cls
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
        sample_duration: length of audio samples to generate (seconds)
    """

    def __init__(self, sample_duration=None):
        self.pipeline = pd.Series({}, dtype=object)
        self.sample_duration = sample_duration

    def __repr__(self):
        return f"Preprocessor with pipeline:\n{self.pipeline}"

    def _repr_html_(self):
        top_str = f"{type(self).__name__} with pipeline:"
        out = f"<b>{top_str}</b>"

        for i, action in self.pipeline.items():
            out += _action_html(i, action)

        return out

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
        profile=False,
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
            trace (boolean - default False): if True, saves the output of each pipeline step in the `sample_info` output argument
                Can be used for analysis/debugging of intermediate values of the sample during preprocessing
            profile (boolean - default False): if True, saves the runtime of each pipeline step in `.runtime`
                (a series indexed like .pipeline)
        Returns:
            sample (instance of AudioSample class)

        """
        # validate input
        if break_on_key is not None:
            assert (
                break_on_key in self.pipeline
            ), f"break_on_key was {break_on_key} but no matching action found in pipeline"

        # create AudioSample from input path
        sample = self._generate_sample(sample)
        if trace:
            sample.trace = pd.Series(index=self.pipeline.index, dtype=str)

        if profile:
            sample.runtime = pd.Series(index=self.pipeline.index)

        # run the pipeline by performing each Action on the AudioSample
        try:
            # perform each action in the pipeline
            for k, action in self.pipeline.items():
                time0 = time.time()

                if type(action) == break_on_type or k == break_on_key:
                    if trace:
                        # saved "output" of this step informs user pipeline was stopped
                        sample.trace.loc[k] = (
                            f"## Pipeline terminated ## {sample.trace[k]}"
                        )
                    break
                if action.bypass:
                    continue
                if action.is_augmentation and bypass_augmentations:
                    if trace:
                        sample.trace.loc[k] = f"## Bypassed ## {sample.trace[k]}"
                    continue

                # perform the action on the sample (modifies the AudioSample in-place)
                action(sample)

                if profile:
                    sample.runtime[k] = time.time() - time0

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
        """create AudioSample object from initial input:

        can override this method in subclasses to modify how samples
        are created, or to add additional attributes to samples

        Args:
            sample: can be any of
            - (path, start time) tuple
            - pd.Series with (file, start_time, end_time) as .name
                (eg index of a pd.DataFrame from which row was taken)
            - AudioSample object
        Returns:
            new AudioSample object
        """
        # handle paths or pd.Series as input for `sample`
        if isinstance(sample, tuple):
            # assume duration should be self.sample_duration
            path, start = sample
            assert isinstance(
                path, (str, Path)
            ), "if passing tuple, first element must be str or pathlib.Path"
            sample = AudioSample(path, start_time=start, duration=self.sample_duration)
        elif isinstance(sample, pd.Series):
            sample = AudioSample.from_series(sample)
        elif isinstance(sample, pd.DataFrame):
            raise AssertionError(
                "sample must be AudioSample, tuple of (path, start_time), "
                "or pd.Series with (path, start_time, end_time) as .name. "
                f"was {type(sample)}. "
                "Perhaps a dataset was accessed like dataset[[0,1,2]] instead of dataset[0]?"
            )
        else:
            assert isinstance(sample, AudioSample), (
                "sample must be AudioSample, tuple of (path, start_time), "
                "or pd.Series with (path, start_time, end_time) as .name. "
                f"was {type(sample)}"
            )
            # make a copy to avoid modifying original
            sample = copy.deepcopy(sample)

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

    def to_dict(self):
        d = {}
        # save attributes
        # remove hidden attributes and methods (starting with _), only keep public attributes
        # also ignore "pipeline" and handle it separatey below
        d["attributes"] = {
            k: v
            for k, v in copy.deepcopy(self.__dict__).items()
            if k[0] != "_" and not callable(v) and k != "pipeline"
        }
        # convert each action in the pipeline to dictionary
        d["pipeline"] = {k: v.to_dict() for k, v in self.pipeline.items()}

        # assume that any attributes with the same name as arguments to __init__ should
        # be passed to __init__, and that all arguments required to re-create this
        # class with __init__ are stored as parameters. Add them to a separate dictionary
        # that we will pass to __init__ in from_dict()
        # note that these are duplicated with d["attributes"]
        d["init_kwargs"] = {
            k: d["attributes"][k]
            for k in get_args(type(self)).keys()
            if k in d["attributes"]
        }

        # save class name for re-creating the object; matches a key in PREPROCESSOR_CLS_DICT
        d["class"] = io.build_name(type(self))

        return d

    @classmethod
    def from_dict(cls, dict):
        # NOTE: might not work properly if child class kwargs are used to initialize the object in a way
        # other than just setting the kwargs as attributes, or for creating .pipeline
        # or if arguments are not set as parameters (dataclass style)
        # (bc they then would not be saved by to_dict())

        # create a new instance of the class
        # pass any stored kwargs to the __init__ method
        instance = cls(**dict["init_kwargs"])

        # set any stored attribute values
        for attr in dict["attributes"]:
            setattr(instance, attr, dict["attributes"][attr])

        # re-create the pipeline using the actions' .from_dict methods
        instance.pipeline = pd.Series(
            {
                k: ACTION_CLS_DICT[v["class"]].from_dict(v)
                for k, v in dict["pipeline"].items()
            }
        )
        return instance

    def save_json(self, path):
        """save preprocessor to a json file

        re-load with load_json(path) or .from_json(path)"""
        d = self.to_dict()
        with open(path, "w") as f:
            json.dump(d, f, cls=io.NumpyTypeEncoder)

    @classmethod
    def from_json(self, path):
        """load preprocessor from a json file

        for instance, file created with .save_json()"""
        with open(path, "r") as f:
            d = json.load(f, cls=io.NumpyTypeDecoder)
        return self.from_dict(d)

    def save_yaml(self, path):
        """save preprocessor to a YAML file

        re-load with load_yaml(path) or .from_yaml(path)
        """
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, Dumper=io.CustomYamlDumper, sort_keys=False)

    @classmethod
    def from_yaml(self, path):
        """load preprocessor from a YAML file

        for instance, file created with .save_yaml()

        note that safe_load is not used, so make sure you trust the author of the file

        Args:
            path: path to the .yaml file

        Returns:
            preprocessor: instance of a preprocessor class
        """
        # Load the dictionary from the YAML file
        with open(path, "r") as f:
            loaded_data = yaml.load(f, Loader=io.CustomYamlLoader)

        return self.from_dict(loaded_data)

    def save(self, path):
        """save preprocessor to a file

        Args:
            path: path to the file, with .json or .yaml extension
        """
        if isinstance(path, pathlib.Path):
            path = str(path)
        if path.lower().endswith(".json"):
            self.save_json(path)
        elif path.lower().endswith(".yaml"):
            self.save_yaml(path)
        else:
            raise ValueError(f"Unsupported file format: {path}. Must be .json or .yaml")


def load_json(path):
    """load preprocessor from a json file

    for instance, file created with .save_json()"""
    with open(path, "r") as f:
        d = json.load(f, cls=io.NumpyTypeDecoder)
    return preprocessor_from_dict(d)


def load_yaml(path):
    """load preprocessor from a YAML file

    for instance, file created with .save_yaml()

    Args:
        path: path to the .yaml file

    Returns:
        preprocessor: instance of a preprocessor class
    """
    # Load the dictionary from the YAML file
    with open(path, "r") as f:
        loaded_data = yaml.load(f, Loader=io.CustomYamlLoader)

    return preprocessor_from_dict(loaded_data)


def load(path):
    """load preprocessor from a file (json or yaml)

    use to load preprocessor definitions saved with .save()

    Args:
        path: path to the file

    Returns:
        preprocessor: instance of a preprocessor class
    """
    if isinstance(path, pathlib.Path):
        path = str(path)
    if path.lower().endswith(".json"):
        return load_json(path)
    elif path.lower().endswith(".yaml"):
        return load_yaml(path)
    else:
        raise ValueError(f"Unsupported file format: {path}. Must be .json or .yaml")


@register_preprocessor_cls
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
            -  default None will use the original width of the spectrogram
        channels: number of channels in output sample (default 1)
        sample_shape: tuple of (height, width, channels) for output sample
            Deprecated in favor of using height, width, channels
            - if not None, will override height, width, channels
            [default: None] means use height, width, channels arguments
    """

    def __init__(
        self,
        sample_duration,
        overlay_df=None,
        height=224,
        width=224,
        channels=1,
        sample_shape=None,
    ):
        super().__init__(sample_duration=sample_duration)

        # allow sample_shape argument for backwards compatability
        if sample_shape is not None:
            height, width, channels = sample_shape
            warnings.warn(
                """sample_shape argument is deprecated. Please use height, width, channels arguments instead. 
                The current behavior is to override height, width, channels with sample_shape 
                when sample_shape is not None.
                """,
                DeprecationWarning,
            )

        self.height = height
        self.width = width
        self.channels = channels

        # define a default set of Actions
        # each Action's .__call__ method is called during preprocessing
        # the .__call__ method takes an AudioSample object as an argument
        # and modifies it _in place_!!
        self.pipeline = pd.Series(
            {
                # load a segment of an audio file into an Audio object
                # references AudioSample attributes: start_time and duration
                "load_audio": AudioClipLoader(),
                # if we are augmenting and get a long file, take a random trim from it
                "random_trim_audio": AudioTrim(
                    target_duration=sample_duration,
                    is_augmentation=True,
                    random_trim=True,
                ),
                # otherwise, we expect to get the correct duration. no random trim
                # trim or extend (w/silence) clips to correct length
                "trim_audio": AudioTrim(
                    target_duration=sample_duration, random_trim=False
                ),
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
                "overlay": (
                    Overlay(
                        is_augmentation=True,
                        overlay_df=pd.DataFrame() if overlay_df is None else overlay_df,
                        update_labels=True,
                    )
                ),
                # add vertical (time) and horizontal (frequency) masking bars
                "time_mask": Action(action_functions.time_mask, is_augmentation=True),
                "frequency_mask": Action(
                    action_functions.frequency_mask, is_augmentation=True
                ),
                # add noise to the sample
                "add_noise": Action(
                    action_functions.tensor_add_noise, is_augmentation=True, std=0.005
                ),
                # linearly scale the _values_ of the sample
                "rescale": Action(action_functions.scale_tensor),
                # apply random affine (rotation, translation, scaling, shearing) augmentation
                # default values are reasonable for spectrograms: no shearing or rotation
                "random_affine": Action(
                    action_functions.torch_random_affine, is_augmentation=True
                ),
            }
        )

        # bypass overlay if overlay_df was not provided (None)
        # keep the action in the pipeline for ease of enabling it later
        if overlay_df is None or len(overlay_df) < 1:
            self.pipeline["overlay"].bypass = True

    def _generate_sample(self, sample):
        """add attributes to the sample specifying desired shape of output sample"""
        sample = super()._generate_sample(sample)

        # add attributes specifying desired shape of output sample
        sample.height = self.height
        sample.width = self.width
        sample.channels = self.channels

        return sample


class PCENPreprocessor(preprocessors.SpectrogramPreprocessor):
    def __init__(self, *args, **kwargs):
        """same arguments as SpectrogramPreprocessor

        adds an action that performs PCEN after making spectrogram
        PCEN is Per Channel Energy Normalization, see https://arxiv.org/abs/1607.05666

        The only other difference from SpectrogramPreprocessor is that we set the dB_scale to False
        when generating the Spectrogram, because PCEN expects a linear-scale spectrogram; and that
        the we normalize the output of PCEN to [0,1], then use range=[0,1] for spec.to_tensor()

        note: user should set self.pipeline['pcen'].params['sr'] and 'hop_length' to match the audio/spectrogram settings
        after instantiating this class

        User can modify parameters, in particular setting PCEN parameters via self.pipeline['pcen'].params
        """
        super().__init__(*args, **kwargs)

        # need to pass linear-value spectrogram to pcen
        self.pipeline["to_spec"].set(dB_scale=False)

        # use Librosa implementation of PCEN (could use a pytorch implementation in the future, and make it trainable)
        pcen_action = actions.Action(fn=action_functions.pcen, is_augmentation=False)
        self.insert_action(action_index="pcen", action=pcen_action, after_key="to_spec")

        # normalize PCEN output to [0,1]
        def normalize_to_01(s):
            new_s = (s.spectrogram - s.spectrogram.min()) / (
                s.spectrogram.max() - s.spectrogram.min()
            )
            return s._spawn(spectrogram=new_s)

        self.insert_action(
            action_index="normalize",
            action=actions.Action(fn=normalize_to_01, is_augmentation=False),
            after_key="pcen",
        )

        self.pipeline.to_tensor.set(range=[0, 1])


@register_preprocessor_cls
class AudioPreprocessor(BasePreprocessor):
    """Child of BasePreprocessor that only loads audio and resamples

    Args:
        sample_duration:
            length in seconds of audio samples generated
        sample_rate: target sample rate. [default: None] does not resample
        extend_short_clips: if True, clips shorter than sample_duration are extended
            to sample_duration by adding silence.
    """

    def __init__(self, sample_duration, sample_rate, extend_short_clips=True):
        super().__init__(sample_duration=sample_duration)
        self.pipeline = pd.Series(
            {
                # load a segment of an audio file into an Audio object
                # references AudioSample attributes: start_time and duration
                "load_audio": AudioClipLoader(sample_rate=sample_rate),
                # trim samples to correct length
                # if extend_short_clips=True, extend short clips with silence
                "trim_audio": AudioTrim(
                    target_duration=sample_duration, extend=extend_short_clips
                ),
            }
        )


@register_preprocessor_cls
class AudioAugmentationPreprocessor(AudioPreprocessor):
    """AudioPreprocessor that applies augmentations to audio samples during training"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # random gain
        # when training data is much louder than application, e.g. xeno-canto focal recordings,
        # large gain reduction such as dB_range=(-40, -10) may be useful
        random_gain_action = actions.Action(
            action_functions.audio_random_gain, is_augmentation=True, dB_range=(-5, 0)
        )
        random_gain_action.bypass = True
        self.insert_action("random_gain", random_gain_action)

        # add noise
        add_noise_action = actions.Action(
            action_functions.audio_add_noise,
            is_augmentation=True,
            noise_dB=(-80, -40),
        )
        self.insert_action("add_noise", add_noise_action)

        # random time wrap: shift sample in time, moving end to beginning
        random_wrap_action = actions.Action(
            action_functions.random_wrap_audio,
            is_augmentation=True,
            probability=0.75,
        )
        self.insert_action("random_wrap", random_wrap_action)

        # time mask: randomly mask short time segments with noise
        time_mask_action = actions.Action(
            action_functions.audio_time_mask, is_augmentation=True
        )
        self.insert_action("time_mask", time_mask_action)


class NoiseReduceAudioPreprocessor(AudioPreprocessor):
    def __init__(
        self,
        sample_duration,
        sample_rate,
        extend_short_clips=True,
        noisereduce_kwargs=None,
    ):
        """Preprocessor that reduces noise in audio signal and returns Audio objects

        uses package noisereduce (documentation: https://pypi.org/project/noisereduce/) by Tim Sainburg

        see also: discusion of stationary and non-stationary noise reduction in this paper
        https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2021.811737/full


        Args:
            sample_duration: length in seconds of audio samples generated
            sample_rate: target sample rate. [default: None] does not resample
            extend_short_clips: if True, clips shorter than sample_duration are extended with silence
            noisereduce_kwargs: dictionary of args to pass to noisereduce.reduce_noise()
        """
        # Note that noisereduce package implements torch integration w/gpu acceleration, which we may want to use?
        noisereduce_kwargs = noisereduce_kwargs or {}
        super().__init__(
            sample_duration=sample_duration,
            sample_rate=sample_rate,
            extend_short_clips=extend_short_clips,
        )
        self.insert_action(
            "noise_reduce",
            after_key="trim_audio",
            action=Action(
                Audio.reduce_noise,
                is_augmentation=False,
                noisereduce_kwargs=noisereduce_kwargs,
            ),
        )


class NoiseReduceSpectrogramPreprocessor(SpectrogramPreprocessor):
    def __init__(
        self,
        sample_duration,
        overlay_df=None,
        height=None,
        width=None,
        channels=1,
        noisereduce_kwargs=None,
    ):
        """Preprocessor that reduces noise in audio samples before creating spectrogram

        uses package noisereduce (documentation: https://pypi.org/project/noisereduce/) by Tim Sainburg

        see also: discusion of stationary and non-stationary noise reduction in this paper
        https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2021.811737/full


        Args:
            sample_duration: length in seconds of audio samples generated
            overlay_df: if not None, will include an overlay action drawing
                samples from this df
            height: height of output sample (frequency axis)
                - default None will use the original height of the spectrogram
            width: width of output sample (time axis)
                -  default None will use the originalwidth of the spectrogram
            channels: number of channels in output sample (default 1)
            noisereduce_kwargs: dictionary of args to pass to noisereduce.reduce_noise()
        """
        # Note that noisereduce package implements torch integration w/gpu acceleration, which we may want to use?
        noisereduce_kwargs = noisereduce_kwargs or {}
        super().__init__(
            sample_duration=sample_duration,
            overlay_df=overlay_df,
            height=height,
            width=width,
            channels=channels,
        )
        self.insert_action(
            "noise_reduce",
            after_key="trim_audio",
            action=Action(
                Audio.reduce_noise,
                is_augmentation=False,
                noisereduce_kwargs=noisereduce_kwargs,
            ),
        )


def replace_nones(value):
    if value != value:
        return "nan"
    elif value is None:
        return "None"
    else:
        return str(value)


from IPython.display import display, HTML


def _action_html(title, action):

    # Check if .bypass attribute is True to set text color to grey
    if action.bypass:
        if action.is_augmentation:
            text_color = "#80a3ba"
        else:
            text_color = "#CCCCCC"
    else:
        if action.is_augmentation:
            text_color = "#1a608f"
        else:
            text_color = "#000000"

    # Format object attributes with one key-value pair per line
    content = "<br>".join(
        [
            f"<strong>{key}:</strong> {replace_nones(value)}"
            for key, value in action.params.items()
        ]
    )

    suffix = " (Bypassed)" if action.bypass else ""
    if action.is_augmentation:
        suffix += " (Augmentation)"
    html_code = f"""

    <details style="margin: 2px 0;">
      <summary style="font-size: 12px; cursor: pointer; margin: 2px 0; color: {text_color};">{title+suffix}</summary>
      <div style="padding: 2px 0; border-left: 2px solid #007acc; margin: 2px 0 0 12px; font-size: 10px; color: {text_color};">
        {content}
      </div>
    </details>
    
    """
    return html_code
