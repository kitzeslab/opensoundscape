"""Actions for augmentation and preprocessing pipelines

This module contains Action classes which act as the elements in
Preprocessor pipelines. Action classes have __call__() method that operates on an audio sample,
using the .params dictionary of parameter values.
They take a single sample of a specific type and return the transformed
or augmented sample, which may or may not be the same type as the original.

See the action_functions.py module for functions that can be used to create actions using the Action class.
Pass the Action class any function to the action_fn argument, and pass additional arguments to 
set parameters of the Action's .params dictionary. 

Note on converting to/from dictionary/json/yaml:
This will break if you use non-built-in preprocessing operations. 
However, will work if you provide any custom functions/classes and 
decorate them with @register_action_cls or @register_action_fn. 
See the docstring of `action_from_dict()` for examples. 

See the preprocessor module and Preprocessing tutorial
for details on how to use and create your own actions.
"""

import random
import warnings
import numpy as np
import torchvision
import pandas as pd
import copy
from types import MethodType, FunctionType

from opensoundscape.preprocess.utils import PreprocessingError, get_args, get_reqd_args
from opensoundscape.preprocess.action_functions import (
    ACTION_FN_DICT,
    list_action_fns,
    register_action_fn,
)
from opensoundscape.preprocess import io
from opensoundscape.sample import AudioSample
from opensoundscape.spectrogram import Spectrogram
from opensoundscape.audio import Audio


ACTION_CLS_DICT = dict()


def list_actions():
    """return list of available Action class keyword strings"""
    return list(ACTION_CLS_DICT.keys())


def register_action_cls(action_cls):
    """add class to ACTION_DICT"""
    # register the model in dictionary
    ACTION_CLS_DICT[io.build_name(action_cls)] = action_cls

    # return the function
    return action_cls


# let's register some functions we might want to use from elsewhere in the code as action functions
# including all public methods of Audio and Spectrogram classes
def register_all_methods(cls, public_only=True):
    for k in cls.__dict__.keys():
        if k.startswith("_") and public_only:
            pass
        attr = getattr(cls, k)
        if callable(attr):
            register_action_fn(attr)


register_all_methods(Audio)
register_all_methods(Spectrogram)


def action_from_dict(dict):
    """load an action from a dictionary

    Args:
        dict: dictionary created with Action.to_dict()
            - contains keys 'class', 'params', and other keys for object attributes

    Note: if the dictionary names a 'class' or 'action_fn' that is not
    built-in to OpenSoundscape, you should define the class/action in your code and
    add the decorator @register_action_cls or @register_action_fn

    For instance, if we used the Action class and passed a custom action_fn:
    @register_action_fn
    def my_special_sauce(...):
        ...

    Now we can use action_from_dict() to re-create an action that specifies
    `'action_fn':'__main__.my_special_sauce'`

    Similarly, say we defined a custom class in a module my_utils.py, we add the decorator before
    the class definition:
    @register_action_cls
    class Magic(BaseAction):
        ...

    now we can use action_from_dict() to re-create the class from a dictionary that has
    `'class' : 'my_utils.Magic'`
    """
    action_cls = ACTION_CLS_DICT[dict["class"]]
    return action_cls.from_dict(dict)


@register_action_cls
class BaseAction:
    """Parent class for all Actions (used in Preprocessor pipelines)

    New actions should subclass this class.
    """

    def __init__(self, is_augmentation=False):
        self.params = pd.Series(dtype="object")
        self.is_augmentation = is_augmentation
        self.bypass = False

    def __repr__(self):
        return (
            f"{'__bypassed__ ' if self.bypass else ''}"
            f"{'Augmentation ' if self.is_augmentation else ''}"
            f"Action with .params: \n{self.params}"
        )

    def __call__(self, x):
        # modifies the sample in-place
        pass

    def set(self, **kwargs):
        # Series.update ignores nan/None values, so we use dictionary.update method
        new_params = dict(self.params)
        new_params.update(kwargs)
        self.params = pd.Series(new_params, dtype=object)

    def get(self, arg):
        return self.params[arg]

    def to_dict(self, ignore_attributes=()):
        """export current attributes and .params to a dictionary

        useful for saving to JSON

        Args:
            ignore_attributes:
                list of str: attributes to not save
                (useful for skipping large objects to reduce memory usage)

        re-load with `.from_dict(dict)`
        """
        # create a dictionary with copies of all attributes except any listed in `ignore_attributes`
        d = {
            copy.deepcopy(k): copy.deepcopy(v)
            for k, v in self.__dict__.items()
            if not k in ignore_attributes
        }
        # note: to_json will fail if params dictionary is not json-able
        d["params"] = d["params"].to_dict()  # pd.Series to dictionary
        # string name of the class, eg "opensoundscape.audio.Audio"
        d["class"] = io.build_name(type(self))
        return d

    @classmethod
    def from_dict(cls, dict):
        """initialize from dictionary created by .to_dict()

        override if subclass should be initialized with any arguments
        """

        instance = cls()

        # set dictionary key:value pairs as attributes
        # handle params and 'class' name as special cases
        for key, value in dict.items():
            if key == "class":
                # this is just the class name as a string
                # which we should generate with build_name() when needed
                pass
            if key == "params":  # dict to series
                value = pd.Series(value, dtype="object")
            # any key-value pairs become attributes
            instance.__setattr__(key, value)

        # instance.bypass = dict["bypass"]
        # instance.is_augmentation = dict["is_augmentation"]
        return instance


@register_action_cls
class Action(BaseAction):
    """Action class for an arbitrary function

    The function must take the sample as the first argument

    Note that this allows two use cases:
    (A) regular function that takes an input object as first argument
        eg. Audio.from_file(path,**kwargs)
    (B) method of a class, which takes 'self' as the first argument,
        eg. Spectrogram.bandpass(self,**kwargs)

    Other arguments are an arbitrary list of kwargs.
    """

    def __init__(self, fn, is_augmentation=False, **kwargs):
        super().__init__()

        self.action_fn = fn
        self.is_augmentation = is_augmentation

        # query action_fn for arguments and default values
        self.params = pd.Series(get_args(self.action_fn), dtype=object)

        # whether the first argument is 'self' or the incoming object,
        # we remove it from the params dict
        self.params = self.params[1:]

        # update self.params with any user-provided parameters
        self.set(**kwargs)

        # make sure all required args are given (skipping the first, which will be provided by go)
        unmatched_reqd_args = set(get_reqd_args(self.action_fn)[1:]) - set(
            list(kwargs.keys())
        )

        assert unmatched_reqd_args == set(
            []
        ), f"These required arguments were not provided: {unmatched_reqd_args}"

    def __repr__(self):
        return (
            f"{'__bypassed__' if self.bypass else ''}"
            f"{'Augmentation ' if self.is_augmentation else ''}"
            f"Action calling {self.action_fn}, "
            f"with .params: \n{self.params}"
        )

    def __call__(self, sample, **kwargs):
        # the syntax is the same regardless of whether
        # first argument is "self" (for a class method) or not
        # we pass self.params to kwargs along with any additional kwargs

        # only pass (and get back) the data of the sample to the action function
        # to use other attributes of sample.data, write another class and override
        # this __call__ method, for example:
        # def __call__(self, sample, **kwargs):
        #   self.action_fn(sample, **dict(self.params, **kwargs))

        # we modify the sample in-place and don't return anything
        sample.data = self.action_fn(sample.data, **dict(self.params, **kwargs))

    def to_dict(self, ignore_attributes=()):
        """export current attributes and .params to a dictionary

        useful for saving to JSON

        re-load with `.from_dict(dict)`
        """
        d = super().to_dict(ignore_attributes=ignore_attributes)
        d["action_fn"] = io.build_name(self.action_fn)
        return d

    @classmethod
    def from_dict(cls, dict):
        """initialize from dictionary created by .to_dict()"""
        instance = cls(
            fn=ACTION_FN_DICT[dict["action_fn"]],
            is_augmentation=dict["is_augmentation"],
            **dict["params"],
        )
        # set any other attributes stored as key:value pairs in the dict
        for key, value in dict.items():
            if key in ("class", "action_fn", "params", "is_augmentation"):
                continue
            # dictionary key-value pairs become attributes
            instance.__setattr__(key, value)
        return instance


@register_action_cls
class AudioClipLoader(Action):
    """Action to load clips from an audio file

    Loads an audio file or part of a file to an Audio object.
    Will load entire audio file if sample.start_time and sample.duration are None.
    If sample.start_time and sample.duration are provided, loads the audio only in the
    specified interval.

    see Audio.from_file() for documentation.

    Args:
        see Audio.from_file()
    """

    def __init__(self, **kwargs):
        if "fn" in kwargs:
            kwargs.pop("fn")
        super().__init__(fn=Audio.from_file, **kwargs)
        # two params are provided by sample.start_time and sample.duration
        self.params = self.params.drop(["offset", "duration"])

    def __call__(self, sample, **kwargs):
        offset = 0 if sample.start_time is None else sample.start_time
        duration = sample.duration
        sample.data = self.action_fn(
            sample.data, offset=offset, duration=duration, **dict(self.params, **kwargs)
        )


@register_action_cls
class AudioTrim(Action):
    """Action to trim/extend audio to desired length

    Args:
        see actions.trim_audio()
    """

    def __init__(self, **kwargs):
        if "fn" in kwargs:
            kwargs.pop("fn")
        super().__init__(fn=trim_audio, **kwargs)

    def __call__(self, sample, **kwargs):
        self.action_fn(sample, **dict(self.params, **kwargs))


@register_action_fn
def trim_audio(sample, target_duration, extend=True, random_trim=False, tol=1e-10):
    """trim audio clips from t=0 or random position (Audio -> Audio)

    Trims an audio file to desired length.

    Allows audio to be trimmed from start or from a random time

    Optionally extends audio shorter than clip_length to sample.duration by
    appending silence.

    Args:
        sample: AudioSample with .data=Audio object, .duration as sample duration
        target_duration: length of resulting clip in seconds. If None,
            no trimming is performed.
        extend: if True, clips shorter than sample.duration are
            extended with silence to required length [Default: True]
        random_trim: if True, chooses a random segment of length sample.duration
            from the input audio. If False, the file is trimmed from 0 seconds
            to sample.duration seconds. [Default: False]
        tol: tolerance for considering a clip to be long enough (sec),
            when raising an error for short clips [Default: 1e-6]

    Effects:
        Updates the sample's .data, .start_time, and .duration attributes
    """

    if target_duration is None:
        return

    audio = sample.data

    if len(audio.samples) == 0:
        raise ValueError("recieved zero-length audio")

    # input audio is not as long as desired length
    if extend:  # extend clip sith silence
        audio = audio.extend_to(target_duration)
    else:
        if audio.duration + tol < target_duration:
            raise ValueError(
                f"the length of the original file ({audio.duration} "
                f"sec) was less than the length to extract "
                f"({target_duration} sec). To extend short "
                f"clips, use extend=True"
            )
    if random_trim:
        # uniformly randomly choose clip time from full audio
        # such that a full-length clip can be extracted
        extra_time = audio.duration - target_duration
        start_time = np.random.uniform() * extra_time
    else:
        start_time = 0

    end_time = start_time + target_duration
    audio = audio.trim(start_time, end_time)

    # update the sample in-place
    sample.data = audio
    if sample.start_time is None:
        sample.start_time = start_time
    else:
        sample.start_time += start_time
    sample.duration = target_duration


@register_action_cls
class SpectrogramToTensor(Action):
    """Action to create Tesnsor of desired shape from Spectrogram

    calls .to_image on sample.data, which should be type Spectrogram

    **kwargs are passed to Spectrogram.to_image()

    """

    def __init__(self, fn=Spectrogram.to_image, is_augmentation=False, **kwargs):
        kwargs.update(dict(return_type="torch"))  # return a tensor, not PIL.Image
        if "fn" in kwargs:
            kwargs.pop("fn")
        super().__init__(fn, is_augmentation, **kwargs)

    def __call__(self, sample, **kwargs):
        """converts sample.data from Spectrogram to Tensor"""
        # sample.data must be Spectrogram object
        # sample should have attributes: height, width, channels
        # use info from sample for desired shape and n channels
        kwargs.update(shape=[sample.height, sample.width], channels=sample.channels)
        sample.data = self.action_fn(sample.data, **dict(self.params, **kwargs))
