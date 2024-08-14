"""Actions for augmentation and preprocessing pipelines

This module contains Action classes which act as the elements in
Preprocessor pipelines. Action classes have go(), on(), off(), and set()
methods. They take a single sample of a specific type and return the transformed
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
from opensoundscape.sample import AudioSample
from opensoundscape.spectrogram import Spectrogram
from opensoundscape.audio import Audio

ACTION_CLS_DICT = dict()
ACTION_FN_DICT = dict()


def list_actions():
    """return list of available Action class keyword strings"""
    return list(ACTION_CLS_DICT.keys())


def list_action_fns():
    """return list of available action function keyword strings
    (can be used to initialize Action class)
    """
    return list(ACTION_FN_DICT.keys())


def build_name(method_or_object):
    """return the full function or class name

    Args:
        method_or_object: a method, object, or class

    Returns: a string like "opensoundscape.module.Class.method"
        - Note: if defined in a script, looks like "__main__.my_function"

    """
    prefix = method_or_object.__module__
    if isinstance(method_or_object, (MethodType, FunctionType, type)):
        # its a method/function or a class
        return f"{prefix}.{method_or_object.__qualname__}"
    return f"{prefix}.{type(method_or_object).__qualname__}"


def register_action_cls(action_cls):
    """add class to ACTION_DICT"""
    # register the model in dictionary
    ACTION_CLS_DICT[build_name(action_cls)] = action_cls

    # return the function
    return action_cls


def register_action_fn(action_fn):
    """add function to ACTION_FN_DICT

    this allows us to recreate the Action class with a named action_fn

    see also: ACTION_DICT (stores list of named classes for preprocessing)
    """
    # register the model in dictionary
    ACTION_FN_DICT[build_name(action_fn)] = action_fn
    # return the function
    return action_fn


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

    Subclasses should set `self.requires_labels = True` if go() expects (X,y)
    instead of (X). y is a row of a dataframe (a pd.Series) with index (.name)
    = original file path, columns=class names, values=labels (0,1). X is the
    sample, and can be of various types (path, Audio, Spectrogram, Tensor, etc).
    See Overlay for an example of an Action that uses labels.
    """

    def __init__(self, is_augmentation=False):
        self.params = pd.Series(dtype="object")
        # self.returns_labels = False
        self.is_augmentation = is_augmentation
        self.bypass = False

    def __repr__(self):
        return (
            f"{'Bypassed ' if self.bypass else ''}"
            f"{'Augmentation ' if self.is_augmentation else ''}"
            "Action"
        )

    def go(self, x):
        # modifies the sample in-place
        pass

    def set(self, **kwargs):
        """only allow keys that exist in self.params"""
        unmatched_args = set(list(kwargs.keys())) - set(list(self.params.keys()))
        assert unmatched_args == set([]), (
            f"unexpected arguments: {unmatched_args}. "
            f"The valid arguments and current values are: \n{self.params}"
        )
        # Series.update ignores nan/None values, so we use dictionary.update method
        new_params = dict(self.params)
        new_params.update(kwargs)
        self.params = pd.Series(new_params, dtype=object)
        # self.params.update(pd.Series(kwargs, dtype=object))

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
        d["class"] = build_name(type(self))  # eg "opensoundscape.audio.Audio.from_file"
        return d

    @classmethod
    def from_dict(cls, dict):
        """initialize from dictionary created by .to_dict()"""

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
        super(Action, self).__init__()

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
            f"{'## Bypassed ## ' if self.bypass else ''}"
            f"{'Augmentation ' if self.is_augmentation else ''}"
            f"Action calling {self.action_fn}"
        )

    def go(self, sample, **kwargs):
        # the syntax is the same regardless of whether
        # first argument is "self" (for a class method) or not
        # we pass self.params to kwargs along with any additional kwargs

        # only pass (and get back) the data of the sample to the action function
        # to use other attributes of sample.data, write another class and override
        # this go() method, for example:
        # def go(self, sample, **kwargs):
        #   self.action_fn(sample, **dict(self.params, **kwargs))

        # should we make a copy to avoid modifying the original object?
        # or accept that we are modifying the original sample in-place?
        # I think its in-place since we now pass an object and update the data
        sample.data = self.action_fn(sample.data, **dict(self.params, **kwargs))

    def to_dict(self, ignore_attributes=()):
        """export current attributes and .params to a dictionary

        useful for saving to JSON

        re-load with `.from_dict(dict)`
        """
        d = super().to_dict(ignore_attributes=ignore_attributes)
        d["action_fn"] = build_name(self.action_fn)
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
    Will load entire audio file if _start_time and _end_time are None.
    If _start_time and _end_time are provided, loads the audio only in the
    specified interval.

    see Audio.from_file() for documentation.

    Args:
        see Audio.from_file()
    """

    def __init__(self, **kwargs):
        if "fn" in kwargs:
            kwargs.pop("fn")
        super(AudioClipLoader, self).__init__(fn=Audio.from_file, **kwargs)
        # two params are provided by sample.start_time and sample.duration
        self.params = self.params.drop(["offset", "duration"])

    def go(self, sample, **kwargs):
        offset = 0 if sample.start_time is None else sample.start_time
        duration = None if sample.duration is None else sample.duration
        sample.data = self.action_fn(
            sample.data, offset=offset, duration=duration, **dict(self.params, **kwargs)
        )


@register_action_cls
class AudioTrim(Action):
    """Action to trim/extend audio to desired length

    Args:
        see actions.audio_trim()
    """

    def __init__(self, **kwargs):
        if "fn" in kwargs:
            kwargs.pop("fn")
        super(AudioTrim, self).__init__(fn=trim_audio, **kwargs)

    def go(self, sample, **kwargs):
        self.action_fn(sample, **dict(self.params, **kwargs))


@register_action_fn
def trim_audio(sample, target_duration, extend=True, random_trim=False, tol=1e-6):
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
        super(SpectrogramToTensor, self).__init__(fn, is_augmentation, **kwargs)

    def go(self, sample, **kwargs):
        """converts sample.data from Spectrogram to Tensor"""
        # sample.data must be Spectrogram object
        # sample should have attributes: height, width, channels
        # use info from sample for desired shape and n channels
        kwargs.update(shape=[sample.height, sample.width], channels=sample.channels)
        sample.data = self.action_fn(sample.data, **dict(self.params, **kwargs))


@register_action_cls
class Overlay(Action):
    """Action Class for augmentation that overlays samples on eachother

    Overlay is a flavor of "mixup" augmentation, where two samples are
    overlayed on top of eachother. The samples are blended with a weighted
    average, where the weight may be chosen randomly from a range of values.

    In this implementation, the overlayed samples are chosen from a dataframe
    of audio files and labels. The dataframe must have the audio file paths as
    the index, and the labels as columns. The labels are used to choose
    overlayed samples based on an "overlay_class" argument.

    Args:
        overlay_df: dataframe of audio files (index) and labels to use for overlay
        update_labels (bool): if True, labels of sample are updated to include
            labels of overlayed sample
        criterion_fn: function that takes AudioSample and returns True or False
            - if True, perform overlay
            - if False, do not perform overlay
            Default is `always_true`, perform overlay on all samples

        See overlay() for **kwargs and default values

    """

    def __init__(self, is_augmentation=True, **kwargs):
        if "fn" in kwargs:
            kwargs.pop("fn")
        super(Overlay, self).__init__(
            overlay,
            is_augmentation=is_augmentation,
            **kwargs,
        )

        # self.returns_labels = True

        overlay_df = kwargs["overlay_df"].copy()  # copy to avoid modifying original
        overlay_df = overlay_df[~overlay_df.index.duplicated()]  # remove duplicates

        # warn the user if using "different" as overlay_class
        # and "different" is one of the model classes
        if (
            "different" in overlay_df.columns
            and "overlay_class" in kwargs
            and kwargs["overlay_class"] == "different"
        ):
            warnings.warn(
                "class name `different` was in columns, but using "
                "kwarg overlay_class='different' has specific behavior and will "
                "not specifically choose files from the `different` class. "
                "Consider renaming the `different` class. "
            )

        # move overlay_df from .params to its own attribute
        self.overlay_df = overlay_df
        self.params = self.params.drop("overlay_df")  # removes it from params Series

    def go(self, sample, **kwargs):
        self.action_fn(
            sample,
            overlay_df=self.overlay_df,
            **dict(self.params, **kwargs),
        )

    def to_dict(self):
        # don't save self.overlay_df since it might be huge and is not json friendly
        # also don't save criterion_fn, will default to always_true on reload
        # user will have to specify these after using from_dict
        d = super().to_dict(ignore_attributes=("overlay_df"))
        d["params"].pop("criterion_fn")
        return d

    @classmethod
    def from_dict(cls, dict):
        dict["params"]["overlay_df"] = pd.DataFrame()
        dict["params"]["criterion_fn"] = always_true
        instance = super().from_dict(dict)
        # since we don't have an overlay df, set this action to bypass mode
        instance.bypass = True
        warnings.warn(
            "Overlay class's .overlay_df will be None after loading from dict and `.criterion_fn` will be always_true(). "
            "Reset these attributes and set .bypass to False to use Overlay after loading with from_dict()."
        )
        assert isinstance(instance, cls)  # make sure its not the parent type?
        return instance


def always_true(x):
    return True


@register_action_fn
def overlay(
    sample,
    overlay_df,
    update_labels,
    overlay_class=None,
    overlay_prob=1,
    max_overlay_num=1,
    overlay_weight=0.5,
    criterion_fn=always_true,
):
    """iteratively overlay 2d samples on top of eachother

    Overlays (blends) image-like samples from overlay_df on top of
    the sample with probability `overlay_prob` until stopping condition.
    If necessary, trims overlay audio to the length of the input audio.

    Optionally provide `criterion_fn` which takes sample and returns True/False
    to determine whether to perform overlay on this sample.

    Overlays can be used in a few general ways:
        1. a separate df where any file can be overlayed (overlay_class=None)
        2. same df as training, where the overlay class is "different" ie,
            does not contain overlapping labels with the original sample
        3. same df as training, where samples from a specific class are used
            for overlays

    Args:
        sample: AudioSample with .labels: labels of the original sample
            and .preprocessor: the preprocessing pipeline
        overlay_df: a labels dataframe with audio files as the index and
            classes as columns

        update_labels: if True, add overlayed sample's labels to original sample
        overlay_class: how to choose files from overlay_df to overlay
            Options [default: None]:
            None - Randomly select any file from overlay_df
            "different" - Select a random file from overlay_df containing none
                of the classes this file contains
            specific class name - always choose files from this class
        overlay_prob: the probability of applying each subsequent overlay
        max_overlay_num: the maximum number of samples to overlay on original
            - for example, if overlay_prob = 0.5 and max_overlay_num=2,
                1/2 of samples will recieve 1 overlay and 1/4 will recieve an
                additional second overlay
        overlay_weight: a float > 0 and < 1, or a list of 2 floats [min, max]
            between which the weight will be randomly chosen. e.g. [0.1,0.7]
            An overlay_weight <0.5 means more emphasis on original sample.
        criterion_fn: function that takes AudioSample and returns True or False
            - if True, perform overlay
            - if False, do not perform overlay
            Default is `always_true`, perform overlay on all samples

    Returns:
        overlayed sample, (possibly updated) labels


    Example:
        check if sample is from a xeno canto file (has "XC" in name),
        and only perform overlay on xeno canto files
        ```
        def is_xc(audio_sample):
            return "XC" in Path(audio_sample.source).stem
        s=overlay(s, overlay_df, False, criterion_fn=is_xc)
        ```
    """

    # Use the criterion_fn to determine if we should perform overlay on this sample
    if not criterion_fn(sample):
        return sample  # no overlay, just return the original sample

    ##  INPUT VALIDATION ##
    assert (
        overlay_class in ["different", None] or overlay_class in overlay_df.columns
    ), (
        "overlay_class must be 'different' or None or in overlay_df.columns. "
        f"got {overlay_class}"
    )
    assert (overlay_prob <= 1) and (overlay_prob >= 0), (
        "overlay_prob" f"should be in range (0,1), was {overlay_prob}"
    )

    weight_error = f"overlay_weight should be between 0 and 1, was {overlay_weight}"

    if hasattr(overlay_weight, "__iter__"):
        assert (
            len(overlay_weight) == 2
        ), "must provide a float or a range of min,max values for overlay_weight"
        assert (
            overlay_weight[1] > overlay_weight[0]
        ), "second value must be greater than first for overlay_weight"
        for w in overlay_weight:
            assert w < 1 and w > 0, weight_error
    else:
        assert overlay_weight < 1 and overlay_weight > 0, weight_error

    if overlay_class is not None:
        assert (
            len(overlay_df.columns) > 0
        ), "overlay_df must have labels if overlay_class is specified"
        if overlay_class != "different":  # user specified a single class
            assert (
                overlay_df[overlay_class].sum() > 0
            ), "overlay_df did not contain positive labels for overlay_class"

    if len(overlay_df.columns) > 0 and sample.labels is not None:
        assert list(overlay_df.columns) == list(
            sample.labels.index
        ), "overlay_df mast have same columns as sample's _labels or no columns"

    ## OVERLAY ##
    # iteratively perform overlays until stopping condition
    # each time, there is an overlay_prob probability of another overlay
    # up to a max number of max_overlay_num overlays
    overlays_performed = 0

    while overlay_prob > np.random.uniform() and overlays_performed < max_overlay_num:
        try:
            # lets pick a sample based on rules
            if overlay_class is None:
                # choose any file from the overlay_df
                overlay_path = random.choice(overlay_df.index)

            elif overlay_class == "different":
                # Select a random file containing none of the classes this file contains
                # because the overlay_df might be huge and sparse, we randomly
                # choose row until one fits criterea rather than filtering overlay_df
                # TODO: revisit this choice
                good_choice = False
                attempt_counter = 0
                max_attempts = 100  # if we try this many times, raise error
                while (not good_choice) and (attempt_counter < max_attempts):
                    attempt_counter += 1

                    # choose a random sample from the overlay df
                    candidate_idx = random.randint(0, len(overlay_df) - 1)

                    # check if this candidate sample has zero overlapping labels
                    label_intersection = np.logical_and(
                        overlay_df.values[candidate_idx, :], sample.labels.values
                    )
                    good_choice = sum(label_intersection) == 0

                if not good_choice:  # tried max_attempts samples, none worked
                    raise ValueError(
                        f"No samples found with non-overlapping labels after {max_attempts} random draws"
                    )

                overlay_path = overlay_df.index[candidate_idx]

            else:
                # Select a random file from a class of choice (may be slow -
                # however, in the case of a fixed overlay class, we could
                # pass an overlay_df containing only that class)
                choose_from = overlay_df[overlay_df[overlay_class] == 1]
                overlay_path = np.random.choice(choose_from.index.values)

            # now we have picked a file to overlay (overlay_path)
            # we also know its labels, if we need them
            # TODO: this will be slow with large index but fast with numeric index, reset_index() somewhere
            overlay_sample = AudioSample.from_series(overlay_df.loc[overlay_path])

            # now we need to run the pipeline to do everything up until the Overlay step
            # create a preprocessor for loading the overlay samples
            # note that if there are multiple Overlay objects in a pipeline,
            # it will cut off the preprocessing of the overlayed sample before
            # the first Overlay object. This may or may not be the desired behavior,
            # but it will at least "work".
            overlay_sample = sample.preprocessor.forward(
                overlay_sample, break_on_type=Overlay
            )

            # the overlay_sample may have a different shape than the original sample
            # force them into the same shape so we can overlay
            if overlay_sample.data.shape != sample.data.shape:
                overlay_sample.data = torchvision.transforms.Resize(
                    sample.data.shape[1:]
                )(overlay_sample.data)

            # now we blend the two tensors together with a weighted average
            # Select weight of overlay; <0.5 means more emphasis on original sample
            # Supports uniform-random selection from a range of weights eg [0.1,0.7]
            weight = overlay_weight
            if hasattr(weight, "__iter__"):
                assert len(weight) == 2, (
                    f"overlay_weight must specify a single value or range of 2 values, "
                    f"got {overlay_weight}"
                )
                weight = random.uniform(weight[0], weight[1])

            # use a weighted sum to overlay (blend) the samples (arrays or tensors)
            sample.data = sample.data * (1 - weight) + overlay_sample.data * weight

            # update the labels with new classes
            if update_labels and len(overlay_sample.labels) > 0:
                # update labels as union of both files' labels
                sample.labels.values[:] = np.logical_or(
                    sample.labels.values, overlay_sample.labels.values
                ).astype(int)

            # overlay was successful, update count:
            overlays_performed += 1

        except PreprocessingError as ex:
            # don't try to load this sample again: remove from overlay df
            overlay_df = overlay_df.drop(overlay_path)
            warnings.warn(f"Invalid overlay sample: {overlay_path}")
            if len(overlay_df) < 1:
                raise ValueError("tried all overlay_df samples, none were safe") from ex

    return sample
