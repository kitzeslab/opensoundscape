"""Actions for augmentation and preprocessing pipelines

This module contains Action classes which act as the elements in
Preprocessor pipelines. Action classes have go(), on(), off(), and set()
methods. They take a single sample of a specific type and return the transformed
or augmented sample, which may or may not be the same type as the original.

See the preprocessor module and Preprocessing tutorial
for details on how to use and create your own actions.
"""
import numpy as np
import random
from torchvision import transforms
import torch
import warnings
import pandas as pd

from opensoundscape.audio import Audio
from opensoundscape.preprocess import tensor_augment as tensaug
from opensoundscape.preprocess.utils import PreprocessingError, get_args, get_reqd_args


class BaseAction:
    """Parent class for all Actions (used in Preprocessor pipelines)

    New actions should subclass this class.

    Subclasses should set `self.requires_labels = True` if go() expects (X,y)
    instead of (X). y is a row of a dataframe (a pd.Series) with index (.name)
    = original file path, columns=class names, values=labels (0,1). X is the
    sample, and can be of various types (path, Audio, Spectrogram, Tensor, etc).
    See Overlay for an example of an Action that uses labels.
    """

    def __init__(self):
        self.params = pd.Series(dtype=object)
        self.extra_args = []
        self.returns_labels = False
        self.is_augmentation = False
        self.bypass = False

    def __repr__(self):
        return (
            f"{'Bypassed ' if self.bypass else ''}"
            f"{'Augmentation ' if self.is_augmentation else ''}"
            "Action"
        )

    def go(self, x, **kwargs):
        return x

    def set(self, **kwargs):
        """only allow keys that exist in self.params"""
        unmatched_args = set(list(kwargs.keys())) - set(list(self.params.keys()))
        assert unmatched_args == set(
            []
        ), f"unexpected arguments: {unmatched_args}. The valid arguments and current values are: \n{self.params}"
        self.params.update(kwargs)

    def get(self, arg):
        return self.params[arg]


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

    def __init__(self, fn, is_augmentation=False, extra_args=[], **kwargs):
        super(Action, self).__init__()

        self.action_fn = fn
        self.is_augmentation = is_augmentation

        # args that vary for each sample, will be passed from preprocessor
        self.extra_args = extra_args

        # query action_fn for arguments and default values
        self.params = pd.Series(get_args(self.action_fn), dtype=object)

        # whether the first argument is 'self' or the incoming object,
        # we remove it from the params dict
        self.params = self.params[1:]

        # remove "extra_args" from self.params if they are present:
        # these sample-specific arguments will be passed to action.go()
        # directly, so they should not be part of the self.params dictionary
        self.params = self.params.drop([p for p in extra_args if p in self.params])

        # update self.params with any user-provided parameters
        self.set(**kwargs)

        # make sure all required args are given (skipping the first, which will be provided by go)
        unmatched_reqd_args = (
            set(get_reqd_args(self.action_fn)[1:])
            - set(list(kwargs.keys()))
            - set(self.extra_args)
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

    def go(self, x, **kwargs):
        # the syntax is the same regardless of whether
        # first argument is "self" (for a class method) or not
        # we pass self.params to kwargs along with any additional kwargs
        return self.action_fn(x, **dict(self.params, **kwargs))


class AudioClipLoader(Action):
    """Action to load clips from an audio file

    Loads an audio file or part of a file to an Audio object.
    Will load entire audio file if _start_time and _end_time are None.
    see Audio.from_file() for documentation.

    Args:
        see Audio.from_file()
    """

    def __init__(self, **kwargs):
        super(AudioClipLoader, self).__init__(
            Audio.from_file, extra_args=["_start_time", "_sample_duration"], **kwargs
        )
        # two params are replaced by "_start_time" and "_sample_duration"
        self.params = self.params.drop(["offset", "duration"])

    def go(self, path, _start_time, _sample_duration, **kwargs):
        offset = 0 if _start_time is None else _start_time
        # only trim to _sample_duration if _start_time is provided
        # ie, we are loading clips from a long audio file
        duration = None if _start_time is None else _sample_duration
        return self.action_fn(
            path, offset=offset, duration=duration, **dict(self.params, **kwargs)
        )


class AudioTrim(Action):
    """Action to trim/extend audio to desired length

    Args:
        see actions.trim_audio
    """

    def __init__(self, **kwargs):
        super(AudioTrim, self).__init__(
            trim_audio, extra_args=["_sample_duration"], **kwargs
        )


def trim_audio(audio, _sample_duration, extend=True, random_trim=False, tol=1e-5):
    """trim audio clips (Audio -> Audio)

    Trims an audio file to desired length
    Allows audio to be trimmed from start or from a random time
    Optionally extends audio shorter than clip_length with silence

    Args:
        audio: Audio object
        _sample_duration: desired final length (sec)
            - if None, no trim is performed
        extend: if True, clips shorter than _sample_duration are
            extended with silence to required length
        random_trim: if True, chooses a random segment of length _sample_duration
            from the input audio. If False, the file is trimmed from 0 seconds
            to _sample_duration seconds.
        tol: tolerance for considering a clip to be of the correct length (sec)

    Returns:
        trimmed audio
    """
    if len(audio.samples) == 0:
        raise ValueError("recieved zero-length audio")

    if _sample_duration is not None:
        if audio.duration() + tol <= _sample_duration:
            # input audio is not as long as desired length
            if extend:  # extend clip sith silence
                audio = audio.extend(_sample_duration)
            else:
                raise ValueError(
                    f"the length of the original file ({audio.duration()} "
                    f"sec) was less than the length to extract "
                    f"({_sample_duration} sec). To extend short "
                    f"clips, use extend=True"
                )
        if random_trim:
            # uniformly randomly choose clip time from full audio
            extra_time = audio.duration() - _sample_duration
            start_time = np.random.uniform() * extra_time
        else:
            start_time = 0

        end_time = start_time + _sample_duration
        audio = audio.trim(start_time, end_time)

    return audio


def torch_color_jitter(tensor, brightness=0.3, contrast=0.3, saturation=0.3, hue=0):
    """Wraps torchvision.transforms.ColorJitter

    (Tensor -> Tensor) or (PIL Img -> PIL Img)

    Args:
        tensor: input sample
        brightness=0.3
        contrast=0.3
        saturation=0.3
        hue=0

    Returns:
        modified tensor
    """
    transform = transforms.Compose(
        [
            transforms.ColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
            )
        ]
    )
    return transform(tensor)


def torch_random_affine(tensor, degrees=0, translate=(0.3, 0.1), fill=0):
    """Wraps for torchvision.transforms.RandomAffine

    (Tensor -> Tensor) or (PIL Img -> PIL Img)

    Args:
        tensor: torch.Tensor input saple
        degrees = 0
        translate = (0.3, 0.1)
        fill = 0-255, duplicated across channels

    Returns:
        modified tensor

    Note: If applying per-image normalization, we recommend applying
    RandomAffine after image normalization. In this case, an intermediate gray
    value is ~0. If normalization is applied after RandomAffine on a PIL image,
    use an intermediate fill color such as (122,122,122).
    """

    channels = tensor.shape[-3]
    fill = [fill] * channels

    transform = transforms.Compose(
        [transforms.RandomAffine(degrees=degrees, translate=translate, fill=fill)]
    )
    return transform(tensor)


def image_to_tensor(img, greyscale=False):
    """Convert PIL image to RGB or greyscale Tensor (PIL.Image -> Tensor)

    convert PIL.Image w/range [0,255] to torch Tensor w/range [0,1]

    Args:
        img: PIL.Image
        greyscale: if False, converts image to RGB (3 channels).
            If True, converts image to one channel.
    """
    if greyscale:
        img = img.convert("L")
    else:
        img = img.convert("RGB")

    transform = transforms.Compose([transforms.ToTensor()])
    return transform(img)


def scale_tensor(tensor, input_mean=0.5, input_std=0.5):
    """linear scaling of tensor values using torch.transforms.Normalize

    (Tensor->Tensor)

    WARNING: This does not perform per-image normalization. Instead,
    it takes as arguments a fixed u and s, ie for the entire dataset,
    and performs X=(X-input_mean)/input_std.

    Args:
        input_mean: mean of input sample pixels (average across dataset)
        input_std: standard deviation of input sample pixels (average across dataset)
        (these are NOT the target mu and sd, but the original mu and sd of img
        for which the output will have mu=0, std=1)

    Returns:
        modified tensor
    """
    transform = transforms.Compose([transforms.Normalize(input_mean, input_std)])
    return transform(tensor)


def time_mask(tensor, max_masks=3, max_width=0.2):
    """add random vertical bars over sample (Tensor -> Tensor)

    Args:
        tensor: input Torch.tensor sample
        max_masks: maximum number of vertical bars [default: 3]
        max_width: maximum size of bars as fraction of sample width

    Returns:
        augmented tensor
    """

    # convert max_width from fraction of sample to pixels
    max_width_px = int(tensor.shape[-1] * max_width)

    # add "batch" dimension expected by tensaug
    tensor = tensor.unsqueeze(0)

    # perform transform
    tensor = tensaug.time_mask(tensor, T=max_width_px, max_masks=max_masks)

    # remove "batch" dimension
    tensor = tensor.squeeze(0)

    return tensor


def frequency_mask(tensor, max_masks=3, max_width=0.2):
    """add random horizontal bars over Tensor

    Args:
        tensor: input Torch.tensor sample
        max_masks: max number of horizontal bars [default: 3]
        max_width: maximum size of horizontal bars as fraction of sample height

    Returns:
        augmented tensor
    """

    # convert max_width from fraction of sample to pixels
    max_width_px = int(tensor.shape[-2] * max_width)

    # add "batch" dimension expected by tensaug
    tensor = tensor.unsqueeze(0)

    # perform transform
    tensor = tensaug.freq_mask(tensor, F=max_width_px, max_masks=max_masks)

    # remove "batch" dimension
    tensor = tensor.squeeze(0)

    return tensor


def tensor_add_noise(tensor, std=1):
    """Add gaussian noise to sample (Tensor -> Tensor)

    Args:
        std: standard deviation for Gaussian noise [default: 1]

    Note: be aware that scaling before/after this action will change the
    effect of a fixed stdev Gaussian noise
    """
    noise = torch.empty_like(tensor).normal_(mean=0, std=std)
    return tensor + noise


class Overlay(Action):
    """Action Class for augmentation that overlays samples on eachother

    Required Args:
        overlay_df: dataframe of audio files (index) and labels to use for overlay
        update_labels (bool): if True, labels of sample are updated to include
            labels of overlayed sample

    See overlay() for other arguments and default values.
    """

    def __init__(self, is_augmentation=True, **kwargs):

        super(Overlay, self).__init__(
            overlay,
            is_augmentation=is_augmentation,
            extra_args=["_labels", "_preprocessor"],
            **kwargs,
        )

        self.returns_labels = True

        overlay_df = kwargs["overlay_df"]
        overlay_df = overlay_df[~overlay_df.index.duplicated()]  # remove duplicates

        # warn the user if using "different" as overlay_class and "different" is one of the model classes
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

        # move overlay_df from params to its own space, so that it doesn't display with print(params)
        self.overlay_df = overlay_df
        self.params = self.params.drop("overlay_df")  # removes it

    def go(self, x, **kwargs):
        return self.action_fn(
            x, overlay_df=self.overlay_df, **dict(self.params, **kwargs)
        )


def overlay(
    x,
    _labels,
    _preprocessor,
    overlay_df,
    update_labels,
    overlay_class=None,
    overlay_prob=1,
    max_overlay_num=1,
    overlay_weight=0.5,
):
    """iteratively overlay 2d samples on top of eachother

    Overlays (blends) image-like samples from overlay_df on top of
    the sample with probability `overlay_prob` until stopping condition.
    If necessary, trims overlay audio to the length of the input audio.

    Overlays can be used in a few general ways:
        1. a separate df where any file can be overlayed (overlay_class=None)
        2. same df as training, where the overlay class is "different" ie,
            does not contain overlapping labels with the original sample
        3. same df as training, where samples from a specific class are used
            for overlays

    Args:
        overlay_df: a labels dataframe with audio files as the index and
            classes as columns
        _labels: labels of the original sample
        _preprocessor: the preprocessing pipeline
        update_labels: if True, add overlayed sample's labels to original sample
        overlay_class: how to choose files from overlay_df to overlay
            Options [default: "different"]:
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

    Returns:
        overlayed sample, (possibly updated) labels

    """
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
                np.sum(overlay_df[overlay_class]) > 0
            ), "overlay_df did not contain positive labels for overlay_class"

    if len(overlay_df.columns) > 0:
        assert list(overlay_df.columns) == list(
            _labels.index
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
                        overlay_df.values[candidate_idx, :], _labels.values
                    )
                    good_choice = sum(label_intersection) == 0

                if not good_choice:  # tried max_attempts samples, none worked
                    raise ValueError(
                        "No samples found with non-overlapping labels after 100 random draws"
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
            overlay_row = overlay_df.loc[overlay_path]
            overlay_labels = overlay_row.values

            # now we need to run the pipeline to do everything up until the Overlay step
            # create a preprocessor for loading the overlay samples
            # note that if there are multiple Overlay objects in a pipeline,
            # it will cut off the preprocessing of the overlayed sample before
            # the first Overlay object. This may or may not be the desired behavior,
            # but it will at least "work".
            x2, sample_info = _preprocessor.forward(overlay_row, break_on_type=Overlay)

            # now we blend the two tensors together with a weighted average
            # Select weight of overlay; <0.5 means more emphasis on original sample
            # Supports uniform-random selection from a range of weights eg [0.1,0.7]
            weight = overlay_weight
            if hasattr(weight, "__iter__"):
                assert (
                    len(weight) == 2
                ), f"overlay_weight must specify a single value or range of 2 values, got {overlay_weight}"
                weight = random.uniform(weight[0], weight[1])

            # use a weighted sum to overlay (blend) the samples
            x = x * (1 - weight) + x2 * weight

            # update the labels with new classes
            if update_labels and len(overlay_labels) > 0:
                # update labels as union of both files' labels
                _labels.values[:] = np.logical_or(
                    _labels.values, overlay_labels
                ).astype(int)

            # overlay was successful, update count:
            overlays_performed += 1

        except PreprocessingError:
            # don't try to load this sample again: remove from overlay df
            overlay_df = overlay_df.drop(overlay_path)
            warnings.warn(f"unsafe overlay sample: {overlay_path}")
            if len(overlay_df) < 1:
                raise ValueError("tried all overlay_df samples, none were safe")

    return x, _labels
