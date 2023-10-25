"""Actions for augmentation and preprocessing pipelines

This module contains Action classes which act as the elements in
Preprocessor pipelines. Action classes have go(), on(), off(), and set()
methods. They take a single sample of a specific type and return the transformed
or augmented sample, which may or may not be the same type as the original.

See the preprocessor module and Preprocessing tutorial
for details on how to use and create your own actions.
"""
import random
import warnings
import numpy as np
import torchvision
import torch
import pandas as pd

from opensoundscape.audio import Audio, mix
from opensoundscape.preprocess import tensor_augment as tensaug
from opensoundscape.preprocess.utils import PreprocessingError, get_args, get_reqd_args
from opensoundscape.sample import AudioSample
from opensoundscape.spectrogram import Spectrogram


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
        self.params = pd.Series(dtype="object")
        self.returns_labels = False
        self.is_augmentation = False
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
        self.params.update(pd.Series(kwargs, dtype=object))

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
        super(AudioClipLoader, self).__init__(Audio.from_file, **kwargs)
        # two params are provided by sample.start_time and sample.duration
        self.params = self.params.drop(["offset", "duration"])

    def go(self, sample, **kwargs):
        offset = 0 if sample.start_time is None else sample.start_time
        duration = None if sample.duration is None else sample.duration
        sample.data = self.action_fn(
            sample.data, offset=offset, duration=duration, **dict(self.params, **kwargs)
        )


class AudioTrim(Action):
    """Action to trim/extend audio to desired length

    Args:
        see actions.trim_audio
    """

    def __init__(self, **kwargs):
        super(AudioTrim, self).__init__(trim_audio, **kwargs)

    def go(self, sample, **kwargs):
        self.action_fn(sample, **dict(self.params, **kwargs))


def trim_audio(sample, extend=True, random_trim=False, tol=1e-5):
    """trim audio clips (Audio -> Audio)

    Trims an audio file to desired length
    Allows audio to be trimmed from start or from a random time
    Optionally extends audio shorter than clip_length with silence

    Args:
        sample: AudioSample with .data=Audio object, .duration as sample duration
        extend: if True, clips shorter than sample.duration are
            extended with silence to required length
        random_trim: if True, chooses a random segment of length sample.duration
            from the input audio. If False, the file is trimmed from 0 seconds
            to sample.duration seconds.
        tol: tolerance for considering a clip to be of the correct length (sec)

    Returns:
        trimmed audio
    """
    audio = sample.data

    if len(audio.samples) == 0:
        raise ValueError("recieved zero-length audio")

    if sample.target_duration is not None:
        if audio.duration + tol <= sample.target_duration:
            # input audio is not as long as desired length
            if extend:  # extend clip sith silence
                audio = audio.extend_to(sample.target_duration)
            else:
                raise ValueError(
                    f"the length of the original file ({audio.duration} "
                    f"sec) was less than the length to extract "
                    f"({sample.target_duration} sec). To extend short "
                    f"clips, use extend=True"
                )
        if random_trim:
            # uniformly randomly choose clip time from full audio
            extra_time = audio.duration - sample.target_duration
            start_time = np.random.uniform() * extra_time
        else:
            start_time = 0

        end_time = start_time + sample.target_duration
        audio = audio.trim(start_time, end_time)

        # update the sample
        sample.data = audio
        if sample.start_time is None:
            sample.start_time = start_time
        else:
            sample.start_time += start_time
        sample.duration = sample.target_duration

    return sample


class SpectrogramToTensor(Action):
    """Action to create Tesnsor of desired shape from Spectrogram

    calls .to_image on sample.data, which should be type Spectrogram

    **kwargs are passed to Spectrogram.to_image()

    """

    def __init__(self, fn=Spectrogram.to_image, is_augmentation=False, **kwargs):
        kwargs.update(dict(return_type="torch"))  # return a tensor, not PIL.Image
        super(SpectrogramToTensor, self).__init__(fn, is_augmentation, **kwargs)

    def go(self, sample, **kwargs):
        """converts sample.data from Spectrogram to Tensor"""
        # sample.data must be Spectrogram object
        # sample should have attributes: height, width, channels
        # use info from sample for desired shape and n channels
        kwargs.update(shape=[sample.height, sample.width], channels=sample.channels)
        sample.data = self.action_fn(sample.data, **dict(self.params, **kwargs))


def audio_random_gain(audio, dB_range=(-30, 0), clip_range=(-1, 1)):
    """Applies a randomly selected gain level to an Audio object

    Gain is selected from a uniform distribution in the range dB_range

    Args:
        audio: an Audio object
        dB_range: (min,max) decibels of gain to apply
            - dB gain applied is chosen from a uniform random
            distribution in this range

    Returns: Audio object with gain applied
    """
    gain = random.uniform(dB_range[0], dB_range[1])
    return audio.apply_gain(dB=gain, clip_range=clip_range)


def audio_add_noise(audio, noise_dB=-30, signal_dB=0, color="white"):
    """Generates noise and adds to audio object

    Args:
        audio: an Audio object
        noise_dB: number or range: dBFS of noise signal generated
            - if number, crates noise with `dB` dBFS level
            - if (min,max) tuple, chooses noise `dBFS` randomly
            from range with a uniform distribution
        signal_dB: dB (decibels) gain to apply to the incoming Audio
            before mixing with noise [default: -3 dB]
            - like noise_dB, can specify (min,max) tuple to
            use random uniform choice in range

    Returns: Audio object with noise added
    """
    if hasattr(noise_dB, "__iter__"):
        # choose noise level randomly from dB range
        noise_dB = random.uniform(noise_dB[0], noise_dB[1])
    # otherwise, it should just be a number

    if hasattr(signal_dB, "__iter__"):
        # choose signal level randomly from dB range
        signal_dB = random.uniform(signal_dB[0], signal_dB[1])
    # otherwise, it should just be a number

    noise = Audio.noise(
        duration=audio.duration,
        sample_rate=audio.sample_rate,
        color=color,
        dBFS=noise_dB,
    )

    return mix([audio, noise], gain=[signal_dB, 0])


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
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ColorJitter(
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

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomAffine(
                degrees=degrees, translate=translate, fill=fill
            )
        ]
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

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
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
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.Normalize(input_mean, input_std)]
    )
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


def always_true(x):
    return True


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
        super(Overlay, self).__init__(
            overlay,
            is_augmentation=is_augmentation,
            **kwargs,
        )

        self.returns_labels = True

        overlay_df = kwargs["overlay_df"]
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

        # move overlay_df from params to its own space so that it doesn't display with print(params)
        self.overlay_df = overlay_df
        self.params = self.params.drop("overlay_df")  # removes it

    def go(self, sample, **kwargs):
        self.action_fn(
            sample,
            overlay_df=self.overlay_df,
            **dict(self.params, **kwargs),
        )


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
