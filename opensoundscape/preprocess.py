"""preprocess.py: utilities for augmentation and preprocessing pipelines"""
# todo: add parameters into docstrings for easy access using "?"
import numpy as np
from PIL import Image
import random
from pathlib import Path
from time import time
import os
from torchvision import transforms
import torch
from torchvision.utils import save_image

from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram


class ParameterRequiredError(Exception):
    """Raised if action.go(x) called when action.go(x,x_labels) is required"""


### Audio transforms ###
class BaseAction:
    """Parent class for all Pipeline Elements"""

    def __init__(self, **kwargs):
        # pass any parameters as kwargs
        self.params = {}
        self.params.update(kwargs)
        self.bypass = False  # if off, no action is performed

    def go(self, x, **kwargs):
        return x

    def set(self, **kwargs):
        self.params.update(kwargs)

    def get(self, arg):
        return self.params[arg]

    def off(self):
        self.bypass = True

    def on(self):
        self.bypass = False


class ActionContainer:
    """this is an empty object which holds instances of Action child-classes

    the instances each has a go() method. Set parameters with set(param=value,...)
    """

    def __init__(self):
        pass

    def list_actions(self):
        return list(vars(self).keys())


class AudioLoader(BaseAction):
    """Action child class for Audio.from_file()

    default sample_rate is None (use file's sample rate)
    """

    def __init__(self, **kwargs):
        super(AudioLoader, self).__init__(**kwargs)
        # default parameters
        self.params["sample_rate"] = None

        # add parameters passed to __init__
        self.params.update(kwargs)

    def go(self, path):
        return Audio.from_file(path, **self.params)


class AudioTrimmer(BaseAction):
    """Action child class for trimming audio"""

    def __init__(self, **kwargs):
        super(AudioTrimmer, self).__init__(**kwargs)
        # default parameters
        self.params["extend_short_clips"] = False
        self.params["random_trim"] = False
        self.params["audio_length"] = None  # trim all audio to fixed length

        # add parameters passed to __init__
        self.params.update(kwargs)

    def go(self, audio):
        if self.params["audio_length"] is not None:
            if self.params["random_trim"]:
                # from opensoundscape.preprocess import random_audio_trim
                # we don't have default
                audio = random_audio_trim(
                    audio,
                    self.params["audio_length"],
                    self.params["extend_short_clips"],
                )
            else:
                audio = audio.trim(0, self.params["audio_length"])

        return audio


class AudioToSpectrogram(BaseAction):
    """Action child class for Audio.from_file()"""

    def go(self, audio):
        return Spectrogram.from_audio(audio, **self.params)


class SpecToImg(BaseAction):
    """Action child class for spec to image"""

    # shape=(self.width, self.height), mode="L" during construction
    def go(self, spectrogram):
        return spectrogram.to_image(**self.params)


class SaveTensorToDisk(BaseAction):
    """save a torch Tensor to disk"""

    def __init__(self, save_path, **kwargs):
        super(SaveTensorToDisk, self).__init__(**kwargs)
        # make this directory if it doesnt exist yet
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def go(self, x, x_labels=None):
        """we require x_labels because the .name gives origin file name"""
        if x_labels is None:
            raise ParameterRequiredError("Pass x_labels to SaveImgToDisk.go()")

        filename = os.path.basename(x_labels.name) + f"_{time()}.png"
        path = Path.joinpath(self.save_path, filename)
        save_image(x, path)
        return x, x_labels


class TorchColorJitter(BaseAction):
    """Action child class for torchvision.transforms.ColorJitter"""

    def __init__(self, **kwargs):
        super(TorchColorJitter, self).__init__(**kwargs)

        # default parameters
        self.params["brightness"] = 0.3
        self.params["contrast"] = 0.3
        self.params["saturation"] = 0.3
        self.params["hue"] = 0

        # add parameters passed to __init__
        self.params.update(kwargs)

    def go(self, x):
        transform = transforms.Compose([transforms.ColorJitter(**self.params)])
        return transform(x)


class TorchRandomAffine(BaseAction):
    """Action child class with torchvision.transforms.RandomAffine

    can act on PIL image or torch Tensor

    note: we recommend applying RandomAffine after image
    normalization. In this case, an intermediate grey value is 0.
    If normalization is applied after RandomAffine on a PIL image, use an
    intermediate fill color such as (122,122,122).
    """

    def __init__(self, **kwargs):
        super(TorchRandomAffine, self).__init__(**kwargs)

        # default parameters
        self.params["degrees"] = 0
        self.params["translate"] = (0.2, 0.03)
        self.params["fill"] = (0, 0, 0)  # 0-255

        # add parameters passed to __init__
        self.params.update(kwargs)

    def go(self, x):
        transform = transforms.Compose([transforms.RandomAffine(**self.params)])
        return transform(x)


class ImgToTensor(BaseAction):
    """convert PIL.Image to torch Tensor"""

    def go(self, x):
        x = x.convert("RGB")
        transform = transforms.Compose([transforms.ToTensor()])
        return transform(x)


class TensorNormalize(BaseAction):
    """torchvision.transforms.Normalize """

    def __init__(self, **kwargs):
        super(TensorNormalize, self).__init__(**kwargs)

        # default parameters
        self.params["mean"] = 0.5
        self.params["std"] = 0.5

        # add parameters passed to __init__
        self.params.update(kwargs)

    def go(self, x):
        normalize = transforms.Normalize(self.params["mean"], self.params["std"])
        transform = transforms.Compose([normalize])
        return transform(x)


class TensorAugment(BaseAction):
    """perform tensor augmentations

    such as time warp, time mask, and frequency mask
    for now, parameters are hard-coded.

    """

    def go(self, x):
        from opensoundscape.torch import tensor_augment as tensaug

        """torch Tensor in, torch Tensor out"""
        # X is currently shape [3, width, height]
        # Take to shape [1, 1, width, height] for use with `tensor_augment`
        # (tensor_augment is design for batch of [1,w,h] tensors)
        # since batch size is '1' (we are only doing one at a time)
        x = x[0, :, :].unsqueeze(0).unsqueeze(0)  # was: X = X[:,0].unsqueeze(1)
        x = tensaug.time_warp(x.clone(), W=10)
        x = tensaug.time_mask(x, T=50, max_masks=5)
        x = tensaug.freq_mask(x, F=50, max_masks=5)

        # remove "batch" dimension
        x = x[0, :]
        # Transform shape from 1 dimension to 3 dimensions
        x = torch.cat([x] * 3, dim=0)  # dim=1)

        return x


class TensorAddNoise(BaseAction):
    """random white noise added to sample"""

    def __init__(self, **kwargs):
        super(TensorAddNoise, self).__init__(**kwargs)

        # default parameters
        self.params["std"] = 1

        # add parameters passed to __init__
        self.params.update(kwargs)

    def go(self, x):
        noise = torch.empty_like(x).normal_(mean=0, std=self.params["std"])
        return x + noise  # do we need to clamp to a range?


class ImgOverlay(BaseAction):
    # iteratively overlay images with overlay_prob until stopping condition
    def __init__(
        self, overlay_df, audio_length, loader_pipeline, update_labels, **kwargs
    ):
        super(ImgOverlay, self).__init__()

        if overlay_df is None:
            raise ParameterRequiredError("ImgOverlay requires overlay_df")

        # required arguments
        self.params["overlay_df"] = overlay_df
        self.params["audio_length"] = audio_length
        self.params["update_labels"] = update_labels
        self.loader_pipeline = loader_pipeline

        # default overlay parameters
        self.params["overlay_class"] = "different"  # or None or specific class
        self.params["overlay_prob"] = 1
        self.params["max_overlay_num"] = 1
        self.params["overlay_weight"] = 0.5  # allows float or range

        # parameters from **kwargs
        self.params.update(kwargs)

    def go(self, x, x_labels=None):
        """Overlay an image from overlay_df

        if overlay_class is None: select any file from overlay_df
        if overlay_class is different
        Select a random file from a different class. Trim if necessary to the
        same length as the given image. Overlay the images on top of each other
        with a weight.

        overlay_weight: can be a float in (0,1) or range of floats (chooses
        randomly from within range) such as [0.1,0.7].
        An overlay_weight <0.5 means more emphasis on original image.

        update_labels: if True, add labels of overlayed class to returned labels
        """
        overlay_class = self.params["overlay_class"]
        df = self.params["overlay_df"]

        # (always) enforce requirement of x_label
        if x_labels is None:  # and overlay_class is not None:
            raise ParameterRequiredError("ImgOverlay requires x_labels")

        overlays_performed = 0
        while (
            self.params["overlay_prob"] > np.random.uniform()
            and overlays_performed < self.params["max_overlay_num"]
        ):
            overlays_performed += 1

            # we want to overlay an image. lets pick one based on rules
            if overlay_class is None:
                # choose any file from the overlay_df
                overlay_path = random.choice(df.index)

            elif overlay_class == "different":
                # Select a random file containing none of the classes this file contains
                good_choice = (
                    False  # keep picking random ones until we satisfy criteria
                )
                while not good_choice:
                    candidate_idx = random.randint(0, len(df) - 1)
                    # check if this choice meets criteria
                    labels_overlap = sum(df.values[candidate_idx, :] * x_labels.values)
                    good_choice = int(labels_overlap) == 0

                # TODO: check that this is working as expected
                overlay_path = df.index[candidate_idx]

            else:
                # Select a random file from a class of choice (may be slow)
                # However, in the case of a fixed overlay class, we could
                # pass an overlay_df containing only that class
                choose_from = df[df[overlay_class] == 1]
                overlay_path = np.random.choice(choose_from.index.values)

            # now we have picked a file to overlay (overlay_path)
            # we also know its labels, if we need them
            overlay_labels = df.loc[overlay_path].values

            # update the labels with new classes
            if self.params["update_labels"]:
                # update labels as union of both files' labels
                new_labels = iter(x_labels.values + overlay_labels)
                x_labels = x_labels.apply(lambda x: min(1, int(next(new_labels))))

            # now we need to run the pipeline to get from audio path -> image
            x2 = overlay_path
            for pipeline_element in self.loader_pipeline:
                x2 = pipeline_element.go(x2)
            overlay_image = x2

            # removed Miao's blur code:
            # blur_r = np.random.randint(0, 8) / 10
            # overlay_image = overlay_image.filter(ImageFilter.GaussianBlur(radius=blur_r))

            # now we blend the two images together
            # Select weight of overlay; <0.5 means more emphasis on original image
            # allows random selection from a range of weights eg [0.1,0.7]
            weight = self.params["overlay_weight"]
            if type(weight) in (list, tuple, np.ndarray):
                if len(weight) != 2:
                    raise ValueError("Weight must be float or have length 2")
                weight = random.uniform(weight[0], weight[1])
            else:
                weight = self.params["overlay_weight"]

            # use a weighted sum to overlay (blend) the images
            x = Image.blend(x, overlay_image, weight)

        return x, x_labels


def random_audio_trim(audio, duration, extend_short_clips=False):
    """randomly select a subsegment of Audio of fixed length

    randomly chooses a time segment of the entire Audio object to cut out,
    from the set of all possible start times that allow a complete extraction

    Args:
        Audio: input Audio object
        length: duration in seconds of the trimmed Audio output

    Returns:
        Audio object trimmed from original
    """
    input_duration = len(audio.samples) / audio.sample_rate
    if duration > input_duration:
        if not extend_short_clips:
            raise ValueError(
                f"the length of the original file ({input_duration} sec) was less than the length to extract ({duration} sec). To extend short clips, use extend_short_clips=True"
            )
        else:
            return audio.extend(duration)
    extra_time = input_duration - duration
    start_time = np.random.uniform() * extra_time
    return audio.trim(start_time, start_time + duration)


### PIL.Image transforms ###


def time_split(img, seed=None):
    """Given a PIL.Image, rotate it

    Choose a random new starting point and append the first section to the end.
    For example, if `h` chosen

    abcdefghijklmnop
           ^
    hijklmnop + abcdefg

    Args:
        img: A PIL.Image

    Returns:
        A PIL.Image
    """

    if not isinstance(img, Image.Image):
        raise TypeError("Expects PIL.Image as input")

    if seed:
        random.seed(seed)

    width, _ = img.size
    idx = random.randint(0, width)
    arr = np.array(img)
    rotated = np.hstack([arr[:, idx:, :], arr[:, 0:idx, :]])
    return Image.fromarray(rotated)
