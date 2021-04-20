"""preprocess.py: utilities for augmentation and preprocessing pipelines"""
# todo: add parameters into docstrings for easy access using "?"
# TODO: documentation for actions should include parameters or reference to docs
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
from opensoundscape.preprocess import tensor_augment as tensaug


class ParameterRequiredError(Exception):
    """Raised if action.go(x) called when action.go(x,x_labels) is required"""


class ActionContainer:
    """this is a container object which holds instances of Action child-classes

    the Actions it contains each have .go(), .on(), .off(), .set(), .get()

    The actions are un-ordered and may not all be used. In preprocessor objects
    such as AudioToSpectrogramPreprocessor, Actions from the action
    container are listed in a pipeline(list), which defines their order of use.

    To add actions to the container: action_container.loader = AudioLoader()
    To set parameters of actions: action_container.loader.set(param=value,...)

    Methods: list_actions()
    """

    def __init__(self):
        pass

    def list_actions(self):
        return list(vars(self).keys())


### Audio transforms ###
class BaseAction:
    """Parent class for all Pipeline Elements

    New actions should subclass this class.
    """

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


class AudioLoader(BaseAction):
    """Action child class for Audio.from_file() (path -> Audio)

    Loads an audio file, see Audio.from_file() for parameters.
    (sample_rate=None, resample_type="kaiser_fast", max_duration=None)

    Note: default sample_rate=None means use file's sample rate, don't resample
    """

    def go(self, path):
        return Audio.from_file(path, **self.params)


class AudioTrimmer(BaseAction):
    """Action child class for trimming audio (Audio -> Audio)

    Trims an audio file to desired length, from start or random segment

    Params:
        audio_length: desired final length (sec); if None, no trim is performed
        extend_short_clips: if True, clips shorter than audio_length are
            extended by looping
        random_trim: if True, a random segment of length audio_length is chosen
            from the input audio. If False, the file is trimmed from 0 seconds
            to audio_length seconds.
    """

    def __init__(self, **kwargs):
        super(AudioTrimmer, self).__init__(**kwargs)
        # default parameters
        self.params["extend_short_clips"] = False
        self.params["random_trim"] = False
        self.params["audio_length"] = None  # can trim all audio to fixed length

        # add parameters passed to __init__
        self.params.update(kwargs)

    def go(self, audio):
        if self.params["audio_length"] is not None:
            # TODO: might want to move this functionality to Audio.random_trim

            if audio.duration() <= self.params["audio_length"]:
                # input audio is not as long as desired length
                if self.params["extend_short_clips"]:  # extend clip by looping
                    audio = audio.extend(self.params["audio_length"])
                else:
                    raise ValueError(
                        f"the length of the original file ({audio.duration()} sec) was less than the length to extract ({self.params['audio_length']} sec). To extend short clips, use extend_short_clips=True"
                    )
            if self.params["random_trim"]:
                extra_time = input_duration - duration
                start_time = np.random.uniform() * extra_time
            else:
                start_time = 0

            end_time = start_time + self.params["audio_length"]
            audio = audio.trim(start_time, end_time)

        return audio


class AudioToSpectrogram(BaseAction):
    """Action child class for Audio.from_file() (Audio -> Spectrogram)

    see spectrogram.Spectrogram for parameters/docs
    ()
    """

    def go(self, audio):
        return Spectrogram.from_audio(audio, **self.params)


class SpectrogramBandpass(BaseAction):
    """Action class for Spectrogram.bandpass() (Spectrogram -> Spectrogram)

    To bandpass the spectrogram from 1kHz to 5Khz:
    action = SpectrogramBandpass(1000,5000)

    Args: min_f, max_f
    see opensoundscape.spectrogram.Spectrogram.bandpass() for documentation

    Spectrogram in, Spectrogram out
    """

    def go(self, spectrogram):
        return spectrogram.bandpass(**self.params)


class SpecToImg(BaseAction):
    """Action child class for spec to image (Spectrogram -> PIL Image)

    Spectrogram in, PIL.Image out"""

    # shape=(self.width, self.height), mode="L" during construction
    def go(self, spectrogram):
        return spectrogram.to_image(**self.params)


class SaveTensorToDisk(BaseAction):
    """save a torch Tensor to disk (Tensor -> Tensor)"""

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
    """Action class for torchvision.transforms.ColorJitter

    (Tensor -> Tensor) or (PIL Img -> PIL Img)
    """

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
    """Action class with torchvision.transforms.RandomAffine

    (Tensor -> Tensor) or (PIL Img -> PIL Img)

    note: we recommend applying RandomAffine after image
    normalization. In this case, an intermediate gray value is ~0.
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
    """(PIL.Image -> Tensor)

    convert PIL.Image w/range [0,255] to torch Tensor w/range [0,1]
    converts image to RGB (3 channels)
    """

    def go(self, x):
        x = x.convert("RGB")
        transform = transforms.Compose([transforms.ToTensor()])
        return transform(x)


class ImgToTensorGrayscale(BaseAction):
    """(PIL.Image -> Tensor)

    convert PIL.Image w/range [0,255] to torch Tensor w/range [0,1]
    converts image to grayscale (1 channel)
    """

    def go(self, x):
        x = x.convert("L")
        transform = transforms.Compose([transforms.ToTensor()])
        return transform(x)


class TensorNormalize(BaseAction):
    """torchvision.transforms.Normalize (WARNING: FIXED shift and scale)

    (Tensor->Tensor)

    WARNING: This does not perform per-image normalization. Instead,
    it takes as arguments a fixed u and s, ie for the entire dataset,
    and performs X=(X-u)/s.
    """

    def __init__(self, **kwargs):
        super(TensorNormalize, self).__init__(**kwargs)

        # default parameters
        self.params["mean"] = 0.5
        self.params["std"] = 0.5
        # these are NOT the target mu and sd, but the assumed mu and sd of img->
        # performs X= (X-mu)/sd

        # add parameters passed to __init__
        self.params.update(kwargs)

    def go(self, x):
        normalize = transforms.Normalize(self.params["mean"], self.params["std"])
        transform = transforms.Compose([normalize])
        return transform(x)


class TimeWarp(BaseAction):
    """perform tensor augmentations

    such as time warp, time mask, and frequency mask
    Args:
        warp_amount: use higher values for more skew and offset (experimental)
    """

    def __init__(self, **kwargs):
        super(TimeWarp, self).__init__(**kwargs)

        # default parameters
        self.params["warp_amount"] = 5

        # add parameters passed to __init__
        self.params.update(kwargs)

    def go(self, x):

        # add "batch" dimension to tensor and use just first channel
        x = x[0, :, :].unsqueeze(0).unsqueeze(0)
        # perform transform
        x = tensaug.time_warp(x.clone(), W=self.params["warp_amount"])
        # remove "batch" dimension
        x = x[0, :]
        # Copy 1 channel to 3 RGB channels
        x = torch.cat([x] * 3, dim=0)  # dim=1)
        return x


class TimeMask(BaseAction):
    """add random vertical bars over image (Tensor -> Tensor)

    Args:
        max_masks: maximum number of bars [default: 3]
        max_width_px: maximum width of bars in pixels [default: 40]
    """

    def __init__(self, **kwargs):
        super(TimeMask, self).__init__(**kwargs)

        # default parameters
        self.params["max_masks"] = 3
        self.params["max_width_px"] = 40

        # add parameters passed to __init__
        self.params.update(kwargs)

    def go(self, x):

        # add "batch" dimension to tensor and use just first channel
        x = x[0, :, :].unsqueeze(0).unsqueeze(0)
        # perform transform
        x = tensaug.time_mask(
            x, T=self.params["max_width_px"], max_masks=self.params["max_masks"]
        )
        # remove "batch" dimension
        x = x[0, :]
        # Copy 1 channel to 3 RGB channels
        x = torch.cat([x] * 3, dim=0)
        return x


class FrequencyMask(BaseAction):
    """add random horizontal bars over image
    #TODO: should it use fraction of img instead of pixels?

    initialize with **kwargs parameters, or
    use .set(**kwargs) to update parameters

    Parameters:
        max_masks: max number of horizontal bars [default: 3]
        max_width_px: maximum height of horizontal bars in pixels [default: 40]
    """

    def __init__(self, **kwargs):
        super(FrequencyMask, self).__init__(**kwargs)

        # default parameters
        self.params["max_masks"] = 3
        self.params["max_width_px"] = 40

        # add parameters passed to __init__
        self.params.update(kwargs)

    def go(self, x):
        """torch Tensor in, torch Tensor out"""

        # add "batch" dimension to tensor and use just first channel
        x = x[0, :, :].unsqueeze(0).unsqueeze(0)
        # perform transform
        x = tensaug.freq_mask(
            x, F=self.params["max_width_px"], max_masks=self.params["max_masks"]
        )
        # remove "batch" dimension
        x = x[0, :]
        # Copy 1 channel to 3 RGB channels
        x = torch.cat([x] * 3, dim=0)
        return x


class TensorAugment(BaseAction):
    """combination of 3 augmentations with hard-coded parameters

    time warp, time mask, and frequency mask

    use (bool) time_warp, time_mask, freq_mask to turn each on/off
    """

    def __init__(self, **kwargs):
        super(TensorAugment, self).__init__(**kwargs)

        # default parameters
        self.params["time_warp"] = True
        self.params["time_mask"] = True
        self.params["freq_mask"] = True

        # add parameters passed to __init__
        self.params.update(kwargs)

    def go(self, x):
        """torch Tensor in, torch Tensor out"""
        # add "batch" dimension to tensor and keep just first channel
        x = x[0, :, :].unsqueeze(0).unsqueeze(0)  # was: X = X[:,0].unsqueeze(1)
        x = tensaug.time_warp(x.clone(), W=10)
        x = tensaug.time_mask(x, T=50, max_masks=5)
        x = tensaug.freq_mask(x, F=50, max_masks=5)
        # remove "batch" dimension
        x = x[0, :]
        # Copy 1 channel to 3 RGB channels
        x = torch.cat([x] * 3, dim=0)  # dim=1)

        return x


class TensorAddNoise(BaseAction):
    """Add gaussian noise to sample (Tensor -> Tensor)

    Args:
        std: standard deviation for Gaussian noise [default: 1]

    Note: be aware that scaling before/after this action will change the
    effect of a fixed stdev Gaussian noise
    """

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
    """iteratively overlay images on top of eachother

    Overlays images from overlay_df on top of the sample with probability
    overlay_prob until stopping condition.
    If necessary, trims overlay audio to the length of the input audio.
    Overlays the images on top of each other with a weight.

    Args:
        overlay_df: a labels dataframe with audio files as the index and
            classes as columns
        overlay_class: how to choose files from overlay_df to overlay
            Options [default: "different"]:
            None - Randomly select any file from overlay_df
            "different" - Select a random file from overlay_df containing none
                of the classes this file contains
            specific class name - always choose files from this class
        overlay_weight: can be a float between 0-1 or range of floats (chooses
            randomly from within range) such as [0.1,0.7].
            An overlay_weight <0.5 means more emphasis on original image.
        update_labels: if True, add labels of overlayed class to returned labels

    #TODO: raise warning for class name "different"

    """

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
        """Overlay images from overlay_df"""

        assert overlay_class in ["different", None] + df.columns, (
            "overlay_class must be 'different' or None or in df.columns"
            f"got {overlay_class}"
        )
        assert (overlay_prob <= 1) and (overlay_prob >= 0), (
            "overlay_prob" f"should be in range (0,1), was {overlay_weight}"
        )
        assert overlay_weight < 1 and overlay_weight > 0, (
            "overlay_weight" f"should be between 0 and 1, was {overlay_weight}"
        )

        overlay_class = self.params["overlay_class"]
        df = self.params["overlay_df"]

        # (always) enforce requirement of x_label
        if x_labels is None:  # and overlay_class is not None:
            # TODO: this way of doing it makes error handling ugly
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


# def random_audio_trim(audio, duration, extend_short_clips=False):
#     """randomly select a subsegment of Audio of fixed length
#
#     randomly chooses a time segment of the entire Audio object to cut out,
#     from the set of all possible start times that allow a complete extraction
#
#     Args:
#         Audio: input Audio object
#         length: duration in seconds of the trimmed Audio output
#
#     Returns:
#         Audio object trimmed from original
#     """
#     input_duration = len(audio.samples) / audio.sample_rate
#     if duration > input_duration:
#         if not extend_short_clips:
#             raise ValueError(
#                 f"the length of the original file ({input_duration} sec) was less than the length to extract ({duration} sec). To extend short clips, use extend_short_clips=True"
#             )
#         else:
#             return audio.extend(duration)
#     extra_time = input_duration - duration
#     start_time = np.random.uniform() * extra_time
#     return audio.trim(start_time, start_time + duration)
