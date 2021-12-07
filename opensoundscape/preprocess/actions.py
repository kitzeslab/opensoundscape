"""Actions for augmentation and preprocessing pipelines

This module contains Action classes which act as the elements in
Preprocessor pipelines. Action classes have go(), on(), off(), and set()
methods. They take a single sample of a specific type and return the transformed
or augmented sample, which may or may not be the same type as the original.

See the preprocessor module and Preprocessing tutorial
for details on how to use and create your own actions.
"""
import numpy as np
from PIL import Image
import random
from pathlib import Path
from time import time
import os
from torchvision import transforms
import torch
from torchvision.utils import save_image
import warnings

from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram, MelSpectrogram
from opensoundscape.preprocess import tensor_augment as tensaug
from opensoundscape.preprocess.utils import PreprocessingError


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


class BaseAction:
    """Parent class for all Actions (used in Preprocessor pipelines)

    New actions should subclass this class.

    Subclasses should set `self.requires_labels = True` if go() expects (X,y)
    instead of (X). y is a row of a dataframe (a pd.Series) with index (.name)
    = original file path, columns=class names, values=labels (0,1). X is the
    sample, and can be of various types (path, Audio, Spectrogram, Tensor, etc).
    See ImgOverlay for an example of an Action that uses labels.
    """

    def __init__(self, **kwargs):
        # pass any parameters as kwargs
        self.params = {}
        self.params.update(kwargs)
        self.bypass = False  # if True, no action is performed
        self.requires_labels = False

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

    Loads an audio file or part of a file.
    see Audio.from_file() for documentation.

    Args:
        see Audio.from_file

    Note: default sample_rate=None means use file's sample rate, don't resample
    """

    def go(self, path):
        return Audio.from_file(path, **self.params)


class AudioClipLoader(BaseAction):
    """Action to load only a specific segment of an audio file

    Loads an audio file or part of a file.
    see Audio.from_file() for documentation.

    Args:
        see Audio.from_file

    Note: default sample_rate=None means use file's sample rate, don't resample
    """

    def __init__(self, **kwargs):
        super(AudioClipLoader, self).__init__(**kwargs)
        self.requires_clip_times = True

    def go(self, path, start_time, end_time):
        return Audio.from_file(
            path, offset=start_time, duration=end_time - start_time, **self.params
        )


class AudioTrimmer(BaseAction):
    """Action child class for trimming audio (Audio -> Audio)

    Trims an audio file to desired length
    Allows audio to be trimmed from start or from a random time
    Optionally extends audio shorter than clip_length with silence


    Args:
        audio_length: desired final length (sec); if None, no trim is performed
        extend: if True, clips shorter than audio_length are
            extended with silence to required length
        random_trim: if True, a random segment of length audio_length is chosen
            from the input audio. If False, the file is trimmed from 0 seconds
            to audio_length seconds.
    """

    def __init__(self, **kwargs):
        super(AudioTrimmer, self).__init__(**kwargs)
        # default parameters
        self.params["extend"] = False
        self.params["random_trim"] = False
        self.params["audio_length"] = None  # can trim all audio to fixed length

        # add parameters passed to __init__
        self.params.update(kwargs)

    def go(self, audio):
        if self.params["audio_length"] is not None:
            if audio.duration() <= self.params["audio_length"]:
                # input audio is not as long as desired length
                if self.params["extend"]:  # extend clip by looping
                    audio = audio.extend(self.params["audio_length"])
                else:
                    raise ValueError(
                        f"the length of the original file ({audio.duration()} "
                        f"sec) was less than the length to extract "
                        f"({self.params['audio_length']} sec). To extend short "
                        f"clips, use extend=True"
                    )
            if self.params["random_trim"]:
                # uniformly randomly choose clip time from full audio
                extra_time = audio.duration() - self.params["audio_length"]
                start_time = np.random.uniform() * extra_time
            else:
                start_time = 0

            end_time = start_time + self.params["audio_length"]
            audio = audio.trim(start_time, end_time)

        return audio


class AudioToSpectrogram(BaseAction):
    """Action child class for Spectrogram.from_audio() (Audio -> Spectrogram)

    see spectrogram.Spectrogram.from_audio for documentation

    Args:
        window_type="hann": see scipy.signal.spectrogram docs for description of window parameter
        window_samples=512: number of audio samples per spectrogram window (pixel)
        overlap_samples=256: number of samples shared by consecutive windows
        decibel_limits = (-100,-20) : limit the dB values to (min,max) (lower values set to min, higher values set to max)
        dB_scale=True : If True, rescales values to decibels, x=10*log10(x)
            - if dB_scale is False, decibel_limits is ignored
    """

    def go(self, audio):
        return Spectrogram.from_audio(audio, **self.params)


class AudioToMelSpectrogram(BaseAction):
    """Action child class for MelSpectrogram.from_audio()
    (Audio -> MelSpectrogram)

    see spectrogram.MelSpectrogram.from_audio for documentation

    Args:
        n_mels: Number of mel bands to generate [default: 128]
            Note: n_mels should be chosen for compatibility with the
            Spectrogram parameter `window_samples`. Choosing a value
            `> ~ window_samples/10` will result in zero-valued rows while
            small values blend rows from the original spectrogram.
        window_type: The windowing function to use [default: "hann"]
        window_samples: n samples per window [default: 512]
        overlap_samples: n samples shared by consecutive windows [default: 256]
        htk: use HTK mel-filter bank instead of Slaney, see Librosa docs [default: False]
        norm='slanley': mel filter bank normalization, see Librosa docs
        dB_scale=True: If True, rescales values to decibels, x=10*log10(x)
            - if dB_scale is False, decibel_limits is ignored
    """

    def go(self, audio):
        return MelSpectrogram.from_audio(audio, **self.params)


class SpectrogramBandpass(BaseAction):
    """Action class for Spectrogram.bandpass() (Spectrogram -> Spectrogram)

    see opensoundscape.spectrogram.Spectrogram.bandpass() for documentation

    To bandpass the spectrogram from 1kHz to 5Khz:
    action = SpectrogramBandpass(1000,5000)

    Args:
        min_f: low frequency in Hz for bandpass
        max_f: high frequency in Hz for bandpass
        out_of_bounds_ok: if False, raises error if min or max beyond spec limits
    """

    def go(self, spectrogram):
        return spectrogram.bandpass(**self.params)


class SpecToImg(BaseAction):
    """Action class to transform Spectrogram to PIL image

    (Spectrogram -> PIL.Image)

    Args:
        destination: a file path (string)
        shape=None: image dimensions for 1 channel, (height, width)
        mode="RGB": RGB for 3-channel color or "L" for 1-channel grayscale
        colormap=None: (str) Matplotlib color map name (if None, greyscale)
    """

    # shape=(self.width, self.height), mode="L" during construction
    def go(self, spectrogram):
        return spectrogram.to_image(**self.params)


class SaveTensorToDisk(BaseAction):
    """save a torch Tensor to disk (Tensor -> Tensor)

    Requires x_labels because the index of the label-row (.name)
    gives the original file name for this sample.

    Uses torchvision.utils.save_image. Creates save_path dir if it doesn't exist

    Args:
        save_path: a directory where tensor will be saved
    """

    def __init__(self, save_path, **kwargs):
        super(SaveTensorToDisk, self).__init__(**kwargs)
        self.requires_labels = True
        # make this directory if it doesnt exist yet
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def go(self, x, x_labels):
        """we require x_labels because the .name gives origin file name"""
        filename = os.path.basename(x_labels.name) + f"_{time()}.png"
        path = Path.joinpath(self.save_path, filename)
        save_image(x, path)
        return x, x_labels


class TorchColorJitter(BaseAction):
    """Action class for torchvision.transforms.ColorJitter

    (Tensor -> Tensor) or (PIL Img -> PIL Img)

    Args:
        brightness=0.3
        contrast=0.3
        saturation=0.3
        hue=0
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
    """Action class for torchvision.transforms.RandomAffine

    (Tensor -> Tensor) or (PIL Img -> PIL Img)

    Args:
        degrees = 0
        translate = (0.3, 0.1)
        fill = (0, 0, 0)  # 0-255

    Note: If applying per-image normalization, we recommend applying
    RandomAffine after image normalization. In this case, an intermediate gray
    value is ~0. If normalization is applied after RandomAffine on a PIL image,
    use an intermediate fill color such as (122,122,122).
    """

    def __init__(self, **kwargs):
        super(TorchRandomAffine, self).__init__(**kwargs)

        # default parameters
        self.params["degrees"] = 0
        self.params["translate"] = (0.3, 0.1)
        self.params["fill"] = (0, 0, 0)  # 0-255

        # add parameters passed to __init__
        self.params.update(kwargs)

    def go(self, x):
        transform = transforms.Compose([transforms.RandomAffine(**self.params)])
        return transform(x)


class ImgToTensor(BaseAction):
    """Convert PIL image to RGB Tensor (PIL.Image -> Tensor)

    convert PIL.Image w/range [0,255] to torch Tensor w/range [0,1]
    converts image to RGB (3 channels)
    """

    def go(self, x):
        x = x.convert("RGB")
        transform = transforms.Compose([transforms.ToTensor()])
        return transform(x)


class ImgToTensorGrayscale(BaseAction):
    """Convert PIL image to greyscale Tensor (PIL.Image -> Tensor)

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

    Params:
        mean=0.5
        std=0.5
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
    """Time warp is an experimental augmentation that creates a tilted image.

    Args:
        warp_amount: use higher values for more skew and offset (experimental)

    Note: this augmentation reduces the image to greyscale and duplicates the
    result across the 3 channels.

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
        max_width: maximum width of horizontal bars as fraction of image width
        [default: 0.2]
    """

    def __init__(self, **kwargs):
        super(TimeMask, self).__init__(**kwargs)

        # default parameters
        self.params["max_masks"] = 3
        self.params["max_width"] = 0.2

        # add parameters passed to __init__
        self.params.update(kwargs)

    def go(self, x):

        # convert max_width from fraction of image to pixels
        max_width_px = int(x.shape[-1] * self.params["max_width"])

        # add "batch" dimension expected by tensaug
        x = x.unsqueeze(0)

        # perform transform
        x = tensaug.time_mask(x, T=max_width_px, max_masks=self.params["max_masks"])

        # remove "batch" dimension
        x = x.squeeze(0)

        return x


class FrequencyMask(BaseAction):
    """add random horizontal bars over image

    Args:
        max_masks: max number of horizontal bars [default: 3]
        max_width: maximum size of horizontal bars as fraction of image height
    """

    def __init__(self, **kwargs):
        super(FrequencyMask, self).__init__(**kwargs)

        # default parameters
        self.params["max_masks"] = 3
        self.params["max_width"] = 0.2

        # add parameters passed to __init__
        self.params.update(kwargs)

    def go(self, x):
        """torch Tensor in, torch Tensor out"""

        # convert max_width from fraction of image to pixels
        max_width_px = int(x.shape[-2] * self.params["max_width"])

        # add "batch" dimension expected by tensaug
        x = x.unsqueeze(0)

        # perform transform
        x = tensaug.freq_mask(x, F=max_width_px, max_masks=self.params["max_masks"])

        # remove "batch" dimension
        x = x.squeeze(0)

        return x


class TensorAugment(BaseAction):
    """combination of 3 augmentations with hard-coded parameters

    time warp, time mask, and frequency mask

    use (bool) time_warp, time_mask, freq_mask to turn each on/off

    Note: This function reduces the image to greyscale then duplicates the
    image across the 3 channels
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

    Overlays can be used in a few general ways:
        1. a separate df where any file can be overlayed (overlay_class=None)
        2. same df as training, where the overlay class is "different" ie,
            does not contain overlapping labels with the original sample
        3. same df as training, where samples from a specific class are used
            for overlays

    Args:
        overlay_df: a labels dataframe with audio files as the index and
            classes as columns
        audio_length: length in seconds of original audio sample
        loader_pipeline: the preprocessing pipeline to load audio -> spec
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
                1/2 of images will recieve 1 overlay and 1/4 will recieve an
                additional second overlay
        overlay_weight: can be a float between 0-1 or range of floats (chooses
            randomly from within range) such as [0.1,0.7].
            An overlay_weight <0.5 means more emphasis on original image.

    """

    def __init__(
        self, overlay_df, audio_length, loader_pipeline, update_labels, **kwargs
    ):
        super(ImgOverlay, self).__init__()

        assert max(overlay_df.index.duplicated()) == 0, (
            "index of overlay_df "
            "must be unique. contained duplicate indices. to drop duplicates "
            "use: overlay_df = overlay_df[~overlay_df.index.duplicated()]"
        )

        self.requires_labels = True

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

        if "different" in overlay_df.columns:
            warnings.warn(
                "class name `different` was in columns, but using "
                "kwarg overlay_class='different' has specific behavior and will "
                "not specifically choose files from the `different` class."
            )

    def go(self, x, x_labels):
        """Overlay images from overlay_df"""

        overlay_class = self.params["overlay_class"]
        df = self.params["overlay_df"]

        # input validation
        assert overlay_class in ["different", None] or overlay_class in df.columns, (
            "overlay_class must be 'different' or None or in df.columns"
            f"got {overlay_class}"
        )
        assert (self.params["overlay_prob"] <= 1) and (
            self.params["overlay_prob"] >= 0
        ), ("overlay_prob" f"should be in range (0,1), was {overlay_weight}")

        wts = self.params["overlay_weight"]
        weight_error = f"overlay_weight should be between 0 and 1, was {wts}"

        if hasattr(wts, "__iter__"):
            assert (
                len(wts) == 2
            ), "must provide a float or a range of min,max values for overlay_weight"
            assert (
                wts[1] > wts[0]
            ), "second value must be greater than first for overlay_weight"
            for w in wts:
                assert w < 1 and w > 0, weight_error
        else:
            assert wts < 1 and wts > 0, weight_error

        if overlay_class is not None:
            assert (
                len(df.columns) > 0
            ), "overlay_df must have labels if overlay_class is specified"
            if overlay_class != "different":  # user specified a single class
                assert (
                    np.sum(df[overlay_class]) > 0
                ), "overlay_df did not contain positive labels for overlay_class"

        if len(df.columns) > 0:
            assert list(df.columns) == list(
                x_labels.index
            ), "overlay_df mast have same columns as x_labels or no columns"

        # iteratively perform overlays until stopping condition
        # each time, there is an overlay_prob probability of another overlay
        # up to a max number of max_overlay_num overlays
        overlays_performed = 0
        while (
            self.params["overlay_prob"] > np.random.uniform()
            and overlays_performed < self.params["max_overlay_num"]
        ):
            overlays_performed += 1

            # lets pick a sample based on rules
            if overlay_class is None:
                # choose any file from the overlay_df
                overlay_path = random.choice(df.index)

            elif overlay_class == "different":
                # Select a random file containing none of the classes this file contains
                # because the df might be huge and sparse, we randomly
                # choose row until one fits criterea rather than filtering df
                good_choice = False
                attempt_counter = 0
                max_attempts = 100  # if we try this many times, raise error
                while (not good_choice) and (attempt_counter < max_attempts):
                    attempt_counter += 1

                    # choose a random sample from the overlay df
                    candidate_idx = random.randint(0, len(df) - 1)

                    # check if this candidate sample has zero overlapping labels
                    label_intersection = np.logical_and(
                        df.values[candidate_idx, :], x_labels.values
                    )
                    good_choice = sum(label_intersection) == 0

                if not good_choice:  # tried max_attempts samples, none worked
                    warnings.warn("No samples found with non-overlapping labels")
                    continue

                overlay_path = df.index[candidate_idx]

            else:
                # Select a random file from a class of choice (may be slow -
                # however, in the case of a fixed overlay class, we could
                # pass an overlay_df containing only that class)
                choose_from = df[df[overlay_class] == 1]
                overlay_path = np.random.choice(choose_from.index.values)

            # now we have picked a file to overlay (overlay_path)
            # we also know its labels, if we need them
            overlay_labels = df.loc[overlay_path].values

            # update the labels with new classes
            if self.params["update_labels"] and len(overlay_labels) > 0:
                # update labels as union of both files' labels
                x_labels.values[:] = np.logical_or(
                    x_labels.values, overlay_labels
                ).astype(int)

            # now we need to run the pipeline to get from audio path -> image
            x2 = overlay_path
            for action in self.loader_pipeline:
                if action.bypass:
                    continue
                if action.requires_labels:  # this never happens
                    x2, _ = action.go(x2, overlay_labels)
                else:
                    x2 = action.go(x2)

            # now we blend the two images together
            # Select weight of overlay; <0.5 means more emphasis on original image
            # allows random selection from a range of weights eg [0.1,0.7]
            weight = self.params["overlay_weight"]
            if hasattr(weight, "__iter__"):
                weight = random.uniform(weight[0], weight[1])

            # use a weighted sum to overlay (blend) the images
            x = Image.blend(x, x2, weight)

        return x, x_labels
