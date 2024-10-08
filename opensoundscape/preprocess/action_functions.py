"""preprocessing and augmentation functions 

these can be passed to the Action class (action_fn=...) to create a preprocessing action that applies the function to a sample
"""

import random

import librosa
import torch
import torchvision

from opensoundscape.audio import Audio, mix, concat
from opensoundscape.preprocess import tensor_augment, io

ACTION_FN_DICT = dict()


def list_action_fns():
    """return list of available action function keyword strings
    (can be used to initialize Action class)
    """
    return list(ACTION_FN_DICT.keys())


def register_action_fn(action_fn):
    """add function to ACTION_FN_DICT

    this allows us to recreate the Action class with a named action_fn

    see also: ACTION_DICT (stores list of named classes for preprocessing)
    """
    # register the model in dictionary
    ACTION_FN_DICT[io.build_name(action_fn)] = action_fn
    # return the function
    return action_fn


@register_action_fn
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


@register_action_fn
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


@register_action_fn
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


@register_action_fn
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


@register_action_fn
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


@register_action_fn
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


@register_action_fn
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

    # add "batch" dimension expected by tensor_augment
    tensor = tensor.unsqueeze(0)

    # perform transform
    tensor = tensor_augment.time_mask(tensor, T=max_width_px, max_masks=max_masks)

    # remove "batch" dimension
    tensor = tensor.squeeze(0)

    return tensor


@register_action_fn
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

    # add "batch" dimension expected by tensor_augment
    tensor = tensor.unsqueeze(0)

    # perform transform
    tensor = tensor_augment.freq_mask(tensor, F=max_width_px, max_masks=max_masks)

    # remove "batch" dimension
    tensor = tensor.squeeze(0)

    return tensor


@register_action_fn
def tensor_add_noise(tensor, std=1):
    """Add gaussian noise to sample (Tensor -> Tensor)

    Args:
        std: standard deviation for Gaussian noise [default: 1]

    Note: be aware that scaling before/after this action will change the
    effect of a fixed stdev Gaussian noise
    """
    noise = torch.empty_like(tensor).normal_(mean=0, std=std)
    return tensor + noise


@register_action_fn
def pcen(s, **kwargs):
    return s._spawn(spectrogram=librosa.pcen(S=s.spectrogram, **kwargs))


@register_action_fn
def random_wrap_audio(audio, probability=0.5, max_shift=None):
    """Randomly splits the audio into two parts, swapping their order

    useful as a "time shift" augmentation when extra audio beyond the bounds is not available

    Args:
        audio: an Audio object
        probability: probability of performing the augmentation
        max_shift: max number of seconds to shift, default None means no limit
    """
    if random.random() > probability:
        # don't augment
        return audio

    # if max_shift is None, allow splitting anywhere
    max_shift = max_shift or audio.duration
    # split audio into two parts
    split_time = random.uniform(0, max_shift)
    audio1 = audio.trim(0, split_time)
    audio2 = audio.trim(split_time, audio.duration)
    return concat([audio2, audio1])


@register_action_fn
def audio_time_mask(
    audio, max_masks=10, max_width=0.02, noise_dBFS=-15, noise_color="white"
):
    """randomly replace time slices with  noise

    Args:
        audio: input Audio object
        max_masks: maximum number of white noise time masks [default: 10]
        max_width: maximum size of bars as fraction of sample width [default: 0.02]
        noise_dBFS & noise_color: see Audio.noise() `dBFS` and `color` args

    Returns:
        augmented Audio object
    """

    # convert max_width from fraction of sample to seconds
    max_width_seconds = audio.duration * max_width

    # generate white noise segments and random start times
    from opensoundscape.audio import Audio
    import numpy as np

    n_masks = np.random.randint(0, max_masks + 1)
    mask_lens = np.random.uniform(0, max_width_seconds, n_masks)
    # randomly choose start positions by divvying up the non-masked space
    unmasked_time = audio.duration - mask_lens.sum()
    splits = [0] + list(np.sort(np.random.uniform(0, 1, n_masks) * unmasked_time))
    unmasked_segment_lens = np.array(splits[1:]) - np.array(splits[:-1])
    unmasked_segment_starts = [0]
    t = 0
    for i in range(n_masks):
        # skip forward by unmasked length + mask len
        t += unmasked_segment_lens[i] + mask_lens[i]
        unmasked_segment_starts.append(t)
    # doesn't include the last one, we'll just get the end of the sample instead
    unmasked_segment_ends = list(
        np.array(unmasked_segment_starts[:-1]) + np.array(unmasked_segment_lens)
    )

    samples = []
    for i in range(n_masks):
        samples.extend(
            audio.trim(unmasked_segment_starts[i], unmasked_segment_ends[i]).samples
        )
        samples.extend(
            Audio.noise(
                duration=mask_lens[i],
                sample_rate=audio.sample_rate,
                color=noise_color,
                dBFS=noise_dBFS,
            ).samples
        )
    # add the last segment of original audio, making sure we end up with correct total number of samples
    samples.extend(audio.samples[len(samples) - len(audio.samples) :])

    return audio._spawn(samples=samples)


@register_action_fn
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

    # add "batch" dimension expected by tensor_augment
    tensor = tensor.unsqueeze(0)

    # perform transform
    tensor = tensor_augment.freq_mask(tensor, F=max_width_px, max_masks=max_masks)

    # remove "batch" dimension
    tensor = tensor.squeeze(0)

    return tensor
