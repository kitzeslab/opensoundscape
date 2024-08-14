"""preprocessing and augmentation functions 

these can be passed to the Action class (action_fn=...) to create a preprocessing action
"""

import random

import torch
import torchvision

from opensoundscape.audio import Audio, mix
from opensoundscape.preprocess.actions import (
    register_action_fn,
)
from opensoundscape.preprocess import tensor_augment


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
