#!/usr/bin/env python3
"""
Augmentations and transforms for torch.Tensors
"""

import random


def freq_mask(spec, F=30, max_masks=3, replace_with_zero=False):
    """draws horizontal bars over the image

    Args:
        spec: a torch.Tensor representing a spectrogram
        F: maximum frequency-width of bars in pixels
        max_masks: maximum number of bars to draw
        replace_with_zero: if True, bars are 0s, otherwise, mean img value

    Returns:
        Augmented tensor
    """
    cloned = spec.clone()
    batch_size = cloned.shape[0]
    num_mel_channels = cloned.shape[2]

    num_masks = random.randint(1, max_masks)

    for _ in range(num_masks):

        f = [random.randrange(0, F) for _ in range(batch_size)]
        f_zero = [
            random.randrange(0, num_mel_channels - f[i])
            for i, _ in enumerate(range(batch_size))
        ]

        mask_end = [
            random.randrange(f_zero[i], f_zero[i] + f[i])
            if f_zero[i] != (f_zero[i] + f[i])
            else (f_zero[i] + F)
            for i, _ in enumerate(range(batch_size))
        ]

        if replace_with_zero:
            mask_value = [0.0] * batch_size
        else:
            mask_value = cloned.mean(dim=(1, 2, 3))

        for i in range(len(cloned)):
            cloned[i, :, f_zero[i] : mask_end[i], :] = mask_value[i]

    return cloned


def time_mask(spec, T=40, max_masks=3, replace_with_zero=False):
    """draws vertical bars over the image

    Args:
        spec: a torch.Tensor representing a spectrogram
        T: maximum time-width of bars in pixels
        max_masks: maximum number of bars to draw
        replace_with_zero: if True, bars are 0s, otherwise, mean img value

    Returns:
        Augmented tensor
    """
    cloned = spec.clone()
    batch_size = cloned.shape[0]
    len_spectro = cloned.shape[3]

    num_masks = random.randint(1, max_masks)

    for _ in range(num_masks):

        t = [random.randrange(0, T) for _ in range(batch_size)]
        t_zero = [
            random.randrange(0, len_spectro - t[i])
            for i, _ in enumerate(range(batch_size))
        ]

        mask_end = [
            random.randrange(t_zero[i], t_zero[i] + t[i])
            if t_zero[i] != (t_zero[i] + t[i])
            else (t_zero[i] + T)
            for i, _ in enumerate(range(batch_size))
        ]

        if replace_with_zero:
            mask_value = [0.0] * batch_size
        else:
            mask_value = cloned.mean(dim=(1, 2, 3))

        for i in range(len(cloned)):
            cloned[i, :, :, t_zero[i] : mask_end[i]] = mask_value[i]

    return cloned
