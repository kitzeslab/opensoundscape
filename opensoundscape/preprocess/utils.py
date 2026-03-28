"""Utilities for preprocessing"""

import copy
import inspect
from matplotlib import pyplot as plt
import matplotlib
import numpy as np


class PreprocessingError(Exception):
    """Custom exception indicating that a Preprocessor pipeline failed"""


def get_args(func):
    """get list of arguments and default values from a function

    ignores 'kwargs' argument, which is included in inspect.signature.parameters"""
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if k != "kwargs"}


def get_reqd_args(func):
    """get list of required arguments from a function"""
    signature = inspect.signature(func)
    return [
        k
        for k, v in signature.parameters.items()
        if v.default is inspect.Parameter.empty and k != "kwargs"
    ]


def process_tensor_for_display(
    tensor,
    channel=None,
    normalize_from_range=[-1, 1],
    invert=False,
    clip=None,
):
    """process tensor for display as image

    Moves channel axis from first to third position, converts
    torch.Tensor to numpy array, rescales values from [min,max] to [0,1]

    Args:
        tensor: torch.Tensor of shape [c,w,h]
        channel: specify an integer to plot only one channel (axis 0)
            otherwise will return all channels
        normalize_from_range: list of [min,max] values to normalize tensor from
        invert: if true, flips value range via x=1-x
        clip: if specified, tuple of (min,max) to clip values to after normalization

    Returns:
        numpy array of shape [w,h] or [w,h,c]
    """

    tensor = copy.deepcopy(tensor)

    if normalize_from_range is not None:
        min_val, max_val = normalize_from_range
        tensor = (tensor - min_val) / (max_val - min_val)

    if invert:
        tensor = 1 - tensor

    # re-arrange dimensions for img plotting
    sample = tensor.detach().numpy().transpose([1, 2, 0])

    if clip is not None:
        min_clip, max_clip = clip
        sample = np.clip(sample, min_clip, max_clip)

    if channel is not None:
        sample = sample[:, :, channel]

    return sample


def show_tensor(
    tensor,
    channel=None,
    normalize_from_range=[-1, 1],
    invert=False,
    cmap=None,
    clip=[0, 1],
    axis=None,
):
    """helper function for displaying a sample as an image

    Args:
        tensor: torch.Tensor of shape [c,w,h]
        channel: specify an integer to plot only one channel, otherwise will
            attempt to plot all channels
        normalize_from_range: list of [min,max] values to normalize tensor from [default: [-1,1]]
        invert:
            if true, flips value range via x=1-x
        cmap: matplotlib colormap passed to plt.imshow()
            - if None, will choose 'Greys' if only one channel
        clip: if specified, tuple of (min,max) to clip values to after normalization
        axis: matplotlib axis to plot on, if None will create new figure
    """
    sample = process_tensor_for_display(
        tensor,
        channel=channel,
        normalize_from_range=normalize_from_range,
        invert=invert,
        clip=clip,
    )

    if cmap is None:
        # choose greyscale if only one channel
        if channel is not None or tensor.shape[0] == 1:
            cmap = "Greys"
        else:
            cmap = None

    if axis is None:
        axis = plt.subplot()
    # this avoids stretching the color range to the min/max of the tensor
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)

    axis.imshow(sample, cmap=cmap, norm=normalize)


def show_tensor_grid(
    tensors,
    columns,
    labels=None,
    channel=None,
    normalize_from_range=[-1, 1],
    invert=False,
    cmap=None,
    clip=[0, 1],
    axes=None,
    pad=0.05,  # outer margin
    gap=0.05,  # inner gap between images
    title_height=0.07,  # extra top margin for titles
):
    """Create a tightly packed image grid of tensors.

    Args:
        tensors: list of torch.Tensor objects to display
        columns: number of columns in the grid
        labels: optional list of titles for each tensor
        channel: specify an integer to plot only one channel, otherwise will
            attempt to plot all channels
        normalize_from_range: list of [min,max] values to normalize tensor from
        invert: if true, flips value range via x=1-x
        cmap: matplotlib colormap passed to plt.imshow()
            - if None, will choose 'Greys' if only one channel
        clip: if specified, tuple of (min,max) to clip values to after normalization
        axes: optional matplotlib axes to plot on, if None will create new figure
        pad: outer margin around the grid (fraction of figure size)
        gap: inner gap between images (fraction of figure size)
        title_height: extra top margin for titles (fraction of figure size)

    Returns:
        axes: numpy array of matplotlib axes objects
    """

    if labels is not None:
        assert len(labels) == len(tensors)

    n = len(tensors)
    n_rows = int(np.ceil(n / columns))

    if labels is None:
        title_height = 0.0

    if axes is None:
        fig, axes = plt.subplots(
            n_rows,
            columns,
            squeeze=False,
            constrained_layout=False,
        )

    axes_flat = axes.flatten()

    for i, ax in enumerate(axes_flat):
        ax.axis("off")
        if i < n:
            show_tensor(
                tensors[i],
                axis=ax,
                channel=channel,
                normalize_from_range=normalize_from_range,
                invert=invert,
                cmap=cmap,
                clip=clip,
            )
            if labels is not None:
                ax.set_title(labels[i], fontsize=8, pad=3)
        else:
            ax.set_visible(False)

    # Adjust spacing: minimal vertical/horizontal whitespace
    plt.subplots_adjust(
        left=pad,
        right=1 - pad,
        bottom=pad,
        top=1 - pad,
        wspace=gap,
        hspace=gap + title_height,
    )

    # Prevent shrinking from aspect ratio differences
    for ax in axes_flat[: len(tensors)]:
        ax.set_aspect("auto")

    return axes
