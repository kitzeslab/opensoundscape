"""Utilities for preprocessing"""

import copy
import inspect
from matplotlib import pyplot as plt
import matplotlib


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
    tensor, channel=None, transform_from_zero_centered=True, invert=False
):

    tensor = copy.deepcopy(tensor)

    if transform_from_zero_centered:
        tensor = tensor / 2 + 0.5

    if invert:
        tensor = 1 - tensor

    # re-arrange dimensions for img plotting
    sample = tensor.detach().numpy().transpose([1, 2, 0])

    if channel is not None:
        sample = sample[:, :, channel]

    return sample


def show_tensor(
    tensor,
    channel=None,
    transform_from_zero_centered=True,
    invert=False,
    cmap=None,
):
    """helper function for displaying a sample as an image

    Args:
        tensor: torch.Tensor of shape [c,w,h] with values centered around zero
        channel: specify an integer to plot only one channel, otherwise will
            attempt to plot all channels
        transform_from_zero_centered: if True, transforms values from [-1,1] to [0,1]
        invert:
            if true, flips value range via x=1-x
        cmap: matplotlib colormap passed to plt.imshow()
            - if None, will choose 'Greys' if only one channel
    """
    sample = process_tensor_for_display(
        tensor,
        channel=channel,
        transform_from_zero_centered=transform_from_zero_centered,
        invert=invert,
    )

    if cmap is None:
        # choose greyscale if only one channel
        if channel is not None or tensor.shape[0] == 1:
            cmap = "Greys"
        else:
            cmap = None

    # this avoids stretching the color range to the min/max of the tensor
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)

    plt.imshow(sample, cmap=cmap, norm=normalize)


def show_tensor_grid(
    tensors,
    columns,
    channel=None,
    transform_from_zero_centered=True,
    invert=False,
    labels=None,
):
    """create image of nxn tensors

    Args:
        tensors:list of samples
        columns: number of columns in grid
        labels: title of each subplot
        for other args, see show_tensor()
    """
    if labels is not None:
        assert len(labels) == len(tensors)

    fig, _ = plt.subplots(figsize=[5 * columns, 5 * (len(tensors) // columns + 1)])
    for i, s in enumerate(tensors):
        ax = plt.subplot(len(tensors) // columns + 1, columns, i + 1)
        show_tensor(s, channel, transform_from_zero_centered, invert)
        if labels is not None:
            ax.set_title(labels[i])
    return fig
