""" Class activation maps (CAM) for OpenSoundscape models"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import os
import warnings
from matplotlib.patches import Patch

from opensoundscape.utils import generate_opacity_colormaps


class CAM:
    """Object to hold and view Class Activation Maps, including guided backprop

    Stores activation maps as .activation_maps, and guided backprop as .gbp_cams

    each is a Series indexed by class
    """

    def __init__(self, base_image, activation_maps=None, gbp_maps=None):
        """Create object to store activation and backprop maps

        Create CAM object holding the original sample plus
        activation maps and guided back propogation maps for a set of classes

        Args:
            base_image: 3d tensor of shape [channel, w, h] containing original sample
            activation_maps:  pd.Series of [w,h] tensors representing spatial activation
                of some layer of a network, indexed by class name [default: None]
            gbp_maps: pd.Series of [channel, w, h] guided back propagation maps;
                indexed by class name

        Note: activation_maps and gbp_maps will be stored as Series indexed by classes
        """
        self.base_image = base_image.detach().cpu()
        self.activation_maps = activation_maps
        self.gbp_maps = gbp_maps

    def plot(
        self,
        class_subset=None,
        mode="activation",
        show_base=True,
        alpha=0.5,
        color_cycle=("#067bc2", "#43a43d", "#ecc30b", "#f37748", "#d56062"),
        interpolation="bilinear",
        figsize=None,
        plt_show=True,
        save_path=None,
        return_numpy=False,
    ):
        """Plot per-class activation maps, guided back propogations, or their products

        Args:
            class_subset: iterable of classes to visualize with activation maps
                - default `None` plots all classes
                - each item must be in the index of self.gbp_map / self.activation_maps
                - note that a class `None` is created by cnn.generate_cams() when classes are not
                specified during CNN.generate_cams()
            mode: str selecting which maps to visualize, one of:
                'activation' [default]: overlay activation map
                'backprop': overlay guided back propogation result
                'backprop_and_activation': overlay product of both maps
                None: do not overlay anything on the original sample
            show_base: if False, does not plot the image of the original sample
                [default: True]
            alpha: opacity of the activation map overlap [default: 0.5]
            color_cycle: iterable of colors activation maps
                - cycles through the list using one color per class
            interpolation: the interpolation method for the activation map
                [default: bilinear] see matplotlib.pyplot.imshow()
            figsize: the figure size for the plot [default: None]
            plt_show: if True, runs plt.show() [default: True]
            save_path: path to save image to [default: None does not save file]
        Returns:
            (fig, ax) of matplotlib figure, or np.array if return_numpy=True

        Note: if base_image does not have 3 channels, channels are averaged then copied
        across 3 RGB channels to create a greyscale image
        """
        fig, ax = plt.subplots(figsize=figsize)

        if show_base:  # plot image of sample
            # remove the first (batch) dimension
            # move the first dimension (Nchannels) to last dimension for imshow
            base_image = -self.base_image.permute(1, 2, 0)
            # if not 3 channels, average over channels and copy to 3 RGB channels
            if base_image.shape[2] != 3:
                base_image = base_image.mean(2).unsqueeze(2).tile([1, 1, 3])
            # ax.imshow(base_image, alpha=1)
            overlayed_image = np.array(base_image * 255, dtype=np.uint8)
        else:
            overlayed_image = None

        # Default is to show all classes contained in the cam:
        if class_subset is None:
            class_subset = (
                self.activation_maps.keys()
                if mode == "activation"
                else self.gbp_maps.keys()
            )

        # generate matplotlib color maps using specified color cycle
        colormaps = generate_opacity_colormaps(color_cycle)

        for i, target_class in enumerate(class_subset):
            # make the overlay mask for this class
            if mode == "activation":
                assert self.activation_maps is not None
                assert target_class in self.activation_maps, (
                    f"passed target class {target_class}, which is"
                    "not a class indexed in self.activation_maps!"
                )
                overlay = self.activation_maps[target_class]
            elif mode == "backprop":
                assert self.gbp_maps is not None
                assert target_class in self.gbp_maps, (
                    f"passed target class {target_class}, which is"
                    "not a class indexed in self.gbp_maps!"
                )
                overlay = self.gbp_maps[target_class]
            elif mode == "backprop_and_activation":
                assert self.activation_maps is not None
                assert self.gbp_maps is not None
                assert (
                    target_class in self.activation_maps
                    and target_class in self.gbp_maps
                ), (
                    f"passed target class {target_class}, which is"
                    "not a class indexed in self.gbp_maps!"
                )
                # we combine them using the product of the two maps
                am = self.activation_maps[target_class][
                    ..., np.newaxis
                ]  # add channel axis
                overlay = am * self.gbp_maps[target_class]
            elif mode is None:
                pass
            else:
                raise ValueError(
                    f"unsupported mode {mode}: choose "
                    "'activation', 'backprop', or 'backprop_and_activation'."
                )

            if mode is not None:
                colormap = colormaps[i % len(colormaps)]  # cycle through color list
                # Converts to RGB and scale to [0, 255]
                heatmap_rgb = colormap(overlay)[:, :, :3] * 255

                # copy overlay to 3 channels in 3rd dimension
                mask = overlay * alpha
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

                # use overlay as mask to combine heatmap with image
                overlayed_image = heatmap_rgb * mask + overlayed_image * (1 - mask)

        overlayed_image = np.array(overlayed_image, dtype=np.uint8)
        ax.imshow(overlayed_image, interpolation=interpolation)

        if mode is not None:
            ax.set_title(f"{mode} for classes {class_subset}")
        else:
            ax.set_title(f"sample without cam")

        # Create a legend for the layers
        colors = [color_cycle[i % len(color_cycle)] for i in range(len(class_subset))]
        legend_patches = [
            Patch(color=color, label=layer)
            for color, layer in zip(colors, class_subset)
        ]

        # Show the overlay image with the legend
        ax.legend(handles=legend_patches, loc="upper right")

        ax.axis("off")

        if save_path is not None:
            fig.savefig(save_path)

        if plt_show:
            # if plt.show() is desired, check that MPLBACKEND is available
            if os.environ.get("MPLBACKEND") is None:
                warnings.warn("MPLBACKEND is 'None' in os.environ. Skipping plot.")
            else:
                plt.show()

        if return_numpy:
            return overlayed_image
        return fig, ax

    def __repr__(self):
        return f"CAM()"
