""" Class activation maps (CAM) for OpenSoundscape models"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class ActivationMap:
    """Base class for activation maps
    This is a container for the activation map of a given class for a given input.
    """

    def __init__(self, base_image, activation_map=None, gbp_maps=None):
        """Create object to store activation and backprop maps

        Create ActivationMap object holding the original sample plus
        activation maps and guided back propogation maps for a set of classes

        Args:
            base_image: 3d tensor of shape [channel, w, h] containing original sample
            activation_map: [w,h] tensor representing spatial activation
                of some layer of a network [default: None]
            gbp_maps: pd.Series of [channel, w, h] guided back propagation maps;
                with class name as index

        Note: activation_maps and gbp_maps will be stored as Series indexed  by classes
        """
        self.base_image = base_image

        # initialize activation maps and guided back propogation maps
        # as series indexed by classes
        self.activation_map = activation_map
        self.gbp_maps = gbp_maps

    def plot(
        self,
        target_class=None,
        mode="activation_map",
        show_base=True,
        alpha=0.5,
        cmap="jet",
        interpolation="bilinear",
        figsize=None,
        plt_show=True,
        save_path=None,
    ):
        """Plot the activation map, guided back propogation, or their product
        Args:
            target_class: which class's gpb_map to visualize
                - must be in the index of self.gbp_map; only used when plotting gbp
            mode: choose overlay of activation map, backprop, both, or None:
                'activation_map': overlay activation map
                'backprop': overlay guided back propogation result
                'backprop_and_activation': overlay product of these two
                None: do not overlay anything
            show_base: if False, does not plot the image of the original sample
                [default: True]
            alpha: opacity of the activation map overlap [default: 0.5]
            cmap: matplotlib colormap for the activation map [default: 'jet']
            interpolation: the interpolation method for the activation map
                [default: bilinear] see matplotlib.pyplot.imshow()
            figsize: the figure size for the plot [default: None]
            plt_show: if True, runs plt.show() [default: True]
            save_path: path to save image to [default: None does not save file]

        Returns:
            (fig, ax) of matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        if show_base:
            if self.base_image is None:
                raise ValueError(
                    "could not plot because self.base_image is None and show_base is True"
                )
            # remove the first (batch) dimension
            # move the first dimension (Nchannels) to last dimension for imshow
            # make negative
            base_image = -self.base_image.squeeze(0).permute(1, 2, 0).detach()
            ax.imshow(base_image, alpha=1)

        # choose what to show, and validate input
        if mode == "activation_map":
            if self.activation_map is None:
                raise ValueError(
                    "could not plot because self.activation_map is None and mode is activation_map"
                )
            overlay = self.activation_map
        elif mode == "backprop":
            if self.gbp_maps is None:
                raise ValueError(
                    "could not plot because self.gbp_maps is None and mode is backprop"
                )
            if target_class not in self.gbp_maps:
                raise ValueError(
                    f"target class {target_class} not in index of self.gbp_maps"
                )
            overlay = self.gbp_maps[target_class]
        elif mode == "backprop_and_activation":
            if self.activation_map is None or self.gbp_maps is None:
                raise ValueError(
                    "could not plot because one of (self.activation_map, self.gbp_maps) is None and mode is backprop_and_activation"
                )
            if target_class not in self.gbp_maps:
                raise ValueError(
                    f"target class {target_class} not in index of self.gbp_maps"
                )
            # we combine them using the product of the two maps
            overlay = self.activation_map[..., np.newaxis] * self.gbp_maps[target_class]
        elif mode is None:
            pass
        else:
            raise ValueError(
                f"unsupported mode {mode}: choose 'activation_map', 'backprop', or 'backprop_and_activation'."
            )

        if mode is not None:
            ax.imshow(overlay, cmap=cmap, alpha=alpha, interpolation=interpolation)
            ax.set_title(f"{mode} for class {target_class}")
        else:
            ax.set_title(f"mode [None]: no overlay")
        ax.axis("off")

        if save_path is not None:
            fig.savefig(save_path)
            # plt.close(fig) #TODO do we need to close it? we are returning it

        if plt_show:
            plt.show()

        return fig, ax

    def __repr__(self):
        return f"ActivationMap()"
