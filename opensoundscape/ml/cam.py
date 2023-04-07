""" Class activation maps (CAM) for OpenSoundscape models"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class CAM:
    """Object to hold and view Class Activation Maps, including guided backprop

    Stores activation maps as .activation_maps, and guided backprop as .gbp_cams

    each is a Series indexed by class

    #TODO: implement plotting multiple classes, each a different color
    basically, create greyscale images, then convert each one to a different color from color cycler
    getting transparency right might be tricky though
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
        self.base_image = base_image
        self.activation_maps = activation_maps
        self.gbp_maps = gbp_maps

    def plot(
        self,
        target_class=None,
        mode="activation",
        show_base=True,
        alpha=0.5,
        cmap="inferno",
        interpolation="bilinear",
        figsize=None,
        plt_show=True,
        save_path=None,
    ):
        """Plot the activation map, guided back propogation, or their product
        Args:
            target_class: which class's maps to visualize
                - must be in the index of self.gbp_map / self.activation_maps
                - note that the class `None` is created when classes are not specified
                during CNN.generate_cams() [default: None]
            mode: str selecting which maps to visualize, one of:
                'activation' [default]: overlay activation map
                'backprop': overlay guided back propogation result
                'backprop_and_activation': overlay product of both maps
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

        Note: if base_image does not have 3 channels, channels are averaged then copied
        across 3 RGB channels to create a greyscale image
        """
        fig, ax = plt.subplots(figsize=figsize)

        if show_base:  # plot image of sample
            # remove the first (batch) dimension
            # move the first dimension (Nchannels) to last dimension for imshow
            base_image = -self.base_image.permute(1, 2, 0).detach()
            # if not 3 channels, average over channels and copy to 3 RGB channels
            if base_image.shape[2] != 3:
                base_image = base_image.mean(2).unsqueeze(2).tile([1, 1, 3])
            ax.imshow(base_image, alpha=1)

        # choose what maps to show
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
                target_class in self.activation_maps and target_class in self.gbp_maps
            ), (
                f"passed target class {target_class}, which is"
                "not a class indexed in self.gbp_maps!"
            )
            # we combine them using the product of the two maps
            am = self.activation_maps[target_class][..., np.newaxis]  # add channel axis
            overlay = am * self.gbp_maps[target_class]
        elif mode is None:
            pass
        else:
            raise ValueError(
                f"unsupported mode {mode}: choose "
                "'activation', 'backprop', or 'backprop_and_activation'."
            )

        if mode is not None:
            ax.imshow(overlay, cmap=cmap, alpha=alpha, interpolation=interpolation)
            ax.set_title(f"{mode} for class {target_class}")
        else:
            ax.set_title(f"sample without cam")
        ax.axis("off")

        if save_path is not None:
            fig.savefig(save_path)

        if plt_show:
            plt.show()

        return fig, ax

    def __repr__(self):
        return f"CAM()"
