"""Transforms and augmentations for PIL.Images"""
from PIL import Image
import numpy as np
import random


def time_split(img, seed=None):
    """Given a PIL.Image, split into left/right parts and swap

    Randomly chooses the slicing location
    For example, if `h` chosen

    abcdefghijklmnop
           ^
    hijklmnop + abcdefg

    Args:
        img: A PIL.Image

    Returns:
        A PIL.Image
    """

    if not isinstance(img, Image.Image):
        raise TypeError("Expects PIL.Image as input")

    if seed:
        random.seed(seed)

    width, _ = img.size
    idx = random.randint(0, width)
    arr = np.array(img)
    rotated = np.hstack([arr[:, idx:, :], arr[:, 0:idx, :]])
    return Image.fromarray(rotated)
