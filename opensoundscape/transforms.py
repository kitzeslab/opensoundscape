#!/usr/bin/env python3
""" transforms.py: Transforms for spectrograms and images
"""

import numpy as np
from PIL import Image
import random


def time_split(img, seed=None):
    """ Given a PIL.Image, rotate it

    Choose a random new starting point and append the first section to the end.
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
