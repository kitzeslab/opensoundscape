"""preprocess.py: utilities for augmentation and preprocessing pipelines"""

import numpy as np
from PIL import Image
import random

from opensoundscape.audio import Audio

### Audio transforms ###


def random_audio_trim(audio, duration, extend_short_clips=False):
    """randomly select a subsegment of Audio of fixed length

    randomly chooses a time segment of the entire Audio object to cut out,
    from the set of all possible start times that allow a complete extraction

    Args:
        Audio: input Audio object
        length: duration in seconds of the trimmed Audio output

    Returns:
        Audio object trimmed from original
    """
    input_duration = len(audio.samples) / audio.sample_rate
    if duration > input_duration:
        if not extend_short_clips:
            raise ValueError(
                f"the length of the original file ({input_duration} sec) was less than the length to extract ({duration} sec). To extend short clips, use extend_short_clips=True"
            )
        else:
            return audio.extend(duration)
    extra_time = input_duration - duration
    start_time = np.random.uniform() * extra_time
    return audio.trim(start_time, start_time + duration)


### PIL.Image transforms ###


def time_split(img, seed=None):
    """Given a PIL.Image, rotate it

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
