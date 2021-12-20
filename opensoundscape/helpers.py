import numpy as np
import pandas as pd
import requests


def isNan(x):
    """check for nan by equating x to itself"""
    return not x == x


def sigmoid(x):
    """sigmoid function"""
    return 1 / (1 + np.exp(-x))


def bound(x, bounds):
    """ restrict x to a range of bounds = [min, max]"""
    return min(max(x, bounds[0]), bounds[1])


def overlap(r1, r2):
    """"calculate the amount of overlap between two real-numbered ranges"""
    assert r1[1] > r1[0]
    assert r2[1] > r2[0]
    lower_bound = max(r1[0], r2[0])
    upper_bound = min(r1[1], r2[1])
    return max(0, upper_bound - lower_bound)


def overlap_fraction(r1, r2):
    """"calculate the fraction of r1 (low, high range) that overlaps with r2"""
    ol = overlap(r1, r2)
    return ol / (r1[1] - r1[0])


def inrange(x, r):
    """return true if x is in range [r[0],r1] (inclusive)"""
    return x >= r[0] and x <= r[1]


def binarize(x, threshold):
    """ return a list of 0, 1 by thresholding vector x """
    if len(np.shape(x)) > 2:
        raise ValueError("shape must be 1 dimensional or 2 dimensional")

    if len(np.shape(x)) == 2:
        return [[1 if xi > threshold else 0 for xi in row] for row in x]

    return [1 if xi > threshold else 0 for xi in x]


def run_command(cmd):
    """ run a bash command with Popen, return response"""
    from subprocess import Popen, PIPE
    from shlex import split

    return Popen(split(cmd), stdout=PIPE, stderr=PIPE).communicate()


def rescale_features(X, rescaling_vector=None):
    """ rescale all features by dividing by the max value for each feature

    optionally provide the rescaling vector (1xlen(X) np.array),
    so that you can rescale a new dataset consistently with an old one

    returns rescaled feature set and rescaling vector"""
    import numpy as np

    if rescaling_vector is None:
        rescaling_vector = 1 / np.nanmax(X, 0)
    rescaledX = np.multiply(X, rescaling_vector).tolist()
    return rescaledX, rescaling_vector


def file_name(path):
    """get file name without extension from a path"""
    import os

    return os.path.splitext(os.path.basename(path))[0]


def hex_to_time(s):
    """convert a hexidecimal, Unix time string to a datetime timestamp in utc

    Example usage:
    ```
    # Get the UTC timestamp
    t = hex_to_time('5F16A04E')

    # Convert it to a desired timezone
    my_timezone = pytz.timezone("US/Mountain")
    t = t.astimezone(my_timezone)
    ```

    Args:
        s (string): hexadecimal Unix epoch time string, e.g. '5F16A04E'

    Returns:
        datetime.datetime object representing the date and time in UTC
    """
    from datetime import datetime
    import pytz

    sec = int(s, 16)
    timestamp = datetime.utcfromtimestamp(sec).replace(tzinfo=pytz.utc)
    return timestamp


def min_max_scale(array, feature_range=(0, 1)):
    """rescale vaues in an a array linearly to feature_range"""
    bottom, top = feature_range
    array_min = np.min(array)
    array_max = np.max(array)
    scale_factor = (top - bottom) / (array_max - array_min)
    return scale_factor * (array - array_min) + bottom


def linear_scale(array, in_range=(0, 1), out_range=(0, 255)):
    """ Translate from range in_range to out_range

    Inputs:
        in_range: The starting range [default: (0, 1)]
        out_range: The output range [default: (0, 255)]

    Outputs:
        new_array: A translated array
    """
    scale_factor = (out_range[1] - out_range[0]) / (in_range[1] - in_range[0])
    return scale_factor * (array - in_range[0]) + out_range[0]


def jitter(x, width, distribution="gaussian"):
    """
    Jitter (add random noise to) each value of x

    Args:
        x: scalar, array, or nd-array of numeric type
        width: multiplier for random variable (stdev for 'gaussian' or r for 'uniform')
        distribution: 'gaussian' (default) or 'uniform'
            if 'gaussian': draw jitter from gaussian with mu = 0, std = width
            if 'uniform': draw jitter from uniform on [-width, width]
    Returns:
        jittered_x: x + random jitter
    """
    if distribution == "gaussian":
        return np.array(x) + np.random.normal(0, width, size=np.shape(x))
    elif distribution == "uniform":
        return np.array(x) + np.random.uniform(-1 * width, width, size=np.shape(x))
    raise ValueError(
        f"distribution must be 'gaussian' or 'uniform'. Got {distribution}."
    )


def generate_clip_times_df(
    full_duration, clip_duration, clip_overlap=0, final_clip=None
):
    """generate start and end times for even-lengthed clips

    The behavior for incomplete final clips at the end of the full_duration
    depends on the final_clip parameter.

    This function only creates a dataframe with start and end times, it does
    not perform any actual trimming of audio or other objects.

    Args:
        full_duration: The amount of time (seconds) to split into clips
        clip_duration (float):  The duration in seconds of the clips
        clip_overlap (float):   The overlap of the clips in seconds [default: 0]
        final_clip (str):       Behavior if final_clip is less than clip_duration
            seconds long. By default, discards remaining time if less than
            clip_duration seconds long [default: None].
            Options:
                - None:         Discard the remainder (do not make a clip)
                - "extend":     Extend the final clip beyond full_duration to reach clip_duration length
                - "remainder":  Use only remainder of full_duration (final clip will be shorter than clip_duration)
                - "full":       Increase overlap with previous clip to yield a clip with clip_duration length
    Returns:
        clip_df: DataFrame with columns for 'start_time', 'end_time', and
        'clip_duration' of each clip (which may differ from `clip_duration`
        argument for final clip only)

    Note: using "remainder" or "full" with clip_overlap>0 is not recommended.
    This combination may result in several duplications of the same final clip.
    """
    if not final_clip in ["remainder", "full", "extend", None]:
        raise ValueError(
            f"final_clip must be 'remainder', 'full', 'extend',"
            f"or None. Got {final_clip}."
        )

    # Lists of start and end times for clips
    increment = clip_duration - clip_overlap
    starts = np.arange(0, full_duration, increment)
    ends = starts + clip_duration

    # Handle the final_clip
    if final_clip is None:
        # Throw away any clips with end times beyond full_duration
        keeps = ends <= full_duration
        ends = ends[keeps]
        starts = starts[keeps]
    elif final_clip == "remainder":
        # Trim clips with end times beyond full_duration to full_duration
        ends[ends > full_duration] = full_duration
    elif final_clip == "full":
        # Increase the overlap of any clips with end_time past full_duration
        # so that they end at full_duration
        # can result in duplicates of the same final_clip
        clip_idxs_to_shift = ends > full_duration
        starts[clip_idxs_to_shift] -= ends[clip_idxs_to_shift] - full_duration
        ends[clip_idxs_to_shift] = full_duration
    elif final_clip == "extend":
        # Keep the end values that extend beyond full_duration
        pass

    return pd.DataFrame({"start_time": starts, "end_time": ends})


def make_clip_df(files, clip_duration, clip_overlap=0, final_clip=None):
    """generate df of fixed-length clip times for a set of file_batch_size

    Used to prepare a dataframe for ClipLoadingSpectrogramPreprocessor

    A typical prediction workflow:
    ```
    #get list of audio files
    files = glob('./dir/*.WAV')

    #generate clip df
    clip_df = make_clip_df(files,clip_duration=5.0,clip_overlap=0)

    #create dataset
    dataset = ClipLoadingSpectrogramPreprocessor(clip_df)

    #generate predictions with a model
    model = load_model('/path/to/saved.model')
    scores, _, _ = model.predict(dataset)

    This function creates a single dataframe with audio files as
    the index and columns: 'start_time', 'end_time'. It will list
    clips of a fixed duration from the beginning to end of each audio file.

    Args:
        files: list of audio file paths
        clip_duration (float): see generate_clip_times_df
        clip_overlap (float): see generate_clip_times_df
        final_clip (str): see generate_clip_times_df
    """

    import librosa

    clip_dfs = []
    for f in files:
        t = librosa.get_duration(filename=f)
        clips = generate_clip_times_df(
            full_duration=t,
            clip_duration=clip_duration,
            clip_overlap=clip_overlap,
            final_clip=final_clip,
        )
        clips.index = [f] * len(clips)
        clips.index.name = "file"
        clip_dfs.append(clips)
    return pd.concat(clip_dfs)
