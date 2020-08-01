import numpy as np


def isNan(x):
    """check for nan by equating x to itself"""
    return not x == x


def sigmoid(x):
    """sigmoid function"""
    return 1 / (1 + np.exp(-x))


def bound(x, bounds):
    """ restrict x to a range of bounds = [min, max]"""
    return min(max(x, bounds[0]), bounds[1])


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
    """convert a hexidecimal, Unix time string to a datetime timestamp"""
    from datetime import datetime

    sec = int(s, 16)
    timestamp = datetime.utcfromtimestamp(sec)
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
