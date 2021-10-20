"""Tools for extracting features from audio signals"""
import numpy as np
import pandas as pd
from scipy import signal
import pywt
import matplotlib.pyplot as plt


def frequency2scale(frequency, wavelet, sr):
    """determine appropriate wavelet scale for desired center frequency

    Args:
        frequency: desired center frequency of wavelet in Hz (1/seconds)
        wavelet: (str) name of pywt wavelet, eg 'morl' for Morlet
        sr: sample rate in Hz (1/seconds)

    Returns:
        scale: (float) scale parameter for pywt.ctw() to extract desired frequency

    Note: this function is not exactly an inverse of pywt.scale2frequency(), because that
    function returns frequency in sample-units (cycles/sample) than frequency in Hz (cycles/second)
    In other words, freuquency_hz = pywt.scale2frequency(w,scale)*sr.
    """
    from pywt import central_frequency

    # get center frequency of this wavelet
    cf = central_frequency(wavelet)

    # calculate scale
    return cf * sr / frequency


def cwt_peaks(
    audio,
    center_frequency,
    wavelet="morl",
    peak_threshold=0.2,
    peak_separation=None,
    plot=False,
):
    """compute a cwt, post-process, then extract peaks

    This function performs a continuous wavelet transform on an audio signal
    at a single frequency. It then squares, max_skips and normalizes the signal.
    Finally, it detects peaks in the resulting signal and returns the times
    and magnitudes of detected peaks. It is used as a feature extractor for
    Ruffed Grouse detection.

    Args:
        audio: an Audio object
        center_frequency: the target frequency to extract peaks from
        wavelet: (str) name of a pywt wavelet, eg 'morl' (see pywt docs)
        peak_threshold: "height" argument to scipy.signal.find_peaks
        peak_distance: minimum time between detected peaks, in seconds
            - if None, no minimum distance
            - see "distance" argument to scipy.signal.find_peaks

    Returns:
        peak_times: list of times (from beginning of signal) of each peak
        peak_levels: list of magnitudes of each detected peak

    Note:
        consider downsampling audio to reduce computational cost. Audio must
        have sample rate of at least 2x target frequency.
    """

    # create cwt feature
    cwt_scale = frequency2scale(center_frequency, wavelet, audio.sample_rate)
    x, _ = pywt.cwt(
        audio.samples, cwt_scale, wavelet, sampling_period=1 / audio.sample_rate
    )
    x = x[0]  # we only used one frequency, so it's the first of the returned list

    # process the cwt signal:
    # normalize, square, hilbert envelope, normalize
    x = x / np.max(x)
    x = x ** 2
    x = abs(signal.hilbert(x))
    x = x / np.max(x)

    # calcualte time vector for each point in cwt signal
    t = np.linspace(0, audio.duration(), len(x))

    # convert minimum time between peaks to minimum distance in points
    min_d = np.round(peak_separation * audio.sample_rate)
    print(min_d)

    # find peaks in cwt signal
    peak_idx, _ = signal.find_peaks(x, height=peak_threshold, distance=min_d)
    peak_times = [t[i] for i in peak_idx]
    peak_levels = [x[i] for i in peak_idx]

    if plot:
        plt.plot(t, x)
        plt.scatter(peak_times, peak_levels, c="red")
        plt.show()

        peak_delta_ts = [
            peak_times[i] - peak_times[i - 1] for i in range(1, len(peak_times))
        ]
        plt.scatter(peak_times[1:], peak_delta_ts)
        plt.show()

    return peak_times, peak_levels


def inrange(x, r):
    return x >= r[0] and x <= r[1]


def find_accel_sequences(
    t,
    dt_range=[0.05, 0.8],
    dy_range=[-0.2, 0],
    d2y_range=[-0.05, 0.15],
    max_skip=3,
    duration_range=[1, 15],
    points_range=[5, 100],
):
    """
    detect accelerating/decelerating sequences in time series

    developed for deteting Ruffed Grouse drumming events in a series of peaks
    extracted from cwt signal

    The algorithm computes the forward difference of t, y(t). It iterates through
    the y(t), t points searching for sequences of points that meet a set of
    conditions. It begins with an empty candidate sequence.

    Conditions for dt, dy, and d2y are applied to each subsequent point and are
    based on previous points in the candidate sequence. If
    they are met, the point is added to the candidate sequence.

    Conditions for max_skip and the upper bound of dt are used to determine
    when a sequence should be terminated.

    When a sequence is terminated, it is evaluated on conditions for duration_range and
    points_range. If it meets these conditions, it is saved as a detected sequence.

    The search continues with the next point and an empty sequence.

    Args:
        t: times of all detected peaks (seconds)
        dt_range = [0.05,0.8]: valid values for t(i) - t(i-1)
        dy_range=[-0.2,0]: valid values for change in y
            (grouse: difference in time between consecutive beats should decrease)
        d2y_range = [-.05,.15]: limit change in dy: should not show large decrease
            (sharp curve downward on y vs t plot)
        max_skip=3: max invalid points between valid points for a sequence
            (grouse: should not have many noisy points between beats)
        duration_range=[1,15]: total duration of sequence
        points_range=[9,100]: total num points in sequence

    Returns:
        sequences_t, sequences_y: lists of t and y for each detected sequence
    """
    # calculate y(t), the forward-difference
    y = [t[i + 1] - t[i] for i in range(len(t) - 1)]
    if len(y) < 2:  # not long enough to do anything
        return [], []

    # initialize lists to store detected sequences
    sequences_t = []
    sequences_y = []

    # start from second point
    last_used_y_val = y[0]
    last_used_t_val = t[0]
    last_used_dy_val = None
    last_used_index = -1
    y = y[1:]
    t = t[1:]

    # temporary vars for building sequences
    current_sequence_y = []
    current_sequence_t = []
    building_sequence = False

    # loop through points (note that y is one shorter than t)
    for i, yi in enumerate(y):
        ti = t[i]
        # calculate dt, the time since last point in candidate sequence
        dt = ti - last_used_t_val
        # calculate dy, the difference in y compared to previous point in candidate sequence
        dy = yi - last_used_y_val
        # calculate d2y, the second backwards difference of y in the candidate sequence
        d2y = None if last_used_dy_val is None else dy - last_used_dy_val

        # Check if the point is valid based on 3 point-by-point criterea
        if (
            inrange(dt, dt_range)
            and inrange(dy, dy_range)
            and (inrange(d2y, d2y_range) if d2y is not None else True)
        ):
            # valid point. add to candidate sequence
            last_used_index = i
            last_used_y_val = yi
            last_used_t_val = ti
            last_used_dy_val = dy
            building_sequence = True
            current_sequence_y.append(yi)
            current_sequence_t.append(ti)
        else:  # invalid point
            # check: should we break the sequence or continue?
            if building_sequence:
                if i - last_used_index > max_skip or not inrange(
                    dt, dt_range
                ):  # break sequence

                    # check if current sequence meets sequence criterea
                    sequence_length_sec = current_sequence_t[-1] - current_sequence_t[0]
                    if inrange(len(current_sequence_y), points_range) and inrange(
                        sequence_length_sec, duration_range
                    ):
                        # this sequence meets the criterea. save it.
                        sequences_y.append(current_sequence_y)
                        sequences_t.append(current_sequence_t)

                    # reset temporary sequences
                    current_sequence_y = []
                    current_sequence_t = []
                    building_sequence = False

                else:
                    # allow sequence to continue past this noisy point
                    pass

            else:  # we are not building a sequence, so update reference x and t values
                last_used_y_val = yi
                last_used_t_val = ti
                last_used_dy_val = None

    # finally, save current sequence if valid
    if len(current_sequence_y) > 1:
        sequence_length_sec = current_sequence_t[-1] - current_sequence_t[0]
        if inrange(len(current_sequence_y), points_range) and inrange(
            sequence_length_sec, duration_range
        ):
            # this sequence meets the criterea. save it.
            sequences.append(current_sequence_y)
            sequence_times.append(current_sequence_t)

    return sequences_t, sequences_y
