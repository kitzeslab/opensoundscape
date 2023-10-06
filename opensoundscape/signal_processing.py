"""Signal processing tools for feature extraction and more"""
import numpy as np
import pandas as pd
import scipy
import pywt
import matplotlib.pyplot as plt
import pywt
import torch

from opensoundscape.utils import inrange


def frequency2scale(frequency, wavelet, sample_rate):
    """determine appropriate wavelet scale for desired center frequency

    Args:
        frequency: desired center frequency of wavelet in Hz (1/seconds)
        wavelet: (str) name of pywt wavelet, eg 'morl' for Morlet
        sample_rate: sample rate in Hz (1/seconds)

    Returns:
        scale: (float) scale parameter for pywt.ctw() to extract desired frequency

    Note: this function is not exactly an inverse of pywt.scale2frequency(),
    because that function returns frequency in sample-units (cycles/sample)
    rather than frequency in Hz (cycles/second). In other words,
    freuquency_hz = pywt.scale2frequency(w,scale)*sr.
    """
    # get center frequency of this wavelet
    center_freq = pywt.central_frequency(wavelet)

    # calculate scale
    return center_freq * sample_rate / frequency


def cwt_peaks(
    audio,
    center_frequency,
    wavelet="morl",
    peak_threshold=0.2,
    peak_separation=None,
    plot=False,
):
    """compute a cwt, post-process, then extract peaks

    Performs a continuous wavelet transform (cwt) on an audio signal
    at a single frequency. It then squares, smooths, and normalizes the signal.
    Finally, it detects peaks in the resulting signal and returns the times
    and magnitudes of detected peaks. It is used as a feature extractor for
    Ruffed Grouse drumming detection.

    Args:
        audio: an Audio object
        center_frequency: the target frequency to extract peaks from
        wavelet: (str) name of a pywt wavelet, eg 'morl' (see pywt docs)
        peak_threshold: minimum height of peaks
            - if None, no minimum peak height
            - see "height" argument to scipy.signal.find_peaks
        peak_separation: minimum time between detected peaks, in seconds
            - if None, no minimum distance
            - see "distance" argument to scipy.signal.find_peaks

    Returns:
        peak_times: list of times (from beginning of signal) of each peak
        peak_levels: list of magnitudes of each detected peak

    Note:
        consider downsampling audio to reduce computational cost. Audio must
        have sample rate of at least 2x target frequency.
    """

    ## create cwt feature ##

    cwt_scale = frequency2scale(center_frequency, wavelet, audio.sample_rate)
    x, _ = pywt.cwt(
        audio.samples, cwt_scale, wavelet, sampling_period=1 / audio.sample_rate
    )
    x = x[0]  # only used one frequency, so it's the first of the returned list

    ## process the cwt signal ##

    # normalize, square, hilbert envelope, normalize
    x = x / np.max(x)
    x = x**2
    x = abs(scipy.signal.hilbert(x))
    x = x / np.max(x)

    # calcualte time vector for each point in cwt signal
    t = np.linspace(0, audio.duration, len(x))

    ## find peaks in cwt signal ##

    # convert minimum time between peaks to minimum distance in points
    min_d = (
        None
        if peak_separation is None
        else np.round(peak_separation * audio.sample_rate)
    )

    # locate peaks
    peak_idx, _ = scipy.signal.find_peaks(x, height=peak_threshold, distance=min_d)
    peak_times = [t[i] for i in peak_idx]
    peak_levels = [x[i] for i in peak_idx]

    ## plotting ##

    if plot:
        # plot cwt signal and detected peaks
        plt.plot(t, x)
        plt.scatter(peak_times, peak_levels, c="red")
        plt.show()

        # plot a graph of delta-t (forward dif) vs t for all peaks
        peak_delta_ts = [
            peak_times[i] - peak_times[i - 1] for i in range(1, len(peak_times))
        ]
        plt.scatter(peak_times[1:], peak_delta_ts)
        plt.show()

    return peak_times, peak_levels


def find_accel_sequences(
    t,
    dt_range=(0.05, 0.8),
    dy_range=(-0.2, 0),
    d2y_range=(-0.05, 0.15),
    max_skip=3,
    duration_range=(1, 15),
    points_range=(5, 100),
):
    """
    detect accelerating/decelerating sequences in time series

    developed for deteting Ruffed Grouse drumming events in a series of peaks
    extracted from cwt signal

    The algorithm computes the forward difference of t, y(t). It iterates through
    the [y(t), t] points searching for sequences of points that meet a set of
    conditions. It begins with an empty candidate sequence.

    "Point-to-point criterea": Valid ranges for dt, dy, and d2y are checked for
    each subsequent point and are based on previous points in the candidate
    sequence. If they are met, the point is added to the candidate sequence.

    "Continuation criterea": Conditions for max_skip and the upper bound of dt
    are used to determine when a sequence should be terminated.
        - max_skip: max number of sequential invalid points before terminating
        - dt<=dt_range[1]: if dt is long, sequence should be broken

    "Sequence criterea": When a sequence is terminated, it is evaluated on
    conditions for duration_range and points_range. If it meets these
    conditions, it is saved as a detected sequence.
        - duration_range: length of sequence in seconds from first to last point
        - points_range: number of points included in sequence

    When a sequence is terminated, the search continues with the next point and
    an empty sequence.

    Args:
        t: (list or np.array) times of all detected peaks (seconds)
        dt_range=(0.05,0.8): valid values for t(i) - t(i-1)
        dy_range=(-0.2,0): valid values for change in y
            (grouse: difference in time between consecutive beats should decrease)
        d2y_range=(-.05,.15): limit change in dy: should not show large decrease
            (sharp curve downward on y vs t plot)
        max_skip=3: max invalid points between valid points for a sequence
            (grouse: should not have many noisy points between beats)
        duration_range=(1,15): total duration of sequence (sec)
        points_range=(9,100): total number of points in sequence

    Returns:
        sequences_t, sequences_y: lists of t and y for each detected sequence
    """
    t = np.array(t)

    # calculate y(t), the forward-difference
    y = [t[i + 1] - t[i] for i in range(len(t) - 1)]
    if len(y) < 2:  # not long enough to do anything
        return [], []

    # initialize lists to store detected sequences
    sequences_t = []
    sequences_y = []

    # initialize sequence and variables
    last_used_y_val = y[0]
    last_used_t_val = t[0]
    last_used_dy_val = None
    last_used_index = -1
    # since the first point is used for initializing y and t "last used" values,
    # we start iterating from the second point
    y = y[1:]
    t = t[1:]
    # temporary vars for building sequences
    current_sequence_y = []
    current_sequence_t = []
    building_sequence = False

    # loop through all (y, t) points
    # (note: y is one shorter than t. we won't use the last t value)
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
            # valid point. add to current sequence
            last_used_index = i
            last_used_y_val = yi
            last_used_t_val = ti
            last_used_dy_val = dy
            building_sequence = True
            current_sequence_y.append(yi)
            current_sequence_t.append(ti)
        else:
            # invalid point

            if building_sequence:
                # check: should we break the sequence or continue?
                if i - last_used_index > max_skip or not inrange(dt, dt_range):
                    # one of the two continuation criterea was broken.
                    # break sequence.

                    # check if current sequence meets sequence criterea
                    sequence_length_sec = current_sequence_t[-1] - current_sequence_t[0]
                    if inrange(len(current_sequence_y), points_range) and inrange(
                        sequence_length_sec, duration_range
                    ):
                        # this sequence meets the sequence criterea. save it.
                        sequences_y.append(current_sequence_y)
                        sequences_t.append(current_sequence_t)

                    # reset temporary sequences
                    current_sequence_y = []
                    current_sequence_t = []
                    building_sequence = False

                else:
                    # continuation criterea were not violated
                    # allow sequence to continue past this invalid point
                    pass

            else:  # we are not building a sequence, so update reference values
                last_used_y_val = yi
                last_used_t_val = ti
                last_used_dy_val = None

    # we have finished iterating through all of the points
    # finally, save current sequence if valid
    if len(current_sequence_y) > 1:
        sequence_length_sec = current_sequence_t[-1] - current_sequence_t[0]
        if inrange(len(current_sequence_y), points_range) and inrange(
            sequence_length_sec, duration_range
        ):
            # this sequence meets the sequence criterea. save it.
            sequences_y.append(current_sequence_y)
            sequences_t.append(current_sequence_t)

    return sequences_t, sequences_y


def detect_peak_sequence_cwt(
    audio,
    sample_rate=400,
    window_len=60,
    center_frequency=50,
    wavelet="morl",
    peak_threshold=0.2,
    peak_separation=15 / 400,
    dt_range=(0.05, 0.8),
    dy_range=(-0.2, 0),
    d2y_range=(-0.05, 0.15),
    max_skip=3,
    duration_range=(1, 15),
    points_range=(9, 100),
    plot=False,
):
    """Use a continuous wavelet transform to detect accellerating sequences

    This function creates a continuous wavelet transform (cwt) feature and
    searches for accelerating sequences of peaks in the feature. It was developed
    to detect Ruffed Grouse drumming events in audio signals. Default parameters
    are tuned for Ruffed Grouse drumming detection.

    Analysis is performed on analysis windows of fixed length without overlap.
    Detections from each analysis window across the audio file are aggregated.

    Args:
        audio: Audio object
        sample_rate=400: resample audio to this sample rate (Hz)
        window_len=60: length of analysis window (sec)
        center_frequency=50: target audio frequency of cwt
        wavelet='morl': (str) pywt wavelet name (see pywavelets docs)
        peak_threshold=0.2: height threhsold (0-1) for peaks in normalized signal
        peak_separation=15/400: min separation (sec) for peak finding
        dt_range=(0.05, 0.8): sequence detection point-to-point criterion 1
            - Note: the upper limit is also used as sequence termination criterion 2
        dy_range=(-0.2, 0): sequence detection point-to-point criterion 2
        d2y_range=(-0.05, 0.15): sequence detection point-to-point criterion 3
        max_skip=3: sequence termination criterion 1: max sequential invalid points
        duration_range=(1, 15): sequence criterion 1: length (sec) of sequence
        points_range=(9, 100): sequence criterion 2: num points in sequence
        plot=False: if True, plot peaks and detected sequences with pyplot

    Returns:
        dataframe summarizing detected sequences

    Note: for Ruffed Grouse drumming, which is very low pitched, audio is resampled
    to 400 Hz. This greatly increases the efficiency of the cwt, but will only
    detect frequencies up to 400/2=200Hz. Generally, choose a resample frequency
    as low as possible but >=2x the target frequency

    Note: the cwt signal is normalized on each analysis window, so changing the
    analysis window size can change the detection results.

    Note: if there is an incomplete window remaining at the end of the audio
    file, it is discarded (not analyzed).
    """

    # resample audio
    audio = audio.resample(sample_rate)

    # save detection dfs from each window in a list (aggregate later)
    dfs = []

    # analyze the audio in analysis windows of length winidow_len seconds
    for window_idx in range(int(audio.duration / window_len)):
        window_start_t = window_idx * window_len

        # trim audio to an analysis window
        audio_window = audio.trim(window_start_t, window_start_t + window_len)

        # perform continuous wavelet transform on audio and extract peaks
        peak_t, _ = cwt_peaks(
            audio_window,
            center_frequency,
            wavelet,
            peak_threshold=peak_threshold,
            peak_separation=peak_separation,
        )

        # search the set of detected peaks for accelerating sequences
        seq_t, seq_y = find_accel_sequences(
            peak_t,
            dt_range=dt_range,
            dy_range=dy_range,
            d2y_range=d2y_range,
            max_skip=max_skip,
            duration_range=duration_range,
            points_range=points_range,
        )

        if plot:
            print(f"detected peaks and sequences for window {window_idx+1}")
            y = [peak_t[i + 1] - peak_t[i] for i in range(len(peak_t) - 1)]
            plt.scatter(peak_t[:-1], y, label="all detected peaks")
            for j, (yi, ti) in enumerate(zip(seq_y, seq_t)):
                plt.scatter(ti, yi, label=f"detected sequence {j+1}")
            plt.xlabel("time (sec)")
            plt.ylabel("y")
            plt.xlim(0, window_len)
            plt.legend()
            plt.show()

        # convert seq_t to time since beginning of audio file
        seq_t = [list(np.array(s) + window_start_t) for s in seq_t]

        # save df of detected sequences in list
        dfs.append(
            pd.DataFrame(
                data={
                    "sequence_y": seq_y,
                    "sequence_t": seq_t,
                    "window_start_t": [window_start_t] * len(seq_y),
                }
            )
        )

    # create a dataframe summarizing all detections
    detection_df = pd.DataFrame(
        columns=[
            "sequence_y",
            "sequence_t",
            "window_start_t",
            "seq_len",
            "seq_start_time",
            "seq_end_time",
            "seq_midpoint_time",
        ]
    )
    if len(dfs) > 0:
        detection_df = pd.concat(dfs).reset_index(drop=True)
        detection_df["seq_len"] = [len(seq_y) for seq_y in detection_df.sequence_y]
        detection_df["seq_start_time"] = [t[0] for t in detection_df.sequence_t]
        detection_df["seq_end_time"] = [t[-1] for t in detection_df.sequence_t]
        detection_df["seq_midpoint_time"] = (
            detection_df.seq_start_time + detection_df.seq_end_time
        ) / 2
    return detection_df


def _get_ones_sequences(x):
    """Find lengths and positions of contiguous 1s in vector of 0s & 1s"""
    starts = []
    lengths = []
    seq_len = 0
    for i, xi in enumerate(x):
        assert xi in [0, 1], "values must be 0 or 1"
        if xi == 0:
            if seq_len > 0:
                lengths.append(seq_len)
            seq_len = 0
        else:
            if seq_len == 0:
                starts.append(i)
            seq_len += 1
    if seq_len > 0:
        lengths.append(seq_len)
    return starts, lengths


def thresholded_event_durations(x, threshold, normalize=False, sample_rate=1):
    """Detect positions and durations of events over threshold in 1D signal

    This function takes a 1D numeric vector and searches for segments that
    are continuously greater than a threshold value. The input signal
    can optionally be normalized, and if a sample rate is provided the
    start positions will be in the units of [sr]^-1 (ie if sr is Hz,
    start positions will be in seconds).

    Args:
        x: 1d input signal, a vector of numeric values
        threshold: minimum value of signal to be a detection
        normalize: if True, performs x=x/max(x)
        sample_rate: sample rate of input signal

    Returns:
        start_times: start time of each detected event
        durations: duration (# samples/sr) of each detected event
    """
    if normalize:
        x = x / np.max(x)
    x_01 = (np.array(x) >= threshold).astype(int)
    starts, lengths = _get_ones_sequences(x_01)

    return np.array(starts) / sample_rate, np.array(lengths) / sample_rate


def gcc(x, y, cc_filter="phat", epsilon=0.001):
    """
    Generalized cross correlation of two signals

    Computes a generalized cross correlation in frequency response.

    The generalized cross correlation algorithm described in Knapp and Carter [1].

    In the case of cc_filter='cc', gcc simplifies to cross correlation and is equivalent to
    scipy.signal.correlate and numpy.correlate.

    code adapted from github.com/axeber01/ngcc

    Args:
        x: 1d numpy array of audio samples
        y: 1d numpy array of audio samples
        cc_filter: which filter to use in the gcc.
            'phat' - Phase transform. Default.
            'roth' - Roth correlation (1971)
            'scot' - Smoothed Coherence Transform,
            'ht' - Hannan and Thomson
            'cc' - normal cross correlation with no filter
            'cc_norm' - normal cross correlation normalized by the length and amplitude of the signal
        epsilon: small value used to ensure denominator when applying a filter is non-zero.

    Returns:
        gcc: 1d numpy array of gcc values

    see also: tdoa() uses this function to estimate time delay between two signals

    [1] Knapp, C.H. and Carter, G.C (1976)
    The Generalized Correlation Method for Estimation of Time Delay. IEEE Trans. Acoust. Speech Signal Process, 24, 320-327.
    http://dx.doi.org/10.1109/TASSP.1976.1162830
    """
    # using torch speeds up our performance
    x = torch.Tensor(x)
    y = torch.Tensor(y)

    if cc_filter == "cc_norm":
        # zero - center the signals
        x = x - torch.mean(x)
        y = y - torch.mean(y)

    # pad the fft shape for "full" cross correlation of both signals
    n = x.shape[0] + y.shape[0] - 1
    if n % 2 != 0:
        n += 1
    # by choosing an optimal length rather than the shortest possible, we can
    # optimize fft speed
    n_fast = scipy.fft.next_fast_len(n, real=True)

    # Take the real Fast Fourier Transform of the signals
    X = torch.fft.rfft(x, n=n_fast)
    Y = torch.fft.rfft(y, n=n_fast)

    # multiply one by the complex conjugate of the other
    Gxy = X * torch.conj(Y)

    # Apply the filter in the fourier domain
    if cc_filter == "phat":
        phi = 1 / (torch.abs(Gxy) + epsilon)

    elif cc_filter == "roth":
        phi = 1 / (X * torch.conj(X) + epsilon)

    elif cc_filter == "scot":
        Gxx = X * torch.conj(X)
        Gyy = Y * torch.conj(Y)
        phi = 1 / (torch.sqrt(Gxx * Gyy) + epsilon)

    elif cc_filter == "ht":
        Gxx = X * torch.conj(X)
        Gyy = Y * torch.conj(Y)
        gamma_xy = Gxy / (torch.sqrt(Gxx * Gyy) + epsilon)
        coherence = torch.abs(gamma_xy) ** 2
        phi = coherence / (torch.abs(Gxy) * (1 - coherence) + epsilon)
    elif cc_filter == "cc" or cc_filter == "cc_norm":
        phi = 1.0

    else:
        raise ValueError(
            "Unsupported cc_filter. Must be one of: 'phat', 'roth', 'ht', 'scot', 'cc', 'cc_norm'"
        )
    # Inverse FFT to get the GCC
    cc = torch.fft.irfft(Gxy * phi, n_fast)

    # reorder the cross-correlation coefficients, trimming out padded regions
    # order of outputs matches torch.correlate and scipy.signal.correlate
    cc = torch.concatenate((cc[-y.shape[0] + 1 :], cc[: x.shape[0]]))

    # normalize the cross-correlation coefficients if using cc_norm
    # this normalizes both by length and amplitude
    if cc_filter == "cc_norm":
        cc = cc / (
            torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
        )  # implicitly assumes mean of x and y are 0

    return cc.numpy()


def tdoa(
    signal,
    reference_signal,
    max_delay,
    cc_filter="phat",
    sample_rate=1,
    return_max=False,
):
    """Estimate time difference of arrival between two signals

    estimates time delay by finding the maximum of the generalized cross correlation (gcc)
    of two signals. The two signals are discrete-time series with the same sample rate.

    Only the central portion of the signal, from max_delay after the start and max_delay before the end, is used for the calculation.
    All of the reference signal is used. This means that tdoa(sig, ref_sig, max_delay) will not necessarily be the same as -tdoa(ref_sig, sig, max_delay

    For example, if the signal arrives 2.5 seconds _after_ the reference signal, returns 2.5;
    if it arrives 0.5 seconds _before_ the reference signal, returns -0.5.

    Args:
        signal: np.array or list object containing the signal of interest
        reference_signal: np.array or list containing the reference signal. Both audio recordings must be time-synchronized.
        max_delay: maximum possible tdoa (seconds) between the two signals. Cannot be longer than 1/2 the duration of the signal.
        The tdoa returned will be between -max_delay and +max_delay.
        cc_filter: see gcc()
        sample_rate: sample rate (Hz) of signals; both signals must have same sample rate
        return_max: if True, returns the maximum value of the generalized cross correlation

            For example, if max_delay=0.5, the tdoa returned will be the delay between -0.5 and +0.5 seconds, that maximizes the cross-correlation.
            This is useful if you know the maximum possible delay between the two signals, and want to ignore any tdoas outside of that range.
            e.g. if receivers are 100m apart, and the speed of sound is 340m/s, then the maximum possible delay is 0.294 seconds.
    Returns:
        estimated delay from reference signal to signal, in seconds
        (note that default samping rate is 1.0 samples/second)

        if return_max is True, returns a second value, the maximum value of the
        result of generalized cross correlation

    See also: gcc() if you want the raw output of generalized cross correlation
    """
    # check that the two signals are the same length

    if (
        len(reference_signal) - len(signal) > 1
    ):  # allow 1 sample difference for rounding errors
        raise ValueError("reference_signal and signal must be the same length.")

    # check that the maximum delay is not longer than 1/2 the audio signal
    audio_len = len(signal) / sample_rate
    if max_delay >= audio_len / 2:
        raise ValueError(
            f"max_delay cannot be longer than 1/2 the signal. max_delay is {max_delay} seconds, signal is {audio_len} seconds long."
        )
    # trim the primary signal to just the central part
    start = int(max_delay * sample_rate)
    end = int(len(reference_signal) - max_delay * sample_rate)
    signal = signal[start:end]

    # compute the generalized cross correlation between the signals
    cc = gcc(signal, reference_signal, cc_filter=cc_filter)

    # filter to only the 'valid' part of the cross correlation, where the signals overlap fully
    cc = cc[len(signal) - 1 : -len(signal) + 1]

    # find max cc value
    # if max is at 0 it indicates the signal arrives max_delay earlier than in the reference signal
    lag = np.argmax(cc)

    # convert lag to time delay
    tdoa = (lag / sample_rate) - max_delay
    max_cc = np.max(cc)

    if return_max:
        return tdoa, np.max(cc)

    else:
        return tdoa
