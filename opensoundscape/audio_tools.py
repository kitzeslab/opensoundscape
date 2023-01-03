"""set of tools that filter or modify audio files or sample arrays (not Audio objects)
"""
from librosa import load
import numpy as np
from scipy.signal import butter, sosfiltfilt


def butter_bandpass(low_f, high_f, sample_rate, order=9):
    """generate coefficients for bandpass_filter()

    Args:
        low_f: low frequency of butterworth bandpass filter
        high_f: high frequency of butterworth bandpass filter
        sample_rate: audio sample rate
        order=9: order of butterworth filter

    Returns:
        set of coefficients used in sosfiltfilt()
    """
    nyq = 0.5 * sample_rate
    low = low_f / nyq
    high = high_f / nyq
    sos = butter(order, [low, high], analog=False, btype="band", output="sos")
    return sos


def bandpass_filter(signal, low_f, high_f, sample_rate, order=9):
    """perform a butterworth bandpass filter on a discrete time signal
    using scipy.signal's butter and solfiltfilt (phase-preserving version of sosfilt)

    Args:
        signal: discrete time signal (audio samples, list of float)
        low_f: -3db point (?) for highpass filter (Hz)
        high_f: -3db point (?) for highpass filter (Hz)
        sample_rate: samples per second (Hz)
        order=9: higher values -> steeper dropoff

    Returns:
        filtered time signal
    """
    sos = butter_bandpass(low_f, high_f, sample_rate, order=order)
    return sosfiltfilt(sos, signal)


def clipping_detector(samples, threshold=0.6):
    """count the number of samples above a threshold value

    Args:
        samples: a time series of float values
        threshold=0.6: minimum value of sample to count as clipping

    Returns:
        number of samples exceeding threshold
    """
    return len(list(filter(lambda x: x > threshold, samples)))


# helper for Tessa's silence detector, used for filtering xeno-canto
# and copied from crc: /ihome/sam/bmooreii/projects/opensoundscape/xeno-canto
# move to Audio module?
def window_energy(samples, window_len_samples=256, overlap_len_samples=128):
    """
    Calculate audio energy with a sliding window

    Calculate the energy in an array of audio samples

    Args:
        samples (np.ndarray): array of audio
            samples loaded using librosa.load
        window_len_samples: samples per window
        overlap_len_samples: number of samples shared between consecutive windows

    Returns:
        list of energy level (float) for each window
    """

    def _energy(samples):
        return np.sum(samples**2) / len(samples)

    windowed = []
    skip = window_len_samples - overlap_len_samples
    for start in range(0, len(samples), skip):
        energy = _energy(samples[start : start + window_len_samples])
        windowed.append(energy)

    return windowed


# Based on Tessa's detect_silence(). Flipped outputs so that 0 is silent 1 is non-silent
def silence_filter(
    filename,
    smoothing_factor=10,
    window_len_samples=256,
    overlap_len_samples=128,
    threshold=None,
):
    """
    Identify whether a file is silent (0) or not (1)

    Load samples from an mp3 file and identify
    whether or not it is likely to be silent.
    Silence is determined by finding the energy
    in windowed regions of these samples, and
    normalizing the detected energy by the average
    energy level in the recording.

    If any windowed region has energy above the
    threshold, returns a 0; else returns 1.

    Args:
        filename (str): file to inspect
        smoothing_factor (int): modifier
            to window_len_samples
        window_len_samples: number of samples per window segment
        overlap_len_samples: number of samples to overlap
            each window segment
        threshold: threshold value (experimentally
            determined)

    Returns:
        0 if file contains no significant energy over bakcground
        1 if file contains significant energy over bakcground
    If threshold is None: returns net_energy over background noise
    """
    try:
        samples, _ = load(filename, sr=None)
    #     except NoBackendError:
    #         return -1.0
    except RuntimeError:
        return -2.0
    except ZeroDivisionError:
        return -3.0
    except:
        return -4.0

    energy = window_energy(
        samples, window_len_samples * smoothing_factor, overlap_len_samples
    )
    norm_factor = np.mean(energy)
    net_energy = (energy - norm_factor) * 100

    # the default of "None" for threshold will return the max value of ys
    if threshold is None:
        return np.max(net_energy)
    # if we pass a threshold (eg .05), we will return 0 or 1
    else:
        return int(np.max(net_energy) > threshold)
