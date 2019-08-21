#!/usr/bin/env python3
""" spectrogram.py: Utilities for dealing with spectrograms
"""

from scipy import signal
import numpy as np


def spectrogram(samples, sample_rate, window="hann", overlap=75, segment_length=512):
    """ From samples, generate a spectrogram

    Given samples and a sample rate, generate a spectrogram

    Args:
        samples: mono samples, np.ndarray [shape=(n,)]
        sample_rate: sample_rate for samples

    """

    frequencies, times, spectrogram = signal.spectrogram(
        samples,
        sample_rate,
        window=window,
        nperseg=segment_length,
        noverlap=segment_length * overlap / 100,
        scaling="spectrum",
    )

    return spectrogram, frequencies, times


def decibel_filter(decibel_threshold=-100.0):
    """ Generate a function to apply a decibel_filter
    """

    def _decibel_filter(spectrogram, frequencies, times):
        remove_zeros = np.copy(spectrogram)
        remove_zeros[remove_zeros == 0.0] = np.nan
        in_decibel = 10.0 * np.log10(remove_zeros)
        in_decibel[in_decibel <= decibel_threshold] = decibel_threshold
        return np.nan_to_num(10.0 ** (in_decibel / 10.0)), frequencies, times

    return _decibel_filter
