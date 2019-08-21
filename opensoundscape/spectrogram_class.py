#!/usr/bin/env python3
""" spectrogram.py: Utilities for dealing with spectrograms
"""

from scipy import signal
import numpy as np


class Spectrogram:
    def __init__(
        self, samples, sample_rate, window="hann", overlap=75, segment_length=512
    ):
        self.frequencies, self.times, self.spectrogram = signal.spectrogram(
            samples,
            sample_rate,
            window=window,
            nperseg=segment_length,
            noverlap=segment_length * overlap / 100,
            scaling="spectrum",
        )

    def decibel_filter(self, decibel_threshold=-100.0, inplace=False):
        remove_zeros = np.copy(self.spectrogram)
        remove_zeros[remove_zeros == 0.0] = np.nan
        in_decibel = 10.0 * np.log10(remove_zeros)
        in_decibel[in_decibel <= decibel_threshold] = decibel_threshold
        if inplace:
            self.spectrogram = np.nan_to_num(10.0 ** (in_decibel / 10.0))
        else:
            return (
                np.nan_to_num(10.0 ** (in_decibel / 10.0)),
                self.frequencies,
                self.times,
            )
