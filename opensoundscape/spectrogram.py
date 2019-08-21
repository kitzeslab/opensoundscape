#!/usr/bin/env python3
""" spectrogram.py: Utilities for dealing with spectrograms
"""

from scipy import signal
import numpy as np
from opensoundscape.audio import Samples


class Spectrogram:
    """ Immutable spectrogram container
    """

    __slots__ = ("frequencies", "times", "spectrogram")

    def __init__(
        self, samples, window="hann", overlap=75, segment_length=512
    ):
        if not isinstance(samples, Samples):
            raise TypeError(f"Spectrogram expects a Samples class as input, got: '{samples.__class__}'")

        frequencies, times, spectrogram = signal.spectrogram(
            samples.samples,
            samples.sample_rate,
            window=window,
            nperseg=segment_length,
            noverlap=segment_length * overlap / 100,
            scaling="spectrum",
        )

        super(Spectrogram, self).__setattr__("frequencies", frequencies)
        super(Spectrogram, self).__setattr__("times", times)
        super(Spectrogram, self).__setattr__("spectrogram", spectrogram)

    def __setattr__(self, name, value):
        raise AttributeError("Spectrogram's cannot be modified")

    def __repr__(self):
        return f"<Spectrogram(spectrogram={self.spectrogram.shape}, frequencies={self.frequencies.shape}, times={self.times.shape})>"

    def decibel_filter(self, decibel_threshold=-100.0):
        """ Apply a decibel based filter
        """

        remove_zeros = np.copy(self.spectrogram)
        remove_zeros[remove_zeros == 0.0] = np.nan
        in_decibel = 10.0 * np.log10(remove_zeros)
        in_decibel[in_decibel <= decibel_threshold] = decibel_threshold
        return np.nan_to_num(10.0 ** (in_decibel / 10.0)), self.frequencies, self.times
