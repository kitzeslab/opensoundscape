#!/usr/bin/env python3
""" spectrogram.py: Utilities for dealing with spectrograms
"""

from scipy import signal
import numpy as np


class Spectrogram:
    """ Immutable spectrogram container
    """

    __slots__ = ("frequencies", "times", "spectrogram")

    def __init__(
        self, samples, sample_rate, window="hann", overlap=75, segment_length=512
    ):

        frequencies, times, spectrogram = signal.spectrogram(
            samples,
            sample_rate,
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
