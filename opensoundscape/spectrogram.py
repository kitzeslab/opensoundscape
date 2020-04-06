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

    def __init__(self, spectrogram, frequencies, times):
        if not isinstance(spectrogram, np.ndarray):
            raise TypeError(
                f"Spectrogram.spectrogram should be a np.ndarray [shape=(n, m)]. Got {spectrogram.__class__}"
            )
        if not isinstance(frequencies, np.ndarray):
            raise TypeError(
                f"Spectrogram.frequencies should be an np.ndarray [shape=(n,)]. Got {frequencies.__class__}"
            )
        if not isinstance(times, np.ndarray):
            raise TypeError(
                f"Spectrogram.times should be an np.ndarray [shape=(m,)]. Got {times.__class__}"
            )

        if spectrogram.ndim != 2:
            raise TypeError(
                f"spectrogram should be a np.ndarray [shape=(n, m)]. Got {spectrogram.shape}"
            )
        if frequencies.ndim != 1:
            raise TypeError(
                f"frequencies should be an np.ndarray [shape=(n,)]. Got {frequencies.shape}"
            )
        if times.ndim != 1:
            raise TypeError(
                f"times should be an np.ndarray [shape=(m,)]. Got {times.shape}"
            )

        if spectrogram.shape != (frequencies.shape[0], times.shape[0]):
            raise TypeError(
                f"Dimension mismatch, spectrogram.shape: {spectrogram.shape}, frequencies.shape: {frequencies.shape}, times.shape: {times.shape}"
            )

        super(Spectrogram, self).__setattr__("frequencies", frequencies)
        super(Spectrogram, self).__setattr__("times", times)
        super(Spectrogram, self).__setattr__("spectrogram", spectrogram)

    @classmethod
    def from_audio(cls, audio, window="hann", overlap=75, segment_length=512):
        if not isinstance(audio, Samples):
            raise TypeError("Class method expects Samples class as input")

        frequencies, times, spectrogram = signal.spectrogram(
            audio.samples,
            audio.sample_rate,
            window=window,
            nperseg=segment_length,
            noverlap=segment_length * overlap / 100,
            scaling="spectrum",
        )

        return cls(spectrogram, frequencies, times)

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
        return Spectrogram(
            np.nan_to_num(10.0 ** (in_decibel / 10.0)), self.frequencies, self.times
        )

    def min_max_scale(self, feature_range=(0, 1)):
        """ Apply a min-max filter
        """

        if len(feature_range) != 2:
            raise AttributeError(
                "Error: `feature_range` doesn't look like a 2-element tuple?"
            )
        if feature_range[1] < feature_range[0]:
            raise AttributeError("Error: `feature_range` isn't increasing?")

        spect_min = self.spectrogram.min()
        spect_max = self.spectrogram.min()
        scale_factor = (feature_range[1] - feature_range[0]) / (spect_max - spect_min)
        return Spectrogram(
            scale_factor * (self.spectrogram - spect_min) + feature_range[0],
            self.frequencies,
            self.times,
        )

    def audacity_gain_range(self, spec_gain=20, spec_range=80):
        """ Apply gain and range similar to Audacity
        """

        _spec = self.spectrogram
        _spec[_spec > -spec_gain] = -spec_gain
        _spec[_spec < -(spec_gain + spec_range)] = -(spec_gain + spec_range)
        return Spectrogram(_spec, self.frequencies, self.times)
