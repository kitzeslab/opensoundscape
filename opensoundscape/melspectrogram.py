#!/usr/bin/env python3
""" melspectrogram.py: Utilities for dealing with mel spectrograms

WARNING: This module has not been thoroughly tested for compatibility with
modules and tools in OpenSoundscape.
"""

from opensoundscape.audio import Audio
from librosa.feature import melspectrogram
from librosa import pcen
from PIL import Image
from opensoundscape.helpers import linear_scale


class MelSpectrogram:
    """ Immutable spectrogram container

    WARNING: This class has not been thoroughly tested for compatibility with
    modules and tools in OpenSoundscape.
    """

    __slots__ = ("S", "sample_rate", "hop_length", "fmin", "fmax")

    def __init__(self, S, sample_rate, hop_length, fmin, fmax):
        super(MelSpectrogram, self).__setattr__("S", S)
        super(MelSpectrogram, self).__setattr__("sample_rate", sample_rate)
        super(MelSpectrogram, self).__setattr__("hop_length", hop_length)
        super(MelSpectrogram, self).__setattr__("fmin", fmin)
        super(MelSpectrogram, self).__setattr__("fmax", fmax)

    @classmethod
    def from_audio(
        cls,
        audio,
        n_fft=1024,
        n_mels=128,
        window="flattop",
        win_length=256,
        hop_length=32,
        htk=True,
        fmin=None,
        fmax=None,
    ):
        """ Create a MelSpectrogram object from an Audio object

        The kwargs are cherry-picked from:

        - https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html#librosa.feature.melspectrogram
        - https://librosa.org/doc/latest/generated/librosa.filters.mel.html?librosa.filters.mel

        Args:
            n_fft: Length of the FFT window [default: 1024]
            n_mels: Number of mel bands to generate [default: 128]
            window: The windowing function to use [default: "flattop"]
            win_length: Each frame of audio is windowed by `window`. The window
                will be of length `win_length` and then padded with zeros to match
                `n_fft` [default: 256]
            hop_length: Number of samples between successive frames [default: 32]
            htk: use HTK formula instead of Slaney [default: True]
            fmin: lowest frequency (in Hz) [default: None]
            fmax: highest frequency (in Hz). If None, use `fmax = sr / 2.0` [default: None]

        Returns:
            opensoundscape.melspectrogram.MelSpectrogram object
        """

        if not isinstance(audio, Audio):
            raise TypeError("Class method expects Audio class as input")

        process_fmin = fmin if fmin else 0
        process_fmax = fmax if fmax else audio.sample_rate / 2

        S = melspectrogram(
            y=audio.samples,
            sr=audio.sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            n_mels=n_mels,
            htk=htk,
            fmin=process_fmin,
            fmax=process_fmax,
        )

        # Make spectrogram "right-side up"
        S = S[::-1]

        return cls(S, audio.sample_rate, hop_length, process_fmin, process_fmax)

    def to_pcen(self, gain=0.8, bias=10.0, power=0.25, time_constant=0.06):
        """ Create PCEN from MelSpectrogram

        Argument descriptions come from https://librosa.org/doc/latest/generated/librosa.pcen.html?highlight=pcen#librosa-pcen

        Args:
            gain: The gain factor. Typical values should be slightly less than 1 [default: 0.8]
            bias: The bias point of the nonlinear compression [default: 10.0]
            power: The compression exponent. Typical values should be between 0
                and 0.5. Smaller values of power result in stronger compression. At
                the limit power=0, polynomial compression becomes logarithmic
                [default: 0.25]
            time_constant: The time constant for IIR filtering, measured in seconds [default: 0.06]

        Returns:
            The per-channel energy normalized version of MelSpectrogram.S
        """

        return MelSpectrogram(
            pcen(
                self.S,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                gain=gain,
                bias=bias,
                power=power,
                time_constant=time_constant,
            ),
            self.sample_rate,
            self.hop_length,
            self.fmin,
            self.fmax,
        )

    def to_image(self, shape=None, mode="RGB", s_range=(0, 20)):
        """ Generate PIL Image from MelSpectrogram

        Given a range of values for S (e.g. default is minimum 0, maximum 20)
        generate a PIL image in 3-channel (RGB) or single channel (L) mode.
        A user can optionally resize the image.

        Args:
            shape: Resize to shape (h, w) [default: None]
            mode: Mode to write out "RGB" or "L" [default: "RGB"]
            s_range: The input range of S [default: (0, 20)]

        Returns:
            PIL.Image
        """

        arr = linear_scale(
            self.S, in_range=(s_range[0], s_range[1]), out_range=(255, 0)
        )
        img = Image.fromarray(arr[::-1, :])
        img = img.convert(mode)
        if shape is not None:
            img = img.resize(shape)
        return img
