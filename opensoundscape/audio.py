#!/usr/bin/env python3
""" audio.py: Utilities for dealing with audio files
"""

import pathlib
import io
import librosa
import soundfile
import numpy as np
from warnings import filterwarnings


class OpsoLoadAudioInputError(Exception):
    """ Custom exception indicating we can't load input
    """

    pass


class OpsoLoadAudioInputTooLong(Exception):
    """ Custom exception indicating length of audio is too long
    """

    pass


class Audio:
    """ Container for audio samples
    """

    __slots__ = ("samples", "sample_rate")

    def __init__(self, samples, sample_rate):

        # Do not move these lines; it will break Pytorch training
        self.samples = samples
        self.sample_rate = sample_rate

        samples_error = None
        if not isinstance(self.samples, np.ndarray):
            samples_error = (
                "Initializing an Audio object requires samples to be a numpy array"
            )

        try:
            self.sample_rate = int(self.sample_rate)
        except ValueError:
            sample_rate_error = f"Initializing an Audio object requires an integer sample_rate, got `{sample_rate}`"
            if samples_error:
                raise ValueError(
                    f"Audio initialization failed with:\n{samples_error}\n{sample_rate_error}"
                )
            raise ValueError(f"Audio initialization failed with:\n{sample_rate_error}")

        if samples_error:
            raise ValueError(f"Audio initialization failed with:\n{samples_error}")

    @classmethod
    def from_file(
        cls, path, sample_rate=None, max_duration=None, resample_type="kaiser_fast"
    ):
        """ Load audio from files

        Deal with the various possible input types to load an audio
        file and generate a spectrogram

        Args:
            path (str, Path): path to an audio file
            sample_rate (int, None): resample audio with value and resample_type,
                if None use source sample_rate (default: None)
            resample_type: method used to resample_type (default: kaiser_fast)
            max_duration: the maximum length of an input file,
                None is no maximum (default: None)

        Returns:
            Audio: attributes samples and sample_rate
        """

        if max_duration:
            if librosa.get_duration(filename=path) > max_duration:
                raise OpsoLoadAudioInputTooLong()

        filterwarnings("ignore")
        samples, sr = librosa.load(
            path, sr=sample_rate, res_type=resample_type, mono=True
        )

        return cls(samples=samples, sample_rate=sr)

    @classmethod
    def from_bytesio(cls, bytesio, sample_rate=None, resample_type="kaiser_fast"):
        """...
        """
        samples, sr = soundfile.read(bytesio)
        if sample_rate:
            samples = librosa.resample(samples, sr, sample_rate, res_type=resample_type)
            sr = sample_rate

        return cls(samples, sr)

    def __repr__(self):
        return f"<Audio(samples={self.samples.shape}, sample_rate={self.sample_rate})>"

    def trim(self, start_time, end_time):
        """ trim Audio object in time

        Args:
            start_time: time in seconds for start of extracted clip
            end_time: time in seconds for end of extracted clip
        Returns:
            a new Audio object containing samples from start_time to end_time
        """
        start_sample = self.time_to_sample(start_time)
        end_sample = self.time_to_sample(end_time)
        samples_trimmed = self.samples[start_sample:end_sample]
        return Audio(samples_trimmed, self.sample_rate)

    def time_to_sample(self, time):
        """ Given a time, convert it to the corresponding sample

        Args:
            time: The time to multiply with the sample_rate
        Returns:
            sample: The rounded sample
        """
        return round(time * self.sample_rate)

    def bandpass(self, low_f, high_f, order=9):
        """ bandpass audio signal frequencies

        uses a phase-preserving algorithm (scipy.signal's butter and solfiltfilt)

        Args:
            low_f: low frequency cutoff (-3 dB)  in Hz of bandpass filter
            high_f: high frequency cutoff (-3 dB)  in Hz of bandpass filter
            order: butterworth filter order (integer) ~= steepness of cutoff

        """
        from opensoundscape.audio_tools import bandpass_filter

        if low_f <= 0:
            raise ValueError("low_f must be greater than zero")

        if high_f >= self.sample_rate / 2:
            raise ValueError("high_f must be less than sample_rate/2")

        filtered_samples = bandpass_filter(
            self.samples, low_f, high_f, self.sample_rate, order=9
        )
        return Audio(filtered_samples, self.sample_rate)

    # can act on an audio file and be moved into Audio class
    def spectrum(self):
        """create frequency spectrum from an Audio object using fft

        Args:
            self

        Returns:
            fft, frequencies
        """
        from scipy.fftpack import fft
        from scipy.fft import fftfreq

        # Compute the fft (fast fourier transform) of the selected clip
        N = len(self.samples)
        T = 1 / self.sample_rate
        fft = fft(self.samples)
        freq = fftfreq(N, d=T)  # the frequencies corresponding to fft bins

        # remove negative frequencies and scale magnitude by 2.0/N:
        fft = 2.0 / N * fft[0 : int(N / 2)]
        frequencies = freq[0 : int(N / 2)]
        fft = np.abs(fft)

        return fft, frequencies

    def save(self, path):
        """save Audio to file

        Args:
            path: destination for output
        """
        from soundfile import write

        write(path, self.samples, self.sample_rate)

    def duration(self):
        """ Return duration of Audio

        Output:
            duration (float): The duration of the Audio
        """

        return len(self.samples) / self.sample_rate
