#!/usr/bin/env python3
""" audio.py: Utilities for dealing with audio files
"""

from pathlib import Path
import io
import librosa
import soundfile
import numpy as np
import pandas as pd
import warnings
from math import ceil


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

    Initializing an `Audio` object directly requires the specification of the
    sample rate. Use `Audio.from_file` or `Audio.from_bytesio` with
    `sample_rate=None` to use a native sampling rate.

    Arguments:
        samples (np.array):     The audio samples
        sample_rate (integer):  The sampling rate for the audio samples
        resample_type (str):    The resampling method to use [default: "kaiser_fast"]
        max_duration (None or integer): The maximum duration allowed for the audio file [default: None]

    Returns:
        An initialized `Audio` object
    """

    __slots__ = ("samples", "sample_rate", "resample_type", "max_duration")

    def __init__(
        self, samples, sample_rate, resample_type="kaiser_fast", max_duration=None
    ):
        # Do not move these lines; it will break Pytorch training
        self.samples = samples
        self.sample_rate = sample_rate
        self.resample_type = resample_type
        self.max_duration = max_duration

        samples_error = None
        if not isinstance(self.samples, np.ndarray):
            samples_error = (
                "Initializing an Audio object requires samples to be a numpy array"
            )

        try:
            self.sample_rate = int(self.sample_rate)
        except ValueError:
            sample_rate_error = (
                "Initializing an Audio object requires the audio samples' sampling rate"
            )
            if samples_error:
                raise ValueError(
                    f"Audio initialization failed with:\n{samples_error}\n{sample_rate_error}"
                )
            raise ValueError(f"Audio initialization failed with:\n{sample_rate_error}")

        if samples_error:
            raise ValueError(f"Audio initialization failed with:\n{samples_error}")

    @classmethod
    def from_file(
        cls, path, sample_rate=None, resample_type="kaiser_fast", max_duration=None
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

        warnings.filterwarnings("ignore")
        samples, sr = librosa.load(
            path, sr=sample_rate, res_type=resample_type, mono=True
        )
        warnings.resetwarnings()

        return cls(samples, sr, resample_type=resample_type, max_duration=max_duration)

    @classmethod
    def from_bytesio(
        cls, bytesio, sample_rate=None, max_duration=None, resample_type="kaiser_fast"
    ):
        """ Read from bytesio object

        Read an Audio object from a BytesIO object. This is primarily used for
        passing Audio over HTTP.

        TODO:
            Describe how to initialize an Audio file as a BytesIO object

        Arguments:
            bytesio: Contents of WAV file as BytesIO
            sample_rate: The final sampling rate of Audio object [default: None]
            max_duration: The maximum duration of the audio file [default: None]
            resample_type: The librosa method to do resampling [default: "kaiser_fast"]

        Returns:
            An initialized Audio object
        """
        samples, sr = soundfile.read(bytesio)
        if sample_rate:
            samples = librosa.resample(samples, sr, sample_rate, res_type=resample_type)
            sr = sample_rate

        return cls(samples, sr, resample_type=resample_type, max_duration=max_duration)

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
        return Audio(
            samples_trimmed,
            self.sample_rate,
            resample_type=self.resample_type,
            max_duration=self.max_duration,
        )

    def extend(self, length):
        """ Extend audio file by looping it

        Args:
            length: the final length in seconds of the extended file
        Returns:
            a new Audio object of the desired length
        """

        total_samples_needed = round(length * self.sample_rate)
        samples_extended = np.resize(self.samples, total_samples_needed)
        return Audio(
            samples_extended,
            self.sample_rate,
            resample_type=self.resample_type,
            max_duration=self.max_duration,
        )

    def time_to_sample(self, time):
        """ Given a time, convert it to the corresponding sample

        Args:
            time: The time to multiply with the sample_rate
        Returns:
            sample: The rounded sample
        """
        return round(time * self.sample_rate)

    def bandpass(self, low_f, high_f, order):
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
            self.samples, low_f, high_f, self.sample_rate, order=order
        )
        return Audio(
            filtered_samples,
            self.sample_rate,
            resample_type=self.resample_type,
            max_duration=self.max_duration,
        )

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

    def split(self, clip_duration=5, clip_overlap=1, final_clip=None):
        """ Split Audio into clips

        The Audio object is split into clips of a specified duration and overlap

        Arguments:
            clip_duration:  The duration in seconds of the clips
            clip_overlap:   The overlap of the clips in seconds
            final_clip:     Possible options (any other input will ignore the final clip entirely),
                                - "remainder":          Include the remainder of the Audio
                                                            (clip will not have clip_duration length)
                                - "full":               Increase the overlap to yield a clip with clip_duration
                                - "extend":             Similar to remainder but extend the clip to clip_duration
        Results:
            A list of dictionaries with keys: ["audio", "begin_time", "end_time"]
        """

        duration = self.duration()
        if clip_duration > duration:
            if final_clip == "remainder":
                return_clip = Audio(
                    self.samples,
                    self.sample_rate,
                    resample_type=self.resample_type,
                    max_duration=self.max_duration,
                )
                return [
                    {
                        "clip": return_clip,
                        "clip_duration": return_clip.duration(),
                        "begin_time": 0,
                        "end_time": duration,
                    }
                ]
            elif final_clip in ["full", "extend"]:
                return_clip = self.extend(clip_duration)
                return [
                    {
                        "clip": return_clip,
                        "clip_duration": return_clip.duration(),
                        "begin_time": 0,
                        "end_time": duration,
                    }
                ]
            else:
                warnings.warn(
                    f"Given Audio object with duration of `{duration}` seconds and `clip_duration={clip_duration}` but `final_clip={final_clip}` produces no clips. Returning empty list."
                )
                return []

        clip_times = np.arange(0.0, duration, duration / len(self.samples))
        num_clips = ceil((duration - clip_overlap) / (clip_duration - clip_overlap))
        to_return = [None] * num_clips
        for idx in range(num_clips):
            if idx == num_clips - 1:
                if final_clip in ["remainder", "extend"]:
                    begin_time = clip_duration * idx - clip_overlap * idx
                    end_time = duration
                elif final_clip == "full":
                    begin_time = int(duration - clip_duration)
                    end_time = duration
                else:
                    begin_time = clip_duration * idx - clip_overlap * idx
                    end_time = begin_time + clip_duration
                    if end_time > duration:
                        return to_return[:-1]
            else:
                begin_time = clip_duration * idx - clip_overlap * idx
                end_time = begin_time + clip_duration

            audio_clip = self.trim(begin_time, end_time)
            if final_clip == "extend":
                audio_clip = audio_clip.extend(clip_duration)
            to_return[idx] = {
                "clip": audio_clip,
                "clip_duration": audio_clip.duration(),
                "begin_time": begin_time,
                "end_time": end_time,
            }

        return to_return


def split_and_save(
    audio,
    destination,
    prefix,
    clip_duration=5,
    clip_overlap=1,
    final_clip=None,
    dry_run=False,
):
    """ Split audio into clips and save them to a folder

    Arguments:
        audio:          The input Audio to split
        destination:    A folder to write clips to
        prefix:         A name to prepend to the written clips
        clip_duration:  The duration of each clip in seconds [default: 5]
        clip_overlap:   The overlap of each clip in seconds [default: 1]
        final_clip:     Possible options (any other input will ignore the final clip entirely) [default: None]
                            - "remainder":          Include the remainder of the Audio
                                                        (clip will not have clip_duration length)
                            - "full":               Increase the overlap to yield a clip with clip_duration
                            - "extend":             Similar to remainder but extend the clip to clip_duration
        dry_run:        If True, skip writing audio and just return clip DataFrame [default: False]
    
    Returns:
        pandas.DataFrame containing begin and end times for each clip from the source audio
    """

    clips = audio.split(
        clip_duration=clip_duration, clip_overlap=clip_overlap, final_clip=final_clip
    )
    for clip in clips:
        clip_name = (
            f"{destination}/{prefix}_{clip['begin_time']}s_{clip['end_time']}s.wav"
        )
        if not dry_run:
            clip["clip"].save(clip_name)

    # Convert [{k: v}] -> {k: [v]}
    return pd.DataFrame(
        {key: [clip[key] for clip in clips] for key in clips[0].keys() if key != "clip"}
    )
