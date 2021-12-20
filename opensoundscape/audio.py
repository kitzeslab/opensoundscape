#!/usr/bin/env python3
""" audio.py: Utilities for loading and modifying Audio objects


**Note: Out-of-place operations**

Functions that modify Audio (and Spectrogram) objects are "out of place",
meaning that they return a new Audio object instead of modifying the
original object. This means that running a line
```
audio_object.resample(22050) # WRONG!
```
will **not** change the sample rate of `audio_object`!
If your goal was to overwrite `audio_object` with the new,
resampled audio, you would instead write
```
audio_object = audio_object.resample(22050)
```

"""

import librosa
import soundfile
import numpy as np
import pandas as pd
import warnings
from math import ceil
from opensoundscape.helpers import generate_clip_times_df
from opensoundscape.audiomoth import parse_audiomoth_metadata
from tinytag import TinyTag
from datetime import timedelta


class OpsoLoadAudioInputError(Exception):
    """Custom exception indicating we can't load input"""

    pass


class OpsoLoadAudioInputTooLong(Exception):
    """Custom exception indicating length of audio is too long"""

    pass


class Audio:
    """Container for audio samples

    Initialization requires sample array. To load audio file, use
    `Audio.from_file()`

    Initializing an `Audio` object directly requires the specification of the
    sample rate. Use `Audio.from_file` or `Audio.from_bytesio` with
    `sample_rate=None` to use a native sampling rate.

    Args:
        samples (np.array):     The audio samples
        sample_rate (integer):  The sampling rate for the audio samples
        resample_type (str):    The resampling method to use [default: "kaiser_fast"]
        max_duration (None or integer):
            The maximum duration in seconds allowed for the audio file
            (longer files will raise an exception)[default: None]
            If None, no limit is enforced

    Returns:
        An initialized `Audio` object
    """

    __slots__ = ("samples", "sample_rate", "resample_type", "max_duration", "metadata")

    def __init__(
        self,
        samples,
        sample_rate,
        resample_type="kaiser_fast",
        max_duration=None,
        metadata=None,
    ):
        self.samples = samples
        self.sample_rate = sample_rate
        self.resample_type = resample_type
        self.max_duration = max_duration
        self.metadata = metadata

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
        cls,
        path,
        sample_rate=None,
        resample_type="kaiser_fast",
        max_duration=None,
        metadata=True,
        offset=0,
        duration=None,
    ):
        """Load audio from files

        Deal with the various possible input types to load an audio file
        Also attempts to load metadata using tinytag.

        Audio objects only support mono (one-channel) at this time. Files
        with multiple channels are mixed down to a single channel.

        Optionally, load only a piece of a file using `offset` and `duration`.
        This will efficiently read sections of a .wav file regardless of where
        the desired clip is in the audio. For mp3 files, access time grows
        linearly with time since the beginning of the file.

        This function relies on librosa.load(), which supports wav natively but
        requires ffmpeg for mp3 support.

        Args:
            path (str, Path): path to an audio file
            sample_rate (int, None): resample audio with value and resample_type,
                if None use source sample_rate (default: None)
            resample_type: method used to resample_type (default: kaiser_fast)
            max_duration: the maximum length of an input file,
                None is no maximum (default: None)
            metadata (bool): if True, attempts to load metadata from the audio
                file. If an exception occurs, self.metadata will be `None`.
                Otherwise self.metadata is a dictionary.
                Note: will also attempt to parse AudioMoth metadata from the
                `comment` field, if the `artist` field includes `AudioMoth`.
                The parsing function for AudioMoth is likely to break when new
                firmware versions change the `comment` metadata field.
            offset: load audio starting at this time (seconds) after the
                start of the file. Default: 0 seconds.
            duration: load audio of this duration (seconds) starting at
                `offset`. If None, loads all the way to the end of the file.

        Returns:
            Audio object with attributes: samples, sample_rate, resample_type,
            max_duration, metadata (dict or None)

        Note: default sample_rate=None means use file's sample rate, don't
        resample

        """
        path = str(path)  # Pathlib path can have dependency issues - use string
        if max_duration:
            if librosa.get_duration(filename=path) > max_duration:
                raise OpsoLoadAudioInputTooLong()

        ## Load samples ##
        warnings.filterwarnings("ignore")
        samples, sr = librosa.load(
            path,
            sr=sample_rate,
            res_type=resample_type,
            mono=True,
            offset=offset,
            duration=duration,
        )
        warnings.resetwarnings()

        ## Load Metadata ##
        try:
            metadata = TinyTag.get(path).as_dict()
            # if this is an AudioMoth file, try to parse out additional
            # metadata from the comment field
            if metadata["artist"] and "AudioMoth" in metadata["artist"]:
                try:
                    metadata = parse_audiomoth_metadata(metadata)
                    # if the offset > 0, we need to update the timestamp
                    metadata["recording_start_time"] += timedelta(seconds=round(offset))
                except Exception as e:
                    warnings.warn(
                        "This seems to be an AudioMoth file, "
                        f"but parse_audiomoth_metadata() raised: {e}"
                    )

            ## Update metadata ##
            metadata["channels"] = 1

            # update the duration because we may have only loaded
            # a piece of the entire audio file.
            metadata["duration"] = len(samples) / sr

            # update the sample rate in metadata
            metadata["samplerate"] = sr

            # if we loaded part we don't know the file size anymore
            if offset > 0 or duration is not None:
                metadata["filesize"] = np.nan

        except Exception as e:
            warnings.warn(f"Failed to load metadata: {e}. Metadata will be None")
            metadata = None

        return cls(
            samples,
            sr,
            resample_type=resample_type,
            max_duration=max_duration,
            metadata=metadata,
        )

    @classmethod
    def from_bytesio(
        cls, bytesio, sample_rate=None, max_duration=None, resample_type="kaiser_fast"
    ):
        """Read from bytesio object

        Read an Audio object from a BytesIO object. This is primarily used for
        passing Audio over HTTP.

        Args:
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

    def resample(self, sample_rate, resample_type=None):
        """Resample Audio object

        Args:
            sample_rate (scalar):   the new sample rate
            resample_type (str):    resampling algorithm to use [default: None
                                    (uses self.resample_type of instance)]

        Returns:
            a new Audio object of the desired sample rate
        """
        if resample_type is None:
            resample_type = self.resample_type

        samples_resampled = librosa.resample(
            self.samples,
            orig_sr=self.sample_rate,
            target_sr=sample_rate,
            res_type=resample_type,
        )

        return Audio(
            samples_resampled,
            sample_rate,
            resample_type=resample_type,
            max_duration=self.max_duration,
        )

    def trim(self, start_time, end_time):
        """Trim Audio object in time

        If start_time is less than zero, output starts from time 0
        If end_time is beyond the end of the sample, trims to end of sample

        Args:
            start_time: time in seconds for start of extracted clip
            end_time: time in seconds for end of extracted clip
        Returns:
            a new Audio object containing samples from start_time to end_time
        """
        start_sample = max(0, self.time_to_sample(start_time))
        end_sample = self.time_to_sample(end_time)
        samples_trimmed = self.samples[start_sample:end_sample]
        return Audio(
            samples_trimmed,
            self.sample_rate,
            resample_type=self.resample_type,
            max_duration=self.max_duration,
        )

    def loop(self, length=None, n=None):
        """Extend audio file by looping it

        Args:
            length:
                the final length in seconds of the looped file
                (cannot be used with n)[default: None]
            n:
                the number of occurences of the original audio sample
                (cannot be used with length) [default: None]
                For example, n=1 returns the original sample, and
                n=2 returns two concatenated copies of the original sample

        Returns:
            a new Audio object of the desired length or repetitions
        """
        if (length is None) + (n is None) != 1:
            raise ValueError("Please enter a value for 'length' OR " "'n', not both")

        if length is not None:
            # loop the audio until it reaches a duration of `length` seconds
            total_samples_needed = round(length * self.sample_rate)
            samples_extended = np.resize(self.samples, total_samples_needed)

        else:  # loop the audio n times
            samples_extended = np.tile(self.samples, n)
        return Audio(
            samples_extended,
            self.sample_rate,
            resample_type=self.resample_type,
            max_duration=self.max_duration,
        )

    def extend(self, length):
        """Extend audio file by adding silence to the end

        Args:
            length: the final duration in seconds of the extended audio object

        Returns:
            a new Audio object of the desired duration
        """

        total_samples_needed = round(length * self.sample_rate)
        samples_extended = np.pad(
            self.samples, pad_width=(0, total_samples_needed - len(self.samples))
        )
        return Audio(
            samples_extended,
            self.sample_rate,
            resample_type=self.resample_type,
            max_duration=self.max_duration,
        )

    def time_to_sample(self, time):
        """Given a time, convert it to the corresponding sample

        Args:
            time: The time to multiply with the sample_rate

        Returns:
            sample: The rounded sample
        """
        return int(time * self.sample_rate)

    def bandpass(self, low_f, high_f, order):
        """Bandpass audio signal with a butterworth filter

        Uses a phase-preserving algorithm (scipy.signal's butter and solfiltfilt)

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

    def spectrum(self):
        """Create frequency spectrum from an Audio object using fft

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
        """Save Audio to file

        NOTE: currently, only saving to .wav format supported

        Args:
            path: destination for output
        """
        from soundfile import write

        if not str(path).split(".")[-1] in ["wav", "WAV"]:
            raise TypeError(
                "Only wav file is currently supported by .save()."
                " File extension must be .wav or .WAV. "
            )

        write(path, self.samples, self.sample_rate)

    def duration(self):
        """Return duration of Audio

        Returns:
            duration (float): The duration of the Audio
        """

        return len(self.samples) / self.sample_rate

    def split(self, clip_duration, clip_overlap=0, final_clip=None):
        """Split Audio into even-lengthed clips

        The Audio object is split into clips of a specified duration and overlap

        Args:
            clip_duration (float):  The duration in seconds of the clips
            clip_overlap (float):   The overlap of the clips in seconds [default: 0]
            final_clip (str):       Behavior if final_clip is less than clip_duration
                seconds long. By default, discards remaining audio if less than
                clip_duration seconds long [default: None].
                Options:
                    - None:         Discard the remainder (do not make a clip)
                    - "extend":     Extend the final clip with silence to reach clip_duration length
                    - "remainder":  Use only remainder of Audio (final clip will be shorter than clip_duration)
                    - "full":       Increase overlap with previous clip to yield a clip with clip_duration length
        Returns:
            - audio_clips: list of audio objects
            - dataframe w/columns for start_time and end_time of each clip
        """
        if not final_clip in ["remainder", "full", "extend", None]:
            raise ValueError(
                f"final_clip must be 'remainder', 'full', 'extend',"
                f"or None. Got {final_clip}."
            )

        duration = self.duration()
        clip_df = generate_clip_times_df(
            full_duration=duration,
            clip_duration=clip_duration,
            clip_overlap=clip_overlap,
            final_clip=final_clip,
        )

        clips = [None] * len(clip_df)
        for idx, (start, end) in enumerate(
            zip(clip_df["start_time"], clip_df["end_time"])
        ):

            # Trim the clip to desired range
            audio_clip = self.trim(start, end)

            # Extend the final clip if necessary
            if end > duration and final_clip == "extend":
                audio_clip = audio_clip.extend(clip_duration)

            # Add clip to list of clips
            clips[idx] = audio_clip

        if len(clips) == 0:
            warnings.warn(
                f"Given Audio object with duration of `{duration}` "
                f"seconds and `clip_duration={clip_duration}` but "
                f" `final_clip={final_clip}` produces no clips. "
                f"Returning empty list."
            )

        return clips, clip_df

    def split_and_save(
        self,
        destination,
        prefix,
        clip_duration,
        clip_overlap=0,
        final_clip=None,
        dry_run=False,
    ):
        """Split audio into clips and save them to a folder

        Args:
            destination:        A folder to write clips to
            prefix:             A name to prepend to the written clips
            clip_duration:      The duration of each clip in seconds
            clip_overlap:       The overlap of each clip in seconds [default: 0]
            final_clip (str):   Behavior if final_clip is less than clip_duration seconds long. [default: None]
                By default, ignores final clip entirely.
                Possible options (any other input will ignore the final clip entirely),
                    - "remainder":  Include the remainder of the Audio (clip will not have clip_duration length)
                    - "full":       Increase the overlap to yield a clip with clip_duration length
                    - "extend":     Similar to remainder but extend (repeat) the clip to reach clip_duration length
                    - None:         Discard the remainder
            dry_run (bool):      If True, skip writing audio and just return clip DataFrame [default: False]

        Returns:
            pandas.DataFrame containing paths and start and end times for each clip
        """

        clips, df = self.split(
            clip_duration=clip_duration,
            clip_overlap=clip_overlap,
            final_clip=final_clip,
        )
        clip_names = []
        for i, clip in enumerate(clips):
            start_t = df.at[i, "start_time"]
            end_t = df.at[i, "end_time"]
            clip_name = f"{destination}/{prefix}_{start_t}s_{end_t}s.wav"
            clip_names.append(clip_name)
            if not dry_run:
                clip.save(clip_name)

        df.index = clip_names
        df.index.name = "file"
        return df
