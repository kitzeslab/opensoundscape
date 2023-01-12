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
import warnings
from datetime import timedelta, datetime
from pathlib import Path
import json

import numpy as np
from scipy.fftpack import fft as scipyfft
from scipy.fft import fftfreq
import librosa
import soundfile

import opensoundscape
from opensoundscape.helpers import generate_clip_times_df, load_metadata
from opensoundscape.audiomoth import parse_audiomoth_metadata
from opensoundscape.audio_tools import bandpass_filter


class OpsoLoadAudioInputError(Exception):
    """Custom exception indicating we can't load input"""


class AudioOutOfBoundsError(Exception):
    """Custom exception indicating the user tried to load audio
    outside of the time period that exists in the audio object"""


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

    Returns:
        An initialized `Audio` object
    """

    __slots__ = ("samples", "sample_rate", "resample_type", "metadata")

    def __init__(
        self, samples, sample_rate, resample_type="kaiser_fast", metadata=None
    ):
        self.samples = samples
        self.sample_rate = sample_rate
        self.resample_type = resample_type
        self.metadata = metadata

        samples_error = None
        if not (isinstance(self.samples, np.ndarray) or isinstance(self.samples, list)):
            samples_error = (
                "Initializing an Audio object requires samples to be a numpy "
                "array or list"
            )
        self.samples = np.array(self.samples).astype("float32")

        try:
            self.sample_rate = int(self.sample_rate)
        except ValueError as exc:
            sample_rate_error = (
                "Initializing an Audio object requires the audio samples' sampling rate"
            )
            if samples_error:
                raise ValueError(
                    f"Audio initialization failed with:\n{samples_error}\n{sample_rate_error}"
                ) from exc
            raise ValueError(
                f"Audio initialization failed with:\n{sample_rate_error}"
            ) from exc

        if samples_error:
            raise ValueError(f"Audio initialization failed with:\n{samples_error}")

    @classmethod
    def silence(cls, duration, sample_rate):
        """ "Create audio object with zero-valued samples

        Args:
            duration: length in seconds
            sample_rate: samples per second

        Note: rounds down to integer number of samples
        """
        return cls(np.zeros(int(duration * sample_rate)), sample_rate)

    @classmethod
    def noise(cls, duration, sample_rate, color="white", dBFS=-10):
        """ "Create audio object with noise of a desired 'color'

        set np.random.seed() for reproducible results

        Based on an implementatino by @Bob in StackOverflow question 67085963

        Args:
            duration: length in seconds
            sample_rate: samples per second
            color: any of these colors, which describe the shape of the power spectral density:
                - white: uniform psd (equal energy per linear frequency band)
                - pink: psd = 1/sqrt(f) (equal energy per octave)
                - brownian: psd = 1/f (aka brown noise)
                - brown: synonym for brownian
                - violet: psd = f
                - blue: psd = sqrt(f)
            [default: 'white']

        Returns: Audio object

        Note: Clips samples to [-1,1] which can result in dBFS different from that
        requested, especially when dBFS is near zero
        """
        # look-up dictionary for relationship of power spectral density with frequency
        psd_functions = dict(
            white=lambda f: 1,
            blue=lambda f: np.sqrt(f),
            violet=lambda f: f,
            brownian=lambda f: 1 / np.where(f == 0, float("inf"), f),
            brown=lambda f: 1 / np.where(f == 0, float("inf"), f),
            pink=lambda f: 1 / np.where(f == 0, float("inf"), np.sqrt(f)),
        )
        n_samples = int(duration * sample_rate)
        assert color in psd_functions, f"Invalid color {color}"
        psd = psd_functions[color]

        white = np.fft.rfft(np.random.randn(n_samples))
        target_psd = psd(np.fft.rfftfreq(n_samples))
        # Normalize S for rms of desired dBFS
        target_psd = (
            target_psd
            / np.sqrt(np.mean(target_psd**2))
            * (10 ** (dBFS / 20))
            / np.sqrt(2)
        )

        shaped = white * target_psd

        samples = np.fft.irfft(shaped)

        return cls(np.clip(samples, -1, 1), sample_rate)

    @classmethod
    def from_file(
        cls,
        path,
        sample_rate=None,
        resample_type="kaiser_fast",
        dtype=np.float32,
        load_metadata=True,
        offset=None,
        duration=None,
        start_timestamp=None,
        out_of_bounds_mode="warn",
    ):
        """Load audio from files

        Deal with the various possible input types to load an audio file
        Also attempts to load metadata using tinytag.

        Audio objects only support mono (one-channel) at this time. Files
        with multiple channels are mixed down to a single channel. To load
        multiple channels as separate Audio objects, use `load_channels_as_audio()`

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
            dtype: data type of samples returned [Default: np.float32]
            load_metadata (bool): if True, attempts to load metadata from the audio
                file. If an exception occurs, self.metadata will be `None`.
                Otherwise self.metadata is a dictionary.
                Note: will also attempt to parse AudioMoth metadata from the
                `comment` field, if the `artist` field includes `AudioMoth`.
                The parsing function for AudioMoth is likely to break when new
                firmware versions change the `comment` metadata field.
            offset: load audio starting at this time (seconds) after the
                start of the file. Defaults to 0 seconds.
                - cannot specify both `offset` and `start_timestamp`
            duration: load audio of this duration (seconds) starting at
                `offset`. If None, loads all the way to the end of the file.
            start_timestamp: load audio starting at this localized datetime.datetime timestamp
                - cannot specify both `offset` and `start_timestamp`
                - will only work if loading metadata results in localized datetime
                    object for 'recording_start_time' key
                - will raise AudioOutOfBoundsError if requested time period
                is not full contained within the audio file
                Example of creating localized timestamp:
                ```
                import pytz; from datetime import datetime;
                local_timestamp = datetime(2020,12,25,23,59,59)
                local_timezone = pytz.timezone('US/Eastern')
                timestamp = local_timezone.localize(local_timestamp)
                ```
            out_of_bounds_mode:
                - 'warn': generate a warning [default]
                - 'raise': raise an AudioOutOfBoundsError
                - 'ignore': return any available audio with no warning/error

        Returns:
            Audio object with attributes: samples, sample_rate, resample_type,
            metadata (dict or None)

        Note: default sample_rate=None means use file's sample rate, does not
        resample

        """
        assert out_of_bounds_mode in ["raise", "warn", "ignore"]

        path = str(path)  # Pathlib path can have dependency issues - use string

        ## Load Metadata ##
        if load_metadata:
            metadata = _metadata_from_file_handler(path)
        else:
            metadata = None

        ## Determine start time / offset ##
        if start_timestamp is not None:
            # user should have provied a localized timestamp as the start_timestamp
            assert (
                offset is None
            ), "You must not specify both `start_timestamp` and `offset`"
            assert (
                type(start_timestamp) == datetime and start_timestamp.tzinfo is not None
            ), "start_timestamp must be a localized datetime object"
            assert (
                metadata is not None
                and "recording_start_time" in metadata
                and type(metadata["recording_start_time"]) == datetime
            ), (
                "metadata did not contain start time timestamp in key `recording_start_time`. "
                "This key is automatically created when parsing AudioMoth metadata."
            )
            audio_start = metadata["recording_start_time"]
            offset = (start_timestamp - audio_start).total_seconds()
            if offset < 0:
                error_msg = "requested time period begins before start of recording"
                if out_of_bounds_mode == "raise":
                    raise AudioOutOfBoundsError(error_msg)
                elif out_of_bounds_mode == "warn":
                    warnings.warn(error_msg)
                # else: pass

        elif offset is None:  # default offset is 0
            offset = 0

        ## Load samples ##
        warnings.filterwarnings("ignore")
        samples, sr = librosa.load(
            path,
            sr=sample_rate,
            res_type=resample_type,
            mono=True,
            offset=offset,
            duration=duration,
            dtype=None,
        )
        # temporary workaround for soundfile issue #349
        # which causes empty sample array if loading float32 from mp3:
        # pass dtype=None, then change it afterwards
        samples = samples.astype(dtype)
        warnings.resetwarnings()

        # out of bounds warning/exception user if no samples or too short
        if len(samples) == 0:
            error_msg = "audio object has zero samples"
            if out_of_bounds_mode == "raise":
                raise AudioOutOfBoundsError(error_msg)
            elif out_of_bounds_mode == "warn":
                warnings.warn(error_msg)
        elif duration is not None and len(samples) < duration * sr:
            if offset < 0:
                error_msg = "requested time period begins before start of recording"
            else:
                error_msg = (
                    f"Audio object is shorter than requested duration: "
                    f"{len(samples)/sr} sec instead of {duration} sec"
                )
            if out_of_bounds_mode == "raise":
                raise AudioOutOfBoundsError(error_msg)
            elif out_of_bounds_mode == "warn":
                warnings.warn(error_msg)

        ## Update metadata ##
        if metadata is not None:
            # update the duration because we may have only loaded
            # a piece of the entire audio file.
            metadata["duration"] = len(samples) / sr

            # update the sample rate in metadata
            metadata["samplerate"] = sr

            # if we loaded part we don't know the file size anymore
            if offset != 0 or duration is not None:
                metadata["filesize"] = np.nan

            # if the offset > 0, we need to update the timestamp
            if "recording_start_time" in metadata and offset > 0:
                metadata["recording_start_time"] += timedelta(seconds=offset)

        return cls(samples, sr, resample_type=resample_type, metadata=metadata)

    @classmethod
    def from_bytesio(cls, bytesio, sample_rate=None, resample_type="kaiser_fast"):
        """Read from bytesio object

        Read an Audio object from a BytesIO object. This is primarily used for
        passing Audio over HTTP.

        Args:
            bytesio: Contents of WAV file as BytesIO
            sample_rate: The final sampling rate of Audio object [default: None]
            resample_type: The librosa method to do resampling [default: "kaiser_fast"]

        Returns:
            An initialized Audio object
        """
        samples, original_sample_rate = soundfile.read(bytesio)
        if sample_rate is not None and sample_rate != original_sample_rate:
            samples = librosa.resample(
                samples,
                orig_sr=original_sample_rate,
                target_sr=sample_rate,
                res_type=resample_type,
            )
        else:
            sample_rate = original_sample_rate

        return cls(samples, sample_rate, resample_type=resample_type)

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

        return Audio(samples_resampled, sample_rate, resample_type=resample_type)

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
        start_sample = max(0, self._get_sample_index(start_time))
        end_sample = self._get_sample_index(end_time)
        samples_trimmed = self.samples[start_sample:end_sample]

        # update metadata with new start time and duration
        if self.metadata is None:
            metadata = None
        else:
            metadata = self.metadata.copy()
            if "recording_start_time" in metadata:
                metadata["recording_start_time"] += timedelta(seconds=start_time)

            if "duration" in metadata:
                metadata["duration"] = len(samples_trimmed) / self.sample_rate

        return Audio(
            samples_trimmed,
            self.sample_rate,
            resample_type=self.resample_type,
            metadata=metadata,
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
            raise ValueError("Please enter a value for 'length' OR 'n', not both")

        if length is not None:
            # loop the audio until it reaches a duration of `length` seconds
            total_samples_needed = round(length * self.sample_rate)
            samples_extended = np.resize(self.samples, total_samples_needed)
        else:  # loop the audio n times
            samples_extended = np.tile(self.samples, n)

        # update metadata to reflect new duration
        if self.metadata is None:
            metadata = None
        else:
            metadata = self.metadata.copy()
            if "duration" in metadata:
                metadata["duration"] = len(samples_extended) / self.sample_rate

        return Audio(
            samples_extended,
            self.sample_rate,
            resample_type=self.resample_type,
            metadata=metadata,
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

        # update metadata to reflect new duration
        if self.metadata is None:
            metadata = None
        else:
            metadata = self.metadata.copy()
            if "duration" in metadata:
                metadata["duration"] = len(samples_extended) / self.sample_rate

        return Audio(
            samples_extended,
            self.sample_rate,
            resample_type=self.resample_type,
            metadata=metadata,
        )

    def bandpass(self, low_f, high_f, order):
        """Bandpass audio signal with a butterworth filter

        Uses a phase-preserving algorithm (scipy.signal's butter and solfiltfilt)

        Args:
            low_f: low frequency cutoff (-3 dB)  in Hz of bandpass filter
            high_f: high frequency cutoff (-3 dB)  in Hz of bandpass filter
            order: butterworth filter order (integer) ~= steepness of cutoff

        """

        if low_f <= 0:
            raise ValueError("low_f must be greater than zero")

        if high_f >= self.sample_rate / 2:
            raise ValueError("high_f must be less than sample_rate/2")

        filtered_samples = bandpass_filter(
            self.samples, low_f, high_f, self.sample_rate, order=order
        )
        return Audio(
            filtered_samples, self.sample_rate, resample_type=self.resample_type
        )

    def spectrum(self):
        """Create frequency spectrum from an Audio object using fft

        Args:
            self

        Returns:
            fft, frequencies
        """

        # Compute the fft (fast fourier transform) of the selected clip
        N = len(self.samples)
        fft = scipyfft(self.samples)

        # create the frequencies corresponding to fft bins
        freq = fftfreq(N, d=1 / self.sample_rate)

        # remove negative frequencies and scale magnitude by 2.0/N:
        fft = 2.0 / N * fft[0 : int(N / 2)]
        frequencies = freq[0 : int(N / 2)]
        fft = np.abs(fft)

        return fft, frequencies

    def normalize(self, peak_level=None, peak_dBFS=None):
        """Return audio object with normalized waveform

        Linearly scales waveform values so that the max absolute value matches
        the specified value (default: 1.0)

        args:
            peak_level: maximum absolute value of resulting waveform
            peak_dBFS: maximum resulting absolute value in decibels Full Scale
                - for example, -3 dBFS equals a peak level of 0.71
                - Note: do not specify both peak_level and peak_dBFS

        returns:
            Audio object with normalized samples

        Note: if all samples are zero, returns the original object (avoids
        division by zero)

        """
        # make sure the user didn't request peak level both ways
        if peak_level is not None and peak_dBFS is not None:
            raise ValueError("Must not specify both peak_level and peak_dBFS")

        if peak_level is None and peak_dBFS is None:
            peak_level = 0
        elif peak_dBFS is not None:
            if peak_dBFS > 0:
                warnings.warn("user requested decibels Full Scale >0 !")

            # dBFS is defined as 20*log10(V), so V = 10^(dBFS/20)
            peak_level = 10 ** (peak_dBFS / 20)

        abs_max = max(max(self.samples), -min(self.samples))
        if abs_max == 0:
            # don't try to normalize 0-valued samples. Return original object
            abs_max = 1

        return Audio(
            self.samples / abs_max * peak_level,
            self.sample_rate,
            resample_type=self.resample_type,
            metadata=self.metadata,
        )

    def apply_gain(self, dB, clip_range=(-1, 1)):
        """apply dB (decibels) of gain to audio signal

        Specifically, multiplies samples by 10^(dB/20)

        Args:
            dB: decibels of gain to apply
            clip_range: [minimum,maximum] values for samples
                - values outside this range will be replaced with the range
                boundary values. Pass `None` to preserve original sample values
                without clipping. [Default: [-1,1]]

        Returns:
            Audio object with gain applied to samples
        """
        samples = self.samples * (10 ** (dB / 20))
        if clip_range is not None:
            samples = np.clip(samples, clip_range[0], clip_range[1])
        return Audio(
            samples,
            self.sample_rate,
            resample_type=self.resample_type,
            metadata=self.metadata,
        )

    def save(
        self,
        path,
        metadata_format="opso",
        soundfile_subtype=None,
        soundfile_format=None,
        suppress_warnings=False,
    ):
        """Save Audio to file

        supports all file formats supported by underlying package soundfile,
        including WAV, MP3, and others

        NOTE: saving metadata is only supported for WAV and AIFF formats

        Supports writing the following metadata fields:
        ["title","copyright","software","artist","comment","date",
        "album","license","tracknumber","genre"]

        Args:
            path: destination for output
            metadata_format: strategy for saving metadata. Can be:
                - 'opso' [Default]: Saves metadata dictionary in the comment
                    field as a JSON string. Uses the most recent version of opso_metadata
                    formats.
                - 'opso_metadata_v0.1': specify the exact version of opso_metadata to use
                - 'soundfile': Saves the default soundfile metadata fields only:
                    ["title","copyright","software","artist","comment","date",
                     "album","license","tracknumber","genre"]
                - None: does not save metadata to file
            soundfile_subtype: soundfile audio subtype choice, see soundfile.write
                or list options with soundfile.available_subtypes()
            soundfile_format: soundfile audio format choice, see soundfile.write
            suppress_warnings: if True, will not warn user when unable to
                save metadata [default: False]
        """
        fmt = Path(path).suffix.upper()

        try:
            soundfile.write(
                file=path,
                data=self.samples,
                samplerate=self.sample_rate,
                subtype=soundfile_subtype,
                format=soundfile_format,
            )
        except ValueError as exc:
            raise NotImplementedError(
                "Failed to save file with soundfinder. "
                "This may be because the underlying package `libsndfile` must be "
                "version >=1.1.0 to write mp3 files. \n"
                "Note that as of Dec 2022, libsndfile 1.1.0 is not available on Ubuntu."
            ) from exc

        if metadata_format is not None and self.metadata is not None:
            if fmt not in [".WAV", ".AIFF"]:
                if not suppress_warnings:
                    warnings.warn(
                        "Saving metadata is only supported for WAV and AIFF formats"
                    )
            else:  # we can write metadata for WAV and AIFF
                _write_metadata(self.metadata, metadata_format, path)

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
                - None: Discard the remainder (do not make a clip)
                - "extend": Extend the final clip with silence to reach
                    clip_duration length
                - "remainder": Use only remainder of Audio (final clip will be
                    shorter than clip_duration)
                - "full": Increase overlap with previous clip to yield a clip with
                    clip_duration length
        Returns:
            - audio_clips: list of audio objects
            - dataframe w/columns for start_time and end_time of each clip
        """
        if not final_clip in ["remainder", "full", "extend", None]:
            raise ValueError(
                f"final_clip must be 'remainder', 'full', 'extend',"
                f"or None. Got {final_clip}."
            )

        duration = self.duration
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
            destination: A folder to write clips to
            prefix: A name to prepend to the written clips
            clip_duration: The duration of each clip in seconds
            clip_overlap: The overlap of each clip in seconds [default: 0]
            final_clip (str): Behavior if final_clip is less than clip_duration seconds long.
            [default: None]
                By default, ignores final clip entirely.
                Possible options (any other input will ignore the final clip entirely),
                    - "remainder": Include the remainder of the Audio (clip will not have
                      clip_duration length)
                    - "full": Increase the overlap to yield a clip with clip_duration length
                    - "extend": Similar to remainder but extend (repeat) the clip to reach
                      clip_duration length
                    - None: Discard the remainder
            dry_run (bool): If True, skip writing audio and just return clip DataFrame
                [default: False]

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

    def _get_sample_index(self, time):
        """Given a time, convert it to the corresponding sample

        Args:
            time: The time to multiply with the sample_rate

        Returns:
            sample: The rounded sample

        Note: always rounds down (casts to int)
        """
        return int(time * self.sample_rate)

    @property
    def duration(self):
        """Calculates the Audio duration in seconds"""
        return len(self.samples) / self.sample_rate

    @property
    def rms(self):
        """Calculates the root-mean-square value of the audio samples"""
        return np.sqrt(np.mean(self.samples**2))

    @property
    def dBFS(self):
        """calculate the root-mean-square dB value relative to a full-scale sine wave"""
        return 20 * np.log10(self.rms * np.sqrt(2))


def load_channels_as_audio(
    path,
    sample_rate=None,
    resample_type="kaiser_fast",
    dtype=np.float32,
    offset=0,
    duration=None,
    metadata=True,
):
    """Load each channel of an audio file to a separate Audio object

    Provides a way to access individual channels, since Audio.from_file
    mixes down to mono by default

    args:
        see Audio.from_file()

    returns:
        list of Audio objects (one per channel)

    Note: metadata is copied to each Audio object, but will contain an
        additional field: "channel"="1 of 3" for first of 3 channels
    """
    path = str(path)  # Pathlib path can have dependency issues - use string

    ## Load Metadata ##
    if metadata:
        metadata_dict = _metadata_from_file_handler(path)
    else:
        metadata_dict = None

    ## Load samples ##
    warnings.filterwarnings("ignore")
    samples, sr = librosa.load(
        path,
        sr=sample_rate,
        res_type=resample_type,
        mono=False,
        offset=offset,
        duration=duration,
        dtype=None,
    )
    # temporary workaround for soundfile issue #349
    # which causes empty sample array if loading float32 from mp3:
    # pass dtype=None, then change it afterwards
    samples = samples.astype(dtype)

    warnings.resetwarnings()
    if len(np.shape(samples)) == 1:
        samples = [samples]

    # create an audio object for each channel
    # adding a metadata field to track which of
    # the original channels it represents
    audio_objects = []
    for i, samples_channel in enumerate(samples):
        channel_metadata = metadata_dict.copy()
        channel_metadata["channels"] = 1
        channel_metadata["channel"] = f"{i+1} of {len(samples)}"
        audio_objects.append(
            Audio(
                samples=samples_channel,
                sample_rate=sr,
                resample_type=resample_type,
                metadata=channel_metadata,
            )
        )
    return audio_objects


def concat(audio_objects, sample_rate=None):
    """concatenate a list of Audio objects end-to-end

    Args:
        audio_objects: iterable of Audio objects
        sample_rate: target sampling rate
            - if None, uses sampling rate of _first_
            Audio object in list
            - default: None

    Returns: a single Audio object

    Notes: discards metadata and retains .resample_type of _first_ audio object
    """

    assert np.all(
        [type(a) == Audio for a in audio_objects]
    ), "all elements in audio_objects must be Audio objects"

    # use first object's sample rate if None provided
    if sample_rate is None:
        sample_rate = audio_objects[0].sample_rate

    # concatenate sample arrays to form new Audio object
    return Audio(
        np.hstack([a.resample(sample_rate).samples for a in audio_objects]),
        sample_rate,
        resample_type=audio_objects[0].resample_type,
    )


def mix(
    audio_objects,
    duration=None,
    gain=-3,
    offsets=None,
    sample_rate=None,
    clip_range=(-1, 1),
):
    """mixdown (superimpose) Audio signals into a single Audio object

    Adds audio samples from multiple audio objects to create a mixdown
    of Audio samples. Resamples all audio to a consistent sample rate,
    and optionally applies individual gain and time-offsets to each Audio.

    Args:
        audio_objects: iterable of Audio objects
        duration: duration in seconds of returned Audio. Can be:
            - number: extends shorter Audio with silence
                and truncates longer Audio
            - None: extends all Audio to the length of the longest
                value of (Audio.duration + offset)
            [default: None]
        gain: number, list of numbers, or None
            - number: decibles of gain to apply to all objects
            - list of numbers: dB of gain to apply to each object
                (length must match length of audio_objects)
            [default: -3 dB on each object]
        offsets: list of time-offsets (seconds) for each Audio object
            For instance [0,1] starts the first Audio at 0 seconds and
            shifts the second Audio to start at 1.0 seconds
            - if None, all objects start at time 0
            - otherwise, length must match length of audio_objects.
        sample_rate: sample rate of returned Audio object
            - integer: resamples all audio to this sample rate
            - None: uses sample rate of _first_ Audio object
            [default: None]
        clip_range: minimum and maximum sample values. Samples outside
            this range will be replaced by the range limit values
            Pass None to keep sample values without clipping.
            [default: (-1,1)]


    Returns:
        Audio object

    Notes:
        Audio metadata is discarded. .resample_type of first Audio is retained.
        Resampling of each Audio uses respective .resample_type of objects.

    """

    ## Input validation ##

    assert np.all(
        [type(a) == Audio for a in audio_objects]
    ), "all elements in audio_objects must be Audio objects"

    if hasattr(duration, "__iter__"):
        assert len(duration) == len(audio_objects), (
            f"duration must be a number, None, or an iterable of the same "
            f"length as audio_objects. duration length {len(duration)} does not "
            f"match audio_objects length {len(audio_objects)}."
        )

    if hasattr(gain, "__iter__"):
        assert len(gain) == len(audio_objects), (
            f"gain must be a number, None, or an iterable of the same "
            f"length as audio_objects. gain length {len(duration)} does not "
            f"match audio_objects length {len(audio_objects)}."
        )

    if offsets is not None:
        assert len(offsets) == len(audio_objects), (
            f"offsets must be None or an iterable of the same "
            f"length as audio_objects. offsets length {len(duration)} does not "
            f"match audio_objects length {len(audio_objects)}."
        )

    if sample_rate is None:
        sample_rate = audio_objects[0].sample_rate

    if duration is None:
        if offsets is not None:
            duration = max(
                [a.duration + offsets[i] for i, a in enumerate(audio_objects)]
            )
        else:
            duration = max([a.duration for a in audio_objects])

    ## Create mixdown ##

    mixdown = np.zeros(int(duration * sample_rate))

    for i, audio in enumerate(audio_objects):
        # apply volume (gain) adjustment to this Audio
        if hasattr(gain, "__iter__"):
            audio = audio.apply_gain(gain[i])
        elif gain is not None:
            audio = audio.apply_gain(gain)

        # resample if required
        if audio.sample_rate != sample_rate:
            audio = audio.resample(sample_rate)

        # add offset if desired
        if offsets is not None:
            audio = concat(
                [Audio.silence(duration=offsets[i], sample_rate=sample_rate), audio]
            )

        # pad or truncate to correct length
        if audio.duration < duration:
            audio = audio.extend(duration)
        elif audio.duration > duration:
            audio = audio.trim(0, duration)

        # add samples to mixdown
        mixdown += audio.samples

    # limit sample values to clip_range
    if clip_range is not None:
        mixdown = np.clip(mixdown, clip_range[0], clip_range[1])

    return Audio(mixdown, sample_rate, resample_type=audio_objects[0].resample_type)


def parse_opso_metadata(comment_string):
    """parse metadata saved by opensoundcsape as json in comment field

    Parses a json string which opensoundscape saves to the comment metadata field
    of WAV files to preserve metadata. The string begins with `opso_metadata`
    The contents of the string after this 13 character prefix should be parsable
    as JSON, and should have a key `opso_metadata_version` specifying the version
    of the metadata format, for instance 'v0.1'.

    see also `generate_opso_metadata` which generates the string parsed by
    this function.

    Args:
        comment_string: a string beginning with `opso_metadata` followed
            by JSON parseable dictionary

    Returns: dictionary of parsed metadata
    """
    assert comment_string[:13] == "opso_metadata", (
        "Comment string did not begin" "with 'opso_metadata'."
    )

    metadata = json.loads(comment_string[13:])
    metadata_version = metadata["opso_metadata_version"]
    if metadata_version == "v0.1":
        # parse and re-format according to opso_metadata_v0.1 formatting
        if "recording_start_time" in metadata:
            metadata["recording_start_time"] = datetime.fromisoformat(
                metadata["recording_start_time"]
            )
    # elif: # implement parsing of future metadata versions here
    else:
        raise NotImplementedError(
            f"Parsing opso_metadata version {metadata_version} "
            "has not been implemented."
        )

    return metadata


def generate_opso_metadata_str(metadata_dictionary, version="v0.1"):
    """generate json string for comment field containing metadata

    Preserve Audio.metadata dictionary by dumping to a json string
    and including it as the 'comment' field when saving WAV files.

    The string begins with `opso_metadata`
    The contents of the string after this 13 character prefix should be parsable
    as JSON, and should have a key `opso_metadata_version` specifying the version
    of the metadata format, for instance 'v0.1'.

    See also: `parse_opso_metadata` which parses the string created by this
    fundtion

    Args:
        metadata_dictionary: dictionary of audio metadata. Should conform
            to opso_metadata version. v0.1 should have only strings and floats
            except the "recording_start_time" key, which should contain a
            localized (ie has timezone) datetime.datetime object. The datetime
            is saved as a string in ISO format using datetime.isoformat()
            and loaded with datetime.fromisoformat().
        version: version number of opso_metadata format.
            Currently implemented: ['v0.1']


    Returns:
        string beginning with `opso_metadata` followed by JSON-parseable
        string containing the metadata.
    """
    metadata = metadata_dictionary.copy()
    metadata["opso_metadata_version"] = version
    if version == "v0.1":
        # formatting rules for v0.1:
        if "recording_start_time" in metadata:
            metadata["recording_start_time"] = metadata[
                "recording_start_time"
            ].isoformat()
        metadata["opensoundscape_version"] = opensoundscape.__version__
    # elif #implement future versions of metadata format here
    else:
        raise NotImplementedError(
            f"Saving opso_metadata version {version} has not been implemented."
        )
    return "opso_metadata" + json.dumps(metadata)


def _metadata_from_file_handler(path):
    try:
        metadata = load_metadata(path)
        # if we have saved this file an opso_metadata json string in
        # the comment field, re-load the metadata dictionary by parsing
        # the json string
        if "comment" in metadata and metadata["comment"][:13] == "opso_metadata":
            try:
                metadata = parse_opso_metadata(metadata["comment"])
            except Exception as exec:
                warnings.warn(
                    "The file seems to contain opensoundcape metadata in the "
                    f"comment field, but parse_opso_metadata raised {exec}"
                )
        # otherwise, if this is an AudioMoth file, try to parse out additional
        # metadata from the comment field
        elif "artist" in metadata and "AudioMoth" in metadata["artist"]:
            try:
                metadata = parse_audiomoth_metadata(metadata)
            except Exception as exc:
                warnings.warn(
                    "This seems to be an AudioMoth file, "
                    f"but parse_audiomoth_metadata() raised: {exc}"
                )

        ## Update metadata ##
        metadata["channels"] = 1  # we sum to mono when we load with librosa

    except Exception as exc:
        warnings.warn(f"Failed to load metadata: {exc}. Metadata will be None")
        metadata = None

    return metadata


def _write_metadata(metadata, metadata_format, path):
    """write metadata using one of the supported formats

    Args:
        metadata: dictionary of metadata
        metadata_format: one of 'opso','opso_metadata_v0.1','soundfile'
            (see Audio.wave documentation)
        path: file path to save metadata in with soundfile
    """
    metadata = metadata.copy()  # avoid changing the existing object
    if metadata_format == "soundfile":
        pass  # just write the metadata as is
    elif metadata_format in ("opso", "opso_metadata_v0.1"):
        # opso_metadata_v0.1 is currently the most recent
        # so metadata_format='opso' will also use this format
        metadata["comment"] = generate_opso_metadata_str(metadata, version="v0.1")
    else:
        raise NotImplementedError(f"unkown metadata_format: {metadata_format}")

    with soundfile.SoundFile(path, "r+") as s:
        # MUST use r+ mode to update the file without overwriting everything
        for field in [
            "title",
            "copyright",
            "software",
            "artist",
            "comment",
            "date",
            "album",
            "license",
            "tracknumber",
            "genre",
        ]:
            if field in metadata and metadata[field] is not None:
                s.__setattr__(field, metadata[field])
