#!/usr/bin/env python3
""" load.py: Utilities for loading audio files
"""

import pathlib
import io
import librosa


class OpsoLoadAudioInputError(Exception):
    """ Custom exception indicating we can't load input
    """

    pass


class OpsoLoadAudioInputTooLong(Exception):
    """ Custom exception indicating length of audio is too long
    """

    pass


def load(audio, sample_rate=22050, max_duration=600):
    """ Load audio in various formats and generate a spectrogram

    Deal with the various possible input types to load an audio
    file and generate a spectrogram

    Args:
        audio: string, pathlib, or io object
        sample_rate: the target sample rate (default: 22050 Hz)
        max_duration: the maximum length of an input file,
                       None is no maximum (default: 600 seconds)

    Returns:
        samples: np.ndarray [shape=(n,)]
        sample_rate: int
    """

    # Deal with a string/path-like object
    # Checks:
    #   1. path exists == True
    #   2. duration is less than max_duration
    path = None
    if audio.__class__ == str:
        # Simply load the audio into a pathlib.Path object
        path = pathlib.Path(audio)
    elif issubclass(audio.__class__, pathlib.PurePath):
        # We already have a pathlib object
        path = audio
    elif issubclass(audio.__class__, io.TextIOBase):
        # We have a StringIO object
        print("StringIO object")
        path = None
    elif issubclass(audio.__class__, io.BufferedIOBase):
        # We have a BytesIO object
        print("BytesIO object")
        path = None
    else:
        raise OpsoLoadAudioInputError(
            f"Error: can't load files of class {audio.__class__}"
        )

    if path:
        if not path.is_file():
            raise FileNotFoundError(f"Error: The file {path} doesn't exist?")
        if librosa.get_duration(filename=path) > max_duration:
            raise OpsoLoadAudioInputTooLong(
                f"Error: The file {path} is longer than {max_duration} seconds"
            )

        # Safe to load w/ librosa

    else:
        print("StringIO or BytesIO object")
