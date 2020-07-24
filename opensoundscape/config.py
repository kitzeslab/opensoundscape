#!/usr/bin/env python3
from schema import Schema, And, Or, Use
import yaml
from pathlib import Path
from io import StringIO

DEFAULT_CONFIG = """\
runtime:
  cores_per_node: 1                 # Number of cores per node (positive integer)
  batch_size: 1                     # Size of batches (positive integer)
raven:
  annotations: false                # Look for Raven annotations (boolean)
  label_corrections: null           # Use this file to correct classes in Raven annotations (null or string)
audio:
  sample_rate: null                 # Sample rate for audio resampling (null or positive integer)
  max_duration: null                # Maximum duration of audio file during read (null or positive integer)
  resample_type: "kaiser_best"      # Resample type for librosa ("kaiser_best" or "kaiser_fast")
  duration: 5                       # Duration in seconds for split audio
  overlap: 1                        # Overlap of audio for split audio
""".strip()

greater_than_zero = lambda n: n > 0

AUDIO_SCHEMA = Schema(
    {
        "sample_rate": Or(
            None,
            And(
                Use(int),
                greater_than_zero,
                error="Sample rate should be null or a positive integer",
            ),
        ),
        "max_duration": Or(
            None,
            And(
                Use(int),
                greater_than_zero,
                error="Max duration should be a positive integer or null",
            ),
        ),
        "resample_type": And(
            Use(str),
            lambda s: s in ["kaiser_best", "kaiser_fast"],
            error="Resample type can be one of kaiser_fast or kaiser_best",
        ),
        "duration": And(Use(int), greater_than_zero),
        "overlap": And(Use(int), greater_than_zero),
    }
)

RUNTIME_SCHEMA = Schema(
    {
        "cores_per_node": And(Use(int), greater_than_zero),
        "batch_size": And(Use(int), greater_than_zero),
    }
)

RAVEN_SCHEMA = Schema(
    {
        "annotations": Use(
            bool, error="Annotations should be a boolean value, e.g. `true` or `false`"
        ),
        "label_corrections": Or(
            None,
            And(
                Use(str),
                lambda fname: Path(fname).exists() and Path(fname).isfile(),
                error="Label corrections should be a file that exists",
            ),
        ),
    }
)

SCHEMA = Schema(
    {"audio": AUDIO_SCHEMA, "runtime": RUNTIME_SCHEMA, "raven": RAVEN_SCHEMA}
)


def validate(config):
    """ Validate a configuration string

    Input:
        config: A string containing an Opensoundscape configuration

    Output:
        dict: A dictionary of the validated Opensoundscape configuration
    """
    with StringIO(config) as f:
        yml = yaml.load(f, Loader=yaml.SafeLoader)
    return SCHEMA.validate(yml)


def validate_file(fname):
    """ Validate a configuration file

    Input:
        fname: A filename containing an Opensoundscape configuration

    Output:
        dict: A dictionary of the validated Opensoundscape configuration
    """
    with open(fname, "r") as f:
        yml = yaml.load(f, Loader=yaml.SafeLoader)
    return SCHEMA.validate(yml)


def get_default_config():
    """ Get the default configuration file as a dictionary

    Output:
        dict: A dictionary containing the default Opensoundscape configuration

    """
    return validate(DEFAULT_CONFIG)
