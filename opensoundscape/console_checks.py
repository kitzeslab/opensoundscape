#!/usr/bin/env python3
""" Utilities related to console checks on docopt args
"""
from pathlib import Path


def positive_integer(args, parameter):
    try:
        r = int(args[parameter])
        if r <= 0:
            raise ValueError
        return r
    except ValueError:
        exit(
            f"Error: `{parameter}` should be a positive whole number! Got `{args[parameter]}`"
        )


def positive_integer_with_default(args, parameter, default=1):
    if args[parameter]:
        return positive_integer(args, parameter)
    else:
        return default


def directory_exists(args, parameter):
    p = Path(args[parameter])
    if not p.exists():
        raise FileNotFoundError(
            f"The directory given to `{parameter}` does not exist, got `{args[parameter]}`"
        )
    if not p.is_dir():
        raise NotADirectoryError(
            f"The directory given to `{parameter}` should be a directory but I found a file, got `{args[parameter]}`"
        )
    return p
