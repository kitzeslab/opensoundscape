#!/usr/bin/env python3
""" opensoundscape.py -- Opensoundscape
Usage:
    opensoundscape.py [-hv]

Options:
    -h --help               Print this screen and exit
    -v --version            Print the version of opensoundscape.py
"""

from docopt import docopt
from opensoundscape import __version__ as opensoundscape_version


def entrypoint():
    """
    The Opensoundscape entrypoint for console interaction
    """

    args = docopt(__doc__, version=f"opensoundscape version {opensoundscape_version}")

    print(args)
