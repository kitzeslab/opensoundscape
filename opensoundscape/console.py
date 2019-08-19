#!/usr/bin/env python3
""" console.py: Entrypoint for opensoundscape
"""

from docopt import docopt
import subprocess
import opensoundscape as opso
from opensoundscape import __version__ as opensoundscape_version


OPSO_DOCOPT = """ opensoundscape.py -- Opensoundscape
Usage:
    opensoundscape.py [-hv]

Options:
    -h --help               Print this screen and exit
    -v --version            Print the version of opensoundscape.py
"""


def entrypoint():
    """The Opensoundscape entrypoint for console interaction
    """

    args = docopt(
        OPSO_DOCOPT, version=f"opensoundscape version {opensoundscape_version}"
    )

    print(args)


def build_docs():
    """ Run sphinx-build for our project
    """
    subprocess.run(["sphinx-build", "doc", "doc/_build"], cwd=f"{opso.__path__[0]}/..")
