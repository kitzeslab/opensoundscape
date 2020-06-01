#!/usr/bin/env python3
""" console.py: Entrypoint for opensoundscape
"""

from docopt import docopt
import subprocess
import opensoundscape as opso
from opensoundscape import __version__ as opensoundscape_version
import opensoundscape.raven as raven
from opensoundscape.completions import COMPLETIONS


OPSO_DOCOPT = """ opensoundscape.py -- Opensoundscape
Usage:
    opensoundscape [-hv]
    opensoundscape completions
    opensoundscape raven_annotation_check <directory>
    opensoundscape raven_lowercase_annotations <directory>
    opensoundscape raven_generate_class_corrections [-l] <directory> <output.csv>

Options:
    -h --help                           Print this screen and exit
    -v --version                        Print the version of opensoundscape.py
    -l --lower                          Generate class corrections for `**/*.selections.txt.lower` instead of `**/*.selections.txt`

Positional Arguments:
    <directory>                         A path to a directory
    <output.csv>                        A CSV file containing the output of an analysis

Descriptions:
    completions                         Generate bash completions `opensoundscape completions > ~/.local/share/bash-completion/completions/opensoundscape`
    raven_annotation_check              Given a directory of Raven annotation files check that a class is specified
    raven_generate_class_corrections    Given a directory of Raven annotation files, generate a CSV file to check classes and correct any issues
"""


def entrypoint():
    """The Opensoundscape entrypoint for console interaction
    """

    args = docopt(
        OPSO_DOCOPT, version=f"opensoundscape version {opensoundscape_version}"
    )

    if args["completions"]:
        print(COMPLETIONS)

    elif args["raven_annotation_check"]:
        raven.annotation_check(args["<directory>"])

    elif args["raven_generate_class_corrections"]:
        csv = raven.generate_class_corrections(args["<directory>"], lower=args["--lower"])
        with open(args["<output.csv>"], "w") as f:
            f.write(csv)

    else:
        raise NotImplementedError(
            "The requested command is not implemented. Please submit an issue."
        )


def build_docs():
    """ Run sphinx-build for our project
    """
    subprocess.run(["sphinx-build", "doc", "doc/_build"], cwd=f"{opso.__path__[0]}/..")
