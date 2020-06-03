#!/usr/bin/env python3
""" console.py: Entrypoint for opensoundscape
"""

from docopt import docopt
import subprocess
import opensoundscape as opso
from opensoundscape import __version__ as opensoundscape_version
import opensoundscape.raven as raven
from opensoundscape.completions import COMPLETIONS
import opensoundscape.console_checks as checks
from opensoundscape.datasets import Splitter
from torch.utils.data import DataLoader
from pathlib import Path
from itertools import chain


OPSO_DOCOPT = """ opensoundscape.py -- Opensoundscape
Usage:
    opensoundscape [-hv]
    opensoundscape completions
    opensoundscape raven_annotation_check <directory>
    opensoundscape raven_lowercase_annotations <directory>
    opensoundscape raven_generate_class_corrections <directory> <output.csv>
    opensoundscape raven_query_annotations <directory> <class>
    opensoundscape split_audio (-i <directory>) (-o <directory>) (-d <duration>) (-p <overlap>)
        [-a -l <labels.csv>] [-n <cores>] [-b <batch_size>] [-s <segments.csv>]

Options:
    -h --help                           Print this screen and exit
    -v --version                        Print the version of opensoundscape.py
    -i --input_directory <directory>    The input directory for the analysis
    -o --output_directory <directory>   The output directory for the analysis
    -d --duration <duration>            The segment duration in seconds
    -p --overlap <overlap>              The segment overlap in seconds
    -a --annotations                    Search for Raven annotation files
    -l --labels <labels.csv>            A CSV file with corrections to labels in Raven annotations files
    -n --num_cores <number>             The number of cores to use for the analysis [default: 1]
    -b --batch_size <number>            The batch_size for the analysis [default: 1]
    -s --segments <segments.csv>        Write segments to this file [default: segments.csv]

Positional Arguments:
    <directory>                         A path to a directory
    <output.csv>                        A CSV file containing the output of an analysis
    <class>                             The class name for the analysis

Descriptions:
    completions                         Generate bash completions
                                        `opensoundscape completions > ~/.local/share/bash-completion/completions/opensoundscape`
    raven_annotation_check              Given a directory of Raven annotation files, check that a class is specified
    raven_generate_class_corrections    Given a directory of Raven annotation files,
                                        generate a CSV file to check classes and correct any issues
    raven_query_annotations             Given a directory of Raven annotation files, search for rows matching a specific class
    split_audio                         Given a directory of WAV files, optional Raven annotation files,
                                            optional label corrects, split then into segments of a given duration and overlap
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
        csv = raven.generate_class_corrections(
            args["<directory>"], lower=args["--lower"]
        )
        with open(args["<output.csv>"], "w") as f:
            f.write(csv)

    elif args["raven_query_annotations"]:
        raven.query_annotations(args["<directory>"], args["<class>"])

    elif args["split_audio"]:
        args["--duration"] = checks.positive_integer(args, "--duration")
        args["--overlap"] = checks.positive_integer(args, "--overlap")
        args["--batch_size"] = checks.positive_integer_with_default(
            args, "--batch_size"
        )
        args["--num_cores"] = checks.positive_integer_with_default(args, "--num_cores")

        input_p = checks.directory_exists(args, "--input_directory")
        output_p = checks.directory_exists(args, "--output_directory")

        if not args["--segments"]:
            args["--segments"] = "segments.csv"
        segments = Path(args["--segments"])
        if segments.exists():
            segments.rename(segments.with_suffix(".csv.bak"))

        wavs = chain(input_p.rglob("**/*.WAV"), input_p.rglob("**/*.wav"))

        dataset = Splitter(
            wavs,
            annotations=args["--annotations"],
            labels=args["--labels"],
            overlap=args["--overlap"],
            duration=args["--duration"],
            output_directory=args["--output_directory"],
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args["--batch_size"],
            shuffle=False,
            num_workers=args["--num_cores"],
            collate_fn=Splitter.collate_fn,
        )

        with open(args["--segments"], "w") as f:
            if args["--annotations"]:
                f.write("Source,Annotations,Begin (s),End (s),Destination,Labels\n")
            else:
                f.write("Source,Begin (s),End (s),Destination\n")
            for idx, data in enumerate(dataloader):
                for output in data:
                    f.write(f"{output}\n")

    else:
        raise NotImplementedError(
            "The requested command is not implemented. Please submit an issue."
        )


def build_docs():
    """ Run sphinx-build for our project
    """
    subprocess.run(["sphinx-build", "doc", "doc/_build"], cwd=f"{opso.__path__[0]}/..")
