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
from opensoundscape.config import validate_file, get_default_config, DEFAULT_CONFIG
from opensoundscape.datasets import Splitter
from torch.utils.data import DataLoader
from pathlib import Path
from itertools import chain


OPSO_DOCOPT = """ opensoundscape.py -- Opensoundscape
Usage:
    opensoundscape [-hv]
    opensoundscape completions
    opensoundscape default_config
    opensoundscape raven_annotation_check <directory>
    opensoundscape raven_lowercase_annotations <directory>
    opensoundscape raven_generate_class_corrections <directory> <output.csv>
    opensoundscape raven_query_annotations <directory> <class>
    opensoundscape split_audio (-i <directory>) (-o <directory>) (-s <segments.csv>) [-c <opensoundscape.yaml>]

Options:
    -h --help                                       Print this screen and exit
    -v --version                                    Print the version of opensoundscape.py
    -i --input_directory <directory>                The input directory for the analysis
    -o --output_directory <directory>               The output directory for the analysis
    -s --segments <segments.csv>                    Write segments to this file [default: segments.csv]
    -c --config <opensoundscape.yaml>               The opensoundscape.yaml config file

Positional Arguments:
    <directory>                         A path to a directory
    <output.csv>                        A CSV file containing the output of an analysis
    <class>                             The class name for the analysis

Descriptions:
    completions                         Generate bash completions
                                        `opensoundscape completions > ~/.local/share/bash-completion/completions/opensoundscape`
    default_config                      Write the default opensoundscape.yaml configuration file to standard out
    raven_annotation_check              Given a directory of Raven annotation files, check that a class is specified
    raven_generate_class_corrections    Given a directory of Raven annotation files,
                                        generate a CSV file to check classes and correct any issues
    raven_query_annotations             Given a directory of Raven annotation files, search for rows matching a specific class
    split_audio                         Given a directory of WAV files, generate splits of the audio
"""


def entrypoint():
    """The Opensoundscape entrypoint for console interaction
    """

    args = docopt(
        OPSO_DOCOPT, version=f"opensoundscape version {opensoundscape_version}"
    )

    if args["completions"]:
        print(COMPLETIONS)

    elif args["default_config"]:
        print(DEFAULT_CONFIG)

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
        config = get_default_config()
        if args["--config"]:
            config = validate_file(args["--config"])

        input_p = checks.directory_exists(args, "--input_directory")
        output_p = checks.directory_exists(args, "--output_directory")

        segments = Path(args["--segments"])
        if segments.exists():
            segments.rename(segments.with_suffix(".csv.bak"))

        wavs = chain(input_p.rglob("**/*.WAV"), input_p.rglob("**/*.wav"))

        dataset = Splitter(
            wavs,
            annotations=config["raven"]["annotations"],
            label_corrections=config["raven"]["label_corrections"],
            overlap=config["audio"]["overlap"],
            duration=config["audio"]["duration"],
            output_directory=args["--output_directory"],
            audio_params={
                k: v
                for k, v in config["audio"].items()
                if k in ["sample_rate", "max_duration", "resample_type"]
            },
        )

        dataloader = DataLoader(
            dataset,
            batch_size=config["runtime"]["batch_size"],
            shuffle=False,
            num_workers=config["runtime"]["cores_per_node"],
            collate_fn=Splitter.collate_fn,
        )

        with open(args["--segments"], "w") as f:
            if config["raven"]["annotations"]:
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
