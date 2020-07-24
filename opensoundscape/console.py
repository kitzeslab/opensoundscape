#!/usr/bin/env python3
""" console.py: Entrypoint for opensoundscape
"""

from docopt import docopt
from pathlib import Path
from itertools import chain
import subprocess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import opensoundscape as opso
from opensoundscape import __version__ as opensoundscape_version
import opensoundscape.raven as raven
from opensoundscape.completions import COMPLETIONS
from opensoundscape.config import validate_file, get_default_config, DEFAULT_CONFIG
import opensoundscape.console_checks as checks
import opensoundscape.datasets as datasets
from tempfile import TemporaryDirectory
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
    opensoundscape predict_from_directory (-i <directory>) (-d <state_dict.pth>) [-c <opensoundscape.yaml>]

Options:
    -h --help                           Print this screen and exit
    -v --version                        Print the version of opensoundscape.py
    -i --input_directory <directory>    The input directory for the analysis
    -o --output_directory <directory>   The output directory for the analysis
    -s --segments <segments.csv>        Write segments to this file
    -c --config <opensoundscape.yaml>   The opensoundscape.yaml config file
    -d --state_dict <state_dict.pth>    A PyTorch state dictionary for ResNet18
                                            e.g. `torch.save(model.state_dict(), "state_dict.pth")`

Positional Arguments:
    <directory>                         A path to a directory
    <output.csv>                        A CSV file containing the output of an analysis
    <class>                             The class name for the analysis

Descriptions:
    completions                         Generate bash completions `opensoundscape completions > ~/.local/share/bash-completion/completions/opensoundscape`
    default_config                      Write the default opensoundscape.yaml configuration file to standard out
    raven_annotation_check              Given a directory of Raven annotation files, check that a class is specified
    raven_generate_class_corrections    Given a directory of Raven annotation files, generate a CSV file to check classes and correct any issues
    raven_query_annotations             Given a directory of Raven annotation files, search for rows matching a specific class
    split_audio                         Given a directory of WAV files, generate splits of the audio
    predict_from_directory              Given a directory of WAV files, run a PyTorch model prediction on 5 second segments
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

        wavs = chain(
            input_p.rglob("**/*.WAV"),
            input_p.rglob("**/*.wav"),
            input_p.rglob("**/*.mp3"),
            input_p.rglob("**/*.MP3"),
        )

        dataset = datasets.SplitterDataset(
            wavs,
            annotations=config["raven"]["annotations"],
            label_corrections=config["raven"]["label_corrections"],
            overlap=config["audio"]["overlap"],
            duration=config["audio"]["duration"],
            output_directory=args["--output_directory"],
        )

        dataloader = DataLoader(
            dataset,
            batch_size=config["runtime"]["batch_size"],
            shuffle=False,
            num_workers=config["runtime"]["cores_per_node"],
            collate_fn=datasets.SplitterDataset.collate_fn,
        )

        with open(args["--segments"], "w") as f:
            if config["raven"]["annotations"]:
                f.write("Source,Annotations,Begin (s),End (s),Destination,Labels\n")
            else:
                f.write("Source,Begin (s),End (s),Destination\n")
            for idx, data in enumerate(dataloader):
                for output in data:
                    f.write(f"{output}\n")

    elif args["predict_from_directory"]:
        config = get_default_config()
        if args["--config"]:
            config = validate_file(args["--config"])

        input_p = checks.directory_exists(args, "--input_directory")

        wavs = chain(
            input_p.rglob("**/*.WAV"),
            input_p.rglob("**/*.wav"),
            input_p.rglob("**/*.mp3"),
            input_p.rglob("**/*.MP3"),
        )

        with TemporaryDirectory() as segments_dir:
            dataset = datasets.SplitterDataset(
                wavs,
                overlap=config["audio"]["overlap"],
                duration=config["audio"]["duration"],
                output_directory=segments_dir,
            )

            dataloader = DataLoader(
                dataset,
                batch_size=config["runtime"]["batch_size"],
                shuffle=False,
                num_workers=config["runtime"]["cores_per_node"],
                collate_fn=datasets.SplitterDataset.collate_fn,
            )

            segments_csv = f"{segments_dir}/segments.csv"
            with open(segments_csv, "w") as f:
                f.write("Source,Begin (s),End (s),Destination\n")
                for idx, data in enumerate(dataloader):
                    for output in data:
                        f.write(f"{output}\n")

            input_df = pd.read_csv(segments_csv)
            dataset = datasets.SingleTargetAudioDataset(input_df)

            dataloader = DataLoader(
                dataset,
                batch_size=config["runtime"]["batch_size"],
                shuffle=False,
                num_workers=config["runtime"]["cores_per_node"],
            )

            try:
                model = resnet18(pretrained=False)
                model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2)
                model.load_state_dict(torch.load(args["--state_dict"]))
            except:
                exit(
                    f"I was unable to load the state dictionary from `{args['--state_dict']}`"
                )

            model.eval()
            with torch.no_grad():
                for idx, data in enumerate(dataloader):
                    X = data["X"]
                    predictions = outputs.clone().detach().argmax(dim=1)
                    start = config["runtime"]["batch_size"] * idx
                    end = start + config["runtime"]["batch_size"]
                    for fname, pred in zip(
                        input_df["Destination"][start:end], predictions
                    ):
                        print(f"{fname},{pred}")

    else:
        raise NotImplementedError(
            "The requested command is not implemented. Please submit an issue."
        )


def build_docs():
    """ Run sphinx-build for our project
    """
    subprocess.run(
        ["sphinx-build", "docs", "docs/_build"], cwd=f"{opso.__path__[0]}/.."
    )
