#!/usr/bin/env python3
""" opensoundscape.py -- Opensoundscape
Usage:
    opensoundscape.py [-hv]
    opensoundscape.py init [-i <ini>]
    opensoundscape.py spect_gen [-i <ini>]
    opensoundscape.py view <label> [<image>] [-i <ini>] [-s]
    opensoundscape.py model_fit [-i <ini>] [-r] [<csv_file>]
    opensoundscape.py predict [-i <ini>]

Positional Arguments:
    <label>                 The label you would like to view, must be in the configs
                                defined as `data_dir`
    <image>                 An image file to write, e.g. 'image.png'
    <csv_file>              Store model_fit metrics in a CSV file

Options:
    -h --help               Print this screen and exit
    -v --version            Print the version of opensoundscape.py
    -i --ini <ini>          Specify an override file [default: opensoundscape.ini]
    -s --segments           View the segments only [default: False]
    -r --rerun_statistics   Override to model_fit statistics lock [default: False]
"""

from docopt import docopt
from opensoundscape.config.config import generate_config
from opensoundscape import __version__ as opso_version
from opensoundscape.init.init import init
from opensoundscape.spect_gen.spect_gen import spect_gen
from opensoundscape.model_fit.model_fit import model_fit
from opensoundscape.predict.predict import predict
from opensoundscape.view.view import view


def run():
    """
    The entry point for OpenSoundscape console interaction
    """

    arguments = docopt(__doc__, version=f"opensoundscape.py version {opso_version}")

    config = generate_config(arguments)

    # Initialize empty string for arguments['<image>']
    # -> (not supported by docopt)
    if not arguments["<image>"]:
        arguments["<image>"] = ""

    if arguments["init"]:
        init(config)

    elif arguments["spect_gen"]:
        spect_gen(config)

    elif arguments["model_fit"]:
        model_fit(config)

    elif arguments["predict"]:
        predict(config)

    elif arguments["view"]:
        view(config)
