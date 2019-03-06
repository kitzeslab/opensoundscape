#!/usr/bin/env python3
""" generate_images.py

Generate all of the images related to a <label>

Prerequisites:
    - `opensoundscape spect_gen -i <ini>`

Usage:
    generate_images.py [<final.png>] [-i <ini>] [-hv]

Positional Arguments:
    <final.png>             The final image [default: final.png]

Options:
    -h --help               Print this screen and exit
    -v --version            Print the version of generate_images.py
    -i --ini <ini>          Specify an override file [default: opensoundscape.ini]
"""

from docopt import docopt
import pandas as pd
import numpy as np
from importlib.util import find_spec
import pathlib
from opensoundscape.config.config import generate_config
from opensoundscape.view.view import view
from opensoundscape import __version__ as opso_version


def run():
    arguments = docopt(__doc__, version=f"generate_images.py version {opso_version}")
    config = generate_config(arguments, store_options=False)

    labels_df = pd.read_csv(
        f"{config['general']['data_dir']}/{config['general']['train_file']}",
        index_col="Filename",
    )

    # Need to inject some config to generate an image
    config["docopt"] = {}
    images = [None] * labels_df.shape[0]
    for idx, label in enumerate(labels_df.index.values):
        config["docopt"]["label"] = label
        images[idx] = f"{label.replace('/', '-')}.png"
        config["docopt"]["image"] = images[idx]
        view(config)

    # Try to find fpdf, fail otherwise
    pil_spec = find_spec("PIL")
    if not pil_spec:
        raise ImportError(
            "This script requires the installation of Pillow, but you haven't installed it"
        )

    if not arguments["<final.png>"]:
        arguments["<final.png>"] = "final.png"

    # Write them to the PNG File
    import PIL

    images_pil = [PIL.Image.open(x) for x in images]
    images_vstack = np.vstack(images_pil)
    images_comb = PIL.Image.fromarray(images_vstack)
    images_comb.save(arguments["<final.png>"])

    # Remove the source images
    for image in images:
        p = pathlib.Path(image)
        p.unlink()
