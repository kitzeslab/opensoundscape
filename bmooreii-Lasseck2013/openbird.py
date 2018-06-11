#!/usr/bin/env python
''' openbird.py -- OpenBird
Usage:
    openbird.py [-hv]
    openbird.py sampling [-i <ini>]
    openbird.py spect_gen [-i <ini>]
    openbird.py template_pool_gen [-i <ini>]
    openbird.py view <label> [<image>] [-i <ini>] [-s]
    openbird.py model_fit [-i <ini>]
    openbird.py predict [-i <ini>]

Positional Arguments:
    <label>             The label you would like to view, must be in the configs
                            defined as `data_dir`
    <image>             An image file to write, e.g. 'image.png'

Options:
    -h --help           Print this screen and exit
    -v --version        Print the version of crc-squeue.py
    -i --ini <ini>      Specify an override file [default: openbird.ini]
    -s --segments       View the segments only [default: False]
'''


from docopt import docopt
from modules.sampling import sampling
from modules.spect_gen import spect_gen
from modules.view import view
from modules.model_fit import model_fit
from modules.predict import predict
from modules.utils import generate_config
from modules.template_pool_gen import template_pool_gen
import sys

# From the docstring, generate the arguments dictionary
arguments = docopt(__doc__, version='openbird.py version 0.0.1')

# Get the default config variables
defaults = generate_config("{}/{}".format(sys.path[0], "config/openbird.ini"), arguments["--ini"], "default")

# Initialize empty string for arguments['<image>'] (not supported by docopt)
if not arguments['<image>']:
    arguments['<image>'] = ''

if arguments['sampling']:
    # Preprocess the file with the defaults
    #sampling(arguments['<file>'], defaults)
    print("sampling")

elif arguments['spect_gen']:
    # Pass the configuration to spect_gen
    spect_gen(defaults)

elif arguments['view']:
    # Preprocess the file with the defaults
    # -> optionally write image to a file
    view(arguments['<label>'], arguments['<image>'], arguments['--segments'], defaults)

elif arguments['model_fit']:
    # Using defined algorithm, create model
    model_fit(defaults)

elif arguments['predict']:
    # Make a prediction based on a model
    predict(defaults)

elif arguments['template_pool_gen']:
    # Build template pool based on config
    template_pool_gen(defaults)
