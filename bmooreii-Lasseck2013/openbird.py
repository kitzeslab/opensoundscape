#!/usr/bin/env python
''' openbird.py -- OpenBird
Usage:
    openbird.py [-hv]
    openbird.py sampling [-i <ini>]
    openbird.py spect_gen [-i <ini>]
    openbird.py view <label> [<image>] [-i <ini>] [-s]
    openbird.py model_fit [-i <ini>]
    openbird.py predict <model> [-i <ini>]

Positional Arguments:
    <image>            An image file to write, e.g. 'image.png'

Options:
    -h --help           Print this screen and exit
    -v --version        Print the version of crc-squeue.py
    -i --ini <ini>      Specify an override file [default: openbird.ini]
    -s --segments       View the segments only [default: False]
'''


def generate_config(f_default, f_override, section):
    '''Get the configuration section

    Simply return a ConfigParser containing the INI section of interest. We
    have a default config in config as well as an override file.  Access
    elements via `config.get{float,boolean,int}('key')`.

    Args:
        f_default: The default config `config/openbird.ini`
        f_override: The override config `openbird.ini`
        section: The parent section of the INI file

    Returns:
        A ConfigParser instance 

    Raises:
        FileNotFoundError if INI file doesn't exist
    '''
    if not isfile(f_default):
        raise FileNotFoundError("{} doesn't exist!".format(f_default))
    if not isfile(f_override):
        raise FileNotFoundError("{} doesn't exist!".format(f_override))

    config = ConfigParser()
    config.read(f_default)
    config.read(f_override)
    return config[section]


from docopt import docopt
from os.path import isfile
from os.path import abspath
from configparser import ConfigParser
from modules.sampling import sampling
from modules.spect_gen import spect_gen
from modules.view import view
from modules.model_fit import model_fit
from modules.predict import predict

# From the docstring, generate the arguments dictionary
arguments = docopt(__doc__, version='openbird.py version 0.0.1')

# Get the default config variables
defaults = generate_config('config/openbird.ini', arguments['--ini'], 'default')

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
    # Given a directory, make a prediction based on a model
    #predict(arguments['<dir>'], defaults)
    print("predict")
