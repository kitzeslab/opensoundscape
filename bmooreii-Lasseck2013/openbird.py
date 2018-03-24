#!/usr/bin/env python
''' openbird.py -- OpenBird
Usage:
    openbird.py [-hv]
    openbird.py sampling <dir> [-i <ini>]
    openbird.py spect_gen <file> [-i <ini>]
    openbird.py view <file> [<image>] [-i <ini>] [-s]
    openbird.py model_fit <dir> [-i <ini>]
    openbird.py prediction <model> <dir> [-i <ini>]

Positional Arguments:
    <file>             A wav file, e.g. 'train/001.wav'
    <dir>              A directory, e.g. 'nips4b/'
    <image>            An image file to write, e.g. 'image.png'

Options:
    -h --help           Print this screen and exit
    -v --version        Print the version of crc-squeue.py
    -i --ini <ini>      Specify a different ini file [default: openbird.ini]
    -s --segments       View the segments only [default: False]
'''


def generate_config(section):
    '''Get the configuration section

    Simply return a ConfigParser containing the INI section.
    Access elements via `config.get{float,boolean,int}('key')`.

    Args:
        section: The parent section of the INI file

    Returns:
        A ConfigParser instance 

    Raises:
        FileNotFoundError if INI file doesn't exist
    '''
    if isfile(arguments['--ini']):
        config = ConfigParser()
        config.read(arguments['--ini'])
        return config[section]
    else:
        raise FileNotFoundError("File: {} doesn't exist!".format(arguments['--ini']))


from docopt import docopt
from os.path import isfile
from configparser import ConfigParser
from modules.sampling import sampling
from modules.spect_gen import spect_gen
from modules.view import view
from modules.model_fit import model_fit
from modules.prediction import prediction

# From the docstring, generate the arguments dictionary
arguments = docopt(__doc__, version='openbird.py version 0.0.1')

# Get the default config variables
defaults = generate_config('default')

# Initialize empty string for arguments['<image>'] (not supported by docopt)
if not arguments['<image>']:
    arguments['<image>'] = ''

if arguments['sampling']:
    # Preprocess the file with the defaults
    sampling(arguments['<file>'], defaults)

elif arguments['spect_gen']:
    # Preprocess the file with the defaults
    spect_gen(arguments['<file>'], defaults)

elif arguments['view']:
    # Preprocess the file with the defaults
    # -> optionally write image to a file
    view(arguments['<file>'], arguments['<image>'], arguments['--segments'], defaults)

elif arguments['model_fit']:
    # Using defined algorithm, create model
    model_fit(arguments['<dir>'], defaults)

elif arguments['prediction']:
    # Given a directory, make a prediction based on a model
    prediction(arguments['<dir>'], defaults)
