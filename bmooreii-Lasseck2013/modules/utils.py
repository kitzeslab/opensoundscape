from multiprocessing import cpu_count
from configparser import ConfigParser
from os.path import isfile
from os.path import join
import sys

def return_cpu_count(config):
    '''Return the number of CPUs requested

    If num_processors defined in the config, return that number. else return
    the number of CPUs on the machine.

    Args:
        config: The parsed ini file for this run

    Returns:
        nprocs: Integer number of cores
    '''
    if config['num_processors'] == '':
        nprocs = cpu_count()
    else:
        nprocs = config.getint('num_processors')
    return nprocs


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
    openbird_dir = sys.path[0]
    f_default = join(openbird_dir, f_default)

    if not isfile(f_default):
        raise FileNotFoundError("{} doesn't exist!".format(f_default))
    if not isfile(f_override):
        raise FileNotFoundError("{} doesn't exist!".format(f_override))

    config = ConfigParser()
    config.read(f_default)
    config.read(f_override)
    return config[section]
