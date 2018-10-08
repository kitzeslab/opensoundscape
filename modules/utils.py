from multiprocessing import cpu_count
from configparser import ConfigParser
from os.path import isfile
from os.path import join
import sys
from difflib import get_close_matches

def return_cpu_count(config):
    '''Return the number of CPUs requested

    If num_processors defined in the config, return that number. else return
    the number of CPUs on the machine minus 1 (leave some space for system).

    Args:
        config: The parsed ini file for this run

    Returns:
        nprocs: Integer number of cores
    '''
    if config['general']['num_processors'] == '':
        nprocs = cpu_count() - 1
    else:
        nprocs = config['general'].getint('num_processors')
    return nprocs


def generate_config(f_default, f_override):
    '''Generate the configuration

    Simply return a ConfigParser for opensoundscape. We have a default config in
    `config/` as well as a potential override file. Access elements via
    `config[<section>].get{float,boolean,int}('key')`.

    Args:
        f_default: The default config `config/opensoundscape.ini`
        f_override: The override config `opensoundscape.ini`

    Returns:
        A ConfigParser instance

    Raises:
        FileNotFoundError if INI file doesn't exist
    '''
    opensoundscape_dir = sys.path[0]
    f_default = join(opensoundscape_dir, f_default)

    if not isfile(f_default):
        raise FileNotFoundError(f"{f_default} doesn't exist!")
    if not isfile(f_override):
        raise FileNotFoundError(f"{f_override} doesn't exist!")

    config = ConfigParser()
    config.read(f_default)

    override_config = ConfigParser()
    override_config.read(f_override)

    # Verification of override sections and keys
    for section in override_config.keys():
        # -> First, does the section even exist?
        if not section in config.keys():
            close_matches = get_close_matches(section, config.keys())
            print(f"ERROR: From {f_override}, section '{section}' isn't recognized!")
            if close_matches:
                print(f"-> did you mean: {' '.join(close_matches)}")
            exit()

        # Second, do the keys even exist?
        for key in override_config[section].keys():
            if not key in config[section].keys():
                close_matches = get_close_matches(key, config[section].keys())
                print(f"ERROR: From {f_override}, section {section}, key '{key}' isn't recognized!")
                if close_matches:
                    print(f"-> did you mean: {' '.join(close_matches)}")
                exit()

    # If we make it here, our override config contains valid sections and keys
    config.read(f_override)
    return config


def yes_no(question, default='no'):
    ''' Ask a yes/no question

    Ask a yes/no question with a default.

    Args:
        question: The y/n question
        default: What should the default be if blank

    Returns:
        True (yes) or False (no)

    Raises:
        Nothing
    '''

    yes = set(['yes', 'y', 'ye'])
    no = set(['no', 'n'])

    if default == 'no':
        yn_def_str = '[yN]'
        yn_def_ret = False
    else:
        yn_def_str = '[Yn]'
        yn_def_ret = True

    while True:
        choice = input(f"{question} {yn_def_str}: ").lower()
        if choice in yes:
           return True
        elif choice in no:
           return False
        elif choice == '':
           return yn_def_ret
        else:
           print("Please respond with 'yes' or 'no'")
