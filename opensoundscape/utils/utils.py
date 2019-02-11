from multiprocessing import cpu_count
from configparser import ConfigParser
from os.path import isfile
from os.path import join
import sys
from opensoundscape.config.checks import ini_section_and_keys_exists
from opensoundscape.config.checks import config_checks
from cv2 import TM_CCOEFF
from cv2 import TM_CCOEFF_NORMED
from cv2 import TM_CCORR
from cv2 import TM_CCORR_NORMED
from cv2 import TM_SQDIFF
from cv2 import TM_SQDIFF_NORMED


def get_percent_from_section(config, section, item):
    """Return a percent

    Get percent and verify it. If it is greater than 1 assume
    it needs to be divided by 100, otherwise return it.

    Args:
        config: The parsed ini file for this run

    Returns:
        percent stratification between 0 and 1
    """

    val = config[section].getfloat(item)
    if val > 0.0 and val <= 1.0:
        return val
    elif val <= 100.0:
        return val / 100.0
    else:
        raise ValueError(
            f"Percent {section}.{item} can be (0,1), e.g. 0.33, or [1,100), e.g. 33.3"
        )


def get_template_matching_algorithm(config):
    """ Return the template matching algorithm for OpenCV

    Input:
        The opensoundscape config

    Output:
        An integer
    """

    options = {
        "TM_CCOEFF": TM_CCOEFF,
        "TM_CCOEFF_NORMED": TM_CCOEFF_NORMED,
        "TM_CCORR": TM_CCORR,
        "TM_CCORR_NORMED": TM_CCORR_NORMED,
        "TM_SQDIFF": TM_SQDIFF,
        "TM_SQDIFF_NORMED": TM_SQDIFF_NORMED,
    }
    return options[config["model_fit"].get("template_match_algorithm")]


def return_cpu_count(config):
    """Return the number of CPUs requested

    If num_processors defined in the config, return that number. else return
    the number of CPUs on the machine minus 1 (leave some space for system).

    Args:
        config: The parsed ini file for this run

    Returns:
        nprocs: Integer number of cores
    """
    if config["general"]["num_processors"] == "":
        nprocs = cpu_count() - 1
    else:
        nprocs = config["general"].getint("num_processors")
    return nprocs


def generate_config(f_default, arguments):
    """Generate the configuration

    Simply return a ConfigParser for opensoundscape. We have a default config in
    `config/` as well as a potential override file (arguments["--ini"]). Access elements via
    `config[<section>].get{float,boolean,int}('key')`.

    Args:
        f_default: The default config `config/opensoundscape.ini`
        arguments: The docopt arguments to store

    Returns:
        A ConfigParser instance

    Raises:
        FileNotFoundError if INI file doesn't exist
    """
    opensoundscape_dir = sys.path[0]
    f_default = join(opensoundscape_dir, f_default)
    f_override = arguments["--ini"]

    if not isfile(f_default):
        raise FileNotFoundError(f"{f_default} doesn't exist!")
    if not isfile(f_override):
        raise FileNotFoundError(f"{f_override} doesn't exist!")

    config = ConfigParser(allow_no_value=True)
    config.read(f_default)

    override_config = ConfigParser()
    override_config.read(f_override)

    # Check that the override config makes sense, then read it
    ini_section_and_keys_exists(config, override_config, f_override)
    config.read(f_override)

    config_checks(config)

    # Finally, store any command line options so we only need to pass config
    config["docopt"] = {}
    config["docopt"]["label"] = arguments["<label>"]
    config["docopt"]["image"] = arguments["<image>"]
    config["docopt"]["print_segments"] = str(arguments["--segments"]).lower()
    config["docopt"]["rerun_statistics"] = str(arguments["--rerun_statistics"]).lower()
    config["docopt"]["csv_file"] = arguments["<csv_file>"]

    return config


def yes_no(question, default="no"):
    """ Ask a yes/no question

    Ask a yes/no question with a default.

    Args:
        question: The y/n question
        default: What should the default be if blank

    Returns:
        True (yes) or False (no)

    Raises:
        Nothing
    """

    yes = set(["yes", "y", "ye"])
    no = set(["no", "n"])

    if default == "no":
        yn_def_str = "[yN]"
        yn_def_ret = False
    else:
        yn_def_str = "[Yn]"
        yn_def_ret = True

    while True:
        choice = input(f"{question} {yn_def_str}: ").lower()
        if choice in yes:
            return True
        elif choice in no:
            return False
        elif choice == "":
            return yn_def_ret
        else:
            print("Please respond with 'yes' or 'no'")
