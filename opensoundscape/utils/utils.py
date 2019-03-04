from multiprocessing import cpu_count
from librosa import load


def get_sample_rate(label, config):
    """Given a label, return the native sample rate

    Get the native sample rate of the file

    Args:
        label: The label of the file to generate a spectrogram for
        config: The opensoundscape config file

    Returns:
        sample_rate: The native sample rate for the label
    """

    _, sample_rate = load(
        f"{config['general']['data_dir']}/{label}",
        mono=False,  # Don't automatically load as mono, so we can warn if we force to mono
        res_type=config["spect_gen"]["resample_type"],
        sr=None,
    )

    return sample_rate


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

    OpenCV is an optional library. If we hit this function, we will
    have checked that OpenCV can be imported.

    Input:
        config: The opensoundscape config

    Output:
        An integer
    """

    from cv2 import TM_CCOEFF
    from cv2 import TM_CCOEFF_NORMED
    from cv2 import TM_CCORR
    from cv2 import TM_CCORR_NORMED
    from cv2 import TM_SQDIFF
    from cv2 import TM_SQDIFF_NORMED

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
