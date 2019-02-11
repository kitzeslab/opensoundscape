from difflib import get_close_matches
from importlib.util import find_spec


def ini_section_and_keys_exists(
    default_config, override_config, override_config_filename
):
    """ Given the parsed INI files

    Do the sections and keys in the override config exist in the default config?

    Input:
        default_config: The unmodified config/opensoundscape.ini INI object
        override_config: The user modifications made to the opensoundscape configuration
        override_config_filename: The filename for the override file
    """

    for section in override_config.keys():
        # -> First, does the section even exist?
        if not section in default_config.keys():
            close_matches = get_close_matches(section, default_config.keys())
            print(
                f"ERROR: From {override_config_filename}, section '{section}' isn't recognized!"
            )
            if close_matches:
                print(f"-> did you mean: {' '.join(close_matches)}")
            exit()

        # Second, do the keys even exist?
        for key in override_config[section].keys():
            if not key in default_config[section].keys():
                close_matches = get_close_matches(key, default_config[section].keys())
                print(
                    f"ERROR: From {override_config_filename}, section {section}, key '{key}' isn't recognized!"
                )
                if close_matches:
                    print(f"-> did you mean: {' '.join(close_matches)}")
                exit()


def match_algorithm_exists(config):
    """ Given the parsed INI files
    
    Check if the user wants to use OpenCV. If yes, can we import OpenCV?
    Make sure that the algorithm they pick exists, potential options:
    TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF,
    TM_SQDIFF_NORMED

    Input:
        config: The opensoundscape configuration
    """

    match_method = config["model_fit"].get("template_match_method")
    if match_method == "opencv":
        cv2_spec = find_spec("cv2")
        if not cv2_spec:
            raise ImportError(
                "You have requested `template_match_method = opencv`, but you haven't installed OpenCV?"
            )

        options = [
            "TM_CCOEFF",
            "TM_CCOEFF_NORMED",
            "TM_CCORR",
            "TM_CCORR_NORMED",
            "TM_SQDIFF",
            "TM_SQDIFF_NORMED",
        ]
        algo = config["model_fit"].get("template_match_algorithm")
        if algo not in options:
            raise ValueError(
                f"The template matching algorithm chosen was {algo}, valid options: {' '.join(options)}"
            )


def config_checks(config):
    """ Given the parsed INI files

    The entry point for configuration checks

    Input:
        default_config: The unmodified config/opensoundscape.ini INI object
        override_config: The user modifications made to the opensoundscape configuration
        override_config_filename: The filename for the override file
    
    Output:
        None

    Raises:
        Nothing, but can various functions will exit
    """

    match_algorithm_exists(config)
