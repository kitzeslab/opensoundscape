from difflib import get_close_matches


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


def matchTemplate_algorithm_exists(config):
    """ Given the parsed INI files
    
    Check that the matchTemplate algorithm exists, potential options:
    TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF,
    TM_SQDIFF_NORMED

    Input:
        config: The opensoundscape configuration
    """

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

    matchTemplate_algorithm_exists(config)
