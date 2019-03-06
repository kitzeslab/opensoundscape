from opensoundscape.utils.db_utils import write_ini_section


class OpensoundscapeSpectGenAlgorithmDoesntExist(Exception):
    """
    spect_gen algorithm doesn't exist exception
    """

    pass


def spect_gen(config):
    """Fit a model

    Given a method (from config), generate spectrum for all input files.

    Args:
        config: The parsed ini file for this run

    Returns:
        Nothing or writes to MongoDB

    Raises:
        Nothing
    """

    if config["spect_gen"]["algo"] == "template_matching":
        from opensoundscape.spect_gen.spect_gen_algo.template_matching.spect_gen_algo import (
            spect_gen_algo,
        )
    else:
        raise OpensoundscapeSpectGenAlgorithmDoesntExist(
            f"The algorithm '{config['spect_gen']['algo']}' doesn't exist!"
        )

    if config["general"].getboolean("db_rw"):
        write_ini_section(config, "spect_gen")

    spect_gen_algo(config)
